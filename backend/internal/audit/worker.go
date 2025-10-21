package audit

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Worker processes async audit tasks from the queue (Phase 5 WP3)
// Handles at-least-once processing with idempotency checks and DLQ for failures
type Worker struct {
	mu                sync.RWMutex
	workerID          string
	queue             AuditQueue
	taskHandlers      map[TaskType]TaskHandler
	dlq               DeadLetterQueue
	pollInterval      time.Duration
	maxRetries        int
	idempotencyStore  IdempotencyStore
	metrics           *WorkerMetrics
	stopCh            chan struct{}
	wg                sync.WaitGroup
}

// WorkerMetrics tracks worker operations
type WorkerMetrics struct {
	mu                    sync.RWMutex
	TasksProcessed        int64
	TasksSucceeded        int64
	TasksFailed           int64
	TasksRetried          int64
	TasksDLQd             int64
	AvgProcessingTimeMs   float64
	LastProcessedAt       time.Time
}

// AuditQueue abstracts the task queue (Redis, RabbitMQ, SQS, etc.)
type AuditQueue interface {
	Poll(ctx context.Context, count int) ([]AuditTask, error)
	Ack(ctx context.Context, taskID string) error
	Nack(ctx context.Context, taskID string) error
}

// DeadLetterQueue handles failed tasks
type DeadLetterQueue interface {
	Push(ctx context.Context, task AuditTask, reason string) error
	List(ctx context.Context, limit int) ([]DLQEntry, error)
	Remove(ctx context.Context, taskID string) error
}

// DLQEntry represents a failed task in the DLQ
type DLQEntry struct {
	Task       AuditTask
	Reason     string
	FailedAt   time.Time
	RetryCount int
}

// IdempotencyStore tracks processed tasks to prevent duplicates
type IdempotencyStore interface {
	MarkProcessed(ctx context.Context, taskID string, ttl time.Duration) error
	IsProcessed(ctx context.Context, taskID string) (bool, error)
}

// TaskHandler processes a specific task type
type TaskHandler interface {
	Handle(ctx context.Context, task AuditTask) error
	Type() TaskType
}

// WorkerConfig holds configuration
type WorkerConfig struct {
	WorkerID         string
	Queue            AuditQueue
	TaskHandlers     map[TaskType]TaskHandler
	DLQ              DeadLetterQueue
	IdempotencyStore IdempotencyStore
	PollInterval     time.Duration // Default: 1 second
	MaxRetries       int           // Default: 3
}

// NewWorker creates a new audit worker
func NewWorker(config WorkerConfig) (*Worker, error) {
	if config.WorkerID == "" {
		config.WorkerID = fmt.Sprintf("worker-%d", time.Now().UnixNano())
	}
	if config.Queue == nil {
		return nil, fmt.Errorf("Queue is required")
	}
	if config.DLQ == nil {
		return nil, fmt.Errorf("DLQ is required")
	}
	if config.IdempotencyStore == nil {
		return nil, fmt.Errorf("IdempotencyStore is required")
	}
	if len(config.TaskHandlers) == 0 {
		return nil, fmt.Errorf("at least one TaskHandler is required")
	}

	if config.PollInterval == 0 {
		config.PollInterval = 1 * time.Second // Default: poll every second
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3 // Default: 3 retries
	}

	worker := &Worker{
		workerID:         config.WorkerID,
		queue:            config.Queue,
		taskHandlers:     config.TaskHandlers,
		dlq:              config.DLQ,
		pollInterval:     config.PollInterval,
		maxRetries:       config.MaxRetries,
		idempotencyStore: config.IdempotencyStore,
		metrics:          &WorkerMetrics{},
		stopCh:           make(chan struct{}),
	}

	return worker, nil
}

// Start begins the worker loop (runs in background)
func (w *Worker) Start(ctx context.Context) {
	w.wg.Add(1)
	go w.workerLoop(ctx)
	fmt.Printf("Audit Worker %s: started (poll interval %v, max retries %d)\n",
		w.workerID, w.pollInterval, w.maxRetries)
}

// Stop gracefully stops the worker
func (w *Worker) Stop() {
	close(w.stopCh)
	w.wg.Wait()
	fmt.Printf("Audit Worker %s: stopped\n", w.workerID)
}

// workerLoop continuously polls for tasks and processes them
func (w *Worker) workerLoop(ctx context.Context) {
	defer w.wg.Done()

	ticker := time.NewTicker(w.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-w.stopCh:
			return
		case <-ticker.C:
			if err := w.processBatch(ctx); err != nil {
				fmt.Printf("Audit Worker %s: error processing batch: %v\n", w.workerID, err)
			}
		}
	}
}

// processBatch polls and processes a batch of tasks
func (w *Worker) processBatch(ctx context.Context) error {
	// Poll for tasks (batch size: 10)
	tasks, err := w.queue.Poll(ctx, 10)
	if err != nil {
		return fmt.Errorf("failed to poll queue: %w", err)
	}

	if len(tasks) == 0 {
		return nil // No tasks available
	}

	// Process each task
	for _, task := range tasks {
		if err := w.processTask(ctx, task); err != nil {
			fmt.Printf("Audit Worker %s: task %s processing failed: %v\n",
				w.workerID, task.ID, err)
		}
	}

	return nil
}

// processTask processes a single audit task with idempotency and retries
func (w *Worker) processTask(ctx context.Context, task AuditTask) error {
	start := time.Now()

	w.metrics.mu.Lock()
	w.metrics.TasksProcessed++
	w.metrics.mu.Unlock()

	// Idempotency check: skip if already processed
	processed, err := w.idempotencyStore.IsProcessed(ctx, task.ID)
	if err != nil {
		fmt.Printf("Audit Worker %s: idempotency check failed for task %s: %v\n",
			w.workerID, task.ID, err)
		// Continue processing (fail-open on idempotency check error)
	} else if processed {
		fmt.Printf("Audit Worker %s: task %s already processed (skipping)\n",
			w.workerID, task.ID)
		w.queue.Ack(ctx, task.ID) // Ack to remove from queue
		return nil
	}

	// Find handler for task type
	handler, ok := w.taskHandlers[task.Type]
	if !ok {
		reason := fmt.Sprintf("no handler for task type: %s", task.Type)
		w.sendToDLQ(ctx, task, reason)
		w.queue.Ack(ctx, task.ID) // Ack to remove from queue (can't process)
		return fmt.Errorf(reason)
	}

	// Execute task with retries
	var lastErr error
	for attempt := 0; attempt <= w.maxRetries; attempt++ {
		if err := handler.Handle(ctx, task); err != nil {
			lastErr = err
			fmt.Printf("Audit Worker %s: task %s attempt %d/%d failed: %v\n",
				w.workerID, task.ID, attempt+1, w.maxRetries+1, err)

			if attempt < w.maxRetries {
				w.metrics.mu.Lock()
				w.metrics.TasksRetried++
				w.metrics.mu.Unlock()

				// Backoff before retry
				time.Sleep(time.Duration(attempt+1) * time.Second)
				continue
			}

			// Max retries exhausted → DLQ
			reason := fmt.Sprintf("max retries exhausted: %v", err)
			w.sendToDLQ(ctx, task, reason)
			w.queue.Ack(ctx, task.ID) // Ack to remove from queue

			w.metrics.mu.Lock()
			w.metrics.TasksFailed++
			w.metrics.mu.Unlock()

			return fmt.Errorf("task failed after %d attempts: %w", w.maxRetries+1, err)
		}

		// Success
		break
	}

	if lastErr == nil {
		// Mark as processed (idempotency)
		if err := w.idempotencyStore.MarkProcessed(ctx, task.ID, 7*24*time.Hour); err != nil {
			fmt.Printf("Audit Worker %s: failed to mark task %s as processed: %v\n",
				w.workerID, task.ID, err)
		}

		// Ack to remove from queue
		if err := w.queue.Ack(ctx, task.ID); err != nil {
			fmt.Printf("Audit Worker %s: failed to ack task %s: %v\n",
				w.workerID, task.ID, err)
		}

		w.metrics.mu.Lock()
		w.metrics.TasksSucceeded++
		w.metrics.LastProcessedAt = time.Now()

		// Update average processing time (exponential moving average)
		duration := time.Since(start).Milliseconds()
		if w.metrics.AvgProcessingTimeMs == 0 {
			w.metrics.AvgProcessingTimeMs = float64(duration)
		} else {
			w.metrics.AvgProcessingTimeMs = 0.9*w.metrics.AvgProcessingTimeMs + 0.1*float64(duration)
		}
		w.metrics.mu.Unlock()

		fmt.Printf("Audit Worker %s: task %s completed successfully (%dms)\n",
			w.workerID, task.ID, time.Since(start).Milliseconds())
	}

	return lastErr
}

// sendToDLQ sends a failed task to the dead letter queue
func (w *Worker) sendToDLQ(ctx context.Context, task AuditTask, reason string) {
	if err := w.dlq.Push(ctx, task, reason); err != nil {
		fmt.Printf("Audit Worker %s: CRITICAL - failed to push task %s to DLQ: %v\n",
			w.workerID, task.ID, err)
	} else {
		w.metrics.mu.Lock()
		w.metrics.TasksDLQd++
		w.metrics.mu.Unlock()

		fmt.Printf("Audit Worker %s: task %s sent to DLQ (reason: %s)\n",
			w.workerID, task.ID, reason)
	}
}

// GetMetrics returns current worker metrics
func (w *Worker) GetMetrics() WorkerMetrics {
	w.metrics.mu.RLock()
	defer w.metrics.mu.RUnlock()
	return *w.metrics
}

// --- Mock Implementations for Testing ---

// MockAuditQueue implements AuditQueue for testing
type MockAuditQueue struct {
	tasks  []AuditTask
	acked  map[string]bool
	nacked map[string]bool
	mu     sync.RWMutex
}

func NewMockAuditQueue() *MockAuditQueue {
	return &MockAuditQueue{
		tasks:  []AuditTask{},
		acked:  make(map[string]bool),
		nacked: make(map[string]bool),
	}
}

func (m *MockAuditQueue) Push(task AuditTask) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tasks = append(m.tasks, task)
}

func (m *MockAuditQueue) Poll(ctx context.Context, count int) ([]AuditTask, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.tasks) == 0 {
		return []AuditTask{}, nil
	}

	n := count
	if n > len(m.tasks) {
		n = len(m.tasks)
	}

	batch := m.tasks[:n]
	m.tasks = m.tasks[n:]

	return batch, nil
}

func (m *MockAuditQueue) Ack(ctx context.Context, taskID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.acked[taskID] = true
	return nil
}

func (m *MockAuditQueue) Nack(ctx context.Context, taskID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nacked[taskID] = true
	return nil
}

// MockDLQ implements DeadLetterQueue for testing
type MockDLQ struct {
	entries []DLQEntry
	mu      sync.RWMutex
}

func NewMockDLQ() *MockDLQ {
	return &MockDLQ{
		entries: []DLQEntry{},
	}
}

func (m *MockDLQ) Push(ctx context.Context, task AuditTask, reason string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	entry := DLQEntry{
		Task:       task,
		Reason:     reason,
		FailedAt:   time.Now(),
		RetryCount: 0,
	}
	m.entries = append(m.entries, entry)
	return nil
}

func (m *MockDLQ) List(ctx context.Context, limit int) ([]DLQEntry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	n := limit
	if n > len(m.entries) {
		n = len(m.entries)
	}

	return m.entries[:n], nil
}

func (m *MockDLQ) Remove(ctx context.Context, taskID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for i, entry := range m.entries {
		if entry.Task.ID == taskID {
			m.entries = append(m.entries[:i], m.entries[i+1:]...)
			return nil
		}
	}

	return fmt.Errorf("task not found in DLQ: %s", taskID)
}

// MockIdempotencyStore implements IdempotencyStore for testing
type MockIdempotencyStore struct {
	processed map[string]bool
	mu        sync.RWMutex
}

func NewMockIdempotencyStore() *MockIdempotencyStore {
	return &MockIdempotencyStore{
		processed: make(map[string]bool),
	}
}

func (m *MockIdempotencyStore) MarkProcessed(ctx context.Context, taskID string, ttl time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.processed[taskID] = true
	return nil
}

func (m *MockIdempotencyStore) IsProcessed(ctx context.Context, taskID string) (bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.processed[taskID], nil
}

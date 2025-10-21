package audit

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// TaskType defines the type of audit task (Phase 5 WP3)
type TaskType string

const (
	// TaskTypeEnrichWORM enriches WORM entries with additional metadata
	TaskTypeEnrichWORM TaskType = "enrich_worm"

	// TaskTypeAnchorBatch anchors a batch of WORM segments with external attestation
	TaskTypeAnchorBatch TaskType = "anchor_batch"

	// TaskTypeAttestExternal writes external attestation (blockchain, timestamp service)
	TaskTypeAttestExternal TaskType = "attest_external"

	// TaskTypeGenerateReport generates audit reports for compliance
	TaskTypeGenerateReport TaskType = "generate_report"

	// TaskTypeCleanupExpired cleans up expired audit logs based on retention policy
	TaskTypeCleanupExpired TaskType = "cleanup_expired"
)

// AuditTask represents a single audit task in the queue
type AuditTask struct {
	ID          string                 `json:"id"`           // Unique task identifier
	Type        TaskType               `json:"type"`         // Task type
	Payload     map[string]interface{} `json:"payload"`      // Task-specific payload
	CreatedAt   time.Time              `json:"created_at"`   // Task creation timestamp
	ScheduledAt time.Time              `json:"scheduled_at"` // When to execute (for delayed tasks)
	Priority    int                    `json:"priority"`     // Priority (0=low, 10=high)
	TenantID    string                 `json:"tenant_id"`    // Associated tenant
	RetryCount  int                    `json:"retry_count"`  // Number of retries attempted
}

// NewAuditTask creates a new audit task
func NewAuditTask(taskType TaskType, payload map[string]interface{}) *AuditTask {
	return &AuditTask{
		ID:          fmt.Sprintf("%s-%d", taskType, time.Now().UnixNano()),
		Type:        taskType,
		Payload:     payload,
		CreatedAt:   time.Now(),
		ScheduledAt: time.Now(), // Execute immediately by default
		Priority:    5,           // Default priority (medium)
		RetryCount:  0,
	}
}

// Serialize serializes the task to JSON
func (t *AuditTask) Serialize() ([]byte, error) {
	return json.Marshal(t)
}

// Deserialize deserializes a task from JSON
func DeserializeTask(data []byte) (*AuditTask, error) {
	var task AuditTask
	if err := json.Unmarshal(data, &task); err != nil {
		return nil, err
	}
	return &task, nil
}

// --- Task Handlers ---

// EnrichWORMHandler handles enrich_worm tasks
type EnrichWORMHandler struct{}

func (h *EnrichWORMHandler) Handle(ctx context.Context, task AuditTask) error {
	// Extract payload fields
	pcsID, ok := task.Payload["pcs_id"].(string)
	if !ok {
		return fmt.Errorf("missing or invalid pcs_id in payload")
	}

	// In production, this would:
	// 1. Read the WORM entry for pcs_id
	// 2. Enrich with additional metadata (e.g., tenant info, policy version)
	// 3. Update the WORM entry (append-only enrichment log)

	fmt.Printf("EnrichWORMHandler: enriching entry %s\n", pcsID)
	// Simulate processing
	time.Sleep(10 * time.Millisecond)

	return nil
}

func (h *EnrichWORMHandler) Type() TaskType {
	return TaskTypeEnrichWORM
}

// AnchorBatchHandler handles anchor_batch tasks
type AnchorBatchHandler struct {
	anchoring *BatchAnchoring
}

func NewAnchorBatchHandler(anchoring *BatchAnchoring) *AnchorBatchHandler {
	return &AnchorBatchHandler{
		anchoring: anchoring,
	}
}

func (h *AnchorBatchHandler) Handle(ctx context.Context, task AuditTask) error {
	// Extract payload fields
	segmentPaths, ok := task.Payload["segment_paths"].([]interface{})
	if !ok {
		return fmt.Errorf("missing or invalid segment_paths in payload")
	}

	// Convert to []string
	paths := make([]string, 0, len(segmentPaths))
	for _, p := range segmentPaths {
		if path, ok := p.(string); ok {
			paths = append(paths, path)
		}
	}

	// In production, this would call the batch anchoring logic
	fmt.Printf("AnchorBatchHandler: anchoring %d segments\n", len(paths))

	return nil
}

func (h *AnchorBatchHandler) Type() TaskType {
	return TaskTypeAnchorBatch
}

// AttestExternalHandler handles attest_external tasks
type AttestExternalHandler struct {
	attestationStore AttestationStore
}

func NewAttestExternalHandler(attestationStore AttestationStore) *AttestExternalHandler {
	return &AttestExternalHandler{
		attestationStore: attestationStore,
	}
}

func (h *AttestExternalHandler) Handle(ctx context.Context, task AuditTask) error {
	// Extract payload fields
	batchID, ok := task.Payload["batch_id"].(string)
	if !ok {
		return fmt.Errorf("missing or invalid batch_id in payload")
	}

	batchRoot, ok := task.Payload["batch_root"].(string)
	if !ok {
		return fmt.Errorf("missing or invalid batch_root in payload")
	}

	// In production, this would:
	// 1. Submit batch_root to blockchain (e.g., Ethereum smart contract)
	// 2. Submit to timestamping service (e.g., RFC 3161)
	// 3. Write attestation record with transaction hash/timestamp token

	fmt.Printf("AttestExternalHandler: attesting batch %s (root %s)\n", batchID, batchRoot)

	// Simulate external attestation
	time.Sleep(100 * time.Millisecond)

	// Write attestation (mock)
	attestation := Attestation{
		BatchID:         batchID,
		BatchRoot:       batchRoot,
		AncoredAt:       time.Now(),
		AttestationType: "blockchain",
		AttestationData: fmt.Sprintf("tx-hash=0x%x", time.Now().UnixNano()),
	}

	if err := h.attestationStore.WriteAttestation(ctx, attestation); err != nil {
		return fmt.Errorf("failed to write attestation: %w", err)
	}

	return nil
}

func (h *AttestExternalHandler) Type() TaskType {
	return TaskTypeAttestExternal
}

// GenerateReportHandler handles generate_report tasks
type GenerateReportHandler struct{}

func (h *GenerateReportHandler) Handle(ctx context.Context, task AuditTask) error {
	// Extract payload fields
	reportType, ok := task.Payload["report_type"].(string)
	if !ok {
		return fmt.Errorf("missing or invalid report_type in payload")
	}

	startDate, ok := task.Payload["start_date"].(string)
	if !ok {
		return fmt.Errorf("missing or invalid start_date in payload")
	}

	endDate, ok := task.Payload["end_date"].(string)
	if !ok {
		return fmt.Errorf("missing or invalid end_date in payload")
	}

	// In production, this would:
	// 1. Query WORM entries for date range
	// 2. Generate compliance report (CSV, PDF, JSON)
	// 3. Store report in object storage
	// 4. Notify requester (email, webhook)

	fmt.Printf("GenerateReportHandler: generating %s report (%s to %s)\n",
		reportType, startDate, endDate)

	// Simulate report generation
	time.Sleep(500 * time.Millisecond)

	return nil
}

func (h *GenerateReportHandler) Type() TaskType {
	return TaskTypeGenerateReport
}

// CleanupExpiredHandler handles cleanup_expired tasks
type CleanupExpiredHandler struct {
	wormStore WORMStore
}

func NewCleanupExpiredHandler(wormStore WORMStore) *CleanupExpiredHandler {
	return &CleanupExpiredHandler{
		wormStore: wormStore,
	}
}

func (h *CleanupExpiredHandler) Handle(ctx context.Context, task AuditTask) error {
	// Extract payload fields
	retentionDays, ok := task.Payload["retention_days"].(float64)
	if !ok {
		return fmt.Errorf("missing or invalid retention_days in payload")
	}

	// In production, this would:
	// 1. List WORM segments older than retention period
	// 2. Archive to cold storage (Glacier, tape)
	// 3. Delete from active WORM store
	// 4. Log cleanup action for compliance

	fmt.Printf("CleanupExpiredHandler: cleaning up entries older than %d days\n",
		int(retentionDays))

	// Simulate cleanup
	time.Sleep(200 * time.Millisecond)

	return nil
}

func (h *CleanupExpiredHandler) Type() TaskType {
	return TaskTypeCleanupExpired
}

// --- Task Queue Backlog Metrics ---

// BacklogMetrics tracks queue backlog and age
type BacklogMetrics struct {
	QueueSize        int64         // Number of tasks in queue
	OldestTaskAge    time.Duration // Age of oldest task
	P95TaskAge       time.Duration // 95th percentile task age
	HighPrioritySize int64         // Number of high-priority tasks
}

// ComputeBacklogMetrics computes backlog metrics from a list of tasks
func ComputeBacklogMetrics(tasks []AuditTask) BacklogMetrics {
	if len(tasks) == 0 {
		return BacklogMetrics{}
	}

	now := time.Now()
	ages := make([]time.Duration, 0, len(tasks))
	highPriorityCount := int64(0)

	for _, task := range tasks {
		age := now.Sub(task.CreatedAt)
		ages = append(ages, age)

		if task.Priority >= 8 {
			highPriorityCount++
		}
	}

	// Find oldest
	oldest := ages[0]
	for _, age := range ages {
		if age > oldest {
			oldest = age
		}
	}

	// Compute p95 (simplified: 95th element in sorted list)
	// In production, use a proper percentile algorithm
	p95 := oldest // Placeholder

	return BacklogMetrics{
		QueueSize:        int64(len(tasks)),
		OldestTaskAge:    oldest,
		P95TaskAge:       p95,
		HighPrioritySize: highPriorityCount,
	}
}

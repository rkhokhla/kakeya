package tiering

import (
	"bytes"
	"compress/zlib"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/fractal-lba/backend/internal/api"
)

// ColdDriver implements StorageDriver for object storage (S3/GCS) - Phase 5 WP2
type ColdDriver struct {
	mu               sync.RWMutex
	objectStore      ObjectStore
	bucket           string
	prefix           string // Key prefix for namespacing (e.g., "cold/")
	compressionLevel int    // zlib compression level (0=none, 1-9)
	encryption       bool   // Server-side encryption enabled
	lifecyclePolicy  *LifecyclePolicy
	metrics          *ColdDriverMetrics
}

// ColdDriverMetrics tracks cold tier operations
type ColdDriverMetrics struct {
	mu                    sync.RWMutex
	GetRequests           int64
	SetRequests           int64
	DeleteRequests        int64
	BytesRead             int64
	BytesWritten          int64
	CompressionRatio      float64 // Average compression ratio
	AvgLatencyMs          float64
	Errors                int64
}

// ObjectStore abstracts S3/GCS operations
type ObjectStore interface {
	Get(ctx context.Context, bucket, key string) ([]byte, error)
	Put(ctx context.Context, bucket, key string, data []byte, opts *PutOptions) error
	Delete(ctx context.Context, bucket, key string) error
	Exists(ctx context.Context, bucket, key string) (bool, error)
	SetLifecyclePolicy(ctx context.Context, bucket string, policy *LifecyclePolicy) error
}

// PutOptions configures object storage write operations
type PutOptions struct {
	ContentType      string
	ServerSideEncrypt bool
	Metadata         map[string]string
}

// LifecyclePolicy defines object lifecycle rules
type LifecyclePolicy struct {
	Rules []LifecycleRule
}

// LifecycleRule defines a single lifecycle action
type LifecycleRule struct {
	ID              string
	Prefix          string
	DaysToTransition int    // Days before transitioning to cheaper storage class (Glacier, etc.)
	DaysToExpire     int    // Days before deletion (0 = no expiration)
	StorageClass     string // Target storage class (e.g., "GLACIER", "COLDLINE")
}

// ColdDriverConfig holds configuration
type ColdDriverConfig struct {
	ObjectStore      ObjectStore
	Bucket           string
	Prefix           string
	CompressionLevel int  // 0 (disabled) to 9 (max compression)
	Encryption       bool // Server-side encryption
	LifecyclePolicy  *LifecyclePolicy
}

// NewColdDriver creates a new cold tier storage driver
func NewColdDriver(config ColdDriverConfig) (*ColdDriver, error) {
	if config.ObjectStore == nil {
		return nil, fmt.Errorf("ObjectStore is required")
	}
	if config.Bucket == "" {
		return nil, fmt.Errorf("Bucket is required")
	}

	if config.Prefix == "" {
		config.Prefix = "cold/" // Default prefix
	}
	if config.CompressionLevel < 0 || config.CompressionLevel > 9 {
		config.CompressionLevel = 6 // Default: balanced compression
	}

	driver := &ColdDriver{
		objectStore:      config.ObjectStore,
		bucket:           config.Bucket,
		prefix:           config.Prefix,
		compressionLevel: config.CompressionLevel,
		encryption:       config.Encryption,
		lifecyclePolicy:  config.LifecyclePolicy,
		metrics:          &ColdDriverMetrics{},
	}

	// Apply lifecycle policy if provided
	if config.LifecyclePolicy != nil {
		ctx := context.Background()
		if err := driver.objectStore.SetLifecyclePolicy(ctx, driver.bucket, config.LifecyclePolicy); err != nil {
			return nil, fmt.Errorf("failed to set lifecycle policy: %w", err)
		}
		fmt.Printf("Cold Driver: lifecycle policy applied (bucket %s)\n", driver.bucket)
	}

	return driver, nil
}

// Get retrieves a value from cold storage
func (c *ColdDriver) Get(ctx context.Context, key string) (*api.VerifyResult, error) {
	start := time.Now()
	defer func() {
		latency := time.Since(start).Milliseconds()
		c.updateAvgLatency(float64(latency))
	}()

	objectKey := c.prefix + key

	// Fetch from object storage
	data, err := c.objectStore.Get(ctx, c.bucket, objectKey)
	if err != nil {
		c.metrics.mu.Lock()
		c.metrics.Errors++
		c.metrics.mu.Unlock()
		return nil, fmt.Errorf("cold tier get failed: %w", err)
	}

	c.metrics.mu.Lock()
	c.metrics.GetRequests++
	c.metrics.BytesRead += int64(len(data))
	c.metrics.mu.Unlock()

	// Decompress if compression is enabled
	if c.compressionLevel > 0 {
		decompressed, err := c.decompress(data)
		if err != nil {
			return nil, fmt.Errorf("decompression failed: %w", err)
		}
		data = decompressed
	}

	// Deserialize
	var result api.VerifyResult
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("deserialization failed: %w", err)
	}

	return &result, nil
}

// Set stores a value in cold storage
func (c *ColdDriver) Set(ctx context.Context, key string, value *api.VerifyResult, ttl time.Duration) error {
	start := time.Now()
	defer func() {
		latency := time.Since(start).Milliseconds()
		c.updateAvgLatency(float64(latency))
	}()

	// Serialize
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("serialization failed: %w", err)
	}

	originalSize := len(data)

	// Compress if compression is enabled
	if c.compressionLevel > 0 {
		compressed, err := c.compress(data)
		if err != nil {
			// Fallback: store uncompressed on compression error
			fmt.Printf("Cold Driver: compression failed (storing uncompressed): %v\n", err)
		} else {
			data = compressed
			ratio := float64(len(compressed)) / float64(originalSize)
			c.metrics.mu.Lock()
			c.metrics.CompressionRatio = (c.metrics.CompressionRatio + ratio) / 2.0 // Running average
			c.metrics.mu.Unlock()
		}
	}

	objectKey := c.prefix + key

	// Metadata for lifecycle management
	metadata := map[string]string{
		"pcs_id":        key,
		"created_at":    time.Now().Format(time.RFC3339),
		"compressed":    fmt.Sprintf("%t", c.compressionLevel > 0),
		"original_size": fmt.Sprintf("%d", originalSize),
	}

	// Note: TTL is enforced by lifecycle policy (DaysToExpire), not per-object metadata
	// S3/GCS don't support per-object TTL like Redis; use lifecycle rules instead

	opts := &PutOptions{
		ContentType:      "application/json",
		ServerSideEncrypt: c.encryption,
		Metadata:         metadata,
	}

	// Upload to object storage
	if err := c.objectStore.Put(ctx, c.bucket, objectKey, data, opts); err != nil {
		c.metrics.mu.Lock()
		c.metrics.Errors++
		c.metrics.mu.Unlock()
		return fmt.Errorf("cold tier put failed: %w", err)
	}

	c.metrics.mu.Lock()
	c.metrics.SetRequests++
	c.metrics.BytesWritten += int64(len(data))
	c.metrics.mu.Unlock()

	return nil
}

// Delete removes a value from cold storage
func (c *ColdDriver) Delete(ctx context.Context, key string) error {
	objectKey := c.prefix + key

	if err := c.objectStore.Delete(ctx, c.bucket, objectKey); err != nil {
		c.metrics.mu.Lock()
		c.metrics.Errors++
		c.metrics.mu.Unlock()
		return fmt.Errorf("cold tier delete failed: %w", err)
	}

	c.metrics.mu.Lock()
	c.metrics.DeleteRequests++
	c.metrics.mu.Unlock()

	return nil
}

// Exists checks if a key exists in cold storage
func (c *ColdDriver) Exists(ctx context.Context, key string) (bool, error) {
	objectKey := c.prefix + key
	return c.objectStore.Exists(ctx, c.bucket, objectKey)
}

// compress compresses data using zlib
func (c *ColdDriver) compress(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	w, err := zlib.NewWriterLevel(&buf, c.compressionLevel)
	if err != nil {
		return nil, err
	}

	if _, err := w.Write(data); err != nil {
		return nil, err
	}

	if err := w.Close(); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// decompress decompresses zlib data
func (c *ColdDriver) decompress(data []byte) ([]byte, error) {
	r, err := zlib.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// updateAvgLatency updates the running average latency
func (c *ColdDriver) updateAvgLatency(latencyMs float64) {
	c.metrics.mu.Lock()
	defer c.metrics.mu.Unlock()

	// Exponential moving average (alpha = 0.1)
	if c.metrics.AvgLatencyMs == 0 {
		c.metrics.AvgLatencyMs = latencyMs
	} else {
		c.metrics.AvgLatencyMs = 0.9*c.metrics.AvgLatencyMs + 0.1*latencyMs
	}
}

// GetMetrics returns current metrics
func (c *ColdDriver) GetMetrics() ColdDriverMetrics {
	c.metrics.mu.RLock()
	defer c.metrics.mu.RUnlock()
	return *c.metrics
}

// --- Mock ObjectStore for testing ---

// MockObjectStore implements ObjectStore for testing
type MockObjectStore struct {
	data   map[string][]byte
	mu     sync.RWMutex
	errors map[string]error // Inject errors for testing
}

func NewMockObjectStore() *MockObjectStore {
	return &MockObjectStore{
		data:   make(map[string][]byte),
		errors: make(map[string]error),
	}
}

func (m *MockObjectStore) Get(ctx context.Context, bucket, key string) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if err, ok := m.errors["Get"]; ok {
		return nil, err
	}

	fullKey := bucket + "/" + key
	data, ok := m.data[fullKey]
	if !ok {
		return nil, fmt.Errorf("object not found: %s", key)
	}
	return data, nil
}

func (m *MockObjectStore) Put(ctx context.Context, bucket, key string, data []byte, opts *PutOptions) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if err, ok := m.errors["Put"]; ok {
		return err
	}

	fullKey := bucket + "/" + key
	m.data[fullKey] = data
	return nil
}

func (m *MockObjectStore) Delete(ctx context.Context, bucket, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if err, ok := m.errors["Delete"]; ok {
		return err
	}

	fullKey := bucket + "/" + key
	delete(m.data, fullKey)
	return nil
}

func (m *MockObjectStore) Exists(ctx context.Context, bucket, key string) (bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if err, ok := m.errors["Exists"]; ok {
		return false, err
	}

	fullKey := bucket + "/" + key
	_, ok := m.data[fullKey]
	return ok, nil
}

func (m *MockObjectStore) SetLifecyclePolicy(ctx context.Context, bucket string, policy *LifecyclePolicy) error {
	// Mock: just log the policy
	fmt.Printf("MockObjectStore: lifecycle policy set for bucket %s (rules: %d)\n",
		bucket, len(policy.Rules))
	return nil
}

func (m *MockObjectStore) InjectError(operation string, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.errors[operation] = err
}

func (m *MockObjectStore) ClearErrors() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.errors = make(map[string]error)
}

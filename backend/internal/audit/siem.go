package audit

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// SIEMStreamer streams audit events to SIEM platforms (Phase 6 WP3)
type SIEMStreamer struct {
	mu            sync.RWMutex
	config        *SIEMConfig
	httpClient    *http.Client
	buffer        []SIEMEvent
	bufferSize    int
	flushInterval time.Duration
	metrics       *SIEMMetrics
	stopCh        chan struct{}
}

// SIEMConfig defines SIEM integration configuration
type SIEMConfig struct {
	// Provider is the SIEM platform (splunk, datadog, elastic, sumo)
	Provider string

	// Endpoint is the SIEM ingestion URL
	Endpoint string

	// APIKey for authentication
	APIKey string

	// SourceType for event categorization
	SourceType string

	// BatchSize for event batching (default: 100)
	BatchSize int

	// FlushInterval for periodic flush (default: 10s)
	FlushInterval time.Duration

	// TLSConfig for secure transport
	TLSConfig *tls.Config

	// CustomFields to add to every event
	CustomFields map[string]interface{}
}

// SIEMEvent represents an audit event for SIEM
type SIEMEvent struct {
	Timestamp   time.Time              `json:"timestamp"`
	EventType   string                 `json:"event_type"`
	Severity    string                 `json:"severity"` // info, warning, error, critical
	Source      string                 `json:"source"`
	TenantID    string                 `json:"tenant_id,omitempty"`
	PCSID       string                 `json:"pcs_id,omitempty"`
	RegionID    string                 `json:"region_id,omitempty"`
	Message     string                 `json:"message"`
	Details     map[string]interface{} `json:"details,omitempty"`
	Correlation string                 `json:"correlation_id,omitempty"`
}

// SIEMMetrics tracks SIEM streaming metrics
type SIEMMetrics struct {
	mu               sync.RWMutex
	EventsSent       int64
	EventsFailed     int64
	FlushesCompleted int64
	FlushesFailed    int64
	LastFlushTime    time.Time
	LastError        string
}

// NewSIEMStreamer creates a new SIEM streamer
func NewSIEMStreamer(config *SIEMConfig) *SIEMStreamer {
	if config.BatchSize == 0 {
		config.BatchSize = 100
	}
	if config.FlushInterval == 0 {
		config.FlushInterval = 10 * time.Second
	}

	return &SIEMStreamer{
		config: config,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				TLSClientConfig: config.TLSConfig,
			},
		},
		buffer:        make([]SIEMEvent, 0, config.BatchSize),
		bufferSize:    config.BatchSize,
		flushInterval: config.FlushInterval,
		metrics:       &SIEMMetrics{},
		stopCh:        make(chan struct{}),
	}
}

// Start starts the SIEM streamer background flush loop
func (ss *SIEMStreamer) Start(ctx context.Context) {
	ticker := time.NewTicker(ss.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := ss.Flush(ctx); err != nil {
				ss.recordError(fmt.Sprintf("Flush error: %v", err))
			}
		case <-ss.stopCh:
			// Final flush
			_ = ss.Flush(ctx)
			return
		case <-ctx.Done():
			_ = ss.Flush(ctx)
			return
		}
	}
}

// Stop stops the SIEM streamer
func (ss *SIEMStreamer) Stop() {
	close(ss.stopCh)
}

// Send adds an event to the buffer
func (ss *SIEMStreamer) Send(ctx context.Context, event SIEMEvent) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	// Add custom fields
	if ss.config.CustomFields != nil {
		if event.Details == nil {
			event.Details = make(map[string]interface{})
		}
		for k, v := range ss.config.CustomFields {
			event.Details[k] = v
		}
	}

	ss.buffer = append(ss.buffer, event)

	// Flush if buffer is full
	if len(ss.buffer) >= ss.bufferSize {
		return ss.flush(ctx)
	}

	return nil
}

// Flush sends buffered events to SIEM
func (ss *SIEMStreamer) Flush(ctx context.Context) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	return ss.flush(ctx)
}

// flush (internal, assumes lock held)
func (ss *SIEMStreamer) flush(ctx context.Context) error {
	if len(ss.buffer) == 0 {
		return nil
	}

	// Take snapshot of buffer
	events := make([]SIEMEvent, len(ss.buffer))
	copy(events, ss.buffer)
	ss.buffer = ss.buffer[:0] // Clear buffer

	// Send to SIEM based on provider
	var err error
	switch ss.config.Provider {
	case "splunk":
		err = ss.sendToSplunk(ctx, events)
	case "datadog":
		err = ss.sendToDatadog(ctx, events)
	case "elastic":
		err = ss.sendToElastic(ctx, events)
	case "sumo":
		err = ss.sendToSumoLogic(ctx, events)
	default:
		err = fmt.Errorf("unsupported SIEM provider: %s", ss.config.Provider)
	}

	// Update metrics
	ss.metrics.mu.Lock()
	if err != nil {
		ss.metrics.FlushesFailed++
		ss.metrics.EventsFailed += int64(len(events))
		ss.metrics.LastError = err.Error()
	} else {
		ss.metrics.FlushesCompleted++
		ss.metrics.EventsSent += int64(len(events))
		ss.metrics.LastFlushTime = time.Now()
	}
	ss.metrics.mu.Unlock()

	return err
}

// sendToSplunk sends events to Splunk HEC (HTTP Event Collector)
func (ss *SIEMStreamer) sendToSplunk(ctx context.Context, events []SIEMEvent) error {
	// Splunk HEC format: one JSON object per line
	var payload bytes.Buffer
	for _, event := range events {
		hecEvent := map[string]interface{}{
			"time":       event.Timestamp.Unix(),
			"sourcetype": ss.config.SourceType,
			"event":      event,
		}
		if err := json.NewEncoder(&payload).Encode(hecEvent); err != nil {
			return fmt.Errorf("failed to encode Splunk HEC event: %w", err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", ss.config.Endpoint, &payload)
	if err != nil {
		return fmt.Errorf("failed to create Splunk request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Splunk %s", ss.config.APIKey))
	req.Header.Set("Content-Type", "application/json")

	resp, err := ss.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("Splunk request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("Splunk returned status %d", resp.StatusCode)
	}

	return nil
}

// sendToDatadog sends events to Datadog Logs API
func (ss *SIEMStreamer) sendToDatadog(ctx context.Context, events []SIEMEvent) error {
	// Datadog Logs API format: JSON array
	payload, err := json.Marshal(events)
	if err != nil {
		return fmt.Errorf("failed to marshal Datadog events: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", ss.config.Endpoint, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("failed to create Datadog request: %w", err)
	}

	req.Header.Set("DD-API-KEY", ss.config.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := ss.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("Datadog request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("Datadog returned status %d", resp.StatusCode)
	}

	return nil
}

// sendToElastic sends events to Elasticsearch Bulk API
func (ss *SIEMStreamer) sendToElastic(ctx context.Context, events []SIEMEvent) error {
	// Elasticsearch Bulk API format: action_and_meta_data\n + optional_source\n
	var payload bytes.Buffer
	for _, event := range events {
		// Index action
		action := map[string]interface{}{
			"index": map[string]interface{}{
				"_index": ss.config.SourceType,
			},
		}
		if err := json.NewEncoder(&payload).Encode(action); err != nil {
			return fmt.Errorf("failed to encode Elastic action: %w", err)
		}

		// Document
		if err := json.NewEncoder(&payload).Encode(event); err != nil {
			return fmt.Errorf("failed to encode Elastic event: %w", err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", ss.config.Endpoint+"/_bulk", &payload)
	if err != nil {
		return fmt.Errorf("failed to create Elastic request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("ApiKey %s", ss.config.APIKey))
	req.Header.Set("Content-Type", "application/x-ndjson")

	resp, err := ss.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("Elastic request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("Elastic returned status %d", resp.StatusCode)
	}

	return nil
}

// sendToSumoLogic sends events to Sumo Logic HTTP Collector
func (ss *SIEMStreamer) sendToSumoLogic(ctx context.Context, events []SIEMEvent) error {
	// Sumo Logic format: JSON array
	payload, err := json.Marshal(events)
	if err != nil {
		return fmt.Errorf("failed to marshal Sumo Logic events: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", ss.config.Endpoint, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("failed to create Sumo Logic request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	// Sumo Logic uses URL with embedded auth token

	resp, err := ss.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("Sumo Logic request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("Sumo Logic returned status %d", resp.StatusCode)
	}

	return nil
}

// recordError records an error in metrics
func (ss *SIEMStreamer) recordError(msg string) {
	ss.metrics.mu.Lock()
	ss.metrics.LastError = msg
	ss.metrics.mu.Unlock()
	fmt.Printf("SIEM Error: %s\n", msg)
}

// GetMetrics returns SIEM streaming metrics
func (ss *SIEMStreamer) GetMetrics() SIEMMetrics {
	ss.metrics.mu.RLock()
	defer ss.metrics.mu.RUnlock()
	return *ss.metrics
}

// --- Event Builders ---

// NewWORMEvent creates a SIEM event for WORM log writes
func NewWORMEvent(pcsID, tenantID, regionID string, entry *WORMEntry) SIEMEvent {
	return SIEMEvent{
		Timestamp: time.Now(),
		EventType: "worm.write",
		Severity:  "info",
		Source:    "fractal-lba",
		TenantID:  tenantID,
		PCSID:     pcsID,
		RegionID:  regionID,
		Message:   "PCS written to immutable audit log",
		Details: map[string]interface{}{
			"merkle_root":      entry.MerkleRoot,
			"D_hat":            entry.DHat,
			"coh_star":         entry.CohStar,
			"r":                entry.R,
			"budget":           entry.Budget,
			"regime":           entry.Regime,
			"verify_outcome":   entry.VerifyOutcome,
			"policy_version":   entry.PolicyVersion,
			"entry_hash":       entry.EntryHash,
		},
	}
}

// NewAnchoringEvent creates a SIEM event for batch anchoring
func NewAnchoringEvent(batchID, regionID string, segmentCount int, merkleRoot string) SIEMEvent {
	return SIEMEvent{
		Timestamp: time.Now(),
		EventType: "anchoring.batch",
		Severity:  "info",
		Source:    "fractal-lba",
		RegionID:  regionID,
		Message:   fmt.Sprintf("Anchored %d audit segments", segmentCount),
		Details: map[string]interface{}{
			"batch_id":      batchID,
			"segment_count": segmentCount,
			"merkle_root":   merkleRoot,
		},
		Correlation: batchID,
	}
}

// NewCRREvent creates a SIEM event for cross-region replication
func NewCRREvent(sourceRegion, targetRegion string, segmentCount int, status string) SIEMEvent {
	severity := "info"
	if status == "failed" {
		severity = "error"
	}

	return SIEMEvent{
		Timestamp: time.Now(),
		EventType: "crr.ship",
		Severity:  severity,
		Source:    "fractal-lba",
		RegionID:  sourceRegion,
		Message:   fmt.Sprintf("CRR %s: %d segments %s→%s", status, segmentCount, sourceRegion, targetRegion),
		Details: map[string]interface{}{
			"source_region": sourceRegion,
			"target_region": targetRegion,
			"segment_count": segmentCount,
			"status":        status,
		},
	}
}

// NewDivergenceEvent creates a SIEM event for divergence detection
func NewDivergenceEvent(region1, region2 string, divergencePct, mismatchPct float64) SIEMEvent {
	severity := "warning"
	if divergencePct > 10.0 || mismatchPct > 20.0 {
		severity = "critical"
	}

	return SIEMEvent{
		Timestamp: time.Now(),
		EventType: "crr.divergence",
		Severity:  severity,
		Source:    "fractal-lba",
		Message:   fmt.Sprintf("Divergence detected: %s vs %s (%.2f%% count, %.2f%% mismatch)", region1, region2, divergencePct, mismatchPct),
		Details: map[string]interface{}{
			"region1":           region1,
			"region2":           region2,
			"count_divergence":  divergencePct,
			"sample_mismatch":   mismatchPct,
		},
	}
}

// NewReconcileEvent creates a SIEM event for auto-reconciliation
func NewReconcileEvent(proposalID, region1, region2, action, status string, safetyScore float64) SIEMEvent {
	severity := "info"
	if status == "failed" {
		severity = "error"
	} else if status == "pending" {
		severity = "warning"
	}

	return SIEMEvent{
		Timestamp: time.Now(),
		EventType: "crr.reconcile",
		Severity:  severity,
		Source:    "fractal-lba",
		Message:   fmt.Sprintf("Reconciliation %s: %s for %s vs %s (safety: %.2f)", status, action, region1, region2, safetyScore),
		Details: map[string]interface{}{
			"proposal_id":  proposalID,
			"region1":      region1,
			"region2":      region2,
			"action":       action,
			"status":       status,
			"safety_score": safetyScore,
		},
		Correlation: proposalID,
	}
}

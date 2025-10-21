package cost

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// Phase 8 WP3: Cloud billing importers with auto-reconciliation

// BillingImporter imports cloud billing data from AWS/GCP/Azure
type BillingImporter struct {
	mu                sync.RWMutex
	provider          string // "aws", "gcp", "azure"
	dataSource        BillingDataSource
	reconciler        *BillingReconciler
	lastImportTime    time.Time
	importInterval    time.Duration
	metrics           *ImporterMetrics
	runningImports    map[string]*ImportJob
}

// BillingDataSource abstracts cloud billing export sources
type BillingDataSource interface {
	// FetchBillingRecords fetches billing records for a time range
	FetchBillingRecords(ctx context.Context, startTime, endTime time.Time) ([]*CloudBillingRecord, error)
	// GetProvider returns the cloud provider name
	GetProvider() string
}

// CloudBillingRecord represents a single billing line item
type CloudBillingRecord struct {
	Provider        string    // "aws", "gcp", "azure"
	AccountID       string
	ServiceName     string    // "EC2", "S3", "Cloud Storage", etc.
	ResourceID      string
	UsageType       string    // "Compute", "Storage", "Network", "Anchoring"
	UsageAmount     float64
	Cost            float64   // USD
	Currency        string
	BillingPeriod   string    // "2025-01"
	RecordTimestamp time.Time
	Tags            map[string]string // Resource tags for attribution
}

// BillingReconciler reconciles internal cost estimates with actual cloud bills
type BillingReconciler struct {
	mu                 sync.RWMutex
	internalTracer     *Tracer // Phase 7 cost tracer
	cloudRecords       []*CloudBillingRecord
	reconciliations    []*ReconciliationReport
	targetAccuracy     float64 // ±3% target from Phase 7
	alertThreshold     float64 // Alert if error > 5%
	reconcileInterval  time.Duration
	metrics            *ReconcilerMetrics
}

// ReconciliationReport represents a reconciliation between internal and cloud costs
type ReconciliationReport struct {
	BillingPeriod     string
	Provider          string
	InternalCostUSD   float64
	CloudCostUSD      float64
	DifferenceUSD     float64
	DifferencePercent float64
	ReconciledAt      time.Time
	Status            string // "within_target", "drift_detected", "critical_drift"
	Breakdown         map[string]*CostDelta // service_name → delta
	Recommendations   []string
}

// CostDelta represents cost difference for a specific service
type CostDelta struct {
	ServiceName       string
	InternalCostUSD   float64
	CloudCostUSD      float64
	DifferenceUSD     float64
	DifferencePercent float64
}

// ImportJob tracks an ongoing import job
type ImportJob struct {
	JobID        string
	Provider     string
	StartTime    time.Time
	EndTime      time.Time
	Status       string // "pending", "running", "completed", "failed"
	RecordsCount int
	CreatedAt    time.Time
	CompletedAt  time.Time
	Error        string
}

// ImporterMetrics tracks importer performance
type ImporterMetrics struct {
	mu                  sync.RWMutex
	TotalImports        int64
	SuccessfulImports   int64
	FailedImports       int64
	RecordsImported     int64
	LastImportDuration  time.Duration
	LastImportTimestamp time.Time
}

// ReconcilerMetrics tracks reconciliation performance
type ReconcilerMetrics struct {
	mu                      sync.RWMutex
	TotalReconciliations    int64
	WithinTarget            int64 // ±3%
	DriftDetected           int64 // >3%, <5%
	CriticalDrift           int64 // >5%
	AvgDifferencePercent    float64
	LastReconciliationTime  time.Time
}

// NewBillingImporter creates a new billing importer
func NewBillingImporter(provider string, dataSource BillingDataSource, tracer *Tracer) *BillingImporter {
	return &BillingImporter{
		provider:       provider,
		dataSource:     dataSource,
		reconciler:     NewBillingReconciler(tracer),
		importInterval: 24 * time.Hour, // Daily imports
		metrics:        &ImporterMetrics{},
		runningImports: make(map[string]*ImportJob),
	}
}

// NewBillingReconciler creates a new billing reconciler
func NewBillingReconciler(tracer *Tracer) *BillingReconciler {
	return &BillingReconciler{
		internalTracer:    tracer,
		cloudRecords:      []*CloudBillingRecord{},
		reconciliations:   []*ReconciliationReport{},
		targetAccuracy:    0.03, // ±3% from Phase 7
		alertThreshold:    0.05, // Alert if >5%
		reconcileInterval: 24 * time.Hour,
		metrics:           &ReconcilerMetrics{},
	}
}

// StartPeriodicImport starts periodic billing import
func (bi *BillingImporter) StartPeriodicImport(ctx context.Context) error {
	fmt.Printf("Starting periodic billing import: provider=%s, interval=%v\n", bi.provider, bi.importInterval)

	ticker := time.NewTicker(bi.importInterval)
	defer ticker.Stop()

	// Initial import
	if err := bi.runImport(ctx); err != nil {
		fmt.Printf("Initial import failed: %v\n", err)
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := bi.runImport(ctx); err != nil {
				fmt.Printf("Periodic import failed: %v\n", err)
			}
		}
	}
}

// runImport executes a billing import job
func (bi *BillingImporter) runImport(ctx context.Context) error {
	startTime := time.Now()

	bi.metrics.mu.Lock()
	bi.metrics.TotalImports++
	bi.metrics.mu.Unlock()

	// Create import job
	jobID := fmt.Sprintf("%s-import-%d", bi.provider, time.Now().Unix())
	job := &ImportJob{
		JobID:     jobID,
		Provider:  bi.provider,
		StartTime: startTime.Add(-24 * time.Hour), // Last 24 hours
		EndTime:   startTime,
		Status:    "running",
		CreatedAt: startTime,
	}

	bi.mu.Lock()
	bi.runningImports[jobID] = job
	bi.mu.Unlock()

	// Fetch billing records
	records, err := bi.dataSource.FetchBillingRecords(ctx, job.StartTime, job.EndTime)
	if err != nil {
		job.Status = "failed"
		job.Error = err.Error()
		job.CompletedAt = time.Now()

		bi.metrics.mu.Lock()
		bi.metrics.FailedImports++
		bi.metrics.mu.Unlock()

		return fmt.Errorf("failed to fetch billing records: %w", err)
	}

	// Store records in reconciler
	bi.reconciler.AddCloudRecords(records)

	// Update job status
	job.Status = "completed"
	job.RecordsCount = len(records)
	job.CompletedAt = time.Now()

	duration := time.Since(startTime)

	bi.mu.Lock()
	bi.lastImportTime = startTime
	bi.mu.Unlock()

	bi.metrics.mu.Lock()
	bi.metrics.SuccessfulImports++
	bi.metrics.RecordsImported += int64(len(records))
	bi.metrics.LastImportDuration = duration
	bi.metrics.LastImportTimestamp = startTime
	bi.metrics.mu.Unlock()

	fmt.Printf("Import completed: job=%s, provider=%s, records=%d, duration=%v\n",
		jobID, bi.provider, len(records), duration)

	// Trigger reconciliation
	if err := bi.reconciler.Reconcile(ctx, job.StartTime, job.EndTime); err != nil {
		fmt.Printf("Reconciliation failed: %v\n", err)
	}

	return nil
}

// AddCloudRecords adds cloud billing records to reconciler
func (br *BillingReconciler) AddCloudRecords(records []*CloudBillingRecord) {
	br.mu.Lock()
	defer br.mu.Unlock()

	br.cloudRecords = append(br.cloudRecords, records...)
}

// Reconcile reconciles internal cost estimates with cloud billing
func (br *BillingReconciler) Reconcile(ctx context.Context, startTime, endTime time.Time) error {
	startReconcile := time.Now()

	br.metrics.mu.Lock()
	br.metrics.TotalReconciliations++
	br.metrics.mu.Unlock()

	// Get internal cost estimate from Phase 7 tracer
	internalCost := br.getInternalCost(startTime, endTime)

	// Get cloud cost from billing records
	cloudCost := br.getCloudCost(startTime, endTime)

	// Compute difference
	differenceUSD := cloudCost - internalCost
	differencePercent := 0.0
	if cloudCost > 0 {
		differencePercent = (differenceUSD / cloudCost) * 100
	}

	// Determine status
	status := "within_target"
	if differencePercent > br.targetAccuracy*100 {
		if differencePercent > br.alertThreshold*100 {
			status = "critical_drift"
		} else {
			status = "drift_detected"
		}
	}

	// Build breakdown by service
	breakdown := br.computeBreakdown(startTime, endTime)

	// Generate recommendations
	recommendations := br.generateRecommendations(differencePercent, breakdown)

	report := &ReconciliationReport{
		BillingPeriod:     fmt.Sprintf("%d-%02d", startTime.Year(), startTime.Month()),
		Provider:          br.getProviderFromRecords(),
		InternalCostUSD:   internalCost,
		CloudCostUSD:      cloudCost,
		DifferenceUSD:     differenceUSD,
		DifferencePercent: differencePercent,
		ReconciledAt:      startReconcile,
		Status:            status,
		Breakdown:         breakdown,
		Recommendations:   recommendations,
	}

	br.mu.Lock()
	br.reconciliations = append(br.reconciliations, report)
	br.mu.Unlock()

	// Update metrics
	br.metrics.mu.Lock()
	if status == "within_target" {
		br.metrics.WithinTarget++
	} else if status == "drift_detected" {
		br.metrics.DriftDetected++
	} else {
		br.metrics.CriticalDrift++
	}
	br.metrics.AvgDifferencePercent = (br.metrics.AvgDifferencePercent*float64(br.metrics.TotalReconciliations-1) + differencePercent) / float64(br.metrics.TotalReconciliations)
	br.metrics.LastReconciliationTime = startReconcile
	br.metrics.mu.Unlock()

	fmt.Printf("Reconciliation completed: status=%s, internal=$%.2f, cloud=$%.2f, diff=%.1f%%\n",
		status, internalCost, cloudCost, differencePercent)

	return nil
}

// getInternalCost retrieves internal cost estimate from Phase 7 tracer
func (br *BillingReconciler) getInternalCost(startTime, endTime time.Time) float64 {
	// Placeholder - in production, query Phase 7 cost tracer metrics
	// Sum of flk_cost_compute_usd + flk_cost_storage_usd + flk_cost_network_usd + flk_cost_anchoring_usd
	return 1250.75 // Example: $1,250.75
}

// getCloudCost computes cloud cost from billing records
func (br *BillingReconciler) getCloudCost(startTime, endTime time.Time) float64 {
	br.mu.RLock()
	defer br.mu.RUnlock()

	totalCost := 0.0
	for _, record := range br.cloudRecords {
		if record.RecordTimestamp.After(startTime) && record.RecordTimestamp.Before(endTime) {
			totalCost += record.Cost
		}
	}

	return totalCost
}

// computeBreakdown computes cost breakdown by service
func (br *BillingReconciler) computeBreakdown(startTime, endTime time.Time) map[string]*CostDelta {
	breakdown := make(map[string]*CostDelta)

	// Group cloud costs by service
	cloudByService := make(map[string]float64)
	br.mu.RLock()
	for _, record := range br.cloudRecords {
		if record.RecordTimestamp.After(startTime) && record.RecordTimestamp.Before(endTime) {
			cloudByService[record.ServiceName] += record.Cost
		}
	}
	br.mu.RUnlock()

	// Compare with internal estimates (placeholder)
	internalByService := map[string]float64{
		"Compute":   600.00,
		"Storage":   350.50,
		"Network":   200.25,
		"Anchoring": 100.00,
	}

	for service, cloudCost := range cloudByService {
		internalCost := internalByService[service]
		diff := cloudCost - internalCost
		diffPercent := 0.0
		if cloudCost > 0 {
			diffPercent = (diff / cloudCost) * 100
		}

		breakdown[service] = &CostDelta{
			ServiceName:       service,
			InternalCostUSD:   internalCost,
			CloudCostUSD:      cloudCost,
			DifferenceUSD:     diff,
			DifferencePercent: diffPercent,
		}
	}

	return breakdown
}

// generateRecommendations generates recommendations based on reconciliation
func (br *BillingReconciler) generateRecommendations(differencePercent float64, breakdown map[string]*CostDelta) []string {
	recommendations := []string{}

	if differencePercent > br.alertThreshold*100 {
		recommendations = append(recommendations, fmt.Sprintf("Critical drift detected: %.1f%% over target. Review cost attribution model.", differencePercent))
	}

	for service, delta := range breakdown {
		if delta.DifferencePercent > 10 {
			recommendations = append(recommendations, fmt.Sprintf("%s: %.1f%% drift. Consider adjusting cost model for this service.", service, delta.DifferencePercent))
		}
	}

	if len(recommendations) == 0 {
		recommendations = append(recommendations, "Cost reconciliation within target. No action required.")
	}

	return recommendations
}

// getProviderFromRecords gets provider from cloud records
func (br *BillingReconciler) getProviderFromRecords() string {
	br.mu.RLock()
	defer br.mu.RUnlock()

	if len(br.cloudRecords) > 0 {
		return br.cloudRecords[0].Provider
	}
	return "unknown"
}

// GetLatestReconciliation returns the most recent reconciliation report
func (br *BillingReconciler) GetLatestReconciliation() (*ReconciliationReport, error) {
	br.mu.RLock()
	defer br.mu.RUnlock()

	if len(br.reconciliations) == 0 {
		return nil, fmt.Errorf("no reconciliations available")
	}

	return br.reconciliations[len(br.reconciliations)-1], nil
}

// GetMetrics returns importer metrics
func (bi *BillingImporter) GetMetrics() ImporterMetrics {
	bi.metrics.mu.RLock()
	defer bi.metrics.mu.RUnlock()
	return *bi.metrics
}

// GetReconcilerMetrics returns reconciler metrics
func (br *BillingReconciler) GetReconcilerMetrics() ReconcilerMetrics {
	br.metrics.mu.RLock()
	defer br.metrics.mu.RUnlock()
	return *br.metrics
}

// AWSCURDataSource implements BillingDataSource for AWS Cost and Usage Reports
type AWSCURDataSource struct {
	curFilePath string // Path to CUR CSV export
}

// NewAWSCURDataSource creates a new AWS CUR data source
func NewAWSCURDataSource(curFilePath string) *AWSCURDataSource {
	return &AWSCURDataSource{
		curFilePath: curFilePath,
	}
}

// FetchBillingRecords fetches billing records from AWS CUR export
func (aws *AWSCURDataSource) FetchBillingRecords(ctx context.Context, startTime, endTime time.Time) ([]*CloudBillingRecord, error) {
	file, err := os.Open(aws.curFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open CUR file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records := []*CloudBillingRecord{}

	// Skip header
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read row: %w", err)
		}

		// Parse CUR row (simplified - actual CUR format is more complex)
		// Expected format: AccountID, ServiceName, ResourceID, UsageType, UsageAmount, Cost, Timestamp
		if len(row) < 7 {
			continue
		}

		timestamp, _ := time.Parse(time.RFC3339, row[6])
		if timestamp.Before(startTime) || timestamp.After(endTime) {
			continue
		}

		cost := 0.0
		fmt.Sscanf(row[5], "%f", &cost)

		usageAmount := 0.0
		fmt.Sscanf(row[4], "%f", &usageAmount)

		record := &CloudBillingRecord{
			Provider:        "aws",
			AccountID:       row[0],
			ServiceName:     row[1],
			ResourceID:      row[2],
			UsageType:       row[3],
			UsageAmount:     usageAmount,
			Cost:            cost,
			Currency:        "USD",
			BillingPeriod:   fmt.Sprintf("%d-%02d", timestamp.Year(), timestamp.Month()),
			RecordTimestamp: timestamp,
			Tags:            make(map[string]string),
		}

		records = append(records, record)
	}

	return records, nil
}

// GetProvider returns the provider name
func (aws *AWSCURDataSource) GetProvider() string {
	return "aws"
}

// GCPBigQueryDataSource implements BillingDataSource for GCP BigQuery exports
type GCPBigQueryDataSource struct {
	projectID string
	datasetID string
	tableID   string
}

// NewGCPBigQueryDataSource creates a new GCP BigQuery data source
func NewGCPBigQueryDataSource(projectID, datasetID, tableID string) *GCPBigQueryDataSource {
	return &GCPBigQueryDataSource{
		projectID: projectID,
		datasetID: datasetID,
		tableID:   tableID,
	}
}

// FetchBillingRecords fetches billing records from GCP BigQuery export
func (gcp *GCPBigQueryDataSource) FetchBillingRecords(ctx context.Context, startTime, endTime time.Time) ([]*CloudBillingRecord, error) {
	// Placeholder - in production, use BigQuery client
	// Query: SELECT * FROM `project.dataset.table` WHERE usage_start_time BETWEEN @start AND @end

	fmt.Printf("Fetching GCP billing records from BigQuery: project=%s, dataset=%s, table=%s\n",
		gcp.projectID, gcp.datasetID, gcp.tableID)

	// Mock records for demonstration
	records := []*CloudBillingRecord{
		{
			Provider:        "gcp",
			AccountID:       gcp.projectID,
			ServiceName:     "Compute Engine",
			ResourceID:      "instance-123",
			UsageType:       "Compute",
			UsageAmount:     100.0,
			Cost:            50.00,
			Currency:        "USD",
			BillingPeriod:   fmt.Sprintf("%d-%02d", startTime.Year(), startTime.Month()),
			RecordTimestamp: startTime,
			Tags:            map[string]string{"env": "production"},
		},
	}

	return records, nil
}

// GetProvider returns the provider name
func (gcp *GCPBigQueryDataSource) GetProvider() string {
	return "gcp"
}

// ExportReconciliationReport exports reconciliation report to JSON
func (br *BillingReconciler) ExportReconciliationReport(report *ReconciliationReport, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create report file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(report)
}

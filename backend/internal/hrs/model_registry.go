package hrs

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// ModelRegistry manages versioned HRS models with immutability (Phase 8 WP1)
type ModelRegistry struct {
	mu                 sync.RWMutex
	registryDir        string
	models             map[string]*RegisteredModel // version → model
	activeModel        string                      // Currently active model version
	previousActiveModel string                      // Previously active model version (for auto-revert)
	abRouter           *ABRouter
	metrics            *RegistryMetrics
}

// RegisteredModel represents a model in the registry
type RegisteredModel struct {
	Version       string
	Model         RiskModel
	ModelCard     *ModelCard
	RegisteredAt  time.Time
	DatasetHash   string
	BinaryHash    string // SHA-256 of model binary
	BinaryPath    string
	Status        string // "registered", "active", "shadow", "deprecated"
}

// ModelCard provides model documentation and metadata
type ModelCard struct {
	Version         string
	ModelType       string
	TrainedAt       time.Time
	DatasetInfo     DatasetInfo
	Metrics         ModelMetrics
	TrainingConfig  TrainingConfig
	Hyperparameters map[string]interface{}
	Limitations     []string
	IntendedUse     string
	EthicalConsiderations string
}

// DatasetInfo describes training dataset
type DatasetInfo struct {
	NumSamples    int
	NumFeatures   int
	PositiveRatio float64
	DatasetHash   string
	StartDate     time.Time
	EndDate       time.Time
	LabelSources  map[string]int // "human_review" → count, "202_escalation" → count
}

// ABRouter manages A/B testing between models
type ABRouter struct {
	mu            sync.RWMutex
	experiments   map[string]*ABExperiment // experiment_id → experiment
	tenantRouting map[string]string        // tenant_id → model_version (sticky routing)
}

// ABExperiment defines an A/B test between models
type ABExperiment struct {
	ID              string
	Name            string
	ControlVersion  string
	TreatmentVersion string
	TrafficSplit    float64 // % to treatment (0.0-1.0)
	StartedAt       time.Time
	EndsAt          time.Time
	Status          string // "pending", "running", "completed", "stopped"
	Metrics         *ABMetrics
}

// ABMetrics tracks A/B experiment results
type ABMetrics struct {
	mu                    sync.RWMutex
	ControlRequests       int64
	TreatmentRequests     int64
	ControlLatencyP95     float64
	TreatmentLatencyP95   float64
	ControlAUC            float64
	TreatmentAUC          float64
	StatisticalSignificance bool
	WinningVariant        string // "control", "treatment", "inconclusive"
}

// RegistryMetrics tracks registry operations
type RegistryMetrics struct {
	mu                sync.RWMutex
	TotalModels       int64
	ActiveModels      int64
	ShadowModels      int64
	DeprecatedModels  int64
	ABExperiments     int64
	LastRegistration  time.Time
}

// NewModelRegistry creates a new model registry
func NewModelRegistry(registryDir string) *ModelRegistry {
	os.MkdirAll(registryDir, 0755)

	return &ModelRegistry{
		registryDir: registryDir,
		models:      make(map[string]*RegisteredModel),
		abRouter:    NewABRouter(),
		metrics:     &RegistryMetrics{},
	}
}

// NewABRouter creates a new A/B router
func NewABRouter() *ABRouter {
	return &ABRouter{
		experiments:   make(map[string]*ABExperiment),
		tenantRouting: make(map[string]string),
	}
}

// RegisterModel registers a trained model in the registry
func (mr *ModelRegistry) RegisterModel(trainedModel *TrainedModel) (*RegisteredModel, error) {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	// Save model binary
	binaryPath, binaryHash, err := mr.saveModelBinary(trainedModel)
	if err != nil {
		return nil, fmt.Errorf("failed to save model binary: %w", err)
	}

	// Create model card
	modelCard := &ModelCard{
		Version:    trainedModel.Version,
		ModelType:  trainedModel.Config.ModelType,
		TrainedAt:  trainedModel.TrainedAt,
		Metrics:    trainedModel.Metrics,
		TrainingConfig: trainedModel.Config,
		Hyperparameters: map[string]interface{}{
			"learning_rate": trainedModel.Config.LearningRate,
			"epochs":        trainedModel.Config.Epochs,
			"batch_size":    trainedModel.Config.BatchSize,
		},
		Limitations: []string{
			"Trained on historical data; may not generalize to novel attack patterns",
			"Requires periodic retraining to maintain accuracy",
			fmt.Sprintf("AUC: %.3f (validation set)", trainedModel.Metrics.AUC),
		},
		IntendedUse: "Real-time hallucination risk prediction for PCS verification (≤10ms p95 latency budget)",
		EthicalConsiderations: "Model trained on PI-safe features only; no raw user content or PII used",
	}

	// Save model card
	if err := mr.saveModelCard(trainedModel.Version, modelCard); err != nil {
		return nil, fmt.Errorf("failed to save model card: %w", err)
	}

	registered := &RegisteredModel{
		Version:      trainedModel.Version,
		Model:        trainedModel.Model,
		ModelCard:    modelCard,
		RegisteredAt: time.Now(),
		DatasetHash:  trainedModel.DatasetHash,
		BinaryHash:   binaryHash,
		BinaryPath:   binaryPath,
		Status:       "registered",
	}

	mr.models[trainedModel.Version] = registered

	// Update metrics
	mr.metrics.mu.Lock()
	mr.metrics.TotalModels++
	mr.metrics.LastRegistration = time.Now()
	mr.metrics.mu.Unlock()

	fmt.Printf("Registered model: version=%s, binary_hash=%s\n", trainedModel.Version, binaryHash[:8])

	return registered, nil
}

// saveModelBinary saves model binary to disk and returns path + hash
func (mr *ModelRegistry) saveModelBinary(trainedModel *TrainedModel) (string, string, error) {
	binaryDir := filepath.Join(mr.registryDir, "binaries")
	os.MkdirAll(binaryDir, 0755)

	// Serialize model (placeholder - in production, use proper serialization)
	modelData, err := json.Marshal(trainedModel)
	if err != nil {
		return "", "", fmt.Errorf("failed to serialize model: %w", err)
	}

	// Compute hash
	hash := sha256.Sum256(modelData)
	binaryHash := hex.EncodeToString(hash[:])

	// Save binary
	binaryPath := filepath.Join(binaryDir, fmt.Sprintf("%s-%s.bin", trainedModel.Version, binaryHash[:8]))
	if err := os.WriteFile(binaryPath, modelData, 0444); err != nil { // Read-only
		return "", "", fmt.Errorf("failed to write binary: %w", err)
	}

	return binaryPath, binaryHash, nil
}

// saveModelCard saves model card to disk
func (mr *ModelRegistry) saveModelCard(version string, card *ModelCard) error {
	cardDir := filepath.Join(mr.registryDir, "cards")
	os.MkdirAll(cardDir, 0755)

	cardPath := filepath.Join(cardDir, fmt.Sprintf("%s-card.json", version))
	file, err := os.Create(cardPath)
	if err != nil {
		return fmt.Errorf("failed to create card file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(card)
}

// ActivateModel promotes a model to active status
func (mr *ModelRegistry) ActivateModel(version string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, ok := mr.models[version]
	if !ok {
		return fmt.Errorf("model not found: %s", version)
	}

	// Deactivate current active model and track it as previous
	if mr.activeModel != "" {
		if activeModel, ok := mr.models[mr.activeModel]; ok {
			activeModel.Status = "shadow"
		}
		mr.previousActiveModel = mr.activeModel
	}

	// Activate new model
	model.Status = "active"
	mr.activeModel = version

	// Update metrics
	mr.metrics.mu.Lock()
	mr.metrics.ActiveModels = 1
	mr.metrics.mu.Unlock()

	fmt.Printf("Activated model: version=%s (previous=%s)\n", version, mr.previousActiveModel)

	return nil
}

// GetActiveModel returns the currently active model
func (mr *ModelRegistry) GetActiveModel() (*RegisteredModel, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	if mr.activeModel == "" {
		return nil, fmt.Errorf("no active model")
	}

	model, ok := mr.models[mr.activeModel]
	if !ok {
		return nil, fmt.Errorf("active model not found: %s", mr.activeModel)
	}

	return model, nil
}

// GetPreviousActiveModel returns the previously active model (for auto-revert)
func (mr *ModelRegistry) GetPreviousActiveModel() *RegisteredModel {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	if mr.previousActiveModel == "" {
		return nil
	}

	model, ok := mr.models[mr.previousActiveModel]
	if !ok {
		return nil
	}

	return model
}

// PromoteModel promotes a model to active status (alias for ActivateModel for fairness audit)
func (mr *ModelRegistry) PromoteModel(version string) error {
	return mr.ActivateModel(version)
}

// GetModel retrieves a model by version
func (mr *ModelRegistry) GetModel(version string) (*RegisteredModel, error) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	model, ok := mr.models[version]
	if !ok {
		return nil, fmt.Errorf("model not found: %s", version)
	}

	return model, nil
}

// ListModels returns all registered models sorted by registration time
func (mr *ModelRegistry) ListModels() []*RegisteredModel {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	models := make([]*RegisteredModel, 0, len(mr.models))
	for _, model := range mr.models {
		models = append(models, model)
	}

	// Sort by registration time (newest first)
	sort.Slice(models, func(i, j int) bool {
		return models[i].RegisteredAt.After(models[j].RegisteredAt)
	})

	return models
}

// StartABExperiment starts an A/B test between two models
func (mr *ModelRegistry) StartABExperiment(name, controlVersion, treatmentVersion string, trafficSplit float64, duration time.Duration) (*ABExperiment, error) {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	// Validate models exist
	if _, ok := mr.models[controlVersion]; !ok {
		return nil, fmt.Errorf("control model not found: %s", controlVersion)
	}
	if _, ok := mr.models[treatmentVersion]; !ok {
		return nil, fmt.Errorf("treatment model not found: %s", treatmentVersion)
	}

	experimentID := fmt.Sprintf("ab-%s-%d", name, time.Now().Unix())
	experiment := &ABExperiment{
		ID:               experimentID,
		Name:             name,
		ControlVersion:   controlVersion,
		TreatmentVersion: treatmentVersion,
		TrafficSplit:     trafficSplit,
		StartedAt:        time.Now(),
		EndsAt:           time.Now().Add(duration),
		Status:           "running",
		Metrics:          &ABMetrics{},
	}

	mr.abRouter.experiments[experimentID] = experiment

	// Update metrics
	mr.metrics.mu.Lock()
	mr.metrics.ABExperiments++
	mr.metrics.mu.Unlock()

	fmt.Printf("Started A/B experiment: %s (control=%s, treatment=%s, split=%.0f%%)\n",
		experimentID, controlVersion, treatmentVersion, trafficSplit*100)

	return experiment, nil
}

// RouteRequest routes a request to a model based on A/B experiment
func (mr *ModelRegistry) RouteRequest(tenantID string, requestHash string) (string, error) {
	mr.abRouter.mu.RLock()
	defer mr.abRouter.mu.RUnlock()

	// Check for sticky tenant routing
	if version, ok := mr.abRouter.tenantRouting[tenantID]; ok {
		return version, nil
	}

	// Find active A/B experiment
	for _, exp := range mr.abRouter.experiments {
		if exp.Status == "running" && time.Now().Before(exp.EndsAt) {
			// Consistent hash routing based on request hash
			hashVal := hashString(requestHash)
			if float64(hashVal%100)/100.0 < exp.TrafficSplit {
				return exp.TreatmentVersion, nil
			}
			return exp.ControlVersion, nil
		}
	}

	// No active experiment, use active model
	mr.mu.RLock()
	activeVersion := mr.activeModel
	mr.mu.RUnlock()

	return activeVersion, nil
}

// hashString computes a consistent hash for routing
func hashString(s string) uint64 {
	hash := sha256.Sum256([]byte(s))
	var result uint64
	for i := 0; i < 8; i++ {
		result = (result << 8) | uint64(hash[i])
	}
	return result
}

// StopABExperiment stops an A/B experiment
func (mr *ModelRegistry) StopABExperiment(experimentID string) error {
	mr.abRouter.mu.Lock()
	defer mr.abRouter.mu.Unlock()

	exp, ok := mr.abRouter.experiments[experimentID]
	if !ok {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}

	exp.Status = "stopped"
	fmt.Printf("Stopped A/B experiment: %s\n", experimentID)

	return nil
}

// GetABMetrics returns A/B experiment metrics
func (mr *ModelRegistry) GetABMetrics(experimentID string) (*ABMetrics, error) {
	mr.abRouter.mu.RLock()
	defer mr.abRouter.mu.RUnlock()

	exp, ok := mr.abRouter.experiments[experimentID]
	if !ok {
		return nil, fmt.Errorf("experiment not found: %s", experimentID)
	}

	return exp.Metrics, nil
}

// DeprecateModel marks a model as deprecated
func (mr *ModelRegistry) DeprecateModel(version string) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()

	model, ok := mr.models[version]
	if !ok {
		return fmt.Errorf("model not found: %s", version)
	}

	if model.Status == "active" {
		return fmt.Errorf("cannot deprecate active model: %s", version)
	}

	model.Status = "deprecated"

	// Update metrics
	mr.metrics.mu.Lock()
	mr.metrics.DeprecatedModels++
	mr.metrics.mu.Unlock()

	fmt.Printf("Deprecated model: version=%s\n", version)

	return nil
}

// VerifyModelIntegrity verifies model binary integrity using hash
func (mr *ModelRegistry) VerifyModelIntegrity(version string) (bool, error) {
	mr.mu.RLock()
	model, ok := mr.models[version]
	mr.mu.RUnlock()

	if !ok {
		return false, fmt.Errorf("model not found: %s", version)
	}

	// Read binary
	data, err := os.ReadFile(model.BinaryPath)
	if err != nil {
		return false, fmt.Errorf("failed to read binary: %w", err)
	}

	// Compute hash
	hash := sha256.Sum256(data)
	computedHash := hex.EncodeToString(hash[:])

	// Compare
	if computedHash != model.BinaryHash {
		return false, fmt.Errorf("hash mismatch: expected %s, got %s", model.BinaryHash, computedHash)
	}

	return true, nil
}

// GetMetrics returns registry metrics
func (mr *ModelRegistry) GetMetrics() RegistryMetrics {
	mr.metrics.mu.RLock()
	defer mr.metrics.mu.RUnlock()
	return *mr.metrics
}

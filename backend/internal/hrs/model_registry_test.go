package hrs

import (
	"testing"
	"time"
)

// TestGetPreviousActiveModel tests the GetPreviousActiveModel method
func TestGetPreviousActiveModel(t *testing.T) {
	// Create temp registry dir
	tmpDir := t.TempDir()
	registry := NewModelRegistry(tmpDir)

	// Initially no previous model
	prevModel := registry.GetPreviousActiveModel()
	if prevModel != nil {
		t.Errorf("Expected nil for no previous model, got %v", prevModel)
	}

	// Register and activate first model
	model1 := createTestModel(t, registry, "v1.0.0")
	err := registry.ActivateModel(model1.Version)
	if err != nil {
		t.Fatalf("Failed to activate model1: %v", err)
	}

	// Still no previous model after first activation
	prevModel = registry.GetPreviousActiveModel()
	if prevModel != nil {
		t.Errorf("Expected nil after first activation, got %v", prevModel)
	}

	// Register and activate second model
	model2 := createTestModel(t, registry, "v2.0.0")
	err = registry.ActivateModel(model2.Version)
	if err != nil {
		t.Fatalf("Failed to activate model2: %v", err)
	}

	// Now previous model should be model1
	prevModel = registry.GetPreviousActiveModel()
	if prevModel == nil {
		t.Fatal("Expected previous model, got nil")
	}
	if prevModel.Version != "v1.0.0" {
		t.Errorf("Expected previous model v1.0.0, got %s", prevModel.Version)
	}
	if prevModel.Status != "shadow" {
		t.Errorf("Expected previous model status 'shadow', got %s", prevModel.Status)
	}
}

// TestPromoteModel tests the PromoteModel method
func TestPromoteModel(t *testing.T) {
	// Create temp registry dir
	tmpDir := t.TempDir()
	registry := NewModelRegistry(tmpDir)

	// Register two models
	model1 := createTestModel(t, registry, "v1.0.0")
	model2 := createTestModel(t, registry, "v2.0.0")

	// Activate model1
	err := registry.ActivateModel(model1.Version)
	if err != nil {
		t.Fatalf("Failed to activate model1: %v", err)
	}

	// Activate model2 (model1 becomes previous)
	err = registry.ActivateModel(model2.Version)
	if err != nil {
		t.Fatalf("Failed to activate model2: %v", err)
	}

	// Verify model2 is active
	activeModel, err := registry.GetActiveModel()
	if err != nil {
		t.Fatalf("Failed to get active model: %v", err)
	}
	if activeModel.Version != "v2.0.0" {
		t.Errorf("Expected active model v2.0.0, got %s", activeModel.Version)
	}

	// Use PromoteModel to revert to model1
	err = registry.PromoteModel("v1.0.0")
	if err != nil {
		t.Fatalf("Failed to promote model1: %v", err)
	}

	// Verify model1 is now active
	activeModel, err = registry.GetActiveModel()
	if err != nil {
		t.Fatalf("Failed to get active model after promotion: %v", err)
	}
	if activeModel.Version != "v1.0.0" {
		t.Errorf("Expected active model v1.0.0 after promotion, got %s", activeModel.Version)
	}
	if activeModel.Status != "active" {
		t.Errorf("Expected promoted model status 'active', got %s", activeModel.Status)
	}

	// Verify model2 is now previous
	prevModel := registry.GetPreviousActiveModel()
	if prevModel == nil {
		t.Fatal("Expected previous model after promotion, got nil")
	}
	if prevModel.Version != "v2.0.0" {
		t.Errorf("Expected previous model v2.0.0, got %s", prevModel.Version)
	}
}

// TestPromoteModelNonExistent tests promoting a non-existent model
func TestPromoteModelNonExistent(t *testing.T) {
	tmpDir := t.TempDir()
	registry := NewModelRegistry(tmpDir)

	// Try to promote non-existent model
	err := registry.PromoteModel("v99.99.99")
	if err == nil {
		t.Error("Expected error when promoting non-existent model, got nil")
	}
}

// Helper: createTestModel creates a minimal test model
func createTestModel(t *testing.T, registry *ModelRegistry, version string) *RegisteredModel {
	t.Helper()

	// Create a mock trained model
	trainedModel := &TrainedModel{
		Version:     version,
		Model:       &MockRiskModel{version: version},
		TrainedAt:   time.Now(),
		DatasetHash: "test-hash-" + version,
		Metrics: ModelMetrics{
			AUC:       0.85,
			Precision: 0.80,
			Recall:    0.75,
			F1:        0.77,
		},
		Config: TrainingConfig{
			ModelType:    "logistic_regression",
			LearningRate: 0.001,
			Epochs:       100,
			BatchSize:    32,
		},
	}

	registered, err := registry.RegisterModel(trainedModel)
	if err != nil {
		t.Fatalf("Failed to register model %s: %v", version, err)
	}

	return registered
}

// MockRiskModel is a simple mock for testing
type MockRiskModel struct {
	version string
}

func (m *MockRiskModel) Predict(features *PCSFeatures) (float64, error) {
	return 0.5, nil
}

func (m *MockRiskModel) PredictWithUncertainty(features *PCSFeatures) (float64, float64, error) {
	return 0.5, 0.1, nil
}

func (m *MockRiskModel) GetVersion() string {
	return m.version
}

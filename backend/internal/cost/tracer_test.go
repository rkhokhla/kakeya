package cost

import (
	"testing"
)

// TestTracerAlias verifies the Tracer type alias compiles correctly (Phase 9 WP4 compatibility fix)
// This ensures backward compatibility with Phase 9 code that references Tracer instead of CostTracer
func TestTracerAlias(t *testing.T) {
	// The mere fact that this test compiles proves the type alias works
	// Phase 9 code uses Tracer in function signatures and it must compile
	var _ Tracer // This line proves Tracer type exists

	// Verify NewCostTracer works
	tracer := NewCostTracer()
	if tracer == nil {
		t.Fatal("NewCostTracer should not return nil")
	}
}

// TestDefaultCostModel tests the default cost model parameters
func TestDefaultCostModel(t *testing.T) {
	model := DefaultCostModel()

	// Verify compute cost
	if model.ComputeUSD != 0.0001 {
		t.Errorf("Expected compute cost $0.0001, got $%.4f", model.ComputeUSD)
	}

	// Verify storage costs
	if model.StorageHotUSD != 0.023 {
		t.Errorf("Expected hot storage $0.023/GB/month, got $%.3f", model.StorageHotUSD)
	}
	if model.StorageWarmUSD != 0.010 {
		t.Errorf("Expected warm storage $0.010/GB/month, got $%.3f", model.StorageWarmUSD)
	}
	if model.StorageColdUSD != 0.004 {
		t.Errorf("Expected cold storage $0.004/GB/month, got $%.3f", model.StorageColdUSD)
	}

	// Verify network cost
	if model.NetworkUSD != 0.09 {
		t.Errorf("Expected network cost $0.09/GB, got $%.2f", model.NetworkUSD)
	}

	// Verify anchoring costs
	if model.AnchoringEthereumUSD != 2.00 {
		t.Errorf("Expected Ethereum anchoring $2.00, got $%.2f", model.AnchoringEthereumUSD)
	}
	if model.AnchoringPolygonUSD != 0.01 {
		t.Errorf("Expected Polygon anchoring $0.01, got $%.2f", model.AnchoringPolygonUSD)
	}
	if model.AnchoringOpenTimestampUSD != 0.00 {
		t.Errorf("Expected OpenTimestamp anchoring $0.00, got $%.2f", model.AnchoringOpenTimestampUSD)
	}
}

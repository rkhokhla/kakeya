package otel

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel/attribute"
)

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig("test-service")

	if config.ServiceName != "test-service" {
		t.Errorf("Expected service name 'test-service', got '%s'", config.ServiceName)
	}

	if config.ServiceVersion == "" {
		t.Error("Service version should not be empty")
	}

	if config.CollectorEndpoint == "" {
		t.Error("Collector endpoint should not be empty")
	}

	if config.SamplingRate < 0.0 || config.SamplingRate > 1.0 {
		t.Errorf("Sampling rate out of bounds: %.2f", config.SamplingRate)
	}
}

func TestPCSAttributes(t *testing.T) {
	attrs := PCSAttributes(
		"pcs-123",
		"merkle-abc",
		"shard-1",
		"sticky",
		42,
		1.5,
		0.7,
		0.85,
	)

	if len(attrs) != 8 {
		t.Errorf("Expected 8 attributes, got %d", len(attrs))
	}

	// Verify key attribute exists
	found := false
	for _, attr := range attrs {
		if attr.Key == AttrPCSID && attr.Value.AsString() == "pcs-123" {
			found = true
			break
		}
	}
	if !found {
		t.Error("PCSID attribute not found")
	}
}

func TestTenantAttributes(t *testing.T) {
	// With userID
	attrs := TenantAttributes("tenant-123", "user-456")
	if len(attrs) != 2 {
		t.Errorf("Expected 2 attributes with userID, got %d", len(attrs))
	}

	// Without userID
	attrs = TenantAttributes("tenant-123", "")
	if len(attrs) != 1 {
		t.Errorf("Expected 1 attribute without userID, got %d", len(attrs))
	}
}

func TestRoutingAttributes(t *testing.T) {
	attrs := RoutingAttributes("fast_path", 0.05, 0.95)

	if len(attrs) != 3 {
		t.Errorf("Expected 3 attributes, got %d", len(attrs))
	}
}

func TestVerificationAttributes(t *testing.T) {
	attrs := VerificationAttributes("accepted", true, true)

	if len(attrs) != 3 {
		t.Errorf("Expected 3 attributes, got %d", len(attrs))
	}
}

func TestPerformanceAttributes(t *testing.T) {
	attrs := PerformanceAttributes(true, 25.5)

	if len(attrs) != 2 {
		t.Errorf("Expected 2 attributes, got %d", len(attrs))
	}
}

func TestStartSpan(t *testing.T) {
	ctx := context.Background()

	// This will use the global no-op tracer since we haven't initialized OTel
	ctx, span := StartSpan(ctx, "test-tracer", "test-span",
		attribute.String("test.key", "test.value"),
	)

	if ctx == nil {
		t.Error("Context should not be nil")
	}

	if span == nil {
		t.Error("Span should not be nil")
	}

	span.End()
}

func TestRecordError(t *testing.T) {
	ctx := context.Background()
	_, span := StartSpan(ctx, "test-tracer", "test-span")

	// Should not panic
	RecordError(span, nil, "")
	RecordError(span, nil, "test message")

	span.End()
}

func TestAddEvent(t *testing.T) {
	ctx := context.Background()
	_, span := StartSpan(ctx, "test-tracer", "test-span")

	// Should not panic
	AddEvent(span, "test-event")
	AddEvent(span, "test-event-with-attrs",
		attribute.String("key", "value"),
	)

	span.End()
}

package otel

import (
	"context"
	"fmt"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"go.opentelemetry.io/otel/trace"
)

// Phase 11 WP7: OpenTelemetry Observability
// Correlate traces, metrics, and logs across controls (VRF/JWT/RAG/routing)

// Config holds OpenTelemetry configuration
type Config struct {
	ServiceName        string
	ServiceVersion     string
	Environment        string
	CollectorEndpoint  string
	CollectorInsecure  bool
	SamplingRate       float64 // 0.0 to 1.0 (1.0 = always sample)
	MaxEventsPerSpan   int
	MaxAttributesPerSpan int
}

// DefaultConfig returns production defaults
func DefaultConfig(serviceName string) *Config {
	return &Config{
		ServiceName:        serviceName,
		ServiceVersion:     "0.11.0",
		Environment:        "production",
		CollectorEndpoint:  "localhost:4317",
		CollectorInsecure:  true, // Use TLS in production
		SamplingRate:       1.0,  // Sample all traces in dev
		MaxEventsPerSpan:   128,
		MaxAttributesPerSpan: 128,
	}
}

// InitTracer initializes OpenTelemetry tracing
func InitTracer(ctx context.Context, config *Config) (*sdktrace.TracerProvider, error) {
	if config == nil {
		config = DefaultConfig("fractal-lba")
	}

	// Create OTLP exporter
	exporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(config.CollectorEndpoint),
		otlptracegrpc.WithInsecure(), // Use WithTLSCredentials in production
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create OTLP exporter: %w", err)
	}

	// Create resource with service information
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(config.ServiceName),
			semconv.ServiceVersion(config.ServiceVersion),
			semconv.DeploymentEnvironment(config.Environment),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Create tracer provider with sampling
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter,
			sdktrace.WithBatchTimeout(5*time.Second),
			sdktrace.WithMaxQueueSize(2048),
			sdktrace.WithMaxExportBatchSize(512),
		),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.TraceIDRatioBased(config.SamplingRate)),
		sdktrace.WithSpanLimits(sdktrace.SpanLimits{
			EventCountLimit:      config.MaxEventsPerSpan,
			AttributeCountLimit:  config.MaxAttributesPerSpan,
		}),
	)

	// Set global tracer provider
	otel.SetTracerProvider(tp)

	// Set global propagator for context propagation
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return tp, nil
}

// Shutdown gracefully shuts down the tracer provider
func Shutdown(ctx context.Context, tp *sdktrace.TracerProvider) error {
	if tp == nil {
		return nil
	}

	// Use context with timeout for shutdown
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	return tp.Shutdown(ctx)
}

// StartSpan is a convenience wrapper for starting a span with common attributes
func StartSpan(ctx context.Context, tracerName, spanName string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	tracer := otel.Tracer(tracerName)
	ctx, span := tracer.Start(ctx, spanName)

	// Add attributes if provided
	if len(attrs) > 0 {
		span.SetAttributes(attrs...)
	}

	return ctx, span
}

// RecordError records an error on a span with optional message
func RecordError(span trace.Span, err error, message string) {
	if span == nil || err == nil {
		return
	}

	if message != "" {
		span.RecordError(err, trace.WithAttributes(
			attribute.String("error.message", message),
		))
	} else {
		span.RecordError(err)
	}

	span.SetStatus(codes.Error, err.Error())
}

// AddEvent adds an event to a span with optional attributes
func AddEvent(span trace.Span, name string, attrs ...attribute.KeyValue) {
	if span == nil {
		return
	}

	span.AddEvent(name, trace.WithAttributes(attrs...))
}

// Common attribute keys for Fractal LBA
const (
	// PCS attributes
	AttrPCSID         = attribute.Key("pcs.id")
	AttrMerkleRoot    = attribute.Key("pcs.merkle_root")
	AttrEpoch         = attribute.Key("pcs.epoch")
	AttrShardID       = attribute.Key("pcs.shard_id")
	AttrDHat          = attribute.Key("pcs.d_hat")
	AttrCohStar       = attribute.Key("pcs.coh_star")
	AttrCompressibility = attribute.Key("pcs.r")
	AttrRegime        = attribute.Key("pcs.regime")

	// Tenant attributes
	AttrTenantID      = attribute.Key("tenant.id")
	AttrUserID        = attribute.Key("user.id")

	// Routing attributes
	AttrRoute         = attribute.Key("routing.route")
	AttrHRSRisk       = attribute.Key("hrs.risk")
	AttrHRSConfidence = attribute.Key("hrs.confidence")

	// Verification attributes
	AttrVerifyResult  = attribute.Key("verify.result")
	AttrSignatureValid = attribute.Key("signature.valid")
	AttrVRFValid      = attribute.Key("vrf.valid")

	// Performance attributes
	AttrDedupHit      = attribute.Key("dedup.hit")
	AttrLatencyMs     = attribute.Key("latency.ms")
)

// Helper functions to create common attributes

func PCSAttributes(pcsID, merkleRoot, shardID, regime string, epoch int, dHat, cohStar, r float64) []attribute.KeyValue {
	return []attribute.KeyValue{
		AttrPCSID.String(pcsID),
		AttrMerkleRoot.String(merkleRoot),
		AttrEpoch.Int(epoch),
		AttrShardID.String(shardID),
		AttrDHat.Float64(dHat),
		AttrCohStar.Float64(cohStar),
		AttrCompressibility.Float64(r),
		AttrRegime.String(regime),
	}
}

func TenantAttributes(tenantID, userID string) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		AttrTenantID.String(tenantID),
	}
	if userID != "" {
		attrs = append(attrs, AttrUserID.String(userID))
	}
	return attrs
}

func RoutingAttributes(route string, hrsRisk, hrsConfidence float64) []attribute.KeyValue {
	return []attribute.KeyValue{
		AttrRoute.String(route),
		AttrHRSRisk.Float64(hrsRisk),
		AttrHRSConfidence.Float64(hrsConfidence),
	}
}

func VerificationAttributes(result string, signatureValid, vrfValid bool) []attribute.KeyValue {
	return []attribute.KeyValue{
		AttrVerifyResult.String(result),
		AttrSignatureValid.Bool(signatureValid),
		AttrVRFValid.Bool(vrfValid),
	}
}

func PerformanceAttributes(dedupHit bool, latencyMs float64) []attribute.KeyValue {
	return []attribute.KeyValue{
		AttrDedupHit.Bool(dedupHit),
		AttrLatencyMs.Float64(latencyMs),
	}
}

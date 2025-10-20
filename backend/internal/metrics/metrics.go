package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics holds all Prometheus counters for the system
type Metrics struct {
	// Global counters (Phase 1/2 compatibility)
	IngestTotal  prometheus.Counter
	DedupHits    prometheus.Counter
	Accepted     prometheus.Counter
	Escalated    prometheus.Counter
	SignatureErr prometheus.Counter
	WALErrors    prometheus.Counter

	// Multi-tenant labeled metrics (Phase 3)
	IngestTotalByTenant  *prometheus.CounterVec
	DedupHitsByTenant    *prometheus.CounterVec
	AcceptedByTenant     *prometheus.CounterVec
	EscalatedByTenant    *prometheus.CounterVec
	SignatureErrByTenant *prometheus.CounterVec
	QuotaExceededByTenant *prometheus.CounterVec
}

// New creates and registers all metrics
func New() *Metrics {
	return &Metrics{
		// Global counters (backward compatible)
		IngestTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "flk_ingest_total",
			Help: "Total number of PCS submissions received",
		}),
		DedupHits: promauto.NewCounter(prometheus.CounterOpts{
			Name: "flk_dedup_hits",
			Help: "Number of duplicate PCS submissions served from cache",
		}),
		Accepted: promauto.NewCounter(prometheus.CounterOpts{
			Name: "flk_accepted",
			Help: "Number of PCS submissions accepted (200)",
		}),
		Escalated: promauto.NewCounter(prometheus.CounterOpts{
			Name: "flk_escalated",
			Help: "Number of PCS submissions escalated (202)",
		}),
		SignatureErr: promauto.NewCounter(prometheus.CounterOpts{
			Name: "flk_signature_errors",
			Help: "Number of signature verification failures",
		}),
		WALErrors: promauto.NewCounter(prometheus.CounterOpts{
			Name: "flk_wal_errors",
			Help: "Number of WAL write errors",
		}),

		// Multi-tenant labeled metrics
		IngestTotalByTenant: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_ingest_total_by_tenant",
				Help: "Total number of PCS submissions received per tenant",
			},
			[]string{"tenant_id"},
		),
		DedupHitsByTenant: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_dedup_hits_by_tenant",
				Help: "Number of duplicate PCS submissions served from cache per tenant",
			},
			[]string{"tenant_id"},
		),
		AcceptedByTenant: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_accepted_by_tenant",
				Help: "Number of PCS submissions accepted (200) per tenant",
			},
			[]string{"tenant_id"},
		),
		EscalatedByTenant: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_escalated_by_tenant",
				Help: "Number of PCS submissions escalated (202) per tenant",
			},
			[]string{"tenant_id"},
		),
		SignatureErrByTenant: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_signature_errors_by_tenant",
				Help: "Number of signature verification failures per tenant",
			},
			[]string{"tenant_id"},
		),
		QuotaExceededByTenant: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_quota_exceeded_by_tenant",
				Help: "Number of requests rejected due to quota exceeded per tenant",
			},
			[]string{"tenant_id"},
		),
	}
}

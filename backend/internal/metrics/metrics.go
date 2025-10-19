package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics holds all Prometheus counters for the system
type Metrics struct {
	IngestTotal  prometheus.Counter
	DedupHits    prometheus.Counter
	Accepted     prometheus.Counter
	Escalated    prometheus.Counter
	SignatureErr prometheus.Counter
	WALErrors    prometheus.Counter
}

// New creates and registers all metrics
func New() *Metrics {
	return &Metrics{
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
	}
}

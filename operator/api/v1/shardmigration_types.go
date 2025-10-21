package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ShardMigrationSpec defines the desired state of ShardMigration
type ShardMigrationSpec struct {
	// SourceShards are the existing shards to migrate from
	SourceShards []string `json:"sourceShards"`

	// TargetShards are the new shard configuration
	TargetShards []string `json:"targetShards"`

	// DedupBackend is the dedup store type (redis, postgres)
	DedupBackend string `json:"dedupBackend"`

	// BatchSize for key copying (default: 1000)
	BatchSize int `json:"batchSize,omitempty"`

	// ThrottleQPS limits migration rate (default: 1000)
	ThrottleQPS int `json:"throttleQPS,omitempty"`

	// HealthGates defines SLO thresholds for safe migration
	HealthGates HealthGateConfig `json:"healthGates,omitempty"`

	// AutoRollback enables automatic rollback on SLO violations
	AutoRollback bool `json:"autoRollback,omitempty"`
}

// HealthGateConfig defines SLO thresholds for health checks
type HealthGateConfig struct {
	// MaxLatencyP95 in milliseconds (default: 200)
	MaxLatencyP95 float64 `json:"maxLatencyP95,omitempty"`

	// MaxErrorRate as percentage (default: 1.0)
	MaxErrorRate float64 `json:"maxErrorRate,omitempty"`

	// MinDedupHitRatio as percentage (default: 40.0)
	MinDedupHitRatio float64 `json:"minDedupHitRatio,omitempty"`
}

// ShardMigrationStatus defines the observed state of ShardMigration
type ShardMigrationStatus struct {
	// Phase is the current migration phase
	// +kubebuilder:validation:Enum=Pending;Planning;Copying;Verifying;DualWrite;Cutover;Cleanup;Completed;Failed;RollingBack
	Phase string `json:"phase"`

	// Progress tracks migration progress
	Progress MigrationProgress `json:"progress,omitempty"`

	// Conditions represent the latest available observations
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// LastTransitionTime is when the phase last changed
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty"`

	// Message provides human-readable status
	Message string `json:"message,omitempty"`
}

// MigrationProgress tracks detailed migration metrics
type MigrationProgress struct {
	// KeysCopied is the number of keys copied so far
	KeysCopied int64 `json:"keysCopied"`

	// TotalKeys is the estimated total keys to migrate
	TotalKeys int64 `json:"totalKeys"`

	// BytesCopied is the total bytes copied
	BytesCopied int64 `json:"bytesCopied"`

	// Percentage is the completion percentage (0-100)
	Percentage float64 `json:"percentage"`

	// EstimatedTimeRemaining in seconds
	EstimatedTimeRemaining int64 `json:"estimatedTimeRemaining,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Progress",type=string,JSONPath=`.status.progress.percentage`
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// ShardMigration is the Schema for the shardmigrations API
type ShardMigration struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ShardMigrationSpec   `json:"spec,omitempty"`
	Status ShardMigrationStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ShardMigrationList contains a list of ShardMigration
type ShardMigrationList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ShardMigration `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ShardMigration{}, &ShardMigrationList{})
}

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// CRRPolicySpec defines the desired state of CRRPolicy
type CRRPolicySpec struct {
	// SourceRegion is the region shipping WAL segments
	SourceRegion string `json:"sourceRegion"`

	// TargetRegions are the regions receiving WAL segments
	TargetRegions []string `json:"targetRegions"`

	// ReplicationMode defines the replication strategy
	// +kubebuilder:validation:Enum=full;selective;multi-way
	ReplicationMode string `json:"replicationMode,omitempty"`

	// TenantSelector selects which tenants to replicate (selective mode)
	TenantSelector *TenantSelector `json:"tenantSelector,omitempty"`

	// ShipInterval is how often to check for new segments (default: 30s)
	ShipInterval metav1.Duration `json:"shipInterval,omitempty"`

	// Priority for shipping (1-10, higher = more urgent)
	Priority int `json:"priority,omitempty"`

	// MaxRetries for failed uploads (default: 3)
	MaxRetries int `json:"maxRetries,omitempty"`

	// HealthGates defines SLO thresholds for CRR health
	HealthGates CRRHealthGateConfig `json:"healthGates,omitempty"`

	// AutoReconcile enables automatic divergence reconciliation
	AutoReconcile bool `json:"autoReconcile,omitempty"`
}

// TenantSelector defines criteria for selecting tenants
type TenantSelector struct {
	// Include is a list of tenant IDs to include
	Include []string `json:"include,omitempty"`

	// Exclude is a list of tenant IDs to exclude
	Exclude []string `json:"exclude,omitempty"`

	// MatchLabels selects tenants by labels
	MatchLabels map[string]string `json:"matchLabels,omitempty"`
}

// CRRHealthGateConfig defines CRR SLO thresholds
type CRRHealthGateConfig struct {
	// MaxLagSeconds is the maximum acceptable replication lag (default: 60)
	MaxLagSeconds float64 `json:"maxLagSeconds,omitempty"`

	// MaxShipErrors per hour (default: 10)
	MaxShipErrors int64 `json:"maxShipErrors,omitempty"`

	// MaxDivergencePercent triggers reconciliation (default: 5.0)
	MaxDivergencePercent float64 `json:"maxDivergencePercent,omitempty"`
}

// CRRPolicyStatus defines the observed state of CRRPolicy
type CRRPolicyStatus struct {
	// Active indicates if the policy is currently active
	Active bool `json:"active"`

	// ReplicationStats tracks CRR metrics
	ReplicationStats ReplicationStats `json:"replicationStats,omitempty"`

	// Conditions represent the latest available observations
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// LastShipTime is when the last segment was shipped
	LastShipTime metav1.Time `json:"lastShipTime,omitempty"`

	// Message provides human-readable status
	Message string `json:"message,omitempty"`
}

// ReplicationStats tracks CRR performance metrics
type ReplicationStats struct {
	// SegmentsShipped in the last hour
	SegmentsShipped int64 `json:"segmentsShipped"`

	// BytesShipped in the last hour
	BytesShipped int64 `json:"bytesShipped"`

	// LagSeconds is the current replication lag (p95)
	LagSeconds float64 `json:"lagSeconds"`

	// ShipErrors in the last hour
	ShipErrors int64 `json:"shipErrors"`

	// DivergencePercent is the current divergence level
	DivergencePercent float64 `json:"divergencePercent,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced
// +kubebuilder:printcolumn:name="Mode",type=string,JSONPath=`.spec.replicationMode`
// +kubebuilder:printcolumn:name="Active",type=boolean,JSONPath=`.status.active`
// +kubebuilder:printcolumn:name="Lag",type=string,JSONPath=`.status.replicationStats.lagSeconds`
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// CRRPolicy is the Schema for the crrpolicies API
type CRRPolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   CRRPolicySpec   `json:"spec,omitempty"`
	Status CRRPolicyStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// CRRPolicyList contains a list of CRRPolicy
type CRRPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []CRRPolicy `json:"items"`
}

func init() {
	SchemeBuilder.Register(&CRRPolicy{}, &CRRPolicyList{})
}

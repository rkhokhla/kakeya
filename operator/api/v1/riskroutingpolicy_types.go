package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// RiskRoutingPolicy defines risk-aware routing policy (Phase 7 WP4)
// Binds HRS risk bands to verification actions and budget adjustments
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=rrp
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
type RiskRoutingPolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   RiskRoutingPolicySpec   `json:"spec,omitempty"`
	Status RiskRoutingPolicyStatus `json:"status,omitempty"`
}

// RiskRoutingPolicySpec defines the desired state of RiskRoutingPolicy
type RiskRoutingPolicySpec struct {
	// TenantSelector selects which tenants this policy applies to
	// +optional
	TenantSelector *TenantSelector `json:"tenantSelector,omitempty"`

	// RiskBands define risk thresholds and corresponding actions
	// +kubebuilder:validation:MinItems=1
	RiskBands []RiskBand `json:"riskBands"`

	// DefaultAction is the action when risk is unknown or below all thresholds
	// +kubebuilder:validation:Enum=accept;escalate;reject
	DefaultAction string `json:"defaultAction"`

	// HRSConfig configures the Hallucination Risk Scorer
	// +optional
	HRSConfig *HRSConfig `json:"hrsConfig,omitempty"`

	// CanaryRollout defines canary rollout configuration
	// +optional
	CanaryRollout *CanaryRollout `json:"canaryRollout,omitempty"`
}

// RiskBand defines a risk threshold and corresponding action
type RiskBand struct {
	// Name of the risk band (e.g., "low", "medium", "high")
	Name string `json:"name"`

	// MinRisk is the minimum risk score [0, 1] for this band
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinRisk float64 `json:"minRisk"`

	// MaxRisk is the maximum risk score [0, 1] for this band
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MaxRisk float64 `json:"maxRisk"`

	// Action defines what to do for requests in this risk band
	// +kubebuilder:validation:Enum=accept;rag_required;human_review;reduce_budget;reject;alternate_region
	Action string `json:"action"`

	// BudgetMultiplier multiplies the budget for requests in this band
	// +optional
	// +kubebuilder:default=1.0
	BudgetMultiplier float64 `json:"budgetMultiplier,omitempty"`

	// AlternateRegion specifies an alternate region for high-risk requests
	// +optional
	AlternateRegion string `json:"alternateRegion,omitempty"`

	// RequireEnsemble specifies if ensemble verification is required
	// +optional
	RequireEnsemble bool `json:"requireEnsemble,omitempty"`

	// Metadata for auditing and observability
	// +optional
	Metadata map[string]string `json:"metadata,omitempty"`
}

// HRSConfig configures the Hallucination Risk Scorer
type HRSConfig struct {
	// Enabled enables HRS for this policy
	Enabled bool `json:"enabled"`

	// ModelVersion specifies the HRS model version to use
	// +optional
	// +kubebuilder:default="lr-v1.0"
	ModelVersion string `json:"modelVersion,omitempty"`

	// MinConfidence is the minimum confidence interval threshold
	// +optional
	// +kubebuilder:default=0.7
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinConfidence float64 `json:"minConfidence,omitempty"`

	// LatencyBudgetMs is the maximum allowed latency for HRS (ms)
	// +optional
	// +kubebuilder:default=10
	LatencyBudgetMs int `json:"latencyBudgetMs,omitempty"`

	// FailOpen specifies if HRS should fail-open (accept on error)
	// +optional
	// +kubebuilder:default=true
	FailOpen bool `json:"failOpen,omitempty"`
}

// CanaryRollout defines canary rollout configuration
type CanaryRollout struct {
	// Enabled enables canary rollout
	Enabled bool `json:"enabled"`

	// InitialPercent is the initial traffic percentage for canary (0-100)
	// +kubebuilder:default=5
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	InitialPercent int `json:"initialPercent"`

	// IncrementPercent is the traffic increment per step (0-100)
	// +kubebuilder:default=10
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	IncrementPercent int `json:"incrementPercent"`

	// IncrementIntervalMinutes is the time between increments
	// +kubebuilder:default=15
	IncrementIntervalMinutes int `json:"incrementIntervalMinutes"`

	// HealthGates define SLO gates that must pass before increment
	// +optional
	HealthGates []HealthGate `json:"healthGates,omitempty"`
}

// HealthGate defines an SLO gate for canary rollout
type HealthGate struct {
	// Name of the health gate
	Name string `json:"name"`

	// MetricQuery is the Prometheus query for the metric
	MetricQuery string `json:"metricQuery"`

	// Threshold is the maximum allowed value
	Threshold float64 `json:"threshold"`

	// WindowMinutes is the evaluation window in minutes
	// +kubebuilder:default=5
	WindowMinutes int `json:"windowMinutes"`
}

// RiskRoutingPolicyStatus defines the observed state of RiskRoutingPolicy
type RiskRoutingPolicyStatus struct {
	// Phase is the current phase of the policy rollout
	// +kubebuilder:validation:Enum=Pending;Canary;Active;Failed;RolledBack
	Phase string `json:"phase,omitempty"`

	// CanaryPercent is the current canary traffic percentage (0-100)
	// +optional
	CanaryPercent int `json:"canaryPercent,omitempty"`

	// ObservedGeneration is the generation observed by the controller
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conditions represent the latest available observations of the policy's state
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// LastTransitionTime is the last time the phase changed
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty"`

	// Metrics track policy effectiveness
	// +optional
	Metrics *RiskRoutingMetrics `json:"metrics,omitempty"`
}

// RiskRoutingMetrics track policy effectiveness
type RiskRoutingMetrics struct {
	// TotalRequests is the total requests processed
	TotalRequests int64 `json:"totalRequests"`

	// AcceptedRequests is the number of accepted requests
	AcceptedRequests int64 `json:"acceptedRequests"`

	// EscalatedRequests is the number of escalated requests
	EscalatedRequests int64 `json:"escalatedRequests"`

	// RejectedRequests is the number of rejected requests
	RejectedRequests int64 `json:"rejectedRequests"`

	// AvgRiskScore is the average risk score
	// +optional
	AvgRiskScore float64 `json:"avgRiskScore,omitempty"`

	// HighRiskRate is the percentage of high-risk requests
	// +optional
	HighRiskRate float64 `json:"highRiskRate,omitempty"`
}

// +kubebuilder:object:root=true

// RiskRoutingPolicyList contains a list of RiskRoutingPolicy
type RiskRoutingPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []RiskRoutingPolicy `json:"items"`
}

func init() {
	SchemeBuilder.Register(&RiskRoutingPolicy{}, &RiskRoutingPolicyList{})
}

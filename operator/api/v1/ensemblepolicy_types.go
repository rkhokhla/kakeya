package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EnsemblePolicy defines ensemble verification policy (Phase 7 WP4)
// Configures N-of-M verification strategy with pluggable checks
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=ep
// +kubebuilder:printcolumn:name="N",type="integer",JSONPath=".spec.nOfM.n"
// +kubebuilder:printcolumn:name="M",type="integer",JSONPath=".spec.nOfM.m"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
type EnsemblePolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   EnsemblePolicySpec   `json:"spec,omitempty"`
	Status EnsemblePolicyStatus `json:"status,omitempty"`
}

// EnsemblePolicySpec defines the desired state of EnsemblePolicy
type EnsemblePolicySpec struct {
	// TenantSelector selects which tenants this policy applies to
	// +optional
	TenantSelector *TenantSelector `json:"tenantSelector,omitempty"`

	// NOfM defines N-of-M acceptance threshold
	NOfM NOfMConfig `json:"nOfM"`

	// EnabledChecks lists which verification strategies to use
	// +kubebuilder:validation:MinItems=1
	EnabledChecks []EnsembleCheck `json:"enabledChecks"`

	// TimeoutMs is the per-strategy timeout in milliseconds
	// +kubebuilder:default=100
	// +kubebuilder:validation:Minimum=10
	// +kubebuilder:validation:Maximum=5000
	TimeoutMs int `json:"timeoutMs"`

	// FailMode defines behavior when N-of-M threshold not met
	// +kubebuilder:validation:Enum=open;closed
	// +kubebuilder:default="open"
	FailMode string `json:"failMode"`

	// MinConfidence is the minimum aggregate confidence threshold [0, 1]
	// +kubebuilder:default=0.7
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinConfidence float64 `json:"minConfidence"`

	// AuditDisagreements enables WORM logging for disagreements
	// +kubebuilder:default=true
	AuditDisagreements bool `json:"auditDisagreements"`

	// SIEMIntegration enables SIEM streaming for ensemble events
	// +optional
	SIEMIntegration *SIEMIntegration `json:"siemIntegration,omitempty"`

	// CanaryRollout defines canary rollout configuration
	// +optional
	CanaryRollout *CanaryRollout `json:"canaryRollout,omitempty"`
}

// NOfMConfig defines N-of-M acceptance threshold
type NOfMConfig struct {
	// N is the minimum number of strategies that must agree
	// +kubebuilder:validation:Minimum=1
	N int `json:"n"`

	// M is the total number of strategies
	// +kubebuilder:validation:Minimum=1
	M int `json:"m"`
}

// EnsembleCheck defines a verification strategy configuration
type EnsembleCheck struct {
	// Name of the check strategy
	// +kubebuilder:validation:Enum=pcs_recompute;retrieval_overlap;micro_vote
	Name string `json:"name"`

	// Enabled enables this check
	// +kubebuilder:default=true
	Enabled bool `json:"enabled"`

	// Weight is the importance weight for this check (0-1)
	// +optional
	// +kubebuilder:default=1.0
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	Weight float64 `json:"weight,omitempty"`

	// Config contains strategy-specific configuration
	// +optional
	Config *EnsembleCheckConfig `json:"config,omitempty"`
}

// EnsembleCheckConfig contains strategy-specific configuration
type EnsembleCheckConfig struct {
	// PCSRecompute configuration
	// +optional
	PCSRecompute *PCSRecomputeConfig `json:"pcsRecompute,omitempty"`

	// RetrievalOverlap configuration
	// +optional
	RetrievalOverlap *RetrievalOverlapConfig `json:"retrievalOverlap,omitempty"`

	// MicroVote configuration
	// +optional
	MicroVote *MicroVoteConfig `json:"microVote,omitempty"`
}

// PCSRecomputeConfig configures PCS recomputation strategy
type PCSRecomputeConfig struct {
	// Tolerance for D̂ recomputation (e.g., 0.15)
	// +kubebuilder:default=0.15
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	Tolerance float64 `json:"tolerance"`
}

// RetrievalOverlapConfig configures retrieval overlap strategy
type RetrievalOverlapConfig struct {
	// MinJaccard is the minimum Jaccard similarity [0, 1]
	// +kubebuilder:default=0.6
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinJaccard float64 `json:"minJaccard"`

	// ShingleSize is the n-gram size for shingling
	// +kubebuilder:default=3
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=10
	ShingleSize int `json:"shingleSize"`
}

// MicroVoteConfig configures micro-vote strategy
type MicroVoteConfig struct {
	// ModelEndpoint is the auxiliary model API endpoint
	// +optional
	ModelEndpoint string `json:"modelEndpoint,omitempty"`

	// Threshold is the minimum agreement score [0, 1]
	// +kubebuilder:default=0.7
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	Threshold float64 `json:"threshold"`

	// TimeoutMs is the model API timeout in milliseconds
	// +kubebuilder:default=200
	// +kubebuilder:validation:Minimum=10
	// +kubebuilder:validation:Maximum=5000
	TimeoutMs int `json:"timeoutMs"`
}

// SIEMIntegration configures SIEM streaming
type SIEMIntegration struct {
	// Enabled enables SIEM streaming
	Enabled bool `json:"enabled"`

	// Provider is the SIEM provider
	// +kubebuilder:validation:Enum=splunk;datadog;elastic;sumologic
	Provider string `json:"provider"`

	// Endpoint is the SIEM ingestion endpoint
	Endpoint string `json:"endpoint"`

	// SecretRef references the secret containing SIEM credentials
	SecretRef SecretReference `json:"secretRef"`

	// EventTypes lists which events to stream
	// +optional
	EventTypes []string `json:"eventTypes,omitempty"`
}

// SecretReference references a Kubernetes secret
type SecretReference struct {
	// Name is the secret name
	Name string `json:"name"`

	// Namespace is the secret namespace
	// +optional
	Namespace string `json:"namespace,omitempty"`

	// Key is the key within the secret
	// +optional
	// +kubebuilder:default="token"
	Key string `json:"key,omitempty"`
}

// EnsemblePolicyStatus defines the observed state of EnsemblePolicy
type EnsemblePolicyStatus struct {
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

	// Metrics track ensemble verification effectiveness
	// +optional
	Metrics *EnsembleMetrics `json:"metrics,omitempty"`
}

// EnsembleMetrics track ensemble verification effectiveness
type EnsembleMetrics struct {
	// TotalVerifications is the total verifications performed
	TotalVerifications int64 `json:"totalVerifications"`

	// Accepted is the number of accepted verifications
	Accepted int64 `json:"accepted"`

	// Rejected is the number of rejected verifications
	Rejected int64 `json:"rejected"`

	// Escalated is the number of escalated verifications
	Escalated int64 `json:"escalated"`

	// Disagreements is the number of N-of-M disagreements
	Disagreements int64 `json:"disagreements"`

	// AvgLatencyMs is the average ensemble latency in milliseconds
	// +optional
	AvgLatencyMs float64 `json:"avgLatencyMs,omitempty"`

	// AvgConfidence is the average aggregate confidence
	// +optional
	AvgConfidence float64 `json:"avgConfidence,omitempty"`

	// StrategyAgreementRates tracks per-strategy agreement rates
	// +optional
	StrategyAgreementRates map[string]float64 `json:"strategyAgreementRates,omitempty"`
}

// +kubebuilder:object:root=true

// EnsemblePolicyList contains a list of EnsemblePolicy
type EnsemblePolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []EnsemblePolicy `json:"items"`
}

func init() {
	SchemeBuilder.Register(&EnsemblePolicy{}, &EnsemblePolicyList{})
}

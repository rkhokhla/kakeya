package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EnsembleBanditPolicy defines bandit-tuned ensemble configuration per tenant
// Phase 9 WP2: Thompson sampling/UCB for N-of-M optimization
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
type EnsembleBanditPolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   EnsembleBanditPolicySpec   `json:"spec,omitempty"`
	Status EnsembleBanditPolicyStatus `json:"status,omitempty"`
}

// EnsembleBanditPolicySpec defines the desired bandit configuration
type EnsembleBanditPolicySpec struct {
	// TenantID for this policy (or "*" for default)
	TenantID string `json:"tenant_id"`

	// ExplorationStrategy: "thompson_sampling" or "ucb"
	ExplorationStrategy string `json:"exploration_strategy"`

	// ExplorationRate: probability of random arm selection (epsilon-greedy)
	ExplorationRate float64 `json:"exploration_rate"`

	// RewardFunction configuration
	RewardFunction RewardFunctionConfig `json:"reward_function"`

	// Constraints
	Constraints BanditConstraints `json:"constraints"`

	// ArmSpace: ensemble configurations to explore
	ArmSpace []EnsembleArmConfig `json:"arm_space"`
}

// RewardFunctionConfig defines reward function weights
type RewardFunctionConfig struct {
	ContainmentWeight float64 `json:"containment_weight"` // e.g., 0.5
	AgreementWeight   float64 `json:"agreement_weight"`   // e.g., 0.3
	LatencyPenalty    float64 `json:"latency_penalty"`    // e.g., 0.01
	CostPenalty       float64 `json:"cost_penalty"`       // e.g., 0.1
}

// BanditConstraints defines SLO constraints
type BanditConstraints struct {
	MaxLatencyMs float64 `json:"max_latency_ms"` // e.g., 120ms p95
	MaxCostDelta float64 `json:"max_cost_delta"` // e.g., +7%
}

// EnsembleArmConfig defines an ensemble configuration option
type EnsembleArmConfig struct {
	N int `json:"n"` // Required agreements
	M int `json:"m"` // Total strategies

	// Strategy weights
	Weights map[string]float64 `json:"weights"`
}

// EnsembleBanditPolicyStatus defines observed state
type EnsembleBanditPolicyStatus struct {
	// Phase: Pending, Active, Failed
	Phase string `json:"phase"`

	// Current best arm
	BestArm int `json:"best_arm"`

	// Performance metrics
	AvgReward    float64 `json:"avg_reward"`
	TotalPulls   int64   `json:"total_pulls"`
	LastUpdate   string  `json:"last_update"`

	// SLO compliance
	LatencyCompliance bool `json:"latency_compliance"`
	CostCompliance    bool `json:"cost_compliance"`
}

// +kubebuilder:object:root=true
// EnsembleBanditPolicyList contains a list of EnsembleBanditPolicy
type EnsembleBanditPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []EnsembleBanditPolicy `json:"items"`
}

func init() {
	SchemeBuilder.Register(&EnsembleBanditPolicy{}, &EnsembleBanditPolicyList{})
}

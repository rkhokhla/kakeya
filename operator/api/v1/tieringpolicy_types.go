package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TieringPolicySpec defines the desired state of TieringPolicy
type TieringPolicySpec struct {
	// HotTier configuration
	HotTier TierConfig `json:"hotTier"`

	// WarmTier configuration
	WarmTier TierConfig `json:"warmTier"`

	// ColdTier configuration
	ColdTier TierConfig `json:"coldTier"`

	// DemotionInterval is how often to run demotion (default: 5m)
	DemotionInterval metav1.Duration `json:"demotionInterval,omitempty"`

	// PredictivePromotion enables ML-based pre-warming
	PredictivePromotion bool `json:"predictivePromotion,omitempty"`

	// CostOptimization enables cost-aware tier selection
	CostOptimization bool `json:"costOptimization,omitempty"`
}

// TierConfig defines configuration for a storage tier
type TierConfig struct {
	// Backend is the storage backend (redis, postgres, s3, gcs)
	Backend string `json:"backend"`

	// TTL is the time-to-live before demotion
	TTL metav1.Duration `json:"ttl"`

	// CompressionLevel (0-9, 0=disabled)
	CompressionLevel int `json:"compressionLevel,omitempty"`

	// Encryption enables server-side encryption
	Encryption bool `json:"encryption,omitempty"`

	// TargetLatencyP95 in milliseconds
	TargetLatencyP95 float64 `json:"targetLatencyP95"`

	// MaxCostPerGB in dollars per month
	MaxCostPerGB float64 `json:"maxCostPerGB,omitempty"`
}

// TieringPolicyStatus defines the observed state of TieringPolicy
type TieringPolicyStatus struct {
	// Active indicates if the policy is currently active
	Active bool `json:"active"`

	// TieringStats tracks tiering metrics
	TieringStats TieringStats `json:"tieringStats,omitempty"`

	// Conditions represent the latest available observations
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// LastDemotionTime is when demotion last ran
	LastDemotionTime metav1.Time `json:"lastDemotionTime,omitempty"`

	// Message provides human-readable status
	Message string `json:"message,omitempty"`
}

// TieringStats tracks tiering performance metrics
type TieringStats struct {
	// HotHits in the last hour
	HotHits int64 `json:"hotHits"`

	// WarmHits in the last hour
	WarmHits int64 `json:"warmHits"`

	// ColdHits in the last hour
	ColdHits int64 `json:"coldHits"`

	// Promotions in the last hour
	Promotions int64 `json:"promotions"`

	// Demotions in the last hour
	Demotions int64 `json:"demotions"`

	// AverageColdLatencyP95 in milliseconds
	AverageColdLatencyP95 float64 `json:"averageColdLatencyP95"`

	// EstimatedMonthlyCost in dollars
	EstimatedMonthlyCost float64 `json:"estimatedMonthlyCost,omitempty"`

	// CostSavings compared to all-hot tier (percentage)
	CostSavings float64 `json:"costSavings,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced
// +kubebuilder:printcolumn:name="Active",type=boolean,JSONPath=`.status.active`
// +kubebuilder:printcolumn:name="ColdLatency",type=string,JSONPath=`.status.tieringStats.averageColdLatencyP95`
// +kubebuilder:printcolumn:name="Savings",type=string,JSONPath=`.status.tieringStats.costSavings`
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// TieringPolicy is the Schema for the tieringpolicies API
type TieringPolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TieringPolicySpec   `json:"spec,omitempty"`
	Status TieringPolicyStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TieringPolicyList contains a list of TieringPolicy
type TieringPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TieringPolicy `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TieringPolicy{}, &TieringPolicyList{})
}

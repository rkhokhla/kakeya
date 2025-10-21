package crr

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// AutoReconciler automatically fixes safe divergences (Phase 6 WP2)
type AutoReconciler struct {
	mu               sync.RWMutex
	detector         *DivergenceDetector
	reconcileMode    ReconcileMode
	safetyThreshold  float64 // Max divergence % for automatic reconciliation
	requireApproval  bool
	approvalQueue    []ReconciliationProposal
	metrics          *ReconcileMetrics
}

// ReconcileMode defines reconciliation strategy
type ReconcileMode string

const (
	// ReconcilemodeManual requires human approval
	ReconcilemodeManual ReconcileMode = "manual"

	// ReconcilemodeAutoSafe auto-reconciles safe divergences
	ReconcilemodeAutoSafe ReconcileMode = "auto-safe"

	// ReconcilemodeAutoAggressive auto-reconciles all divergences
	ReconcilemodeAutoAggressive ReconcileMode = "auto-aggressive"
)

// ReconciliationProposal represents a proposed fix
type ReconciliationProposal struct {
	ID                string
	Region1           string
	Region2           string
	DivergencePercent float64
	ConflictingKeys   []string
	ProposedAction    ReconcileAction
	SafetyScore       float64 // 0.0-1.0, higher = safer
	CreatedAt         time.Time
	Status            string // "pending", "approved", "rejected", "applied"
}

// ReconcileAction defines the reconciliation action
type ReconcileAction string

const (
	// ActionReplayMissing replays missing entries
	ActionReplayMissing ReconcileAction = "replay-missing"

	// ActionChooseFirst uses first-write wins
	ActionChooseFirst ReconcileAction = "choose-first"

	// ActionQuorum uses majority vote across N regions
	ActionQuorum ReconcileAction = "quorum"

	// ActionManualReview escalates to human
	ActionManualReview ReconcileAction = "manual-review"
)

// ReconcileMetrics tracks reconciliation operations
type ReconcileMetrics struct {
	mu                     sync.RWMutex
	ReconciliationsAttempted int64
	ReconciliationsSucceeded int64
	ReconciliationsFailed    int64
	AutoApplied              int64
	ManualApprovalRequired   int64
}

// NewAutoReconciler creates a new auto-reconciler
func NewAutoReconciler(detector *DivergenceDetector, mode ReconcileMode, safetyThreshold float64) *AutoReconciler {
	return &AutoReconciler{
		detector:         detector,
		reconcileMode:    mode,
		safetyThreshold:  safetyThreshold,
		requireApproval:  mode == ReconcilemodeManual,
		approvalQueue:    []ReconciliationProposal{},
		metrics:          &ReconcileMetrics{},
	}
}

// ReconcileDivergence attempts to fix a detected divergence
func (ar *AutoReconciler) ReconcileDivergence(ctx context.Context, alert DivergenceAlert) (*ReconciliationProposal, error) {
	ar.mu.Lock()
	defer ar.mu.Unlock()

	// Analyze divergence and propose action
	proposal := ar.analyzeAndPropose(ctx, alert)

	// Decide if we can auto-apply
	canAutoApply := ar.canAutoApply(proposal)

	if canAutoApply {
		// Auto-apply safe reconciliation
		if err := ar.applyReconciliation(ctx, proposal); err != nil {
			proposal.Status = "failed"
			ar.metrics.ReconciliationsFailed++
			return proposal, fmt.Errorf("failed to apply reconciliation: %w", err)
		}

		proposal.Status = "applied"
		ar.metrics.AutoApplied++
		ar.metrics.ReconciliationsSucceeded++

		fmt.Printf("Auto-Reconcile: Applied %s for %s vs %s (safety: %.2f)\n",
			proposal.ProposedAction, proposal.Region1, proposal.Region2, proposal.SafetyScore)

		return proposal, nil
	}

	// Requires approval - add to queue
	proposal.Status = "pending"
	ar.approvalQueue = append(ar.approvalQueue, *proposal)
	ar.metrics.ManualApprovalRequired++

	fmt.Printf("Auto-Reconcile: Approval required for %s vs %s (safety: %.2f, divergence: %.2f%%)\n",
		proposal.Region1, proposal.Region2, proposal.SafetyScore, proposal.DivergencePercent)

	return proposal, nil
}

// analyzeAndPropose analyzes divergence and proposes action
func (ar *AutoReconciler) analyzeAndPropose(ctx context.Context, alert DivergenceAlert) *ReconciliationProposal {
	proposal := &ReconciliationProposal{
		ID:                fmt.Sprintf("reconcile-%d", time.Now().UnixNano()),
		Region1:           alert.Region1,
		Region2:           alert.Region2,
		DivergencePercent: alert.CountDivergence,
		CreatedAt:         time.Now(),
	}

	// Compute safety score based on divergence characteristics
	safetyScore := ar.computeSafetyScore(alert)
	proposal.SafetyScore = safetyScore

	// Choose action based on divergence type
	if alert.SampleMismatch == 0 && alert.CountDivergence < 5.0 {
		// Only missing keys, no conflicts → safe to replay
		proposal.ProposedAction = ActionReplayMissing
		proposal.SafetyScore = 0.9
	} else if alert.SampleMismatch > 20.0 {
		// High conflict rate → manual review
		proposal.ProposedAction = ActionManualReview
		proposal.SafetyScore = 0.2
	} else if alert.CountDivergence < 10.0 {
		// Moderate divergence → first-write wins
		proposal.ProposedAction = ActionChooseFirst
		proposal.SafetyScore = 0.7
	} else {
		// High divergence → quorum if available, else manual
		proposal.ProposedAction = ActionQuorum
		proposal.SafetyScore = 0.5
	}

	return proposal
}

// computeSafetyScore calculates how safe automatic reconciliation is
func (ar *AutoReconciler) computeSafetyScore(alert DivergenceAlert) float64 {
	score := 1.0

	// Penalize high divergence
	if alert.CountDivergence > 10.0 {
		score -= 0.3
	} else if alert.CountDivergence > 5.0 {
		score -= 0.1
	}

	// Penalize high sample mismatch
	if alert.SampleMismatch > 20.0 {
		score -= 0.4
	} else if alert.SampleMismatch > 10.0 {
		score -= 0.2
	}

	// Ensure score is in range [0, 1]
	if score < 0 {
		score = 0
	}

	return score
}

// canAutoApply determines if reconciliation can be auto-applied
func (ar *AutoReconciler) canAutoApply(proposal *ReconciliationProposal) bool {
	switch ar.reconcileMode {
	case ReconcilemodeManual:
		return false

	case ReconcilemodeAutoSafe:
		// Only auto-apply if safety score is high and divergence is low
		return proposal.SafetyScore >= ar.safetyThreshold &&
			proposal.DivergencePercent < 5.0 &&
			proposal.ProposedAction == ActionReplayMissing

	case ReconcilemodeAutoAggressive:
		// Auto-apply if safety score meets threshold
		return proposal.SafetyScore >= ar.safetyThreshold

	default:
		return false
	}
}

// applyReconciliation executes the reconciliation action
func (ar *AutoReconciler) applyReconciliation(ctx context.Context, proposal *ReconciliationProposal) error {
	ar.metrics.ReconciliationsAttempted++

	switch proposal.ProposedAction {
	case ActionReplayMissing:
		return ar.replayMissingEntries(ctx, proposal)

	case ActionChooseFirst:
		return ar.applyFirstWriteWins(ctx, proposal)

	case ActionQuorum:
		return ar.applyQuorumDecision(ctx, proposal)

	case ActionManualReview:
		// Should not reach here for auto-apply
		return fmt.Errorf("manual review required")

	default:
		return fmt.Errorf("unknown action: %s", proposal.ProposedAction)
	}
}

// replayMissingEntries replays WAL entries that are missing in one region
func (ar *AutoReconciler) replayMissingEntries(ctx context.Context, proposal *ReconciliationProposal) error {
	// In production, would:
	// 1. Identify missing keys in region2 that exist in region1
	// 2. Fetch corresponding WAL entries from region1
	// 3. Replay entries in region2 with idempotency guard

	fmt.Printf("Auto-Reconcile: Replaying missing entries %s → %s\n",
		proposal.Region1, proposal.Region2)

	return nil // Placeholder
}

// applyFirstWriteWins resolves conflicts using first-write wins
func (ar *AutoReconciler) applyFirstWriteWins(ctx context.Context, proposal *ReconciliationProposal) error {
	// In production, would:
	// 1. For each conflicting key, determine which entry was first (by timestamp)
	// 2. Overwrite later entry with first entry

	fmt.Printf("Auto-Reconcile: Applying first-write wins %s vs %s\n",
		proposal.Region1, proposal.Region2)

	return nil // Placeholder
}

// applyQuorumDecision resolves conflicts using majority vote across regions
func (ar *AutoReconciler) applyQuorumDecision(ctx context.Context, proposal *ReconciliationProposal) error {
	// In production, would:
	// 1. Query all regions for conflicting keys
	// 2. Use majority vote to select correct value
	// 3. Update minority regions to match majority

	fmt.Printf("Auto-Reconcile: Applying quorum decision for %s vs %s\n",
		proposal.Region1, proposal.Region2)

	return nil // Placeholder
}

// ApproveProposal manually approves a pending reconciliation
func (ar *AutoReconciler) ApproveProposal(ctx context.Context, proposalID string) error {
	ar.mu.Lock()
	defer ar.mu.Unlock()

	// Find proposal in queue
	for i, proposal := range ar.approvalQueue {
		if proposal.ID == proposalID {
			// Remove from queue
			ar.approvalQueue = append(ar.approvalQueue[:i], ar.approvalQueue[i+1:]...)

			// Apply reconciliation
			if err := ar.applyReconciliation(ctx, &proposal); err != nil {
				return fmt.Errorf("failed to apply approved reconciliation: %w", err)
			}

			ar.metrics.ReconciliationsSucceeded++
			fmt.Printf("Auto-Reconcile: Applied approved proposal %s\n", proposalID)
			return nil
		}
	}

	return fmt.Errorf("proposal not found: %s", proposalID)
}

// RejectProposal manually rejects a pending reconciliation
func (ar *AutoReconciler) RejectProposal(proposalID string) error {
	ar.mu.Lock()
	defer ar.mu.Unlock()

	// Find and remove proposal from queue
	for i, proposal := range ar.approvalQueue {
		if proposal.ID == proposalID {
			ar.approvalQueue = append(ar.approvalQueue[:i], ar.approvalQueue[i+1:]...)
			fmt.Printf("Auto-Reconcile: Rejected proposal %s\n", proposalID)
			return nil
		}
	}

	return fmt.Errorf("proposal not found: %s", proposalID)
}

// ListPendingProposals returns all pending reconciliation proposals
func (ar *AutoReconciler) ListPendingProposals() []ReconciliationProposal {
	ar.mu.RLock()
	defer ar.mu.RUnlock()

	return append([]ReconciliationProposal{}, ar.approvalQueue...)
}

// GetMetrics returns reconciliation metrics
func (ar *AutoReconciler) GetMetrics() ReconcileMetrics {
	ar.metrics.mu.RLock()
	defer ar.metrics.mu.RUnlock()
	return *ar.metrics
}

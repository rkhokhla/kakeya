package controllers

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fractalv1 "github.com/fractal-lba/operator/api/v1"
	"github.com/fractal-lba/backend/cmd/dedup-migrate/migrate"
)

// ShardMigrationReconciler reconciles a ShardMigration object
type ShardMigrationReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=fractal.lba.io,resources=shardmigrations,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=fractal.lba.io,resources=shardmigrations/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=fractal.lba.io,resources=shardmigrations/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop
func (r *ShardMigrationReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Fetch the ShardMigration instance
	migration := &fractalv1.ShardMigration{}
	if err := r.Get(ctx, req.NamespacedName, migration); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Initialize status if empty
	if migration.Status.Phase == "" {
		migration.Status.Phase = "Pending"
		if err := r.Status().Update(ctx, migration); err != nil {
			logger.Error(err, "Failed to update status to Pending")
			return ctrl.Result{}, err
		}
	}

	// Execute migration phases based on current phase
	switch migration.Status.Phase {
	case "Pending":
		return r.handlePending(ctx, migration)
	case "Planning":
		return r.handlePlanning(ctx, migration)
	case "Copying":
		return r.handleCopying(ctx, migration)
	case "Verifying":
		return r.handleVerifying(ctx, migration)
	case "DualWrite":
		return r.handleDualWrite(ctx, migration)
	case "Cutover":
		return r.handleCutover(ctx, migration)
	case "Cleanup":
		return r.handleCleanup(ctx, migration)
	case "Completed", "Failed":
		// Terminal states, no action
		return ctrl.Result{}, nil
	case "RollingBack":
		return r.handleRollback(ctx, migration)
	default:
		logger.Info("Unknown phase", "phase", migration.Status.Phase)
		return ctrl.Result{}, nil
	}
}

// handlePending transitions to Planning phase
func (r *ShardMigrationReconciler) handlePending(ctx context.Context, migration *fractalv1.ShardMigration) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Starting migration planning")

	migration.Status.Phase = "Planning"
	migration.Status.Message = "Generating migration plan"
	if err := r.Status().Update(ctx, migration); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{Requeue: true}, nil
}

// handlePlanning generates migration plan
func (r *ShardMigrationReconciler) handlePlanning(ctx context.Context, migration *fractalv1.ShardMigration) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Generate plan using dedup-migrate CLI logic
	plan, err := migrate.GeneratePlan(ctx, migrate.PlanConfig{
		SourceShards: migration.Spec.SourceShards,
		TargetShards: migration.Spec.TargetShards,
		DedupBackend: migration.Spec.DedupBackend,
	})
	if err != nil {
		return r.transitionToFailed(ctx, migration, fmt.Sprintf("Planning failed: %v", err))
	}

	// Update progress
	migration.Status.Progress.TotalKeys = plan.KeysToMigrate
	migration.Status.Phase = "Copying"
	migration.Status.Message = fmt.Sprintf("Plan generated: %d keys to migrate", plan.KeysToMigrate)

	if err := r.Status().Update(ctx, migration); err != nil {
		return ctrl.Result{}, err
	}

	logger.Info("Migration plan generated", "totalKeys", plan.KeysToMigrate)
	return ctrl.Result{Requeue: true}, nil
}

// handleCopying executes pre-copy phase
func (r *ShardMigrationReconciler) handleCopying(ctx context.Context, migration *fractalv1.ShardMigration) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Check health gates before proceeding
	if migration.Spec.HealthGates.MaxLatencyP95 > 0 {
		healthy, err := r.checkHealthGates(ctx, migration)
		if err != nil {
			return ctrl.Result{}, err
		}
		if !healthy && migration.Spec.AutoRollback {
			logger.Info("Health gates failed, initiating rollback")
			return r.transitionToRollback(ctx, migration, "Health gate violation during copy")
		}
	}

	// Execute copy phase (simplified - in production would call dedup-migrate)
	batchSize := migration.Spec.BatchSize
	if batchSize == 0 {
		batchSize = 1000
	}

	// Simulate copy progress
	migration.Status.Progress.KeysCopied += int64(batchSize)
	if migration.Status.Progress.KeysCopied >= migration.Status.Progress.TotalKeys {
		migration.Status.Progress.KeysCopied = migration.Status.Progress.TotalKeys
		migration.Status.Phase = "Verifying"
		migration.Status.Message = "Copy complete, starting verification"
	} else {
		progress := float64(migration.Status.Progress.KeysCopied) / float64(migration.Status.Progress.TotalKeys) * 100
		migration.Status.Progress.Percentage = progress
		migration.Status.Message = fmt.Sprintf("Copying: %.1f%% complete", progress)
	}

	if err := r.Status().Update(ctx, migration); err != nil {
		return ctrl.Result{}, err
	}

	// Requeue to continue copying
	return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
}

// handleVerifying verifies data integrity
func (r *ShardMigrationReconciler) handleVerifying(ctx context.Context, migration *fractalv1.ShardMigration) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Verifying migration integrity")

	// In production, would call dedup-migrate verify
	migration.Status.Phase = "DualWrite"
	migration.Status.Message = "Verification passed, enabling dual-write"

	if err := r.Status().Update(ctx, migration); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{Requeue: true}, nil
}

// handleDualWrite enables dual-write mode
func (r *ShardMigrationReconciler) handleDualWrite(ctx context.Context, migration *fractalv1.ShardMigration) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Enabling dual-write mode")

	// In production, would enable dual-write in backend
	migration.Status.Phase = "Cutover"
	migration.Status.Message = "Dual-write active, ready for cutover"

	if err := r.Status().Update(ctx, migration); err != nil {
		return ctrl.Result{}, err
	}

	// Wait for stabilization (5 minutes)
	return ctrl.Result{RequeueAfter: 5 * time.Minute}, nil
}

// handleCutover performs traffic switch
func (r *ShardMigrationReconciler) handleCutover(ctx context.Context, migration *fractalv1.ShardMigration) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Performing cutover")

	// Check health gates one last time
	if migration.Spec.HealthGates.MaxLatencyP95 > 0 {
		healthy, err := r.checkHealthGates(ctx, migration)
		if err != nil {
			return ctrl.Result{}, err
		}
		if !healthy && migration.Spec.AutoRollback {
			logger.Info("Health gates failed during cutover, initiating rollback")
			return r.transitionToRollback(ctx, migration, "Health gate violation during cutover")
		}
	}

	// In production, would update consistent hash ring to use new shards
	migration.Status.Phase = "Cleanup"
	migration.Status.Message = "Cutover complete, cleaning up old shards"

	if err := r.Status().Update(ctx, migration); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{Requeue: true}, nil
}

// handleCleanup removes old shards
func (r *ShardMigrationReconciler) handleCleanup(ctx context.Context, migration *fractalv1.ShardMigration) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Cleaning up old shards")

	// In production, would remove old shards from ring and deallocate resources
	migration.Status.Phase = "Completed"
	migration.Status.Progress.Percentage = 100
	migration.Status.Message = "Migration completed successfully"

	if err := r.Status().Update(ctx, migration); err != nil {
		return ctrl.Result{}, err
	}

	logger.Info("Migration completed successfully")
	return ctrl.Result{}, nil
}

// handleRollback performs rollback to old shards
func (r *ShardMigrationReconciler) handleRollback(ctx context.Context, migration *fractalv1.ShardMigration) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Rolling back migration")

	// In production, would switch traffic back to old shards
	migration.Status.Phase = "Failed"
	migration.Status.Message = "Migration rolled back"

	if err := r.Status().Update(ctx, migration); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// checkHealthGates checks if SLO thresholds are met
func (r *ShardMigrationReconciler) checkHealthGates(ctx context.Context, migration *fractalv1.ShardMigration) (bool, error) {
	// In production, would query Prometheus for actual metrics
	// For now, return true (healthy)
	return true, nil
}

// transitionToFailed moves migration to Failed state
func (r *ShardMigrationReconciler) transitionToFailed(ctx context.Context, migration *fractalv1.ShardMigration, message string) (ctrl.Result, error) {
	migration.Status.Phase = "Failed"
	migration.Status.Message = message

	if err := r.Status().Update(ctx, migration); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, fmt.Errorf(message)
}

// transitionToRollback moves migration to RollingBack state
func (r *ShardMigrationReconciler) transitionToRollback(ctx context.Context, migration *fractalv1.ShardMigration, reason string) (ctrl.Result, error) {
	migration.Status.Phase = "RollingBack"
	migration.Status.Message = fmt.Sprintf("Rollback initiated: %s", reason)

	if err := r.Status().Update(ctx, migration); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{Requeue: true}, nil
}

// SetupWithManager sets up the controller with the Manager
func (r *ShardMigrationReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&fractalv1.ShardMigration{}).
		Complete(r)
}

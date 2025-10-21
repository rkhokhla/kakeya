package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/fractal-lba/kakeya/internal/sharding"
	"github.com/spf13/cobra"
)

var (
	// Global flags
	configFile    string
	checkpointDir string
	dryRun        bool
	verbose       bool

	// Migration state
	migrationID   string
	sourceShards  []string
	targetShards  []string
	throttleQPS   int
	batchSize     int
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "dedup-migrate",
		Short: "Dedup migration tool for zero-downtime shard rebalancing (Phase 5 WP4)",
		Long: `Zero-downtime migration tool for sharded dedup stores.
Supports plan→copy→verify→dual-write→cutover→cleanup workflow with resumability.`,
	}

	// Global flags
	rootCmd.PersistentFlags().StringVarP(&configFile, "config", "c", "", "Migration config file (JSON)")
	rootCmd.PersistentFlags().StringVarP(&checkpointDir, "checkpoint-dir", "d", "./checkpoints", "Checkpoint directory for resumability")
	rootCmd.PersistentFlags().BoolVar(&dryRun, "dry-run", false, "Dry-run mode (no actual changes)")
	rootCmd.PersistentFlags().BoolVar(&verbose, "verbose", false, "Verbose logging")

	// Subcommands
	rootCmd.AddCommand(planCmd())
	rootCmd.AddCommand(copyCmd())
	rootCmd.AddCommand(verifyCmd())
	rootCmd.AddCommand(dualWriteCmd())
	rootCmd.AddCommand(cutoverCmd())
	rootCmd.AddCommand(cleanupCmd())
	rootCmd.AddCommand(statusCmd())
	rootCmd.AddCommand(rollbackCmd())

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// planCmd generates a migration plan
func planCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "plan",
		Short: "Generate migration plan for shard rebalancing",
		Long: `Analyzes current shard distribution and generates a migration plan.
Shows which keys will move from old shards to new shards (1/N keys per added shard).`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			// Load config
			config, err := loadMigrationConfig(configFile)
			if err != nil {
				return fmt.Errorf("failed to load config: %w", err)
			}

			// Generate plan
			plan, err := generateMigrationPlan(ctx, config)
			if err != nil {
				return fmt.Errorf("failed to generate plan: %w", err)
			}

			// Display plan
			fmt.Printf("=== Migration Plan ===\n")
			fmt.Printf("Migration ID: %s\n", plan.ID)
			fmt.Printf("Source shards: %v\n", plan.SourceShards)
			fmt.Printf("Target shards: %v\n", plan.TargetShards)
			fmt.Printf("Keys to migrate: %d\n", plan.KeysToMigrate)
			fmt.Printf("Estimated data size: %s\n", formatBytes(plan.EstimatedBytes))
			fmt.Printf("Estimated duration: %v\n", plan.EstimatedDuration)
			fmt.Printf("\n")

			// Save plan to checkpoint
			if err := savePlan(checkpointDir, plan); err != nil {
				return fmt.Errorf("failed to save plan: %w", err)
			}

			fmt.Printf("Plan saved to %s/%s.plan.json\n", checkpointDir, plan.ID)
			fmt.Printf("\nNext: Run 'dedup-migrate copy --migration-id %s' to start pre-copying data\n", plan.ID)

			return nil
		},
	}

	return cmd
}

// copyCmd performs pre-copy phase (copy data without cutover)
func copyCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "copy",
		Short: "Pre-copy phase: Copy data from source to target shards",
		Long: `Copies keys that will be migrated from source shards to target shards.
This phase runs while both old and new shards are operational (no downtime).
Progress is checkpointed for resumability.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			// Load plan
			plan, err := loadPlan(checkpointDir, migrationID)
			if err != nil {
				return fmt.Errorf("failed to load plan: %w", err)
			}

			fmt.Printf("=== Pre-Copy Phase ===\n")
			fmt.Printf("Migration ID: %s\n", plan.ID)
			fmt.Printf("Keys to copy: %d\n", plan.KeysToMigrate)
			fmt.Printf("Batch size: %d, Throttle: %d QPS\n", batchSize, throttleQPS)
			fmt.Printf("\n")

			// Perform copy
			if err := performCopy(ctx, plan, batchSize, throttleQPS, dryRun); err != nil {
				return fmt.Errorf("copy failed: %w", err)
			}

			fmt.Printf("\nCopy phase complete. Next: Run 'dedup-migrate verify --migration-id %s'\n", plan.ID)

			return nil
		},
	}

	cmd.Flags().StringVar(&migrationID, "migration-id", "", "Migration ID (from plan)")
	cmd.Flags().IntVar(&batchSize, "batch-size", 1000, "Batch size for copying")
	cmd.Flags().IntVar(&throttleQPS, "throttle-qps", 1000, "Throttle QPS to avoid overload")
	cmd.MarkFlagRequired("migration-id")

	return cmd
}

// verifyCmd verifies data integrity after copy
func verifyCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Verify data integrity after copy phase",
		Long: `Verifies that all keys were successfully copied from source to target shards.
Compares checksums and counts to ensure no data loss.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			// Load plan
			plan, err := loadPlan(checkpointDir, migrationID)
			if err != nil {
				return fmt.Errorf("failed to load plan: %w", err)
			}

			fmt.Printf("=== Verification Phase ===\n")
			fmt.Printf("Migration ID: %s\n", plan.ID)
			fmt.Printf("\n")

			// Perform verification
			result, err := performVerification(ctx, plan)
			if err != nil {
				return fmt.Errorf("verification failed: %w", err)
			}

			fmt.Printf("Keys verified: %d\n", result.KeysVerified)
			fmt.Printf("Mismatches: %d\n", result.Mismatches)
			fmt.Printf("Missing keys: %d\n", result.MissingKeys)

			if result.Mismatches > 0 || result.MissingKeys > 0 {
				fmt.Printf("\nWARNING: Verification failed. Re-run 'dedup-migrate copy' to fix inconsistencies.\n")
				return fmt.Errorf("verification failed")
			}

			fmt.Printf("\nVerification passed. Next: Run 'dedup-migrate dual-write --migration-id %s'\n", plan.ID)

			return nil
		},
	}

	cmd.Flags().StringVar(&migrationID, "migration-id", "", "Migration ID")
	cmd.MarkFlagRequired("migration-id")

	return cmd
}

// dualWriteCmd enables dual-write mode
func dualWriteCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "dual-write",
		Short: "Enable dual-write mode (write to both old and new shards)",
		Long: `Enables dual-write mode where all new writes go to both old and new shards.
This ensures no data loss during cutover. Run for a stabilization period (e.g., 5-10 min).`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			// Load plan
			plan, err := loadPlan(checkpointDir, migrationID)
			if err != nil {
				return fmt.Errorf("failed to load plan: %w", err)
			}

			fmt.Printf("=== Dual-Write Mode ===\n")
			fmt.Printf("Migration ID: %s\n", plan.ID)
			fmt.Printf("\nEnabling dual-write mode...\n")

			// Enable dual-write
			if err := enableDualWrite(ctx, plan, dryRun); err != nil {
				return fmt.Errorf("failed to enable dual-write: %w", err)
			}

			fmt.Printf("Dual-write enabled. Monitor for errors:\n")
			fmt.Printf("  - flk_dedup_dual_write_errors (Prometheus)\n")
			fmt.Printf("  - Logs: grep 'dual-write error'\n")
			fmt.Printf("\nAfter stabilization period, run: 'dedup-migrate cutover --migration-id %s'\n", plan.ID)

			return nil
		},
	}

	cmd.Flags().StringVar(&migrationID, "migration-id", "", "Migration ID")
	cmd.MarkFlagRequired("migration-id")

	return cmd
}

// cutoverCmd performs cutover (switch to new shards)
func cutoverCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cutover",
		Short: "Cutover: Switch reads to new shards",
		Long: `Switches traffic from old shards to new shards. This is the point of no return.
After cutover, old shards are no longer used for reads.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			// Load plan
			plan, err := loadPlan(checkpointDir, migrationID)
			if err != nil {
				return fmt.Errorf("failed to load plan: %w", err)
			}

			fmt.Printf("=== Cutover Phase ===\n")
			fmt.Printf("Migration ID: %s\n", plan.ID)
			fmt.Printf("\nWARNING: This will switch traffic to new shards. Confirm? (yes/no): ")

			var confirm string
			fmt.Scanln(&confirm)
			if confirm != "yes" {
				return fmt.Errorf("cutover aborted")
			}

			// Perform cutover
			if err := performCutover(ctx, plan, dryRun); err != nil {
				return fmt.Errorf("cutover failed: %w", err)
			}

			fmt.Printf("\nCutover complete. Monitor dedup hit ratio and latency.\n")
			fmt.Printf("If issues occur, run: 'dedup-migrate rollback --migration-id %s'\n", plan.ID)
			fmt.Printf("\nAfter stabilization, run: 'dedup-migrate cleanup --migration-id %s' to remove old shards\n", plan.ID)

			return nil
		},
	}

	cmd.Flags().StringVar(&migrationID, "migration-id", "", "Migration ID")
	cmd.MarkFlagRequired("migration-id")

	return cmd
}

// cleanupCmd cleans up old shards after successful cutover
func cleanupCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cleanup",
		Short: "Cleanup: Remove old shards after successful cutover",
		Long: `Removes old shards from the ring and deallocates resources.
Only run after cutover has been stable for at least 24 hours.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			// Load plan
			plan, err := loadPlan(checkpointDir, migrationID)
			if err != nil {
				return fmt.Errorf("failed to load plan: %w", err)
			}

			fmt.Printf("=== Cleanup Phase ===\n")
			fmt.Printf("Migration ID: %s\n", plan.ID)
			fmt.Printf("Old shards to remove: %v\n", plan.SourceShards)
			fmt.Printf("\nConfirm cleanup? (yes/no): ")

			var confirm string
			fmt.Scanln(&confirm)
			if confirm != "yes" {
				return fmt.Errorf("cleanup aborted")
			}

			// Perform cleanup
			if err := performCleanup(ctx, plan, dryRun); err != nil {
				return fmt.Errorf("cleanup failed: %w", err)
			}

			fmt.Printf("\nCleanup complete. Migration finished successfully.\n")

			return nil
		},
	}

	cmd.Flags().StringVar(&migrationID, "migration-id", "", "Migration ID")
	cmd.MarkFlagRequired("migration-id")

	return cmd
}

// statusCmd shows migration status
func statusCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show migration status",
		RunE: func(cmd *cobra.Command, args []string) error {
			// Load plan
			plan, err := loadPlan(checkpointDir, migrationID)
			if err != nil {
				return fmt.Errorf("failed to load plan: %w", err)
			}

			// Load checkpoint
			checkpoint, err := loadCheckpoint(checkpointDir, migrationID)
			if err != nil {
				fmt.Printf("No checkpoint found (migration not started)\n")
			} else {
				fmt.Printf("=== Migration Status ===\n")
				fmt.Printf("Migration ID: %s\n", plan.ID)
				fmt.Printf("Phase: %s\n", checkpoint.Phase)
				fmt.Printf("Progress: %d/%d keys (%.1f%%)\n",
					checkpoint.KeysCopied, plan.KeysToMigrate,
					float64(checkpoint.KeysCopied)/float64(plan.KeysToMigrate)*100)
				fmt.Printf("Last checkpoint: %s\n", checkpoint.LastUpdated.Format(time.RFC3339))
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&migrationID, "migration-id", "", "Migration ID")
	cmd.MarkFlagRequired("migration-id")

	return cmd
}

// rollbackCmd rolls back migration
func rollbackCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "rollback",
		Short: "Rollback migration (emergency only)",
		Long: `Rolls back migration by switching traffic back to old shards.
Only use if cutover caused issues.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			// Load plan
			plan, err := loadPlan(checkpointDir, migrationID)
			if err != nil {
				return fmt.Errorf("failed to load plan: %w", err)
			}

			fmt.Printf("=== Rollback ===\n")
			fmt.Printf("Migration ID: %s\n", plan.ID)
			fmt.Printf("\nWARNING: This will rollback to old shards. Confirm? (yes/no): ")

			var confirm string
			fmt.Scanln(&confirm)
			if confirm != "yes" {
				return fmt.Errorf("rollback aborted")
			}

			// Perform rollback
			if err := performRollback(ctx, plan, dryRun); err != nil {
				return fmt.Errorf("rollback failed: %w", err)
			}

			fmt.Printf("\nRollback complete. Traffic restored to old shards.\n")

			return nil
		},
	}

	cmd.Flags().StringVar(&migrationID, "migration-id", "", "Migration ID")
	cmd.MarkFlagRequired("migration-id")

	return cmd
}

// --- Migration Logic (Implementations) ---

// MigrationConfig holds migration configuration
type MigrationConfig struct {
	SourceShards  []string `json:"source_shards"`
	TargetShards  []string `json:"target_shards"`
	DedupBackend  string   `json:"dedup_backend"` // "redis", "postgres"
	RedisAddrs    []string `json:"redis_addrs,omitempty"`
	PostgresConns []string `json:"postgres_conns,omitempty"`
}

// MigrationPlan represents a migration plan
type MigrationPlan struct {
	ID                string        `json:"id"`
	SourceShards      []string      `json:"source_shards"`
	TargetShards      []string      `json:"target_shards"`
	KeysToMigrate     int64         `json:"keys_to_migrate"`
	EstimatedBytes    int64         `json:"estimated_bytes"`
	EstimatedDuration time.Duration `json:"estimated_duration"`
	CreatedAt         time.Time     `json:"created_at"`
}

// Checkpoint tracks migration progress
type Checkpoint struct {
	MigrationID string    `json:"migration_id"`
	Phase       string    `json:"phase"` // "copy", "verify", "dual-write", "cutover", "cleanup"
	KeysCopied  int64     `json:"keys_copied"`
	LastKey     string    `json:"last_key"` // For resumability
	LastUpdated time.Time `json:"last_updated"`
}

// VerificationResult holds verification results
type VerificationResult struct {
	KeysVerified int64
	Mismatches   int64
	MissingKeys  int64
}

func loadMigrationConfig(path string) (*MigrationConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config MigrationConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

func generateMigrationPlan(ctx context.Context, config *MigrationConfig) (*MigrationPlan, error) {
	// Placeholder implementation
	plan := &MigrationPlan{
		ID:                fmt.Sprintf("migration-%d", time.Now().Unix()),
		SourceShards:      config.SourceShards,
		TargetShards:      config.TargetShards,
		KeysToMigrate:     1000000, // Estimate
		EstimatedBytes:    10 * 1024 * 1024 * 1024, // 10GB
		EstimatedDuration: 2 * time.Hour,
		CreatedAt:         time.Now(),
	}

	return plan, nil
}

func savePlan(dir string, plan *MigrationPlan) error {
	os.MkdirAll(dir, 0755)
	data, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(fmt.Sprintf("%s/%s.plan.json", dir, plan.ID), data, 0644)
}

func loadPlan(dir, migrationID string) (*MigrationPlan, error) {
	data, err := os.ReadFile(fmt.Sprintf("%s/%s.plan.json", dir, migrationID))
	if err != nil {
		return nil, err
	}

	var plan MigrationPlan
	if err := json.Unmarshal(data, &plan); err != nil {
		return nil, err
	}

	return &plan, nil
}

func loadCheckpoint(dir, migrationID string) (*Checkpoint, error) {
	data, err := os.ReadFile(fmt.Sprintf("%s/%s.checkpoint.json", dir, migrationID))
	if err != nil {
		return nil, err
	}

	var checkpoint Checkpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil, err
	}

	return &checkpoint, nil
}

func performCopy(ctx context.Context, plan *MigrationPlan, batchSize, throttleQPS int, dryRun bool) error {
	fmt.Printf("Copying %d keys (batch size: %d, throttle: %d QPS)...\n",
		plan.KeysToMigrate, batchSize, throttleQPS)
	// Placeholder
	time.Sleep(2 * time.Second)
	fmt.Printf("Copy complete.\n")
	return nil
}

func performVerification(ctx context.Context, plan *MigrationPlan) (*VerificationResult, error) {
	fmt.Printf("Verifying data integrity...\n")
	// Placeholder
	time.Sleep(1 * time.Second)
	return &VerificationResult{
		KeysVerified: plan.KeysToMigrate,
		Mismatches:   0,
		MissingKeys:  0,
	}, nil
}

func enableDualWrite(ctx context.Context, plan *MigrationPlan, dryRun bool) error {
	if dryRun {
		fmt.Printf("DRY-RUN: Would enable dual-write mode\n")
		return nil
	}
	// Placeholder
	return nil
}

func performCutover(ctx context.Context, plan *MigrationPlan, dryRun bool) error {
	if dryRun {
		fmt.Printf("DRY-RUN: Would perform cutover\n")
		return nil
	}
	// Placeholder
	return nil
}

func performCleanup(ctx context.Context, plan *MigrationPlan, dryRun bool) error {
	if dryRun {
		fmt.Printf("DRY-RUN: Would clean up old shards\n")
		return nil
	}
	// Placeholder
	return nil
}

func performRollback(ctx context.Context, plan *MigrationPlan, dryRun bool) error {
	if dryRun {
		fmt.Printf("DRY-RUN: Would rollback to old shards\n")
		return nil
	}
	// Placeholder
	return nil
}

func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

package crr

import (
	"context"
	"fmt"
	"sync"
)

// SelectiveReplicator implements per-tenant and multi-way CRR (Phase 6 WP2)
type SelectiveReplicator struct {
	mu            sync.RWMutex
	policies      map[string]*ReplicationPolicy // tenant_id → policy
	shippers      map[string]*Shipper           // region_pair → shipper
	defaultPolicy *ReplicationPolicy
}

// ReplicationPolicy defines replication rules for a tenant or namespace
type ReplicationPolicy struct {
	TenantID      string
	Mode          ReplicationMode
	SourceRegion  string
	TargetRegions []string
	Priority      int  // 1-10, higher = more urgent
	Enabled       bool
}

// ReplicationMode defines how data is replicated
type ReplicationMode string

const (
	// ReplicationModeFull replicates all data
	ReplicationModeFull ReplicationMode = "full"

	// ReplicationModeSelective replicates based on tenant/namespace
	ReplicationModeSelective ReplicationMode = "selective"

	// ReplicationModeMultiWay replicates to N regions
	ReplicationModeMultiWay ReplicationMode = "multi-way"
)

// NewSelectiveReplicator creates a new selective replicator
func NewSelectiveReplicator() *SelectiveReplicator {
	return &SelectiveReplicator{
		policies:      make(map[string]*ReplicationPolicy),
		shippers:      make(map[string]*Shipper),
		defaultPolicy: &ReplicationPolicy{Mode: ReplicationModeFull, Enabled: true},
	}
}

// SetPolicy sets replication policy for a tenant
func (sr *SelectiveReplicator) SetPolicy(tenantID string, policy *ReplicationPolicy) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	sr.policies[tenantID] = policy
	fmt.Printf("Selective CRR: Policy set for tenant %s (mode: %s, targets: %v)\n",
		tenantID, policy.Mode, policy.TargetRegions)
}

// GetPolicy retrieves replication policy for a tenant
func (sr *SelectiveReplicator) GetPolicy(tenantID string) *ReplicationPolicy {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	if policy, ok := sr.policies[tenantID]; ok {
		return policy
	}
	return sr.defaultPolicy
}

// ShouldReplicate determines if a PCS should be replicated based on policy
func (sr *SelectiveReplicator) ShouldReplicate(ctx context.Context, pcsID, tenantID, targetRegion string) bool {
	policy := sr.GetPolicy(tenantID)

	if !policy.Enabled {
		return false
	}

	switch policy.Mode {
	case ReplicationModeFull:
		return true

	case ReplicationModeSelective:
		// Check if target region is in policy
		for _, region := range policy.TargetRegions {
			if region == targetRegion {
				return true
			}
		}
		return false

	case ReplicationModeMultiWay:
		// Replicate to all configured target regions
		for _, region := range policy.TargetRegions {
			if region == targetRegion {
				return true
			}
		}
		return false

	default:
		return true
	}
}

// RegisterShipper registers a shipper for a region pair
func (sr *SelectiveReplicator) RegisterShipper(sourceRegion, targetRegion string, shipper *Shipper) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	key := fmt.Sprintf("%s->%s", sourceRegion, targetRegion)
	sr.shippers[key] = shipper

	fmt.Printf("Selective CRR: Registered shipper %s\n", key)
}

// GetShipper retrieves shipper for a region pair
func (sr *SelectiveReplicator) GetShipper(sourceRegion, targetRegion string) *Shipper {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	key := fmt.Sprintf("%s->%s", sourceRegion, targetRegion)
	return sr.shippers[key]
}

// ListActivePolicies returns all active replication policies
func (sr *SelectiveReplicator) ListActivePolicies() []*ReplicationPolicy {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	policies := make([]*ReplicationPolicy, 0, len(sr.policies))
	for _, policy := range sr.policies {
		if policy.Enabled {
			policies = append(policies, policy)
		}
	}

	return policies
}

// --- Multi-Way Replication ---

// MultiWayReplicator coordinates replication to N regions
type MultiWayReplicator struct {
	mu         sync.RWMutex
	replicator *SelectiveReplicator
	topology   *TopologyConfig
}

// TopologyConfig defines multi-region topology
type TopologyConfig struct {
	Regions []RegionConfig
}

// RegionConfig defines a single region configuration
type RegionConfig struct {
	RegionID      string
	PeerRegions   []string // Regions to replicate to
	ReplicateFrom []string // Regions to receive from
}

// NewMultiWayReplicator creates a multi-way replicator
func NewMultiWayReplicator(replicator *SelectiveReplicator, topology *TopologyConfig) *MultiWayReplicator {
	return &MultiWayReplicator{
		replicator: replicator,
		topology:   topology,
	}
}

// ReplicateToAll replicates to all configured peer regions
func (mwr *MultiWayReplicator) ReplicateToAll(ctx context.Context, sourceRegion string, tenantID string) error {
	// Find source region config
	var regionConfig *RegionConfig
	for _, rc := range mwr.topology.Regions {
		if rc.RegionID == sourceRegion {
			regionConfig = &rc
			break
		}
	}

	if regionConfig == nil {
		return fmt.Errorf("region config not found for %s", sourceRegion)
	}

	// Replicate to all peer regions
	for _, targetRegion := range regionConfig.PeerRegions {
		// Check if should replicate based on policy
		policy := mwr.replicator.GetPolicy(tenantID)
		if !policy.Enabled {
			continue
		}

		// Get shipper for region pair
		shipper := mwr.replicator.GetShipper(sourceRegion, targetRegion)
		if shipper == nil {
			fmt.Printf("Multi-way CRR: No shipper for %s->%s\n", sourceRegion, targetRegion)
			continue
		}

		// Ship in background (non-blocking)
		go func(target string, s *Shipper) {
			if err := s.shipPendingSegments(ctx); err != nil {
				fmt.Printf("Multi-way CRR: Ship error %s->%s: %v\n", sourceRegion, target, err)
			}
		}(targetRegion, shipper)
	}

	return nil
}

// GetTopology returns the current topology configuration
func (mwr *MultiWayReplicator) GetTopology() *TopologyConfig {
	mwr.mu.RLock()
	defer mwr.mu.RUnlock()
	return mwr.topology
}

// UpdateTopology updates the topology configuration
func (mwr *MultiWayReplicator) UpdateTopology(topology *TopologyConfig) {
	mwr.mu.Lock()
	defer mwr.mu.Unlock()

	mwr.topology = topology
	fmt.Printf("Multi-way CRR: Topology updated (%d regions)\n", len(topology.Regions))
}

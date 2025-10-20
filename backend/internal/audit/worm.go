package audit

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/fractal-lba/kakeya/internal/api"
)

// WORMEntry represents a single immutable audit log entry
type WORMEntry struct {
	Timestamp       time.Time        `json:"timestamp"`
	PCSID           string           `json:"pcs_id"`
	TenantID        string           `json:"tenant_id,omitempty"`
	MerkleRoot      string           `json:"merkle_root"`
	Epoch           int              `json:"epoch"`
	ShardID         string           `json:"shard_id"`
	DHat            float64          `json:"D_hat"`
	CohStar         float64          `json:"coh_star"`
	R               float64          `json:"r"`
	Budget          float64          `json:"budget"`
	Regime          string           `json:"regime"`
	VerifyOutcome   string           `json:"verify_outcome"` // "accepted", "escalated", "rejected"
	VerifyParamsHash string          `json:"verify_params_hash"`
	PolicyVersion   string           `json:"policy_version,omitempty"`
	RegionID        string           `json:"region_id,omitempty"`
	EntryHash       string           `json:"entry_hash"` // SHA-256 of canonical entry
}

// WORMLog implements an append-only, tamper-evident audit log (Phase 3 WP3)
type WORMLog struct {
	mu           sync.Mutex
	baseDir      string
	currentFile  *os.File
	writer       *bufio.Writer
	segmentStart time.Time
	segmentSize  int64
	maxSegmentSize int64 // Rotate after this many bytes
	entries      int64
}

// NewWORMLog creates a new write-once-read-many audit log
func NewWORMLog(baseDir string) (*WORMLog, error) {
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create WORM log directory: %w", err)
	}

	w := &WORMLog{
		baseDir:        baseDir,
		maxSegmentSize: 100 * 1024 * 1024, // 100MB segments
	}

	// Open initial segment
	if err := w.rotateSegment(); err != nil {
		return nil, fmt.Errorf("failed to open initial segment: %w", err)
	}

	return w, nil
}

// Append adds an immutable entry to the WORM log
func (w *WORMLog) Append(pcs *api.PCS, result *api.VerifyResult, tenantID string, policyVersion string, regionID string) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Create entry
	entry := WORMEntry{
		Timestamp:  time.Now().UTC(),
		PCSID:      pcs.PCSID,
		TenantID:   tenantID,
		MerkleRoot: pcs.MerkleRoot,
		Epoch:      pcs.Epoch,
		ShardID:    pcs.ShardID,
		DHat:       pcs.DHat,
		CohStar:    pcs.CohStar,
		R:          pcs.R,
		Budget:     pcs.Budget,
		Regime:     pcs.Regime,
		PolicyVersion: policyVersion,
		RegionID:   regionID,
	}

	// Determine outcome
	if result.Accepted && !result.Escalated {
		entry.VerifyOutcome = "accepted"
	} else if result.Escalated {
		entry.VerifyOutcome = "escalated"
	} else {
		entry.VerifyOutcome = "rejected"
	}

	// Compute verify params hash (simplified - in production, would hash full params)
	entry.VerifyParamsHash = fmt.Sprintf("v1-%d", time.Now().Unix()/86400) // Daily versioning

	// Compute entry hash for tamper-evidence
	entryJSON, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("failed to marshal WORM entry: %w", err)
	}
	hash := sha256.Sum256(entryJSON)
	entry.EntryHash = hex.EncodeToString(hash[:])

	// Re-marshal with hash
	finalJSON, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("failed to marshal final WORM entry: %w", err)
	}

	// Append to current segment (newline-delimited JSON)
	if _, err := w.writer.Write(finalJSON); err != nil {
		return fmt.Errorf("failed to write WORM entry: %w", err)
	}
	if _, err := w.writer.WriteString("\n"); err != nil {
		return fmt.Errorf("failed to write newline: %w", err)
	}

	// Flush immediately for durability (WORM contract)
	if err := w.writer.Flush(); err != nil {
		return fmt.Errorf("failed to flush WORM log: %w", err)
	}
	if err := w.currentFile.Sync(); err != nil {
		return fmt.Errorf("failed to fsync WORM log: %w", err)
	}

	w.segmentSize += int64(len(finalJSON) + 1)
	w.entries++

	// Rotate segment if needed
	if w.segmentSize >= w.maxSegmentSize {
		if err := w.rotateSegment(); err != nil {
			return fmt.Errorf("failed to rotate WORM segment: %w", err)
		}
	}

	return nil
}

// rotateSegment creates a new time-based segment file
func (w *WORMLog) rotateSegment() error {
	// Close current file if open
	if w.writer != nil {
		if err := w.writer.Flush(); err != nil {
			return err
		}
	}
	if w.currentFile != nil {
		if err := w.currentFile.Close(); err != nil {
			return err
		}
	}

	// Create new segment with timestamp-based name
	now := time.Now().UTC()
	segmentDir := filepath.Join(w.baseDir, now.Format("2006/01/02"))
	if err := os.MkdirAll(segmentDir, 0755); err != nil {
		return fmt.Errorf("failed to create segment directory: %w", err)
	}

	segmentName := now.Format("150405.jsonl") // HHmmss.jsonl
	segmentPath := filepath.Join(segmentDir, segmentName)

	// Open in append mode with strict permissions (write-only for immutability)
	file, err := os.OpenFile(segmentPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0444)
	if err != nil {
		return fmt.Errorf("failed to create segment file: %w", err)
	}

	w.currentFile = file
	w.writer = bufio.NewWriter(file)
	w.segmentStart = now
	w.segmentSize = 0

	return nil
}

// Close closes the WORM log
func (w *WORMLog) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.writer != nil {
		if err := w.writer.Flush(); err != nil {
			return err
		}
	}
	if w.currentFile != nil {
		return w.currentFile.Close()
	}
	return nil
}

// Stats returns statistics about the WORM log
func (w *WORMLog) Stats() (entries int64, segmentSize int64, segmentStart time.Time) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.entries, w.segmentSize, w.segmentStart
}

// ComputeSegmentRoot computes Merkle root of a segment for anchoring (Phase 3 WP3)
func ComputeSegmentRoot(segmentPath string) (string, error) {
	file, err := os.Open(segmentPath)
	if err != nil {
		return "", fmt.Errorf("failed to open segment: %w", err)
	}
	defer file.Close()

	// Read all entry hashes
	var hashes [][]byte
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var entry WORMEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			return "", fmt.Errorf("failed to parse entry: %w", err)
		}
		hashBytes, err := hex.DecodeString(entry.EntryHash)
		if err != nil {
			return "", fmt.Errorf("failed to decode entry hash: %w", err)
		}
		hashes = append(hashes, hashBytes)
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("failed to scan segment: %w", err)
	}

	if len(hashes) == 0 {
		return "", fmt.Errorf("empty segment")
	}

	// Compute simple Merkle root (simplified - production would use full Merkle tree)
	root := hashes[0]
	for i := 1; i < len(hashes); i++ {
		combined := append(root, hashes[i]...)
		hash := sha256.Sum256(combined)
		root = hash[:]
	}

	return hex.EncodeToString(root), nil
}

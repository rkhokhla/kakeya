// Package fractal_lba provides a Go SDK for the Fractal LBA + Kakeya FT Stack API (Phase 4 WP5)
package fractal_lba

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"
)

// Client is the Fractal LBA API client (Phase 4 WP5)
type Client struct {
	baseURL    string
	tenantID   string
	signingKey string
	signingAlg string
	httpClient *http.Client
}

// PCS represents a Proof-of-Computation Summary
type PCS struct {
	PCSID      string                 `json:"pcs_id"`
	Schema     string                 `json:"schema"`
	Version    string                 `json:"version"`
	ShardID    string                 `json:"shard_id"`
	Epoch      int                    `json:"epoch"`
	Attempt    int                    `json:"attempt"`
	SentAt     string                 `json:"sent_at"`
	Seed       int64                  `json:"seed"`
	Scales     []int                  `json:"scales"`
	Nj         map[string]int         `json:"N_j"`
	CohStar    float64                `json:"coh_star"`
	VStar      []float64              `json:"v_star"`
	DHat       float64                `json:"D_hat"`
	R          float64                `json:"r"`
	Regime     string                 `json:"regime"`
	Budget     float64                `json:"budget"`
	MerkleRoot string                 `json:"merkle_root"`
	Sig        string                 `json:"sig"`
	FT         FaultToleranceInfo     `json:"ft"`
}

// FaultToleranceInfo contains metadata about delivery and system state
type FaultToleranceInfo struct {
	OutboxSeq   int64    `json:"outbox_seq"`
	Degraded    bool     `json:"degraded"`
	Fallbacks   []string `json:"fallbacks"`
	ClockSkewMs int      `json:"clock_skew_ms"`
}

// VerifyResult contains the outcome of PCS verification
type VerifyResult struct {
	Accepted        bool    `json:"accepted"`
	RecomputedDHat  float64 `json:"recomputed_D_hat,omitempty"`
	RecomputedBudget float64 `json:"recomputed_budget,omitempty"`
	Reason          string  `json:"reason,omitempty"`
	Escalated       bool    `json:"escalated"`
}

// NewClient creates a new Fractal LBA API client
func NewClient(baseURL, tenantID, signingKey, signingAlg string) *Client {
	return &Client{
		baseURL:    baseURL,
		tenantID:   tenantID,
		signingKey: signingKey,
		signingAlg: signingAlg,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// SubmitPCS submits a Proof-of-Computation Summary
func (c *Client) SubmitPCS(ctx context.Context, pcs *PCS) (*VerifyResult, error) {
	// Validate PCS
	if err := c.validatePCS(pcs); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	// Sign PCS if signing is enabled
	if c.signingAlg != "none" {
		if err := c.signPCS(pcs); err != nil {
			return nil, fmt.Errorf("signing failed: %w", err)
		}
	}

	// Marshal PCS to JSON
	body, err := json.Marshal(pcs)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal PCS: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/v1/pcs/submit", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "fractal-lba-go-sdk/0.4.0")
	if c.tenantID != "" {
		req.Header.Set("X-Tenant-Id", c.tenantID)
	}

	// Send request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Handle response status
	switch resp.StatusCode {
	case http.StatusOK, http.StatusAccepted:
		var result VerifyResult
		if err := json.Unmarshal(respBody, &result); err != nil {
			return nil, fmt.Errorf("failed to unmarshal response: %w", err)
		}
		return &result, nil
	case http.StatusUnauthorized:
		return nil, fmt.Errorf("signature verification failed (401)")
	case http.StatusTooManyRequests:
		return nil, fmt.Errorf("rate limit exceeded (429)")
	default:
		return nil, fmt.Errorf("API error (%d): %s", resp.StatusCode, string(respBody))
	}
}

// HealthCheck checks if the API is healthy
func (c *Client) HealthCheck(ctx context.Context) error {
	url := fmt.Sprintf("%s/health", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}

	return nil
}

// validatePCS performs basic validation
func (c *Client) validatePCS(pcs *PCS) error {
	if pcs.PCSID == "" {
		return fmt.Errorf("pcs_id is required")
	}
	if pcs.Schema != "fractal-lba-kakeya" {
		return fmt.Errorf("invalid schema: %s", pcs.Schema)
	}
	if pcs.CohStar < 0 || pcs.CohStar > 1.05 {
		return fmt.Errorf("coh_star out of bounds: %f", pcs.CohStar)
	}
	if pcs.R < 0 || pcs.R > 1 {
		return fmt.Errorf("r out of bounds: %f", pcs.R)
	}
	if pcs.Budget < 0 || pcs.Budget > 1 {
		return fmt.Errorf("budget out of bounds: %f", pcs.Budget)
	}
	return nil
}

// signPCS signs a PCS using the configured algorithm (Phase 1 canonicalization)
func (c *Client) signPCS(pcs *PCS) error {
	if c.signingAlg == "hmac" {
		if c.signingKey == "" {
			return fmt.Errorf("HMAC key not configured")
		}

		// Create signature subset (Phase 1 spec: 8 fields, 9-decimal rounding)
		subset := map[string]interface{}{
			"budget":      round9(pcs.Budget),
			"coh_star":    round9(pcs.CohStar),
			"D_hat":       round9(pcs.DHat),
			"epoch":       pcs.Epoch,
			"merkle_root": pcs.MerkleRoot,
			"pcs_id":      pcs.PCSID,
			"r":           round9(pcs.R),
			"shard_id":    pcs.ShardID,
		}

		// Canonical JSON (sorted keys, no spaces)
		canonicalJSON, err := json.Marshal(subset)
		if err != nil {
			return fmt.Errorf("failed to marshal signature subset: %w", err)
		}

		// SHA-256 digest
		digest := sha256.Sum256(canonicalJSON)

		// HMAC-SHA256
		mac := hmac.New(sha256.New, []byte(c.signingKey))
		mac.Write(digest[:])
		signature := mac.Sum(nil)

		// Base64 encode
		pcs.Sig = base64.StdEncoding.EncodeToString(signature)

		return nil
	}

	return fmt.Errorf("unsupported signing algorithm: %s", c.signingAlg)
}

// round9 rounds a float64 to 9 decimal places (Phase 1 spec)
func round9(x float64) float64 {
	return math.Round(x*1e9) / 1e9
}

package signing

import (
	"encoding/base64"
	"fmt"

	"github.com/fractal-lba/kakeya/internal/api"
)

// Verifier validates PCS signatures
type Verifier interface {
	Verify(pcs *api.PCS) error
}

// HMACVerifier verifies HMAC-SHA256 signatures
type HMACVerifier struct {
	key []byte
}

// NewHMACVerifier creates an HMAC verifier with the given key
func NewHMACVerifier(key string) *HMACVerifier {
	return &HMACVerifier{key: []byte(key)}
}

func (h *HMACVerifier) Verify(pcs *api.PCS) error {
	if pcs.Sig == "" {
		return fmt.Errorf("signature is empty")
	}

	// Get canonical signature payload (WP2: sign payload directly, not digest)
	payload, err := SignaturePayload(pcs)
	if err != nil {
		return fmt.Errorf("failed to generate signature payload: %w", err)
	}

	// Verify HMAC directly on payload
	if err := VerifyHMAC(payload, pcs.Sig, h.key); err != nil {
		return fmt.Errorf("HMAC verification failed: %w", err)
	}

	return nil
}

// Ed25519Verifier verifies Ed25519 signatures
type Ed25519Verifier struct {
	pubKey []byte
}

// NewEd25519Verifier creates an Ed25519 verifier from base64-encoded public key
func NewEd25519Verifier(pubKeyB64 string) (*Ed25519Verifier, error) {
	pubKey, err := base64.StdEncoding.DecodeString(pubKeyB64)
	if err != nil {
		return nil, fmt.Errorf("failed to decode public key: %w", err)
	}

	if len(pubKey) != 32 { // ed25519.PublicKeySize
		return nil, fmt.Errorf("invalid Ed25519 public key size: expected 32, got %d", len(pubKey))
	}

	return &Ed25519Verifier{pubKey: pubKey}, nil
}

func (e *Ed25519Verifier) Verify(pcs *api.PCS) error {
	if pcs.Sig == "" {
		return fmt.Errorf("signature is empty")
	}

	// Get canonical signature digest
	digest, err := SignatureDigest(pcs)
	if err != nil {
		return fmt.Errorf("failed to generate signature digest: %w", err)
	}

	// Verify Ed25519
	if err := VerifyEd25519(digest, pcs.Sig, e.pubKey); err != nil {
		return fmt.Errorf("Ed25519 verification failed: %w", err)
	}

	return nil
}

// NoOpVerifier skips signature verification (for testing/disabled signing)
type NoOpVerifier struct{}

func (n *NoOpVerifier) Verify(pcs *api.PCS) error {
	return nil
}

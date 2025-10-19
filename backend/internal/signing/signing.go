package signing

import (
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/sha256"
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

	// Get canonical signing payload
	payload, err := pcs.SigningPayload()
	if err != nil {
		return fmt.Errorf("failed to generate signing payload: %w", err)
	}

	// Compute expected HMAC
	mac := hmac.New(sha256.New, h.key)
	mac.Write(payload)
	expectedMAC := mac.Sum(nil)

	// Decode provided signature
	providedMAC, err := base64.StdEncoding.DecodeString(pcs.Sig)
	if err != nil {
		return fmt.Errorf("failed to decode signature: %w", err)
	}

	// Compare
	if !hmac.Equal(expectedMAC, providedMAC) {
		return fmt.Errorf("HMAC signature mismatch")
	}

	return nil
}

// Ed25519Verifier verifies Ed25519 signatures
type Ed25519Verifier struct {
	pubKey ed25519.PublicKey
}

// NewEd25519Verifier creates an Ed25519 verifier from base64-encoded public key
func NewEd25519Verifier(pubKeyB64 string) (*Ed25519Verifier, error) {
	pubKey, err := base64.StdEncoding.DecodeString(pubKeyB64)
	if err != nil {
		return nil, fmt.Errorf("failed to decode public key: %w", err)
	}

	if len(pubKey) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("invalid Ed25519 public key size: expected %d, got %d", ed25519.PublicKeySize, len(pubKey))
	}

	return &Ed25519Verifier{pubKey: ed25519.PublicKey(pubKey)}, nil
}

func (e *Ed25519Verifier) Verify(pcs *api.PCS) error {
	if pcs.Sig == "" {
		return fmt.Errorf("signature is empty")
	}

	// Get canonical signing payload
	payload, err := pcs.SigningPayload()
	if err != nil {
		return fmt.Errorf("failed to generate signing payload: %w", err)
	}

	// Compute digest (Ed25519 signs the hash)
	digest := sha256.Sum256(payload)

	// Decode provided signature
	sig, err := base64.StdEncoding.DecodeString(pcs.Sig)
	if err != nil {
		return fmt.Errorf("failed to decode signature: %w", err)
	}

	// Verify
	if !ed25519.Verify(e.pubKey, digest[:], sig) {
		return fmt.Errorf("Ed25519 signature verification failed")
	}

	return nil
}

// NoOpVerifier skips signature verification (for testing/disabled signing)
type NoOpVerifier struct{}

func (n *NoOpVerifier) Verify(pcs *api.PCS) error {
	return nil
}

package signing

import (
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"errors"
)

var (
	// ErrBadHMAC indicates HMAC signature verification failed
	ErrBadHMAC = errors.New("HMAC signature verification failed")

	// ErrBadEd25519 indicates Ed25519 signature verification failed
	ErrBadEd25519 = errors.New("Ed25519 signature verification failed")

	// ErrInvalidSignature indicates signature is malformed
	ErrInvalidSignature = errors.New("invalid signature format")
)

// VerifyHMAC verifies an HMAC-SHA256 signature.
//
// Process (WP2 simplified):
// 1. Compute HMAC-SHA256 of the payload directly with the key
// 2. Decode base64 signature from PCS
// 3. Compare using constant-time comparison
//
// Pre-hashing is unnecessary since HMAC provides cryptographic security.
//
// Args:
//   - payload: Canonical JSON payload bytes
//   - sigB64: Base64-encoded HMAC signature from PCS
//   - key: HMAC secret key (bytes)
//
// Returns:
//   - nil if verification succeeds
//   - ErrBadHMAC if signature doesn't match
//   - ErrInvalidSignature if base64 decoding fails
func VerifyHMAC(payload []byte, sigB64 string, key []byte) error {
	// Compute expected HMAC directly on payload
	mac := hmac.New(sha256.New, key)
	mac.Write(payload)
	expected := mac.Sum(nil)

	// Decode provided signature
	got, err := base64.StdEncoding.DecodeString(sigB64)
	if err != nil {
		return ErrInvalidSignature
	}

	// Constant-time comparison
	if !hmac.Equal(expected, got) {
		return ErrBadHMAC
	}

	return nil
}

// VerifyEd25519 verifies an Ed25519 signature.
//
// Process:
// 1. Decode base64 signature from PCS
// 2. Use Ed25519 public key to verify signature over digest
//
// Args:
//   - digest: SHA-256 hash of canonical payload (32 bytes)
//   - sigB64: Base64-encoded Ed25519 signature from PCS
//   - pubKey: Ed25519 public key (32 bytes)
//
// Returns:
//   - nil if verification succeeds
//   - ErrBadEd25519 if signature doesn't match
//   - ErrInvalidSignature if base64 decoding fails or key size wrong
func VerifyEd25519(digest []byte, sigB64 string, pubKey []byte) error {
	// Validate public key size
	if len(pubKey) != ed25519.PublicKeySize {
		return ErrInvalidSignature
	}

	// Decode provided signature
	sig, err := base64.StdEncoding.DecodeString(sigB64)
	if err != nil {
		return ErrInvalidSignature
	}

	// Verify signature
	if !ed25519.Verify(ed25519.PublicKey(pubKey), digest, sig) {
		return ErrBadEd25519
	}

	return nil
}

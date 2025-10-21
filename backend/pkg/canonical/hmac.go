package canonical

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"errors"
)

var (
	// ErrInvalidSignature indicates signature verification failed
	ErrInvalidSignature = errors.New("invalid HMAC signature")
)

// SignHMAC signs the signature subset using HMAC-SHA256.
//
// Process (WP2 simplified):
//  1. Generate canonical signature payload
//  2. HMAC-SHA256 the payload directly with provided key
//  3. Return base64-encoded signature
//
// Pre-hashing is unnecessary since HMAC provides cryptographic security.
//
// Args:
//
//	subset: SignatureSubset with required fields
//	key: HMAC secret key (bytes)
//
// Returns:
//
//	Base64-encoded HMAC signature, or error if payload generation fails
//
// Example:
//
//	subset := &SignatureSubset{
//	    PCSID:      "test",
//	    DHat:       1.0,
//	    CohStar:    0.75,
//	    R:          0.5,
//	    Budget:     0.35,
//	    MerkleRoot: "abc",
//	    Epoch:      1,
//	    ShardID:    "s1",
//	}
//	signature, err := SignHMAC(subset, []byte("my-secret-key"))
func SignHMAC(subset *SignatureSubset, key []byte) (string, error) {
	payload, err := CanonicalJSONBytes(subset)
	if err != nil {
		return "", err
	}

	mac := hmac.New(sha256.New, key)
	mac.Write(payload)
	sig := mac.Sum(nil)

	return base64.StdEncoding.EncodeToString(sig), nil
}

// VerifyHMAC verifies an HMAC-SHA256 signature.
//
// Process (WP2 simplified):
//  1. Generate canonical signature payload
//  2. HMAC-SHA256 the payload directly with provided key
//  3. Decode base64 signature and compare using constant-time comparison
//
// Args:
//
//	subset: SignatureSubset with required fields
//	sigB64: Base64-encoded HMAC signature
//	key: HMAC secret key (bytes)
//
// Returns:
//
//	nil if verification succeeds, error otherwise
func VerifyHMAC(subset *SignatureSubset, sigB64 string, key []byte) error {
	payload, err := CanonicalJSONBytes(subset)
	if err != nil {
		return err
	}

	// Compute expected HMAC directly on payload
	mac := hmac.New(sha256.New, key)
	mac.Write(payload)
	expected := mac.Sum(nil)

	// Decode provided signature
	got, err := base64.StdEncoding.DecodeString(sigB64)
	if err != nil {
		return err
	}

	// Constant-time comparison
	if !hmac.Equal(expected, got) {
		return ErrInvalidSignature
	}

	return nil
}

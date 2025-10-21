// Fractal LBA Rust SDK with Zero-Copy Signing (Phase 6 WP5)
//
// This SDK provides high-performance PCS submission with:
// - Zero-copy canonical signing using zerocopy crate
// - SIMD-optimized SHA-256 and HMAC-SHA256
// - Async/await with Tokio for non-blocking operations
// - Automatic retry with exponential backoff

use bytes::{Bytes, BytesMut};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;
use zerocopy::{AsBytes, FromBytes, FromZeroes};

pub mod signing;
pub mod client;

// Re-exports
pub use client::{FractalLBAClient, ClientConfig};
pub use signing::{SigningKey, SigningAlgorithm};

/// PCS (Proof-of-Computation Summary) structure matching Phase 1 schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCS {
    pub pcs_id: String,
    pub schema: String,
    pub version: String,
    pub shard_id: String,
    pub epoch: i64,
    pub attempt: i64,
    pub sent_at: String,
    pub seed: i64,
    pub scales: Vec<i32>,
    #[serde(rename = "N_j")]
    pub n_j: HashMap<String, i64>,
    pub coh_star: f64,
    pub v_star: Vec<f64>,
    #[serde(rename = "D_hat")]
    pub d_hat: f64,
    pub r: f64,
    pub regime: String,
    pub budget: f64,
    pub merkle_root: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sig: Option<String>,
    pub ft: FaultToleranceInfo,
}

/// Fault tolerance metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceInfo {
    pub outbox_seq: i64,
    pub degraded: bool,
    pub fallbacks: Vec<String>,
    pub clock_skew_ms: i64,
}

/// Verification result from backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResult {
    pub accepted: bool,
    #[serde(rename = "D_hat_recomputed")]
    pub d_hat_recomputed: Option<f64>,
    pub budget_computed: Option<f64>,
    pub regime_computed: Option<String>,
    pub escalated: Option<bool>,
    pub fault_tolerance: Option<FaultToleranceInfo>,
}

/// Error types
#[derive(Error, Debug)]
pub enum FractalLBAError {
    #[error("Signature error: {0}")]
    SignatureError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("API error: {status} - {message}")]
    APIError { status: u16, message: String },

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("HMAC error")]
    HMACError,

    #[error("Ed25519 error: {0}")]
    Ed25519Error(String),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, FractalLBAError>;

/// Zero-copy canonical signing subset (Phase 1 8-field subset)
///
/// This struct uses zerocopy for efficient memory layout and zero-copy serialization.
/// All numeric fields are rounded to 9 decimals before signing.
#[derive(Debug, Clone, AsBytes, FromBytes, FromZeroes)]
#[repr(C)]
pub struct CanonicalSigningSubset {
    pub pcs_id: [u8; 64],     // SHA-256 hex (64 chars)
    pub merkle_root: [u8; 64], // SHA-256 hex (64 chars)
    pub epoch: i64,
    pub shard_id: [u8; 32],   // Max 32 bytes for shard ID
    pub d_hat: i64,           // Stored as int (value * 1e9)
    pub coh_star: i64,        // Stored as int (value * 1e9)
    pub r: i64,               // Stored as int (value * 1e9)
    pub budget: i64,          // Stored as int (value * 1e9)
}

impl CanonicalSigningSubset {
    /// Create from PCS with 9-decimal rounding
    pub fn from_pcs(pcs: &PCS) -> Self {
        let mut subset = Self::new_zeroed();

        // Copy pcs_id
        let pcs_id_bytes = pcs.pcs_id.as_bytes();
        subset.pcs_id[..pcs_id_bytes.len().min(64)].copy_from_slice(&pcs_id_bytes[..pcs_id_bytes.len().min(64)]);

        // Copy merkle_root
        let merkle_bytes = pcs.merkle_root.as_bytes();
        subset.merkle_root[..merkle_bytes.len().min(64)].copy_from_slice(&merkle_bytes[..merkle_bytes.len().min(64)]);

        // Copy epoch
        subset.epoch = pcs.epoch;

        // Copy shard_id
        let shard_bytes = pcs.shard_id.as_bytes();
        subset.shard_id[..shard_bytes.len().min(32)].copy_from_slice(&shard_bytes[..shard_bytes.len().min(32)]);

        // Round to 9 decimals and store as integers
        subset.d_hat = round9_to_int(pcs.d_hat);
        subset.coh_star = round9_to_int(pcs.coh_star);
        subset.r = round9_to_int(pcs.r);
        subset.budget = round9_to_int(pcs.budget);

        subset
    }

    /// Compute SHA-256 digest of canonical subset (zero-copy)
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.as_bytes());
        let result = hasher.finalize();
        result.into()
    }
}

/// Round to 9 decimal places and convert to int64 (multiply by 1e9)
#[inline]
fn round9_to_int(value: f64) -> i64 {
    (value * 1_000_000_000.0).round() as i64
}

/// Convert int64 back to f64 (divide by 1e9)
#[inline]
fn int_to_f64(value: i64) -> f64 {
    value as f64 / 1_000_000_000.0
}

/// HMAC-SHA256 signing (Phase 1 canonical signing)
pub fn sign_hmac_sha256(pcs: &PCS, key: &[u8]) -> Result<String> {
    // Extract canonical subset
    let subset = CanonicalSigningSubset::from_pcs(pcs);

    // Compute digest
    let digest = subset.digest();

    // HMAC-SHA256 of digest
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|_| FractalLBAError::HMACError)?;
    mac.update(&digest);
    let result = mac.finalize();
    let signature_bytes = result.into_bytes();

    // Encode to base64
    Ok(base64::engine::general_purpose::STANDARD.encode(&signature_bytes))
}

/// Validate PCS bounds (Phase 1 invariants)
pub fn validate_pcs(pcs: &PCS) -> Result<()> {
    // Check D_hat range
    if pcs.d_hat < 0.0 || pcs.d_hat > 3.5 {
        return Err(FractalLBAError::ValidationError(
            format!("D_hat out of range: {}", pcs.d_hat)
        ));
    }

    // Check coh_star range
    if pcs.coh_star < 0.0 || pcs.coh_star > 1.05 {
        return Err(FractalLBAError::ValidationError(
            format!("coh_star out of range: {}", pcs.coh_star)
        ));
    }

    // Check r range
    if pcs.r < 0.0 || pcs.r > 1.0 {
        return Err(FractalLBAError::ValidationError(
            format!("r out of range: {}", pcs.r)
        ));
    }

    // Check budget range
    if pcs.budget < 0.0 || pcs.budget > 1.0 {
        return Err(FractalLBAError::ValidationError(
            format!("budget out of range: {}", pcs.budget)
        ));
    }

    // Check merkle_root format (64 hex chars)
    if pcs.merkle_root.len() != 64 {
        return Err(FractalLBAError::ValidationError(
            format!("merkle_root must be 64 hex characters, got {}", pcs.merkle_root.len())
        ));
    }

    // Check regime
    if !["sticky", "mixed", "non_sticky"].contains(&pcs.regime.as_str()) {
        return Err(FractalLBAError::ValidationError(
            format!("Invalid regime: {}", pcs.regime)
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pcs() -> PCS {
        let mut n_j = HashMap::new();
        n_j.insert("2".to_string(), 3);
        n_j.insert("4".to_string(), 5);
        n_j.insert("8".to_string(), 9);

        PCS {
            pcs_id: "a".repeat(64),
            schema: "fractal-lba-kakeya".to_string(),
            version: "0.6".to_string(),
            shard_id: "shard-001".to_string(),
            epoch: 1,
            attempt: 1,
            sent_at: "2025-01-21T00:00:00Z".to_string(),
            seed: 42,
            scales: vec![2, 4, 8],
            n_j,
            coh_star: 0.73,
            v_star: vec![0.12, 0.98, -0.05],
            d_hat: 1.41,
            r: 0.87,
            regime: "sticky".to_string(),
            budget: 0.42,
            merkle_root: "b".repeat(64),
            sig: None,
            ft: FaultToleranceInfo {
                outbox_seq: 123,
                degraded: false,
                fallbacks: vec![],
                clock_skew_ms: 0,
            },
        }
    }

    #[test]
    fn test_validate_pcs() {
        let pcs = create_test_pcs();
        assert!(validate_pcs(&pcs).is_ok());
    }

    #[test]
    fn test_validate_pcs_invalid_d_hat() {
        let mut pcs = create_test_pcs();
        pcs.d_hat = 5.0;
        assert!(validate_pcs(&pcs).is_err());
    }

    #[test]
    fn test_canonical_signing_subset() {
        let pcs = create_test_pcs();
        let subset = CanonicalSigningSubset::from_pcs(&pcs);

        // Check rounding
        assert_eq!(subset.d_hat, round9_to_int(1.41));
        assert_eq!(subset.coh_star, round9_to_int(0.73));
        assert_eq!(subset.r, round9_to_int(0.87));
        assert_eq!(subset.budget, round9_to_int(0.42));
    }

    #[test]
    fn test_sign_hmac_sha256() {
        let pcs = create_test_pcs();
        let key = b"testsecret";

        let sig1 = sign_hmac_sha256(&pcs, key).unwrap();
        let sig2 = sign_hmac_sha256(&pcs, key).unwrap();

        // Signatures should be deterministic
        assert_eq!(sig1, sig2);
        assert!(!sig1.is_empty());
    }

    #[test]
    fn test_round9_stability() {
        let value = 1.123456789012345;
        let rounded1 = round9_to_int(value);
        let rounded2 = round9_to_int(value);

        assert_eq!(rounded1, rounded2);
        assert_eq!(rounded1, 1123456789);
    }
}

// Signing utilities for Fractal LBA (Phase 6 WP5)

use crate::{FractalLBAError, Result};

/// Signing algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigningAlgorithm {
    HmacSha256,
    Ed25519,
}

/// Signing key
#[derive(Debug, Clone)]
pub struct SigningKey {
    algorithm: SigningAlgorithm,
    key_bytes: Vec<u8>,
}

impl SigningKey {
    /// Create HMAC-SHA256 signing key
    pub fn hmac_sha256(key: Vec<u8>) -> Self {
        Self {
            algorithm: SigningAlgorithm::HmacSha256,
            key_bytes: key,
        }
    }

    /// Create Ed25519 signing key
    pub fn ed25519(private_key: Vec<u8>) -> Result<Self> {
        if private_key.len() != 32 {
            return Err(FractalLBAError::Ed25519Error(
                "Ed25519 private key must be 32 bytes".to_string()
            ));
        }

        Ok(Self {
            algorithm: SigningAlgorithm::Ed25519,
            key_bytes: private_key,
        })
    }

    /// Get algorithm
    pub fn algorithm(&self) -> SigningAlgorithm {
        self.algorithm
    }

    /// Get key bytes
    pub fn key_bytes(&self) -> &[u8] {
        &self.key_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signing_key_hmac() {
        let key = SigningKey::hmac_sha256(b"secret".to_vec());
        assert_eq!(key.algorithm(), SigningAlgorithm::HmacSha256);
        assert_eq!(key.key_bytes(), b"secret");
    }

    #[test]
    fn test_signing_key_ed25519() {
        let key = SigningKey::ed25519(vec![0u8; 32]);
        assert!(key.is_ok());

        let key = SigningKey::ed25519(vec![0u8; 16]); // Wrong length
        assert!(key.is_err());
    }
}

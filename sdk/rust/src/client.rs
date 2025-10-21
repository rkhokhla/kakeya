// Fractal LBA Async Client (Phase 6 WP5)

use crate::{sign_hmac_sha256, validate_pcs, FractalLBAError, Result, PCS, VerifyResult};
use reqwest::{Client, StatusCode};
use std::time::Duration;

/// Client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Backend API endpoint
    pub endpoint: String,

    /// Tenant ID for multi-tenant deployments (Phase 3)
    pub tenant_id: Option<String>,

    /// Signing key (HMAC secret)
    pub signing_key: Vec<u8>,

    /// Request timeout (default: 30s)
    pub timeout: Duration,

    /// Max retries (default: 3)
    pub max_retries: u32,

    /// Initial backoff (default: 1s)
    pub initial_backoff: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8080/v1/pcs/submit".to_string(),
            tenant_id: None,
            signing_key: vec![],
            timeout: Duration::from_secs(30),
            max_retries: 3,
            initial_backoff: Duration::from_secs(1),
        }
    }
}

/// Fractal LBA Async Client
pub struct FractalLBAClient {
    config: ClientConfig,
    http_client: Client,
}

impl FractalLBAClient {
    /// Create a new client
    pub fn new(config: ClientConfig) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(FractalLBAError::NetworkError)?;

        Ok(Self {
            config,
            http_client,
        })
    }

    /// Submit a PCS (async)
    pub async fn submit_pcs(&self, mut pcs: PCS) -> Result<VerifyResult> {
        // Validate PCS
        validate_pcs(&pcs)?;

        // Sign PCS
        let signature = sign_hmac_sha256(&pcs, &self.config.signing_key)?;
        pcs.sig = Some(signature);

        // Submit with retries
        let mut retries = 0;
        let mut backoff = self.config.initial_backoff;

        loop {
            match self.submit_with_retry(&pcs).await {
                Ok(result) => return Ok(result),
                Err(err) => {
                    retries += 1;
                    if retries > self.config.max_retries {
                        return Err(err);
                    }

                    // Exponential backoff with jitter
                    tokio::time::sleep(backoff).await;
                    backoff = backoff * 2 + Duration::from_millis(rand::random::<u64>() % 1000);
                }
            }
        }
    }

    /// Submit PCS (single attempt)
    async fn submit_with_retry(&self, pcs: &PCS) -> Result<VerifyResult> {
        let mut req = self.http_client
            .post(&self.config.endpoint)
            .json(pcs);

        // Add tenant header if configured
        if let Some(tenant_id) = &self.config.tenant_id {
            req = req.header("X-Tenant-Id", tenant_id);
        }

        let resp = req.send().await.map_err(FractalLBAError::NetworkError)?;

        let status = resp.status();
        let body_text = resp.text().await.map_err(FractalLBAError::NetworkError)?;

        match status {
            StatusCode::OK | StatusCode::ACCEPTED => {
                // Parse response
                serde_json::from_str(&body_text).map_err(FractalLBAError::SerializationError)
            }
            StatusCode::UNAUTHORIZED => {
                Err(FractalLBAError::SignatureError("Signature verification failed (401)".to_string()))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(FractalLBAError::APIError {
                    status: 429,
                    message: "Rate limit exceeded".to_string(),
                })
            }
            _ => {
                Err(FractalLBAError::APIError {
                    status: status.as_u16(),
                    message: body_text,
                })
            }
        }
    }

    /// Health check (async)
    pub async fn health_check(&self) -> Result<bool> {
        let endpoint = self.config.endpoint.replace("/v1/pcs/submit", "/health");

        let resp = self.http_client
            .get(&endpoint)
            .send()
            .await
            .map_err(FractalLBAError::NetworkError)?;

        Ok(resp.status().is_success())
    }
}

// Use a simple random number generator for jitter (no need for full rand crate)
mod rand {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static SEED: Cell<u64> = Cell::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        );
    }

    pub fn random<T>() -> T
    where
        T: From<u64>,
    {
        SEED.with(|seed| {
            let mut s = seed.get();
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            seed.set(s);
            T::from(s)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::FaultToleranceInfo;

    fn create_test_pcs() -> PCS {
        let mut n_j = HashMap::new();
        n_j.insert("2".to_string(), 3);

        PCS {
            pcs_id: "a".repeat(64),
            schema: "fractal-lba-kakeya".to_string(),
            version: "0.6".to_string(),
            shard_id: "shard-001".to_string(),
            epoch: 1,
            attempt: 1,
            sent_at: "2025-01-21T00:00:00Z".to_string(),
            seed: 42,
            scales: vec![2],
            n_j,
            coh_star: 0.73,
            v_star: vec![0.12],
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

    #[tokio::test]
    async fn test_client_creation() {
        let config = ClientConfig::default();
        let client = FractalLBAClient::new(config);
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_submit_pcs_validation() {
        let config = ClientConfig {
            signing_key: b"testsecret".to_vec(),
            ..Default::default()
        };
        let client = FractalLBAClient::new(config).unwrap();

        let mut pcs = create_test_pcs();
        pcs.d_hat = 5.0; // Invalid

        let result = client.submit_pcs(pcs).await;
        assert!(result.is_err());
    }
}

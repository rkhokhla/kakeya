// Fractal LBA WASM SDK for Browser-Based Agents (Phase 6 WP5)
//
// This SDK compiles to WebAssembly and can run in browsers, allowing
// client-side agents to compute PCS and submit to the backend.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use hmac::{Hmac, Mac};
use std::collections::HashMap;

/// Console log helper
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// PCS (Proof-of-Computation Summary) structure
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCS {
    pcs_id: String,
    schema: String,
    version: String,
    shard_id: String,
    epoch: i64,
    attempt: i64,
    sent_at: String,
    seed: i64,
    #[wasm_bindgen(skip)]
    pub scales: Vec<i32>,
    #[wasm_bindgen(skip)]
    pub n_j: HashMap<String, i64>,
    coh_star: f64,
    #[wasm_bindgen(skip)]
    pub v_star: Vec<f64>,
    d_hat: f64,
    r: f64,
    regime: String,
    budget: f64,
    merkle_root: String,
    sig: Option<String>,
    #[wasm_bindgen(skip)]
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
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResult {
    accepted: bool,
    d_hat_recomputed: Option<f64>,
    budget_computed: Option<f64>,
    regime_computed: Option<String>,
    escalated: Option<bool>,
}

#[wasm_bindgen]
impl VerifyResult {
    #[wasm_bindgen(getter)]
    pub fn accepted(&self) -> bool {
        self.accepted
    }

    #[wasm_bindgen(getter)]
    pub fn d_hat_recomputed(&self) -> Option<f64> {
        self.d_hat_recomputed
    }

    #[wasm_bindgen(getter)]
    pub fn budget_computed(&self) -> Option<f64> {
        self.budget_computed
    }
}

/// Fractal LBA WASM Client
#[wasm_bindgen]
pub struct FractalLBAClient {
    endpoint: String,
    tenant_id: Option<String>,
    signing_key: Vec<u8>,
}

#[wasm_bindgen]
impl FractalLBAClient {
    /// Create a new client
    #[wasm_bindgen(constructor)]
    pub fn new(endpoint: String, tenant_id: Option<String>, signing_key: String) -> Self {
        console_log!("Fractal LBA WASM Client initialized");

        Self {
            endpoint,
            tenant_id,
            signing_key: signing_key.into_bytes(),
        }
    }

    /// Submit PCS (async)
    pub async fn submit_pcs(&self, pcs_json: JsValue) -> Result<JsValue, JsValue> {
        // Parse PCS from JS
        let mut pcs: PCS = serde_wasm_bindgen::from_value(pcs_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse PCS: {:?}", e)))?;

        // Validate PCS
        validate_pcs(&pcs).map_err(|e| JsValue::from_str(&e))?;

        // Sign PCS
        let signature = sign_hmac_sha256(&pcs, &self.signing_key)
            .map_err(|e| JsValue::from_str(&e))?;
        pcs.sig = Some(signature);

        // Submit via fetch API
        let result = self.submit_via_fetch(&pcs).await?;

        // Convert to JS
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }

    /// Submit via browser fetch API
    async fn submit_via_fetch(&self, pcs: &PCS) -> Result<VerifyResult, JsValue> {
        let window = web_sys::window().ok_or("no global window")?;

        // Serialize PCS
        let body = serde_json::to_string(pcs)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))?;

        // Create request
        let mut opts = RequestInit::new();
        opts.method("POST");
        opts.mode(RequestMode::Cors);
        opts.body(Some(&JsValue::from_str(&body)));

        let request = Request::new_with_str_and_init(&self.endpoint, &opts)?;

        // Set headers
        let headers = request.headers();
        headers.set("Content-Type", "application/json")?;
        if let Some(tenant_id) = &self.tenant_id {
            headers.set("X-Tenant-Id", tenant_id)?;
        }

        // Fetch
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        let resp: Response = resp_value.dyn_into()?;

        // Check status
        let status = resp.status();
        if status < 200 || status >= 300 {
            let text = JsFuture::from(resp.text()?).await?;
            return Err(JsValue::from_str(&format!("API error {}: {}", status, text.as_string().unwrap_or_default())));
        }

        // Parse response
        let json = JsFuture::from(resp.json()?).await?;
        let result: VerifyResult = serde_wasm_bindgen::from_value(json)
            .map_err(|e| JsValue::from_str(&format!("Parse error: {:?}", e)))?;

        Ok(result)
    }

    /// Health check
    pub async fn health_check(&self) -> Result<bool, JsValue> {
        let window = web_sys::window().ok_or("no global window")?;

        let endpoint = self.endpoint.replace("/v1/pcs/submit", "/health");

        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(&endpoint, &opts)?;

        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        let resp: Response = resp_value.dyn_into()?;

        Ok(resp.ok())
    }
}

/// Validate PCS bounds (Phase 1 invariants)
fn validate_pcs(pcs: &PCS) -> Result<(), String> {
    if pcs.d_hat < 0.0 || pcs.d_hat > 3.5 {
        return Err(format!("D_hat out of range: {}", pcs.d_hat));
    }

    if pcs.coh_star < 0.0 || pcs.coh_star > 1.05 {
        return Err(format!("coh_star out of range: {}", pcs.coh_star));
    }

    if pcs.r < 0.0 || pcs.r > 1.0 {
        return Err(format!("r out of range: {}", pcs.r));
    }

    if pcs.budget < 0.0 || pcs.budget > 1.0 {
        return Err(format!("budget out of range: {}", pcs.budget));
    }

    if pcs.merkle_root.len() != 64 {
        return Err(format!("merkle_root must be 64 hex characters, got {}", pcs.merkle_root.len()));
    }

    if !["sticky", "mixed", "non_sticky"].contains(&pcs.regime.as_str()) {
        return Err(format!("Invalid regime: {}", pcs.regime));
    }

    Ok(())
}

/// Round to 9 decimal places
fn round9(value: f64) -> f64 {
    (value * 1_000_000_000.0).round() / 1_000_000_000.0
}

/// HMAC-SHA256 signing (Phase 1 canonical signing)
fn sign_hmac_sha256(pcs: &PCS, key: &[u8]) -> Result<String, String> {
    // Create canonical subset (8 fields)
    let subset = serde_json::json!({
        "pcs_id": pcs.pcs_id,
        "merkle_root": pcs.merkle_root,
        "epoch": pcs.epoch,
        "shard_id": pcs.shard_id,
        "D_hat": round9(pcs.d_hat),
        "coh_star": round9(pcs.coh_star),
        "r": round9(pcs.r),
        "budget": round9(pcs.budget),
    });

    // Serialize with sorted keys (no spaces)
    let canonical_json = serde_json::to_string(&subset)
        .map_err(|e| format!("JSON error: {}", e))?;

    // SHA-256 digest
    let mut hasher = Sha256::new();
    hasher.update(canonical_json.as_bytes());
    let digest = hasher.finalize();

    // HMAC-SHA256 of digest
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|_| "HMAC key error".to_string())?;
    mac.update(&digest);
    let result = mac.finalize();
    let signature_bytes = result.into_bytes();

    // Encode to base64
    Ok(base64::engine::general_purpose::STANDARD.encode(&signature_bytes))
}

/// Utility: Compute SHA-256 hash
#[wasm_bindgen]
pub fn sha256(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}

/// Utility: Compute Merkle root
#[wasm_bindgen]
pub fn compute_merkle_root(leaves: Vec<JsValue>) -> Result<String, JsValue> {
    let leaf_hashes: Vec<String> = leaves.iter()
        .map(|v| v.as_string().ok_or("Invalid leaf hash"))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| JsValue::from_str(e))?;

    if leaf_hashes.is_empty() {
        return Err(JsValue::from_str("No leaves provided"));
    }

    let mut current_level = leaf_hashes;

    while current_level.len() > 1 {
        let mut next_level = Vec::new();

        for chunk in current_level.chunks(2) {
            let combined = if chunk.len() == 2 {
                format!("{}{}", chunk[0], chunk[1])
            } else {
                chunk[0].clone()
            };

            let mut hasher = Sha256::new();
            hasher.update(combined.as_bytes());
            let result = hasher.finalize();
            next_level.push(hex::encode(result));
        }

        current_level = next_level;
    }

    Ok(current_level[0].clone())
}

// Hex encoding helper
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_validate_pcs() {
        let pcs = create_test_pcs();
        assert!(validate_pcs(&pcs).is_ok());
    }

    #[wasm_bindgen_test]
    fn test_round9() {
        assert_eq!(round9(1.123456789012345), 1.123456789);
    }

    #[wasm_bindgen_test]
    fn test_sign_hmac_sha256() {
        let pcs = create_test_pcs();
        let sig = sign_hmac_sha256(&pcs, b"testsecret");
        assert!(sig.is_ok());
    }

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
}

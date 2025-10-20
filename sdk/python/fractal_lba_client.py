#!/usr/bin/env python3
"""
Fractal LBA + Kakeya FT Stack Python SDK (Phase 3 WP7)

Official Python client for submitting Proof-of-Computation Summaries to the
Fractal LBA backend with built-in signature support, retry logic, and
error handling.

Usage:
    from fractal_lba_client import FractalLBAClient

    client = FractalLBAClient(
        base_url="https://api.fractal-lba.example.com",
        tenant_id="tenant-001",
        signing_key="your-hmac-key",
        signing_alg="hmac"
    )

    response = client.submit_pcs(pcs_dict)
"""

import hashlib
import hmac
import json
import time
from typing import Dict, Any, Optional, List
from decimal import Decimal, ROUND_HALF_UP
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class FractalLBAError(Exception):
    """Base exception for Fractal LBA client errors"""
    pass


class SignatureError(FractalLBAError):
    """Raised when signature generation or verification fails"""
    pass


class ValidationError(FractalLBAError):
    """Raised when PCS validation fails"""
    pass


class APIError(FractalLBAError):
    """Raised when API returns an error"""
    def __init__(self, message: str, status_code: int, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class FractalLBAClient:
    """
    Python SDK client for Fractal LBA API (Phase 3 WP7)

    Features:
    - Automatic PCS signing (HMAC-SHA256)
    - Retry with exponential backoff + jitter
    - Multi-tenant support
    - Request validation
    - Response handling
    """

    def __init__(
        self,
        base_url: str,
        tenant_id: Optional[str] = None,
        signing_key: Optional[str] = None,
        signing_alg: str = "none",
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff_factor: float = 1.0
    ):
        """
        Initialize Fractal LBA client

        Args:
            base_url: Base URL of the API (e.g., https://api.fractal-lba.example.com)
            tenant_id: Tenant ID for multi-tenant mode (Phase 3)
            signing_key: HMAC signing key (if alg=hmac)
            signing_alg: Signature algorithm ("none", "hmac", "ed25519")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_backoff_factor: Backoff factor for retries
        """
        self.base_url = base_url.rstrip('/')
        self.tenant_id = tenant_id
        self.signing_key = signing_key
        self.signing_alg = signing_alg
        self.timeout = timeout

        # Create session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "fractal-lba-python-sdk/0.3.0"
        })

        if self.tenant_id:
            self.session.headers.update({"X-Tenant-Id": self.tenant_id})

    def submit_pcs(self, pcs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a Proof-of-Computation Summary

        Args:
            pcs: PCS dictionary with all required fields

        Returns:
            VerifyResult dict with acceptance status

        Raises:
            ValidationError: If PCS is invalid
            SignatureError: If signature generation fails
            APIError: If API returns error
        """
        # Validate PCS
        self._validate_pcs(pcs)

        # Sign PCS if signing is enabled
        if self.signing_alg != "none":
            pcs = self._sign_pcs(pcs)

        # Submit to API
        url = f"{self.base_url}/v1/pcs/submit"

        try:
            response = self.session.post(
                url,
                json=pcs,
                timeout=self.timeout
            )

            # Handle different response codes
            if response.status_code == 200:
                return response.json()  # Accepted
            elif response.status_code == 202:
                return response.json()  # Escalated
            elif response.status_code == 401:
                raise APIError(
                    "Signature verification failed",
                    status_code=401,
                    response=response.json() if response.text else None
                )
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "10")
                raise APIError(
                    f"Rate limit exceeded. Retry after {retry_after}s",
                    status_code=429,
                    response=response.json() if response.text else None
                )
            else:
                raise APIError(
                    f"API error: {response.status_code}",
                    status_code=response.status_code,
                    response=response.json() if response.text else None
                )

        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {e}", status_code=0)

    def _validate_pcs(self, pcs: Dict[str, Any]) -> None:
        """Validate PCS structure and required fields"""
        required_fields = [
            "pcs_id", "schema", "version", "shard_id", "epoch", "attempt",
            "sent_at", "seed", "scales", "N_j", "coh_star", "v_star",
            "D_hat", "r", "regime", "budget", "merkle_root", "ft"
        ]

        for field in required_fields:
            if field not in pcs:
                raise ValidationError(f"Missing required field: {field}")

        # Validate schema
        if pcs["schema"] != "fractal-lba-kakeya":
            raise ValidationError(f"Invalid schema: {pcs['schema']}")

        # Validate bounds
        if not (0 <= pcs["coh_star"] <= 1.05):
            raise ValidationError(f"coh_star out of bounds: {pcs['coh_star']}")

        if not (0 <= pcs["r"] <= 1):
            raise ValidationError(f"r out of bounds: {pcs['r']}")

        if not (0 <= pcs["budget"] <= 1):
            raise ValidationError(f"budget out of bounds: {pcs['budget']}")

        if pcs["regime"] not in ["sticky", "mixed", "non_sticky"]:
            raise ValidationError(f"Invalid regime: {pcs['regime']}")

    def _sign_pcs(self, pcs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign PCS using configured algorithm

        Returns copy of PCS with 'sig' field populated
        """
        pcs_copy = pcs.copy()

        if self.signing_alg == "hmac":
            if not self.signing_key:
                raise SignatureError("HMAC key not configured")

            # Generate signature using Phase 1 canonicalization
            signature = self._sign_hmac(pcs_copy)
            pcs_copy["sig"] = signature

        elif self.signing_alg == "ed25519":
            raise SignatureError("Ed25519 signing not yet implemented in SDK")

        else:
            raise SignatureError(f"Unknown signing algorithm: {self.signing_alg}")

        return pcs_copy

    def _sign_hmac(self, pcs: Dict[str, Any]) -> str:
        """
        Generate HMAC-SHA256 signature (Phase 1 compatibility)

        Uses 8-field signature subset with 9-decimal rounding
        """
        # Extract signature subset (Phase 1 spec)
        subset = {
            "budget": self._round9(pcs["budget"]),
            "coh_star": self._round9(pcs["coh_star"]),
            "D_hat": self._round9(pcs["D_hat"]),
            "epoch": pcs["epoch"],
            "merkle_root": pcs["merkle_root"],
            "pcs_id": pcs["pcs_id"],
            "r": self._round9(pcs["r"]),
            "shard_id": pcs["shard_id"]
        }

        # Canonical JSON (sorted keys, no spaces)
        canonical_json = json.dumps(subset, sort_keys=True, separators=(',', ':'))

        # SHA-256 digest
        digest = hashlib.sha256(canonical_json.encode('utf-8')).digest()

        # HMAC-SHA256
        sig_bytes = hmac.new(
            self.signing_key.encode('utf-8'),
            digest,
            hashlib.sha256
        ).digest()

        # Base64 encode
        import base64
        return base64.b64encode(sig_bytes).decode('utf-8')

    def _round9(self, x: float) -> float:
        """Round float to 9 decimal places (Phase 1 spec)"""
        return float(Decimal(str(x)).quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP))

    def health_check(self) -> bool:
        """
        Check if API is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            url = f"{self.base_url}/health"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def close(self):
        """Close session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage
if __name__ == "__main__":
    # Example PCS
    example_pcs = {
        "pcs_id": "a7b3c8d9e0f1234567890abcdef12345",
        "schema": "fractal-lba-kakeya",
        "version": "0.1",
        "shard_id": "shard-001",
        "epoch": 12345,
        "attempt": 1,
        "sent_at": "2025-01-15T10:30:00Z",
        "seed": 42,
        "scales": [2, 4, 8, 16, 32],
        "N_j": {"2": 3, "4": 5, "8": 9, "16": 17, "32": 31},
        "coh_star": 0.85,
        "v_star": [0.12, 0.98, -0.05],
        "D_hat": 1.35,
        "r": 0.65,
        "regime": "sticky",
        "budget": 0.55,
        "merkle_root": "a1b2c3d4e5f6789012345678901234567890abcdefabcdef1234567890abcd",
        "sig": "",  # Will be populated by client
        "ft": {
            "outbox_seq": 123,
            "degraded": False,
            "fallbacks": [],
            "clock_skew_ms": 0
        }
    }

    # Create client
    client = FractalLBAClient(
        base_url="http://localhost:8080",
        tenant_id="tenant-001",
        signing_key="testsecret",
        signing_alg="hmac"
    )

    # Check health
    if client.health_check():
        print("✓ API is healthy")
    else:
        print("✗ API is not healthy")
        exit(1)

    # Submit PCS
    try:
        result = client.submit_pcs(example_pcs)
        print(f"✓ PCS submitted successfully")
        print(f"  Accepted: {result['accepted']}")
        print(f"  Escalated: {result['escalated']}")
        if 'recomputed_D_hat' in result:
            print(f"  Recomputed D̂: {result['recomputed_D_hat']:.4f}")
    except Exception as e:
        print(f"✗ Failed to submit PCS: {e}")
        exit(1)

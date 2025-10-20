"""
Unit tests for signature verification (CLAUDE_PHASE1.md T4).

Tests:
- Canonicalization: subset extraction, 9-decimal rounding, sorted keys JSON
- HMAC signing: canonical digest → HMAC-SHA256 → base64
- Signature verification: valid signature passes, tampered data fails
"""

import pytest
import base64
import hashlib
import hmac
from agent.src.utils import canonical_json, signing


class TestCanonicalization:
    """Test canonical JSON and signature subset extraction."""

    def test_signature_subset_extracts_8_fields(self):
        """Signature subset must contain exactly 8 fields per CLAUDE.md."""
        pcs = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345678901234,
            "coh_star": 0.734567890123456,
            "r": 0.871234567890123,
            "budget": 0.421234567890123,
            "version": "0.1",  # Not in signature subset
            "scales": [2, 4, 8],  # Not in signature subset
        }

        subset = canonical_json.signature_subset(pcs)

        # Exactly 8 fields
        assert len(subset) == 8
        assert "pcs_id" in subset
        assert "merkle_root" in subset
        assert "epoch" in subset
        assert "shard_id" in subset
        assert "D_hat" in subset
        assert "coh_star" in subset
        assert "r" in subset
        assert "budget" in subset

        # Excluded fields
        assert "version" not in subset
        assert "scales" not in subset

    def test_round9_applied_to_floats(self):
        """Floats in signature subset must be rounded to 9 decimals."""
        pcs = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345678901234,  # More than 9 decimals
            "coh_star": 0.734567890123456,
            "r": 0.871234567890123,
            "budget": 0.421234567890123,
        }

        subset = canonical_json.signature_subset(pcs)

        # Check rounding
        assert subset["D_hat"] == 1.412345679
        assert subset["coh_star"] == 0.73456789
        assert subset["r"] == 0.871234568
        assert subset["budget"] == 0.421234568

    def test_canonical_json_sorted_keys_no_spaces(self):
        """Canonical JSON must have sorted keys and no spaces."""
        obj = {"z": 3, "a": 1, "m": 2}

        canonical = canonical_json.dumps_canonical(obj)

        # Should be sorted and compact
        assert canonical == b'{"a":1,"m":2,"z":3}'

    def test_signature_payload_stable(self):
        """Same PCS should produce identical signature payload."""
        pcs = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345678901234,
            "coh_star": 0.734567890123456,
            "r": 0.871234567890123,
            "budget": 0.421234567890123,
        }

        payload1 = signing.signature_payload(pcs)
        payload2 = signing.signature_payload(pcs)

        assert payload1 == payload2


class TestHMACSigningVerification:
    """Test HMAC-SHA256 signing and verification."""

    def test_sign_hmac_produces_base64(self):
        """sign_hmac should return base64-encoded signature."""
        pcs = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345679,
            "coh_star": 0.73456789,
            "r": 0.871234568,
            "budget": 0.421234568,
        }
        key = b"testsecret"

        sig = signing.sign_hmac(pcs, key)

        # Should be base64 string
        assert isinstance(sig, str)
        # Should be decodable
        decoded = base64.b64decode(sig)
        # HMAC-SHA256 produces 32 bytes
        assert len(decoded) == 32

    def test_verify_hmac_valid_signature(self):
        """Verify valid signature passes."""
        pcs = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345679,
            "coh_star": 0.73456789,
            "r": 0.871234568,
            "budget": 0.421234568,
        }
        key = b"testsecret"

        # Sign
        sig = signing.sign_hmac(pcs, key)
        pcs["sig"] = sig

        # Verify
        result = signing.verify_hmac(pcs, key)

        assert result is True

    def test_verify_hmac_tampered_field_fails(self):
        """Changing any field in signature subset must fail verification."""
        pcs = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345679,
            "coh_star": 0.73456789,
            "r": 0.871234568,
            "budget": 0.421234568,
        }
        key = b"testsecret"

        # Sign
        sig = signing.sign_hmac(pcs, key)
        pcs["sig"] = sig

        # Tamper with D_hat (change one digit)
        pcs["D_hat"] = 1.412345689  # Changed last digit from 9 to 9 -> 8 to 9

        # Verify must fail
        result = signing.verify_hmac(pcs, key)

        assert result is False

    def test_verify_hmac_tampered_merkle_root_fails(self):
        """Changing merkle_root must fail verification."""
        pcs = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345679,
            "coh_star": 0.73456789,
            "r": 0.871234568,
            "budget": 0.421234568,
        }
        key = b"testsecret"

        # Sign
        sig = signing.sign_hmac(pcs, key)
        pcs["sig"] = sig

        # Tamper with merkle_root
        pcs["merkle_root"] = "abc124"  # Changed last digit

        # Verify must fail
        result = signing.verify_hmac(pcs, key)

        assert result is False

    def test_verify_hmac_wrong_key_fails(self):
        """Using wrong key must fail verification."""
        pcs = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345679,
            "coh_star": 0.73456789,
            "r": 0.871234568,
            "budget": 0.421234568,
        }
        key = b"testsecret"
        wrong_key = b"wrongsecret"

        # Sign with correct key
        sig = signing.sign_hmac(pcs, key)
        pcs["sig"] = sig

        # Verify with wrong key
        result = signing.verify_hmac(pcs, wrong_key)

        assert result is False

    def test_verify_hmac_invalid_base64_fails(self):
        """Invalid base64 signature must fail verification."""
        pcs = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345679,
            "coh_star": 0.73456789,
            "r": 0.871234568,
            "budget": 0.421234568,
            "sig": "not-valid-base64!!!"
        }
        key = b"testsecret"

        # Verify must fail gracefully
        result = signing.verify_hmac(pcs, key)

        assert result is False


class TestGoldenPCSVerification:
    """Test signature verification against golden PCS files."""

    def test_golden_tiny_case_1_signature_verifies(self):
        """Golden PCS tiny_case_1 signature should verify with known key."""
        import json
        from pathlib import Path

        golden_path = Path(__file__).parent / "golden" / "pcs_tiny_case_1.json"

        if not golden_path.exists():
            pytest.skip("Golden file not found")

        with open(golden_path) as f:
            pcs = json.load(f)

        key = b"testsecret"  # Known key used in build_pcs.py

        # Verify
        result = signing.verify_hmac(pcs, key)

        assert result is True, f"Golden PCS signature verification failed for {golden_path}"

    def test_golden_tiny_case_2_signature_verifies(self):
        """Golden PCS tiny_case_2 signature should verify with known key."""
        import json
        from pathlib import Path

        golden_path = Path(__file__).parent / "golden" / "pcs_tiny_case_2.json"

        if not golden_path.exists():
            pytest.skip("Golden file not found")

        with open(golden_path) as f:
            pcs = json.load(f)

        key = b"testsecret"  # Known key used in build_pcs.py

        # Verify
        result = signing.verify_hmac(pcs, key)

        assert result is True, f"Golden PCS signature verification failed for {golden_path}"


class TestSignatureStability:
    """Test that signature remains stable across different executions."""

    def test_repeated_signing_identical(self):
        """Signing same PCS multiple times should yield identical signature."""
        pcs = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345679,
            "coh_star": 0.73456789,
            "r": 0.871234568,
            "budget": 0.421234568,
        }
        key = b"testsecret"

        sig1 = signing.sign_hmac(pcs, key)
        sig2 = signing.sign_hmac(pcs, key)
        sig3 = signing.sign_hmac(pcs, key)

        assert sig1 == sig2
        assert sig2 == sig3

    def test_non_signature_fields_dont_affect_signature(self):
        """Changing fields outside signature subset should not affect signature."""
        pcs1 = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345679,
            "coh_star": 0.73456789,
            "r": 0.871234568,
            "budget": 0.421234568,
            "version": "0.1",  # Not in signature subset
            "scales": [2, 4, 8],
        }

        pcs2 = {
            "pcs_id": "test-id",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.412345679,
            "coh_star": 0.73456789,
            "r": 0.871234568,
            "budget": 0.421234568,
            "version": "0.2",  # Changed
            "scales": [2, 4, 8, 16, 32],  # Changed
        }

        key = b"testsecret"

        sig1 = signing.sign_hmac(pcs1, key)
        sig2 = signing.sign_hmac(pcs2, key)

        # Signatures should be identical
        assert sig1 == sig2

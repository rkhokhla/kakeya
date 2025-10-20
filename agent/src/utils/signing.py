"""
PCS signing utilities for HMAC-SHA256 and Ed25519.

This module provides functions for creating cryptographic signatures over
PCS data according to CLAUDE.md specifications.
"""

import base64
import hmac
import hashlib
from typing import Dict, Any
from .canonical_json import signature_subset, dumps_canonical


def signature_payload(pcs: Dict[str, Any]) -> bytes:
    """
    Generate the canonical signature payload for a PCS.

    Per CLAUDE.md §2.1:
    1. Extract signature subset (pcs_id, merkle_root, epoch, shard_id, D_hat, coh_star, r, budget)
    2. Round numeric fields to 9 decimals
    3. Serialize to canonical JSON (sorted keys, no spaces)
    4. Return UTF-8 bytes

    Args:
        pcs: Complete PCS dictionary

    Returns:
        Canonical JSON bytes ready for hashing/signing
    """
    subset = signature_subset(pcs)
    return dumps_canonical(subset)


def sign_hmac(pcs: Dict[str, Any], key: bytes) -> str:
    """
    Sign a PCS using HMAC-SHA256.

    Process:
    1. Generate canonical signature payload
    2. Compute SHA-256 digest of payload
    3. HMAC-SHA256 the digest with provided key
    4. Return base64-encoded signature

    Args:
        pcs: Complete PCS dictionary
        key: HMAC secret key (bytes)

    Returns:
        Base64-encoded HMAC signature
    """
    payload = signature_payload(pcs)
    digest = hashlib.sha256(payload).digest()
    sig = hmac.new(key, digest, hashlib.sha256).digest()
    return base64.b64encode(sig).decode("utf-8")


def sign_ed25519(pcs: Dict[str, Any], private_key) -> str:
    """
    Sign a PCS using Ed25519.

    Process:
    1. Generate canonical signature payload
    2. Compute SHA-256 digest of payload
    3. Sign digest with Ed25519 private key
    4. Return base64-encoded signature

    Args:
        pcs: Complete PCS dictionary
        private_key: Ed25519 private key (from cryptography library)

    Returns:
        Base64-encoded Ed25519 signature
    """
    payload = signature_payload(pcs)
    digest = hashlib.sha256(payload).digest()
    signature = private_key.sign(digest)
    return base64.b64encode(signature).decode("utf-8")


def verify_hmac(pcs: Dict[str, Any], key: bytes) -> bool:
    """
    Verify HMAC-SHA256 signature on a PCS.

    Process:
    1. Extract signature from pcs["sig"]
    2. Generate canonical signature payload (excluding sig field)
    3. Compute SHA-256 digest of payload
    4. HMAC-SHA256 the digest with provided key
    5. Compare with extracted signature (constant-time)

    Args:
        pcs: Complete PCS dictionary with "sig" field
        key: HMAC secret key (bytes)

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # Extract signature
        sig_b64 = pcs.get("sig", "")
        if not sig_b64:
            return False

        # Decode signature
        expected_sig = base64.b64decode(sig_b64)

        # Compute signature over PCS (excluding sig field)
        pcs_without_sig = {k: v for k, v in pcs.items() if k != "sig"}
        payload = signature_payload(pcs_without_sig)
        digest = hashlib.sha256(payload).digest()
        computed_sig = hmac.new(key, digest, hashlib.sha256).digest()

        # Constant-time comparison
        return hmac.compare_digest(expected_sig, computed_sig)

    except Exception:
        # Any error in verification should return False
        return False

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

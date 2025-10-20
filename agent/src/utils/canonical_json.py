"""
Canonical JSON serialization for PCS signature stability.

This module provides utilities for creating stable, reproducible JSON
representations of PCS data for cryptographic signing.
"""

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any


# Exact keys included in signature payload (CLAUDE.md contract)
SIGN_KEYS = ("pcs_id", "merkle_root", "epoch", "shard_id", "D_hat", "coh_star", "r", "budget")


def round9(x: float) -> float:
    """
    Round a float to 9 decimal places for signature stability.

    Uses Decimal for precise rounding to avoid floating-point drift.

    Args:
        x: Float value to round

    Returns:
        Float rounded to 9 decimal places
    """
    return float(Decimal(str(x)).quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP))


def signature_subset(pcs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalize the signature subset from a PCS.

    Per CLAUDE.md §2.1, only specific fields are signed, and numeric
    fields must be rounded to 9 decimals for stability.

    Args:
        pcs: Complete PCS dictionary

    Returns:
        Dictionary with only signature fields, numeric values rounded
    """
    subset = {k: pcs[k] for k in SIGN_KEYS}

    # Round numeric fields to 9 decimal places
    for k in ("D_hat", "coh_star", "r", "budget"):
        subset[k] = round9(subset[k])

    return subset


def dumps_canonical(obj: Dict[str, Any]) -> bytes:
    """
    Serialize a dictionary to canonical JSON bytes.

    Uses sorted keys and no whitespace for stability across implementations.

    Args:
        obj: Dictionary to serialize

    Returns:
        UTF-8 encoded JSON bytes with sorted keys and no spaces
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

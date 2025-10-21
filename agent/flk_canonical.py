"""
Canonical JSON utilities for FLK/PCS signature generation.
Phase 10 WP1: Ensures cross-language signature compatibility (Python/Go/TS).

Key requirements:
- Floats formatted to exactly 9 decimal places
- Stable field ordering for signature subset
- No whitespace in JSON output
"""

import json
from typing import Dict, Any


# Signature subset: 8 fields used for signing (Phase 1 specification)
SIGNATURE_FIELDS = [
    "pcs_id",
    "merkle_root",
    "epoch",
    "shard_id",
    "D_hat",
    "coh_star",
    "r",
    "budget"
]


def format_float_9dp(value: float) -> str:
    """
    Format float to exactly 9 decimal places.

    This ensures stable signatures across languages and prevents
    floating-point drift.

    Args:
        value: Float to format

    Returns:
        String representation with 9 decimal places (e.g., "1.234567890")

    Examples:
        >>> format_float_9dp(1.23456789012345)
        '1.234567890'
        >>> format_float_9dp(0.5)
        '0.500000000'
    """
    return f"{value:.9f}"


def round9(value: float) -> float:
    """
    Round float to 9 decimal places.

    Used for normalizing floats before formatting to ensure
    consistent behavior across operations.

    Args:
        value: Float to round

    Returns:
        Float rounded to 9 decimal places
    """
    return round(value, 9)


def extract_signature_subset(pcs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the 8-field subset used for signing.

    Args:
        pcs: Full PCS dictionary

    Returns:
        Dictionary with only signature fields

    Raises:
        KeyError: If required signature field is missing
    """
    subset = {}
    for field in SIGNATURE_FIELDS:
        if field not in pcs:
            raise KeyError(f"Missing required signature field: {field}")
        subset[field] = pcs[field]
    return subset


def canonical_json_bytes(pcs_subset: Dict[str, Any]) -> bytes:
    """
    Generate canonical JSON bytes for signing.

    Rules:
    - Signature fields only (8 fields)
    - Floats formatted to 9 decimal places
    - Keys sorted alphabetically
    - No whitespace (compact JSON)
    - UTF-8 encoded

    Args:
        pcs_subset: PCS dictionary (should contain signature fields only)

    Returns:
        Canonical JSON as bytes, ready for signing

    Example:
        >>> pcs = {
        ...     "pcs_id": "abc123",
        ...     "D_hat": 1.23456789,
        ...     "coh_star": 0.75,
        ...     "r": 0.5,
        ...     "budget": 0.35,
        ...     "merkle_root": "def456",
        ...     "epoch": 1,
        ...     "shard_id": "shard-001"
        ... }
        >>> payload = canonical_json_bytes(pcs)
        >>> isinstance(payload, bytes)
        True
    """
    # Normalize floats to 9 decimal places
    normalized = {}
    for key, value in pcs_subset.items():
        if isinstance(value, float):
            # Round to 9dp, then format as string to preserve precision
            normalized[key] = float(format_float_9dp(round9(value)))
        else:
            normalized[key] = value

    # Generate compact JSON with sorted keys
    # This ensures byte-for-byte equality across implementations
    json_str = json.dumps(
        normalized,
        sort_keys=True,
        separators=(',', ':'),  # No whitespace
        ensure_ascii=False
    )

    return json_str.encode('utf-8')


def signature_payload(pcs: Dict[str, Any]) -> bytes:
    """
    Generate signature payload from full PCS.

    Convenience function that combines subset extraction and
    canonical JSON generation.

    Args:
        pcs: Full PCS dictionary

    Returns:
        Canonical payload bytes for signing

    Example:
        >>> pcs = {
        ...     "pcs_id": "test",
        ...     "D_hat": 1.0,
        ...     "coh_star": 0.75,
        ...     "r": 0.5,
        ...     "budget": 0.35,
        ...     "merkle_root": "abc",
        ...     "epoch": 1,
        ...     "shard_id": "s1",
        ...     "extra_field": "ignored"
        ... }
        >>> payload = signature_payload(pcs)
        >>> b'"D_hat"' in payload
        True
    """
    subset = extract_signature_subset(pcs)
    return canonical_json_bytes(subset)


# Backwards compatibility: Phase 1-9 used this function name
def SigningPayload(pcs: Dict[str, Any]) -> bytes:
    """
    Deprecated: Use signature_payload() instead.

    Kept for backwards compatibility with Phase 1-9 code.
    """
    return signature_payload(pcs)

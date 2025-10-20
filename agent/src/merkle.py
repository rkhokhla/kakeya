"""
Merkle tree implementation for PCS integrity.
"""

import hashlib
from typing import List, Tuple


def hash_leaf(data: bytes) -> str:
    """Hash a single leaf node."""
    return hashlib.sha256(data).hexdigest()


def hash_pair(left: str, right: str) -> str:
    """Hash a pair of nodes."""
    combined = (left + right).encode('utf-8')
    return hashlib.sha256(combined).hexdigest()


def build_merkle_tree(leaves: List[bytes]) -> str:
    """
    Build a Merkle tree from leaf data and return the root hash.

    Args:
        leaves: List of byte data for leaf nodes

    Returns:
        Hex-encoded root hash
    """
    if not leaves:
        # Empty tree has zero hash
        return "0" * 64

    # Hash all leaves
    current_level = [hash_leaf(leaf) for leaf in leaves]

    # Build tree bottom-up
    while len(current_level) > 1:
        next_level = []

        # Pair up nodes
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else left

            parent = hash_pair(left, right)
            next_level.append(parent)

        current_level = next_level

    return current_level[0]


def verify_merkle_proof(
    leaf: bytes,
    proof: List[Tuple[str, str]],
    root: str
) -> bool:
    """
    Verify a Merkle proof for a leaf.

    Args:
        leaf: The leaf data
        proof: List of (sibling_hash, position) where position is "left" or "right"
        root: Expected root hash

    Returns:
        True if proof is valid
    """
    current = hash_leaf(leaf)

    for sibling, position in proof:
        if position == "left":
            current = hash_pair(sibling, current)
        else:
            current = hash_pair(current, sibling)

    return current == root

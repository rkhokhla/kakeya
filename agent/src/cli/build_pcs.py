#!/usr/bin/env python3
"""
CLI tool to build PCS from CSV event data.

Usage:
    python -m agent.src.cli.build_pcs \
        --in tests/data/tiny_case_1.csv \
        --out tests/golden/pcs_tiny_case_1.json \
        --key testsecret
"""

import argparse
import json
import sys
import csv
import hashlib
from pathlib import Path

import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from agent.src import signals, merkle
    from agent.src.utils import sign_hmac
except ImportError:
    # Try relative imports
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src import signals, merkle
    from src.utils import sign_hmac


def load_csv(csv_path: str) -> tuple:
    """Load CSV file and return points array and raw data."""
    points = []
    rows = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract 3D points
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            points.append([x, y, z])

            # Build canonical row for compressibility
            t = float(row['timestamp'])
            rows.append(f"{t:.9f},{row['x']},{row['y']},{row['z']}")

    points_array = np.array(points)
    raw_data = "\n".join(rows).encode('utf-8')

    return points_array, raw_data


def compute_scales_and_nj(points: np.ndarray, scales: list) -> dict:
    """
    Compute N_j (unique non-empty voxel count) for each scale.

    Simple box-counting: partition space into voxels and count non-empty ones.
    """
    N_j = {}

    for scale in scales:
        # Quantize points to voxel grid
        voxel_coords = np.floor(points * scale).astype(int)

        # Count unique voxels
        unique_voxels = set(map(tuple, voxel_coords))
        N_j[scale] = len(unique_voxels)

    return N_j


def build_pcs(csv_path: str, shard_id: str, epoch: int, hmac_key: str, seed: int = 42) -> dict:
    """Build a complete PCS from CSV data."""

    # Load data
    points, raw_data = load_csv(csv_path)

    # Define scales
    scales = [2, 4, 8, 16]

    # Compute N_j for each scale
    N_j = compute_scales_and_nj(points, scales)

    # Compute signals
    D_hat = signals.compute_D_hat(scales, N_j)
    coh_star, v_star = signals.compute_coherence(points, num_directions=100, num_bins=20, seed=seed)
    r = signals.compute_compressibility(raw_data)

    # Classify regime and compute budget
    regime = signals.classify_regime(D_hat, coh_star)
    budget = signals.compute_budget(D_hat, coh_star, r)

    # Build merkle tree from data chunks
    chunk_size = 64
    chunks = [raw_data[i:i+chunk_size] for i in range(0, len(raw_data), chunk_size)]
    merkle_root = merkle.build_merkle_tree(chunks)

    # Compute PCS ID
    pcs_id_data = f"{merkle_root}|{epoch}|{shard_id}"
    pcs_id = hashlib.sha256(pcs_id_data.encode('utf-8')).hexdigest()

    # Build PCS
    pcs = {
        "pcs_id": pcs_id,
        "schema": "fractal-lba-kakeya",
        "version": "0.1",
        "shard_id": shard_id,
        "epoch": epoch,
        "attempt": 1,
        "sent_at": "2025-01-19T12:00:00Z",  # Fixed for reproducibility
        "seed": seed,
        "scales": scales,
        "N_j": {str(k): v for k, v in N_j.items()},
        "coh_star": coh_star,
        "v_star": v_star.tolist(),
        "D_hat": D_hat,
        "r": r,
        "regime": regime,
        "budget": budget,
        "merkle_root": merkle_root,
        "sig": "",  # Will be filled below
        "ft": {
            "outbox_seq": 0,
            "degraded": False,
            "fallbacks": [],
            "clock_skew_ms": 0
        }
    }

    # Sign PCS
    pcs["sig"] = sign_hmac(pcs, hmac_key.encode('utf-8'))

    return pcs


def main():
    parser = argparse.ArgumentParser(description='Build PCS from CSV')
    parser.add_argument('--in', dest='input', required=True, help='Input CSV file')
    parser.add_argument('--out', dest='output', required=True, help='Output JSON file')
    parser.add_argument('--key', required=True, help='HMAC key for signing')
    parser.add_argument('--shard-id', default='test-shard-001', help='Shard ID')
    parser.add_argument('--epoch', type=int, default=1, help='Epoch number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Build PCS
    pcs = build_pcs(args.input, args.shard_id, args.epoch, args.key, args.seed)

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(pcs, f, indent=2)

    print(f"✓ Generated PCS: {args.output}")
    print(f"  pcs_id: {pcs['pcs_id']}")
    print(f"  D_hat: {pcs['D_hat']}")
    print(f"  coh_star: {pcs['coh_star']}")
    print(f"  r: {pcs['r']}")
    print(f"  regime: {pcs['regime']}")
    print(f"  budget: {pcs['budget']}")


if __name__ == '__main__':
    main()

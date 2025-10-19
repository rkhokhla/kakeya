"""Fractal LBA + Kakeya FT Stack - Agent Package"""

from .agent import PCSAgent
from .signals import compute_D_hat, compute_coherence, compute_compressibility, classify_regime, compute_budget
from .merkle import build_merkle_tree
from .outbox import OutboxWAL
from .client import PCSClient, DLQ

__version__ = "0.1.0"

__all__ = [
    "PCSAgent",
    "compute_D_hat",
    "compute_coherence",
    "compute_compressibility",
    "classify_regime",
    "compute_budget",
    "build_merkle_tree",
    "OutboxWAL",
    "PCSClient",
    "DLQ",
]

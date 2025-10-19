"""
Main agent module for PCS generation and submission.
"""

import hashlib
import hmac
import base64
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np

from . import signals
from . import merkle
from . import outbox
from . import client


class PCSAgent:
    """Agent for computing and submitting Proof-of-Computation Summaries."""

    def __init__(
        self,
        shard_id: str,
        endpoint: str,
        sign_alg: str = "hmac",
        hmac_key: Optional[str] = None,
        ed25519_priv_b64: Optional[str] = None,
        outbox_path: str = "data/outbox.wal",
        dlq_path: str = "data/dlq.jsonl",
        verify_tls: bool = True
    ):
        """
        Initialize PCS agent.

        Args:
            shard_id: Unique shard identifier
            endpoint: Backend submission URL
            sign_alg: Signature algorithm ("hmac", "ed25519", or "none")
            hmac_key: HMAC secret key (required if sign_alg="hmac")
            ed25519_priv_b64: Ed25519 private key base64 (required if sign_alg="ed25519")
            outbox_path: Path to outbox WAL
            dlq_path: Path to dead letter queue
            verify_tls: Whether to verify TLS certificates
        """
        self.shard_id = shard_id
        self.sign_alg = sign_alg
        self.hmac_key = hmac_key
        self.ed25519_priv = None

        if sign_alg == "ed25519" and ed25519_priv_b64:
            from cryptography.hazmat.primitives.asymmetric import ed25519
            priv_bytes = base64.b64decode(ed25519_priv_b64)
            self.ed25519_priv = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)

        self.outbox_wal = outbox.OutboxWAL(outbox_path)
        self.client = client.PCSClient(endpoint, verify_tls=verify_tls)
        self.dlq = client.DLQ(dlq_path)

    def compute_pcs(
        self,
        epoch: int,
        scales: List[int],
        N_j: Dict[int, int],
        points: np.ndarray,
        raw_data: bytes,
        seed: int,
        attempt: int = 1
    ) -> Dict[str, Any]:
        """
        Compute a complete PCS from input data.

        Args:
            epoch: Time epoch
            scales: List of spatial scales
            N_j: Unique nonempty cells per scale
            points: Nx3 array of 3D points for coherence
            raw_data: Raw event data for compressibility
            seed: Random seed
            attempt: Attempt number

        Returns:
            Complete PCS dictionary
        """
        # Compute signals
        D_hat = signals.compute_D_hat(scales, N_j)
        coh_star, v_star = signals.compute_coherence(points)
        r = signals.compute_compressibility(raw_data)
        regime = signals.classify_regime(D_hat, coh_star)
        budget = signals.compute_budget(D_hat, coh_star, r)

        # Build merkle root (from raw data chunks)
        chunk_size = 1024
        chunks = [raw_data[i:i+chunk_size] for i in range(0, len(raw_data), chunk_size)]
        merkle_root = merkle.build_merkle_tree(chunks)

        # Compute PCS ID
        pcs_id = self._compute_pcs_id(merkle_root, epoch, self.shard_id)

        # Build PCS
        pcs = {
            "pcs_id": pcs_id,
            "schema": "fractal-lba-kakeya",
            "version": "0.1",
            "shard_id": self.shard_id,
            "epoch": epoch,
            "attempt": attempt,
            "sent_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "seed": seed,
            "scales": scales,
            "N_j": {str(k): v for k, v in N_j.items()},  # Keys must be strings
            "coh_star": coh_star,
            "v_star": v_star.tolist() if isinstance(v_star, np.ndarray) else list(v_star),
            "D_hat": D_hat,
            "r": r,
            "regime": regime,
            "budget": budget,
            "merkle_root": merkle_root,
            "sig": "",
            "ft": {
                "outbox_seq": 0,  # Will be set on append
                "degraded": False,
                "fallbacks": [],
                "clock_skew_ms": 0
            }
        }

        # Sign
        pcs["sig"] = self._sign_pcs(pcs)

        return pcs

    def submit_pcs(self, pcs: Dict[str, Any]) -> bool:
        """
        Submit PCS via outbox WAL with retry logic.

        Args:
            pcs: Complete PCS dictionary

        Returns:
            True if successfully submitted and acked
        """
        # Append to outbox WAL first (durability)
        entry = self.outbox_wal.append(pcs["pcs_id"], pcs)
        pcs["ft"]["outbox_seq"] = entry.seq

        # Attempt submission
        response = self.client.submit(pcs)

        if response is not None:
            # Success - mark as acked
            self.outbox_wal.mark_acked(entry.seq)
            return True
        else:
            # Failed - add to DLQ
            self.dlq.add(pcs, "All retries exhausted")
            return False

    def retry_pending(self):
        """Retry all pending (unacked) entries from outbox."""
        pending = self.outbox_wal.pending()

        for entry in pending:
            pcs = entry.payload
            response = self.client.submit(pcs)

            if response is not None:
                self.outbox_wal.mark_acked(entry.seq)
            else:
                self.dlq.add(pcs, "Retry failed")

    def _compute_pcs_id(self, merkle_root: str, epoch: int, shard_id: str) -> str:
        """Compute pcs_id = sha256(merkle_root|epoch|shard_id)."""
        data = f"{merkle_root}|{epoch}|{shard_id}"
        digest = hashlib.sha256(data.encode('utf-8')).digest()
        return digest.hex()

    def _sign_pcs(self, pcs: Dict[str, Any]) -> str:
        """Sign PCS using configured algorithm."""
        if self.sign_alg == "none":
            return ""

        # Build canonical signing payload
        payload = {
            "pcs_id": pcs["pcs_id"],
            "merkle_root": pcs["merkle_root"],
            "epoch": pcs["epoch"],
            "shard_id": pcs["shard_id"],
            "D_hat": signals.round_9(pcs["D_hat"]),
            "coh_star": signals.round_9(pcs["coh_star"]),
            "r": signals.round_9(pcs["r"]),
            "budget": signals.round_9(pcs["budget"])
        }

        # Serialize with sorted keys, no spaces
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))

        if self.sign_alg == "hmac":
            if not self.hmac_key:
                raise ValueError("HMAC key not configured")

            mac = hmac.new(
                self.hmac_key.encode('utf-8'),
                payload_json.encode('utf-8'),
                hashlib.sha256
            ).digest()

            return base64.b64encode(mac).decode('utf-8')

        elif self.sign_alg == "ed25519":
            if not self.ed25519_priv:
                raise ValueError("Ed25519 private key not configured")

            # Sign the SHA-256 digest
            digest = hashlib.sha256(payload_json.encode('utf-8')).digest()
            signature = self.ed25519_priv.sign(digest)

            return base64.b64encode(signature).decode('utf-8')

        else:
            raise ValueError(f"Unknown sign algorithm: {self.sign_alg}")

    def close(self):
        """Clean up resources."""
        self.client.close()

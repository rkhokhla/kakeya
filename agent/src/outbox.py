"""
Outbox WAL for agent-side at-least-once delivery.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import fcntl


@dataclass
class OutboxEntry:
    """Single outbox entry."""
    seq: int
    pcs_id: str
    payload: dict
    created_at: str
    acked: bool = False
    acked_at: Optional[str] = None


class OutboxWAL:
    """Write-ahead log for PCS submissions with fsync."""

    def __init__(self, wal_path: str):
        self.wal_path = wal_path
        self.next_seq = 1

        # Create directory if needed
        os.makedirs(os.path.dirname(wal_path) or '.', exist_ok=True)

        # Load existing entries to determine next sequence
        if os.path.exists(wal_path):
            entries = self.load_all()
            if entries:
                self.next_seq = max(e.seq for e in entries) + 1

    def append(self, pcs_id: str, payload: dict) -> OutboxEntry:
        """
        Append a new PCS to the outbox with fsync.

        Args:
            pcs_id: The PCS identifier
            payload: Complete PCS JSON

        Returns:
            Created OutboxEntry
        """
        entry = OutboxEntry(
            seq=self.next_seq,
            pcs_id=pcs_id,
            payload=payload,
            created_at=datetime.utcnow().isoformat() + 'Z',
            acked=False
        )

        self.next_seq += 1

        # Append to WAL with fsync
        with open(self.wal_path, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(asdict(entry)) + '\n')
                f.flush()
                os.fsync(f.fileno())  # Critical: ensure durability
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return entry

    def mark_acked(self, seq: int) -> bool:
        """
        Mark an entry as acknowledged.

        Note: This is implemented as append-only. A compaction job
        should periodically remove old acked entries.

        Args:
            seq: Sequence number to mark

        Returns:
            True if marked successfully
        """
        ack_entry = {
            'ack': seq,
            'acked_at': datetime.utcnow().isoformat() + 'Z'
        }

        with open(self.wal_path, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(ack_entry) + '\n')
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return True

    def load_all(self) -> List[OutboxEntry]:
        """
        Load all entries from WAL, applying acks.

        Returns:
            List of OutboxEntry objects
        """
        if not os.path.exists(self.wal_path):
            return []

        entries = {}
        acks = {}

        with open(self.wal_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    if 'ack' in data:
                        # This is an ack marker
                        seq = data['ack']
                        acks[seq] = data.get('acked_at')
                    else:
                        # This is an entry
                        entry = OutboxEntry(**data)
                        entries[entry.seq] = entry
                except (json.JSONDecodeError, TypeError):
                    continue

        # Apply acks
        for seq, acked_at in acks.items():
            if seq in entries:
                entries[seq].acked = True
                entries[seq].acked_at = acked_at

        return list(entries.values())

    def pending(self) -> List[OutboxEntry]:
        """
        Get all unacked entries.

        Returns:
            List of entries where acked=False
        """
        all_entries = self.load_all()
        return [e for e in all_entries if not e.acked]

    def compact(self, horizon_days: int = 14):
        """
        Remove acked entries older than horizon.

        Args:
            horizon_days: Keep entries within this many days
        """
        from datetime import timedelta

        all_entries = self.load_all()
        cutoff = datetime.utcnow() - timedelta(days=horizon_days)

        # Keep: unacked OR recent
        to_keep = []
        for e in all_entries:
            created = datetime.fromisoformat(e.created_at.rstrip('Z'))
            if not e.acked or created > cutoff:
                to_keep.append(e)

        # Rewrite WAL
        backup_path = self.wal_path + '.bak'
        os.rename(self.wal_path, backup_path)

        try:
            with open(self.wal_path, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    for e in to_keep:
                        f.write(json.dumps(asdict(e)) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            os.remove(backup_path)
        except Exception as e:
            # Restore backup on error
            if os.path.exists(backup_path):
                os.rename(backup_path, self.wal_path)
            raise

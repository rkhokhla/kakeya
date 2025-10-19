"""
HTTP client for PCS submission with retries and exponential backoff.
"""

import time
import random
import requests
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PCSClient:
    """Client for submitting PCS to the backend."""

    def __init__(
        self,
        endpoint: str,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: int = 30,
        verify_tls: bool = True
    ):
        """
        Initialize PCS client.

        Args:
            endpoint: Backend URL (e.g., https://api.example.com/v1/pcs/submit)
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            timeout: HTTP timeout (seconds)
            verify_tls: Whether to verify TLS certificates
        """
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.verify_tls = verify_tls
        self.session = requests.Session()

    def submit(self, pcs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Submit a PCS with exponential backoff and jitter.

        Args:
            pcs: PCS dictionary

        Returns:
            Response dict if successful, None if all retries exhausted
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.endpoint,
                    json=pcs,
                    timeout=self.timeout,
                    verify=self.verify_tls,
                    headers={'Content-Type': 'application/json'}
                )

                # Success cases
                if response.status_code in (200, 202):
                    logger.info(f"PCS {pcs['pcs_id']} submitted successfully (status={response.status_code})")
                    return response.json()

                # Auth failure - don't retry
                if response.status_code == 401:
                    logger.error(f"Signature verification failed for {pcs['pcs_id']}")
                    return None

                # Client error - don't retry
                if 400 <= response.status_code < 500:
                    logger.error(f"Client error {response.status_code} for {pcs['pcs_id']}: {response.text}")
                    return None

                # Server error or rate limit - retry
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After', self.base_delay)
                    try:
                        retry_after = float(retry_after)
                    except ValueError:
                        retry_after = self.base_delay

                    logger.warning(f"Rate limited for {pcs['pcs_id']}, retrying after {retry_after}s")
                    time.sleep(retry_after)
                    continue

                # Other server errors
                logger.warning(f"Server error {response.status_code} for {pcs['pcs_id']}, attempt {attempt+1}/{self.max_retries}")
                last_error = f"HTTP {response.status_code}"

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout submitting {pcs['pcs_id']}, attempt {attempt+1}/{self.max_retries}")
                last_error = "Timeout"

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error for {pcs['pcs_id']}: {e}, attempt {attempt+1}/{self.max_retries}")
                last_error = f"Connection error: {e}"

            except Exception as e:
                logger.error(f"Unexpected error for {pcs['pcs_id']}: {e}")
                last_error = str(e)

            # Exponential backoff with jitter
            if attempt < self.max_retries - 1:
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                jitter = delay * random.uniform(0.5, 1.5)
                logger.debug(f"Backing off for {jitter:.2f}s before retry")
                time.sleep(jitter)

        # All retries exhausted
        logger.error(f"Failed to submit {pcs['pcs_id']} after {self.max_retries} attempts: {last_error}")
        return None

    def close(self):
        """Close the HTTP session."""
        self.session.close()


class DLQ:
    """Dead Letter Queue for failed submissions."""

    def __init__(self, path: str):
        self.path = path

    def add(self, pcs: Dict[str, Any], error: str):
        """Add a failed PCS to the DLQ."""
        import json
        import os

        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)

        entry = {
            'pcs_id': pcs['pcs_id'],
            'error': error,
            'timestamp': time.time(),
            'pcs': pcs
        }

        with open(self.path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        logger.info(f"Added {pcs['pcs_id']} to DLQ")

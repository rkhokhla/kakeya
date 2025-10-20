"""
SDK Golden Vector Tests (Phase 5 WP5)

Validates that Python, Go, and TypeScript SDKs produce identical signatures
for the same PCS input. Ensures Phase 1 canonical signing is implemented
identically across all SDKs.

Test Approach:
1. Load reference PCS from golden vectors file (tests/golden/pcs_vectors.json)
2. Sign PCS with Python SDK
3. Compare signature with Go SDK output (via subprocess)
4. Compare signature with TypeScript SDK output (via subprocess)
5. Assert all three signatures are byte-identical

Requirements:
- Phase 1 canonical signing: 8-field subset, 9-decimal rounding, sorted keys
- HMAC-SHA256 with shared key
- Signature bytes must be identical (not just semantically equivalent)
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sdk" / "python"))

from fractal_lba_client import FractalLBAClient


class TestGoldenVectors(unittest.TestCase):
    """Golden vector tests for SDK signature compatibility"""

    @classmethod
    def setUpClass(cls):
        """Load golden vectors once for all tests"""
        golden_file = Path(__file__).parent.parent / "golden" / "pcs_vectors.json"
        if not golden_file.exists():
            raise FileNotFoundError(
                f"Golden vectors file not found: {golden_file}\n"
                "Run 'python agent/src/cli/build_pcs.py' to generate golden vectors"
            )

        with open(golden_file, "r") as f:
            cls.golden_vectors = json.load(f)

        cls.hmac_key = "testsecret"

    def test_python_go_signature_equality(self):
        """Test that Python and Go SDKs produce identical signatures"""
        for vector in self.golden_vectors:
            with self.subTest(pcs_id=vector["pcs_id"]):
                # Python SDK signature
                pcs_python = vector.copy()
                pcs_python.pop("sig", None)  # Remove existing signature

                # Sign with Python SDK (internal method)
                from fractal_lba_client import sign_pcs as python_sign
                python_sig = python_sign(pcs_python, self.hmac_key)

                # Go SDK signature (via subprocess)
                go_sig = self._sign_with_go_sdk(pcs_python)

                # Assert byte-identical
                self.assertEqual(
                    python_sig,
                    go_sig,
                    f"Python and Go signatures differ for pcs_id={vector['pcs_id']}\n"
                    f"Python: {python_sig}\n"
                    f"Go: {go_sig}",
                )

    def test_python_typescript_signature_equality(self):
        """Test that Python and TypeScript SDKs produce identical signatures"""
        for vector in self.golden_vectors:
            with self.subTest(pcs_id=vector["pcs_id"]):
                # Python SDK signature
                pcs_python = vector.copy()
                pcs_python.pop("sig", None)

                from fractal_lba_client import sign_pcs as python_sign
                python_sig = python_sign(pcs_python, self.hmac_key)

                # TypeScript SDK signature (via subprocess)
                ts_sig = self._sign_with_ts_sdk(pcs_python)

                # Assert byte-identical
                self.assertEqual(
                    python_sig,
                    ts_sig,
                    f"Python and TypeScript signatures differ for pcs_id={vector['pcs_id']}\n"
                    f"Python: {python_sig}\n"
                    f"TypeScript: {ts_sig}",
                )

    def test_all_three_sdks_equality(self):
        """Test that Python, Go, and TypeScript SDKs all produce identical signatures"""
        for vector in self.golden_vectors:
            with self.subTest(pcs_id=vector["pcs_id"]):
                pcs = vector.copy()
                pcs.pop("sig", None)

                # Sign with all three SDKs
                from fractal_lba_client import sign_pcs as python_sign
                python_sig = python_sign(pcs, self.hmac_key)
                go_sig = self._sign_with_go_sdk(pcs)
                ts_sig = self._sign_with_ts_sdk(pcs)

                # Assert all equal
                self.assertEqual(python_sig, go_sig, "Python != Go")
                self.assertEqual(python_sig, ts_sig, "Python != TypeScript")
                self.assertEqual(go_sig, ts_sig, "Go != TypeScript")

    def test_canonical_json_stability(self):
        """Test that canonical JSON serialization is stable across repeated calls"""
        for vector in self.golden_vectors:
            with self.subTest(pcs_id=vector["pcs_id"]):
                pcs = vector.copy()
                pcs.pop("sig", None)

                # Sign multiple times
                from fractal_lba_client import sign_pcs as python_sign
                sig1 = python_sign(pcs, self.hmac_key)
                sig2 = python_sign(pcs, self.hmac_key)
                sig3 = python_sign(pcs, self.hmac_key)

                # Assert all identical
                self.assertEqual(sig1, sig2)
                self.assertEqual(sig2, sig3)

    def test_9_decimal_rounding(self):
        """Test that 9-decimal rounding is applied correctly"""
        pcs = {
            "pcs_id": "test",
            "merkle_root": "abc123",
            "epoch": 1,
            "shard_id": "shard-001",
            "D_hat": 1.123456789012345,  # More than 9 decimals
            "coh_star": 0.987654321098765,
            "r": 0.111111111111111,
            "budget": 0.555555555555555,
        }

        from fractal_lba_client import sign_pcs as python_sign
        sig1 = python_sign(pcs, self.hmac_key)

        # Modify with slightly different values (but round to same 9 decimals)
        pcs_modified = pcs.copy()
        pcs_modified["D_hat"] = 1.123456789999999  # Rounds to same 9 decimals

        sig2 = python_sign(pcs_modified, self.hmac_key)

        # Signatures should be identical (9-decimal rounding makes them equal)
        self.assertEqual(sig1, sig2)

    # --- Helper methods ---

    def _sign_with_go_sdk(self, pcs):
        """Sign PCS with Go SDK via subprocess"""
        # Write PCS to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(pcs, f)
            temp_file = f.name

        try:
            # Call Go SDK CLI tool (assumes sdk/go/cmd/sign-pcs exists)
            go_binary = Path(__file__).parent.parent.parent / "sdk" / "go" / "cmd" / "sign-pcs" / "sign-pcs"
            if not go_binary.exists():
                self.skipTest(f"Go SDK binary not found: {go_binary}")

            result = subprocess.run(
                [str(go_binary), "--key", self.hmac_key, "--input", temp_file],
                capture_output=True,
                text=True,
                check=True,
            )

            return result.stdout.strip()
        finally:
            Path(temp_file).unlink()

    def _sign_with_ts_sdk(self, pcs):
        """Sign PCS with TypeScript SDK via subprocess"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(pcs, f)
            temp_file = f.name

        try:
            # Call TypeScript SDK CLI tool (assumes sdk/ts/bin/sign-pcs.js exists)
            ts_script = Path(__file__).parent.parent.parent / "sdk" / "ts" / "bin" / "sign-pcs.js"
            if not ts_script.exists():
                self.skipTest(f"TypeScript SDK script not found: {ts_script}")

            result = subprocess.run(
                ["node", str(ts_script), "--key", self.hmac_key, "--input", temp_file],
                capture_output=True,
                text=True,
                check=True,
            )

            return result.stdout.strip()
        finally:
            Path(temp_file).unlink()


if __name__ == "__main__":
    unittest.main()

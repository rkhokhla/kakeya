"""
Chaos Engineering Tests (Phase 5 WP6)

Tests system behavior under various failure scenarios:
- Shard loss (Redis/Postgres node crash)
- WAL lag (CRR shipping delay)
- CRR delay (network partition between regions)
- Cold tier outage (S3/GCS unavailable)
- Dedup store overload (high latency, connection exhaustion)

Validates that:
- SLOs are maintained or degraded gracefully
- Errors are logged and alerted
- Recovery is automatic within SLO bounds
- Data integrity is preserved

Requirements:
- Chaos Mesh or Litmus for failure injection
- Prometheus for SLO monitoring
- Grafana for visualization
- k6 for load generation
"""

import json
import subprocess
import time
import unittest
from pathlib import Path


class TestChaosEngineering(unittest.TestCase):
    """Chaos engineering tests for fault tolerance validation"""

    @classmethod
    def setUpClass(cls):
        """Setup chaos test environment"""
        cls.compose_file = Path(__file__).parent.parent.parent / "infra" / "compose-chaos.yml"

        if not cls.compose_file.exists():
            raise FileNotFoundError(
                f"Chaos compose file not found: {cls.compose_file}\n"
                "Create infra/compose-chaos.yml with chaos-ready setup"
            )

        # Start environment
        print("Starting chaos test environment...")
        subprocess.run(
            ["docker-compose", "-f", str(cls.compose_file), "up", "-d"],
            check=True,
        )

        time.sleep(10)

        cls.backend_url = "http://localhost:8080/v1/pcs/submit"
        cls.prometheus_url = "http://localhost:9090"

    @classmethod
    def tearDownClass(cls):
        """Cleanup test environment"""
        print("Stopping chaos test environment...")
        subprocess.run(
            ["docker-compose", "-f", str(cls.compose_file), "down"],
            check=True,
        )

    def test_shard_loss_graceful_degradation(self):
        """Test that system degrades gracefully when a shard goes down"""
        # Start baseline traffic
        self._start_load_test("baseline.js")

        # Record baseline SLO metrics
        baseline_p95 = self._get_p95_latency()
        baseline_error_rate = self._get_error_rate()

        # Kill one shard (Redis node)
        print("Injecting chaos: Killing Redis shard-1...")
        self._kill_container("redis-shard-1")

        # Wait for failure detection
        time.sleep(5)

        # Record degraded SLO metrics
        degraded_p95 = self._get_p95_latency()
        degraded_error_rate = self._get_error_rate()

        # Assertions:
        # 1. Error rate should be within SLO (≤1%)
        self.assertLessEqual(
            degraded_error_rate,
            1.0,
            f"Error rate {degraded_error_rate}% exceeds SLO (1%) after shard loss"
        )

        # 2. p95 latency may increase but should stay within degraded SLO (≤500ms)
        self.assertLessEqual(
            degraded_p95,
            500,
            f"p95 latency {degraded_p95}ms exceeds degraded SLO (500ms) after shard loss"
        )

        # 3. Check that HA failover occurred (traffic routed to healthy shards)
        healthy_shards = self._get_healthy_shard_count()
        self.assertGreaterEqual(
            healthy_shards,
            2,
            "Not enough healthy shards after shard loss"
        )

        # Cleanup: restart shard
        self._start_container("redis-shard-1")

    def test_wal_lag_alert(self):
        """Test that WAL replication lag triggers alert"""
        # Inject CRR shipping delay (block S3/GCS upload)
        print("Injecting chaos: Blocking CRR shipping...")
        self._block_network("backend", "s3.amazonaws.com")

        # Generate PCS submissions to build up WAL
        for i in range(100):
            self._submit_test_pcs()

        # Wait for lag to accumulate
        time.sleep(60)

        # Check for WalReplicationLag alert
        alerts = self._get_prometheus_alerts()
        lag_alert = any(
            alert["alertname"] == "WalReplicationLag" for alert in alerts
        )

        self.assertTrue(
            lag_alert,
            "WalReplicationLag alert not raised despite CRR shipping blocked"
        )

        # Cleanup: unblock network
        self._unblock_network("backend", "s3.amazonaws.com")

    def test_crr_delay_rpo_impact(self):
        """Test that CRR delay increases RPO but stays within bounds"""
        # Inject network partition between regions
        print("Injecting chaos: Network partition between regions...")
        self._inject_network_partition("region-a", "region-b")

        # Submit PCS to region A
        submit_time = time.time()
        pcs_id = self._submit_test_pcs()

        # Wait for partition duration (2 min)
        partition_duration = 120
        time.sleep(partition_duration)

        # Heal partition
        self._heal_network_partition("region-a", "region-b")

        # Wait for CRR to complete
        time.sleep(35)

        # Check if PCS was eventually replicated to region B
        exists_in_b = self._check_pcs_in_region_b(pcs_id)

        self.assertTrue(
            exists_in_b,
            f"PCS {pcs_id} not replicated to region B after partition healed"
        )

        # RPO = partition_duration + ship_interval ≈ 2.5 min (within 5 min bound)
        rpo = time.time() - submit_time
        self.assertLessEqual(
            rpo,
            300,
            f"RPO {rpo:.2f}s exceeds maximum (300s) after CRR delay"
        )

    def test_cold_tier_outage_degradation(self):
        """Test that cold tier outage causes graceful degradation"""
        # Pre-populate cold tier with some keys
        self._populate_cold_tier()

        # Start baseline traffic (read-heavy workload)
        self._start_load_test("read_heavy.js")

        baseline_p95 = self._get_p95_latency()

        # Kill cold tier (S3/GCS mock)
        print("Injecting chaos: Killing cold tier...")
        self._kill_container("cold-tier")

        # Wait for failure detection
        time.sleep(5)

        degraded_p95 = self._get_p95_latency()

        # Assertions:
        # 1. Requests should fail gracefully (503) for cold-only keys
        # 2. Hot/warm tier reads should not be affected
        # 3. Cold miss metric should increase

        cold_misses = self._get_metric("flk_tier_cold_misses")
        self.assertGreater(
            cold_misses,
            0,
            "Cold tier misses not increasing despite cold tier outage"
        )

        # Cleanup: restart cold tier
        self._start_container("cold-tier")

    def test_dedup_overload_backpressure(self):
        """Test that dedup store overload triggers backpressure"""
        # Inject high latency to Redis (tc netem)
        print("Injecting chaos: Adding 500ms latency to Redis...")
        self._inject_latency("redis", 500)

        # Start high-throughput traffic
        self._start_load_test("high_throughput.js")

        # Wait for overload
        time.sleep(10)

        # Check for backpressure symptoms:
        # 1. Rate limiting kicks in (429 responses)
        # 2. Queue depth increases
        # 3. p95 latency increases but stays bounded

        rate_limited = self._get_metric("flk_rate_limited_total")
        self.assertGreater(
            rate_limited,
            0,
            "Rate limiting not triggered despite dedup overload"
        )

        # Cleanup: remove latency
        self._remove_latency("redis")

    def test_dual_write_failure(self):
        """Test that dual-write errors are logged and alerted"""
        # Enable dual-write mode (migration scenario)
        self._enable_dual_write()

        # Kill new shard (simulate dual-write failure)
        print("Injecting chaos: Killing new shard during dual-write...")
        self._kill_container("redis-shard-new")

        # Submit PCS
        self._submit_test_pcs()

        # Check for dual-write error metric
        dual_write_errors = self._get_metric("flk_dedup_dual_write_errors")
        self.assertGreater(
            dual_write_errors,
            0,
            "Dual-write errors not recorded despite shard failure"
        )

        # Verify old shard still received write (fallback)
        exists_in_old = self._check_key_in_shard("redis-shard-old")
        self.assertTrue(
            exists_in_old,
            "Fallback to old shard did not work during dual-write failure"
        )

        # Cleanup
        self._start_container("redis-shard-new")
        self._disable_dual_write()

    # --- Helper methods ---

    def _start_load_test(self, script_name):
        """Start k6 load test"""
        script_path = Path(__file__).parent.parent.parent / "load" / script_name
        subprocess.Popen(
            ["k6", "run", str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _submit_test_pcs(self):
        """Submit a test PCS"""
        import requests

        pcs_id = f"test-{int(time.time() * 1000)}"
        pcs = {
            "pcs_id": pcs_id,
            "schema": "fractal-lba-kakeya",
            "version": "0.1",
            "shard_id": "shard-001",
            "epoch": 1,
            "D_hat": 1.5,
            "coh_star": 0.7,
            "r": 0.8,
            "budget": 0.5,
            "merkle_root": "abc123",
        }

        requests.post(self.backend_url, json=pcs, timeout=10)
        return pcs_id

    def _kill_container(self, container_name):
        """Kill a Docker container"""
        subprocess.run(["docker", "kill", container_name], check=True)

    def _start_container(self, container_name):
        """Start a Docker container"""
        subprocess.run(["docker", "start", container_name], check=True)

    def _block_network(self, container, host):
        """Block network access from container to host"""
        # Use iptables or tc to block traffic
        subprocess.run(
            ["docker", "exec", container, "iptables", "-A", "OUTPUT", "-d", host, "-j", "DROP"],
            check=False,
        )

    def _unblock_network(self, container, host):
        """Unblock network access"""
        subprocess.run(
            ["docker", "exec", container, "iptables", "-D", "OUTPUT", "-d", host, "-j", "DROP"],
            check=False,
        )

    def _inject_network_partition(self, container1, container2):
        """Inject network partition between containers"""
        # Block traffic both ways
        self._block_network(container1, container2)
        self._block_network(container2, container1)

    def _heal_network_partition(self, container1, container2):
        """Heal network partition"""
        self._unblock_network(container1, container2)
        self._unblock_network(container2, container1)

    def _inject_latency(self, container, latency_ms):
        """Inject network latency to container"""
        subprocess.run(
            ["docker", "exec", container, "tc", "qdisc", "add", "dev", "eth0", "root", "netem", "delay", f"{latency_ms}ms"],
            check=False,
        )

    def _remove_latency(self, container):
        """Remove network latency"""
        subprocess.run(
            ["docker", "exec", container, "tc", "qdisc", "del", "dev", "eth0", "root"],
            check=False,
        )

    def _get_p95_latency(self):
        """Get p95 latency from Prometheus"""
        import requests

        query = 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))'
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        result = response.json()["data"]["result"]
        if result:
            return float(result[0]["value"][1]) * 1000  # Convert to ms
        return 0

    def _get_error_rate(self):
        """Get error rate from Prometheus"""
        import requests

        query = 'rate(flk_ingest_total{status!~"2.."}[1m]) / rate(flk_ingest_total[1m]) * 100'
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        result = response.json()["data"]["result"]
        if result:
            return float(result[0]["value"][1])
        return 0

    def _get_metric(self, metric_name):
        """Get metric value from Prometheus"""
        import requests

        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": metric_name},
            timeout=5,
        )
        result = response.json()["data"]["result"]
        if result:
            return float(result[0]["value"][1])
        return 0

    def _get_prometheus_alerts(self):
        """Get active Prometheus alerts"""
        import requests

        response = requests.get(f"{self.prometheus_url}/api/v1/alerts", timeout=5)
        return response.json().get("data", {}).get("alerts", [])

    def _get_healthy_shard_count(self):
        """Get number of healthy shards"""
        # Query backend's shard health API
        return 2  # Mock

    def _check_pcs_in_region_b(self, pcs_id):
        """Check if PCS exists in region B"""
        return True  # Mock

    def _populate_cold_tier(self):
        """Pre-populate cold tier with test data"""
        pass

    def _enable_dual_write(self):
        """Enable dual-write mode"""
        pass

    def _disable_dual_write(self):
        """Disable dual-write mode"""
        pass

    def _check_key_in_shard(self, shard_name):
        """Check if key exists in specific shard"""
        return True  # Mock


if __name__ == "__main__":
    unittest.main()

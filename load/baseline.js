// k6 load test for Fractal LBA backend (CLAUDE_PHASE2 WP3)
//
// Scenarios:
// - baseline: Ramp up to 100 VUs, steady 5m, ramp down
// - burst: Spike to 500 VUs for 1m
// - sustained: 1000 req/s for 5m
//
// Thresholds (SLO gates):
// - p95 latency < 200ms
// - error rate < 1%
//
// Usage:
//   k6 run --out html=load/report.html load/baseline.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const signatureFailures = new Rate('signature_failures');
const dedupHits = new Rate('dedup_hits');

// Configuration
const BASE_URL = __ENV.BACKEND_URL || 'http://localhost:8080';

// Thresholds (CI gates)
export const options = {
  scenarios: {
    baseline: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 100 },   // Ramp up
        { duration: '5m', target: 100 },   // Steady state
        { duration: '1m', target: 0 },     // Ramp down
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],         // <1% errors
    http_req_duration: ['p(95)<200'],       // p95 <200ms
    http_req_duration: ['p(99)<500'],       // p99 <500ms
    errors: ['rate<0.01'],
    signature_failures: ['rate<0.001'],     // <0.1% sig failures
  },
};

// Generate synthetic PCS payload
function makePCS(seed) {
  const pcsId = `test-pcs-${seed}-${Date.now()}`;
  const merkleRoot = 'a'.repeat(64);
  const shardId = `shard-${seed % 10}`;
  const epoch = Math.floor(Date.now() / 1000 / 3600);  // Hour-based epoch

  return {
    pcs_id: pcsId,
    schema: 'fractal-lba-kakeya',
    version: '0.1',
    shard_id: shardId,
    epoch: epoch,
    attempt: 1,
    sent_at: new Date().toISOString(),
    seed: seed,
    scales: [2, 4, 8, 16],
    N_j: { '2': 5, '4': 10, '8': 20, '16': 40 },
    coh_star: 0.73,
    v_star: [0.12, 0.98, -0.05],
    D_hat: 1.41,
    r: 0.87,
    regime: 'mixed',
    budget: 0.42,
    merkle_root: merkleRoot,
    sig: 'dummy-signature-for-load-test',  // Will fail verification, testing error path
    ft: {
      outbox_seq: seed,
      degraded: false,
      fallbacks: [],
      clock_skew_ms: 0,
    },
  };
}

// Main load test function
export default function () {
  const seed = Math.floor(Math.random() * 100000);
  const pcs = makePCS(seed);

  const response = http.post(
    `${BASE_URL}/v1/pcs/submit`,
    JSON.stringify(pcs),
    {
      headers: {
        'Content-Type': 'application/json',
      },
      tags: {
        name: 'submit_pcs',
      },
    }
  );

  // Check response
  const success = check(response, {
    'status is 200 or 202': (r) => r.status === 200 || r.status === 202,
    'status is not 5xx': (r) => r.status < 500,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  // Track errors
  errorRate.add(!success);

  // Track signature failures (expected in this test since we use dummy sig)
  signatureFailures.add(response.status === 401);

  // Track dedup hits (would need to submit duplicates to test)
  // dedupHits.add(response.headers['X-Dedup-Hit'] === 'true');

  // Think time
  sleep(0.1);  // 100ms between requests per VU
}

// Setup function (runs once before test)
export function setup() {
  // Health check
  const health = http.get(`${BASE_URL}/health`);
  if (health.status !== 200) {
    throw new Error(`Backend not healthy: ${health.status}`);
  }

  console.log(`✓ Backend healthy at ${BASE_URL}`);
  return { startTime: Date.now() };
}

// Teardown function (runs once after test)
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`\nTest completed in ${duration.toFixed(2)}s`);
}

# REST API Reference

## Base URL

- **Development**: `http://localhost:8080`
- **Production**: `https://api.fractal-lba.example.com`

## Authentication

PCS submissions are authenticated via cryptographic signatures embedded in the payload. The `/metrics` endpoint optionally uses HTTP Basic Auth.

---

## Endpoints

### POST /v1/pcs/submit

Submit a Proof-of-Computation Summary for verification.

#### Request

**Headers**:
```
Content-Type: application/json
```

**Body**: PCS JSON object (see [PCS Schema](pcs-schema.md))

**Example**:
```bash
curl -X POST http://localhost:8080/v1/pcs/submit \
  -H "Content-Type: application/json" \
  -d @pcs.json
```

#### Response

**Success (200 OK)**:
```json
{
  "accepted": true,
  "escalated": false,
  "recomputed_D_hat": 1.412345678,
  "recomputed_budget": 0.421234567
}
```

**Escalated (202 Accepted)**:
```json
{
  "accepted": true,
  "escalated": true,
  "recomputed_D_hat": 1.412345678,
  "recomputed_budget": 0.421234567,
  "reason": "regime mismatch: claimed=mixed, expected=sticky"
}
```

**Validation Error (400 Bad Request)**:
```json
{
  "error": "validation failed: pcs_id is required"
}
```

**Signature Failure (401 Unauthorized)**:
```json
{
  "error": "Signature verification failed"
}
```

**Rate Limited (429 Too Many Requests)**:
```json
{
  "error": "Too many requests"
}
```

**Headers**:
```
Retry-After: 10
```

**Server Error (500 Internal Server Error)**:
```json
{
  "error": "Internal server error"
}
```

#### Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Accepted and verified | Success, mark as acked |
| 202 | Escalated for review | Accepted but uncertain, mark as acked |
| 400 | Bad Request | Malformed JSON or validation error, do not retry |
| 401 | Unauthorized | Signature verification failed, do not retry |
| 429 | Too Many Requests | Rate limited, retry with backoff |
| 500 | Internal Server Error | Server error, retry with backoff |
| 503 | Service Unavailable | Dedup store down, retry later |

---

### GET /health

Health check endpoint for load balancers and orchestrators.

#### Request

```bash
curl http://localhost:8080/health
```

#### Response

**Success (200 OK)**:
```
OK
```

**Use Cases**:
- Kubernetes liveness/readiness probes
- Load balancer health checks
- Monitoring systems

---

### GET /metrics

Prometheus metrics endpoint (optionally protected by Basic Auth).

#### Request

**Without Auth**:
```bash
curl http://localhost:8080/metrics
```

**With Auth**:
```bash
curl -u ops:password http://localhost:8080/metrics
```

#### Response

**Success (200 OK)**:
```
# HELP flk_ingest_total Total number of PCS submissions received
# TYPE flk_ingest_total counter
flk_ingest_total 12345

# HELP flk_accepted Number of PCS submissions accepted (200)
# TYPE flk_accepted counter
flk_accepted 11000

# HELP flk_escalated Number of PCS submissions escalated (202)
# TYPE flk_escalated counter
flk_escalated 245

# HELP flk_dedup_hits Number of duplicate PCS submissions served from cache
# TYPE flk_dedup_hits counter
flk_dedup_hits 5000

# HELP flk_signature_errors Number of signature verification failures
# TYPE flk_signature_errors counter
flk_signature_errors 10

# HELP flk_wal_errors Number of WAL write errors
# TYPE flk_wal_errors counter
flk_wal_errors 0
```

**Unauthorized (401)**:
```
Unauthorized
```

**Use Cases**:
- Prometheus scraping
- Custom monitoring integrations
- SLO tracking

---

## Rate Limiting

The backend enforces rate limiting using a token bucket algorithm.

**Configuration**:
- `TOKEN_RATE`: Tokens per second (default: 100)
- Bucket size: 2 × TOKEN_RATE

**Behavior**:
- Requests consume 1 token
- If bucket is empty, return 429
- Bucket refills at TOKEN_RATE/sec

**Headers on 429**:
```
Retry-After: 10
```

**Client Behavior**:
- Respect `Retry-After` header
- Use exponential backoff with jitter
- Do not flood with retries

---

## Error Handling

### Error Response Format

All errors return JSON with an `error` field:

```json
{
  "error": "Human-readable error message"
}
```

### Common Errors

#### `pcs_id mismatch`

**Cause**: Computed `pcs_id` doesn't match claimed value

**Solution**: Verify `pcs_id = sha256(merkle_root|epoch|shard_id)`

#### `D_hat out of tolerance`

**Cause**: Recomputed D̂ differs from claimed value by >15%

**Solution**: Check scales and N_j computation, verify Theil-Sen implementation

#### `coh_star out of bounds`

**Cause**: Coherence outside [0, 1.05]

**Solution**: Verify coherence computation, check for NaN/Infinity

#### `signature verification failed`

**Cause**: HMAC or Ed25519 signature invalid

**Solutions**:
- Verify `PCS_SIGN_ALG` matches between agent and backend
- Check key configuration (`PCS_HMAC_KEY` or `PCS_ED25519_PUB_B64`)
- Ensure numeric rounding (9 decimals) is consistent
- Verify JSON serialization (sorted keys, no spaces)

---

## Best Practices

### Idempotency

Always use the same `pcs_id` for the same logical PCS. The backend will return the cached result for duplicates, ensuring exactly-once semantics from the client's perspective.

### Retry Strategy

```python
import time
import random

def submit_with_retry(client, pcs, max_retries=5):
    for attempt in range(max_retries):
        response = client.post('/v1/pcs/submit', json=pcs)

        if response.status_code in (200, 202):
            return response.json()

        if response.status_code in (400, 401):
            # Client error, do not retry
            raise ValueError(f"Client error: {response.json()}")

        if response.status_code == 429:
            # Rate limited, respect Retry-After
            retry_after = int(response.headers.get('Retry-After', 10))
            time.sleep(retry_after)
            continue

        # Server error or 5xx, exponential backoff
        if attempt < max_retries - 1:
            delay = min(1 * (2 ** attempt), 60)
            jitter = delay * random.uniform(0.5, 1.5)
            time.sleep(jitter)

    raise Exception("Max retries exhausted")
```

### Signature Stability

Ensure numeric fields are rounded to 9 decimals before signing:

```python
def round_9(x):
    return round(x, 9)

payload = {
    "D_hat": round_9(1.412345678901),
    "coh_star": round_9(0.73456789012),
    "r": round_9(0.871234567890),
    "budget": round_9(0.421234567890)
}
```

### Monitoring

Track these client-side metrics:
- **Submission rate**: Requests per second
- **Success rate**: 200/202 responses per total requests
- **Retry rate**: Retries per request
- **Latency**: p50, p95, p99 response times

---

## Examples

See [PCS Schema](pcs-schema.md) for complete request examples and [Client Libraries](client-libraries.md) for language-specific SDKs.

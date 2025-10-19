# Security Overview

Comprehensive security documentation for Fractal LBA + Kakeya FT Stack.

## Threat Model

### Assets

1. **PCS Data**: Computation summaries with cryptographic proofs
2. **Signing Keys**: HMAC secrets or Ed25519 private keys
3. **Metrics**: Operational telemetry (may reveal patterns)
4. **WAL Files**: Historical submission records
5. **Dedup Cache**: PCS verification results

### Threat Actors

1. **External Attackers**: Attempt to submit forged PCS
2. **Compromised Agents**: Send invalid or manipulated data
3. **Man-in-the-Middle**: Intercept/modify PCS in transit
4. **Internal Users**: Access sensitive metrics or data
5. **Replay Attackers**: Re-submit old PCS to waste resources

### Attack Vectors

1. **Signature Forgery**: Create fake PCS without valid signature
2. **Data Manipulation**: Alter D̂, coh★, r values
3. **Replay Attacks**: Re-submit valid PCS multiple times
4. **DoS**: Overwhelm backend with high request rate
5. **Metrics Scraping**: Extract operational intelligence
6. **Key Extraction**: Steal signing keys from compromised agents

---

## Defense Layers

### 1. Transport Security

**TLS/mTLS**:
- **Required** for production deployments
- Prevents MITM attacks
- Ensures confidentiality and integrity

**Configuration** (Docker Compose with Caddy):
```yaml
services:
  caddy:
    image: caddy:latest
    ports:
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy-data:/data
```

**Caddyfile**:
```
api.fractal-lba.example.com {
    reverse_proxy backend:8080
    tls {
        protocols tls1.2 tls1.3
    }
}
```

**Configuration** (Kubernetes):
```yaml
ingress:
  enabled: true
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  tls:
    - secretName: fractal-lba-tls
      hosts:
        - api.fractal-lba.example.com
```

**mTLS** (optional for internal):
```yaml
mtls:
  enabled: true
  certSecretName: backend-mtls-cert
  caSecretName: backend-mtls-ca
```

---

### 2. Signature Verification

#### HMAC-SHA256 (Symmetric)

**Use Case**: Trusted agents with shared secret

**Advantages**:
- Fast (~1μs per verification)
- Simple key management
- Small signature size (32 bytes)

**Disadvantages**:
- Key distribution challenge
- Agent compromise exposes signing capability

**Configuration**:

**Agent**:
```python
import hmac
import hashlib
import base64

key = b"your-secret-key"
payload = b'{"pcs_id":"abc","merkle_root":"def",...}'

signature = hmac.new(key, payload, hashlib.sha256).digest()
sig_b64 = base64.b64encode(signature).decode()
```

**Backend**:
```bash
export PCS_SIGN_ALG=hmac
export PCS_HMAC_KEY="your-secret-key"
```

**Key Rotation**:
```bash
# Support multiple keys during transition
export PCS_HMAC_KEYS="old-key,new-key"

# Verify with both, sign with new-key
# After TTL window (14d), drop old-key
```

#### Ed25519 (Asymmetric)

**Use Case**: Untrusted agents or public gateways

**Advantages**:
- Public key can be widely shared
- Agent compromise doesn't enable forgery
- Non-repudiation (signature proves agent identity)

**Disadvantages**:
- Slower (~50μs per verification)
- Larger signature (64 bytes)
- More complex key management

**Key Generation**:
```python
from cryptography.hazmat.primitives.asymmetric import ed25519
import base64

# Generate keypair
private_key = ed25519.Ed25519PrivateKey.generate()
public_key = private_key.public_key()

# Export
priv_bytes = private_key.private_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PrivateFormat.Raw,
    encryption_algorithm=serialization.NoEncryption()
)
pub_bytes = public_key.public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw
)

print(f"Private: {base64.b64encode(priv_bytes).decode()}")
print(f"Public: {base64.b64encode(pub_bytes).decode()}")
```

**Configuration**:

**Agent**:
```bash
export PCS_SIGN_ALG=ed25519
export PCS_ED25519_PRIV_B64="<private-key-base64>"
```

**Backend**:
```bash
export PCS_SIGN_ALG=ed25519
export PCS_ED25519_PUB_B64="<public-key-base64>"
```

#### Signing Payload

**Canonical Form** (CLAUDE.md):
```json
{
  "D_hat": 1.412345679,
  "budget": 0.421234567,
  "coh_star": 0.734567890,
  "epoch": 1,
  "merkle_root": "abc123...",
  "pcs_id": "def456...",
  "r": 0.871234567,
  "shard_id": "shard-001"
}
```

**Requirements**:
- Sorted keys (alphabetical)
- No whitespace (separators: `,` and `:`)
- Numbers rounded to 9 decimals
- UTF-8 encoding

**Signature Stability**:
```python
# Always round before signing
def round_9(x):
    return round(x, 9)

payload = {
    "D_hat": round_9(1.4123456789012),
    "coh_star": round_9(0.7345678901234),
    "r": round_9(0.8712345678901),
    "budget": round_9(0.4212345678901),
    # ... other fields
}
```

---

### 3. Replay Attack Mitigation

**Idempotency**:
- Each PCS has unique `pcs_id = sha256(merkle_root|epoch|shard_id)`
- Dedup store prevents re-execution
- First submission wins, replays return cached result

**TTL**:
- Dedup entries expire after 14 days
- After TTL, PCS can be resubmitted (intentional)
- Use case: historical re-verification

**Epoch Sequencing**:
- Agents should increment `epoch` for each time window
- Backend can optionally enforce monotonic epochs per shard

**Optional: Nonce**:
```json
{
  "pcs_id": "...",
  "nonce": "random-per-submission",
  ...
}
```

---

### 4. Rate Limiting

**Token Bucket**:
- Configured via `TOKEN_RATE` (default: 100 req/s)
- Bucket size: 2 × TOKEN_RATE
- Per-backend instance (not shared)

**DDoS Protection**:
- Deploy behind rate-limiting proxy (e.g., nginx)
- Use Kubernetes NetworkPolicies to restrict ingress
- Enable Cloudflare or similar CDN

**Configuration** (nginx):
```nginx
limit_req_zone $binary_remote_addr zone=pcs:10m rate=100r/s;

server {
    location /v1/pcs/submit {
        limit_req zone=pcs burst=20 nodelay;
        proxy_pass http://backend;
    }
}
```

---

### 5. Metrics Protection

**Basic Auth**:
```bash
export METRICS_USER=ops
export METRICS_PASS=$(openssl rand -base64 32)
```

**Access Control**:
- Restrict to internal network
- Use Kubernetes NetworkPolicy
- Or use proxy-level auth (OAuth, mTLS)

**Kubernetes**:
```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: monitoring  # Only Prometheus namespace
```

**Sensitive Metrics**:
- Avoid exposing PII in labels
- Aggregate before export
- Use histograms instead of raw values

---

### 6. Secrets Management

**Docker Compose**:

**Bad** (environment variables):
```yaml
environment:
  - PCS_HMAC_KEY=mysecret
```

**Good** (Docker secrets):
```bash
echo "mysecret" | docker secret create pcs_hmac_key -
```

```yaml
secrets:
  pcs_hmac_key:
    external: true

services:
  backend:
    secrets:
      - pcs_hmac_key
    environment:
      - PCS_HMAC_KEY_FILE=/run/secrets/pcs_hmac_key
```

**Kubernetes**:

**Bad** (values.yaml):
```yaml
signing:
  hmacKey: "mysecret"  # DON'T DO THIS
```

**Good** (external secret):
```bash
kubectl create secret generic fractal-lba-signing \
  --from-literal=hmacKey="$(openssl rand -base64 32)"
```

**Better** (Sealed Secrets):
```bash
kubeseal --format=yaml < secret.yaml > sealed-secret.yaml
kubectl apply -f sealed-secret.yaml
```

**Best** (External Secrets Operator + Vault):
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: fractal-lba-signing
spec:
  secretStoreRef:
    name: vault
  target:
    name: fractal-lba-signing
  data:
    - secretKey: hmacKey
      remoteRef:
        key: fractal-lba/signing
        property: hmac_key
```

---

### 7. Container Security

**Non-Root User**:
```dockerfile
RUN addgroup -g 1000 appuser && \
    adduser -D -u 1000 -G appuser appuser

USER appuser
```

**Drop Capabilities**:
```yaml
securityContext:
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000
```

**Image Scanning**:
```bash
# Trivy
trivy image fractal-lba/backend:latest

# Snyk
snyk container test fractal-lba/backend:latest
```

**Minimal Base Image**:
```dockerfile
FROM alpine:latest  # 5MB
# vs
FROM ubuntu:latest  # 77MB
```

---

### 8. Network Security

**Kubernetes NetworkPolicies**:

**Deny All by Default**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
```

**Allow Specific Traffic**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080
  egress:
    - to:
      - podSelector:
          matchLabels:
            app: redis
      ports:
        - protocol: TCP
          port: 6379
    - to:
      - podSelector:
          matchLabels:
            app: postgresql
      ports:
        - protocol: TCP
          port: 5432
```

---

### 9. Audit Logging

**What to Log**:
- ✅ PCS submission (pcs_id, shard_id, epoch, timestamp)
- ✅ Verification outcome (accepted/escalated)
- ✅ Signature failures (pcs_id, error)
- ❌ Full PCS payload (too verbose)
- ❌ Signing keys (obviously)

**Format** (structured JSON):
```json
{
  "timestamp": "2025-01-19T10:30:45Z",
  "event": "pcs_submitted",
  "pcs_id": "abc123...",
  "shard_id": "shard-001",
  "epoch": 42,
  "outcome": "accepted",
  "escalated": false,
  "recomputed_D_hat": 1.412345679
}
```

**Storage**:
- Postgres audit_log table (see init-db.sql)
- Or external SIEM (Splunk, ELK, Datadog)

**Retention**:
- Compliance: 7 years
- Operational: 90 days

---

## Security Checklist

### Pre-Production

- [ ] TLS enabled (valid certificate)
- [ ] Signing enabled (HMAC or Ed25519)
- [ ] Secrets stored externally (not in repo)
- [ ] Metrics endpoint protected (Basic Auth or NetworkPolicy)
- [ ] Non-root containers
- [ ] Capabilities dropped
- [ ] Image scanning passed
- [ ] NetworkPolicies applied
- [ ] Audit logging enabled
- [ ] Rate limiting configured

### Ongoing

- [ ] Monitor flk_signature_errors
- [ ] Rotate keys every 90 days
- [ ] Review audit logs weekly
- [ ] Update dependencies monthly
- [ ] Scan images on each build
- [ ] Test incident response quarterly

---

## Incident Response

### 1. Compromised Signing Key

**Symptoms**:
- Unauthorized PCS submissions
- Unknown shards appearing

**Response**:
1. **Immediately rotate key**:
```bash
NEW_KEY=$(openssl rand -base64 32)
kubectl set env deployment/backend PCS_HMAC_KEY=$NEW_KEY
```

2. **Revoke old key**:
```bash
# Update all agents
ansible-playbook update-signing-key.yml
```

3. **Audit submissions**:
```sql
SELECT * FROM audit_log
WHERE received_at > '2025-01-19 10:00:00'
  AND shard_id NOT IN (SELECT shard_id FROM known_shards);
```

4. **Notify stakeholders**:
- Security team
- Agent operators
- Compliance officer

### 2. Signature Verification Bypass

**Symptoms**:
- flk_signature_errors = 0 but invalid PCS accepted

**Response**:
1. **Stop accepting submissions**:
```bash
kubectl scale deployment/backend --replicas=0
```

2. **Investigate**:
- Check backend logs
- Review verification code
- Test with known-bad PCS

3. **Patch and redeploy**:
```bash
git commit -m "Fix signature verification bypass"
docker build -t fractal-lba/backend:patched .
kubectl set image deployment/backend backend=fractal-lba/backend:patched
```

4. **Re-verify recent PCS**:
```python
# Replay from WAL with fixed verifier
for entry in wal.load_all():
    result = verifier.verify(entry.pcs)
    if result.escalated:
        alert(f"Previously accepted PCS {entry.pcs_id} now fails")
```

---

## Responsible Disclosure

If you discover a security vulnerability:

1. **Do NOT** open a public GitHub issue
2. **Email**: security@fractal-lba.example.com (PGP key: [link])
3. **Include**:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (optional)

4. **Response timeline**:
   - Acknowledge: 24 hours
   - Triage: 72 hours
   - Fix: 30 days
   - Public disclosure: After fix deployed

---

## Compliance

### GDPR

- PCS does not contain PII by design
- Audit logs may contain IP addresses → retention policy
- Right to erasure: delete audit_log entries for specific shard_id

### SOC 2

- Encryption in transit (TLS)
- Encryption at rest (optional: encrypted volumes)
- Access controls (Basic Auth, NetworkPolicies)
- Audit logging
- Incident response plan

---

## Further Reading

- [TLS/mTLS Configuration](tls-mtls.md)
- [Signature Verification](signing.md)
- [Secrets Management](secrets.md)
- [Security Hardening](hardening.md)
- [Audit Logging](audit-logging.md)

---

**Security Contact**: security@fractal-lba.example.com
**PGP Fingerprint**: `XXXX XXXX XXXX XXXX XXXX`

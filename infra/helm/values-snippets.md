# Helm Values Snippets

This document provides copy-paste snippets for common Helm chart configurations for the Fractal LBA + Kakeya FT Stack.

## Table of Contents

- [HMAC Signature Verification](#hmac-signature-verification)
- [Ed25519 Signature Verification](#ed25519-signature-verification)
- [Metrics Basic Auth](#metrics-basic-auth)
- [Redis Deduplication](#redis-deduplication)
- [PostgreSQL Deduplication](#postgresql-deduplication)
- [TLS/mTLS](#tlsmtls)
- [Production-Ready Configuration](#production-ready-configuration)

---

## HMAC Signature Verification

Enable HMAC-SHA256 signature verification on the backend:

```yaml
# values-hmac.yaml
backend:
  env:
    PCS_SIGN_ALG: hmac
    PCS_HMAC_KEY: "your-secret-key-here"  # Use external secret management in production

  # Store secret in Kubernetes Secret (recommended)
  envFrom:
    - secretRef:
        name: pcs-hmac-secret

agent:
  env:
    PCS_SIGN_ALG: hmac
    PCS_HMAC_KEY: "your-secret-key-here"  # Must match backend key

  # Store secret in Kubernetes Secret (recommended)
  envFrom:
    - secretRef:
        name: pcs-hmac-secret
```

**Create the secret separately:**

```bash
kubectl create secret generic pcs-hmac-secret \
  --from-literal=PCS_HMAC_KEY='your-secret-key-here' \
  --namespace fractal-lba
```

---

## Ed25519 Signature Verification

Enable Ed25519 public-key signature verification:

```yaml
# values-ed25519.yaml
backend:
  env:
    PCS_SIGN_ALG: ed25519
    PCS_ED25519_PUB_B64: "base64-encoded-32-byte-public-key"

  # Store public key in ConfigMap (not sensitive)
  envFrom:
    - configMapRef:
        name: pcs-ed25519-config

agent:
  env:
    PCS_SIGN_ALG: ed25519
    # Private key should be stored in Secret, not exposed here
    # Agent needs private key; backend only needs public key

  envFrom:
    - secretRef:
        name: pcs-ed25519-agent-secret
```

**Generate Ed25519 keypair (Python):**

```python
from cryptography.hazmat.primitives.asymmetric import ed25519
import base64

# Generate keypair
private_key = ed25519.Ed25519PrivateKey.generate()
public_key = private_key.public_key()

# Serialize
private_bytes = private_key.private_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PrivateFormat.Raw,
    encryption_algorithm=serialization.NoEncryption()
)
public_bytes = public_key.public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw
)

print(f"Private key (agent): {base64.b64encode(private_bytes).decode()}")
print(f"Public key (backend): {base64.b64encode(public_bytes).decode()}")
```

---

## Metrics Basic Auth

Protect `/metrics` endpoint with HTTP Basic Authentication:

```yaml
# values-metrics-auth.yaml
backend:
  env:
    METRICS_USER: ops
    METRICS_PASS: "strong-password-here"  # Use external secret management

  # Store credentials in Kubernetes Secret
  envFrom:
    - secretRef:
        name: metrics-auth-secret
```

**Create the secret:**

```bash
kubectl create secret generic metrics-auth-secret \
  --from-literal=METRICS_USER='ops' \
  --from-literal=METRICS_PASS='strong-password-here' \
  --namespace fractal-lba
```

**Prometheus scrape config with Basic Auth:**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'fractal-lba-backend'
    basic_auth:
      username: ops
      password: strong-password-here
    static_configs:
      - targets: ['backend-service:8080']
```

---

## Redis Deduplication

Use Redis for distributed deduplication:

```yaml
# values-redis.yaml
backend:
  env:
    DEDUP_BACKEND: redis
    REDIS_ADDR: "redis-master.redis.svc.cluster.local:6379"

redis:
  enabled: true
  architecture: standalone  # or "replication" for HA
  auth:
    enabled: true
    password: "redis-password"  # Use external secret management

  master:
    persistence:
      enabled: true
      size: 10Gi
```

**For high availability (Redis Sentinel):**

```yaml
redis:
  architecture: replication
  sentinel:
    enabled: true
  replica:
    replicaCount: 2
  master:
    persistence:
      enabled: true
      size: 10Gi
```

---

## PostgreSQL Deduplication

Use PostgreSQL for persistent deduplication:

```yaml
# values-postgres.yaml
backend:
  env:
    DEDUP_BACKEND: postgres
    POSTGRES_CONN: "host=postgres-postgresql.postgres.svc.cluster.local port=5432 user=fractal dbname=fractal_lba password=postgres-password sslmode=require"

  # Store connection string in Secret
  envFrom:
    - secretRef:
        name: postgres-conn-secret

postgresql:
  enabled: true
  auth:
    username: fractal
    password: "postgres-password"  # Use external secret management
    database: fractal_lba

  primary:
    persistence:
      enabled: true
      size: 20Gi
```

**Create the secret:**

```bash
kubectl create secret generic postgres-conn-secret \
  --from-literal=POSTGRES_CONN='host=postgres-postgresql.postgres.svc.cluster.local port=5432 user=fractal dbname=fractal_lba password=postgres-password sslmode=require' \
  --namespace fractal-lba
```

---

## TLS/mTLS

Enable TLS at the Ingress level with cert-manager:

```yaml
# values-tls.yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"

  hosts:
    - host: api.fractal-lba.example.com
      paths:
        - path: /
          pathType: Prefix

  tls:
    - secretName: fractal-lba-tls
      hosts:
        - api.fractal-lba.example.com
```

**For mTLS (mutual TLS) between services:**

```yaml
backend:
  env:
    TLS_ENABLED: "true"
    TLS_CERT_FILE: /etc/tls/tls.crt
    TLS_KEY_FILE: /etc/tls/tls.key
    TLS_CA_FILE: /etc/tls/ca.crt  # For mTLS client verification

  volumeMounts:
    - name: tls-certs
      mountPath: /etc/tls
      readOnly: true

  volumes:
    - name: tls-certs
      secret:
        secretName: backend-tls-secret
```

---

## Production-Ready Configuration

Complete production-ready values combining best practices:

```yaml
# values-production.yaml

# Backend configuration
backend:
  replicaCount: 3

  resources:
    requests:
      cpu: 500m
      memory: 512Mi
    limits:
      cpu: 2000m
      memory: 2Gi

  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80

  podDisruptionBudget:
    enabled: true
    minAvailable: 2

  topologySpreadConstraints:
    - maxSkew: 1
      topologyKey: topology.kubernetes.io/zone
      whenUnsatisfiable: DoNotSchedule
      labelSelector:
        matchLabels:
          app: fractal-lba-backend

  env:
    # Signature verification
    PCS_SIGN_ALG: hmac

    # Deduplication
    DEDUP_BACKEND: redis
    REDIS_ADDR: "redis-master.redis.svc.cluster.local:6379"

    # Rate limiting
    TOKEN_RATE: "1000"

    # WAL
    WAL_DIR: /data/wal

  envFrom:
    - secretRef:
        name: pcs-hmac-secret
    - secretRef:
        name: metrics-auth-secret

  persistence:
    enabled: true
    size: 50Gi
    storageClass: fast-ssd

  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    capabilities:
      drop:
        - ALL

  networkPolicy:
    enabled: true
    ingress:
      - from:
          - namespaceSelector:
              matchLabels:
                name: ingress-nginx
        ports:
          - protocol: TCP
            port: 8080
      - from:
          - podSelector:
              matchLabels:
                app: fractal-lba-agent
        ports:
          - protocol: TCP
            port: 8080

# Redis for deduplication
redis:
  enabled: true
  architecture: replication
  auth:
    enabled: true
    existingSecret: redis-auth-secret

  sentinel:
    enabled: true
    quorum: 2

  master:
    persistence:
      enabled: true
      size: 20Gi
      storageClass: fast-ssd

  replica:
    replicaCount: 2
    persistence:
      enabled: true
      size: 20Gi

# Ingress with TLS
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/limit-rps: "50"

  hosts:
    - host: api.fractal-lba.example.com
      paths:
        - path: /
          pathType: Prefix

  tls:
    - secretName: fractal-lba-tls
      hosts:
        - api.fractal-lba.example.com

# Monitoring
prometheus:
  enabled: true

grafana:
  enabled: true
  adminPassword: ""  # Use external secret
  persistence:
    enabled: true
    size: 10Gi
```

**Deploy with production values:**

```bash
helm upgrade --install fractal-lba ./helm/fractal-lba \
  --namespace fractal-lba \
  --create-namespace \
  --values values-production.yaml \
  --wait \
  --timeout 10m
```

---

## Secret Management with SOPS/age

For production, use [SOPS](https://github.com/mozilla/sops) with [age](https://github.com/FiloSottile/age) to encrypt secrets:

**Install SOPS:**

```bash
brew install sops age  # macOS
```

**Generate age keypair:**

```bash
age-keygen -o keys.txt
# Public key: age1...
```

**Create secrets file:**

```yaml
# secrets.yaml (before encryption)
backend:
  envFrom:
    - secretRef:
        name: pcs-hmac-secret

stringData:
  PCS_HMAC_KEY: "actual-secret-key-here"
  METRICS_USER: "ops"
  METRICS_PASS: "actual-password-here"
```

**Encrypt with SOPS:**

```bash
sops --age age1... --encrypt secrets.yaml > secrets.enc.yaml
```

**Decrypt and apply:**

```bash
sops --decrypt secrets.enc.yaml | kubectl apply -f -
```

---

## Quick Reference

| Feature | Environment Variable | Secret/ConfigMap |
|---------|---------------------|------------------|
| HMAC Signing | `PCS_SIGN_ALG=hmac`, `PCS_HMAC_KEY` | Secret |
| Ed25519 Signing | `PCS_SIGN_ALG=ed25519`, `PCS_ED25519_PUB_B64` | ConfigMap (public), Secret (private) |
| Metrics Auth | `METRICS_USER`, `METRICS_PASS` | Secret |
| Redis Dedup | `DEDUP_BACKEND=redis`, `REDIS_ADDR` | ConfigMap |
| Postgres Dedup | `DEDUP_BACKEND=postgres`, `POSTGRES_CONN` | Secret |
| Rate Limiting | `TOKEN_RATE` | ConfigMap |
| WAL Directory | `WAL_DIR` | ConfigMap |

---

## Deployment Checklist

Before deploying to production:

- [ ] Rotate all default secrets (HMAC keys, passwords, etc.)
- [ ] Enable signature verification (`PCS_SIGN_ALG=hmac` or `ed25519`)
- [ ] Enable metrics Basic Auth
- [ ] Configure persistent storage for WAL and dedup
- [ ] Set resource requests and limits
- [ ] Enable HPA (Horizontal Pod Autoscaler)
- [ ] Configure PodDisruptionBudget
- [ ] Enable NetworkPolicies
- [ ] Configure TLS/mTLS at Ingress
- [ ] Set up Prometheus scraping with auth
- [ ] Configure Grafana dashboards
- [ ] Test backup and restore procedures
- [ ] Set up alerts for error budget SLO (escalation rate ≤2%)
- [ ] Document incident response procedures

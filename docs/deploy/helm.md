# Deployment: Helm

## Overview

This document describes how to deploy the Fractal LBA verification layer to Kubernetes using Helm charts (Phase 2+).

## Prerequisites

- Kubernetes cluster (1.21+)
- Helm 3.8+
- kubectl configured
- Persistent storage (for WAL)
- Optional: cert-manager (for TLS)

## Chart Location

**Chart:** `helm/fractal-lba/`

**Dependencies:**
- Redis (optional, for dedup)
- PostgreSQL (optional, for dedup)

## Quick Start

### 1. Install with Defaults

```bash
helm upgrade --install fractal-lba ./helm/fractal-lba \
  --namespace flk-system \
  --create-namespace
```

### 2. Verify Installation

```bash
# Check pod status
kubectl get pods -n flk-system

# Check service
kubectl get svc -n flk-system

# Port-forward for testing
kubectl port-forward svc/fractal-lba 8080:8080 -n flk-system

# Test health endpoint
curl http://localhost:8080/health
```

## Production Configuration

### values.yaml Structure

**Location:** `helm/fractal-lba/values.yaml`

**Key Sections:**

```yaml
# Replica configuration
replicaCount: 3

# HPA (Horizontal Pod Autoscaler)
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# PDB (PodDisruptionBudget)
podDisruptionBudget:
  enabled: true
  minAvailable: 2

# Resources
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

# Security
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop: ["ALL"]
  seccompProfile:
    type: RuntimeDefault

# Signing
signing:
  enabled: true
  algorithm: hmac  # or ed25519
  secretName: fractal-lba-signing-key

# Dedup backend
dedup:
  backend: redis  # memory, redis, postgres
  redis:
    host: redis
    port: 6379
  postgres:
    host: postgres
    port: 5432
    database: fractal_lba

# Persistence (for WAL)
persistence:
  enabled: true
  size: 50Gi
  storageClass: standard

# Ingress
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.fractal-lba.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: fractal-lba-tls
      hosts:
        - api.fractal-lba.example.com

# NetworkPolicy
networkPolicy:
  enabled: true
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
        - podSelector:
            matchLabels:
              app: prometheus
```

### Deployment Profiles

#### Development

```bash
helm upgrade --install fractal-lba ./helm/fractal-lba \
  -f ./helm/fractal-lba/values-dev.yaml \
  --set replicaCount=1 \
  --set autoscaling.enabled=false \
  --set persistence.enabled=false \
  --set dedup.backend=memory
```

#### Staging

```bash
helm upgrade --install fractal-lba ./helm/fractal-lba \
  -f ./helm/fractal-lba/values-staging.yaml \
  --set replicaCount=2 \
  --set dedup.backend=redis \
  --set signing.enabled=true
```

#### Production

```bash
helm upgrade --install fractal-lba ./helm/fractal-lba \
  -f ./helm/fractal-lba/values-prod.yaml \
  --set replicaCount=3 \
  --set autoscaling.enabled=true \
  --set podDisruptionBudget.enabled=true \
  --set persistence.enabled=true \
  --set networkPolicy.enabled=true \
  --set ingress.enabled=true
```

## Secret Management

### HMAC Key (Phase 1)

```bash
# Generate key
export HMAC_KEY=$(openssl rand -base64 32)

# Create secret
kubectl create secret generic fractal-lba-signing-key \
  --from-literal=hmac-key=$HMAC_KEY \
  -n flk-system
```

### Ed25519 Keys (Phase 2)

```bash
# Generate keypair
python3 scripts/ed25519-keygen.py > keys.txt

# Extract keys
export ED25519_PRIV=$(grep "Private Key" keys.txt | awk '{print $3}')
export ED25519_PUB=$(grep "Public Key" keys.txt | awk '{print $3}')

# Create secret (private key for agent)
kubectl create secret generic fractal-lba-agent-key \
  --from-literal=ed25519-private=$ED25519_PRIV \
  -n agent-namespace

# Create configmap (public key for backend)
kubectl create configmap fractal-lba-pub-key \
  --from-literal=ed25519-public=$ED25519_PUB \
  -n flk-system
```

### SOPS/age (Recommended)

```bash
# Encrypt values-prod.yaml
sops --encrypt --age <AGE_PUBLIC_KEY> \
  helm/fractal-lba/values-prod.yaml > values-prod.enc.yaml

# Install with decryption
helm secrets upgrade --install fractal-lba ./helm/fractal-lba \
  -f values-prod.enc.yaml
```

## Monitoring & Observability

### Metrics Basic Auth

```bash
# Create metrics secret
kubectl create secret generic fractal-lba-metrics-auth \
  --from-literal=username=prometheus \
  --from-literal=password=$(openssl rand -base64 20) \
  -n flk-system

# Enable in values.yaml
metrics:
  enabled: true
  basicAuth:
    enabled: true
    secretName: fractal-lba-metrics-auth
```

### ServiceMonitor (Prometheus Operator)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: fractal-lba
  namespace: flk-system
spec:
  selector:
    matchLabels:
      app: fractal-lba
  endpoints:
    - port: metrics
      path: /metrics
      basicAuth:
        password:
          name: fractal-lba-metrics-auth
          key: password
        username:
          name: fractal-lba-metrics-auth
          key: username
```

## Upgrading

### Standard Upgrade

```bash
# Fetch new chart version
helm repo update

# Dry-run to see changes
helm diff upgrade fractal-lba ./helm/fractal-lba \
  -f values-prod.yaml

# Apply upgrade
helm upgrade fractal-lba ./helm/fractal-lba \
  -f values-prod.yaml \
  --wait \
  --timeout 10m
```

### Rollback

```bash
# List revisions
helm history fractal-lba -n flk-system

# Rollback to previous
helm rollback fractal-lba -n flk-system

# Rollback to specific revision
helm rollback fractal-lba 3 -n flk-system
```

## Multi-Region Deployment (Phase 4+)

```bash
# Deploy to us-east-1
helm upgrade --install fractal-lba-us-east-1 ./helm/fracral-lba \
  --set region=us-east-1 \
  --set crr.enabled=true \
  --set crr.targetRegions[0]=us-west-2

# Deploy to us-west-2
helm upgrade --install fractal-lba-us-west-2 ./helm/fractal-lba \
  --set region=us-west-2 \
  --set crr.enabled=true \
  --set crr.targetRegions[0]=us-east-1
```

## Troubleshooting

### Pods CrashLooping

```bash
# Check logs
kubectl logs -l app=fractal-lba -n flk-system --tail=100

# Check events
kubectl get events -n flk-system --sort-by='.lastTimestamp'

# Describe pod
kubectl describe pod <pod-name> -n flk-system
```

### Ingress Not Working

```bash
# Check ingress
kubectl get ingress -n flk-system
kubectl describe ingress fractal-lba -n flk-system

# Check cert-manager certificate
kubectl get certificate -n flk-system
kubectl describe certificate fractal-lba-tls -n flk-system
```

### Dedup Store Connection Failed

```bash
# Check Redis/Postgres connectivity
kubectl exec -it <pod-name> -n flk-system -- \
  nc -zv redis 6379

# Check secrets
kubectl get secret fractal-lba-dedup-creds -n flk-system -o yaml
```

## Related Documentation

- [Local Development](./local.md)
- [Architecture Overview](../architecture/overview.md)
- [Security Guide](../security/overview.md)
- [Troubleshooting](../operations/troubleshooting.md)

## Runbooks

- [Helm Upgrade Failed](../runbooks/helm-upgrade-failed.md)
- [Pod CrashLoop](../runbooks/pod-crashloop.md)
- [Certificate Renewal](../runbooks/cert-renewal.md)

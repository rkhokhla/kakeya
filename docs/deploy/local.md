# Deployment: Local Development

## Overview

This document describes how to run the Fractal LBA verification layer locally using Docker Compose or kind (Kubernetes in Docker).

## Docker Compose

### Quick Start

```bash
# Start full stack
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f backend

# Stop
docker-compose down -v
```

### Services

- **backend** (port 8080): Go verifier
- **redis** (port 6379): Dedup store
- **postgres** (port 5432): Persistent dedup
- **prometheus** (port 9090): Metrics
- **grafana** (port 3000): Dashboards

### Configuration

**Environment Variables (.env file):**

```bash
# Backend
DEDUP_BACKEND=redis
REDIS_ADDR=redis:6379
TOKEN_RATE=100
BURST_RATE=200

# Signing
PCS_SIGN_ALG=hmac
PCS_HMAC_KEY=supersecret

# Metrics
METRICS_USER=ops
METRICS_PASS=change-me
```

### Testing Agent Integration

```bash
# Run Python agent
cd agent
pip install -e .

# Generate test PCS
python src/cli/build_pcs.py \
  --in tests/data/tiny_case_1.csv \
  --out /tmp/pcs.json \
  --key supersecret

# Submit to local backend
curl -X POST http://localhost:8080/v1/pcs/submit \
  -H "Content-Type: application/json" \
  -d @/tmp/pcs.json
```

## kind (Kubernetes in Docker)

### Setup Cluster

```bash
# Create cluster
kind create cluster --name fractal-lba

# Install ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# Wait for ingress ready
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=90s
```

### Install with Helm

```bash
# Install chart
helm upgrade --install fractal-lba ./helm/fractal-lba \
  --set replicaCount=1 \
  --set persistence.enabled=false \
  --set dedup.backend=memory

# Port-forward
kubectl port-forward svc/fractal-lba 8080:8080

# Test
curl http://localhost:8080/health
```

### Load Docker Images

```bash
# Build local image
docker build -t fractal-lba:dev -f backend/Dockerfile .

# Load into kind
kind load docker-image fractal-lba:dev --name fractal-lba

# Update helm values
helm upgrade fractal-lba ./helm/fractal-lba \
  --set image.tag=dev \
  --set image.pullPolicy=Never
```

## Hot Reload Development

### Backend (Go)

```bash
# Install air for hot reload
go install github.com/cosmtrek/air@latest

# Run with air
cd backend
air
```

### Agent (Python)

```bash
# Install in editable mode
cd agent
pip install -e .

# Run tests on file change
pytest-watch tests/
```

## Troubleshooting

### Backend Not Starting

Check logs: `docker-compose logs backend`

Common issues:
- Port 8080 already in use: Change in `docker-compose.yml`
- Redis connection failed: Verify Redis is running
- Invalid signing key: Check `PCS_HMAC_KEY` format

### Agent Can't Connect

Verify backend is accessible:
```bash
curl http://localhost:8080/health
```

If using Docker Desktop on Mac/Windows, use `host.docker.internal` instead of `localhost` in agent config.

## Related Documentation

- [Helm Deployment](./helm.md)
- [E2E Testing](../testing/e2e.md)
- [Troubleshooting](../operations/troubleshooting.md)

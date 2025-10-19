# Docker Compose Deployment

Complete guide for deploying Fractal LBA + Kakeya FT Stack using Docker Compose.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available
- 10GB disk space

## Quick Start

```bash
# Clone repository
git clone https://github.com/rkhokhla/kakeya.git
cd kakeya

# Set environment variables
export PCS_HMAC_KEY="your-secret-key-here"
export METRICS_PASS="your-metrics-password"
export POSTGRES_PASSWORD="your-db-password"
export GRAFANA_PASSWORD="admin"

# Start all services
cd deployments/docker
docker-compose up -d

# Verify services are running
docker-compose ps

# View logs
docker-compose logs -f backend
```

## Architecture

The Docker Compose stack includes:

```
┌─────────────┐
│   Backend   │ :8080  ← HTTP API
└──────┬──────┘
       │
   ┌───┴───┬────────┬────────┐
   │       │        │        │
┌──▼──┐ ┌─▼───┐ ┌──▼────┐ ┌─▼──────┐
│Redis│ │Postgres│ │Prometheus│ │Grafana│
│:6379│ │:5432  │ │:9090    │ │:3000  │
└─────┘ └───────┘ └─────────┘ └────────┘
```

## Services

### Backend

**Image**: Built from `backend/Dockerfile`
**Port**: `8080`
**Depends on**: Redis, Postgres

**Environment Variables**:
```yaml
environment:
  - PORT=8080
  - DEDUP_BACKEND=redis
  - REDIS_ADDR=redis:6379
  - TOKEN_RATE=100
  - METRICS_USER=ops
  - METRICS_PASS=${METRICS_PASS}
  - PCS_SIGN_ALG=hmac
  - PCS_HMAC_KEY=${PCS_HMAC_KEY}
  - WAL_DIR=/data/wal
```

**Volumes**:
- `backend-data:/data` - WAL and dedup snapshot

**Health Check**:
```yaml
healthcheck:
  test: ["CMD", "wget", "-q", "--spider", "http://localhost:8080/health"]
  interval: 10s
  timeout: 5s
  retries: 3
```

### Redis

**Image**: `redis:7-alpine`
**Port**: `6379`
**Persistence**: Append-only file (AOF)

**Volumes**:
- `redis-data:/data`

**Command**:
```bash
redis-server --appendonly yes
```

### PostgreSQL

**Image**: `postgres:16-alpine`
**Port**: `5432`
**Database**: `fractal_lba`

**Environment Variables**:
```yaml
environment:
  - POSTGRES_DB=fractal_lba
  - POSTGRES_USER=flk
  - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
```

**Volumes**:
- `postgres-data:/var/lib/postgresql/data`
- Init script: `init-db.sql` (creates tables)

**Schema**:
```sql
-- Dedup store
CREATE TABLE dedup_store (
    pcs_id VARCHAR(64) PRIMARY KEY,
    result JSONB NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    pcs_id VARCHAR(64) NOT NULL,
    shard_id VARCHAR(255),
    epoch INTEGER,
    accepted BOOLEAN,
    escalated BOOLEAN,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason TEXT
);
```

### Prometheus

**Image**: `prom/prometheus:latest`
**Port**: `9090`

**Configuration**: `observability/prometheus/prometheus.yml`

**Scrape Targets**:
- Backend: `http://backend:8080/metrics` (with Basic Auth)

**Volume**:
- `prometheus-data:/prometheus`

### Grafana

**Image**: `grafana/grafana:latest`
**Port**: `3000`

**Default Credentials**:
- Username: `admin`
- Password: `${GRAFANA_PASSWORD:-admin}`

**Provisioning**:
- Datasource: Prometheus (auto-configured)
- Dashboard: Fractal LBA dashboard (auto-imported)

**Volumes**:
- `grafana-data:/var/lib/grafana`
- Provisioning configs: `observability/grafana/provisioning`

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Signing
PCS_SIGN_ALG=hmac
PCS_HMAC_KEY=your-256-bit-secret-key

# Metrics auth
METRICS_USER=ops
METRICS_PASS=secure-metrics-password

# Postgres
POSTGRES_PASSWORD=secure-db-password

# Grafana
GRAFANA_PASSWORD=secure-grafana-password

# Dedup backend (memory, redis, or postgres)
DEDUP_BACKEND=redis

# Rate limiting
TOKEN_RATE=100
```

### Switching Dedup Backend

**Memory** (default for dev):
```yaml
environment:
  - DEDUP_BACKEND=memory
  - DEDUP_SNAPSHOT=/data/dedup.json
```

**Redis** (recommended):
```yaml
environment:
  - DEDUP_BACKEND=redis
  - REDIS_ADDR=redis:6379
```

**Postgres**:
```yaml
environment:
  - DEDUP_BACKEND=postgres
  - POSTGRES_CONN=host=postgres port=5432 user=flk password=${POSTGRES_PASSWORD} dbname=fractal_lba sslmode=disable
```

## Operations

### Starting Services

```bash
docker-compose up -d
```

### Stopping Services

```bash
docker-compose down
```

### Restarting a Service

```bash
docker-compose restart backend
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Scaling Backend

```bash
docker-compose up -d --scale backend=3
```

Note: Requires external load balancer (HAProxy, nginx) for distribution.

### Accessing Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Backend API | http://localhost:8080 | - |
| Health Check | http://localhost:8080/health | - |
| Metrics | http://localhost:8080/metrics | ops / ${METRICS_PASS} |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / ${GRAFANA_PASSWORD} |
| Redis CLI | `docker exec -it flk-redis redis-cli` | - |
| Postgres | `docker exec -it flk-postgres psql -U flk fractal_lba` | - |

## Backup & Restore

### Backup Volumes

```bash
# Backup Redis
docker exec flk-redis redis-cli SAVE
docker cp flk-redis:/data/dump.rdb ./backup/redis-dump-$(date +%Y%m%d).rdb

# Backup Postgres
docker exec flk-postgres pg_dump -U flk fractal_lba > ./backup/postgres-$(date +%Y%m%d).sql

# Backup backend data
docker cp flk-backend:/data ./backup/backend-data-$(date +%Y%m%d)
```

### Restore from Backup

```bash
# Stop services
docker-compose down

# Restore Redis
docker cp ./backup/redis-dump-YYYYMMDD.rdb flk-redis:/data/dump.rdb

# Restore Postgres
docker exec -i flk-postgres psql -U flk fractal_lba < ./backup/postgres-YYYYMMDD.sql

# Start services
docker-compose up -d
```

## Troubleshooting

### Backend Won't Start

**Check logs**:
```bash
docker-compose logs backend
```

**Common issues**:
- Missing `PCS_HMAC_KEY` (if signing enabled)
- Redis/Postgres not ready (healthcheck failing)
- Port 8080 already in use

**Solution**:
```bash
# Wait for dependencies
docker-compose up -d redis postgres
sleep 10
docker-compose up -d backend
```

### Redis Connection Errors

**Symptom**: Backend logs show "connection refused"

**Check Redis**:
```bash
docker exec flk-redis redis-cli ping
# Should return: PONG
```

**Restart Redis**:
```bash
docker-compose restart redis
```

### Postgres Connection Errors

**Check Postgres**:
```bash
docker exec flk-postgres pg_isready -U flk
# Should return: accepting connections
```

**View logs**:
```bash
docker-compose logs postgres
```

### Metrics Not Showing in Grafana

**Check Prometheus targets**:
1. Open http://localhost:9090/targets
2. Verify `backend` target is UP
3. Check Basic Auth credentials match

**Verify datasource**:
1. Open Grafana → Configuration → Datasources
2. Test Prometheus connection

### Disk Space Issues

**Check volume usage**:
```bash
docker system df -v
```

**Clean old logs**:
```bash
docker system prune -f
```

**WAL cleanup**:
```bash
# Backend WAL rotation
docker exec flk-backend sh -c "find /data/wal -name '*.wal' -mtime +14 -delete"
```

## Performance Tuning

### Redis Optimization

```yaml
redis:
  command: >
    redis-server
    --appendonly yes
    --maxmemory 2gb
    --maxmemory-policy allkeys-lru
```

### Postgres Tuning

```yaml
postgres:
  command: >
    postgres
    -c shared_buffers=256MB
    -c effective_cache_size=1GB
    -c work_mem=16MB
    -c maintenance_work_mem=64MB
```

### Backend Scaling

Horizontal scaling requires load balancer:

```yaml
# docker-compose.override.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
```

**nginx.conf**:
```nginx
upstream backend {
    least_conn;
    server backend:8080;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
    }
}
```

## Upgrading

### Rolling Update

```bash
# Pull latest images
docker-compose pull

# Recreate services (zero downtime with dependencies)
docker-compose up -d --no-deps backend
```

### Major Version Upgrade

```bash
# Backup first
./backup.sh

# Stop all services
docker-compose down

# Pull new version
git pull origin main

# Update and restart
docker-compose up -d
```

## Security Hardening

### Use Secrets

Store sensitive data in Docker secrets:

```bash
echo "your-secret-key" | docker secret create pcs_hmac_key -
```

Update compose:
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

### Network Isolation

```yaml
networks:
  frontend:
  backend:

services:
  backend:
    networks:
      - frontend
      - backend

  redis:
    networks:
      - backend
```

### Non-Root Containers

Already implemented in Dockerfile:

```dockerfile
USER appuser
```

## Production Considerations

Docker Compose is suitable for:
- ✅ Development
- ✅ Testing
- ✅ Small-scale production (<10k req/day)

For large-scale production, use [Kubernetes](kubernetes.md).

## Next Steps

- [Kubernetes Deployment](kubernetes.md) - Production-grade orchestration
- [Configuration Guide](configuration.md) - Detailed settings
- [Operations Guide](../operations/operations-guide.md) - Day-to-day management

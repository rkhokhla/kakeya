# Runbook: Tenant SLO Breach (Phase 3)

**Alert:** `FLKTenantSLOBreach`
**Severity:** P2 (High)
**Owner:** SRE, Backend Team

---

## 1. Symptoms

* Prometheus alert fires: `FLKTenantSLOBreach`
* Specific tenant's error rate exceeds SLO threshold (>2% escalations or >1% errors)
* Per-tenant metrics show degradation:
  * `rate(flk_escalated_by_tenant{tenant_id="X"}[5m]) > 0.02`
  * `rate(flk_signature_errors_by_tenant{tenant_id="X"}[5m]) > 0.01`

---

## 2. Impact

* **Tenant**: Specific tenant experiencing degraded service
* **Other tenants**: Isolated (multi-tenant isolation should prevent noisy neighbor)
* **Business**: Potential SLA violation, customer escalation

---

## 3. Immediate Triage (5 minutes)

### 3.1 Identify Affected Tenant

```bash
# Check Prometheus for which tenant is breaching SLO
kubectl -n fractal-lba logs -l app=backend --tail=100 | grep "tenant_id=" | sort | uniq -c

# Get tenant-specific metrics
curl -s http://prometheus:9090/api/v1/query?query='flk_escalated_by_tenant' | jq '.data.result[] | select(.value[1] | tonumber > 0) | .metric.tenant_id'
```

**Result:** Identify `tenant_id = TENANT_X`

### 3.2 Check Tenant Health Dashboard

* Navigate to Grafana → "Tenant Health" dashboard
* Select tenant: `TENANT_X`
* Observe:
  * Error rate trend (last 1h, 6h, 24h)
  * Latency (p95, p99)
  * Quota usage (rate limit hits, daily quota)
  * Signature failures

### 3.3 Check Recent Changes

```bash
# Check recent policy changes for tenant
kubectl -n fractal-lba get configmap policy-registry -o json | jq '.data | to_entries[] | select(.key | contains("TENANT_X"))'

# Check recent tenant config updates
git log --oneline --since="24 hours ago" -- infra/tenants/TENANT_X.yaml
```

---

## 4. Root Cause Analysis (15 minutes)

### Scenario A: Signature Failures Spike

**Symptom:**
```bash
flk_signature_errors_by_tenant{tenant_id="TENANT_X"} > 0.01
```

**Diagnosis:**
1. Agent using wrong signing key
2. Recent key rotation not synchronized
3. Clock skew causing timestamp issues (signature payload includes timestamps in some modes)

**Action:**
```bash
# Check tenant's signing configuration
kubectl -n fractal-lba get secret tenant-TENANT_X-signing -o json | jq '.data | keys'

# Check recent key rotation events
kubectl -n fractal-lba get events --field-selector involvedObject.kind=Secret,involvedObject.name=tenant-TENANT_X-signing

# Verify agent is using correct key (coordinate with tenant)
```

**Resolution:**
* If key mismatch: coordinate tenant to update agent config
* If key rotation incomplete: enable multi-key verification window (see [key-rotation.md](./key-rotation.md))
* If clock skew: agent must sync NTP

---

### Scenario B: Escalation Rate Spike (Data Quality)

**Symptom:**
```bash
flk_escalated_by_tenant{tenant_id="TENANT_X"} / flk_ingest_total_by_tenant{tenant_id="TENANT_X"} > 0.02
```

**Diagnosis:**
1. Tenant's data distribution changed (D̂, coh★ out of tolerance)
2. Policy thresholds too strict for tenant's use case
3. Upstream data corruption

**Action:**
```bash
# Sample recent escalations from WORM log
grep "tenant_id.*TENANT_X" /data/audit-worm/$(date +%Y/%m/%d)/*.jsonl | \
  jq 'select(.verify_outcome == "escalated") | {pcs_id, D_hat, coh_star, reason}' | head -20

# Check D̂ distribution for tenant
# (requires analytics pipeline - query from audit warehouse)

# Compare to policy thresholds
kubectl -n fractal-lba get configmap policy-TENANT_X -o yaml | grep -A 10 "tol_D"
```

**Resolution:**
* If data distribution shifted: discuss with tenant; may need policy adjustment
* If policy too strict: propose tolerance relaxation (requires approval + canary)
* If data corruption: investigate tenant's agent logs, check upstream systems

---

### Scenario C: Quota Exceeded (Rate Limit)

**Symptom:**
```bash
flk_quota_exceeded_by_tenant{tenant_id="TENANT_X"} > threshold
```

**Diagnosis:**
1. Tenant exceeded configured rate limit (requests/sec)
2. Burst traffic pattern
3. Attack or misconfigured retry loop

**Action:**
```bash
# Check tenant quota configuration
kubectl -n fractal-lba exec -it deploy/backend -- \
  curl localhost:8080/admin/tenants/TENANT_X | jq '.token_rate, .burst_rate, .daily_quota'

# Check request pattern
# (query logs for request timestamps, look for retry storms)
kubectl -n fractal-lba logs -l app=backend --since=1h | \
  grep "tenant_id=TENANT_X" | grep "quota exceeded" | wc -l

# Identify source IPs (if applicable)
kubectl -n fractal-lba logs -l app=backend --since=1h | \
  grep "tenant_id=TENANT_X" | awk '{print $NF}' | sort | uniq -c | sort -rn | head -10
```

**Resolution:**
* If legitimate burst: temporarily increase `burst_rate` (requires config update)
* If retry storm: coordinate with tenant to fix retry logic
* If attack: apply rate limit at ingress (Nginx/Envoy), block malicious IPs

---

### Scenario D: Per-Tenant Policy Misconfiguration

**Symptom:**
* SLO breach started immediately after policy rollout
* Other tenants unaffected

**Diagnosis:**
* Newly promoted policy version has incorrect thresholds for this tenant
* Feature flag misconfigured

**Action:**
```bash
# Check active policy version for tenant
kubectl -n fractal-lba logs -l app=backend --since=5m | \
  grep "tenant_id=TENANT_X" | grep "policy_version" | tail -1

# Compare with previous version
kubectl -n fractal-lba get configmap policy-registry -o json | \
  jq '.data | to_entries[] | select(.key | contains("v")) | {version: .key, created: .value.created_at}'

# Check feature flags
kubectl -n fractal-lba exec -it deploy/backend -- \
  curl localhost:8080/admin/flags | jq '.tenants.TENANT_X'
```

**Resolution:**
* Rollback to previous policy version (see [policy-rollback.md](./policy-rollback.md))
* Fix policy and re-promote with canary testing
* Update feature flags if misconfigured

---

## 5. Mitigation Actions

### Immediate (< 5 min)

1. **Notify tenant** (if customer-facing SLA breach)
   ```
   Subject: [P2] Service degradation for tenant TENANT_X
   Body: We are investigating elevated error rates for your tenant. ETA for resolution: 30 minutes.
   ```

2. **Enable degraded mode** (if applicable)
   ```bash
   # Disable strict checks for tenant temporarily
   kubectl -n fractal-lba patch configmap tenant-TENANT_X --type=merge -p '{"data":{"strict_mode":"false"}}'
   kubectl -n fractal-lba rollout restart deploy/backend
   ```

3. **Increase quota temporarily** (if rate limited)
   ```bash
   # Double rate limit (example)
   kubectl -n fractal-lba patch configmap tenant-TENANT_X --type=merge -p '{"data":{"token_rate":"200"}}'
   ```

### Short-term (< 30 min)

1. **Rollback policy** (if policy issue)
   * See [policy-rollback.md](./policy-rollback.md)

2. **Re-sync signing keys** (if signature issue)
   * Coordinate with tenant to verify agent config
   * Enable multi-key verification window if needed

3. **Analyze WORM logs** for patterns
   * Export last 1000 escalations to CSV
   * Share with data science team for tolerance tuning

---

## 6. Resolution & Verification

### Verify SLO Restored

```bash
# Check metrics (wait 5-10 minutes after mitigation)
curl -s "http://prometheus:9090/api/v1/query?query=rate(flk_escalated_by_tenant{tenant_id=\"TENANT_X\"}[5m])"

# Expected: rate < 0.02
```

### Update Grafana Dashboard

* Add annotation: "SLO breach - Root cause: [X] - Mitigation: [Y]"
* Update alert silence (if applicable)

### Notify Stakeholders

* Post-incident Slack update:
  ```
  ✓ RESOLVED: TENANT_X SLO breach
  Root cause: [Summary]
  Resolution: [Actions taken]
  Duration: [Time to resolve]
  Next steps: [Follow-up tasks]
  ```

---

## 7. Post-Incident Follow-up

1. **RCA Document** (within 48h)
   * Timeline of events
   * Root cause analysis
   * Actions taken
   * Preventive measures

2. **Policy Review** (if applicable)
   * Evaluate if tolerance thresholds are appropriate
   * Propose policy version update

3. **Tenant Communication**
   * Share RCA summary
   * Discuss preventive measures
   * Review SLA implications

4. **Monitoring Improvements**
   * Add more granular alerts (if gaps identified)
   * Update dashboard with new visualizations

---

## 8. Related Runbooks

* [policy-rollback.md](./policy-rollback.md) - Rollback policy changes
* [signature-spike.md](./signature-spike.md) - Signature verification failures
* [region-failover.md](./region-failover.md) - Multi-region failover

---

## 9. References

* Tenant isolation: `CLAUDE.md` § Multi-Tenant Architecture
* SLO thresholds: `CLAUDE_PHASE2.md` § Performance & SLOs
* Policy management: `CLAUDE_PHASE3.md` § WP4 Policy DSL

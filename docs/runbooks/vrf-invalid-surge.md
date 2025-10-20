# Runbook: VRF Invalid Proof Surge (Phase 3 WP6)

**Alert:** `FLKVRFInvalidSurge`
**Severity:** P1 (Critical - Potential Attack)
**Owner:** Security Team, SRE

---

## 1. Symptoms

* Prometheus alert fires: `FLKVRFInvalidSurge`
* High rate of VRF verification failures:
  * `rate(flk_vrf_invalid_total[5m]) > 10/s`
* Backend logs show repeated VRF errors:
  ```
  VRF verification failed for tenant X, PCS Y: invalid proof
  ```
* Potentially correlated with:
  * Signature failures
  * Anomaly score spikes
  * Specific tenant or IP range

---

## 2. Impact

**Security Risk:** HIGH
* Indicates potential adversarial attack to manipulate direction sampling
* Could be attempt to bias coherence computation
* May indicate compromised agent keys or man-in-the-middle attack

**Service Impact:**
* Affected requests rejected (401/202)
* Legitimate traffic may be blocked if false positives

---

## 3. Immediate Triage (3 minutes)

### 3.1 Assess Scope

```bash
# Count VRF failures in last 10 minutes
kubectl -n fractal-lba logs -l app=backend --since=10m | grep "VRF verification failed" | wc -l

# Identify affected tenants
kubectl -n fractal-lba logs -l app=backend --since=10m | \
  grep "VRF verification failed" | \
  grep -oP 'tenant_id=\K[^ ]+' | sort | uniq -c | sort -rn

# Identify source IPs (if logged)
kubectl -n fractal-lba logs -l app=backend --since=10m | \
  grep "VRF verification failed" | \
  awk '{print $NF}' | sort | uniq -c | sort -rn | head -10
```

**Decision Point:**
* **Single tenant affected** → Likely agent misconfiguration or compromised key
* **Multiple tenants affected** → Potential backend bug or coordinated attack
* **Single IP/CIDR** → Targeted attack, proceed to block

### 3.2 Check for Correlated Anomalies

```bash
# Check for signature failures spike (may indicate key compromise)
curl -s http://prometheus:9090/api/v1/query?query='rate(flk_signature_errors_by_tenant[5m]) > 0.1'

# Check anomaly scores
kubectl -n fractal-lba logs -l app=backend --since=10m | \
  grep "anomaly_score" | jq '.score' | awk '{sum+=$1; n++} END {print "Avg:", sum/n}'

# Check if escalations also spiking (data quality attack)
curl -s http://prometheus:9090/api/v1/query?query='rate(flk_escalated_by_tenant[5m]) > 0.05'
```

---

## 4. Root Cause Analysis (10 minutes)

### Scenario A: Agent Misconfiguration (Single Tenant)

**Symptoms:**
* One tenant accounts for >90% of VRF failures
* Started recently (< 24h ago)
* No signature failures

**Diagnosis:**
* Agent updated to version with broken VRF proof generation
* Agent using wrong VRF public key
* Clock skew affecting VRF proof timestamps

**Action:**
```bash
# Get failing PCS samples
kubectl -n fractal-lba logs -l app=backend --since=10m | \
  grep "VRF verification failed" | \
  grep "tenant_id=TENANT_X" | head -5 | jq '.pcs_id'

# Check WORM log for full PCS details
grep -E "$(kubectl logs ... | jq -r '.pcs_id' | paste -sd '|')" \
  /data/audit-worm/$(date +%Y/%m/%d)/*.jsonl | jq .

# Verify VRF public key configuration
kubectl -n fractal-lba get configmap tenant-TENANT_X-vrf-config -o yaml
```

**Resolution:**
1. Contact tenant to verify agent version and VRF configuration
2. Provide correct VRF public key if mismatch
3. Temporarily disable VRF for tenant if blocking legitimate traffic (report-only mode)

---

### Scenario B: Compromised Agent Key

**Symptoms:**
* VRF failures + signature failures from same tenant
* Unusual request patterns (rate spikes, odd timing)
* PCS data anomalies (extreme D̂, coh★ values)

**Diagnosis:**
* Agent private key leaked or compromised
* Attacker attempting to submit manipulated PCS with stolen key but invalid VRF

**Action:**
```bash
# Check signature + VRF correlation
kubectl -n fractal-lba logs -l app=backend --since=1h | \
  jq 'select(.tenant_id == "TENANT_X") | select(.vrf_valid == false or .sig_valid == false)'

# Check anomaly scores for tenant
kubectl -n fractal-lba logs -l app=backend --since=1h | \
  jq 'select(.tenant_id == "TENANT_X") | .anomaly_score' | \
  awk '{if($1 > 0.7) print}'

# Get source IPs for suspicious requests
kubectl -n fractal-lba logs -l app=backend --since=1h | \
  jq 'select(.tenant_id == "TENANT_X" and .vrf_valid == false) | .source_ip' | \
  sort | uniq -c
```

**Resolution:**
1. **URGENT:** Disable tenant's signing key immediately
   ```bash
   kubectl -n fractal-lba patch secret tenant-TENANT_X-signing \
     --type=json -p='[{"op": "add", "path": "/metadata/annotations/disabled", "value": "true"}]'
   kubectl -n fractal-lba rollout restart deploy/backend
   ```
2. Contact tenant security team immediately
3. Initiate key rotation procedure (see [key-rotation.md](./key-rotation.md))
4. Audit recent PCS submissions for tampering
5. Report to security incident response team

---

### Scenario C: Coordinated Attack (Multiple Tenants)

**Symptoms:**
* Multiple tenants affected simultaneously
* Started at specific time (suggests coordinated)
* May include DDoS component (rate spike)

**Diagnosis:**
* Adversarial attack to test system defenses
* Reconnaissance for vulnerability discovery
* Attempt to cause service degradation

**Action:**
```bash
# Identify attack pattern
kubectl -n fractal-lba logs -l app=backend --since=30m | \
  grep "VRF verification failed" | \
  awk '{print $1, $2}' | uniq -c | head -20

# Check for rate spike
curl -s http://prometheus:9090/api/v1/query?query='rate(flk_ingest_total[1m])'

# Identify source networks
kubectl -n fractal-lba logs -l app=backend --since=30m | \
  grep "VRF verification failed" | \
  awk '{print $NF}' | cut -d. -f1-3 | sort | uniq -c | sort -rn
```

**Resolution:**
1. **Block malicious IPs at ingress:**
   ```bash
   # Add to Nginx/Envoy deny list
   kubectl -n ingress-nginx edit configmap ingress-nginx-controller
   # Add: block-cidrs: "203.0.113.0/24,198.51.100.0/24"
   ```

2. **Enable strict mode for all tenants:**
   ```bash
   kubectl -n fractal-lba patch configmap global-security --type=merge \
     -p '{"data":{"vrf_strict_mode":"true","sanity_checks_strict":"true"}}'
   ```

3. **Notify security team:**
   * Provide attack timeline, source IPs, affected tenants
   * Request threat intelligence analysis

4. **Monitor for escalation:**
   * Watch for DDoS patterns
   * Check for lateral movement (other service endpoints)

---

### Scenario D: Backend Bug or Verifier Misconfiguration

**Symptoms:**
* All VRF proofs failing suddenly
* No pattern by tenant, IP, or time
* Recent backend deployment

**Diagnosis:**
* Bug in VRF verification code
* VRF public keys not loaded correctly
* Library version incompatibility

**Action:**
```bash
# Check recent deployments
kubectl -n fractal-lba rollout history deploy/backend

# Check backend logs for initialization errors
kubectl -n fractal-lba logs -l app=backend --since-time=$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) | \
  grep -i "vrf\|init\|config"

# Test VRF verification manually
kubectl -n fractal-lba exec -it deploy/backend -- \
  curl -X POST localhost:8080/internal/test/vrf \
  -H "Content-Type: application/json" \
  -d @/tmp/test-vrf-proof.json
```

**Resolution:**
1. **Rollback deployment if recent change:**
   ```bash
   kubectl -n fractal-lba rollout undo deploy/backend
   kubectl -n fractal-lba rollout status deploy/backend
   ```

2. **Fix configuration:**
   * Verify VRF public keys loaded correctly
   * Check environment variables: `VRF_ENABLED`, `VRF_PUBLIC_KEYS`

3. **Emergency bypass (last resort):**
   ```bash
   # Temporarily disable VRF verification (report-only mode)
   kubectl -n fractal-lba set env deploy/backend VRF_MODE=report_only
   ```
   **⚠️ WARNING:** This degrades security posture. Only use during critical outage.

---

## 5. Immediate Mitigation Actions

### Priority 1: Contain Attack Surface

1. **If single tenant compromised:**
   ```bash
   # Disable tenant
   kubectl -n fractal-lba patch configmap tenant-TENANT_X --type=merge \
     -p '{"data":{"active":"false"}}'
   ```

2. **If coordinated attack:**
   ```bash
   # Block source IPs at ingress (Nginx example)
   kubectl -n ingress-nginx exec -it deploy/ingress-nginx-controller -- \
     nginx -s reload  # After updating deny list
   ```

3. **If DDoS component:**
   ```bash
   # Tighten global rate limits
   kubectl -n fractal-lba set env deploy/backend GLOBAL_RATE_LIMIT=50
   ```

### Priority 2: Enable Enhanced Monitoring

```bash
# Increase log verbosity for VRF
kubectl -n fractal-lba set env deploy/backend LOG_LEVEL=debug VRF_LOG_VERBOSE=true

# Export recent VRF failures to incident bucket
kubectl -n fractal-lba logs -l app=backend --since=1h | \
  grep "VRF verification failed" > /tmp/vrf-failures-$(date +%s).log
gsutil cp /tmp/vrf-failures-*.log gs://incident-artifacts/vrf-surge-$(date +%Y%m%d)/
```

### Priority 3: Notify Stakeholders

**Security Team (Immediate):**
```
Subject: [P1] VRF Invalid Proof Surge - Potential Attack
Body:
- Rate: X failures/sec
- Affected tenants: [list]
- Source IPs: [list top 5]
- Mitigation: [actions taken]
- Investigation: [ongoing]
```

**Affected Tenants (If service impact):**
```
Subject: [P1] Service interruption - Security incident
Body:
We are responding to a security incident affecting PCS submissions.
Your service may experience temporary disruptions.
ETA for resolution: [estimate]
```

---

## 6. Resolution & Verification

### Verify Attack Stopped

```bash
# Check VRF failure rate (should drop to near-zero)
curl -s http://prometheus:9090/api/v1/query?query='rate(flk_vrf_invalid_total[5m])'

# Expected: < 0.1/s (baseline noise)

# Verify legitimate traffic flowing
curl -s http://prometheus:9090/api/v1/query?query='rate(flk_ingest_total[5m])'
# Should return to normal levels
```

### Re-enable Affected Tenants

```bash
# After key rotation complete
kubectl -n fractal-lba patch configmap tenant-TENANT_X --type=merge \
  -p '{"data":{"active":"true"}}'

# Monitor for recurrence
kubectl -n fractal-lba logs -l app=backend -f | grep "tenant_id=TENANT_X"
```

---

## 7. Post-Incident Follow-up

### Immediate (< 4h)

1. **Security Incident Report**
   * Timeline of attack
   * Attack vectors identified
   * Mitigation actions
   * Service impact assessment

2. **Forensic Analysis**
   * Export WORM logs for affected period
   * Analyze PCS data patterns
   * Check for data exfiltration attempts

3. **Key Rotation** (if compromised)
   * Follow [key-rotation.md](./key-rotation.md) for affected tenants
   * Verify new keys deployed correctly

### Short-term (< 7d)

1. **Enhanced Monitoring**
   * Add specific VRF failure rate alerts per tenant
   * Create dashboard for VRF health metrics
   * Implement automated IP blocking (if pattern detected)

2. **Code Review**
   * Audit VRF verification implementation
   * Add additional validation checks
   * Consider rate limiting VRF failures per tenant

3. **Tenant Communication**
   * Share incident details (sanitized)
   * Provide security best practices guide
   * Offer security assessment support

### Long-term (< 30d)

1. **Architecture Review**
   * Evaluate VRF proof requirements
   * Consider adding proof-of-work for high-risk operations
   * Implement anomaly-based circuit breakers

2. **Threat Model Update**
   * Document new attack vector
   * Update defenses and detection rules
   * Share with security community (if appropriate)

---

## 8. Related Runbooks

* [key-rotation.md](./key-rotation.md) - Signing key rotation
* [signature-spike.md](./signature-spike.md) - Signature failures
* [tenant-slo-breach.md](./tenant-slo-breach.md) - Tenant SLO issues

---

## 9. References

* VRF Specification: RFC 9381 (Verifiable Random Functions)
* Adversarial Robustness: `CLAUDE_PHASE3.md` § WP6
* Sanity Checks: `backend/internal/security/vrf.go`
* Anomaly Scoring: `backend/internal/security/vrf.go::AnomalyScorer`

# Runbook: Signature Verification Failures Spike

**Alert:** `FLKSignatureFailuresSpike`
**Severity:** Warning
**Component:** Backend signature verification

## Symptoms

- High rate of HTTP 401 responses (>10/s for 5 minutes)
- `flk_signature_err` counter increasing rapidly
- Users reporting "Signature verification failed" errors

## Impact

- Legitimate PCS submissions are being rejected
- Users cannot submit proof-of-computation summaries
- System may appear unavailable to clients

## Possible Causes

1. **Key Rotation In Progress** - New keys deployed but agents still using old keys
2. **Clock Skew** - System clock drift causing timestamp/nonce issues
3. **Canonicalization Drift** - Agent and backend disagree on canonical format
4. **Malicious Activity** - Attempted replay or tampering attacks
5. **Configuration Error** - Wrong HMAC/Ed25519 key configured
6. **Library Version Mismatch** - Agent using different signing library version

## Diagnostic Steps

###  1 Check Recent Deployments

```bash
# Check if key rotation happened recently
kubectl get events -n fractal-lba --sort-by='.lastTimestamp' | grep -i secret

# Check backend pod restarts
kubectl get pods -n fractal-lba -l app=fractal-lba-backend
```

### 2. Inspect Error Logs

```bash
# Get recent signature verification errors
kubectl logs -n fractal-lba -l app=fractal-lba-backend --tail=100 | grep "Signature verification failed"

# Look for patterns in failed pcs_ids
kubectl logs -n fractal-lba -l app=fractal-lba-backend --tail=1000 | grep "pcs_id" | grep "401"
```

### 3. Check Key Configuration

```bash
# Verify HMAC key is correctly set (first 8 chars only for safety)
kubectl exec -n fractal-lba deployment/fractal-lba-backend -- \
  sh -c 'echo $PCS_HMAC_KEY | cut -c1-8'

# Check signing algorithm
kubectl exec -n fractal-lba deployment/fractal-lba-backend -- \
  sh -c 'echo $PCS_SIGN_ALG'
```

### 4. Test with Golden PCS

```bash
# Submit known-good golden PCS
curl -X POST https://api.fractal-lba.example.com/v1/pcs/submit \
  -H "Content-Type: application/json" \
  -d @tests/golden/pcs_tiny_case_1.json

# Should return 200/202, not 401
```

### 5. Check Clock Skew

```bash
# Check backend system time
kubectl exec -n fractal-lba deployment/fractal-lba-backend -- date -u

# Compare with NTP server
ntpdate -q pool.ntp.org
```

## Resolution

### Scenario A: Key Rotation In Progress

**Action:** Implement key overlap period

```bash
# Backend should support both old and new keys during overlap window
# Update backend to accept both keys:

kubectl create secret generic pcs-hmac-keys-multi \
  --from-literal=PCS_HMAC_KEY_OLD="old-key-here" \
  --from-literal=PCS_HMAC_KEY_NEW="new-key-here"

# Update deployment to use multi-key verification
# See: docs/operations/key-rotation.md
```

**Timeline:** 14 days (dedup TTL window)

### Scenario B: Clock Skew

**Action:** Sync clocks with NTP

```bash
# Enable NTP on all nodes
kubectl label nodes --all ntp=enabled

# Verify time synchronization
for pod in $(kubectl get pods -n fractal-lba -l app=fractal-lba-backend -o name); do
  kubectl exec -n fractal-lba $pod -- date -u
done
```

### Scenario C: Canonicalization Drift

**Action:** Run golden file verification

```bash
# Test Python agent canonicalization
python -m pytest tests/test_signing.py::TestGoldenPCSVerification -v

# Test Go backend canonicalization
cd backend && go test ./internal/signing/... -v
```

If tests fail:
1. **DO NOT** change canonicalization (breaks existing signatures)
2. Document as breaking change
3. Plan major version bump
4. Coordinate agent and backend upgrades

### Scenario D: Malicious Activity

**Action:** Enable rate limiting and investigate

```bash
# Increase rate limiting temporarily
kubectl set env deployment/fractal-lba-backend TOKEN_RATE=50 -n fractal-lba

# Analyze failed submission patterns
kubectl logs -n fractal-lba -l app=fractal-lba-backend | \
  grep "401" | awk '{print $NF}' | sort | uniq -c | sort -rn | head -20

# If specific IP/agent pattern detected, add to network policy
```

### Scenario E: Configuration Error

**Action:** Rollback to last known good configuration

```bash
# Rollback deployment
kubectl rollout undo deployment/fractal-lba-backend -n fractal-lba

# Verify rollback succeeded
kubectl rollout status deployment/fractal-lba-backend -n fractal-lba

# Check if alert clears
curl https://prometheus.example.com/api/v1/alerts | jq '.data.alerts[] | select(.labels.alertname=="FLKSignatureFailuresSpike")'
```

## Prevention

1. **Key Rotation Procedure**
   - Always use overlap period (min 14 days)
   - Test in staging first
   - Monitor signature errors during rollout
   - Have rollback plan ready

2. **Clock Synchronization**
   - Ensure NTP is enabled on all nodes
   - Monitor clock skew with `node_time_drift_seconds`
   - Alert on drift >1 second

3. **CI/CD Gates**
   - Run golden file verification in CI
   - Fail build if canonicalization changes detected
   - Require explicit approval for signature-related changes

4. **Agent Version Management**
   - Maintain backward compatibility for 3 versions
   - Deprecate old signature algorithms with 90-day notice
   - Document breaking changes prominently

## Communication Template

**Subject:** [P2] Signature Verification Failures - Investigating

**Body:**
```
We are currently investigating elevated signature verification failures
affecting PCS submissions.

Status: [In Progress/Resolved]
Impact: Users may see 401 errors when submitting PCS
Timeline: Investigating - ETA 30 minutes
Next Update: [Timestamp]

Actions Taken:
- [List steps from Resolution section]

Mitigation:
- [If applicable: "Rolled back to previous version"]
- [If applicable: "Enabled key overlap period"]

We apologize for the inconvenience. Updates will be posted here.
```

## Related Runbooks

- [Key Rotation Procedure](./key-rotation.md)
- [Error Budget Burn](./error-budget-burn.md)
- [Dedup Outage](./dedup-outage.md)

## Post-Incident

1. **Root Cause Analysis**
   - Document what triggered the alert
   - Identify gaps in monitoring/testing
   - Propose improvements

2. **Metrics Review**
   - Analyze signature failure patterns
   - Check if SLO was breached
   - Update error budget tracking

3. **Documentation Updates**
   - Update this runbook if new scenarios discovered
   - Share learnings with team
   - Update key rotation procedures if needed

---

**Last Updated:** 2025-01-20
**Owner:** SRE Team
**Escalation:** Page on-call if unresolved after 30 minutes

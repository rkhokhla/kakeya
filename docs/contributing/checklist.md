# Contributing Checklist

Use this checklist before submitting a pull request to ensure your contribution meets project standards.

## Pre-Development

- [ ] Read [Contributing Guide](./guide.md)
- [ ] Read [Architecture Overview](../architecture/overview.md)
- [ ] Read [Invariants](../architecture/invariants.md)
- [ ] Check for existing similar features/fixes
- [ ] Open GitHub Discussion for large features
- [ ] Fork and clone repository
- [ ] Create feature branch

## Development

### Code Quality

- [ ] Code follows style guide (gofmt for Go, PEP 8 for Python)
- [ ] Meaningful variable and function names
- [ ] Comments for non-obvious logic
- [ ] No hardcoded secrets or credentials
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate (no sensitive data)

### Testing

- [ ] Unit tests added for new code
- [ ] Unit tests pass locally: `pytest tests/` or `go test ./...`
- [ ] Integration tests added (if applicable)
- [ ] E2E tests added (if user-facing feature)
- [ ] Test coverage ≥80% for new code
- [ ] Edge cases covered (empty inputs, boundaries, errors)

### Safety Invariants (CRITICAL)

- [ ] **WAL-first write ordering** preserved
- [ ] **Verify-before-dedup contract** preserved
- [ ] **First-write wins idempotency** preserved
- [ ] **Canonical signing** unchanged (or coordinated rollout plan)
- [ ] **pcs_id formula** unchanged
- [ ] No changes to signature subset without version bump

### Documentation

- [ ] README.md updated (if user-facing)
- [ ] CLAUDE.md updated (if invariants changed)
- [ ] Relevant docs/ files updated
- [ ] API documentation updated (if API changed)
- [ ] Changelog entry added (`docs/roadmap/changelog.md`)
- [ ] Comments/docstrings for public functions

### Observability (if applicable)

- [ ] Prometheus metrics added for new features
- [ ] Grafana dashboard updated
- [ ] Alerts added for failure scenarios
- [ ] Runbook created for operational features
- [ ] SLOs defined (if new critical path)

### Security

- [ ] No secrets in code or logs
- [ ] Input validation for all external inputs
- [ ] Authentication/authorization preserved
- [ ] TLS/mTLS configuration correct
- [ ] PII handling follows privacy controls

### Backward Compatibility

- [ ] Existing APIs unchanged (or deprecated gracefully)
- [ ] Database migrations backward compatible
- [ ] Configuration changes have defaults
- [ ] Old clients continue to work

## Pre-Submission

### Local Testing

- [ ] All unit tests pass: `pytest tests/ -v` or `go test ./... -v`
- [ ] E2E tests pass: `pytest tests/e2e/ -v`
- [ ] Linting passes: `ruff check .` or `golangci-lint run`
- [ ] Formatting applied: `black .` or `gofmt -w .`
- [ ] No security scan issues: `trufflehog filesystem .`

### Git

- [ ] Branch rebased on latest `main`
- [ ] Commits squashed (if many small commits)
- [ ] Commit messages follow conventional commits
- [ ] No merge conflicts

### PR Description

- [ ] Title follows format: `type(scope): description`
- [ ] Summary explains what and why
- [ ] Motivation section present
- [ ] Changes listed in bullets
- [ ] Testing approach described
- [ ] Screenshots included (if UI change)
- [ ] References issue number (if applicable)

## Submission

- [ ] Pull request created
- [ ] CI checks triggered
- [ ] Linked to relevant issue (if applicable)
- [ ] Requested review from maintainers

## Post-Submission

- [ ] Address reviewer feedback promptly
- [ ] Add new commits (don't force-push during review)
- [ ] Re-request review after changes
- [ ] Thank reviewers for their time

## Checklist for Specific Change Types

### Adding a New Metric

- [ ] Metric defined in `backend/internal/metrics/metrics.go`
- [ ] Metric documented in `docs/observability/dashboards.md`
- [ ] Alert added (if failure condition)
- [ ] Grafana panel added to relevant dashboard
- [ ] Tested with Prometheus query

### Adding a New API Endpoint

- [ ] OpenAPI spec updated (`api/openapi.yaml`)
- [ ] SDK updated (Python/Go/TypeScript)
- [ ] Rate limiting considered
- [ ] Authentication/authorization enforced
- [ ] Input validation implemented
- [ ] Error responses documented
- [ ] E2E test added

### Adding a New CRD (Kubernetes Operator)

- [ ] CRD defined in `operator/api/v1/`
- [ ] Controller implemented
- [ ] Reconciliation logic tested
- [ ] Helm chart updated with CRD
- [ ] Example manifests provided
- [ ] Operator documentation updated

### Adding a New Runbook

- [ ] Template followed (Symptoms, Triage, Resolution, Prevention)
- [ ] Linked from relevant alert
- [ ] Command examples tested
- [ ] Decision trees included (if complex)
- [ ] Communication templates provided

### Schema/Protocol Change (MAJOR)

- [ ] **Architecture review required**
- [ ] Version bump planned (MAJOR)
- [ ] Migration plan documented
- [ ] Backward compatibility period defined
- [ ] Formal verification updated (TLA+/Coq if applicable)
- [ ] All stakeholders notified

## Quick Reference

**Safety Invariants:** [docs/architecture/invariants.md](../architecture/invariants.md)

**Phase Reports:**
- Phase 1-8: `PHASE{1-8}_REPORT.md` in repo root

**Key Files:**
- `CLAUDE.md`: Project memory, don't break without review
- `README.md`: First impression for new users
- `docs/roadmap/changelog.md`: Keep updated

**Tests:**
- Unit: `tests/` (33 Phase 1 + more in later phases)
- E2E: `tests/e2e/` (15 integration + 5 geo-DR + 6 chaos)
- SDK: `tests/sdk/` (golden vectors)

**Contact:**
- Questions: GitHub Discussions
- Bugs: GitHub Issues
- Security: See `docs/security/overview.md`

---

## Final Check

Before clicking "Create Pull Request":

✅ I have reviewed this checklist
✅ All applicable items are checked
✅ I understand this will be reviewed by maintainers
✅ I am prepared to address feedback

**Thank you for contributing to Fractal LBA!**

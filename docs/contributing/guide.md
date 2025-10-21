# Contributing Guide

## Welcome!

Thank you for your interest in contributing to the Fractal LBA verification layer. This guide will help you get started.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and considerate
- Focus on constructive feedback
- Assume good intentions

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/fractal-lba.git
cd fractal-lba
```

### 2. Set Up Development Environment

**Backend (Go):**
```bash
cd backend
go mod download
go test ./...
```

**Agent (Python):**
```bash
cd agent
pip install -e .
pytest tests/
```

**Local Stack:**
```bash
docker-compose up -d
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## Development Workflow

### Making Changes

1. **Write tests first** (TDD encouraged)
2. Implement feature/fix
3. Run tests locally
4. Update documentation
5. Commit with clear messages

### Commit Messages

Follow conventional commits:

```
type(scope): subject

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding/updating tests
- `refactor`: Code change that neither fixes bug nor adds feature
- `perf`: Performance improvement
- `chore`: Build/tooling changes

**Examples:**
```
feat(hrs): add VAE-based anomaly detector

Implements Phase 8 WP4 anomaly detection v2 with VAE model,
semantic clustering, and auto-thresholding.

Closes #123
```

### Running Tests

**Unit Tests:**
```bash
# Python
pytest tests/ -v

# Go
go test ./... -v
```

**E2E Tests:**
```bash
docker-compose -f infra/compose-tests.yml up -d
pytest tests/e2e/ -v
docker-compose -f infra/compose-tests.yml down -v
```

**Linting:**
```bash
# Python
ruff check .
black --check .

# Go
golangci-lint run
```

## Design Principles

### Safety Invariants (DO NOT BREAK)

1. **WAL-first write ordering** (Phase 1)
2. **Verify-before-dedup contract** (Phase 1)
3. **First-write wins idempotency** (Phase 1)
4. **Canonical signing (8-field subset, 9-decimal rounding)** (Phase 1)

See [Invariants](../architecture/invariants.md) for complete list.

### Code Style

**Go:**
- Follow `gofmt` and `goimports`
- Use meaningful variable names (no single-letter except loops)
- Add comments for exported functions
- Keep functions small (<50 lines)

**Python:**
- Follow PEP 8
- Use type hints
- Docstrings for all public functions (Google style)
- Prefer composition over inheritance

### Adding New Features

**Before implementing:**
1. Check if similar feature exists
2. Open GitHub Discussion for large features
3. Review [Architecture Overview](../architecture/overview.md)
4. Ensure backward compatibility

**Implementation checklist:**
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests (if applicable)
- [ ] Documentation (README, docs/, CHANGELOG)
- [ ] Metrics/alerts (if observable)
- [ ] Runbook (if operational impact)
- [ ] Update CLAUDE.md (if changes invariants)

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass locally
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] Branch rebased on latest main

### 2. Submit PR

**Title:** `type(scope): clear description`

**Description template:**
```markdown
## Summary
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Bullet list of key changes

## Testing
How was this tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Backward compatible
- [ ] Invariants preserved
```

### 3. Review Process

- Maintainers will review within 2 business days
- Address feedback with new commits (don't force-push)
- CI must pass (all tests, linting, security scan)

### 4. Merging

- Maintainers will squash and merge
- Your commits will be attributed in merge commit

## Common Scenarios

### Fixing a Bug

1. Open issue describing bug
2. Add failing test reproducing bug
3. Fix bug
4. Verify test passes
5. Submit PR referencing issue

### Adding a Metric

```go
// 1. Define metric in backend/internal/metrics/metrics.go
var MyMetric = promauto.NewCounter(prometheus.CounterOpts{
    Name: "flk_my_metric_total",
    Help: "Description of metric",
})

// 2. Increment where needed
MyMetric.Inc()

// 3. Add alert in observability/prometheus/alerts.yml
- alert: MyMetricHigh
  expr: rate(flk_my_metric_total[5m]) > 100

// 4. Update docs/observability/dashboards.md
```

### Adding a Runbook

1. Create `docs/runbooks/my-incident.md`
2. Use template:
   - Symptoms
   - Triage (< 5 min)
   - Resolution steps
   - Prevention
3. Link from alert annotations

## Resources

- [Architecture Overview](../architecture/overview.md)
- [Testing Guide](../testing/e2e.md)
- [Deployment Guide](../deploy/helm.md)
- [CLAUDE.md](../../CLAUDE.md) (Project invariants)

## Getting Help

- GitHub Discussions: General questions
- GitHub Issues: Bug reports, feature requests
- Code review: Tag `@maintainers` in PR

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

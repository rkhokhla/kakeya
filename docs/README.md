# Fractal LBA + Kakeya FT Stack - Documentation

Welcome to the comprehensive documentation for the Fractal LBA + Kakeya FT Stack.

## Documentation Structure

### Getting Started
- [Quick Start Guide](../README.md#quick-start) - Get up and running in 5 minutes
- [Architecture Overview](architecture/overview.md) - Understand the system design
- [Core Concepts](architecture/concepts.md) - PCS, signals, and mathematical foundations

### Deployment
- [Docker Compose Deployment](deployment/docker-compose.md) - Local and development environments
- [Kubernetes Deployment](deployment/kubernetes.md) - Production deployment with Helm
- [Configuration Guide](deployment/configuration.md) - Environment variables and settings
- [Scaling Guide](deployment/scaling.md) - Performance tuning and capacity planning

### API Reference
- [REST API](api/rest-api.md) - HTTP endpoints and request/response formats
- [PCS Schema](api/pcs-schema.md) - Detailed field documentation
- [Error Codes](api/error-codes.md) - HTTP status codes and error handling
- [Client Libraries](api/client-libraries.md) - SDKs and integration examples

### Operations
- [Operations Guide](operations/operations-guide.md) - Day-to-day management
- [Monitoring & Alerting](operations/monitoring.md) - Prometheus metrics and Grafana dashboards
- [Troubleshooting](operations/troubleshooting.md) - Common issues and solutions
- [Runbooks](operations/runbooks.md) - Incident response procedures
- [Backup & Recovery](operations/backup-recovery.md) - Data durability strategies

### Development
- [Development Setup](development/setup.md) - Local environment configuration
- [Contributing Guide](development/contributing.md) - Code standards and workflow
- [Testing Guide](development/testing.md) - Unit, integration, and load testing
- [Building from Source](development/building.md) - Compilation and packaging

### Security
- [Security Overview](security/overview.md) - Threat model and defenses
- [TLS/mTLS Configuration](security/tls-mtls.md) - Transport encryption
- [Signature Verification](security/signing.md) - HMAC and Ed25519 implementation
- [Secrets Management](security/secrets.md) - Key rotation and storage
- [Security Hardening](security/hardening.md) - Production best practices
- [Audit Logging](security/audit-logging.md) - Compliance and forensics

### Architecture Deep-Dive
- [System Architecture](architecture/system-architecture.md) - Component interactions
- [Data Flow](architecture/data-flow.md) - Request lifecycle
- [Fault Tolerance](architecture/fault-tolerance.md) - WAL, deduplication, retry logic
- [Signal Computation](architecture/signal-computation.md) - Mathematical background
- [Design Decisions](architecture/design-decisions.md) - Why we built it this way

### Advanced Topics
- [Performance Tuning](operations/performance.md) - Optimization strategies
- [Multi-Region Deployment](deployment/multi-region.md) - Geographic distribution
- [Custom Metrics](operations/custom-metrics.md) - Extending observability
- [DSL Integration](architecture/dsl.md) - Policy configuration language

## Quick Links

- **Main README**: [../README.md](../README.md)
- **Technical Specification**: [../CLAUDE.md](../CLAUDE.md)
- **GitHub Repository**: [rkhokhla/kakeya](https://github.com/rkhokhla/kakeya)
- **Issue Tracker**: [GitHub Issues](https://github.com/rkhokhla/kakeya/issues)

## Getting Help

- **Documentation Issues**: File on [GitHub](https://github.com/rkhokhla/kakeya/issues) with label `documentation`
- **Questions**: Open a discussion on GitHub Discussions
- **Security Issues**: See [security/overview.md](security/overview.md) for responsible disclosure

## Contributing to Documentation

We welcome documentation improvements! See [development/contributing.md](development/contributing.md) for guidelines.

Key principles:
- Keep examples runnable and tested
- Include both success and failure scenarios
- Link to related concepts
- Update CLAUDE.md for design changes

---

**Document Version**: 0.1.0
**Last Updated**: 2025-01-19

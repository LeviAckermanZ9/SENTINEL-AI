# Security Policy — SENTINEL-AI

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within SENTINEL-AI, please report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Email: [sentinel-security@example.com](mailto:sentinel-security@example.com)
3. Include a detailed description, reproduction steps, and potential impact.
4. We will acknowledge receipt within 48 hours and provide a fix timeline within 5 business days.

## Security Architecture

### Authentication & Authorization
- All endpoints except `/health` and `/metrics` require **JWT Bearer token** authentication.
- JWT tokens expire after **30 minutes** (configurable via `JWT_EXPIRE_MINUTES` environment variable).
- Tokens are signed using HS256 algorithm with a server-side secret key.
- Password hashing uses **bcrypt** via the `passlib` library.

### API Security
- **CORS**: Restricted to known origins only (`localhost:8501`, `localhost:3000`).
- **Request body limit**: Maximum 5,000 characters per request to prevent abuse.
- **Rate limiting**: Recommended to configure at the reverse-proxy/ingress level.
- **Input validation**: All inputs validated via Pydantic models with strict constraints.

### Container Security
- **Non-root execution**: All containers run as a dedicated `sentinel` user (UID 1001).
- **Multi-stage Docker build**: Production image contains only runtime dependencies.
- **No secrets in images**: All sensitive configuration via environment variables.
- **Health checks**: Built-in Docker HEALTHCHECK for container orchestration.

### Data Privacy & GDPR Compliance
- **No PII storage**: All text processing is performed in-memory and is stateless.
- **No user tracking**: The platform does not collect, store, or transmit any user data.
- **Knowledge base**: Contains only publicly available fact-check articles from LIAR, FEVER, and Snopes datasets.
- **Ephemeral processing**: Input text is discarded after the response is returned.
- **Right to erasure**: Not applicable — no personal data is retained.

### Supply Chain Security
- **Dependabot**: Automated weekly dependency updates for `pip`, `docker`, and `github-actions`.
- **CodeQL SAST**: Static Application Security Testing on every pull request and weekly scheduled scans using GitHub's `security-extended` query suite.
- **Safety check**: `safety check` runs in CI to detect known vulnerabilities in Python dependencies.
- **Bandit**: Static security linter scans all source code for common Python security issues.

### Infrastructure Security
- **Terraform**: AWS resources provisioned with least-privilege IAM policies.
- **S3 buckets**: Versioning enabled, no public access.
- **Security Groups**: Restrict inbound traffic to necessary ports only.
- **Kubernetes**: Network policies and resource limits enforced.

### Secrets Management
- **Environment variables**: All secrets passed via `.env` file (never committed to version control).
- **`.env.example`**: Template provided with placeholder values — no real secrets.
- **GitHub Secrets**: CI/CD pipelines use GitHub's encrypted secrets for deployment credentials.
- **ChromaDB**: Token-based authentication for vector store access.

## Security Checklist

- [x] JWT Bearer authentication on protected endpoints
- [x] CORS restricted to known origins
- [x] Request body size limits
- [x] Non-root Docker containers
- [x] Multi-stage Docker builds
- [x] Dependabot automated updates
- [x] CodeQL SAST scanning
- [x] Bandit security linting in CI
- [x] Safety dependency vulnerability checks
- [x] No PII storage or retention
- [x] Least-privilege IAM policies
- [x] Encrypted secrets management
- [x] Health check endpoints (unauthenticated for orchestration)

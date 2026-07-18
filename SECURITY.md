# Security policy

## Supported version

Security fixes are applied to the current default branch. This demonstration project is not represented as production-ready.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository:

<https://github.com/KripaMishra/cuda-documentation-rag/security/advisories/new>

If private reporting is unavailable, contact the repository owner through their GitHub profile and request a private channel. Do not publish exploit details or credentials in an issue.

Include affected versions, reproduction steps, impact, and any suggested mitigation. Remove API keys, tokens, proprietary data, and other secrets from reports.

## Operator responsibilities

- Keep `GEMINI_API_KEY` outside source control.
- Place public deployments behind an authenticated, rate-limited TLS reverse proxy.
- Do not expose Milvus, Elasticsearch, or etcd ports to the public internet.
- Review crawl permissions and data provenance before ingesting content.
- Treat retrieved documents as untrusted input.

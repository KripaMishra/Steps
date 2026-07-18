# Deployment notes

The bundled Compose configuration is for reproducible self-hosting, not an internet-ready production stack.

## Network boundary

Milvus and Elasticsearch are attached only to the internal `backend` network and have no published host ports. Keep that boundary. Expose only the Streamlit service through a reverse proxy that provides TLS, authentication, request/body limits, rate limiting, and security headers.

## Secrets

Pass `GEMINI_API_KEY` through an environment/secret manager. Do not bake it into an image, Compose file, log, result export, or screenshot. Rotate a key immediately if it appears in Git history or output.

## Operations

- Pin and periodically review container image versions.
- Back up volumes only when their data provenance and retention policy are known.
- Monitor Gemini quota/cost and application error rates outside the app.
- Treat retrieved text as untrusted input.
- Use the configured query, context, token, and timeout bounds.
- Run `docker compose config` before deployment changes.

The app has no built-in user accounts, tenant isolation, billing, or abuse controls. Add those at the platform boundary only if a real deployment requires them.

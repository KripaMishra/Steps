# Contributing

Thanks for improving CUDA Documentation Copilot.

## Development setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
make evaluate
```

Demo mode must remain usable without an API key, model downloads, Milvus, or Elasticsearch. Full-mode changes must preserve the existing Milvus schema, Elasticsearch mapping, and hybrid ranking behavior unless a change is explicitly discussed first.

## Pull requests

- Keep changes focused and add tests for new pure behavior.
- Run `make test`, `make evaluate`, and `git diff --check`.
- Never commit credentials, `.env`, model caches, crawl output, runtime results, or database volumes.
- Do not add copied NVIDIA documentation. Fixture changes must be original summaries with source and provenance metadata; see [DATA_PROVENANCE.md](DATA_PROVENANCE.md).
- Do not claim accuracy, latency, cost, or production readiness without reproducible evidence.

Use a concise conventional commit subject such as `feat(demo): add fixture retrieval`.

## Reporting issues

Include the mode (`demo` or `full`), Python version, command used, and a sanitized error summary. Remove API keys, tokens, local paths, and document content that you cannot redistribute.

Security issues should follow [SECURITY.md](SECURITY.md), not a public issue.

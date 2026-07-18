# CUDA Documentation Copilot

A citation-first CUDA documentation demo with two deliberately separate paths:

- **Demo mode (default):** a small, provenance-labeled fixture corpus, deterministic local retrieval, and an offline extractive answer when no Gemini key is present.
- **Full mode:** the existing DPR + T5, Milvus, Elasticsearch, and Gemini pipeline for operators who want to crawl and index their own permitted corpus.

The project does not claim production readiness or measured answer accuracy. Every answer is expected to expose its supporting sources; unsupported questions return an explicit insufficient-context response.

## Run the demo

Python 3.11 or newer is required.

```bash
git clone https://github.com/KripaMishra/cuda-documentation-rag.git
cd cuda-documentation-rag
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
make demo
```

Open <http://localhost:8501>. No API key, database, GPU model, or crawl is needed.

Try these scripted questions:

1. `How do CUDA streams allow work to overlap?`
2. `What factors can limit CUDA occupancy?`
3. `Why is pinned host memory used for asynchronous copies?`

CLI:

```bash
RAG_MODE=demo python -m components.s5_rag_llm \
  "How do CUDA streams allow work to overlap?" --top_k 3
```

The CLI prints JSON and writes nothing unless `--file_path result/answer.json` is supplied.

## What the response contains

Both modes return the same JSON contract:

```text
query, answer, sources[], mode, latency,
retrieval_metadata, model, corpus_version, insufficient_context
```

Each source includes its URL, title, section, chunk/document ID, retrieval score, excerpt, source date, and provenance. Latency values are milliseconds measured in the current process; they are diagnostics, not a benchmark. The Streamlit UI shows citation cards and keeps retrieval details in a collapsed trace.

## Generation behavior

Demo mode uses Gemini through Google's OpenAI-compatible endpoint when `GEMINI_API_KEY` is configured. Without a key, it returns a clearly labeled deterministic extract from the retrieved project-authored summaries. Full mode requires Gemini configuration.

Gemini is an external paid/quota-limited service. Review current Google AI pricing, quotas, data-use terms, and regional availability before enabling it. Query, context, generation-token, and request-timeout limits are configurable.

## Architecture

```text
                         ┌─ demo fixture → local TF-IDF retrieval ─┐
question → mode router ──┤                                        ├→ response contract → CLI / Streamlit
                         └─ T5 expansion → BM25 + DPR/Milvus ──────┘
                                                    │
                                  Gemini if configured; demo otherwise uses extractive fallback
```

Demo mode does not import or initialize Torch, Transformers, Milvus, or Elasticsearch. Full mode retains the existing Milvus fields (`id`, `embedding`, `content`), Elasticsearch content search, and hybrid score formula. Provenance metadata is added to Elasticsearch and merged into the response after retrieval.

## Docker

Default demo:

```bash
docker compose up --build app
```

Full profile (requires `GEMINI_API_KEY` and operator-managed ingestion):

```bash
docker compose --profile full up --build app-full
```

The demo is served on port 8501 and full mode on 8502. Milvus and Elasticsearch have no host-published ports and share an internal Compose network. For public exposure, place the app behind an authenticated, rate-limited TLS reverse proxy; see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Configuration

Copy `.env.example` to `.env` for local values. `.env` is ignored; never commit credentials.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_MODE` | `demo` | `demo` or `full` |
| `GEMINI_API_KEY` | unset | Enables Gemini in demo; required in full |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `GEMINI_BASE_URL` | Google compatibility URL | API endpoint |
| `RAG_QUERY_MAX_CHARS` | `500` | Maximum normalized query length |
| `RAG_TOP_K_MAX` | `10` | Maximum requested sources |
| `RAG_CONTEXT_MAX_CHARS` | `12000` | Maximum generation context |
| `RAG_GENERATION_MAX_TOKENS` | `512` | Gemini output limit |
| `RAG_REQUEST_TIMEOUT_SECONDS` | `60` | Gemini request timeout |
| `RAG_DEMO_FIXTURE_PATH` | `demo/fixtures/cuda_docs.json` | Demo corpus path |
| `MILVUS_*`, `ELASTICSEARCH_*` | local defaults | Full-mode stores and index |

See [.env.example](.env.example) for every variable.

## Full crawl-to-index mode

Full mode downloads large DPR/T5 models and requires Milvus, Elasticsearch, and a Gemini key. Read [docs/FULL_MODE.md](docs/FULL_MODE.md) before crawling or ingesting any data. Generated crawl and result files are ignored and are not distributed.

## Tests and evaluation

```bash
python -m pip install -r requirements-demo.txt pytest
make test
make lint
make evaluate
```

`make evaluate` runs six offline factual, ambiguous, and unanswerable questions. Its reported rates describe only that tiny checked fixture set and must not be generalized to CUDA documentation quality.

## Data, limitations, and safety

The demo fixture contains 12 original project-authored summaries with links to NVIDIA pages. It does not redistribute an NVIDIA documentation dataset or relicense linked NVIDIA material. Read [DATA_PROVENANCE.md](DATA_PROVENANCE.md) before modifying the corpus.

Known limitations:

- The demo corpus is intentionally small and cannot answer broad CUDA questions.
- Lexical retrieval does not understand all synonyms.
- Gemini adds external cost, quota, availability, and privacy considerations.
- Full mode is operationally heavy and has not been represented as production-ready.
- Public deployments need authentication/rate limiting at a reverse proxy.

## Troubleshooting

- **No sources:** ask one of the scripted questions or inspect the collapsed trace; unrelated questions intentionally return insufficient context.
- **Gemini failure:** remove `GEMINI_API_KEY` to verify the offline demo, then check quota and endpoint settings without logging the key.
- **Full-mode connection failure:** run `docker compose --profile full ps` and confirm service health and environment names.
- **Slow full startup:** DPR/T5 downloads and service health checks can take time; demo mode avoids them.
- **Docker port in use:** change only the host side of `8501:8501` or `8502:8501`.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Original project code and fixture summaries are available under the [MIT License](LICENSE); linked NVIDIA content remains under its own terms.

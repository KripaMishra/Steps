# CUDA Documentation Copilot

CUDA Documentation Copilot is a citation-first Retrieval-Augmented Generation (RAG) demo for CUDA documentation questions. It has two intentionally separate paths:

- **Demo mode (default):** a 12-record project-authored fixture corpus at [`demo/fixtures/cuda_docs.json`](demo/fixtures/cuda_docs.json), deterministic local TF-IDF retrieval, and an offline extractive fallback when `GEMINI_API_KEY` is not set.
- **Full mode:** Gemini embeddings plus Milvus/Zilliz Cloud vector search; Elasticsearch BM25 is optional and is fused with dense results when available.

The project is not represented as production-ready or accuracy-benchmarked. Answers must expose supporting sources; unsupported questions return an explicit insufficient-context response.

**Limitations:** full mode sends generation and embedding requests to Gemini, so operators must manage API quota, cost, and input privacy. It also depends on Milvus/Zilliz and optional Elasticsearch infrastructure, making it operationally heavier than the offline demo path.

## Repository map

| Path | Purpose |
| --- | --- |
| [`main.py`](main.py) | Streamlit UI for asking questions, viewing citation cards, retrieval traces, and JSON downloads. |
| [`components/demo_retrieval.py`](components/demo_retrieval.py) | Dependency-light fixture loader and deterministic TF-IDF cosine retriever. |
| [`components/s5_rag_llm.py`](components/s5_rag_llm.py) | Shared demo/full query orchestration, Gemini generation, extractive fallback, CLI entry point. |
| [`components/response_schema.py`](components/response_schema.py) | Shared response/source/latency dataclasses used by both modes. |
| [`components/settings.py`](components/settings.py) | Environment loading plus query/top-k validation at the trust boundary. |
| [`components/gemini_embeddings.py`](components/gemini_embeddings.py) | Gemini embedding wrapper for full-mode document and query vectors. |
| [`components/s3_data_ingestion.py`](components/s3_data_ingestion.py) | Full-mode ingestion into Milvus, with optional Elasticsearch indexing. |
| [`components/s4_data_retrieval.py`](components/s4_data_retrieval.py) | Full-mode Milvus retrieval and optional Elasticsearch + dense RRF ranking. |
| [`components/s1_data_cleaning.py`](components/s1_data_cleaning.py), [`components/s2_semantic_chunking.py`](components/s2_semantic_chunking.py) | Legacy crawl-output cleaning and semantic chunking helpers for operator-managed full-mode data prep. |
| [`components/evaluate.py`](components/evaluate.py), [`eval/questions.json`](eval/questions.json) | Offline demo evaluator and checked question set. |
| [`nvidia_docs/`](nvidia_docs/) | Scrapy project for operator-run NVIDIA documentation crawling. Generated crawl output is ignored. |
| [`scripts/check_ast_imports.py`](scripts/check_ast_imports.py) | Lightweight lint/smoke check used by `make lint` and CI. |
| [`docs/FULL_MODE.md`](docs/FULL_MODE.md), [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md), [`docs/DEMO_CHECKLIST.md`](docs/DEMO_CHECKLIST.md), [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) | Detailed full-mode, deployment, manual demo, and provenance notes. |
| `notebooks/` | Optional local exploratory notebooks when present; ignored by Git. |
| `result/` | Local query/evaluation/crawl output directory; ignored by Git. |

## Prerequisites

- Python 3.11 or newer.
- For the public demo: dependencies in [`requirements-demo.txt`](requirements-demo.txt). No API key, Milvus, Elasticsearch, GPU, model download, or crawl is required.
- For full mode: dependencies in [`requirements.txt`](requirements.txt), `GEMINI_API_KEY`, `MILVUS_ENDPOINT`, and `MILVUS_TOKEN`. Elasticsearch is optional.
- Docker Compose is optional for containerized demo/full-mode app startup.

## Quick start: offline demo

```bash
git clone https://github.com/KripaMishra/cuda-documentation-rag.git
cd cuda-documentation-rag
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
make demo
```

Open <http://localhost:8501>. Try:

1. `How do CUDA streams allow work to overlap?`
2. `What factors can limit CUDA occupancy?`
3. `Why is pinned host memory used for asynchronous copies?`

CLI demo:

```bash
RAG_MODE=demo python -m components.s5_rag_llm \
  "How do CUDA streams allow work to overlap?" --top_k 3
```

The CLI prints JSON. It writes a file only when `--file_path result/answer.json` is supplied.

## Response contract

Both modes return the same top-level JSON fields:

```text
query, answer, sources[], mode, latency,
retrieval_metadata, model, corpus_version, insufficient_context
```

Each source includes URL, title, section, chunk/document ID, retrieval score, excerpt, source date, and provenance. Latency values are current-process diagnostics, not benchmarks.

## Architecture

```text
question
  │
  ▼
settings + validation ── mode router
  │                         │
  │                         ├─ demo fixture → TF-IDF cosine retrieval
  │                         │
  │                         └─ Gemini query embedding → Milvus dense search
  │                                                     + optional Elasticsearch BM25
  │                                                     + reciprocal-rank fusion
  ▼
Gemini answer generation when configured
or offline extractive fallback in demo mode
  │
  ▼
shared response contract → Streamlit UI / CLI JSON
```

Demo mode is deliberately dependency-light and should not import or initialize Torch, Transformers, Milvus, or Elasticsearch. Full mode stores `id`, `embedding`, `content`, and provenance metadata in Milvus using the configured Gemini embedding dimension; Elasticsearch adds optional BM25 retrieval only when `ELASTICSEARCH_ENABLED=true` and the service is reachable.

## Configuration

Copy `.env.example` to `.env` for local values. `.env` is ignored; never commit credentials. Before local host full-mode commands such as Python, Make, or Streamlit, export those values into the shell:

```bash
set -a; . ./.env; set +a
```

Docker Compose reads `.env` for variable substitution, so Compose commands do not need this manual shell export.

| Variable | Default | Used for |
| --- | --- | --- |
| `RAG_MODE` | `demo` | Selects `demo` or `full`. |
| `GEMINI_API_KEY` | unset | Optional Gemini generation in demo; required in full mode. |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Gemini chat/generation model. |
| `GEMINI_BASE_URL` | Google OpenAI-compatible endpoint | Gemini API endpoint. |
| `RAG_QUERY_MAX_CHARS` | `500` | Maximum normalized query length. |
| `RAG_TOP_K_MAX` | `10` | Maximum requested sources. |
| `RAG_CONTEXT_MAX_CHARS` | `12000` | Maximum context passed to generation. |
| `RAG_GENERATION_MAX_TOKENS` | `512` | Gemini output token limit. |
| `RAG_REQUEST_TIMEOUT_SECONDS` | `60` | Gemini request timeout. |
| `RAG_DEMO_FIXTURE_PATH` | `demo/fixtures/cuda_docs.json` | Demo corpus path. |
| `RAG_RESULT_DIR` | `result` | Default output directory for saved query results. |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Full-mode embedding model. |
| `GEMINI_EMBEDDING_DIMENSIONS` | `768` | Full-mode Milvus vector dimension. |
| `MILVUS_ENDPOINT`, `MILVUS_TOKEN` | unset | Milvus/Zilliz Cloud connection; required in full mode. |
| `MILVUS_COLLECTION` | `Test_collection` | Milvus collection name. |
| `ELASTICSEARCH_ENABLED` | `false` | Enables optional BM25 indexing/retrieval. |
| `ELASTICSEARCH_HOST` | `localhost` | Elasticsearch host. Compose full mode defaults this to `elasticsearch`. |
| `ELASTICSEARCH_PORT` | `9200` | Elasticsearch port. |
| `ELASTICSEARCH_INDEX` | `documents` | Elasticsearch index name. |

Changing `GEMINI_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_DIMENSIONS`, or the Milvus schema requires recreating/reindexing the full-mode collection.

## Data and indexing workflow

### Demo fixture

The demo corpus is already committed at [`demo/fixtures/cuda_docs.json`](demo/fixtures/cuda_docs.json). It contains original project-authored summaries with NVIDIA source links and provenance metadata. To change it, follow [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md), then run:

```bash
make test
make evaluate
```

### Full-mode crawl and ingestion

Read [`docs/FULL_MODE.md`](docs/FULL_MODE.md) before crawling or ingesting. Generated crawl and result files are local operator data and are ignored by Git.

```bash
cp .env.example .env
# edit .env: set GEMINI_API_KEY, MILVUS_ENDPOINT, MILVUS_TOKEN, and optional Elasticsearch values
cd nvidia_docs
scrapy crawl nvidia_docs -O nvidia_docs/spiders/output.json
cd ..
python components/s1_data_cleaning.py \
  --file_path nvidia_docs/nvidia_docs/spiders/output.json \
  --output_path result/cleaned_data.txt
python components/s2_semantic_chunking.py \
  --file_path result/cleaned_data.txt \
  --similarity_threshold 0.15 \
  --max_chunk_length 400 \
  --output_json_file result/preprocessed_chunks.json \
  --micro_json_file result/micro_chunks.json \
  --micro_threshold 100
```

Before ingestion, chunks used for cited answers need real provenance fields:

```text
id, content, source_url, title, section, chunk_id,
source_date, corpus_version, provenance
```

Do not invent missing provenance. With a provenance-enriched JSON file and full-mode environment values loaded:

```bash
set -a; source .env; set +a
RAG_MODE=full python -m components.s3_data_ingestion \
  --input_path result/provenance_enriched_chunks.json \
  --collection_name "$MILVUS_COLLECTION" \
  --es_index "$ELASTICSEARCH_INDEX" \
  --es_host "$ELASTICSEARCH_HOST" --es_port "$ELASTICSEARCH_PORT"
```

For a wiring check against the demo fixture shape, `make ingest-demo` runs `components.s3_data_ingestion` with `demo/fixtures/cuda_docs.json` in full mode after full-mode services and secrets are available.

## Running the app and queries

| Task | Command |
| --- | --- |
| Streamlit demo | `make demo` |
| Demo CLI question | `make demo-cli` |
| Custom demo CLI question | `RAG_MODE=demo python -m components.s5_rag_llm "What factors can limit CUDA occupancy?" --top_k 3` |
| Full-mode CLI question | `RAG_MODE=full python -m components.s5_rag_llm "What is CUDA occupancy?" --top_k 3` |
| Full-mode Streamlit | `RAG_MODE=full streamlit run main.py` |
| Offline evaluation | `make evaluate` |
| Tests | `make test` |
| AST/import smoke + whitespace check | `make lint` |

## Docker Compose

Default demo service:

```bash
docker compose up --build app
```

Full-mode app service:

```bash
docker compose --profile full up --build app-full
```

Optional internal Elasticsearch service:

```bash
docker compose --profile elasticsearch up -d elasticsearch
```

The demo app publishes port 8501; full mode publishes host port 8502 to the Streamlit container port. Elasticsearch is exposed only on the internal Compose network. See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) before exposing the app outside a local/self-hosted boundary.

## Notebooks and scripts

- `notebooks/chunking.ipynb`, `notebooks/formatting.ipynb`, `notebooks/complete.ipynb`, and `notebooks/dpr.ipynb` are exploratory/local notebooks when present; they are not part of CI.
- [`standalone_embed.sh`](standalone_embed.sh) starts/stops/deletes a local Milvus standalone container with Docker and `sudo`; the current Compose full-mode path expects Milvus/Zilliz connection values from the environment.
- [`scripts/check_ast_imports.py`](scripts/check_ast_imports.py) parses tracked and untracked, unignored Python files and smoke-imports lightweight demo modules.

## Validation and troubleshooting

Developer checks:

```bash
python -m pip install -r requirements-demo.txt pytest
make test
make lint
make evaluate
```

`make evaluate` runs the offline question set in [`eval/questions.json`](eval/questions.json). Its hit rate and MRR describe only the small checked fixture set and must not be generalized to CUDA documentation quality.

Common issues:

- **No sources:** use one of the scripted demo questions or inspect the retrieval trace; unrelated questions intentionally return insufficient context.
- **Demo import starts heavy services:** run `make lint`; demo modules should stay lightweight.
- **Gemini failure:** unset `GEMINI_API_KEY` to verify the offline demo, then check quota, model, endpoint, and timeout settings without logging secrets.
- **Full-mode configuration error:** confirm `RAG_MODE=full`, `GEMINI_API_KEY`, `MILVUS_ENDPOINT`, and `MILVUS_TOKEN` are set.
- **Elasticsearch unavailable:** full-mode ingestion/retrieval should continue with Milvus/Gemini vector search and skip BM25 fusion.
- **Docker port in use:** change only the host side of `8501:8501` or `8502:8501` in Compose.

## Data, safety, and license

The fixture summaries are original project content with NVIDIA source links; this repository does not redistribute an NVIDIA documentation dataset or relicense NVIDIA documentation. Generated crawl data, result files, credentials, caches, databases, and local notebooks should stay out of Git unless a separate rights review says otherwise.

See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), and [`LICENSE`](LICENSE). Original project code and fixture summaries are available under the MIT License; linked NVIDIA content remains under its own terms.

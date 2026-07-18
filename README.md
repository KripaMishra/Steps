# NVIDIA CUDA Documentation RAG

A Retrieval-Augmented Generation (RAG) application for NVIDIA CUDA documentation.

## Architecture

```text
NVIDIA docs → cleaning → semantic chunks → DPR embeddings
                                      ↓
                         Milvus + Elasticsearch
                                      ↓
                              hybrid retrieval
                                      ↓
                  Gemini via LangChain ChatOpenAI
```

- **Milvus** stores DPR vector embeddings and performs semantic search.
- **Elasticsearch** provides BM25/full-text search.
- **Gemini** generates answers through Google's OpenAI-compatible endpoint.
- Retrieval models, embeddings, Milvus schema, and Elasticsearch mappings remain unchanged in this migration.

## Setup

Use Python 3.11+ in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy the environment template and set the Google AI Studio key:

```bash
cp .env.example .env
export GEMINI_API_KEY="your-key"
```

The application reads environment variables directly; `.env` is for local reference and is not loaded automatically.

Required Gemini settings:

```text
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

Optional retrieval settings are documented in `.env.example` and default to local Milvus (`localhost:19530`) and Elasticsearch (`localhost:9200`).

## Run

Start Milvus and Elasticsearch, then ingest processed documents:

```bash
python components/s3_data_ingestion.py \
  --input_path result/preprocessed_chunks.json \
  --collection_name Test_collection \
  --es_host localhost --es_port 9200 \
  --milvus_host localhost --milvus_port 19530
```

Run the CLI:

```bash
python -m components.s5_rag_llm "What is CUDA used for?" --top_k 3
```

Run the Streamlit UI:

```bash
streamlit run main.py
```

## Tests

Run the offline test suite:

```bash
python -m pytest
```

The unit tests mock the retriever and Gemini client. A live smoke test requires running Milvus, Elasticsearch, and a valid `GEMINI_API_KEY`.

## Repository layout

- `components/settings.py` — environment-backed runtime configuration.
- `components/s1_data_cleaning.py` — document cleaning.
- `components/s2_semantic_chunking.py` — semantic chunking.
- `components/s3_data_ingestion.py` — Milvus and Elasticsearch ingestion.
- `components/s4_data_retrieval.py` — hybrid retrieval and reranking.
- `components/s5_rag_llm.py` — Gemini-backed answer generation.
- `nvidia_docs/` — Scrapy crawler.
- `tests/` — offline regression tests.
- `result/` — generated runtime artifacts; ignored for new files.

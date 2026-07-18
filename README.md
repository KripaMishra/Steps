# CUDA Documentation RAG

Retrieval-Augmented Generation (RAG) over NVIDIA CUDA documentation.

The application crawls and chunks CUDA docs, indexes them in a hybrid Milvus + Elasticsearch store, and answers questions with Google Gemini through LangChain's OpenAI-compatible client.

Repository: <https://github.com/KripaMishra/cuda-documentation-rag>

## At a glance

| Layer | Technology | Responsibility |
| --- | --- | --- |
| Crawl | Scrapy | Collect NVIDIA CUDA documentation |
| Clean/chunk | Python, scikit-learn | Normalize text and create chunks |
| Vector search | Milvus + DPR | Semantic retrieval |
| Keyword search | Elasticsearch BM25 | Full-text retrieval |
| Answer generation | Gemini + `langchain-openai` | Grounded response generation |
| UI | Streamlit | Interactive query interface |

Gemini is used for **answer generation only**. The existing DPR embeddings, T5 query expansion, Milvus schema, Elasticsearch index, and hybrid ranking remain unchanged.

## Architecture

```text
NVIDIA CUDA docs
      │
      ▼
Scrapy crawler → cleaning → semantic chunking
                                  │
                                  ▼
                         DPR embeddings
                            ┌─────┴─────┐
                            ▼           ▼
                         Milvus   Elasticsearch
                            └─────┬─────┘
                                  ▼
                          hybrid retrieval
                                  ▼
                   Gemini via ChatOpenAI compatibility API
                                  ▼
                             answer / UI
```

## Requirements

- Python 3.11+
- Milvus available at `localhost:19530` by default
- Elasticsearch available at `localhost:9200` by default
- Google AI Studio Gemini API key
- Enough disk and memory for DPR/T5 model downloads

## Quick start

```bash
git clone https://github.com/KripaMishra/cuda-documentation-rag.git
cd cuda-documentation-rag

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Configure the environment:

```bash
cp .env.example .env
$EDITOR .env
set -a
source .env
set +a
```

At minimum, set:

```text
GEMINI_API_KEY=your-google-ai-studio-key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

`.env` is ignored by Git. Never commit an API key.

## Run the pipeline

### 1. Crawl CUDA documentation

```bash
cd nvidia_docs
scrapy crawl nvidia_docs -O nvidia_docs/spiders/output.json
cd ..
```

The crawler depth is controlled by `nvidia_docs/nvidia_docs/settings.py` (`DEPTH_LIMIT=1` by default).

### 2. Clean the crawl output

```bash
python components/s1_data_cleaning.py \
  --file_path nvidia_docs/nvidia_docs/spiders/output.json \
  --output_path result/cleaned_data.txt
```

### 3. Create semantic chunks

```bash
python components/s2_semantic_chunking.py \
  --file_path result/cleaned_data.txt \
  --similarity_threshold 0.15 \
  --max_chunk_length 400 \
  --output_json_file result/preprocessed_chunks.json \
  --micro_json_file result/micro_chunks.json \
  --micro_threshold 100
```

### 4. Start Milvus and ingest documents

If using the bundled Milvus helper:

```bash
bash standalone_embed.sh start
```

Then ingest the regular chunks into Milvus and Elasticsearch:

```bash
python components/s3_data_ingestion.py \
  --input_path result/preprocessed_chunks.json \
  --collection_name Test_collection \
  --es_host localhost --es_port 9200 \
  --milvus_host localhost --milvus_port 19530
```

### 5. Query the RAG system

CLI:

```bash
python -m components.s5_rag_llm \
  "What is CUDA used for?" \
  --top_k 3
```

Streamlit:

```bash
streamlit run main.py
```

## Configuration

All runtime settings are read from environment variables. See `.env.example` for the complete template.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | required | Google AI Studio credential |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `GEMINI_BASE_URL` | Google OpenAI-compatible endpoint | Gemini API base URL |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Milvus connection |
| `MILVUS_COLLECTION` | `Test_collection` | Milvus collection |
| `ELASTICSEARCH_HOST` / `ELASTICSEARCH_PORT` | `localhost` / `9200` | Elasticsearch connection |
| `ELASTICSEARCH_INDEX` | `documents` | Elasticsearch index |
| `RAG_RESULT_DIR` | `result` | Generated result directory |
| `RAG_CONTEXT_MAX_CHARS` | `12000` | Maximum context sent to Gemini |

## Tests and verification

Run the offline suite:

```bash
python -m pytest
```

The tests mock Gemini and retrieval, so they do not require API credentials, Milvus, Elasticsearch, or model downloads.

For a live smoke test, start both data services, export `GEMINI_API_KEY`, ingest documents, and run the CLI command above. Results are written under `result/`, which is ignored for new runtime files.

## Repository layout

```text
components/
├── settings.py          # environment-backed configuration
├── s1_data_cleaning.py  # crawl-output cleaning
├── s2_semantic_chunking.py
├── s3_data_ingestion.py # Milvus + Elasticsearch ingestion
├── s4_data_retrieval.py # hybrid retrieval and reranking
└── s5_rag_llm.py        # Gemini-backed answer generation

nvidia_docs/              # Scrapy project
notebooks/                # exploratory notebooks
tests/                   # offline regression tests
main.py                  # Streamlit entry point
requirements.txt         # runtime and test dependencies
.env.example             # safe configuration template
```

## Troubleshooting

- **`GEMINI_API_KEY is required`**: export the key or source `.env` in the current shell.
- **Milvus/Elasticsearch connection errors**: confirm both services are running and match the host/port variables.
- **Slow first run**: DPR and T5 models are downloaded and loaded locally.
- **No answer context**: rerun ingestion and confirm that the configured Milvus collection and Elasticsearch index contain documents.
- **Generated files appearing in Git**: keep runtime output under `result/`; do not force-add ignored files.

## Current boundaries

This repository intentionally keeps retrieval and storage stable while migrating answer generation to Gemini. Evaluation of answer quality, latency, and cost should be performed with a populated index and representative CUDA questions before production use.

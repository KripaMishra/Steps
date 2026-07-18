PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)

.PHONY: demo demo-cli test lint evaluate up up-full down ingest-demo

demo:
	RAG_MODE=demo $(PYTHON) -m streamlit run main.py

demo-cli:
	RAG_MODE=demo $(PYTHON) -m components.s5_rag_llm "How do CUDA streams allow work to overlap?"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) scripts/check_ast_imports.py
	git diff --check

evaluate:
	RAG_MODE=demo $(PYTHON) -m components.evaluate --questions eval/questions.json

up:
	docker compose up --build app

up-full:
	docker compose --profile full up --build app-full

down:
	docker compose --profile full down

ingest-demo:
	RAG_MODE=full $(PYTHON) -m components.s3_data_ingestion --input_path demo/fixtures/cuda_docs.json

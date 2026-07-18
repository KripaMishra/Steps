# Manual demo checklist

Run with no API key:

```bash
env -u GEMINI_API_KEY RAG_MODE=demo python -m streamlit run main.py
```

For each question below, confirm a labeled offline extractive answer, at least one clickable NVIDIA source card, non-negative latency values, the demo mode/corpus indicator, a collapsed retrieval trace, and valid JSON download.

- [ ] `How do CUDA streams allow work to overlap?` — first citation is the asynchronous streams fixture.
- [ ] `What factors can limit CUDA occupancy?` — citations include the occupancy fixture.
- [ ] `Why is pinned host memory used for asynchronous copies?` — citations include the pinned-memory fixture.

Also confirm:

- [ ] An empty query shows a validation warning.
- [ ] `Who won the international football tournament?` returns explicit insufficient context and no fabricated citation.
- [ ] A simulated service exception shows a generic error without exception text or secrets.

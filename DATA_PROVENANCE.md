# Data provenance

## Demo fixture corpus

`demo/fixtures/cuda_docs.json` contains short, project-authored factual summaries created for this repository. The summaries cite publicly accessible NVIDIA documentation pages but are not copied documentation excerpts and are not an NVIDIA dataset. Each record includes its source URL, page title, section, source access date, corpus version, and provenance statement.

The fixture summaries are covered by the repository's code license. NVIDIA documentation, product names, trademarks, and linked web content remain the property of NVIDIA and their respective owners. The repository's license does not relicense NVIDIA material.

Before changing or redistributing fixture content, verify the current terms and robots/crawl policies of every source. Prefer original summaries and links over copied passages.

## Full-mode crawl output

The crawler and processing code can create local files under `nvidia_docs/nvidia_docs/spiders/output.json` and `result/`. Those generated files are intentionally ignored and are not distributed. Operators are responsible for reviewing NVIDIA's current website terms, documentation license, robots policy, and any third-party content before crawling, storing, or sharing data.

The historical tracked crawl output, processed chunks, and query results were removed for the public baseline because their provenance and redistribution permissions were not documented.

## Refresh policy

1. Review the source page and its current terms.
2. Write a concise original summary; do not paste documentation text.
3. Update `source_date`, `corpus_version`, and the provenance field.
4. Run `make test` and `make evaluate`.
5. Review fixture changes for accidental copied text or secrets before committing.

This provenance record is informational and is not legal advice.

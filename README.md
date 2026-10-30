# Maxx Content Q&A

Ask questions in plain English about Maxx Orthopedics' PDF library and get back grounded answers that cite the source document and page. It runs on open-weight models only: BGE-M3 embeddings, the BGE reranker, and a Qwen3 model served by Ollama. Vectors live in Qdrant.

The full spec is in [`CLAUDE.md`](CLAUDE.md). v1 is for internal staff and distributors only. It is not a medical-advice tool, and it needs legal/regulatory review before any pilot (§8).

## Quick start

Prerequisites: Python 3.11+, Docker, [Ollama](https://ollama.com). For scanned PDFs you also need `tesseract-ocr`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in PARSE_APP_ID / PARSE_REST_KEY (and ADMIN_TOKEN)
docker compose up -d qdrant
ollama pull qwen3
```

1. **Corpus report** (Phase 0, step 2). This shows how many PDFs there are and how many are scanned:

   ```bash
   python -m ingest.enumerate --probe all
   ```

2. **Ingest.** Start with a subset, then run the full corpus. Re-runs are safe: unchanged PDFs are skipped, and on a full run, PDFs that were deactivated or hidden get removed.

   ```bash
   python -m ingest.run_ingest --limit 25
   python -m ingest.run_ingest
   ```

   The first run downloads the BGE models from Hugging Face, about 2.3 GB for embeddings plus 2.3 GB for the reranker.

3. **Serve.** Open <http://localhost:8000>.

   ```bash
   uvicorn app.main:app --port 8000
   ```

## API

| Endpoint | Body / headers | Response |
|---|---|---|
| `POST /api/chat` | `{"question": "...", "sessionId": "optional"}` | `{"answer", "citations": [{"title","url","page"}], "foundInMaterials"}` |
| `POST /api/ingest/refresh` | header `X-Admin-Token: $ADMIN_TOKEN` | `{"indexed","skipped","removed","failed"}` |
| `GET /api/health` | | `{"status": "ok"}` |

Citation markers in `answer` (`[1]`, `[2]`) are numbered to match the `citations` list. If nothing scores above the relevance threshold, or the model says the materials don't cover the question, the response is a fixed "not found" message with `foundInMaterials: false`. The model is never asked to answer from general knowledge.

Every question gets logged to `logs/qa.jsonl`, along with the answer, the citations, and every retrieved passage with its score, for review.

## How it works

```
Parse Media (type=pdf, active, !hidden)
  → download + SHA-256 (data/pdfs, data/manifest.json)
  → PyMuPDF text per page, Tesseract OCR for pages without a text layer
  → ~900-token chunks, 15% overlap, split on headings/paragraphs, tagged with title/product/category/page
  → BGE-M3 embeddings → Qdrant collection `maxx_media`

question → BGE-M3 → top 20 by cosine → BGE reranker → keep ≥ threshold, top 5
  → grounded prompt (numbered sources) → Ollama → answer + citations actually used
```

A document gets re-indexed when its file hash, title, URL, product, or category changes, or when `PIPELINE_VERSION` in `ingest/run_ingest.py` is bumped.

## Tuning

- `RERANK_THRESHOLD` / `VECTOR_THRESHOLD` control when the system says "not found". The defaults are starting guesses, so calibrate them on the eval set. If real questions come back "not found", lower them. If off-topic questions get answered, raise them.
- `RERANK_ENABLED=false` skips the reranker. Answers get faster but worse, and only the vector threshold applies.
- `LLM_MODEL` takes any Ollama tag. Qwen3's thinking mode is turned off by default for latency (`LLM_DISABLE_THINKING`).

## Tests

```bash
pytest                                            # offline unit + pipeline tests (fake models, in-memory Qdrant)
EVAL_BASE_URL=http://localhost:8000 pytest tests/test_eval.py -v   # acceptance eval against a running server
```

The positive questions in `tests/eval_questions.yaml` are still placeholders. Fill in about 20 real questions with the PDF (and page) each answer should cite.

## Security notes

- The Parse endpoint is **plain HTTP**. It's only ever called server-side, and the keys stay in `.env`, never in the browser. Switch `PARSE_BASE_URL` to HTTPS if an HTTPS endpoint exists.
- `/api/ingest/refresh` is disabled until `ADMIN_TOKEN` is set.
- The service has no auth in v1. Keep it on an internal network.

# CLAUDE.md — Maxx Content Q&A (RAG) build spec

> This file is the build spec and working context for a **new, standalone project**
> built with Claude Code. It is self-contained: read it top to bottom, confirm the
> assumptions in §1, then execute the build plan in §10. It is **not** part of the
> Maxx mobile app repo — it's a separate web service that reuses the same content.

---

## 1. What we're building & current assumptions

Build a standalone web app that answers natural-language questions over Maxx
Orthopedics' existing PDF library and returns **grounded answers that cite the
source document (title + page)**, using an **open-weight** LLM (RAG). It will be
proven standalone first, then integrated into the Maxx mobile app later via a
shared API (§8).

**Scope for v1:** "find and summarize what our materials say," with citations.
Not a medical-advice tool. No login/SSO. No video (phase 2). Read-only against the
content backend.

**Decisions already made (defaults — change only with reason, flag if you do):**

| Decision | Default for v1 | Why |
|----------|----------------|-----|
| Audience | Internal staff + distributors | Lower regulatory risk than surgeons/patients |
| LLM (dev) | **Ollama** running a Qwen3 instruct model locally | Self-contained, no cloud GPU for PoC |
| LLM (scale) | vLLM self-host **or** a hosted open-weight provider | Production path |
| Embeddings | **BGE-M3** (MIT) via `sentence-transformers` | Strong, multilingual, CPU-runnable for a small corpus |
| Reranker | **BGE-reranker-v2-m3** (optional in PoC) | Big quality lift, ~4GB VRAM |
| Vector DB | **Qdrant** via Docker | Self-contained local dev |
| Backend | **Python + FastAPI** | Thin API |
| Frontend | Minimal static **HTML/JS chat page** served by FastAPI | Avoid a second toolchain in PoC; upgrade to Next.js later |

**Open questions to confirm with Andy (proceed with the defaults above until told otherwise):**
1. Corpus: ~how many PDFs, digital-text or scanned? → the build's first step (§10, Phase 0) **measures this**.
2. Hosting/GPU budget for the pilot.
3. Is there an **HTTPS** Parse endpoint? (today it's plain HTTP — see §9.)
4. Any PDFs to exclude beyond `active=1 && !hidden`?

> **Model recency:** this file was written Oct 2026; verify current open-weight model
> versions/names when you start (the landscape moves fast). Qwen3 / Llama 4 / DeepSeek
> V4 and BGE-M3 / Qwen3-Embedding are the current reference points.

---

## 2. How to access the content (the Maxx Parse backend)

All content lives in a **Parse Server REST API** — the same one the Maxx mobile app
uses. Read it **read-only, server-side only**.

- **Base URL:** `http://beanstalk.maxxortho.com/`  ⚠️ plain HTTP — never call from the browser (§9).
- **Auth headers on every request:**
  - `X-Parse-Application-Id: <PARSE_APP_ID>`
  - `X-Parse-REST-API-Key: <PARSE_REST_KEY>`
- **Where the key values come from:** the Maxx mobile repo, `src/services/queries.js`
  (the `X-Parse-Application-Id` / `X-Parse-REST-API-Key` set in the request
  interceptor, ~lines 58–59). Copy them into `.env` (§9) — **do not** hardcode or
  commit them. If that repo isn't attached to your session, ask Andy for the two values.

**Data model (hierarchy):**
`Company` → `Products` (`companyId`) → `Category` (`parent_object_id` = product's `objectId`) → `Media` (`categoryId` = category's `objectId`)

**`Media` object fields:** `objectId`, `type` (`"pdf"` | `"video"`), `url` (direct file
URL to download), `title`, `text` (description), `thumbnail`, `categoryId`,
`active` (1/true), `hidden` (bool), `createdAt`.

**Enumerate every PDF** — page the `Media` class filtered to PDFs:

```bash
# total first
curl -s 'http://beanstalk.maxxortho.com/1/classes/Media?where={"type":"pdf","active":1}&count=1&limit=0' \
  -H "X-Parse-Application-Id: $PARSE_APP_ID" -H "X-Parse-REST-API-Key: $PARSE_REST_KEY"
# then page: skip = 0, 100, 200, ... until a page returns < 100 rows
curl -s 'http://beanstalk.maxxortho.com/1/classes/Media?where={"type":"pdf","active":1}&limit=100&skip=0' \
  -H "X-Parse-Application-Id: $PARSE_APP_ID" -H "X-Parse-REST-API-Key: $PARSE_REST_KEY"
```

Each row's `url` is the PDF to download. Filter out `hidden == true`.

**Optional — walk the tree** to tag each chunk with product/category names:
- `GET 1/classes/Company?order=orderIndex`
- `GET 1/classes/Products?where[companyId]=<companyObjectId>`
- `GET 1/classes/Category?where[parent_object_id]=<productObjectId>`
- `GET 1/classes/Media?where[categoryId]=<categoryObjectId>&limit=1000&where[active]=1`

---

## 3. Repository layout (create this)

```
maxx-content-qa/
├── CLAUDE.md                 # this file
├── README.md                 # how to run (generate from this spec)
├── .env.example              # config template (§9)
├── docker-compose.yml        # Qdrant (+ optional Ollama)
├── requirements.txt          # or pyproject.toml
├── config.py                 # loads env vars
├── ingest/
│   ├── enumerate.py          # list PDFs from Parse; report count + scanned-PDF detection
│   ├── download.py           # fetch PDFs, hash, skip-if-unchanged
│   ├── extract.py            # PyMuPDF text; OCR fallback for scanned pages
│   ├── chunk.py              # ~800–1000 token chunks, 15% overlap, with metadata
│   ├── embed_index.py        # embed chunks + upsert to Qdrant
│   └── run_ingest.py         # orchestrates the full pipeline (idempotent)
├── app/
│   ├── main.py               # FastAPI: /api/chat, /api/ingest/refresh, /api/health
│   ├── retrieval.py          # embed query → vector search → rerank
│   └── generate.py           # grounded prompt → LLM (Ollama) → answer + citations
├── web/
│   └── index.html            # minimal chat UI (served by FastAPI)
└── tests/
    ├── eval_questions.yaml    # ~20 real Q/A for acceptance
    └── test_eval.py           # runs the eval set against /api/chat
```

---

## 4. Ingestion pipeline (idempotent)

1. **Enumerate** PDFs via §2; keep `objectId`, `title`, `url`, `categoryId` (+ product/company if walking the tree).
2. **Download** each PDF. Store a content hash keyed by `objectId`; skip unchanged files on re-runs.
3. **Extract text** with PyMuPDF. If a page has little/no extractable text (scanned image), **OCR** it (OCRmyPDF/Tesseract). Keep page numbers.
4. **Chunk** into ~800–1,000-token passages, ~15% overlap, split on headings/paragraphs. Attach metadata per chunk: `{title, product, category, sourceUrl, page, objectId}`.
5. **Embed** each chunk (BGE-M3).
6. **Upsert** chunk vectors + payload metadata into Qdrant (collection e.g. `maxx_media`).
7. **Re-ingest** safely: use `active`/`hidden` to drop removed items, `createdAt`/`updatedAt` to add new ones. `run_ingest.py` must be re-runnable without duplicating.

---

## 5. Retrieval & answer service

Per question:
1. Embed the question with the **same** model used at ingestion.
2. Vector search top ~20 by cosine.
3. Rerank with the cross-encoder; keep top ~5.
4. Build the grounded prompt: system rules + retrieved chunks (with title/page) + question.
5. Call the LLM (Ollama in dev).
6. Return `answer` + `citations` (title, url, page) for the chunks actually used + `foundInMaterials`.

**System prompt (use verbatim as a starting point):**

> You answer only using the provided Maxx Orthopedics materials. If the answer is
> not in them, say you could not find it in the materials and suggest contacting a
> Maxx representative. Do not give medical, clinical, dosing, or surgical advice.
> For every claim, cite the document title and page.

**Refusal behavior:** if top reranked scores are all below a threshold (nothing
relevant), return the "not found" message and `foundInMaterials: false` — do **not**
let the model answer from general knowledge.

---

## 6. API contract (design for later mobile reuse)

- `POST /api/chat` — `{ "question": string, "sessionId"?: string }` →
  `{ "answer": string, "citations": [{ "title": string, "url": string, "page": number }], "foundInMaterials": boolean }`
- `POST /api/ingest/refresh` — admin/cron; re-runs ingestion → `{ "indexed": number, "skipped": number }`
- `GET /api/health` → `{ "status": "ok" }`

Keep v1 **stateless and single-turn**. The mobile app will later add a chat screen
calling `/api/chat` over **HTTPS**, reusing its existing screen/loader patterns;
multi-turn history can be added to the request body then.

---

## 7. Frontend (minimal, 55+ friendly)

A single `web/index.html` chat page served by FastAPI: a question box, a send
button, an answer area, and a citations list (each citation links to its PDF `url`).
Meet the app's accessibility bar: **≥14px text, ≥44px touch targets, high contrast.**
Show a short disclaimer line (see §8). Upgrade to Next.js only if needed.

---

## 8. Guardrails & regulatory (do not skip)

This is **medical-device** content.
- Ground every answer and cite it; prefer "not found" over guessing.
- Enforce "no medical/clinical advice" in the system prompt; show a visible UI disclaimer (the mobile app already gates on a surgeon disclaimer).
- Log every question, answer, and sources used, for review.
- **Legal/regulatory review before any pilot with real users** (FDA / medical-device labeling, off-label). Run it through the Maxx legal playbook.
- Do not expose to patients in v1.

---

## 9. Secrets & configuration

All config via env vars; nothing hardcoded or committed. Create `.env` from `.env.example`.

| Env var | Value / source |
|---------|----------------|
| `PARSE_BASE_URL` | `http://beanstalk.maxxortho.com/` |
| `PARSE_APP_ID` | from Maxx mobile repo `src/services/queries.js` (~line 59) |
| `PARSE_REST_KEY` | from Maxx mobile repo `src/services/queries.js` (~line 58) |
| `QDRANT_URL` | e.g. `http://localhost:6333` |
| `OLLAMA_URL` | e.g. `http://localhost:11434` |
| `LLM_MODEL` | e.g. `qwen3` (whatever you pull in Ollama) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` |

- ⚠️ The Parse base URL is **plain HTTP**. Call it **server-side only**; never ship the key to the browser. Ask Andy whether an HTTPS endpoint exists and prefer it.
- The Parse REST key already ships inside the mobile app (low sensitivity), but still keep it out of source control and the browser bundle.
- `.gitignore` must include `.env` and any downloaded-PDF / index data dirs.

---

## 10. Build plan (execute in order)

**Phase 0 — PoC (target ~1–2 weeks of work):**
1. Scaffold the repo (§3), `docker-compose up` Qdrant, install deps, `ollama pull` the LLM model.
2. Write `ingest/enumerate.py` and run it first — **report the PDF count and how many look scanned (image-only)**. This answers Open Question #1; adjust OCR effort accordingly.
3. Build the ingestion pipeline (§4) end-to-end on a **20–30 PDF** subset.
4. Build retrieval + generation (§5) and the `/api/chat` endpoint (§6).
5. Minimal chat UI (§7).
6. Write `tests/eval_questions.yaml` with ~20 real questions + expected source docs; `test_eval.py` runs them.

**Phase 0 acceptance criteria:**
- On the ~20 eval questions, answers are correct and cite the right PDF/page.
- The system returns "not found" when the answer isn't in the corpus (no hallucination).
- Latency is acceptable for interactive use.

**Phase 1 — Pilot:** full corpus, reranker on, logging, citation UI polish, guardrails (§8), small real-user group. Acceptance: spot-check accuracy target + legal sign-off + stable idempotent re-ingest.

**Phase 2 — Mobile integration + video:** mobile chat screen calling `/api/chat` over HTTPS; transcribe videos with open-weight ASR (e.g., Whisper) and add to the corpus.

---

## 11. Reference (Oct 2026, verify at build time)

Open-weight stack guidance (vendor blogs — benchmark on the real PDFs before committing):
- https://blog.premai.io/best-open-source-llms-for-rag-in-2026-10-models-ranked-by-retrieval-accuracy
- https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting

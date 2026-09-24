# Architecture

NeuraX is an offline-first multimodal RAG system. This document describes the
service layout, the request path, and the configuration model. For HTTP
endpoints see [api.md](api.md); for environment/setup see
[deployment.md](deployment.md).

## Service layout

```text
┌────────────────────────────┐
│ Next.js frontend (:3000)   │  frontend/ — App Router workspace
│ chat · documents · search  │  Talks to the API over HTTP + SSE
│ sources · graph · settings │
└─────────────┬──────────────┘
              │ HTTP / SSE (NEXT_PUBLIC_API_URL)
┌─────────────▼──────────────┐
│ FastAPI service (:8000)    │  backend/
│ api/routes  → HTTP layer   │  Thin: validation, status codes, SSE framing
│ services/   → orchestration│  ComponentRegistry builds domain objects
└─────────────┬──────────────┘
              │ direct Python calls
┌─────────────▼──────────────────────────────────────────────┐
│ Domain modules                                             │
│                                                            │
│  ingestion/   file → text (PDF/DOCX/OCR/Whisper/notes)     │
│  indexing/    chunk → embed → persist (ChromaDB)           │
│  retrieval/   query → hybrid search → optional rerank      │
│  generation/  context → streamed answer + citations        │
│  kg_security/ security graph + Graphify document graph     │
└─────────────┬──────────────────────────────────────────────┘
              │
   ChromaDB (vector_db/) · embedding models (models/)
   LM Studio (:1234, local LLM) · optional NIM / cloud LLM
```

The backend is deliberately thin: `backend/api/routes` handles HTTP concerns
and delegates to `backend/services`, which own the domain wiring. Domain logic
never imports FastAPI.

## Ingestion path

1. `POST /api/documents` validates the upload (extension allowlist, size cap)
   and stores it under `NEURAX_UPLOAD_DIR`.
2. A background indexing job (`backend/services/job_service.py`) runs the file
   through `ingestion/ingestion_manager.py`, which dispatches by type:
   - **Documents** (`document_processor.py`): PDF/DOCX/DOC/TXT, OCR fallback via Tesseract
   - **Images** (`image_processor.py`): OCR + CLIP visual embedding
   - **Audio** (`audio_processor.py`): Whisper transcription
3. `indexing/text_chunker.py` splits extracted text into chunks with
   **deterministic chunk IDs** keyed on the source path. IDs are stable across
   re-ingests, which is what lets `scripts/reindex_corpus.py` do a clean
   delete-and-replace after chunking changes.
4. `indexing/embedding_manager.py` embeds chunks. The text provider is either
   local sentence-transformers (MiniLM) or the NVIDIA NIM provider
   (`indexing/nvidia_nim_embedding_provider.py`) when enabled. Each provider
   writes to **its own ChromaDB collection** — dimensionality differs, and the
   default collection name has a single source of truth in `config.py`
   (`NIM_EMBEDDING_CONFIG["collection_name"]`).
5. Image CLIP vectors go to a separate 512-d collection (`CLIP_IMAGE_CONFIG`).

Cross-file duplicate chunks are skipped at ingest so re-uploading a file that
shares content with an indexed one does not double-index the overlap.

## Retrieval pipeline

`retrieval/query_processor.py` implements a four-stage pipeline. The UI shows
the active mode next to search latency.

```text
query
  │
  ├─► dense search (ChromaDB, 3×k candidates when hybrid)
  ├─► BM25 sparse search (bm25_k candidates, in-memory index)
  │
  ▼
weighted RRF fusion            score(d) = Σ  weight_i / (rrf_k + rank_i)
  dense_weight 0.9 · sparse_weight 0.1   (dense-favored; equal weights let
  ▼                                      BM25 misses outrank dense picks)
optional NIM cross-encoder rerank        rerank_candidates fused passages
  ▼                                      sent to retrieval/nim_reranker.py
top-k results → citation formatting → LLM context
```

Key behaviors:

- **Hybrid is on by default** (`SEARCH_CONFIG["enable_hybrid"]`).
- **Reranking is two-key opt-in**: the `enable_reranking` flag *and* a NIM API
  key must both be present. `SEARCH_CONFIG` alone can never trigger a cloud
  call — this preserves the local-first default.
- The reranker is lazy: it is constructed on first use and cached; a broken or
  unconfigured reranker degrades to fused order, not an error.
- NIM **query** embeddings are cached with a bounded cache, so repeated queries
  do not hit the API.

## Generation path

`backend/services/chat_service.py` selects a generator by chat mode:

- **local (default under `offline_first`)** — `generation/lmstudio_generator.py`
  calls the LM Studio OpenAI-compatible endpoint (`:1234`). Gemma 3n serves
  multimodal queries, Qwen3 4B serves text reasoning; auto-switching is
  configured in `LM_STUDIO_CONFIG`.
- **cloud (default under `online_first`)** — `generation/cloud_generator.py`
  calls any OpenAI-compatible endpoint (e.g. Groq) configured via
  `NEURAX_CLOUD_*`.

Both generators stream token-by-token. The HTTP layer forwards tokens to the
browser as Server-Sent Events (`POST /api/chat/stream`); the non-streaming
`POST /api/chat` aggregates internally. Cloud failures degrade to a guidance
delta in the stream — the UI never fakes a successful generation.

When **Use Knowledge Graph Context** is enabled in Chat, compact graph
relationships (bounded by `max_rag_context_*`) are appended to the vector RAG
context. Graph context supplements ChromaDB retrieval; it never replaces it.

## Knowledge graph layers

Two distinct graphs, both under `kg_security/`:

| | Security graph | Document graph (Graphify) |
|---|---|---|
| Implementation | `knowledge_graph_manager.py` (NetworkX, in-process) | `graphify_service.py` wrapping the external Graphify CLI |
| Data | Query/document activity metadata | Uploaded corpus files |
| Purpose | Anomaly detection, tamper monitoring, audit analytics | Corpus-level entity/relation intelligence |
| Isolation | — | Subprocess calls only (never `shell=True`), managed workspace under `data/graphify/`, loopback-only model endpoints by default |

Graphify is optional: it requires Python 3.10+ on the CLI side, so it is never
an import-time dependency of NeuraX. If the CLI is missing the product still
launches and the Graph page shows install guidance.

## Configuration model

Configuration has a strict hierarchy; the offline-first posture depends on it:

1. **`config.py`** — domain defaults (LM Studio URL, Chroma paths, retrieval
   weights, Graphify limits). Single source of truth for collection names and
   fusion weights.
2. **`.env` / environment variables** — deployment-level overrides read at
   import time (`NEURAX_*`, `GRAPHIFY_*`). See
   [deployment.md](deployment.md#environment-variables).
3. **`NEURAX_STRATEGY`** — one switch that sets the *defaults* for all
   cloud-service flags:
   - `offline_first` (default): NIM embeddings off, reranking off, chat local.
   - `online_first`: NIM embeddings + reranking on when a key exists, chat cloud.
   - An explicit individual flag (`NEURAX_NIM_EMBEDDINGS_ENABLED`, …) always
     beats the strategy default.

The frontend reads only `NEXT_PUBLIC_API_URL` at build time; everything else is
queried from the API (`/api/system/status`, `/api/models/status`).

## Frontend structure

`frontend/` is a Next.js 15 App Router app (product UI — the old Gradio app was
removed; Streamlit remains optional analytics):

- `app/` — routes: `/` (chat), `documents`, `search`, `sources`, `graph`, `settings`
- `features/` — per-feature UI modules mirroring the routes
- `components/` — shared UI
- Theme (light/dark) honors the stored user choice with system fallback

## Testing surface

- `tests/` — domain units: hybrid retrieval, NIM reranker, NIM embedding
  provider, text chunker, LM Studio streaming, cloud generator, Graphify
  service/regression (uses a fake CLI; the Graphify package is not required).
- `backend/tests/` — API-level tests incl. document dedup.
- `frontend/` — `npm run typecheck`, `npm run build`, Playwright suite
  (`npm run test`, requires API + frontend running).
- `pytest.ini` collects `tests` and `backend/tests` together; plain `pytest`
  runs both.

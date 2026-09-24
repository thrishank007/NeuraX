# Deployment Guide

NeuraX is offline-first: the default posture runs entirely on local
infrastructure. This guide covers environment configuration, both deployment
strategies, and packaging for air-gapped systems.

## Prerequisites

- Python 3.9+ (3.10+ recommended); Node.js 18+ for the frontend
- [LM Studio](https://lmstudio.ai/) with downloaded models:
  - **Gemma 3n** — multimodal queries
  - **Qwen3 4B Thinking 2507** — reasoning
- Tesseract OCR and FFmpeg for full format coverage
- Optional: an NVIDIA NIM API key (embeddings + reranking) and an
  OpenAI-compatible cloud LLM endpoint (e.g. Groq)

## Environment variables

Copy `.env.example` to `.env` and adjust. Everything is optional — defaults are
fully local.

### API server

| Variable | Default | Purpose |
|---|---|---|
| `NEURAX_API_HOST` | `127.0.0.1` | FastAPI bind host |
| `NEURAX_API_PORT` | `8000` | FastAPI port |
| `NEURAX_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed browser origins. Keep tight; avoid `*` |
| `NEURAX_UPLOAD_DIR` | `data/uploads` | Upload storage path |
| `NEURAX_MAX_UPLOAD_MB` | `100` | Upload size cap |

### Deployment strategy

| Variable | Default | Purpose |
|---|---|---|
| `NEURAX_STRATEGY` | `offline_first` | `offline_first` or `online_first` (see below) |

`offline_first` — everything local: NIM embeddings and reranking stay **off**,
chat defaults to LM Studio. Each cloud service must be enabled explicitly.

`online_first` — prefer cloud when credentials exist: NIM embeddings and
reranking default **on**, chat defaults to cloud generation.

Notes:
- `NEURAX_CD` is accepted as a legacy alias.
- Hyphenated values (`online-first`) normalize to underscores.
- **Individual flags always override the strategy default.**

### Cloud LLM (optional)

| Variable | Default | Purpose |
|---|---|---|
| `NEURAX_CLOUD_API_URL` | unset | OpenAI-compatible base URL (e.g. `https://api.groq.com/openai/v1`) |
| `NEURAX_CLOUD_API_KEY` | unset | API key — never commit the real key |
| `NEURAX_CLOUD_MODEL` | unset | Model id (e.g. `llama-3.3-70b-versatile`) |
| `NEURAX_CLOUD_MAX_TOKENS` | `1024` | Generation cap |
| `NEURAX_CLOUD_TEMPERATURE` | `0.7` | Sampling temperature |
| `NEURAX_CLOUD_TIMEOUT` | `60` | Request timeout (seconds) |

Cloud streaming errors degrade to a guidance delta in the SSE stream — the UI
shows the problem instead of a fake answer.

### NVIDIA NIM embeddings + reranking (optional)

| Variable | Default | Purpose |
|---|---|---|
| `NEURAX_NIM_API_KEY` | unset | Shared key for embeddings and reranking |
| `NEURAX_NIM_EMBEDDINGS_ENABLED` | strategy default | `true`/`false` override |
| `NEURAX_NIM_EMBEDDING_API_URL` | `https://integrate.api.nvidia.com/v1/embeddings` | Embeddings endpoint |
| `NEURAX_NIM_EMBEDDING_MODEL` | `nvidia/nemotron-3-embed-1b` | Embedding model (2048-d) |
| `NEURAX_NIM_COLLECTION_NAME` | `neurax_nim_nemotron_3_embed_1b_v1` | Dedicated ChromaDB collection |
| `NEURAX_NIM_EMBEDDING_TIMEOUT` | `60` | Timeout (seconds) |
| `NEURAX_NIM_RERANK_ENABLED` | strategy default | `true`/`false` override |
| `NEURAX_NIM_RERANK_API_URL` | NVIDIA reranking endpoint | Reranking endpoint |
| `NEURAX_NIM_RERANK_MODEL` | `nvidia/rerank-qa-mistral-4b` | Must be invocable by your account |
| `NEURAX_NIM_RERANK_TIMEOUT` | `30` | Timeout (seconds) |

Reranking activates only when the flag **and** the API key are both present —
`SEARCH_CONFIG` alone can never trigger a cloud call. Switching embedding
providers changes the target collection; run
`scripts/reindex_corpus.py` to (re)populate the active collection.

### Graphify (optional)

| Variable | Default | Purpose |
|---|---|---|
| `GRAPHIFY_EXECUTABLE` | `graphify` | CLI name or absolute path |
| `GRAPHIFY_OPENAI_API_KEY` | `lm-studio` | Dummy local key |
| `GRAPHIFY_MODEL` | LM Studio Qwen model | Extraction model |

By default Graphify model endpoints must be loopback (`localhost` /
`127.0.0.1` / `::1`); set `allow_non_local_endpoint: true` in
`GRAPHIFY_CONFIG` to permit otherwise.

## Running

### Development

```powershell
# Start FastAPI (:8000) + Next.js (:3000) together
pwsh scripts/dev.ps1
```

Or separately:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

### Production process

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000   # no --reload
cd frontend && npm run build && npm run start
```

Surfaces: Next.js `http://127.0.0.1:3000` · API `http://127.0.0.1:8000` ·
interactive docs `http://127.0.0.1:8000/docs` · optional Streamlit analytics
`http://127.0.0.1:8501`.

## Portable executable / USB

```bash
python build_executables.py                # NeuraX-Windows-x64.zip in packages/
python build_executables.py --usb-deployment   # USB_Deployment/ with autorun.inf
```

## Air-gapped checklist

Prepare while online:

1. `pip install -r requirements.txt` into the target venv (or build the
   portable executable).
2. `npm install` + `npm run build` the frontend on the target machine's
   Node version.
3. Download embedding models (MiniLM, CLIP) and Whisper weights into
   `models/`.
4. Install LM Studio and download Gemma 3n + Qwen3 4B.
5. Install Tesseract OCR and FFmpeg (bundled automatically in the executable
   build).
6. Optional: `uv tool install "graphifyy[openai]"` (Python 3.10+).

At runtime, offline:

7. Keep `NEURAX_STRATEGY=offline_first` (default) — no cloud service
   activates implicitly.
8. Leave all `NEURAX_CLOUD_*` / `NEURAX_NIM_*` variables unset.
9. Verify: the Settings page and `/api/system/status` should report local
   modes with no cloud endpoints configured.
10. For LAN access, bind hosts explicitly and extend `NEURAX_CORS_ORIGINS`
    with the exact origin(s) — never `*`.

## Maintenance

```bash
# Re-index the corpus after chunker/embedding changes (deterministic IDs
# make this a clean delete-and-replace)
venv\Scripts\python.exe scripts\reindex_corpus.py

# Evaluate retrieval quality + latency (dense / bm25 / hybrid)
venv\Scripts\python.exe scripts\eval_rag.py --k 5 --runs 5
```

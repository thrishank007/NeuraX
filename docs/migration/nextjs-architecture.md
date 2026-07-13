# Next.js + FastAPI Architecture

## Overview

```text
Next.js frontend (frontend/, :3000)   ← sole product UI
    ↓ HTTP / SSE
FastAPI application service (backend/, :8000)
    ↓
Existing NeuraX Python domain modules (repo root)
    ↓
ChromaDB, embedding models, Whisper, CLIP, LM Studio
```

Gradio has been **removed**. Streamlit (`:8501`) is optional analytics only.

## Component boundaries

| Layer | Responsibility | Must not |
|---|---|---|
| `frontend/` | Workspace UI | Run embeddings/retrieval |
| `backend/api/` | HTTP routes, validation, CORS, SSE | Own retrieval algorithms |
| `backend/services/` | Domain orchestration | Reimplement RAG math |
| Domain modules | Ingestion, embed, store, query, generate | Know about HTTP |

## Local development

```bash
# Product UI (primary)
pwsh scripts/dev.ps1

# Or separately:
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

## Production

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
cd frontend && npm run build && npm run start
```

## Offline

Frontend → local FastAPI only. FastAPI → local Chroma / models / LM Studio. No required cloud APIs.

# Next.js + FastAPI Architecture

## Overview

```text
Next.js frontend (frontend/, :3000)
    ↓ HTTP, SSE/streaming fetch
FastAPI application service (backend/, :8000)
    ↓
Existing NeuraX Python domain modules (repo root)
    ↓
ChromaDB, embedding models, Whisper, CLIP, LM Studio
```

Gradio (`:7860`) and Streamlit (`:8501`) remain available during migration.

## Component boundaries

| Layer | Responsibility | Must not |
|---|---|---|
| `frontend/` | Workspace UI, status, upload, chat, sources | Run embeddings/retrieval |
| `backend/api/` | HTTP routes, validation, CORS, SSE | Own retrieval algorithms |
| `backend/services/` | Orchestration mirrored from Gradio | Reimplement RAG math |
| Domain modules | Ingestion, embed, store, query, generate | Know about HTTP |
| `config.py` | Paths and model/search defaults | Frontend env secrets |

## Backend layout

```text
backend/
├── main.py
├── config.py
├── api/
│   ├── dependencies.py
│   ├── schemas/
│   └── routes/
├── services/
└── tests/
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness |
| GET | `/api/system/status` | Components + collection stats |
| GET | `/api/models/status` | LM Studio / model info |
| GET | `/api/documents` | List indexed docs (Chroma metadata) |
| POST | `/api/documents` | Upload + queue indexing job |
| GET | `/api/documents/{id}` | Document detail |
| DELETE | `/api/documents/{id}` | Delete from index |
| GET | `/api/index/jobs/{job_id}` | Indexing progress |
| POST | `/api/index/jobs/{job_id}/cancel` | Cancel job |
| POST | `/api/search` | Multimodal search |
| POST | `/api/chat` | Retrieve + generate + cite |
| POST | `/api/chat/stream` | SSE phase events + full answer |
| GET | `/api/sources/{source_id}` | Source payload |
| GET | `/api/graph` | KG viz data when available |
| POST | `/api/feedback` | Rating feedback |

## Request / response schemas (summary)

- **Error:** `{ "error": { "code": string, "message": string, "details": object? } }`  
- **Search:** `{ query, modality, similarity_threshold?, k?, image?, audio? }` → results with `id`, `file_path`, `file_type`, `similarity_score`, `content_preview`, `metadata`  
- **Chat:** `{ query, similarity_threshold?, max_docs?, image? }` → `response`, `citations[]`, `confidence`, `processing_time`, `sources[]`  
- **Job:** `{ job_id, status, progress, total, processed, logs[], errors[] }`  

## Streaming approach

SSE events on `/api/chat/stream`:

1. `status` — retrieval_started / generation_started  
2. `retrieval` — sources payload  
3. `message` — full generated text (Gradio parity; non-token stream by default)  
4. `citations` — structured citations  
5. `done` / `error`

Retrieval prompts and context assembly stay identical to Gradio.

## Error model

- 4xx for client validation; 5xx for unexpected failures  
- Never expose Python stack traces to the browser  
- Codes: `validation_error`, `not_found`, `lmstudio_unavailable`, `empty_index`, `processing_error`, `job_not_found`, `upload_rejected`

## CORS and security

- CORS allowlist: `http://localhost:3000`, `http://127.0.0.1:3000` (configurable)  
- Upload size limit from `SECURITY_CONFIG`  
- Extension allowlist from processing config  
- Safe filenames; store under `data/uploads/`  
- Path traversal rejected  

## Offline mode

- Frontend only calls local FastAPI  
- FastAPI only calls local Chroma / local models / LM Studio  
- No analytics, hosted auth, or required cloud APIs  
- After models/deps installed, air-gapped operation matches Gradio design  

## Local development

```bash
# Existing Gradio
python main_launcher.py --mode gradio_only

# FastAPI
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Next.js
cd frontend && npm run dev

# Both API + frontend
pwsh scripts/dev.ps1
```

## Production launch

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
cd frontend && npm run build && npm run start
```

Serve on trusted LAN by binding host explicitly; keep CORS origins tight.

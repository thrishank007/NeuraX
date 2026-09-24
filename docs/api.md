# API Reference

The FastAPI service layer exposes the product API on `NEURAX_API_HOST:NEURAX_API_PORT`
(default `http://127.0.0.1:8000`). Interactive OpenAPI docs are served at
[`/docs`](http://127.0.0.1:8000/docs) — this page is a human-readable summary.

## Health & status

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness / component readiness |
| GET | `/api/system/status` | System state: offline/cloud mode flags, index health, service posture |
| GET | `/api/models/status` | LM Studio and cloud model availability |

## Chat

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/chat` | RAG answer with citations (aggregated) |
| POST | `/api/chat/stream` | Same, streamed token-by-token as Server-Sent Events |
| POST | `/api/feedback` | Submit user feedback on an answer |

`ChatRequest` fields:

| Field | Type | Notes |
|---|---|---|
| `query` | string | Required |
| `similarity_threshold` | float \| null | Retrieval cutoff override |
| `max_docs` | int \| null | Context size override |
| `use_knowledge_graph` | bool | Append compact Graphify relations to RAG context |
| `mode` | string | `local` (LM Studio) or `cloud` — default follows `NEURAX_STRATEGY` |

Responses include `response`, `citations[]`, `confidence`, `processing_time`,
`sources[]`, `model_used`, `lm_studio_available`, `graph_context_used`, and
`graph_warning`. The SSE stream emits tokens as they arrive; cloud-side errors
degrade to a guidance delta event rather than closing silently.

## Search

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/search` | Retrieval without generation |
| GET | `/api/sources/{source_id}` | Full metadata for one retrieved source |

`POST /api/search` is multipart form data:

| Field | Type | Notes |
|---|---|---|
| `query` | form string | Query text |
| `modality` | form string | `text` / `image` / `audio` / `multimodal` |
| `similarity_threshold` | form float \| null | Cutoff override |
| `k` | form int \| null | Result count override |
| `image` | file \| null | For image / multimodal modalities |
| `audio` | file \| null | For audio / multimodal modalities |

Text search runs the hybrid pipeline (dense + BM25 → weighted RRF → optional
NIM rerank). The response reports the active pipeline mode alongside latency.

## Documents & indexing

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/documents` | List indexed documents (Chroma-backed) |
| POST | `/api/documents` | Upload one or more files; returns `202` with a job id |
| GET | `/api/documents/{doc_id}` | One document's metadata |
| DELETE | `/api/documents/{doc_id}` | Delete an indexed document (with its chunks) |
| GET | `/api/index/jobs/{job_id}` | Poll indexing job status/progress |
| POST | `/api/index/jobs/{job_id}/cancel` | Cancel a running indexing job |

Uploads are validated against the extension allowlist and `NEURAX_MAX_UPLOAD_MB`.

## Knowledge graph

Security graph:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/graph` | Security graph view (NetworkX) |

Graphify document graph (CLI-backed):

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/graphify/status` | CLI availability + workspace state |
| POST | `/api/graphify/build` | Extract the knowledge graph from the corpus |
| POST | `/api/graphify/update` | Incremental update when available |
| POST | `/api/graphify/rebuild` | Clear artifacts and re-extract from scratch |
| GET | `/api/graphify/stats` | Node/edge/community statistics |
| GET | `/api/graphify/data` | Graph payload for visualization |
| GET | `/api/graphify/nodes` | Node list |
| POST | `/api/graphify/query` | Natural-language graph query |
| POST | `/api/graphify/explain` | Explanation for a node/relationship |
| POST | `/api/graphify/path` | Path finding between entities |
| GET | `/api/graphify/corpus` | Managed corpus file listing |
| GET | `/api/graphify/artifacts/{kind}` | Download `graph.html` / `graph.json` / report |

## Errors

Domain errors use a consistent envelope — `{"error": {"code", "message", "details"?}}` —
with appropriate HTTP status codes: `400` for validation, `404` for missing
resources, `500` for processing failures. See `backend/api/errors.py`.

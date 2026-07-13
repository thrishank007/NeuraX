# Feature Parity: Gradio (historical) vs Next.js

**Branch:** `feat/nextjs-frontend-migration`  
**Date:** 2026-07-13  
**Status:** Gradio UI **removed**. Product interface is **Next.js only**.

Status values: **complete** | **intentionally changed** | **blocked** | **not applicable**

| Feature | Historical Gradio | Next.js status | Verification | Notes |
|---|---|---|---|---|
| Multi-file upload | Yes | complete | API + Documents UI | `POST /api/documents` |
| Supported formats | Yes | complete | Config-driven | PROCESSING_CONFIG |
| Upload size limit | Yes | complete | Backend validation | SECURITY_CONFIG |
| Indexing progress | Yes | complete | Job polling UI | In-memory jobs |
| Document listing | Session-only | intentionally changed | Documents page | Chroma-backed list |
| Delete indexed docs | No UI | complete | DELETE + confirm | VectorStore.delete |
| Text search / chat | Yes | complete | Chat + `/api/chat` | Same QueryProcessor |
| Multimodal search UI | Yes | complete | Search page | text / image / voice / multimodal |
| Search without generation | Yes | complete | Search page | Gradio Query tab parity |
| Query history | Yes | complete | Search sidebar + session | Last 20 entries |
| Feedback rating form | Yes | complete | Chat after answer | `POST /api/feedback` |
| Help / instructions | Yes | complete | Settings accordion | Formats + troubleshooting |
| Multimodal search API | Yes | complete | `/api/search` | Form modalities |
| Similarity threshold | Yes | complete | Chat + Search controls | Same semantics |
| Citations | Yes | complete | Sources panel | CitationGenerator |
| System / LM status | Partial | complete | Status bar + Settings | Explicit probes |
| Knowledge graph | Streamlit | complete (thin) | `/api/graph` | Optional export |
| Gradio UI | Yes | **removed** | — | `ui/gradio_app.py` deleted |
| Dual-run fallback | Planned | **not applicable** | — | Next.js is sole product UI |

## Remaining non-Gradio gaps

- Token-level LM streaming not enabled (SSE full-message events)  
- Streamlit analytics/metrics dashboards not ported (ops, not Gradio product UI)

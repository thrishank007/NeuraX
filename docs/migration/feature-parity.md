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
| Multimodal search API | Yes | complete | `/api/search` | Form modalities |
| Similarity threshold | Yes | complete | Chat controls | Same semantics |
| Citations | Yes | complete | Sources panel | CitationGenerator |
| System / LM status | Partial | complete | Status bar + Settings | Explicit probes |
| Knowledge graph | Streamlit | complete (thin) | `/api/graph` | Optional export |
| Gradio UI | Yes | **removed** | — | `ui/gradio_app.py` deleted |
| Dual-run fallback | Planned | **not applicable** | — | Next.js is sole product UI |

## Gaps (product, not fallback)

- Multimodal **composer attach** (image/voice) is API-ready; text chat is the primary UI path  
- Feedback form UI not ported (API `POST /api/feedback` exists)  
- Token-level LM streaming not enabled (SSE full-message events)

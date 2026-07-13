# Feature Parity: Gradio vs Next.js

**Branch:** `feat/nextjs-frontend-migration`  
**Date:** 2026-07-13

Status values: **complete** | **intentionally changed** | **blocked** | **not applicable**

| Feature | Gradio baseline | Next.js status | Verification | Notes |
|---|---|---|---|---|
| Multi-file upload | Yes | complete | API tests + UI upload | `POST /api/documents` |
| Supported formats | Yes | complete | Config-driven | Same PROCESSING_CONFIG formats |
| Upload size limit | Yes | complete | Backend validation | SECURITY_CONFIG max MB |
| Indexing progress | Yes | complete | Job polling UI | In-memory jobs |
| Document listing | Session-only list | intentionally changed | Documents page | Chroma-backed inventory (improvement) |
| Clear processing UI | Yes | complete | N/A after job ends | Job panel replaces clear button |
| Delete indexed docs | No UI | intentionally changed | DELETE API + confirm UI | Uses existing VectorStore.delete |
| Text search | Yes | complete | `/api/search` | Same QueryProcessor |
| Image search | Yes | complete | API supports | UI chat is text-first; search API ready |
| Voice search | Yes | complete | API supports | STT path preserved |
| Multimodal search | Yes | complete | API supports | Form fields on search endpoint |
| Similarity threshold | Yes | complete | Chat controls | Same slider semantics |
| Search results display | Yes | complete | Chat sources panel | Structured JSON vs HTML |
| Query history | In-session | intentionally changed | Chat thread | Message list in session; not durable |
| AI response + citations | Yes | complete | `/api/chat` + stream | Same generation + CitationGenerator |
| Confidence + latency | Yes | complete | Message metadata | From GeneratedResponse |
| Feedback rating | Yes | blocked | Endpoint exists | UI form deferred; API `POST /api/feedback` |
| System / component status | Yes | complete | Status bar + Settings | Health split backend/vector/LM |
| LM Studio connectivity | Partial | complete | Models status | Explicit probe |
| Help / instructions | Accordion | intentionally changed | Empty states + Settings | Inline product copy |
| Knowledge graph | Streamlit only | complete (thin) | `/api/graph` | Honest empty if KG unavailable |
| Import/export index | No UI | not applicable | — | Domain methods exist; not exposed |
| Streaming tokens | No | intentionally changed | SSE phases | Full message event; not token stream |
| Offline / local-only | Yes | complete | No cloud deps | CORS localhost only |
| Gradio still runnable | Yes | complete | Unchanged `ui/gradio_app.py` | Dual-run documented |

## Core workflow verification checklist

1. Backend health displayed — covered by status bar + tests  
2. LM Studio unavailable state — Settings + status chips  
3. Supported document upload — Documents workspace  
4. Indexing progress — job panel  
5. Indexed documents list — Documents workspace  
6. Query submit — Chat workspace  
7. Streamed answer phases — SSE events rendered incrementally as events arrive  
8. Citations open sources — Chat sources panel / Sources page  
9. Document remove — confirm delete  
10. Keyboard navigation — skip link + focusable controls  
11. Responsive widths — Playwright multi-viewport  
12. Refresh safety — client state reloads cleanly  
13. No stack traces in browser — structured API errors  
14. Production Next.js build — `npm run build`  

## Honest gaps

- Multimodal **query UI** (image/voice attach in composer) is API-ready but not fully mirrored as separate Gradio-style query sections in Chat; Documents + text chat cover primary Gradio AI Response workflow.  
- Feedback form UI not ported (API only).  
- Token-level streaming not enabled (matches Gradio non-stream LM client; SSE sends full message).  

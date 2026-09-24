# Troubleshooting

Quick symptom → fix tables. For setup and configuration background see
[deployment.md](deployment.md); for how a subsystem works see
[architecture.md](architecture.md).

## General

| Symptom | What to do |
|---|---|
| Status bar: Backend unavailable | Start the API: `uvicorn backend.main:app --host 127.0.0.1 --port 8000` (or `pwsh scripts/dev.ps1`) |
| LM Studio unavailable | LM Studio → Local Server tab → load a model → serve on `localhost:1234`; check the green status indicator |
| Empty search / weak answers | Index documents first (Documents page); lower the similarity threshold in Settings; confirm hybrid mode is enabled |
| Upload rejected | Extension allowlist (`SECURITY_CONFIG`) and max size (`NEURAX_MAX_UPLOAD_MB`, default 100) in Settings |
| CORS errors in browser | Add the exact origin to `NEURAX_CORS_ORIGINS` (e.g. `http://127.0.0.1:3000`); never use `*` |
| Port already in use | Change `NEURAX_API_PORT` / run Next.js on another port; check for orphaned uvicorn/node processes |

## Cloud services & strategy

| Symptom | What to do |
|---|---|
| Cloud chat shows a guidance/error delta instead of an answer | This is by design on failure — check `NEURAX_CLOUD_API_URL` / `NEURAX_CLOUD_API_KEY` / `NEURAX_CLOUD_MODEL`; the stream degrades rather than faking success |
| Expected cloud mode but chat is local | `NEURAX_STRATEGY` defaults to `offline_first`; set `online_first` or pick cloud mode in the Chat UI |
| NIM features inactive despite `online_first` | A `NEURAX_NIM_API_KEY` must exist; reranking additionally requires the strategy default or `NEURAX_NIM_RERANK_ENABLED=true` |
| NIM reranker silently not used | The rerank model id must be invocable by your account — the API returns the valid list on a miss; set `NEURAX_NIM_RERANK_MODEL` accordingly. A broken reranker degrades to fused order |
| Search quality changed after switching embedding providers | Providers write to different ChromaDB collections; run `venv\Scripts\python.exe scripts\reindex_corpus.py` to populate the active collection |
| Repeat NIM queries still slow | Query embeddings are cached and bounded — confirm the cache directory is writable; first occurrence of each unique query does hit the API |

## Retrieval

| Symptom | What to do |
|---|---|
| BM25 dominates / wrong results | Fusion is dense-favored by design (`dense_weight` 0.9 / `sparse_weight` 0.1 in `SEARCH_CONFIG`); tune `rrf_k`, `bm25_k`, or weights, then validate with `scripts/eval_rag.py` |
| Hybrid retrieval errors on empty corpus | Disable `enable_hybrid` until at least one document is indexed |
| Duplicate chunks in results | Cross-file duplicates are skipped at ingest; re-ingest the file or run `scripts/reindex_corpus.py` after chunker changes |
| Documents indexed but not retrievable | Verify the collection name matches the active embedding provider (`NIM_EMBEDDING_CONFIG["collection_name"]` is the single source of truth) |

## Graphify

| Symptom | What to do |
|---|---|
| Graphify executable not found | Install with `uv tool install "graphifyy[openai]"` or `pipx install "graphifyy[openai]"`; ensure `graphify` is on `PATH` or set `GRAPHIFY_EXECUTABLE` |
| Unsupported Python for Graphify | Use Python 3.10+ for the Graphify tool only; NeuraX can keep an older runtime |
| LM Studio unavailable | Start LM Studio server on the configured base URL; load a chat model (e.g. Qwen) |
| Empty graph | Ensure corpus has files (upload + process documents first), then rebuild |
| Graph build timeout | Increase `process_timeout_seconds` / `api_timeout_seconds`; reduce corpus size or concurrency |
| Corrupted `graph.json` | Rebuild from scratch; check disk space and LM Studio logs |
| Non-loopback model endpoint rejected | Intentional safety default; set `allow_non_local_endpoint: true` in `GRAPHIFY_CONFIG` only if you accept the exposure |

## Frontend

| Symptom | What to do |
|---|---|
| UI cannot reach API | Confirm FastAPI is running and `frontend/.env.local` → `NEXT_PUBLIC_API_URL` points at it; restart `npm run dev` after editing |
| Theme resets after reload | Stored theme takes precedence with system fallback; clear the locally stored theme key to re-follow system |
| Streaming not visible in Chat | Token streaming requires `/api/chat/stream` (SSE); check browser console for connection errors and that no proxy buffers SSE |
| Playwright tests fail | They need a running API + frontend (`npm run test` from `frontend/`); run `npx playwright install chromium` once |

## Logs and diagnostics

- Backend logs: `logs/` (loguru, rotated 10 MB, 1-week retention).
- System status: `GET /api/system/status` (mode flags, index health) and
  `GET /api/models/status` (LM Studio / cloud model states).
- Interactive API docs: http://127.0.0.1:8000/docs

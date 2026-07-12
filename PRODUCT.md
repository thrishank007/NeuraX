# NeuraX Product

## What it is

NeuraX is a **local-first multimodal document intelligence** workspace for secure, offline RAG over documents, images, and audio. It is a productivity product for analysts—not a marketing site or generic chat toy.

## Users

- Analysts and operators who must keep data on-prem / air-gapped  
- Engineers evaluating retrieval quality with visible sources  
- Teams who already run LM Studio locally  

## Jobs to be done

1. Ingest multimodal files into a local index  
2. Search across modalities with adjustable strictness  
3. Ask grounded questions and inspect citations  
4. Trust system status (backend, vector store, LM Studio) at a glance  
5. Work without cloud services  

## Interface principles

1. **Status over chrome** — connectivity and index health always visible  
2. **Sources are first-class** — every answer can open provenance  
3. **Dense, calm productivity** — high information density without clutter  
4. **Honest empty and failure states** — never fake generation success  
5. **Local-first language** — privacy and offline are product features, not footnotes  
6. **Keyboard-friendly** — primary workflows usable without a mouse  

## Navigation model

```text
Workspace
├── Chat              Primary Q&A + citations + source panel
├── Documents         Upload, index progress, inventory, delete
├── Sources           Inspect selected retrieval provenance
├── Knowledge Graph   Graph export when available (Streamlit-parity data)
└── Settings          Thresholds, context size, connection diagnostics
```

## Critical states (must be explicit)

- Backend starting / unavailable  
- LM Studio unavailable / no model loaded  
- No documents indexed  
- Upload / indexing in progress or failed  
- Query running / response ready / generation failed  
- Retrieval returned no useful context  
- Offline mode active  

## Non-goals for the UI migration

- Hosted accounts or multi-tenant auth  
- Cloud vector databases  
- Telemetry / analytics SDKs  
- Rewriting retrieval quality experiments into the UI layer  

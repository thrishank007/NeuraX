"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, HelpCircle } from "lucide-react";

export function HelpPanel({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="rounded-lg border border-border bg-surface shadow-2xs overflow-hidden">
      <button
        type="button"
        className="flex w-full items-center justify-between px-4 py-3 text-left text-xs font-semibold text-text uppercase tracking-wider font-mono hover:bg-surface-2 transition-colors cursor-pointer"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <HelpCircle className="h-4 w-4 text-accent" />
          <span>Operator Manual &amp; Directives</span>
        </div>
        {open ? (
          <ChevronDown className="h-4 w-4 text-dim" aria-hidden />
        ) : (
          <ChevronRight className="h-4 w-4 text-dim" aria-hidden />
        )}
      </button>

      {open && (
        <div className="space-y-4 border-t border-border-subtle p-4 text-xs leading-relaxed text-muted bg-surface-2/40">
          <div className="space-y-1">
            <h3 className="font-semibold text-text font-mono text-[11px] uppercase text-accent">
              01. Document Ingestion Protocol
            </h3>
            <ol className="list-decimal space-y-0.5 pl-4 font-mono text-[11px] text-text/90">
              <li>Drop PDF, DOCX, TXT, images (OCR), or audio files (Whisper).</li>
              <li>Monitor real-time chunking and vectorization progress bar.</li>
              <li>Review indexed items in the collection inventory table.</li>
            </ol>
          </div>

          <div className="space-y-1">
            <h3 className="font-semibold text-text font-mono text-[11px] uppercase text-signal-cyan">
              02. Multimodal Search vs. Chat Briefing
            </h3>
            <ul className="list-disc space-y-0.5 pl-4 font-mono text-[11px] text-text/90">
              <li>
                <strong>Multi-Search</strong>: Direct similarity vector matching across text, CLIP visual embeddings, and audio transcripts without LLM generation.
              </li>
              <li>
                <strong>Chat Briefing</strong>: Streaming grounded reasoning powered by local LM Studio with verified citation provenance.
              </li>
            </ul>
          </div>

          <div className="space-y-1">
            <h3 className="font-semibold text-text font-mono text-[11px] uppercase text-signal-amber">
              03. Air-Gapped Security Architecture
            </h3>
            <p className="text-[11px] font-mono text-text/90">
              All vector storage, token generation, and speech/vision processing execute entirely on local hardware (FastAPI + ChromaDB + LM Studio). Zero telemetry, tracking, or external cloud requests.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

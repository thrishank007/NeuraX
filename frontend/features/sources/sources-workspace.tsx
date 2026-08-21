"use client";

import { useEffect, useState } from "react";
import {
  FileText,
  Library,
  ExternalLink,
  Search,
  Sparkles,
  Layers,
  FileCode,
  FileAudio,
  FileImage,
} from "lucide-react";
import type { CitationItem, SearchResultItem } from "@/types/api";
import { Badge } from "@/components/ui/badge";

interface StoredProvenance {
  sources: SearchResultItem[];
  citations: CitationItem[];
  query: string;
}

export function SourcesWorkspace() {
  const [data, setData] = useState<StoredProvenance | null>(null);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("neurax-last-sources");
      if (raw) {
        setData(JSON.parse(raw) as StoredProvenance);
      }
    } catch {
      /* ignore */
    }
  }, []);

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-4 sm:p-8">
      <header className="space-y-1">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase text-accent">
          <Library className="h-3.5 w-3.5" />
          <span>Provenance Explorer</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-text">
          Grounding Citations &amp; Evidence
        </h1>
        <p className="text-xs text-muted">
          Deep-dive inspection into the exact context chunks and confidence scores from your latest query.
        </p>
      </header>

      {data ? (
        <div className="space-y-6">
          {data.query && (
            <div className="rounded-2xl border border-border bg-surface p-4 text-xs text-text">
              <span className="text-dim uppercase text-[10px] font-bold block mb-1">Originating Query:</span>
              <span className="font-semibold text-accent text-sm">{data.query}</span>
            </div>
          )}

          {data.citations && data.citations.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-xs font-bold uppercase text-text">
                Verified Citations ({data.citations.length})
              </h2>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {data.citations.map((c) => (
                  <div
                    key={`cit-${c.citation_id}`}
                    className="rounded-2xl border border-border bg-surface p-4 space-y-3 shadow-hud text-xs"
                  >
                    <div className="flex items-center justify-between gap-2 border-b border-border/50 pb-2">
                      <span className="font-bold text-accent px-2 py-0.5 rounded bg-accent/15">
                        [{c.citation_id}]
                      </span>
                      <span className="truncate font-semibold text-text text-xs">
                        {c.file_path.split(/[/\\]/).pop() || c.source_document}
                      </span>
                      <Badge tone="emerald">
                        {(c.confidence_score * 100).toFixed(0)}%
                      </Badge>
                    </div>

                    <p className="text-xs text-text/90 leading-relaxed bg-surface-2 p-3 rounded-xl">
                      {c.content_snippet}
                    </p>

                    <div className="flex items-center justify-between text-[11px] text-dim pt-1 font-medium">
                      <span>Type: {c.source_type || "text"}</span>
                      {c.page_number != null && <span>Page: {c.page_number}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {data.sources && data.sources.length > 0 && (
            <section className="space-y-3 pt-2">
              <h2 className="text-xs font-bold uppercase text-text">
                Retrieved Vector Chunks ({data.sources.length})
              </h2>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {data.sources.map((s, idx) => (
                  <div
                    key={s.id || idx}
                    className="rounded-2xl border border-border bg-surface p-4 space-y-2.5 shadow-hud text-xs"
                  >
                    <div className="flex items-center justify-between gap-2 border-b border-border/50 pb-2">
                      <span className="truncate font-semibold text-text">
                        {s.file_name || s.file_path}
                      </span>
                      <span className="text-xs text-signal-cyan font-bold">
                        Score: {s.similarity_score.toFixed(2)}
                      </span>
                    </div>

                    <p className="text-xs text-text/90 leading-relaxed bg-surface-2 p-3 rounded-xl">
                      {s.content_preview}
                    </p>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      ) : (
        <div className="rounded-2xl border border-border bg-surface p-12 text-center text-xs text-dim shadow-hud">
          No recent query context stored in session. Run a search or chat briefing to view grounding provenance records.
        </div>
      )}
    </div>
  );
}

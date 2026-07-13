"use client";

import { useEffect, useState } from "react";
import type { CitationItem, SearchResultItem } from "@/types/api";

type Stored = {
  query?: string;
  sources?: SearchResultItem[];
  citations?: CitationItem[];
};

export function SourcesWorkspace() {
  const [data, setData] = useState<Stored>({});

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("neurax-last-sources");
      if (raw) setData(JSON.parse(raw) as Stored);
    } catch {
      /* ignore */
    }
  }, []);

  const citations = data.citations || [];
  const sources = data.sources || [];

  return (
    <div className="mx-auto max-w-3xl space-y-4 p-4">
      <header>
        <h1 className="text-base font-semibold">Sources</h1>
        <p className="text-xs text-muted">
          Inspect provenance from the last chat retrieval. Fields are limited to
          what the backend provides.
        </p>
      </header>

      {data.query && (
        <p className="rounded-md border border-border bg-surface px-3 py-2 text-sm">
          <span className="text-muted">Last query: </span>
          {data.query}
        </p>
      )}

      {citations.length === 0 && sources.length === 0 && (
        <div className="rounded-lg border border-dashed border-border bg-surface p-6 text-sm text-muted">
          No sources yet. Run a chat query to populate this panel.
        </div>
      )}

      {citations.map((c) => (
        <article
          key={c.citation_id}
          className="rounded-lg border border-border bg-surface p-3"
        >
          <h2 className="text-sm font-medium">
            [{c.citation_id}]{" "}
            {c.file_path.split(/[/\\]/).pop() || c.source_document}
          </h2>
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            <dt className="text-muted">Type</dt>
            <dd>{c.source_type || "—"}</dd>
            <dt className="text-muted">Confidence</dt>
            <dd className="font-mono">{c.confidence_score.toFixed(3)}</dd>
            <dt className="text-muted">Page</dt>
            <dd>{c.page_number ?? "—"}</dd>
            <dt className="text-muted">Path</dt>
            <dd className="break-all font-mono">{c.file_path || "—"}</dd>
          </dl>
          <p className="mt-2 text-sm leading-relaxed">{c.content_snippet}</p>
        </article>
      ))}

      {citations.length === 0 &&
        sources.map((s) => (
          <article
            key={s.id || s.file_path}
            className="rounded-lg border border-border bg-surface p-3"
          >
            <h2 className="text-sm font-medium">
              {s.file_name || s.file_path || s.id}
            </h2>
            <p className="font-mono text-xs text-muted">
              score {s.similarity_score.toFixed(3)} · {s.file_type}
            </p>
            <p className="mt-2 text-sm">{s.content_preview}</p>
          </article>
        ))}
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { api, ApiClientError } from "@/lib/api";
import { Button } from "@/components/ui/button";

export function GraphWorkspace() {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function load(signal?: AbortSignal) {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getGraph(signal);
      setStats(res.stats || {});
    } catch (err) {
      setStats(null);
      if (err instanceof ApiClientError) {
        setError(
          err.code === "not_found"
            ? "Knowledge graph data is not available yet. Graph views in Gradio were Streamlit-only; this endpoint wraps export_viz_data when the KG manager can initialize."
            : err.message,
        );
      } else {
        setError("Failed to load graph");
      }
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const c = new AbortController();
    void load(c.signal);
    return () => c.abort();
  }, []);

  return (
    <div className="mx-auto max-w-3xl space-y-4 p-4">
      <header className="flex items-start justify-between gap-2">
        <div>
          <h1 className="text-base font-semibold">Knowledge Graph</h1>
          <p className="text-xs text-muted">
            Optional security graph export from existing Python modules
          </p>
        </div>
        <Button variant="secondary" size="sm" onClick={() => void load()}>
          Refresh
        </Button>
      </header>

      {loading && <p className="text-sm text-muted">Loading…</p>}
      {error && (
        <div
          className="rounded-lg border border-border bg-surface p-4 text-sm text-muted"
          role="status"
        >
          {error}
        </div>
      )}
      {stats && !error && (
        <pre className="overflow-auto rounded-lg border border-border bg-surface p-3 font-mono text-xs">
          {JSON.stringify(stats, null, 2)}
        </pre>
      )}
    </div>
  );
}

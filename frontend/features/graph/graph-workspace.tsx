"use client";

import { useEffect, useState } from "react";
import {
  GitBranch,
  Play,
  RefreshCw,
  Search,
  Route,
  Activity,
  Layers,
  Sparkles,
  Loader2,
} from "lucide-react";
import { api, ApiClientError } from "@/lib/api";
import type { GraphifyStatus } from "@/types/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export function GraphWorkspace() {
  const [status, setStatus] = useState<GraphifyStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [building, setBuilding] = useState(false);
  const [query, setQuery] = useState("");
  const [queryAnswer, setQueryAnswer] = useState<string | null>(null);
  const [sourceNode, setSourceNode] = useState("");
  const [targetNode, setTargetNode] = useState("");
  const [pathResult, setPathResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function loadStatus() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.graphifyStatus();
      setStatus(res);
    } catch (err) {
      setError(
        err instanceof ApiClientError
          ? err.message
          : "Failed to load knowledge graph stats",
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadStatus();
  }, []);

  async function handleBuild() {
    setBuilding(true);
    setError(null);
    try {
      await api.graphifyBuild(true);
      void loadStatus();
    } catch (err) {
      setError(
        err instanceof ApiClientError
          ? err.message
          : "Knowledge graph build job failed",
      );
    } finally {
      setBuilding(false);
    }
  }

  async function handleQuery() {
    if (!query.trim()) return;
    setError(null);
    try {
      const res = await api.graphifyQuery({ question: query.trim() });
      setQueryAnswer(res.output || JSON.stringify(res));
    } catch (err) {
      setError(
        err instanceof ApiClientError ? err.message : "Graph query execution failed",
      );
    }
  }

  async function handleFindPath() {
    if (!sourceNode.trim() || !targetNode.trim()) return;
    setError(null);
    try {
      const res = await api.graphifyPath({
        source: sourceNode.trim(),
        target: targetNode.trim(),
      });
      setPathResult(res.output || "Path search finished.");
    } catch (err) {
      setError(
        err instanceof ApiClientError ? err.message : "Path query failed",
      );
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-4 sm:p-8">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase text-accent">
            <GitBranch className="h-3.5 w-3.5" />
            <span>Knowledge Graph</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-text">
            Entity Graph &amp; Ontologies
          </h1>
          <p className="text-xs text-muted">
            Explore concepts, entities, and cross-document relationships indexed via Graphify.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => void loadStatus()}
            disabled={loading}
            className="rounded-xl font-medium"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          </Button>

          <Button
            variant="primary"
            size="sm"
            onClick={() => void handleBuild()}
            disabled={building}
            className="rounded-xl px-5 font-semibold"
          >
            {building ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
                <span>Building…</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 mr-1.5" />
                <span>Build Graph</span>
              </>
            )}
          </Button>
        </div>
      </header>

      {error && (
        <div className="text-xs text-signal-rose bg-signal-rose/10 p-3.5 rounded-xl border border-signal-rose/20" role="alert">
          {error}
        </div>
      )}

      {/* Telemetry Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="rounded-2xl border border-border bg-surface p-4 shadow-hud space-y-1">
          <span className="text-xs font-medium text-muted uppercase">Corpus Files</span>
          <p className="text-2xl font-bold text-text">{status?.corpus_file_count ?? 0}</p>
        </div>

        <div className="rounded-2xl border border-border bg-surface p-4 shadow-hud space-y-1">
          <span className="text-xs font-medium text-muted uppercase">Artifacts</span>
          <p className="text-2xl font-bold text-accent">
            {status?.artifacts_available ? "Ready" : "None"}
          </p>
        </div>

        <div className="rounded-2xl border border-border bg-surface p-4 shadow-hud space-y-1">
          <span className="text-xs font-medium text-muted uppercase">Engine Backend</span>
          <p className="text-sm font-bold text-signal-cyan capitalize">
            {status?.backend || "Graphify"}
          </p>
        </div>

        <div className="rounded-2xl border border-border bg-surface p-4 shadow-hud space-y-1">
          <span className="text-xs font-medium text-muted uppercase">Status</span>
          <p className="text-sm font-bold text-text capitalize">
            {status?.build_running ? "Building" : status?.available ? "Online" : "Offline"}
          </p>
        </div>
      </div>

      {/* Natural Language Graph Query Box */}
      <div className="rounded-2xl border border-border bg-surface shadow-hud p-6 space-y-3">
        <div className="flex items-center gap-2 text-xs font-bold uppercase text-text">
          <Search className="h-4 w-4 text-accent" />
          <span>Natural Language Graph Query</span>
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask questions about relationships, e.g. How is component A linked to service B?"
            className="flex-1 rounded-xl border border-border bg-surface-2 px-4 py-2.5 text-xs sm:text-sm text-text placeholder:text-dim focus:outline-none focus:border-accent/40"
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleQuery();
            }}
          />
          <Button
            variant="primary"
            size="sm"
            onClick={() => void handleQuery()}
            disabled={!query.trim()}
            className="rounded-xl px-5 font-semibold"
          >
            Query
          </Button>
        </div>

        {queryAnswer && (
          <div className="rounded-xl border border-border bg-surface-2 p-4 text-xs sm:text-sm text-text leading-relaxed whitespace-pre-wrap">
            {queryAnswer}
          </div>
        )}
      </div>

      {/* Shortest Path Finder */}
      <div className="rounded-2xl border border-border bg-surface shadow-hud p-6 space-y-3">
        <div className="flex items-center gap-2 text-xs font-bold uppercase text-text">
          <Route className="h-4 w-4 text-signal-cyan" />
          <span>Entity Path Finder</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <input
            type="text"
            value={sourceNode}
            onChange={(e) => setSourceNode(e.target.value)}
            placeholder="Source Entity Name"
            className="rounded-xl border border-border bg-surface-2 px-4 py-2.5 text-xs sm:text-sm text-text placeholder:text-dim focus:outline-none"
          />
          <input
            type="text"
            value={targetNode}
            onChange={(e) => setTargetNode(e.target.value)}
            placeholder="Target Entity Name"
            className="rounded-xl border border-border bg-surface-2 px-4 py-2.5 text-xs sm:text-sm text-text placeholder:text-dim focus:outline-none"
          />
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={() => void handleFindPath()}
          disabled={!sourceNode.trim() || !targetNode.trim()}
          className="rounded-xl font-medium"
        >
          Find Path
        </Button>

        {pathResult && (
          <div className="rounded-xl border border-border bg-surface-2 p-4 text-xs sm:text-sm text-text leading-relaxed whitespace-pre-wrap">
            {pathResult}
          </div>
        )}
      </div>
    </div>
  );
}

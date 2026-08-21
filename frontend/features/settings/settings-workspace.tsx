"use client";

import { useEffect, useState } from "react";
import { api, ApiClientError, isAbortError } from "@/lib/api";
import type { ModelsStatus, SystemStatus } from "@/types/api";
import { getApiBaseUrl } from "@/lib/env";
import { HelpPanel } from "@/features/help/help-panel";
import {
  Activity,
  Cloud,
  Cpu,
  Database,
  HardDrive,
  RefreshCw,
  Server,
  Settings,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export function SettingsWorkspace() {
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [models, setModels] = useState<ModelsStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function loadData(signal?: AbortSignal) {
    setLoading(true);
    setError(null);
    try {
      const [s, m] = await Promise.all([
        api.systemStatus(signal),
        api.modelsStatus(signal),
      ]);
      setSystem(s);
      setModels(m);
    } catch (err) {
      if (isAbortError(err) || signal?.aborted) return;
      setError(
        err instanceof ApiClientError
          ? err.message
          : "Could not connect to backend diagnostics service",
      );
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }

  useEffect(() => {
    const c = new AbortController();
    void loadData(c.signal);
    return () => c.abort();
  }, []);

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-4 sm:p-8">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase text-accent">
            <Settings className="h-3.5 w-3.5" />
            <span>Diagnostics &amp; System Health</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-text">
            Offline Engine Status
          </h1>
          <p className="text-xs text-muted">
            Health metrics for FastAPI backend, ChromaDB vector store, and local LM Studio models.
          </p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={() => void loadData()}
          disabled={loading}
          className="rounded-xl gap-1.5 font-medium"
        >
          <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          <span>Refresh</span>
        </Button>
      </header>

      {error && (
        <div className="text-xs text-signal-rose bg-signal-rose/10 p-3.5 rounded-xl border border-signal-rose/20" role="alert">
          {error}
        </div>
      )}

      {/* Services Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="rounded-2xl border border-border bg-surface p-5 shadow-hud space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-muted uppercase">FastAPI Core</span>
            <Badge tone={system?.backend === "ok" ? "emerald" : "warning"}>
              {system?.backend?.toUpperCase() || "UNKNOWN"}
            </Badge>
          </div>
          <p className="text-xs text-dim">Local HTTP REST API Service</p>
        </div>

        <div className="rounded-2xl border border-border bg-surface p-5 shadow-hud space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-muted uppercase">ChromaDB Store</span>
            <Badge tone={system?.vector_store === "ok" ? "emerald" : "warning"}>
              {system?.vector_store?.toUpperCase() || "UNKNOWN"}
            </Badge>
          </div>
          <p className="text-xs text-dim">
            {system?.collection?.total_documents ?? 0} indexed documents
          </p>
        </div>

        <div className="rounded-2xl border border-border bg-surface p-5 shadow-hud space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-muted uppercase">LM Studio</span>
            <Badge tone={models?.lm_studio_reachable ? "emerald" : "danger"}>
              {models?.lm_studio_reachable ? "ONLINE" : "OFFLINE"}
            </Badge>
          </div>
          <p className="text-xs text-dim">
            {models?.models_loaded ?? 0} active models loaded
          </p>
        </div>
      </div>

      {/* LM Studio Details */}
      <section className="rounded-2xl border border-border bg-surface shadow-hud p-6 space-y-3">
        <div className="flex items-center gap-2 border-b border-border pb-3">
          <Cpu className="h-4 w-4 text-accent" />
          <h2 className="text-xs font-bold text-text uppercase tracking-wider">
            Local LLM Registry
          </h2>
        </div>

        {models?.model_ids && models.model_ids.length > 0 ? (
          <div className="space-y-2">
            <span className="text-xs text-dim">Loaded Model Endpoints:</span>
            <div className="space-y-1.5">
              {models.model_ids.map((id) => (
                <div
                  key={id}
                  className="flex items-center justify-between p-3.5 rounded-xl border border-border bg-surface-2 text-xs font-medium text-text"
                >
                  <span>{id}</span>
                  <Badge tone="cyan">ACTIVE</Badge>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-xs text-muted">
            No active models found in LM Studio. Open LM Studio and start local server at port 1234.
          </p>
        )}
      </section>

      {/* Network & Storage Capacity */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="rounded-2xl border border-border bg-surface shadow-hud p-5 space-y-2">
          <span className="text-xs font-medium text-muted uppercase">API Base URL</span>
          <div className="rounded-xl border border-border bg-surface-2 p-3 text-xs font-mono text-accent break-all">
            {getApiBaseUrl()}
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-surface shadow-hud p-5 space-y-2">
          <span className="text-xs font-medium text-muted uppercase">Vector Store Capacity</span>
          <dl className="grid grid-cols-2 gap-2 text-xs">
            <dt className="text-dim">Collection:</dt>
            <dd className="text-text font-semibold">{system?.collection?.collection_name || "neurax_docs"}</dd>
            <dt className="text-dim">Max Upload:</dt>
            <dd className="text-text">{system?.max_upload_mb || 100} MB</dd>
          </dl>
        </div>
      </div>

      <HelpPanel defaultOpen={false} />
    </div>
  );
}

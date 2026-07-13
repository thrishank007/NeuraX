"use client";

import { useEffect, useState } from "react";
import { api, ApiClientError } from "@/lib/api";
import type { ModelsStatus, SystemStatus } from "@/types/api";
import { getApiBaseUrl } from "@/lib/env";
import { StatusChip, mapSystemKind } from "@/components/status/status-chip";
import { HelpPanel } from "@/features/help/help-panel";

export function SettingsWorkspace() {
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [models, setModels] = useState<ModelsStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const c = new AbortController();
    (async () => {
      try {
        const [s, m] = await Promise.all([
          api.systemStatus(c.signal),
          api.modelsStatus(c.signal),
        ]);
        setSystem(s);
        setModels(m);
      } catch (err) {
        setError(
          err instanceof ApiClientError
            ? err.message
            : "Could not load settings diagnostics",
        );
      }
    })();
    return () => c.abort();
  }, []);

  return (
    <div className="mx-auto max-w-2xl space-y-4 p-4">
      <header>
        <h1 className="text-base font-semibold">Settings</h1>
        <p className="text-xs text-muted">
          Local connection diagnostics. No cloud accounts or telemetry.
        </p>
      </header>

      {error && (
        <p className="text-sm text-danger" role="alert">
          {error}
        </p>
      )}

      <section className="space-y-2 rounded-lg border border-border bg-surface p-3">
        <h2 className="text-sm font-medium">API</h2>
        <p className="font-mono text-xs break-all">{getApiBaseUrl()}</p>
        <p className="text-xs text-muted">
          Set <code className="font-mono">NEXT_PUBLIC_API_URL</code> in{" "}
          <code className="font-mono">frontend/.env.local</code>
        </p>
      </section>

      <section className="space-y-2 rounded-lg border border-border bg-surface p-3">
        <h2 className="text-sm font-medium">System</h2>
        <div className="flex flex-wrap gap-2">
          <StatusChip
            kind={mapSystemKind(system?.backend || "unknown")}
            label={`Backend: ${system?.backend || "unknown"}`}
          />
          <StatusChip
            kind={mapSystemKind(system?.vector_store || "unknown")}
            label={`Vector store: ${system?.vector_store || "unknown"}`}
          />
          <StatusChip
            kind={mapSystemKind(system?.lm_studio || "unknown")}
            label={`LM Studio: ${system?.lm_studio || "unknown"}`}
          />
        </div>
        {system && (
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            <dt className="text-muted">Collection</dt>
            <dd className="font-mono">
              {system.collection?.collection_name || "—"} (
              {system.collection?.total_documents ?? 0} docs)
            </dd>
            <dt className="text-muted">Max upload</dt>
            <dd>{system.max_upload_mb} MB</dd>
            <dt className="text-muted">Formats</dt>
            <dd className="font-mono text-[11px]">
              {system.supported_formats?.join(" ") || "—"}
            </dd>
          </dl>
        )}
      </section>

      <section className="space-y-2 rounded-lg border border-border bg-surface p-3">
        <h2 className="text-sm font-medium">Models</h2>
        {models ? (
          <>
            <p className="text-sm">{models.message}</p>
            {models.model_ids?.length > 0 && (
              <ul className="list-disc pl-5 font-mono text-xs text-muted">
                {models.model_ids.map((id) => (
                  <li key={id}>{id}</li>
                ))}
              </ul>
            )}
          </>
        ) : (
          <p className="text-sm text-muted">No model status yet.</p>
        )}
      </section>

      <section className="rounded-lg border border-border bg-surface p-3 text-xs text-muted">
        <p>
          Chat similarity threshold and max context documents are controlled on
          the Chat page. Multimodal retrieval (text, image, voice) is on the
          Search page. Retrieval defaults come from Python{" "}
          <code className="font-mono">SEARCH_CONFIG</code>.
        </p>
      </section>

      <HelpPanel defaultOpen />
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { api, ApiClientError } from "@/lib/api";
import type { ModelsStatus, SystemStatus } from "@/types/api";
import { StatusChip, mapSystemKind } from "@/components/status/status-chip";

export function StatusBar() {
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [models, setModels] = useState<ModelsStatus | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    async function load() {
      try {
        const [s, m] = await Promise.all([
          api.systemStatus(controller.signal),
          api.modelsStatus(controller.signal),
        ]);
        if (cancelled) return;
        setSystem(s);
        setModels(m);
        setBackendError(null);
      } catch (err) {
        if (cancelled) return;
        setSystem(null);
        setModels(null);
        setBackendError(
          err instanceof ApiClientError
            ? err.message
            : "Backend unavailable",
        );
      }
    }

    load();
    const id = window.setInterval(load, 15000);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(id);
    };
  }, []);

  const docs = system?.collection?.total_documents ?? null;

  return (
    <div
      className="flex flex-wrap items-center gap-2 border-b border-border bg-surface-2/60 px-3 py-1.5"
      role="status"
      aria-live="polite"
    >
      {backendError ? (
        <StatusChip kind="unavailable" label="Backend unavailable" />
      ) : (
        <>
          <StatusChip
            kind={mapSystemKind(system?.backend || "loading")}
            label="Backend ready"
          />
          <StatusChip
            kind={mapSystemKind(system?.vector_store || "unknown")}
            label={`Index: ${docs == null ? "…" : `${docs} docs`}`}
          />
          <StatusChip
            kind={
              models?.lm_studio_reachable
                ? models.models_loaded > 0
                  ? "ok"
                  : "degraded"
                : "unavailable"
            }
            label={
              models?.lm_studio_reachable
                ? models.models_loaded > 0
                  ? `LM Studio: ${models.models_loaded} model(s)`
                  : "LM Studio: no model loaded"
                : "LM Studio unavailable"
            }
          />
          {system?.offline_mode && (
            <StatusChip kind="offline" label="Offline mode" />
          )}
        </>
      )}
      {backendError && (
        <span className="text-xs text-danger">{backendError}</span>
      )}
    </div>
  );
}

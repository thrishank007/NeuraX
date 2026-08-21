"use client";

import { useEffect, useState } from "react";
import { api, ApiClientError, isAbortError } from "@/lib/api";
import type { ModelsStatus, SystemStatus } from "@/types/api";
import { cn } from "@/lib/utils";

export function StatusBar() {
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [models, setModels] = useState<ModelsStatus | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let active: AbortController | null = null;

    async function load() {
      active?.abort();
      const controller = new AbortController();
      active = controller;
      try {
        const [s, m] = await Promise.all([
          api.systemStatus(controller.signal),
          api.modelsStatus(controller.signal),
        ]);
        if (cancelled || controller.signal.aborted) return;
        setSystem(s);
        setModels(m);
        setBackendError(null);
      } catch (err) {
        if (cancelled || isAbortError(err) || controller.signal.aborted) return;
        setSystem(null);
        setModels(null);
        setBackendError(
          err instanceof ApiClientError
            ? err.message
            : "Backend offline",
        );
      }
    }

    load();
    const id = window.setInterval(load, 15000);
    return () => {
      cancelled = true;
      active?.abort();
      window.clearInterval(id);
    };
  }, []);

  const docs = system?.collection?.total_documents ?? null;

  return (
    <div className="flex items-center gap-2 text-xs">
      {backendError ? (
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-signal-rose/10 border border-signal-rose/20 text-signal-rose text-xs font-medium">
          <span className="h-1.5 w-1.5 rounded-full bg-signal-rose" />
          <span>Backend Offline</span>
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-surface-2 border border-border text-xs text-muted">
          <span className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse-dot" />
          <span className="text-text font-medium">Air-Gapped</span>
          <span className="text-border mx-0.5">•</span>
          <span className="font-medium">{docs == null ? "…" : `${docs} docs`}</span>
          <span className="text-border mx-0.5">•</span>
          <span className={cn("font-medium", models?.lm_studio_reachable ? "text-accent" : "text-signal-amber")}>
            {models?.lm_studio_reachable
              ? `${models.models_loaded} model${models.models_loaded === 1 ? "" : "s"}`
              : "LM Studio off"}
          </span>
        </div>
      )}
    </div>
  );
}

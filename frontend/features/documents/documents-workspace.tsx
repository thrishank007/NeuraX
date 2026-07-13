"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Trash2, Upload } from "lucide-react";
import { api, ApiClientError } from "@/lib/api";
import type { DocumentItem, JobStatus } from "@/types/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const ACCEPT =
  ".pdf,.docx,.doc,.txt,.jpg,.jpeg,.png,.bmp,.tiff,.webp,.wav,.mp3,.m4a,.flac,.ogg";

export function DocumentsWorkspace() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [filter, setFilter] = useState("");
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const pollRef = useRef<number | null>(null);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const res = await api.listDocuments(signal);
      setDocs(res.documents);
      setError(null);
    } catch (err) {
      setError(
        err instanceof ApiClientError
          ? err.message
          : "Failed to load documents",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const c = new AbortController();
    void refresh(c.signal);
    return () => c.abort();
  }, [refresh]);

  useEffect(() => {
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  function startPolling(jobId: string) {
    if (pollRef.current) window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(async () => {
      try {
        const j = await api.getJob(jobId);
        setJob(j);
        if (
          j.status === "completed" ||
          j.status === "failed" ||
          j.status === "cancelled"
        ) {
          if (pollRef.current) window.clearInterval(pollRef.current);
          pollRef.current = null;
          void refresh();
        }
      } catch {
        /* keep last job state */
      }
    }, 1000);
  }

  async function uploadFiles(fileList: FileList | File[]) {
    const files = Array.from(fileList);
    if (!files.length) return;
    setError(null);
    try {
      const j = await api.uploadDocuments(files);
      setJob(j);
      startPolling(j.job_id);
    } catch (err) {
      setError(
        err instanceof ApiClientError ? err.message : "Upload failed",
      );
    }
  }

  async function removeDoc(id: string) {
    try {
      await api.deleteDocument(id);
      setConfirmId(null);
      void refresh();
    } catch (err) {
      setError(
        err instanceof ApiClientError ? err.message : "Delete failed",
      );
    }
  }

  const filtered = docs.filter((d) => {
    const q = filter.toLowerCase();
    if (!q) return true;
    return (
      d.file_name.toLowerCase().includes(q) ||
      d.file_type.toLowerCase().includes(q) ||
      d.id.toLowerCase().includes(q)
    );
  });

  return (
    <div className="mx-auto max-w-5xl space-y-4 p-4">
      <header>
        <h1 className="text-base font-semibold">Documents</h1>
        <p className="text-xs text-muted">
          Upload and index local files. Supported: PDF, DOCX, DOC, TXT, images,
          audio.
        </p>
      </header>

      <div
        className={cn(
          "rounded-lg border border-dashed p-6 text-center transition-colors",
          dragOver
            ? "border-accent bg-accent-subtle"
            : "border-border bg-surface",
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (e.dataTransfer.files?.length) {
            void uploadFiles(e.dataTransfer.files);
          }
        }}
      >
        <Upload className="mx-auto h-8 w-8 text-muted" aria-hidden />
        <p className="mt-2 text-sm text-text">Drag and drop files here</p>
        <p className="text-xs text-muted">or</p>
        <Button
          className="mt-2"
          variant="secondary"
          onClick={() => inputRef.current?.click()}
        >
          Choose files
        </Button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT}
          className="sr-only"
          aria-label="Upload documents"
          onChange={(e) => {
            if (e.target.files) void uploadFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {job && (
        <div
          className="rounded-lg border border-border bg-surface p-3"
          role="status"
          aria-live="polite"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-sm">
              {(job.status === "running" || job.status === "queued") && (
                <Loader2 className="h-4 w-4 animate-spin text-info" />
              )}
              <span className="font-medium">Indexing: {job.status}</span>
              <Badge tone="neutral">
                {job.processed}/{job.total}
              </Badge>
            </div>
            {(job.status === "running" || job.status === "queued") && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => void api.cancelJob(job.job_id).then(setJob)}
              >
                Cancel
              </Button>
            )}
          </div>
          <div className="mt-2 h-2 overflow-hidden rounded bg-surface-2">
            <div
              className="h-full bg-accent transition-[width]"
              style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
            />
          </div>
          {job.logs.length > 0 && (
            <pre className="mt-2 max-h-32 overflow-auto rounded bg-surface-2 p-2 font-mono text-[11px] text-muted">
              {job.logs.slice(-12).join("\n")}
            </pre>
          )}
          {job.errors.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-xs text-danger">
              {job.errors.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {error && (
        <p className="text-sm text-danger" role="alert">
          {error}
        </p>
      )}

      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">
          Collection inventory ({docs.length})
        </h2>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter…"
          aria-label="Filter documents"
          className="h-9 w-48 rounded-md border border-border bg-surface px-2 text-sm"
        />
      </div>

      {loading ? (
        <p className="text-sm text-muted">Loading documents…</p>
      ) : filtered.length === 0 ? (
        <div className="rounded-lg border border-border bg-surface p-6 text-sm text-muted">
          No documents indexed yet. Upload files to build the collection.
        </div>
      ) : (
        <ul className="divide-y divide-border rounded-lg border border-border bg-surface">
          {filtered.map((d) => (
            <li
              key={d.id}
              className="flex flex-wrap items-start justify-between gap-2 px-3 py-2"
            >
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">
                  {d.file_name || d.id}
                </div>
                <div className="font-mono text-[11px] text-muted">
                  {d.file_type || "unknown"} · {d.id}
                </div>
                {d.content_preview && (
                  <p className="mt-1 line-clamp-2 text-xs text-muted">
                    {d.content_preview}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-2">
                {confirmId === d.id ? (
                  <>
                    <Button
                      size="sm"
                      variant="danger"
                      onClick={() => void removeDoc(d.id)}
                    >
                      Confirm delete
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setConfirmId(null)}
                    >
                      Cancel
                    </Button>
                  </>
                ) : (
                  <Button
                    size="sm"
                    variant="ghost"
                    aria-label={`Delete ${d.file_name || d.id}`}
                    onClick={() => setConfirmId(d.id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import {
  FileText,
  FolderUp,
  Loader2,
  Trash2,
  Upload,
  RefreshCw,
  Library,
  FileCheck,
  HardDrive,
  FileCode,
  FileAudio,
  FileImage,
} from "lucide-react";
import { api, ApiClientError, isAbortError } from "@/lib/api";
import type { DocumentItem, JobStatus } from "@/types/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

function getFileIcon(ext?: string) {
  const e = (ext || "").toLowerCase();
  if (e.includes("pdf") || e.includes("doc") || e.includes("txt")) {
    return <FileText className="h-4 w-4 text-signal-cyan" />;
  }
  if (e.includes("jpg") || e.includes("png") || e.includes("image")) {
    return <FileImage className="h-4 w-4 text-accent" />;
  }
  if (e.includes("audio") || e.includes("wav") || e.includes("mp3")) {
    return <FileAudio className="h-4 w-4 text-signal-amber" />;
  }
  return <FileCode className="h-4 w-4 text-muted" />;
}

export function DocumentsWorkspace() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [files, setFiles] = useState<FileList | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  async function loadData(signal?: AbortSignal) {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listDocuments(signal);
      setDocuments(res.documents || []);
      setTotal(res.total || 0);
    } catch (err) {
      if (isAbortError(err) || signal?.aborted) return;
      setError(
        err instanceof ApiClientError ? err.message : "Failed to load document inventory",
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

  async function handleUpload() {
    if (!files || files.length === 0 || uploading) return;
    setUploading(true);
    setError(null);
    setLogs(["Submitting batch to background indexer…"]);
    setProgress(10);

    try {
      const arr = Array.from(files);
      const job = await api.uploadDocuments(arr);
      
      // Poll job status until done
      let done = false;
      while (!done) {
        await new Promise((r) => setTimeout(r, 1000));
        const status = await api.getJob(job.job_id);
        setProgress(Math.max(10, Math.min(100, Math.round((status.processed / (status.total || 1)) * 100))));
        if (status.logs && status.logs.length > 0) {
          setLogs(status.logs);
        }
        if (status.status === "completed" || status.status === "failed" || status.status === "cancelled") {
          done = true;
          if (status.status === "failed") {
            throw new Error(status.errors?.join(", ") || "Ingestion job failed");
          }
        }
      }

      setLogs((prev) => [...prev, `Ingestion completed successfully.`]);
      setFiles(null);
      void loadData();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Upload / ingestion pipeline failed",
      );
    } finally {
      setUploading(false);
      setProgress(null);
    }
  }

  async function handleDelete(docId: string, fileName: string) {
    if (!window.confirm(`Are you sure you want to delete ${fileName} from the vector index?`)) {
      return;
    }
    setDeleting(docId);
    try {
      await api.deleteDocument(docId);
      void loadData();
    } catch (err) {
      setError(
        err instanceof ApiClientError ? err.message : `Failed to delete ${fileName}`,
      );
    } finally {
      setDeleting(null);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-4 sm:p-8">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase text-accent">
            <Library className="h-3.5 w-3.5" />
            <span>Corpus Library</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-text">
            Indexed Documents &amp; Media
          </h1>
          <p className="text-xs text-muted">
            Manage local PDF, DOCX, text files, images, and audio files stored in ChromaDB.
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

      {/* Upload & Ingestion Dropzone */}
      <div className="rounded-2xl border border-border bg-surface shadow-hud p-6 space-y-4">
        <div className="flex items-center gap-2 text-xs font-bold uppercase text-text">
          <FolderUp className="h-4 w-4 text-accent" />
          <span>Ingest New Documents</span>
        </div>

        <div className="rounded-2xl border-2 border-dashed border-border hover:border-accent/40 bg-surface-2/50 p-6 text-center transition-colors">
          <Upload className="mx-auto h-8 w-8 text-dim mb-2" />
          <p className="text-sm font-semibold text-text">Drag and drop files here, or browse</p>
          <p className="text-xs text-muted mt-1">Supports PDF, DOCX, TXT, PNG, JPG, MP3, WAV</p>

          <input
            type="file"
            multiple
            onChange={(e) => setFiles(e.target.files)}
            className="mt-4 text-xs text-muted file:mr-3 file:py-1.5 file:px-4 file:rounded-xl file:border-0 file:text-xs file:bg-accent/15 file:text-accent file:font-semibold file:cursor-pointer"
          />

          {files && files.length > 0 && (
            <div className="mt-3 text-xs text-accent font-semibold">
              Ready to index: {files.length} file(s) selected
            </div>
          )}
        </div>

        {progress != null && (
          <div className="space-y-1.5 pt-2">
            <div className="flex justify-between text-xs font-medium text-muted">
              <span>Embedding &amp; Chunking Progress:</span>
              <span className="text-accent font-bold">{progress}%</span>
            </div>
            <div className="h-2 w-full rounded-full bg-surface-2 overflow-hidden border border-border">
              <div
                className="h-full bg-accent transition-all duration-300 rounded-full"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {logs.length > 0 && (
          <div className="rounded-xl border border-border bg-surface-2 p-3.5 text-xs text-muted max-h-32 overflow-y-auto scroll-panel space-y-1">
            {logs.map((l, i) => (
              <div key={i} className="leading-relaxed text-text/80">{l}</div>
            ))}
          </div>
        )}

        <div className="flex justify-end pt-2">
          <Button
            variant="primary"
            size="sm"
            onClick={() => void handleUpload()}
            disabled={!files || files.length === 0 || uploading}
            className="rounded-xl px-5 font-semibold"
          >
            {uploading ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
                <span>Processing…</span>
              </>
            ) : (
              <>
                <Upload className="h-3.5 w-3.5 mr-1.5" />
                <span>Start Ingestion</span>
              </>
            )}
          </Button>
        </div>
      </div>

      {error && (
        <div className="text-xs text-signal-rose bg-signal-rose/10 p-3.5 rounded-xl border border-signal-rose/20" role="alert">
          {error}
        </div>
      )}

      {/* Collection Inventory */}
      <section className="rounded-2xl border border-border bg-surface shadow-hud p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-border pb-3">
          <div className="flex items-center gap-2">
            <FileCheck className="h-4 w-4 text-accent" />
            <h2 className="text-xs font-bold text-text uppercase tracking-wider">
              Indexed Collection ({total} Documents)
            </h2>
          </div>
        </div>

        {documents.length > 0 ? (
          <div className="divide-y divide-border/60">
            {documents.map((doc) => (
              <div
                key={doc.id || doc.file_name}
                className="flex items-center justify-between py-3 px-1 gap-3 hover:bg-surface-2/40 transition-colors rounded-xl"
              >
                <div className="flex items-center gap-3 min-w-0">
                  {getFileIcon(doc.file_type || doc.file_name)}
                  <div className="min-w-0">
                    <p className="text-xs font-semibold text-text truncate">
                      {doc.file_name}
                    </p>
                    <p className="text-[11px] text-dim truncate">
                      {doc.file_type} • <span className="font-mono">{doc.file_path}</span>
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <Badge tone="cyan">INDEXED</Badge>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => void handleDelete(doc.id, doc.file_name)}
                    disabled={deleting === doc.id}
                    className="h-7 w-7 text-muted hover:text-signal-rose rounded-lg"
                    title={`Delete ${doc.file_name}`}
                  >
                    {deleting === doc.id ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Trash2 className="h-3.5 w-3.5" />
                    )}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="py-8 text-center text-xs text-dim">
            {loading ? "Loading corpus inventory…" : "No documents indexed yet. Upload files above to begin."}
          </div>
        )}
      </section>
    </div>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import { Image as ImageIcon, Loader2, Mic, Search, Shuffle, Trash2 } from "lucide-react";
import { api, ApiClientError } from "@/lib/api";
import type { SearchResultItem } from "@/types/api";
import {
  clearQueryHistory,
  loadQueryHistory,
  pushQueryHistory,
  storeLastSearch,
  type QueryHistoryEntry,
} from "@/lib/query-history";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type Modality = "text" | "image" | "voice" | "multimodal";

const MODES: { id: Modality; label: string; icon: typeof Search }[] = [
  { id: "text", label: "Text", icon: Search },
  { id: "image", label: "Image", icon: ImageIcon },
  { id: "voice", label: "Voice", icon: Mic },
  { id: "multimodal", label: "Multimodal", icon: Shuffle },
];

export function SearchWorkspace() {
  const [modality, setModality] = useState<Modality>("text");
  const [query, setQuery] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [k, setK] = useState(10);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState("Ready for queries");
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [history, setHistory] = useState<QueryHistoryEntry[]>([]);
  const imageRef = useRef<HTMLInputElement | null>(null);
  const audioRef = useRef<HTMLInputElement | null>(null);
  const multiImageRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    setHistory(loadQueryHistory());
  }, []);

  async function runSearch() {
    setError(null);
    setTranscription(null);

    if (modality === "text" && !query.trim()) {
      setError("Enter a text query");
      return;
    }
    if (modality === "image" && !imageFile) {
      setError("Upload an image for image search");
      return;
    }
    if (modality === "voice" && !audioFile) {
      setError("Upload an audio file for voice search");
      return;
    }
    if (modality === "multimodal") {
      if (!query.trim()) {
        setError("Enter text for multimodal search");
        return;
      }
      if (!imageFile) {
        setError("Upload an image for multimodal search");
        return;
      }
    }

    setBusy(true);
    setStatus("Searching…");
    const controller = new AbortController();

    try {
      const payload = await api.search(
        {
          query: query.trim(),
          modality,
          similarity_threshold: threshold,
          k,
          image: imageFile,
          audio: audioFile,
        },
        controller.signal,
      );

      setResults(payload.results);
      if (payload.transcription) {
        setTranscription(payload.transcription);
      }

      const label =
        modality === "image"
          ? "[Image Query]"
          : modality === "voice"
            ? payload.transcription || "[Voice Query]"
            : modality === "multimodal"
              ? `${query.trim()} + [Image]`
              : payload.query;

      if (payload.results.length) {
        setStatus(
          `Found ${payload.results.length} results in ${payload.processing_time.toFixed(2)}s`,
        );
      } else {
        setStatus(
          `No results (threshold ${payload.similarity_threshold.toFixed(2)}). Try lowering the threshold.`,
        );
      }

      const next = pushQueryHistory({
        query: label,
        type: modality,
        results_count: payload.results.length,
        processing_time: payload.processing_time,
        transcription: payload.transcription || undefined,
      });
      setHistory(next);
      storeLastSearch({
        query: label,
        sources: payload.results,
        modality,
      });
    } catch (err) {
      const msg =
        err instanceof ApiClientError ? err.message : "Search failed";
      setError(msg);
      setStatus("Search failed");
      setResults([]);
    } finally {
      setBusy(false);
    }
  }

  function clearAll() {
    setQuery("");
    setImageFile(null);
    setAudioFile(null);
    setResults([]);
    setTranscription(null);
    setError(null);
    setStatus("Ready for queries");
    if (imageRef.current) imageRef.current.value = "";
    if (audioRef.current) audioRef.current.value = "";
    if (multiImageRef.current) multiImageRef.current.value = "";
  }

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4 p-4">
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-base font-semibold">Search</h1>
          <p className="text-xs text-muted">
            Multimodal retrieval without generation — same pipeline as Gradio
            Search &amp; Query
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
          <label className="flex items-center gap-1">
            Threshold
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              aria-label="Similarity threshold"
              className="w-24"
            />
            <span className="font-mono text-text">{threshold.toFixed(2)}</span>
          </label>
          <label className="flex items-center gap-1">
            Max results
            <input
              type="number"
              min={1}
              max={50}
              value={k}
              onChange={(e) => setK(Number(e.target.value) || 10)}
              aria-label="Maximum results"
              className="w-12 rounded border border-border bg-surface px-1 py-0.5 font-mono text-text"
            />
          </label>
        </div>
      </header>

      <div
        role="tablist"
        aria-label="Search modality"
        className="flex flex-wrap gap-1 rounded-lg border border-border bg-surface p-1"
      >
        {MODES.map((m) => {
          const Icon = m.icon;
          const active = modality === m.id;
          return (
            <button
              key={m.id}
              type="button"
              role="tab"
              aria-selected={active}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm",
                active
                  ? "bg-accent-subtle text-accent"
                  : "text-muted hover:bg-surface-2 hover:text-text",
              )}
              onClick={() => {
                setModality(m.id);
                setError(null);
              }}
            >
              <Icon className="h-3.5 w-3.5" aria-hidden />
              {m.label}
            </button>
          );
        })}
      </div>

      <section className="rounded-lg border border-border bg-surface p-3">
        {(modality === "text" || modality === "multimodal") && (
          <label className="block text-xs font-medium text-muted">
            {modality === "multimodal" ? "Text component" : "Text query"}
            <Textarea
              className="mt-1"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                modality === "multimodal"
                  ? "Describe what you're looking for…"
                  : "e.g. security protocols, network configuration…"
              }
              rows={2}
              aria-label="Search query text"
            />
          </label>
        )}

        {(modality === "image" || modality === "multimodal") && (
          <div className="mt-3">
            <label className="text-xs font-medium text-muted">
              {modality === "multimodal" ? "Image component" : "Image query"}
            </label>
            <input
              ref={modality === "image" ? imageRef : multiImageRef}
              type="file"
              accept=".jpg,.jpeg,.png,.bmp,.tiff,.gif,.webp"
              aria-label="Upload image for search"
              className="mt-1 block w-full text-sm text-muted file:mr-3 file:rounded-md file:border-0 file:bg-surface-2 file:px-3 file:py-1.5 file:text-sm file:text-text"
              onChange={(e) => setImageFile(e.target.files?.[0] || null)}
            />
            {imageFile && (
              <p className="mt-1 font-mono text-[11px] text-muted">
                {imageFile.name}
              </p>
            )}
          </div>
        )}

        {modality === "voice" && (
          <div>
            <label className="text-xs font-medium text-muted">
              Voice query (audio file)
            </label>
            <input
              ref={audioRef}
              type="file"
              accept=".wav,.mp3,.m4a,.flac,.ogg"
              aria-label="Upload audio for voice search"
              className="mt-1 block w-full text-sm text-muted file:mr-3 file:rounded-md file:border-0 file:bg-surface-2 file:px-3 file:py-1.5 file:text-sm file:text-text"
              onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
            />
            {audioFile && (
              <p className="mt-1 font-mono text-[11px] text-muted">
                {audioFile.name}
              </p>
            )}
            {transcription && (
              <p className="mt-2 rounded-md border border-border bg-surface-2 px-2 py-1.5 text-sm">
                <span className="text-muted">Transcription: </span>
                {transcription}
              </p>
            )}
          </div>
        )}

        <div className="mt-3 flex flex-wrap gap-2">
          <Button onClick={() => void runSearch()} disabled={busy}>
            {busy ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            ) : (
              <Search className="h-4 w-4" aria-hidden />
            )}
            Search
          </Button>
          <Button variant="secondary" onClick={clearAll} disabled={busy}>
            Clear
          </Button>
        </div>

        <p className="mt-2 text-xs text-muted" role="status" aria-live="polite">
          {status}
        </p>
        {error && (
          <p className="mt-1 text-xs text-danger" role="alert">
            {error}
          </p>
        )}
      </section>

      <div className="grid gap-4 lg:grid-cols-[1fr_240px]">
        <section aria-label="Search results">
          <h2 className="mb-2 text-sm font-semibold">Results</h2>
          {results.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border bg-surface p-6 text-sm text-muted">
              No results yet. Run a search to retrieve matching chunks without
              generating an answer.
            </div>
          ) : (
            <ul className="space-y-2">
              {results.map((r, i) => (
                <li
                  key={r.id || `${r.file_path}-${i}`}
                  className="rounded-lg border border-border bg-surface p-3"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-sm font-medium">
                      {r.file_name || r.file_path || r.id || "Result"}
                    </span>
                    <Badge tone="neutral">
                      <span className="font-mono">
                        {r.similarity_score.toFixed(3)}
                      </span>
                    </Badge>
                  </div>
                  <p className="mt-0.5 font-mono text-[11px] text-muted">
                    {r.file_type || "unknown"}
                    {r.text_similarity != null
                      ? ` · text ${r.text_similarity.toFixed(3)}`
                      : ""}
                    {r.image_similarity != null
                      ? ` · image ${r.image_similarity.toFixed(3)}`
                      : ""}
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-text">
                    {r.content_preview || "No preview"}
                  </p>
                  {r.file_path && (
                    <p className="mt-1 break-all font-mono text-[10px] text-muted">
                      {r.file_path}
                    </p>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>

        <aside className="rounded-lg border border-border bg-surface p-3">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold">Query history</h2>
            <Button
              size="sm"
              variant="ghost"
              aria-label="Clear query history"
              onClick={() => {
                clearQueryHistory();
                setHistory([]);
              }}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
          {history.length === 0 ? (
            <p className="text-xs text-muted">No queries yet.</p>
          ) : (
            <ul className="max-h-80 space-y-2 overflow-y-auto scroll-panel">
              {history.map((h) => (
                <li
                  key={h.id}
                  className="rounded-md border border-border bg-surface-2 px-2 py-1.5 text-xs"
                >
                  <div className="font-medium text-text">
                    <span className="text-muted">[{h.type}] </span>
                    {h.query.length > 60 ? `${h.query.slice(0, 60)}…` : h.query}
                  </div>
                  <div className="mt-0.5 font-mono text-[10px] text-muted">
                    {h.results_count} hits · {h.processing_time.toFixed(2)}s
                  </div>
                </li>
              ))}
            </ul>
          )}
        </aside>
      </div>
    </div>
  );
}

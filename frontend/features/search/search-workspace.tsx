"use client";

import { useState } from "react";
import {
  FileText,
  Image as ImageIcon,
  Mic,
  Search,
  Sparkles,
  Layers,
  Clock,
  Loader2,
  FileCode,
  FileAudio,
} from "lucide-react";
import { api, ApiClientError } from "@/lib/api";
import type { SearchResultItem } from "@/types/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { pushQueryHistory } from "@/lib/query-history";

type Modality = "text" | "image" | "voice" | "multimodal";

export function SearchWorkspace() {
  const [modality, setModality] = useState<Modality>("text");
  const [query, setQuery] = useState("");
  const [threshold, setThreshold] = useState(0.4);
  const [maxResults, setMaxResults] = useState(6);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState<number | null>(null);

  async function handleSearch() {
    setError(null);
    setTranscription(null);
    setBusy(true);

    try {
      if (modality === "text" && !query.trim()) {
        setError("Enter a query text to search");
        setBusy(false);
        return;
      }
      if (modality === "image" && !imageFile) {
        setError("Select an image file for visual search");
        setBusy(false);
        return;
      }
      if (modality === "voice" && !audioFile) {
        setError("Select an audio recording for voice search");
        setBusy(false);
        return;
      }

      const res = await api.search({
        query: query.trim() || undefined,
        modality,
        similarity_threshold: threshold,
        k: maxResults,
        image: imageFile,
        audio: audioFile,
      });

      setResults(res.results || []);
      setTranscription(res.transcription || null);
      setTime(res.processing_time);

      pushQueryHistory({
        query: query.trim() || imageFile?.name || audioFile?.name || "Multimodal Search",
        type: modality,
        results_count: res.results?.length || 0,
        processing_time: res.processing_time,
      });
    } catch (err) {
      setError(
        err instanceof ApiClientError ? err.message : "Search query execution failed",
      );
    } finally {
      setBusy(false);
    }
  }

  const TABS = [
    { id: "text", label: "Semantic Text", icon: FileText },
    { id: "image", label: "CLIP Vision", icon: ImageIcon },
    { id: "voice", label: "Whisper Audio", icon: Mic },
    { id: "multimodal", label: "Fused Multi", icon: Layers },
  ] as const;

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-4 sm:p-8">
      {/* Search Hub Header */}
      <header className="space-y-1">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase text-accent">
          <Sparkles className="h-3.5 w-3.5" />
          <span>Multimodal Discovery</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-text">
          Search the entire offline index
        </h1>
        <p className="text-xs text-muted">
          Direct vector similarity matching across text embeddings, visual CLIP projections, and audio transcripts.
        </p>
      </header>

      {/* Perplexity-Style Modality Segmented Bar */}
      <div className="flex rounded-2xl border border-border bg-surface p-1.5 max-w-lg shadow-2xs">
        {TABS.map((t) => {
          const Icon = t.icon;
          const active = modality === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => {
                setModality(t.id);
                setError(null);
              }}
              className={cn(
                "flex flex-1 items-center justify-center gap-1.5 rounded-xl py-2 text-xs font-medium transition-all cursor-pointer",
                active
                  ? "bg-accent/15 text-accent font-semibold shadow-xs"
                  : "text-muted hover:text-text",
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{t.label}</span>
            </button>
          );
        })}
      </div>

      {/* Input Search Console */}
      <div className="rounded-2xl border border-border bg-surface shadow-hud p-5 space-y-4">
        {(modality === "text" || modality === "multimodal") && (
          <div>
            <label className="block text-xs font-semibold text-text mb-1.5">
              Search Query
            </label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search concepts, technical specifications, or keywords..."
              className="w-full rounded-xl border border-border bg-surface-2 px-3.5 py-2.5 text-sm text-text placeholder:text-dim focus:outline-none focus:border-accent/40"
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleSearch();
              }}
            />
          </div>
        )}

        {(modality === "image" || modality === "multimodal") && (
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-text">
              Image Target (CLIP Embedding)
            </label>
            <div className="rounded-xl border border-dashed border-border bg-surface-2/60 p-4 text-center">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setImageFile(e.target.files?.[0] || null)}
                className="text-xs text-muted file:mr-3 file:py-1 file:px-3 file:rounded-lg file:border-0 file:text-xs file:bg-accent/15 file:text-accent file:cursor-pointer"
              />
              {imageFile && (
                <p className="mt-2 text-xs text-accent font-medium">Selected: {imageFile.name}</p>
              )}
            </div>
          </div>
        )}

        {(modality === "voice" || modality === "multimodal") && (
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-text">
              Audio Target (Whisper Speech-To-Text)
            </label>
            <div className="rounded-xl border border-dashed border-border bg-surface-2/60 p-4 text-center">
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
                className="text-xs text-muted file:mr-3 file:py-1 file:px-3 file:rounded-lg file:border-0 file:text-xs file:bg-accent/15 file:text-accent file:cursor-pointer"
              />
              {audioFile && (
                <p className="mt-2 text-xs text-accent font-medium">Selected: {audioFile.name}</p>
              )}
            </div>
          </div>
        )}

        {/* Action strip & threshold controls */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-border/50 text-xs">
          <div className="flex flex-wrap items-center gap-4 text-muted">
            <div className="flex items-center gap-2">
              <span>Threshold:</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-16 accent-accent h-1.5 cursor-pointer"
              />
              <span className="text-text font-semibold">{threshold.toFixed(2)}</span>
            </div>

            <div className="flex items-center gap-2">
              <span>Max Matches:</span>
              <input
                type="number"
                min={1}
                max={20}
                value={maxResults}
                onChange={(e) => setMaxResults(Number(e.target.value) || 6)}
                className="w-12 rounded border border-border bg-surface-2 px-1.5 py-0.5 text-center text-text font-medium"
              />
            </div>
          </div>

          <Button
            variant="primary"
            size="sm"
            onClick={() => void handleSearch()}
            disabled={busy}
            className="rounded-xl px-5"
          >
            {busy ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
                <span>Searching…</span>
              </>
            ) : (
              <>
                <Search className="h-3.5 w-3.5 mr-1.5" />
                <span>Run Search</span>
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

      {transcription && (
        <div className="rounded-xl border border-border bg-surface p-4 space-y-1.5 text-xs">
          <span className="text-accent text-xs uppercase font-bold">
            Whisper Audio Transcription:
          </span>
          <p className="text-text font-medium">{transcription}</p>
        </div>
      )}

      {/* Results Grid */}
      {results.length > 0 && (
        <div className="space-y-3 pt-2">
          <div className="flex items-center justify-between text-xs text-muted font-medium">
            <span>Found {results.length} matched vectors</span>
            {time != null && <span>Search completed in {time.toFixed(2)}s</span>}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {results.map((r, idx) => (
              <div
                key={r.id || idx}
                className="rounded-2xl border border-border bg-surface hover:bg-surface-2 p-4 space-y-2.5 text-xs transition-colors shadow-2xs"
              >
                <div className="flex items-center justify-between gap-2 border-b border-border/40 pb-2">
                  <span className="truncate font-semibold text-text text-xs">
                    {r.file_name || r.file_path}
                  </span>
                  <span className="text-xs text-accent font-bold px-2 py-0.5 rounded bg-accent/15">
                    {r.similarity_score.toFixed(2)} match
                  </span>
                </div>

                <p className="text-xs text-text/90 leading-relaxed bg-surface-2/60 p-3 rounded-xl">
                  {r.content_preview}
                </p>

                <div className="flex items-center justify-between text-[11px] text-dim pt-1 font-medium">
                  <span>Type: {r.file_type || "document"}</span>
                  <span className="truncate max-w-[150px] font-mono">{r.file_path}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Brain,
  Check,
  ChevronRight,
  Cloud,
  Compass,
  Database,
  ExternalLink,
  FileAudio,
  FileCode,
  FileImage,
  FileSearch,
  FileText,
  GitBranch,
  Globe,
  Layers,
  Library,
  Loader2,
  Paperclip,
  RotateCcw,
  Search,
  SlidersHorizontal,
  Sparkles,
  Square,
  Volume2,
  X,
  Zap,
} from "lucide-react";
import { api, ApiClientError } from "@/lib/api";
import type { ChatMessage, CitationItem, SearchResultItem } from "@/types/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { FeedbackForm } from "@/features/chat/feedback-form";
import { pushQueryHistory } from "@/lib/query-history";
import { StatusBar } from "@/components/layout/status-bar";

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function getSourceIcon(type?: string) {
  const t = (type || "").toLowerCase();
  if (t.includes("pdf") || t.includes("doc") || t.includes("txt")) {
    return <FileText className="h-4 w-4 text-sky-400" />;
  }
  if (t.includes("jpg") || t.includes("png") || t.includes("image")) {
    return <FileImage className="h-4 w-4 text-teal-400" />;
  }
  if (t.includes("audio") || t.includes("wav") || t.includes("mp3")) {
    return <FileAudio className="h-4 w-4 text-amber-400" />;
  }
  return <FileCode className="h-4 w-4 text-zinc-400" />;
}

export function ChatWorkspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"local" | "cloud">("local");
  const [threshold, setThreshold] = useState(0.5);
  const [maxDocs, setMaxDocs] = useState(5);
  const [useKnowledgeGraph, setUseKnowledgeGraph] = useState(false);
  const [graphWarning, setGraphWarning] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedCitationPreview, setSelectedCitationPreview] = useState<CitationItem | null>(null);
  const [showConfig, setShowConfig] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("neurax-chat-mode");
      if (stored === "cloud" || stored === "local") {
        setMode(stored);
      }
    } catch {
      /* ignore */
    }
  }, []);

  function handleModeChange(newMode: "local" | "cloud") {
    setMode(newMode);
    try {
      localStorage.setItem("neurax-chat-mode", newMode);
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, phase]);

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
    setPhase(null);
  }

  async function submitQuery(promptText?: string) {
    const text = (promptText || query).trim();
    if (!text || busy) return;

    setError(null);
    setQuery("");
    const userMsg: ChatMessage = {
      id: uid(),
      role: "user",
      content: text,
    };
    const assistantId = uid();
    setMessages((m) => [
      ...m,
      userMsg,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        query: text,
        status: "streaming",
      },
    ]);
    setBusy(true);
    setPhase("Searching corpus…");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let responseText = "";
      let citations: CitationItem[] = [];
      let sources: SearchResultItem[] = [];
      let confidence = 0;
      let processing_time = 0;
      let model_used = "";

      setGraphWarning(null);
      for await (const evt of api.chatStream(
        {
          query: text,
          similarity_threshold: threshold,
          max_docs: maxDocs,
          use_knowledge_graph: useKnowledgeGraph,
          mode,
        },
        controller.signal,
      )) {
        if (evt.event === "status") {
          const p = String(evt.data.phase || "");
          if (p === "graph_context") {
            if (evt.data.graph_warning) {
              setGraphWarning(String(evt.data.graph_warning));
            }
            setPhase(
              evt.data.graph_context_used
                ? "Synthesizing graph ontology…"
                : "Retrieval active…",
            );
          } else {
            setPhase(
              p === "retrieval_started"
                ? "Retrieving source chunks…"
                : p === "generation_started"
                  ? "Generating answer…"
                  : p,
            );
          }
        } else if (evt.event === "retrieval") {
          sources = (evt.data.sources as SearchResultItem[]) || [];
        } else if (evt.event === "message") {
          responseText = String(evt.data.response || "");
          confidence = Number(evt.data.confidence || 0);
          model_used = String(evt.data.model_used || "");
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: responseText,
                    status: "streaming",
                    confidence,
                    model_used,
                  }
                : m,
            ),
          );
        } else if (evt.event === "citations") {
          citations = (evt.data.citations as CitationItem[]) || [];
        } else if (evt.event === "done") {
          processing_time = Number(evt.data.processing_time || 0);
          if (evt.data.graph_warning) {
            setGraphWarning(String(evt.data.graph_warning));
          }
        } else if (evt.event === "error") {
          throw new ApiClientError(500, {
            error: {
              code: String(evt.data.code || "processing_error"),
              message: String(evt.data.message || "Generation failed"),
            },
          });
        }
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content:
                  responseText ||
                  "No grounded answer returned. Verify inference server.",
                query: text,
                citations,
                sources,
                confidence,
                processing_time,
                model_used,
                status: "done",
              }
            : m,
        ),
      );

      pushQueryHistory({
        query: text,
        type: "chat",
        results_count: sources.length,
        processing_time,
      });

      try {
        sessionStorage.setItem(
          "neurax-last-sources",
          JSON.stringify({ sources, citations, query: text }),
        );
      } catch {
        /* ignore */
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: m.content || "Cancelled by user.", status: "error" }
              : m,
          ),
        );
      } else {
        const msg =
          err instanceof ApiClientError ? err.message : "Search request failed";
        setError(msg);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: msg, status: "error" }
              : m,
          ),
        );
      }
    } finally {
      setBusy(false);
      setPhase(null);
      abortRef.current = null;
    }
  }

  const promptSuggestions = [
    { label: "Corpus Summary", text: "Summarize the key architectural points from all indexed documents." },
    { label: "Security Anomalies", text: "Are there any anomalies or tamper alerts in the knowledge graph?" },
    { label: "Air-Gapped Setup", text: "What are the requirements for offline deployment with LM Studio?" },
  ];

  return (
    <div className="flex flex-col min-h-full w-full bg-bg text-text">
      {/* Top Header Bar */}
      <div className="flex h-12 shrink-0 items-center justify-between px-6 border-b border-border/40 bg-surface/50 backdrop-blur-sm z-10">
        <div className="flex items-center gap-2 text-xs text-muted">
          <span className="font-bold text-sm text-text">NeuraX</span>
          <span>•</span>
          <span>Local Multimodal Search &amp; Intelligence</span>
        </div>
        <StatusBar />
      </div>

      {/* Main Conversation / Hero Canvas */}
      <div className="flex-1 overflow-y-auto scroll-panel px-4 py-8 sm:px-8">
        <div className="mx-auto max-w-3xl space-y-8">
          {/* Hero State (When no messages) */}
          {messages.length === 0 && (
            <div className="py-12 sm:py-20 text-center space-y-6">
              <div className={cn(
                "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold border transition-all",
                mode === "local"
                  ? "bg-teal-500/10 text-teal-400 border-teal-500/30"
                  : "bg-sky-500/10 text-sky-400 border-sky-500/30",
              )}>
                <Sparkles className="h-3.5 w-3.5" />
                <span>{mode === "local" ? "Air-Gapped Local LM Studio (Offline)" : "Cloud LLM Mode (API Fallback)"}</span>
              </div>

              <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-text">
                Where knowledge begins.
              </h1>

              {/* Main Search Capsule */}
              <div className="max-w-2xl mx-auto text-left">
                <div className="rounded-2xl border border-border bg-surface shadow-hud p-4 focus-within:border-teal-500/50 focus-within:ring-2 focus-within:ring-teal-500/15 transition-all">
                  <textarea
                    ref={inputRef}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={`Ask anything across documents (${mode === "local" ? "Local LM Studio" : "Cloud LLM"} mode)...`}
                    rows={2}
                    className="w-full resize-none bg-transparent text-sm sm:text-base text-text placeholder:text-dim focus:outline-none scroll-panel leading-relaxed"
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        void submitQuery();
                      }
                    }}
                  />

                  {/* Capsule Action Strip */}
                  <div className="flex flex-wrap items-center justify-between gap-2 pt-3 border-t border-border/50">
                    <div className="flex flex-wrap items-center gap-2">
                      {/* Focus Mode Pill */}
                      <button
                        type="button"
                        onClick={() => setUseKnowledgeGraph(!useKnowledgeGraph)}
                        className={cn(
                          "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors cursor-pointer select-none",
                          useKnowledgeGraph
                            ? "bg-teal-500/15 text-teal-400 border-teal-500/30 font-semibold"
                            : "bg-surface-2 text-muted border-border hover:text-text",
                        )}
                      >
                        <GitBranch className="h-3.5 w-3.5" />
                        <span>Graph Context</span>
                      </button>

                      {/* Engine Quick Switcher inside Capsule */}
                      <button
                        type="button"
                        onClick={() => handleModeChange(mode === "local" ? "cloud" : "local")}
                        className={cn(
                          "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold border transition-all cursor-pointer select-none",
                          mode === "local"
                            ? "bg-teal-500/15 text-teal-400 border-teal-500/40 hover:bg-teal-500/25"
                            : "bg-sky-500/15 text-sky-400 border-sky-500/40 hover:bg-sky-500/25",
                        )}
                        title="Click to toggle Local vs Cloud Engine"
                      >
                        {mode === "local" ? (
                          <>
                            <Zap className="h-3.5 w-3.5 text-teal-400" />
                            <span>Local</span>
                          </>
                        ) : (
                          <>
                            <Cloud className="h-3.5 w-3.5 text-sky-400" />
                            <span>Cloud</span>
                          </>
                        )}
                      </button>

                      {/* Config Button */}
                      <button
                        type="button"
                        onClick={() => setShowConfig(!showConfig)}
                        className={cn(
                          "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors cursor-pointer select-none",
                          showConfig
                            ? "bg-surface-3 text-text border-border"
                            : "bg-surface-2 text-muted border-border hover:text-text",
                        )}
                      >
                        <SlidersHorizontal className="h-3.5 w-3.5" />
                        <span>Params</span>
                      </button>
                    </div>

                    {/* Submit Arrow */}
                    <Button
                      variant="primary"
                      size="icon-sm"
                      onClick={() => void submitQuery()}
                      disabled={!query.trim() || busy}
                      className="h-8 w-8 rounded-full bg-teal-500 text-slate-950 hover:bg-teal-400 font-bold disabled:opacity-40"
                    >
                      <ArrowUp className="h-4 w-4" />
                    </Button>
                  </div>

                  {/* Config Tray */}
                  {showConfig && (
                    <div className="mt-3 pt-3 border-t border-border/50 flex flex-wrap items-center gap-4 text-xs text-muted">
                      <div className="flex items-center gap-2">
                        <span>Min Confidence:</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.05}
                          value={threshold}
                          onChange={(e) => setThreshold(Number(e.target.value))}
                          className="w-16 accent-teal-500 h-1.5 cursor-pointer"
                        />
                        <span className="text-text font-semibold">{threshold.toFixed(2)}</span>
                      </div>

                      <div className="flex items-center gap-2">
                        <span>Max Sources:</span>
                        <input
                          type="number"
                          min={1}
                          max={10}
                          value={maxDocs}
                          onChange={(e) => setMaxDocs(Number(e.target.value) || 5)}
                          className="w-12 rounded border border-border bg-surface-2 px-1.5 py-0.5 text-center text-text font-medium"
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Suggestions List */}
                <div className="mt-4 flex flex-wrap gap-2">
                  {promptSuggestions.map((item) => (
                    <button
                      key={item.label}
                      type="button"
                      onClick={() => void submitQuery(item.text)}
                      className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-border bg-surface hover:bg-surface-2 text-xs font-medium text-muted hover:text-text transition-colors cursor-pointer shadow-2xs select-none"
                    >
                      <Search className="h-3 w-3 text-teal-400" />
                      <span>{item.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Thread Conversation Stream */}
          {messages.map((m, idx) => {
            if (m.role === "user") {
              return (
                <div key={m.id} className="space-y-2 pt-4">
                  <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-text">
                    {m.content}
                  </h2>
                  <div className="w-full h-px bg-border/60" />
                </div>
              );
            }

            // Assistant Answer Block
            return (
              <article key={m.id} className="space-y-5 pb-6">
                {/* 1. Perplexity Sources Carousel */}
                {((m.citations && m.citations.length > 0) || (m.sources && m.sources.length > 0)) && (
                  <div className="space-y-2.5">
                    <div className="flex items-center gap-2 text-xs font-semibold text-muted uppercase tracking-wider">
                      <Library className="h-3.5 w-3.5 text-teal-400" />
                      <span>Sources ({m.citations?.length || m.sources?.length})</span>
                    </div>

                    {/* Horizontal Scrollable Carousel */}
                    <div className="flex gap-3 overflow-x-auto pb-2 scroll-panel no-scrollbar">
                      {m.citations && m.citations.length > 0
                        ? m.citations.map((c) => (
                            <button
                              key={`cit-${c.citation_id}`}
                              type="button"
                              onClick={() => setSelectedCitationPreview(c)}
                              className="flex flex-col justify-between w-52 shrink-0 rounded-2xl border border-border bg-surface hover:bg-surface-2 p-3.5 text-left transition-all hover:scale-[1.02] cursor-pointer shadow-2xs group"
                            >
                              <div className="flex items-center justify-between gap-2">
                                <div className="flex items-center gap-2 truncate">
                                  {getSourceIcon(c.source_type)}
                                  <span className="font-semibold text-xs text-text truncate">
                                    {c.file_path.split(/[/\\]/).pop() || c.source_document}
                                  </span>
                                </div>
                                <span className="text-xs text-teal-400 font-bold shrink-0">
                                  [{c.citation_id}]
                                </span>
                              </div>

                              <p className="mt-2.5 line-clamp-2 text-xs text-muted group-hover:text-text/90 leading-relaxed">
                                {c.content_snippet}
                              </p>

                              <div className="mt-3 flex items-center justify-between text-[11px] text-dim border-t border-border/40 pt-2 font-medium">
                                <span>{(c.confidence_score * 100).toFixed(0)}% match</span>
                                {c.page_number != null && <span>p.{c.page_number}</span>}
                              </div>
                            </button>
                          ))
                        : m.sources?.map((s, sIdx) => (
                            <div
                              key={s.id || sIdx}
                              className="flex flex-col justify-between w-52 shrink-0 rounded-2xl border border-border bg-surface p-3.5 text-left text-xs shadow-2xs"
                            >
                              <div className="flex items-center gap-2 truncate font-semibold text-text">
                                {getSourceIcon(s.file_type)}
                                <span className="truncate">{s.file_name || s.file_path}</span>
                              </div>
                              <p className="mt-2.5 line-clamp-2 text-xs text-muted leading-relaxed">
                                {s.content_preview}
                              </p>
                              <span className="mt-3 text-xs text-teal-400 font-semibold">
                                Score: {s.similarity_score.toFixed(2)}
                              </span>
                            </div>
                          ))}
                    </div>
                  </div>
                )}

                {/* 2. Answer Header Status */}
                <div className="flex items-center justify-between text-xs text-dim">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-sm text-text">Answer</span>
                    {m.status === "streaming" && (
                      <span className="flex items-center gap-1.5 text-teal-400 animate-pulse font-medium">
                        <span className="h-1.5 w-1.5 rounded-full bg-teal-400" />
                        Synthesizing…
                      </span>
                    )}
                  </div>
                  {m.processing_time != null && m.status === "done" && (
                    <span className="font-medium">Generated in {m.processing_time.toFixed(2)}s</span>
                  )}
                </div>

                {/* 3. Answer Markdown Prose */}
                <div className="text-base leading-relaxed text-text font-normal space-y-4">
                  <div className="whitespace-pre-wrap leading-relaxed">
                    {m.content || (m.status === "streaming" ? (
                      <span className="text-dim italic flex items-center gap-2 text-xs">
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-teal-400" />
                        Generating grounded intelligence answer…
                      </span>
                    ) : "")}
                  </div>
                </div>

                {/* 4. Follow-up suggestions & Feedback */}
                {m.status === "done" && (
                  <div className="pt-4 border-t border-border/50 space-y-3">
                    <div className="flex items-center gap-2 text-xs font-semibold text-muted">
                      <Sparkles className="h-3.5 w-3.5 text-teal-400" />
                      <span>Suggested Follow-Ups</span>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => void submitQuery(`Explain more details regarding ${m.query}`)}
                        className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-border bg-surface hover:bg-surface-2 text-xs font-medium text-muted hover:text-text transition-colors cursor-pointer"
                      >
                        <span>Deep dive on {m.query?.slice(0, 30)}…</span>
                        <ChevronRight className="h-3 w-3 text-dim" />
                      </button>
                      <button
                        type="button"
                        onClick={() => void submitQuery(`Extract entities and relationships from this answer`)}
                        className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-border bg-surface hover:bg-surface-2 text-xs font-medium text-muted hover:text-text transition-colors cursor-pointer"
                      >
                        <span>Extract Entity Graph</span>
                        <ChevronRight className="h-3 w-3 text-dim" />
                      </button>
                    </div>

                    {m.query && m.content && (
                      <FeedbackForm query={m.query} response={m.content} />
                    )}
                  </div>
                )}
              </article>
            );
          })}

          <div ref={endRef} />
        </div>
      </div>

      {/* Floating Bottom Input Capsule (When in active thread) */}
      {messages.length > 0 && (
        <div className="shrink-0 p-4 sm:p-6 bg-gradient-to-t from-bg via-bg to-transparent border-t border-border/40">
          <div className="mx-auto max-w-3xl space-y-2">
            {error && (
              <p className="text-xs text-signal-rose" role="alert">
                Error: {error}
              </p>
            )}

            <div className="rounded-2xl border border-border bg-surface shadow-hud p-3 focus-within:border-teal-500/40 focus-within:ring-2 focus-within:ring-teal-500/10 transition-all space-y-2">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={`Ask follow-up (${mode === "local" ? "Local LM Studio" : "Cloud LLM"} mode)...`}
                rows={1}
                className="w-full resize-none bg-transparent px-2 py-1 text-sm text-text placeholder:text-dim focus:outline-none scroll-panel"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void submitQuery();
                  }
                }}
              />

              <div className="flex items-center justify-between pt-2 border-t border-border/40">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handleModeChange(mode === "local" ? "cloud" : "local")}
                    className={cn(
                      "flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold transition-all cursor-pointer select-none",
                      mode === "local"
                        ? "bg-teal-500/15 text-teal-400 border border-teal-500/40"
                        : "bg-sky-500/15 text-sky-400 border border-sky-500/40",
                    )}
                  >
                    {mode === "local" ? (
                      <>
                        <Zap className="h-3.5 w-3.5 text-teal-400" />
                        <span>Local (LM Studio)</span>
                      </>
                    ) : (
                      <>
                        <Cloud className="h-3.5 w-3.5 text-sky-400" />
                        <span>Cloud LLM</span>
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={() => setUseKnowledgeGraph(!useKnowledgeGraph)}
                    className={cn(
                      "flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold border transition-colors cursor-pointer select-none",
                      useKnowledgeGraph
                        ? "bg-teal-500/15 text-teal-400 border-teal-500/30"
                        : "bg-surface-2 text-muted border-border hover:text-text",
                    )}
                  >
                    <GitBranch className="h-3 w-3" />
                    <span>Graph</span>
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  {busy ? (
                    <Button
                      variant="danger"
                      size="sm"
                      onClick={stop}
                      className="h-8 px-3 text-xs rounded-xl font-medium"
                    >
                      <Square className="h-3 w-3 mr-1" />
                      <span>Stop</span>
                    </Button>
                  ) : (
                    <Button
                      variant="primary"
                      size="icon-sm"
                      onClick={() => void submitQuery()}
                      disabled={!query.trim()}
                      className="h-8 w-8 rounded-full bg-teal-500 text-slate-950 hover:bg-teal-400 font-bold disabled:opacity-40 shrink-0"
                    >
                      <ArrowUp className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Citation Preview Modal */}
      {selectedCitationPreview && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-fadeIn"
          onClick={() => setSelectedCitationPreview(null)}
        >
          <div
            className="rounded-2xl border border-border bg-surface max-w-lg w-full p-6 shadow-hud space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-teal-400 px-2 py-0.5 rounded bg-teal-500/15 border border-teal-500/20">
                  [{selectedCitationPreview.citation_id}]
                </span>
                <h3 className="font-bold text-sm text-text truncate">
                  {selectedCitationPreview.file_path.split(/[/\\]/).pop() || selectedCitationPreview.source_document}
                </h3>
              </div>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => setSelectedCitationPreview(null)}
                className="h-7 w-7 text-muted hover:text-text rounded-lg"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="grid grid-cols-2 gap-2.5 text-xs text-dim bg-surface-2 p-3 rounded-xl border border-border/50 font-medium">
              <div>
                <span>Type: </span>
                <span className="text-text font-semibold">{selectedCitationPreview.source_type || "text"}</span>
              </div>
              <div>
                <span>Page: </span>
                <span className="text-text font-semibold">{selectedCitationPreview.page_number ?? "N/A"}</span>
              </div>
              <div className="col-span-2">
                <span>Confidence: </span>
                <span className="text-teal-400 font-bold">
                  {(selectedCitationPreview.confidence_score * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            <div className="space-y-1.5">
              <span className="text-xs font-semibold text-muted uppercase">Grounding Snippet:</span>
              <p className="rounded-xl border border-border bg-surface-2 p-4 text-xs text-text leading-relaxed">
                {selectedCitationPreview.content_snippet}
              </p>
            </div>

            <p className="text-[11px] text-dim break-all font-mono">
              Path: {selectedCitationPreview.file_path}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Send, Square } from "lucide-react";
import { api, ApiClientError } from "@/lib/api";
import type { ChatMessage, CitationItem, SearchResultItem } from "@/types/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function ChatWorkspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [maxDocs, setMaxDocs] = useState(5);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedSources, setSelectedSources] = useState<SearchResultItem[]>(
    [],
  );
  const [selectedCitations, setSelectedCitations] = useState<CitationItem[]>(
    [],
  );
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, phase]);

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
    setPhase(null);
  }

  async function submit() {
    const text = query.trim();
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
        status: "streaming",
      },
    ]);
    setBusy(true);
    setPhase("Starting…");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let responseText = "";
      let citations: CitationItem[] = [];
      let sources: SearchResultItem[] = [];
      let confidence = 0;
      let processing_time = 0;
      let model_used = "";

      for await (const evt of api.chatStream(
        {
          query: text,
          similarity_threshold: threshold,
          max_docs: maxDocs,
        },
        controller.signal,
      )) {
        if (evt.event === "status") {
          const p = String(evt.data.phase || "");
          setPhase(
            p === "retrieval_started"
              ? "Retrieving sources…"
              : p === "generation_started"
                ? "Generating answer…"
                : p,
          );
        } else if (evt.event === "retrieval") {
          sources = (evt.data.sources as SearchResultItem[]) || [];
          setSelectedSources(sources);
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
          setSelectedCitations(citations);
        } else if (evt.event === "done") {
          processing_time = Number(evt.data.processing_time || 0);
          if (evt.data.lm_studio_available === false) {
            setPhase("LM Studio unavailable — response may be degraded");
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
                  "No response returned. Check LM Studio and indexed documents.",
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
      setSelectedSources(sources);
      setSelectedCitations(citations);
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
              ? {
                  ...m,
                  content: m.content || "Cancelled.",
                  status: "error",
                }
              : m,
          ),
        );
      } else {
        const msg =
          err instanceof ApiClientError
            ? err.message
            : "Chat request failed";
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

  return (
    <div className="flex h-[calc(100vh-6.5rem)] min-h-[28rem] flex-col lg:flex-row">
      <section className="flex min-w-0 flex-1 flex-col border-b border-border lg:border-b-0 lg:border-r">
        <div className="flex items-center justify-between border-b border-border px-4 py-2">
          <div>
            <h1 className="text-base font-semibold">Chat</h1>
            <p className="text-xs text-muted">
              Grounded answers with local retrieval and LM Studio
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
              Max docs
              <input
                type="number"
                min={1}
                max={10}
                value={maxDocs}
                onChange={(e) => setMaxDocs(Number(e.target.value) || 5)}
                aria-label="Maximum context documents"
                className="w-12 rounded border border-border bg-surface px-1 py-0.5 font-mono text-text"
              />
            </label>
          </div>
        </div>

        <div
          className="scroll-panel flex-1 space-y-4 overflow-y-auto px-4 py-4"
          aria-live="polite"
        >
          {messages.length === 0 && (
            <div className="mx-auto max-w-chat rounded-lg border border-dashed border-border bg-surface p-6 text-sm text-muted">
              <p className="font-medium text-text">Ask about your index</p>
              <p className="mt-1">
                Index documents first if the collection is empty. Answers cite
                retrieved sources when context is available.
              </p>
            </div>
          )}
          {messages.map((m) => (
            <article
              key={m.id}
              className={cn(
                "mx-auto max-w-chat rounded-lg border px-3 py-2",
                m.role === "user"
                  ? "border-accent/30 bg-accent-subtle"
                  : "border-border bg-surface",
              )}
            >
              <header className="mb-1 flex items-center gap-2 text-xs text-muted">
                <span className="font-medium text-text">
                  {m.role === "user" ? "You" : "NeuraX"}
                </span>
                {m.status === "streaming" && (
                  <Badge tone="info">
                    <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
                    Working
                  </Badge>
                )}
                {m.status === "error" && <Badge tone="danger">Error</Badge>}
                {m.confidence != null && m.status === "done" && (
                  <span className="font-mono">
                    conf {m.confidence.toFixed(2)}
                  </span>
                )}
                {m.processing_time != null && m.status === "done" && (
                  <span className="font-mono">
                    {m.processing_time.toFixed(2)}s
                  </span>
                )}
              </header>
              <div className="whitespace-pre-wrap text-[15px] leading-relaxed text-text">
                {m.content || (m.status === "streaming" ? "…" : "")}
              </div>
              {m.citations && m.citations.length > 0 && (
                <ul className="mt-3 space-y-1 border-t border-border pt-2">
                  {m.citations.map((c) => (
                    <li key={c.citation_id}>
                      <button
                        type="button"
                        className="text-left text-xs text-accent hover:underline"
                        onClick={() => {
                          setSelectedCitations(m.citations || []);
                          setSelectedSources(m.sources || []);
                        }}
                      >
                        [{c.citation_id}]{" "}
                        {c.file_path.split(/[/\\]/).pop() || c.source_document}{" "}
                        <span className="font-mono text-muted">
                          ({c.confidence_score.toFixed(2)})
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </article>
          ))}
          <div ref={endRef} />
        </div>

        <div className="border-t border-border bg-surface p-3">
          {phase && (
            <p className="mb-2 flex items-center gap-2 text-xs text-info" role="status">
              <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
              {phase}
            </p>
          )}
          {error && (
            <p className="mb-2 text-xs text-danger" role="alert">
              {error}
            </p>
          )}
          <div className="mx-auto flex max-w-chat gap-2">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your documents…"
              aria-label="Chat query"
              rows={2}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  void submit();
                }
              }}
            />
            {busy ? (
              <Button
                variant="secondary"
                onClick={stop}
                aria-label="Stop generation"
              >
                <Square className="h-4 w-4" />
                Stop
              </Button>
            ) : (
              <Button
                onClick={() => void submit()}
                disabled={!query.trim()}
                aria-label="Send query"
              >
                <Send className="h-4 w-4" />
                Send
              </Button>
            )}
          </div>
          <p className="mx-auto mt-1 max-w-chat text-[11px] text-muted">
            Ctrl/Cmd+Enter to send. Retrieval uses the same pipeline as Gradio.
          </p>
        </div>
      </section>

      <aside
        className="w-full shrink-0 overflow-y-auto scroll-panel bg-surface lg:w-80"
        aria-label="Sources panel"
      >
        <div className="border-b border-border px-3 py-2">
          <h2 className="text-sm font-semibold">Sources</h2>
          <p className="text-xs text-muted">
            Provenance from the latest retrieval
          </p>
        </div>
        <div className="space-y-3 p-3">
          {selectedCitations.length === 0 && selectedSources.length === 0 && (
            <p className="text-sm text-muted">
              Citations and retrieved chunks appear here after you ask a
              question.
            </p>
          )}
          {selectedCitations.map((c) => (
            <div
              key={`c-${c.citation_id}`}
              className="rounded-md border border-border p-2"
            >
              <div className="text-xs font-medium">
                [{c.citation_id}]{" "}
                {c.file_path.split(/[/\\]/).pop() || "Source"}
              </div>
              <div className="mt-1 font-mono text-[11px] text-muted">
                score {c.confidence_score.toFixed(2)}
                {c.page_number != null ? ` · p.${c.page_number}` : ""}
              </div>
              <p className="mt-1 text-xs leading-relaxed text-text">
                {c.content_snippet || "No snippet"}
              </p>
              <p className="mt-1 break-all font-mono text-[10px] text-muted">
                {c.file_path}
              </p>
            </div>
          ))}
          {selectedCitations.length === 0 &&
            selectedSources.map((s) => (
              <div key={s.id || s.file_path} className="rounded-md border border-border p-2">
                <div className="text-xs font-medium">
                  {s.file_name || s.file_path || s.id}
                </div>
                <div className="font-mono text-[11px] text-muted">
                  {s.similarity_score.toFixed(3)} · {s.file_type || "unknown"}
                </div>
                <p className="mt-1 text-xs">{s.content_preview}</p>
              </div>
            ))}
        </div>
      </aside>
    </div>
  );
}

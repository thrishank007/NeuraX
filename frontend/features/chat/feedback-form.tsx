"use client";

import { useState } from "react";
import { api, ApiClientError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Check, MessageSquare, Star } from "lucide-react";
import { cn } from "@/lib/utils";

export function FeedbackForm({
  query,
  response,
}: {
  query: string;
  response: string;
}) {
  const [rating, setRating] = useState(5);
  const [comments, setComments] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!query || !response) {
      setStatus("No response to rate");
      return;
    }
    setBusy(true);
    setStatus(null);
    try {
      const res = await api.feedback({
        query,
        response,
        rating,
        comments,
      });
      setStatus(`Saved: ID #${res.feedback_id.slice(0, 8)}`);
      setComments("");
    } catch (err) {
      setStatus(
        err instanceof ApiClientError ? err.message : "Failed to record evaluation",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-3 rounded-md border border-border-subtle bg-surface-2 p-3 text-xs">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 font-mono text-[11px] font-semibold text-dim uppercase">
          <MessageSquare className="h-3.5 w-3.5 text-accent" />
          <span>Evaluation Feedback</span>
        </div>
        {status && (
          <span className="font-mono text-[10px] text-accent flex items-center gap-1" role="status">
            <Check className="h-3 w-3" /> {status}
          </span>
        )}
      </div>

      <div className="mt-2.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1.5">
          {[1, 2, 3, 4, 5].map((num) => (
            <button
              key={num}
              type="button"
              onClick={() => setRating(num)}
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded border font-mono text-[11px] font-bold transition-colors cursor-pointer",
                rating === num
                  ? "bg-accent text-bg border-accent shadow-2xs"
                  : "bg-surface-3 text-dim border-border hover:text-text",
              )}
            >
              {num}
            </button>
          ))}
          <span className="font-mono text-[11px] text-dim ml-1">/ 5 Quality Score</span>
        </div>

        <Button size="sm" variant="secondary" disabled={busy} onClick={() => void submit()}>
          Record Evaluation
        </Button>
      </div>

      <Textarea
        className="mt-2 min-h-[50px] text-xs resize-none"
        value={comments}
        onChange={(e) => setComments(e.target.value)}
        placeholder="Add context on accuracy, hallucinations, or retrieval gaps (optional)..."
        aria-label="Feedback comments"
        rows={1}
      />
    </div>
  );
}

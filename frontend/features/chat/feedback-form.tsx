"use client";

import { useState } from "react";
import { api, ApiClientError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export function FeedbackForm({
  query,
  response,
}: {
  query: string;
  response: string;
}) {
  const [rating, setRating] = useState(3);
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
      setStatus(`Thanks — feedback saved (${res.feedback_id.slice(0, 8)}…)`);
      setComments("");
    } catch (err) {
      setStatus(
        err instanceof ApiClientError ? err.message : "Failed to submit feedback",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-3 rounded-md border border-border bg-surface-2/60 p-2">
      <p className="text-xs font-medium text-text">Feedback</p>
      <p className="text-[11px] text-muted">
        Rate this answer (1–5). Stored locally via the feedback system.
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-xs text-muted">
          Rating
          <input
            type="range"
            min={1}
            max={5}
            step={1}
            value={rating}
            onChange={(e) => setRating(Number(e.target.value))}
            aria-label="Feedback rating"
            className="w-28"
          />
          <span className="font-mono text-text">{rating}</span>
        </label>
        <Button size="sm" variant="secondary" disabled={busy} onClick={() => void submit()}>
          Submit feedback
        </Button>
      </div>
      <Textarea
        className="mt-2 min-h-[56px] text-xs"
        value={comments}
        onChange={(e) => setComments(e.target.value)}
        placeholder="Optional comments…"
        aria-label="Feedback comments"
        rows={2}
      />
      {status && (
        <p className="mt-1 text-[11px] text-muted" role="status">
          {status}
        </p>
      )}
    </div>
  );
}

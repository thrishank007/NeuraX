"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

export function HelpPanel({ defaultOpen = false }: { defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="rounded-lg border border-border bg-surface">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? (
          <ChevronDown className="h-4 w-4 shrink-0" aria-hidden />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0" aria-hidden />
        )}
        Help &amp; instructions
      </button>
      {open && (
        <div className="space-y-3 border-t border-border px-3 py-3 text-xs leading-relaxed text-muted">
          <div>
            <h3 className="font-medium text-text">Documents</h3>
            <ol className="mt-1 list-decimal space-y-0.5 pl-4">
              <li>Upload PDF, DOCX, DOC, TXT, images, or audio</li>
              <li>Wait for indexing progress to complete</li>
              <li>Confirm files appear in the collection inventory</li>
            </ol>
          </div>
          <div>
            <h3 className="font-medium text-text">Search</h3>
            <ul className="mt-1 list-disc space-y-0.5 pl-4">
              <li>
                <strong className="text-text">Text</strong> — natural language
                retrieval only
              </li>
              <li>
                <strong className="text-text">Image</strong> — visual similarity
                search
              </li>
              <li>
                <strong className="text-text">Voice</strong> — STT then text
                search
              </li>
              <li>
                <strong className="text-text">Multimodal</strong> — text + image
                combined
              </li>
              <li>Lower the similarity threshold if you get no hits</li>
            </ul>
          </div>
          <div>
            <h3 className="font-medium text-text">Chat</h3>
            <p className="mt-1">
              Generates grounded answers from retrieved context with citations.
              Rate responses with the feedback control after an answer.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-text">Formats &amp; limits</h3>
            <p className="mt-1">
              Documents: pdf, docx, doc, txt · Images: jpg, png, bmp, tiff, webp
              · Audio: wav, mp3, m4a, flac, ogg · Default max size 100 MB
            </p>
          </div>
          <div>
            <h3 className="font-medium text-text">Troubleshooting</h3>
            <ul className="mt-1 list-disc space-y-0.5 pl-4">
              <li>Backend unavailable — start FastAPI on port 8000</li>
              <li>LM Studio unavailable — load a model, enable Local Server</li>
              <li>Empty answers — index documents first</li>
              <li>No search hits — lower threshold or rephrase</li>
            </ul>
          </div>
        </div>
      )}
    </section>
  );
}

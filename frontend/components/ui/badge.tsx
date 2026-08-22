import * as React from "react";
import { cn } from "@/lib/utils";

type Tone = "neutral" | "success" | "warning" | "danger" | "info" | "cyan" | "emerald";

const tones: Record<Tone, string> = {
  neutral: "bg-surface-2 text-muted border-border",
  success: "bg-accent-subtle text-accent border-accent/30 dark:text-accent",
  emerald: "bg-accent-subtle text-accent border-accent/30 dark:text-accent",
  warning: "bg-signal-amber-subtle text-signal-amber border-signal-amber/30",
  danger: "bg-signal-rose-subtle text-signal-rose border-signal-rose/30",
  info: "bg-signal-cyan-subtle text-signal-cyan border-signal-cyan/30",
  cyan: "bg-signal-cyan-subtle text-signal-cyan border-signal-cyan/30",
};

export function Badge({
  children,
  tone = "neutral",
  dot = false,
  className,
}: {
  children: React.ReactNode;
  tone?: Tone;
  dot?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 font-mono text-[11px] font-medium tracking-tight",
        tones[tone],
        className,
      )}
    >
      {dot && (
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            tone === "success" || tone === "emerald"
              ? "bg-accent"
              : tone === "warning"
                ? "bg-signal-amber"
                : tone === "danger"
                  ? "bg-signal-rose"
                  : tone === "info" || tone === "cyan"
                    ? "bg-signal-cyan"
                    : "bg-muted",
          )}
          aria-hidden="true"
        />
      )}
      {children}
    </span>
  );
}

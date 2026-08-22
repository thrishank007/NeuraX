import * as React from "react";
import { cn } from "@/lib/utils";

export type TextareaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement> & {
  mono?: boolean;
};

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, mono = false, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        "flex min-h-[88px] w-full resize-y rounded-md border border-border bg-surface-2 px-3 py-2 text-xs text-text placeholder:text-dim scroll-panel transition-colors duration-150 focus-visible:border-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-accent/40 disabled:opacity-50 disabled:cursor-not-allowed",
        mono && "font-mono text-[11px]",
        className,
      )}
      {...props}
    />
  ),
);
Textarea.displayName = "Textarea";

import * as React from "react";
import { cn } from "@/lib/utils";

export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn(
      "flex min-h-[88px] w-full resize-y rounded-md border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-accent disabled:opacity-50",
      className,
    )}
    {...props}
  />
));
Textarea.displayName = "Textarea";

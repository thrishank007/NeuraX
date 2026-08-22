import * as React from "react";
import { cn } from "@/lib/utils";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement> & {
  mono?: boolean;
};

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, mono = false, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "flex h-9 w-full rounded-md border border-border bg-surface-2 px-3 text-xs text-text placeholder:text-dim transition-colors duration-150 focus-visible:border-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-accent/40 disabled:opacity-50 disabled:cursor-not-allowed",
        mono && "font-mono text-[11px]",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";

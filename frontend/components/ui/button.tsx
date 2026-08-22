import * as React from "react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "outline" | "ghost" | "danger" | "accent-glow";
type Size = "sm" | "md" | "lg" | "icon-sm" | "icon-md";

const variants: Record<Variant, string> = {
  primary:
    "bg-accent text-bg font-semibold hover:bg-accent-hover shadow-xs active:scale-[0.98] disabled:opacity-40",
  secondary:
    "bg-surface-2 text-text border border-border hover:bg-surface-3 hover:border-border active:scale-[0.98] disabled:opacity-40",
  outline:
    "bg-transparent text-text border border-border hover:bg-surface-2 hover:text-text active:scale-[0.98] disabled:opacity-40",
  ghost:
    "bg-transparent text-muted hover:bg-surface-2 hover:text-text active:scale-[0.98] disabled:opacity-40",
  danger:
    "bg-signal-rose/10 text-signal-rose border border-signal-rose/30 hover:bg-signal-rose/20 active:scale-[0.98] disabled:opacity-40",
  "accent-glow":
    "bg-accent text-bg font-semibold hover:bg-accent-hover glow-accent active:scale-[0.98] disabled:opacity-40",
};

const sizes: Record<Size, string> = {
  sm: "h-8 px-2.5 text-xs gap-1.5",
  md: "h-9 px-3.5 text-xs font-medium gap-2",
  lg: "h-10 px-4 text-sm font-medium gap-2.5",
  "icon-sm": "h-8 w-8 p-0",
  "icon-md": "h-9 w-9 p-0",
};

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", type = "button", ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex items-center justify-center rounded-md whitespace-nowrap transition-all duration-150 ease-out focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:pointer-events-none cursor-pointer select-none",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  ),
);
Button.displayName = "Button";

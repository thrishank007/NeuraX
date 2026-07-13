import * as React from "react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const variants: Record<Variant, string> = {
  primary:
    "bg-accent text-white hover:bg-accent-hover disabled:opacity-40",
  secondary:
    "bg-surface-2 text-text border border-border hover:bg-border/40 disabled:opacity-40",
  ghost: "bg-transparent text-text hover:bg-surface-2 disabled:opacity-40",
  danger:
    "bg-transparent text-danger border border-danger/40 hover:bg-danger/10 disabled:opacity-40",
};

const sizes: Record<Size, string> = {
  sm: "h-8 px-2.5 text-xs",
  md: "h-10 px-3 text-sm",
  lg: "h-11 px-4 text-sm",
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
        "inline-flex items-center justify-center gap-2 rounded-md font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:pointer-events-none",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  ),
);
Button.displayName = "Button";

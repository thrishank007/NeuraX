import { Badge } from "@/components/ui/badge";

export type StatusKind =
  | "ok"
  | "ready"
  | "degraded"
  | "unavailable"
  | "error"
  | "loading"
  | "offline"
  | "unknown";

function toneFor(kind: StatusKind) {
  switch (kind) {
    case "ok":
    case "ready":
      return "emerald" as const;
    case "degraded":
    case "loading":
      return "warning" as const;
    case "unavailable":
    case "error":
      return "danger" as const;
    case "offline":
      return "cyan" as const;
    default:
      return "neutral" as const;
  }
}

export function StatusChip({
  kind,
  label,
  value,
}: {
  kind: StatusKind;
  label: string;
  value?: string | number;
}) {
  return (
    <Badge tone={toneFor(kind)} dot className="px-2 py-0.5" aria-label={`${label}: ${value ?? kind}`}>
      <span className="font-sans font-medium text-text">{label}</span>
      {value !== undefined && (
        <span className="font-mono text-muted ml-1">
          ({value})
        </span>
      )}
    </Badge>
  );
}

export function mapSystemKind(value: string): StatusKind {
  const v = value.toLowerCase();
  if (v === "ok" || v === "ready") return "ok";
  if (v === "degraded") return "degraded";
  if (v === "unavailable" || v === "error") return "unavailable";
  if (v === "not_initialized" || v === "unknown") return "unknown";
  return "unknown";
}

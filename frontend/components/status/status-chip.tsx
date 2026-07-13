import {
  AlertCircle,
  CheckCircle2,
  CircleDashed,
  Loader2,
  WifiOff,
} from "lucide-react";
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
      return "success" as const;
    case "degraded":
    case "loading":
      return "warning" as const;
    case "unavailable":
    case "error":
      return "danger" as const;
    case "offline":
      return "info" as const;
    default:
      return "neutral" as const;
  }
}

function Icon({ kind }: { kind: StatusKind }) {
  const cls = "h-3.5 w-3.5 shrink-0";
  switch (kind) {
    case "ok":
    case "ready":
      return <CheckCircle2 className={cls} aria-hidden />;
    case "loading":
      return <Loader2 className={`${cls} animate-spin`} aria-hidden />;
    case "offline":
      return <WifiOff className={cls} aria-hidden />;
    case "degraded":
      return <CircleDashed className={cls} aria-hidden />;
    default:
      return <AlertCircle className={cls} aria-hidden />;
  }
}

export function StatusChip({
  kind,
  label,
}: {
  kind: StatusKind;
  label: string;
}) {
  return (
    <Badge tone={toneFor(kind)} aria-label={label}>
      <Icon kind={kind} />
      <span>{label}</span>
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

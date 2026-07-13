import type { SearchResultItem } from "@/types/api";

export type QueryHistoryEntry = {
  id: string;
  query: string;
  type: "text" | "image" | "voice" | "multimodal" | "chat";
  results_count: number;
  processing_time: number;
  timestamp: string;
  transcription?: string;
};

const KEY = "neurax-query-history";
const MAX = 20;

export function loadQueryHistory(): QueryHistoryEntry[] {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as QueryHistoryEntry[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function pushQueryHistory(
  entry: Omit<QueryHistoryEntry, "id" | "timestamp"> & {
    id?: string;
    timestamp?: string;
  },
): QueryHistoryEntry[] {
  const full: QueryHistoryEntry = {
    id: entry.id || `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    timestamp: entry.timestamp || new Date().toISOString(),
    query: entry.query,
    type: entry.type,
    results_count: entry.results_count,
    processing_time: entry.processing_time,
    transcription: entry.transcription,
  };
  const next = [full, ...loadQueryHistory().filter((e) => e.id !== full.id)].slice(
    0,
    MAX,
  );
  try {
    sessionStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* ignore quota */
  }
  return next;
}

export function clearQueryHistory(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

export function storeLastSearch(payload: {
  query: string;
  sources: SearchResultItem[];
  modality: string;
}): void {
  try {
    sessionStorage.setItem(
      "neurax-last-sources",
      JSON.stringify({
        query: payload.query,
        sources: payload.sources,
        citations: [],
        modality: payload.modality,
      }),
    );
  } catch {
    /* ignore */
  }
}

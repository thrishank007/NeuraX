import { getApiBaseUrl } from "@/lib/env";
import type {
  ApiError,
  ChatResponse,
  DocumentItem,
  GraphifyBuildResult,
  GraphifyQueryResult,
  GraphifyStatus,
  HealthResponse,
  JobStatus,
  ModelsStatus,
  SearchResultItem,
  SystemStatus,
} from "@/types/api";

export class ApiClientError extends Error {
  code: string;
  status: number;
  details?: Record<string, unknown>;

  constructor(status: number, payload: ApiError | string) {
    if (typeof payload === "string") {
      super(payload);
      this.code = "http_error";
    } else {
      super(payload.error.message);
      this.code = payload.error.code;
      this.details = payload.error.details;
    }
    this.status = status;
    this.name = "ApiClientError";
  }
}

/** True when a fetch was cancelled via AbortController (Strict Mode remount, navigation). */
export function isAbortError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  if (err instanceof DOMException && err.name === "AbortError") return true;
  if ("name" in err && (err as { name?: string }).name === "AbortError") {
    return true;
  }
  return false;
}

async function parseError(res: Response): Promise<ApiClientError> {
  try {
    const data = (await res.json()) as ApiError;
    if (data?.error?.message) {
      return new ApiClientError(res.status, data);
    }
  } catch {
    /* ignore */
  }
  return new ApiClientError(res.status, res.statusText || "Request failed");
}

async function request<T>(
  path: string,
  init?: RequestInit & { signal?: AbortSignal },
): Promise<T> {
  const url = `${getApiBaseUrl()}${path}`;
  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData
          ? {}
          : { "Content-Type": "application/json" }),
        ...init?.headers,
      },
      cache: "no-store",
    });
  } catch (err) {
    // Preserve abort so callers can ignore Strict Mode / unmount cleanup.
    if (isAbortError(err)) {
      throw err;
    }
    throw new ApiClientError(0, {
      error: {
        code: "backend_unavailable",
        message:
          err instanceof Error
            ? `Backend unreachable: ${err.message}`
            : "Backend unreachable",
      },
    });
  }

  if (!res.ok) {
    throw await parseError(res);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export const api = {
  health: (signal?: AbortSignal) =>
    request<HealthResponse>("/api/health", { signal }),

  systemStatus: (signal?: AbortSignal) =>
    request<SystemStatus>("/api/system/status", { signal }),

  modelsStatus: (signal?: AbortSignal) =>
    request<ModelsStatus>("/api/models/status", { signal }),

  listDocuments: (signal?: AbortSignal) =>
    request<{ documents: DocumentItem[]; total: number }>("/api/documents", {
      signal,
    }),

  getDocument: (id: string, signal?: AbortSignal) =>
    request<DocumentItem>(`/api/documents/${encodeURIComponent(id)}`, {
      signal,
    }),

  deleteDocument: (id: string, signal?: AbortSignal) =>
    request<{ status: string; id: string }>(
      `/api/documents/${encodeURIComponent(id)}`,
      { method: "DELETE", signal },
    ),

  uploadDocuments: (files: File[], signal?: AbortSignal) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request<JobStatus>("/api/documents", {
      method: "POST",
      body: form,
      signal,
    });
  },

  getJob: (jobId: string, signal?: AbortSignal) =>
    request<JobStatus>(`/api/index/jobs/${encodeURIComponent(jobId)}`, {
      signal,
    }),

  cancelJob: (jobId: string, signal?: AbortSignal) =>
    request<JobStatus>(
      `/api/index/jobs/${encodeURIComponent(jobId)}/cancel`,
      { method: "POST", signal },
    ),

  search: async (
    params: {
      query?: string;
      modality?: "text" | "image" | "voice" | "multimodal";
      similarity_threshold?: number;
      k?: number;
      image?: File | null;
      audio?: File | null;
    },
    signal?: AbortSignal,
  ) => {
    const form = new FormData();
    form.append("query", params.query ?? "");
    form.append("modality", params.modality ?? "text");
    if (params.similarity_threshold != null) {
      form.append("similarity_threshold", String(params.similarity_threshold));
    }
    if (params.k != null) {
      form.append("k", String(params.k));
    }
    if (params.image) {
      form.append("image", params.image);
    }
    if (params.audio) {
      form.append("audio", params.audio);
    }
    return request<{
      query: string;
      query_type: string;
      results: SearchResultItem[];
      total_results: number;
      similarity_threshold: number;
      processing_time: number;
      transcription?: string | null;
    }>("/api/search", { method: "POST", body: form, signal });
  },

  chat: (
    body: {
      query: string;
      similarity_threshold?: number;
      max_docs?: number;
      use_knowledge_graph?: boolean;
      mode?: "local" | "cloud";
    },
    signal?: AbortSignal,
  ) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),

  chatStream: async function* (
    body: {
      query: string;
      similarity_threshold?: number;
      max_docs?: number;
      use_knowledge_graph?: boolean;
      mode?: "local" | "cloud";
    },
    signal?: AbortSignal,
  ): AsyncGenerator<{ event: string; data: Record<string, unknown> }> {
    const url = `${getApiBaseUrl()}/api/chat/stream`;
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
        cache: "no-store",
      });
    } catch (err) {
      if (isAbortError(err)) {
        throw err;
      }
      throw new ApiClientError(0, {
        error: {
          code: "backend_unavailable",
          message:
            err instanceof Error
              ? `Backend unreachable: ${err.message}`
              : "Backend unreachable",
        },
      });
    }

    if (!res.ok) {
      throw await parseError(res);
    }
    if (!res.body) {
      throw new ApiClientError(500, "Empty stream body");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        const lines = part.split("\n");
        let event = "message";
        let dataStr = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            event = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataStr += line.slice(5).trim();
          }
        }
        if (!dataStr) continue;
        try {
          yield { event, data: JSON.parse(dataStr) as Record<string, unknown> };
        } catch {
          yield { event, data: { raw: dataStr } };
        }
      }
    }
  },

  getSource: (id: string, signal?: AbortSignal) =>
    request<DocumentItem>(`/api/sources/${encodeURIComponent(id)}`, { signal }),

  getGraph: (signal?: AbortSignal) =>
    request<{ graph: unknown; stats: Record<string, unknown> }>("/api/graph", {
      signal,
    }),

  graphifyStatus: (workspaceId = "default", signal?: AbortSignal) =>
    request<GraphifyStatus>(
      `/api/graphify/status?workspace_id=${encodeURIComponent(workspaceId)}`,
      { signal },
    ),

  graphifyBuild: (force = false, workspaceId = "default", signal?: AbortSignal) =>
    request<GraphifyBuildResult>("/api/graphify/build", {
      method: "POST",
      body: JSON.stringify({ workspace_id: workspaceId, force }),
      signal,
    }),

  graphifyUpdate: (workspaceId = "default", signal?: AbortSignal) =>
    request<GraphifyBuildResult>("/api/graphify/update", {
      method: "POST",
      body: JSON.stringify({ workspace_id: workspaceId }),
      signal,
    }),

  graphifyRebuild: (workspaceId = "default", signal?: AbortSignal) =>
    request<GraphifyBuildResult>("/api/graphify/rebuild", {
      method: "POST",
      body: JSON.stringify({ workspace_id: workspaceId, force: true }),
      signal,
    }),

  graphifyStats: (workspaceId = "default", signal?: AbortSignal) =>
    request<{ stats: Record<string, unknown> }>(
      `/api/graphify/stats?workspace_id=${encodeURIComponent(workspaceId)}`,
      { signal },
    ),

  graphifyData: (
    params: {
      workspace_id?: string;
      node_type?: string;
      community?: string;
      source_file?: string;
      confidence?: string;
    } = {},
    signal?: AbortSignal,
  ) => {
    const q = new URLSearchParams();
    q.set("workspace_id", params.workspace_id || "default");
    if (params.node_type) q.set("node_type", params.node_type);
    if (params.community) q.set("community", params.community);
    if (params.source_file) q.set("source_file", params.source_file);
    if (params.confidence) q.set("confidence", params.confidence);
    return request<{
      nodes: unknown[];
      edges: unknown[];
      stats?: Record<string, unknown>;
    }>(`/api/graphify/data?${q.toString()}`, { signal });
  },

  graphifyNodes: (workspaceId = "default", signal?: AbortSignal) =>
    request<{ labels: string[] }>(
      `/api/graphify/nodes?workspace_id=${encodeURIComponent(workspaceId)}`,
      { signal },
    ),

  graphifyQuery: (
    body: { question: string; budget?: number; workspace_id?: string },
    signal?: AbortSignal,
  ) =>
    request<GraphifyQueryResult>("/api/graphify/query", {
      method: "POST",
      body: JSON.stringify({
        workspace_id: body.workspace_id || "default",
        question: body.question,
        budget: body.budget,
      }),
      signal,
    }),

  graphifyExplain: (
    body: { node_name: string; workspace_id?: string },
    signal?: AbortSignal,
  ) =>
    request<GraphifyQueryResult>("/api/graphify/explain", {
      method: "POST",
      body: JSON.stringify({
        workspace_id: body.workspace_id || "default",
        node_name: body.node_name,
      }),
      signal,
    }),

  graphifyPath: (
    body: { source: string; target: string; workspace_id?: string },
    signal?: AbortSignal,
  ) =>
    request<GraphifyQueryResult>("/api/graphify/path", {
      method: "POST",
      body: JSON.stringify({
        workspace_id: body.workspace_id || "default",
        source: body.source,
        target: body.target,
      }),
      signal,
    }),

  graphifyCorpus: (workspaceId = "default", signal?: AbortSignal) =>
    request<{ files: Record<string, unknown>[]; count: number }>(
      `/api/graphify/corpus?workspace_id=${encodeURIComponent(workspaceId)}`,
      { signal },
    ),

  feedback: (
    body: {
      query: string;
      response: string;
      rating: number;
      comments?: string;
    },
    signal?: AbortSignal,
  ) =>
    request<{ status: string; feedback_id: string }>("/api/feedback", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
};

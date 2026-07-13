import { getApiBaseUrl } from "@/lib/env";
import type {
  ApiError,
  ChatResponse,
  DocumentItem,
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

import { apiClient } from "./client";
import type { QueryRequest, QueryResponse } from "@/lib/types/api";

export const queryApi = {
  async processQuery(request: QueryRequest): Promise<QueryResponse> {
    return apiClient.post<QueryResponse>("/api/query", request);
  },

  async processVoiceQuery(audioFile: File): Promise<QueryResponse> {
    const formData = new FormData();
    formData.append("audio_file", audioFile);
    
    return apiClient.post<QueryResponse>("/api/query/voice", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  },

  async getQueryHistory(limit: number = 20) {
    return apiClient.get(`/api/query/history?limit=${limit}`);
  },

  async getQuerySuggestions(partial: string, limit: number = 5) {
    return apiClient.get(`/api/query/suggestions?partial=${encodeURIComponent(partial)}&limit=${limit}`);
  },
};

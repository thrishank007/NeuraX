import { apiClient } from "./client";
import type { AnalyticsMetrics } from "@/lib/types/api";

export const analyticsApi = {
  async getMetrics(timeRangeHours: number = 24): Promise<AnalyticsMetrics> {
    return apiClient.get<AnalyticsMetrics>(
      `/api/analytics/metrics?time_range_hours=${timeRangeHours}`
    );
  },

  async getUsageStats(timeRangeHours: number = 24) {
    return apiClient.get(`/api/analytics/usage?time_range_hours=${timeRangeHours}`);
  },

  async getSecurityEvents(limit: number = 50) {
    return apiClient.get(`/api/analytics/security?limit=${limit}`);
  },

  async getKnowledgeGraph() {
    return apiClient.get("/api/knowledge-graph");
  },
};

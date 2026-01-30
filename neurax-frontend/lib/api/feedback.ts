import { apiClient } from "./client";
import type { FeedbackRequest } from "@/lib/types/api";

export const feedbackApi = {
  async submitFeedback(feedback: FeedbackRequest) {
    return apiClient.post("/api/feedback", feedback);
  },

  async getFeedbackHistory(limit: number = 20) {
    return apiClient.get(`/api/feedback/history?limit=${limit}`);
  },

  async getFeedbackAnalytics() {
    return apiClient.get("/api/feedback/analytics");
  },
};

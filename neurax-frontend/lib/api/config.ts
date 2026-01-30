import { apiClient } from "./client";
import type { ConfigResponse } from "@/lib/types/api";

export const configApi = {
  async getConfig(): Promise<ConfigResponse> {
    return apiClient.get<ConfigResponse>("/api/config");
  },

  async updateConfig(config: Partial<ConfigResponse>) {
    return apiClient.put("/api/config", config);
  },

  async validateConfig(config: Partial<ConfigResponse>) {
    return apiClient.post("/api/config/validate", config);
  },
};

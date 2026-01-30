import { apiClient } from "./client";
import type { FileUploadResponse } from "@/lib/types/api";

export const documentsApi = {
  async uploadFiles(
    files: File[],
    onProgress?: (progress: number) => void
  ): Promise<FileUploadResponse[]> {
    return apiClient.uploadFiles(files, onProgress);
  },

  async listFiles() {
    return apiClient.get("/api/files");
  },

  async deleteFile(fileId: string) {
    return apiClient.delete(`/api/files/${fileId}`);
  },

  async getProcessingStatus(fileId: string) {
    return apiClient.get(`/api/processing-status/${fileId}`);
  },
};

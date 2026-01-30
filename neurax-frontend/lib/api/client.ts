import axios, { AxiosInstance, AxiosError } from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      timeout: 300000, // 5 minutes for file uploads (processing can take time)
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response) {
          // Server responded with error
          const message = (error.response.data as any)?.detail || error.message;
          return Promise.reject(new Error(message));
        } else if (error.request) {
          // Request made but no response
          return Promise.reject(new Error("Network error. Please check your connection."));
        } else {
          // Something else happened
          return Promise.reject(error);
        }
      }
    );
  }

  async get<T>(url: string, config?: any): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: any): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }

  // File upload helper
  async uploadFiles(files: File[], onProgress?: (progress: number) => void): Promise<any> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const response = await this.client.post("/api/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            onProgress(progress);
          }
        },
        timeout: 300000, // 5 minutes for processing
      });

      return response.data;
    } catch (error: any) {
      // Enhanced error logging
      console.error("Upload error:", error);
      if (error.response) {
        throw new Error(error.response.data?.detail || error.response.data?.message || "Upload failed");
      } else if (error.request) {
        throw new Error("No response from server. Check if backend is running.");
      } else {
        throw new Error(error.message || "Upload failed");
      }
    }
  }
}

export const apiClient = new ApiClient();

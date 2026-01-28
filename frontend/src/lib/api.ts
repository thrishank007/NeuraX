import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  ApiResponse,
  HealthResponse,
  DetailedHealthResponse,
  DocumentResponse,
  DocumentListResponse,
  SearchResponse,
  ChatResponse,
  ChatRequest,
  Document,
  SearchResult,
  ChatMessage,
  EvaluationMetrics,
} from '@/types/api';

class NeuraXApiClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('[API] Request error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`[API] Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('[API] Response error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Health Check Endpoints
  async getHealth(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  async getDetailedHealth(): Promise<DetailedHealthResponse> {
    const response = await this.client.get<DetailedHealthResponse>('/health/detailed');
    return response.data;
  }

  async getReadiness(): Promise<{ status: string }> {
    const response = await this.client.get<{ status: string }>('/health/ready');
    return response.data;
  }

  async getLiveness(): Promise<{ status: string; timestamp: number }> {
    const response = await this.client.get<{ status: string; timestamp: number }>('/health/live');
    return response.data;
  }

  // Document Management Endpoints
  async uploadDocument(file: File): Promise<DocumentResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<DocumentResponse>(
      '/api/v1/documents/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  }

  async listDocuments(limit: number = 50, offset: number = 0): Promise<DocumentListResponse> {
    const response = await this.client.get<DocumentListResponse>(
      `/api/v1/documents/?limit=${limit}&offset=${offset}`
    );
    return response.data;
  }

  async searchDocuments(query: string, limit: number = 10): Promise<SearchResponse> {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('limit', limit.toString());

    const response = await this.client.post<SearchResponse>(
      '/api/v1/documents/search',
      formData
    );
    return response.data;
  }

  async deleteDocument(documentId: string): Promise<{ success: boolean; message: string }> {
    const response = await this.client.delete<{ success: boolean; message: string }>(
      `/api/v1/documents/${documentId}`
    );
    return response.data;
  }

  // Chat Endpoints
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await this.client.post<ChatResponse>(
      '/api/v1/documents/chat',
      request
    );
    return response.data;
  }

  // Evaluation Endpoints
  async getEvaluationMetrics(timeRangeHours: number = 24): Promise<EvaluationMetrics> {
    const response = await this.client.get<EvaluationMetrics>(
      `/api/v1/evaluation/metrics?time_range_hours=${timeRangeHours}`
    );
    return response.data;
  }

  async getRetrievalMetrics(timeRangeHours: number = 24): Promise<{ metrics: any; time_range_hours: number; timestamp: string }> {
    const response = await this.client.get<{ metrics: any; time_range_hours: number; timestamp: string }>(
      `/api/v1/evaluation/retrieval?time_range_hours=${timeRangeHours}`
    );
    return response.data;
  }

  async getGenerationMetrics(timeRangeHours: number = 24): Promise<{ metrics: any; time_range_hours: number; timestamp: string }> {
    const response = await this.client.get<{ metrics: any; time_range_hours: number; timestamp: string }>(
      `/api/v1/evaluation/generation?time_range_hours=${timeRangeHours}`
    );
    return response.data;
  }

  async getLatencyMetrics(timeRangeHours: number = 24): Promise<{ metrics: any; time_range_hours: number; timestamp: string }> {
    const response = await this.client.get<{ metrics: any; time_range_hours: number; timestamp: string }>(
      `/api/v1/evaluation/latency?time_range_hours=${timeRangeHours}`
    );
    return response.data;
  }

  // Utility Methods
  async ping(): Promise<{ status: string; timestamp: string }> {
    const response = await this.client.get<{ status: string; timestamp: string }>('/ping');
    return response.data;
  }

  async getApiInfo(): Promise<{ message: string; version: string; status: string; docs: string }> {
    const response = await this.client.get<{ message: string; version: string; status: string; docs: string }>('/');
    return response.data;
  }

  // Error handling helper
  private handleError(error: any): string {
    if (error.response) {
      // Server responded with error status
      return error.response.data?.detail || error.response.data?.error || `HTTP ${error.response.status}`;
    } else if (error.request) {
      // Request was made but no response received
      return 'Network error - no response from server';
    } else {
      // Something else happened
      return error.message || 'Unknown error occurred';
    }
  }

  // Test connection
  async testConnection(): Promise<boolean> {
    try {
      await this.ping();
      return true;
    } catch (error) {
      console.error('Connection test failed:', error);
      return false;
    }
  }
}

// Create and export singleton instance
export const apiClient = new NeuraXApiClient();

// Export types and utilities
export default NeuraXApiClient;
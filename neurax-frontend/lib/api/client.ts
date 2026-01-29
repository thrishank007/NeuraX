import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  ApiResponse,
  PaginatedResponse,
  UploadResponse,
  QueryResponse,
  ResponseGenerationResponse,
  SystemConfig,
  Analytics,
  SecurityEvent,
  Feedback,
  ProcessingStatus,
} from '@/types';

// API Client Configuration
class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
      timeout: 120000, // 2 minutes for file processing
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Generic API methods
  private async handleRequest<T>(promise: Promise<AxiosResponse<T>>): Promise<T> {
    try {
      const response = await promise;
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.message || error.message || 'API request failed';
      throw new Error(message);
    }
  }

  // File Upload API
  async uploadFiles(files: File[]): Promise<UploadResponse> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    const response = await this.client.post<UploadResponse>('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          // Emit progress event for UI updates
          window.dispatchEvent(new CustomEvent('upload-progress', { detail: progress }));
        }
      },
    });

    return response.data;
  }

  async getUploadedFiles(page = 1, limit = 20): Promise<PaginatedResponse<any>> {
    return this.handleRequest(
      this.client.get(`/api/files?page=${page}&limit=${limit}`)
    );
  }

  async deleteFile(fileId: string): Promise<ApiResponse> {
    return this.handleRequest(
      this.client.delete(`/api/files/${fileId}`)
    );
  }

  async getFileStatus(fileId: string): Promise<ProcessingStatus> {
    return this.handleRequest(
      this.client.get(`/api/processing-status/${fileId}`)
    );
  }

  // Query API
  async processTextQuery(
    text: string,
    similarityThreshold = 0.5,
    options?: {
      includeImages?: boolean;
      includeDocuments?: boolean;
      includeAudio?: boolean;
      maxResults?: number;
    }
  ): Promise<QueryResponse> {
    return this.handleRequest(
      this.client.post('/api/query/text', {
        text,
        similarity_threshold: similarityThreshold,
        options: {
          include_images: options?.includeImages ?? true,
          include_documents: options?.includeDocuments ?? true,
          include_audio: options?.includeAudio ?? true,
          max_results: options?.maxResults ?? 10,
        },
      })
    );
  }

  async processImageQuery(
    imageFile: File,
    similarityThreshold = 0.5,
    options?: {
      maxResults?: number;
      textQuery?: string;
    }
  ): Promise<QueryResponse> {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('similarity_threshold', similarityThreshold.toString());
    
    if (options?.textQuery) {
      formData.append('text_query', options.textQuery);
    }
    if (options?.maxResults) {
      formData.append('max_results', options.maxResults.toString());
    }

    return this.handleRequest(
      this.client.post('/api/query/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
    );
  }

  async processVoiceQuery(
    audioFile: File,
    similarityThreshold = 0.5,
    options?: {
      maxResults?: number;
      language?: string;
    }
  ): Promise<QueryResponse> {
    const formData = new FormData();
    formData.append('audio', audioFile);
    formData.append('similarity_threshold', similarityThreshold.toString());
    
    if (options?.maxResults) {
      formData.append('max_results', options.maxResults.toString());
    }
    if (options?.language) {
      formData.append('language', options.language);
    }

    return this.handleRequest(
      this.client.post('/api/query/voice', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
    );
  }

  async processMultimodalQuery(
    text: string,
    imageFile?: File,
    similarityThreshold = 0.5,
    options?: {
      maxResults?: number;
    }
  ): Promise<QueryResponse> {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('similarity_threshold', similarityThreshold.toString());
    
    if (imageFile) {
      formData.append('image', imageFile);
    }
    if (options?.maxResults) {
      formData.append('max_results', options.maxResults.toString());
    }

    return this.handleRequest(
      this.client.post('/api/query/multimodal', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
    );
  }

  async getQueryHistory(page = 1, limit = 20): Promise<PaginatedResponse<any>> {
    return this.handleRequest(
      this.client.get(`/api/query/history?page=${page}&limit=${limit}`)
    );
  }

  async getQuerySuggestions(query: string): Promise<ApiResponse<string[]>> {
    return this.handleRequest(
      this.client.get(`/api/query/suggestions?q=${encodeURIComponent(query)}`)
    );
  }

  // Response Generation API
  async generateResponse(request: {
    query: string;
    context: any[];
    model?: 'gemma-3n' | 'qwen3-4b';
    maxTokens?: number;
    temperature?: number;
    includeCitations?: boolean;
  }): Promise<ResponseGenerationResponse> {
    return this.handleRequest(
      this.client.post('/api/generate-response', {
        query: request.query,
        context: request.context,
        model: request.model || 'gemma-3n',
        max_tokens: request.maxTokens || 1024,
        temperature: request.temperature || 0.7,
        include_citations: request.includeCitations ?? true,
      })
    );
  }

  // Analytics API
  async getAnalytics(timeRange?: {
    start?: string;
    end?: string;
  }): Promise<Analytics> {
    const params = new URLSearchParams();
    if (timeRange?.start) params.append('start', timeRange.start);
    if (timeRange?.end) params.append('end', timeRange.end);

    return this.handleRequest(
      this.client.get(`/api/analytics?${params.toString()}`)
    );
  }

  async getSystemMetrics(): Promise<any> {
    return this.handleRequest(
      this.client.get('/api/analytics/metrics')
    );
  }

  async getUsageStatistics(timeRange?: {
    start?: string;
    end?: string;
  }): Promise<any> {
    const params = new URLSearchParams();
    if (timeRange?.start) params.append('start', timeRange.start);
    if (timeRange?.end) params.append('end', timeRange.end);

    return this.handleRequest(
      this.client.get(`/api/analytics/usage?${params.toString()}`)
    );
  }

  // Security API
  async getSecurityEvents(page = 1, limit = 20): Promise<PaginatedResponse<SecurityEvent>> {
    return this.handleRequest(
      this.client.get(`/api/security/events?page=${page}&limit=${limit}`)
    );
  }

  async getAnomalyDetection(): Promise<any> {
    return this.handleRequest(
      this.client.get('/api/security/anomalies')
    );
  }

  async getAuditLogs(page = 1, limit = 20): Promise<PaginatedResponse<any>> {
    return this.handleRequest(
      this.client.get(`/api/audit/logs?page=${page}&limit=${limit}`)
    );
  }

  // Feedback API
  async submitFeedback(feedback: {
    queryId: string;
    responseId?: string;
    rating: number;
    comments?: string;
    isHelpful?: boolean;
    metadata?: Record<string, any>;
  }): Promise<ApiResponse> {
    return this.handleRequest(
      this.client.post('/api/feedback', feedback)
    );
  }

  async getFeedbackHistory(page = 1, limit = 20): Promise<PaginatedResponse<Feedback>> {
    return this.handleRequest(
      this.client.get(`/api/feedback/history?page=${page}&limit=${limit}`)
    );
  }

  async getFeedbackAnalytics(): Promise<any> {
    return this.handleRequest(
      this.client.get('/api/feedback/analytics')
    );
  }

  // Configuration API
  async getSystemConfig(): Promise<SystemConfig> {
    return this.handleRequest(
      this.client.get('/api/config')
    );
  }

  async updateSystemConfig(config: Partial<SystemConfig>): Promise<SystemConfig> {
    return this.handleRequest(
      this.client.put('/api/config', config)
    );
  }

  async validateConfig(config: Partial<SystemConfig>): Promise<ApiResponse> {
    return this.handleRequest(
      this.client.post('/api/config/validate', config)
    );
  }

  // Knowledge Graph API
  async getKnowledgeGraph(): Promise<any> {
    return this.handleRequest(
      this.client.get('/api/knowledge-graph')
    );
  }

  // Export API
  async exportResults(format: 'json' | 'csv' | 'pdf', data: any): Promise<Blob> {
    const response = await this.client.post('/api/export', {
      format,
      data,
    }, {
      responseType: 'blob',
    });

    return response.data;
  }

  async exportAnalytics(format: 'json' | 'csv' | 'pdf', timeRange?: {
    start?: string;
    end?: string;
  }): Promise<Blob> {
    const params = new URLSearchParams();
    params.append('format', format);
    if (timeRange?.start) params.append('start', timeRange.start);
    if (timeRange?.end) params.append('end', timeRange.end);

    const response = await this.client.get(`/api/export/analytics?${params.toString()}`, {
      responseType: 'blob',
    });

    return response.data;
  }

  // Health Check
  async healthCheck(): Promise<ApiResponse> {
    return this.handleRequest(
      this.client.get('/health')
    );
  }

  // System Status
  async getSystemStatus(): Promise<any> {
    return this.handleRequest(
      this.client.get('/api/status')
    );
  }
}

// Create singleton instance
export const apiClient = new ApiClient();
export default apiClient;
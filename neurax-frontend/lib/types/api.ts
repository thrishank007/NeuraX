export interface QueryRequest {
  query: string;
  query_type?: "text" | "image" | "multimodal";
  k?: number;
  similarity_threshold?: number;
  filters?: Record<string, any>;
  generate_response?: boolean;
}

export interface QueryResponse {
  query_id: string;
  query: string;
  response_text?: string;
  results: SearchResult[];
  citations: Citation[];
  processing_time: number;
  total_results: number;
  model_used?: string;
}

export interface SearchResult {
  document_id: string;
  similarity_score: number;
  content_preview: string;
  file_path: string;
  file_type: string;
  metadata: Record<string, any>;
}

export interface Citation {
  citation_id: number;
  source_document: string;
  source_type: string;
  content_snippet: string;
  confidence_score: number;
  file_path: string;
  page_number?: number;
}

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  file_type: string;
  status: "success" | "error";
  processing_time: number;
  message: string;
}

export interface FeedbackRequest {
  query: string;
  response: string;
  rating: number;
  comments?: string;
  query_metadata?: Record<string, any>;
}

export interface AnalyticsMetrics {
  metrics: {
    retrieval?: any;
    generation?: any;
    latency?: any;
  };
  time_range_hours: number;
  timestamp: string;
}

export interface ConfigResponse {
  lm_studio_url: string;
  similarity_threshold: number;
  max_results: number;
  model_preference: string;
  supported_formats: string[];
}

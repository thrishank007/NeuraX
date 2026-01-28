// API Types
export interface ApiResponse<T = any> {
  success?: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Health Check Types
export interface HealthResponse {
  status: string;
  timestamp: number;
  uptime: number;
  version: string;
  components: Record<string, boolean>;
  errors: string[];
}

export interface DetailedHealthResponse extends HealthResponse {
  overall_healthy: boolean;
  component_errors: string[];
  system_info: Record<string, any>;
}

// Document Types
export interface Document {
  document_id: string;
  filename: string;
  status: 'processed' | 'processing' | 'failed';
  upload_time: number;
  chunks_count: number;
  file_size: number;
  content_type?: string;
}

export interface DocumentResponse {
  success: boolean;
  document_id: string;
  filename: string;
  status: string;
  chunks_created: number;
  processing_time: number;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
}

export interface SearchResult {
  content: string;
  source: string;
  similarity: number;
  chunk_id: string;
}

export interface SearchResponse {
  success: boolean;
  query: string;
  results: SearchResult[];
  total_results: number;
  search_time: number;
}

// Chat Types
export interface ChatMessage {
  id: string;
  message: string;
  response: string;
  sources: SearchResult[];
  session_id: string;
  timestamp: number;
  generation_time: number;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  include_sources?: boolean;
}

export interface ChatResponse {
  success: boolean;
  response: string;
  sources: SearchResult[];
  session_id: string;
  generation_time: number;
}

// Evaluation Types
export interface EvaluationMetrics {
  retrieval: Record<string, any>;
  generation: Record<string, any>;
  latency: Record<string, any>;
  overall_quality_score: number;
  total_test_cases: number;
  time_range_hours: number;
  timestamp: string;
}

export interface TestCase {
  query: string;
  expected_documents: string[];
}

export interface EvaluationConfig {
  name: string;
  description?: string;
  test_cases?: TestCase[];
  k_values?: number[];
  similarity_thresholds?: number[];
  include_generation_metrics?: boolean;
  include_latency_metrics?: boolean;
  max_concurrent_requests?: number;
}

export interface EvaluationRun {
  run_id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at?: string;
  completed_at?: string;
  total_cases: number;
  overall_score?: number;
  mrr?: number;
  grounding_score?: number;
}

// UI State Types
export interface AppState {
  isLoading: boolean;
  error: string | null;
  currentSession: string | null;
  documents: Document[];
  chatHistory: ChatMessage[];
  systemHealth: DetailedHealthResponse | null;
}

// Component Props Types
export interface FileUploadProps {
  onUpload: (file: File) => Promise<void>;
  isUploading: boolean;
  maxSize?: number;
  allowedTypes?: string[];
}

export interface ChatInterfaceProps {
  onSendMessage: (message: string) => Promise<void>;
  messages: ChatMessage[];
  isLoading: boolean;
  sessionId: string | null;
}

export interface SearchInterfaceProps {
  onSearch: (query: string) => Promise<void>;
  results: SearchResult[];
  isLoading: boolean;
  query: string;
}

export interface MetricsDisplayProps {
  metrics: EvaluationMetrics | null;
  isLoading: boolean;
}

// Form Types
export interface UploadFormData {
  file: File;
}

export interface ChatFormData {
  message: string;
  session_id?: string;
  include_sources?: boolean;
}

export interface SearchFormData {
  query: string;
  limit?: number;
}
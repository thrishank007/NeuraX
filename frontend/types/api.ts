export type ApiError = {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

export type HealthResponse = {
  status: string;
  service: string;
  version: string;
};

export type SystemStatus = {
  overall: string;
  backend: string;
  vector_store: string;
  lm_studio: string;
  components: Record<string, boolean>;
  collection: {
    total_documents?: number;
    file_types?: Record<string, number>;
    collection_name?: string;
  };
  supported_formats: string[];
  max_upload_mb: number;
  offline_mode: boolean;
};

export type ModelsStatus = {
  lm_studio_reachable: boolean;
  models_loaded: number;
  model_ids: string[];
  current_model?: string | null;
  supports_multimodal: boolean;
  details: Record<string, unknown>;
  message: string;
};

export type DocumentItem = {
  id: string;
  file_path: string;
  file_name: string;
  file_type: string;
  embedding_type: string;
  timestamp: string;
  content_preview: string;
  metadata: Record<string, unknown>;
};

export type JobStatus = {
  job_id: string;
  status: string;
  progress: number;
  total: number;
  processed: number;
  logs: string[];
  errors: string[];
  document_ids: string[];
  cancel_requested: boolean;
};

export type SearchResultItem = {
  id: string;
  file_path: string;
  file_name: string;
  file_type: string;
  similarity_score: number;
  content_preview: string;
  metadata: Record<string, unknown>;
  text_similarity?: number | null;
  image_similarity?: number | null;
};

export type CitationItem = {
  citation_id: number;
  source_document: string;
  source_type: string;
  content_snippet: string;
  confidence_score: number;
  file_path: string;
  page_number?: number | null;
  expandable_link: string;
  timestamp: string;
};

export type ChatResponse = {
  query: string;
  response: string;
  citations: CitationItem[];
  confidence: number;
  processing_time: number;
  sources: SearchResultItem[];
  model_used: string;
  lm_studio_available: boolean;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  /** Original user query (on assistant messages) for feedback */
  query?: string;
  citations?: CitationItem[];
  sources?: SearchResultItem[];
  confidence?: number;
  processing_time?: number;
  model_used?: string;
  status?: "pending" | "streaming" | "done" | "error";
};

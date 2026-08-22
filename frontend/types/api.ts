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
  graphify?: {
    available?: boolean;
    enabled?: boolean;
    version?: string | null;
    corpus_file_count?: number;
    artifacts_available?: boolean;
    build_running?: boolean;
    model?: string;
    error?: string;
  };
  cloud_llm?: {
    configured: boolean;
    api_url: string;
    model: string;
  };
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
  graph_context_used?: boolean;
  graph_warning?: string | null;
};

export type GraphifyStatus = {
  enabled: boolean;
  available: boolean;
  executable?: string | null;
  version?: string | null;
  install_hint: string;
  workspace_id: string;
  corpus_file_count: number;
  last_build_time?: string | null;
  build_running: boolean;
  artifacts_available: boolean;
  artifact_paths: Record<string, string | null | undefined>;
  lm_studio_base_url: string;
  model: string;
  backend: string;
  auto_update_after_ingestion: boolean;
  endpoint_allowed: boolean;
  endpoint_warning?: string | null;
  capabilities?: Record<string, unknown>;
  last_error?: string | null;
};

export type GraphifyBuildResult = {
  success: boolean;
  workspace_id: string;
  mode: string;
  message: string;
  duration_seconds?: number;
  artifacts?: Record<string, string | null | undefined>;
  logs?: string;
  error?: string | null;
};

export type GraphifyQueryResult = {
  success: boolean;
  kind: string;
  output: string;
  nodes?: Record<string, unknown>[];
  sources?: string[];
  error?: string | null;
  duration_seconds?: number;
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
  mode?: "local" | "cloud";
  status?: "pending" | "streaming" | "done" | "error";
};

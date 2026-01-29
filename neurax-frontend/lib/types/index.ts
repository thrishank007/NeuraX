// Core types for NeuraX frontend

export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  role: 'admin' | 'user' | 'viewer';
  createdAt: string;
  lastActive: string;
}

export interface FileUpload {
  id: string;
  fileName: string;
  filePath: string;
  fileType: 'document' | 'image' | 'audio' | 'unknown';
  fileSize: number;
  mimeType: string;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
  metadata?: Record<string, any>;
  uploadedAt: string;
  processedAt?: string;
}

export interface SearchResult {
  id: string;
  filePath: string;
  fileName: string;
  fileType: 'document' | 'image' | 'audio' | 'unknown';
  similarityScore: number;
  confidence: number;
  contentPreview: string;
  metadata: Record<string, any>;
  pageNumber?: number;
  timestamp: string;
  highlights?: string[];
  thumbnailUrl?: string;
  textEmbedding?: number[];
  imageEmbedding?: number[];
}

export interface Query {
  id: string;
  type: 'text' | 'image' | 'voice' | 'multimodal';
  text?: string;
  image?: File;
  audio?: File;
  timestamp: string;
  similarityThreshold: number;
  results?: SearchResult[];
  processingTime?: number;
  status: 'pending' | 'processing' | 'completed' | 'error';
  error?: string;
}

export interface AIResponse {
  id: string;
  queryId: string;
  responseText: string;
  confidence: number;
  citations: Citation[];
  processingTime: number;
  modelUsed: 'gemma-3n' | 'qwen3-4b' | 'unknown';
  timestamp: string;
  feedback?: Feedback;
}

export interface Citation {
  id: string;
  citationId: string;
  filePath: string;
  fileName: string;
  sourceType: 'document' | 'image' | 'audio';
  pageNumber?: number;
  contentSnippet: string;
  confidenceScore: number;
  timestamp: string;
  url?: string;
  position?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface Feedback {
  id: string;
  queryId: string;
  responseId: string;
  rating: number; // 1-5
  comments?: string;
  isHelpful: boolean;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface Analytics {
  queryStats: {
    totalQueries: number;
    textQueries: number;
    imageQueries: number;
    voiceQueries: number;
    multimodalQueries: number;
    avgProcessingTime: number;
    successRate: number;
  };
  fileStats: {
    totalFiles: number;
    documentFiles: number;
    imageFiles: number;
    audioFiles: number;
    totalSize: number;
    avgProcessingTime: number;
  };
  systemStats: {
    uptime: number;
    memoryUsage: number;
    cpuUsage: number;
    diskUsage: number;
  };
  usageTrends: {
    date: string;
    queries: number;
    uploads: number;
  }[];
  popularQueries: {
    query: string;
    count: number;
    avgRating: number;
  }[];
}

export interface SecurityEvent {
  id: string;
  type: 'anomaly' | 'audit' | 'alert' | 'error';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  source: string;
  timestamp: string;
  metadata?: Record<string, any>;
  resolved: boolean;
  resolvedAt?: string;
  resolvedBy?: string;
}

export interface SystemConfig {
  apiUrl: string;
  wsUrl: string;
  lmStudioUrl: string;
  maxFileSize: number;
  allowedFileTypes: string[];
  enableAnalytics: boolean;
  enableDarkMode: boolean;
  defaultSimilarityThreshold: number;
  maxQueryHistory: number;
  enableVoiceInput: boolean;
  models: {
    primary: 'gemma-3n' | 'qwen3-4b';
    fallback: 'gemma-3n' | 'qwen3-4b';
  };
  performance: {
    batchSize: number;
    maxConcurrency: number;
    cacheEnabled: boolean;
    cacheTimeout: number;
  };
  security: {
    auditLogging: boolean;
    anomalyDetection: boolean;
    rateLimiting: boolean;
    maxUploadsPerHour: number;
  };
}

export interface KnowledgeGraph {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  metadata: {
    totalNodes: number;
    totalEdges: number;
    createdAt: string;
    lastUpdated: string;
  };
}

export interface KnowledgeNode {
  id: string;
  label: string;
  type: 'document' | 'concept' | 'entity' | 'topic';
  properties: Record<string, any>;
  position?: {
    x: number;
    y: number;
    z: number;
  };
}

export interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  type: 'similarity' | 'related' | 'contained_in' | 'mentions';
  weight: number;
  properties: Record<string, any>;
}

export interface ProcessingStatus {
  jobId: string;
  type: 'file_upload' | 'query' | 'indexing';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number; // 0-100
  message: string;
  result?: any;
  error?: string;
  startedAt: string;
  completedAt?: string;
  metadata?: Record<string, any>;
}

// UI-specific types
export interface ThemeConfig {
  mode: 'light' | 'dark' | 'system';
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  foreground: string;
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  action?: {
    label: string;
    handler: () => void;
  };
  duration?: number; // Auto-dismiss duration in ms
}

// Form types
export interface QueryFormData {
  type: 'text' | 'image' | 'voice' | 'multimodal';
  text?: string;
  image?: File | null;
  audio?: File | null;
  similarityThreshold: number;
  includeImages: boolean;
  includeDocuments: boolean;
  includeAudio: boolean;
  maxResults: number;
}

export interface FeedbackFormData {
  queryId: string;
  responseId: string;
  rating: number;
  comments?: string;
  tags?: string[];
  isPublic: boolean;
}

export interface SettingsFormData {
  general: {
    defaultSimilarityThreshold: number;
    maxResults: number;
    enableNotifications: boolean;
    enableSounds: boolean;
    language: string;
  };
  models: {
    primaryModel: 'gemma-3n' | 'qwen3-4b';
    fallbackModel: 'gemma-3n' | 'qwen3-4b';
    maxTokens: number;
    temperature: number;
  };
  privacy: {
    enableAnalytics: boolean;
    enableUsageTracking: boolean;
    dataRetentionDays: number;
    anonymizeLogs: boolean;
  };
}

// API Response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
  requestId: string;
}

export interface PaginatedResponse<T = any> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

export interface UploadResponse {
  success: boolean;
  files: FileUpload[];
  totalFiles: number;
  totalSize: number;
  errors?: {
    fileName: string;
    error: string;
  }[];
}

export interface QueryResponse {
  query: Query;
  results: SearchResult[];
  totalResults: number;
  processingTime: number;
  similarQueries?: string[];
  suggestions?: string[];
}

export interface ResponseGenerationRequest {
  query: string;
  context: SearchResult[];
  model?: 'gemma-3n' | 'qwen3-4b';
  maxTokens?: number;
  temperature?: number;
  includeCitations: boolean;
}

export interface ResponseGenerationResponse extends ApiResponse<AIResponse> {
  alternatives?: AIResponse[];
  usedContext: SearchResult[];
  modelInfo: {
    name: string;
    parameters: Record<string, any>;
    capabilities: string[];
  };
}
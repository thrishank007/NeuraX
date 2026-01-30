import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const queryApi = {
  submit: async (query: string, image?: File) => {
    const formData = new FormData();
    formData.append('query', query);
    if (image) {
      formData.append('image', image);
    }
    
    const response = await axios.post(`${API_BASE_URL}/api/query`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  getHistory: async () => {
    const response = await apiClient.get('/api/query/history');
    return response.data;
  },
};

export const documentApi = {
  upload: async (files: File[]) => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    
    const response = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  list: async () => {
    const response = await apiClient.get('/api/files');
    return response.data;
  },
  
  delete: async (id: string) => {
    const response = await apiClient.delete(`/api/files/${id}`);
    return response.data;
  },
};

export const analyticsApi = {
  getMetrics: async () => {
    const response = await apiClient.get('/api/analytics/metrics');
    return response.data;
  },
  
  getKnowledgeGraph: async () => {
    const response = await apiClient.get('/api/knowledge-graph');
    return response.data;
  },
};

export const feedbackApi = {
  submit: async (feedback: { query: string; response: string; rating: number; comments: string }) => {
    const response = await apiClient.post('/api/feedback', feedback);
    return response.data;
  },
};

export default apiClient;

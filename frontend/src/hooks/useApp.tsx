'use client';

import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { AppState, ChatMessage, Document, DetailedHealthResponse } from '@/types/api';
import { apiClient } from '@/lib/api';

// Initial state
const initialState: AppState = {
  isLoading: false,
  error: null,
  currentSession: null,
  documents: [],
  chatHistory: [],
  systemHealth: null,
};

// Action types
type AppAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_CURRENT_SESSION'; payload: string | null }
  | { type: 'ADD_DOCUMENT'; payload: Document }
  | { type: 'SET_DOCUMENTS'; payload: Document[] }
  | { type: 'REMOVE_DOCUMENT'; payload: string }
  | { type: 'ADD_CHAT_MESSAGE'; payload: ChatMessage }
  | { type: 'SET_CHAT_HISTORY'; payload: ChatMessage[] }
  | { type: 'SET_SYSTEM_HEALTH'; payload: DetailedHealthResponse | null }
  | { type: 'RESET_STATE' };

// Reducer
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    
    case 'SET_CURRENT_SESSION':
      return { ...state, currentSession: action.payload };
    
    case 'ADD_DOCUMENT':
      return { ...state, documents: [...state.documents, action.payload] };
    
    case 'SET_DOCUMENTS':
      return { ...state, documents: action.payload };
    
    case 'REMOVE_DOCUMENT':
      return {
        ...state,
        documents: state.documents.filter(doc => doc.document_id !== action.payload)
      };
    
    case 'ADD_CHAT_MESSAGE':
      return { ...state, chatHistory: [...state.chatHistory, action.payload] };
    
    case 'SET_CHAT_HISTORY':
      return { ...state, chatHistory: action.payload };
    
    case 'SET_SYSTEM_HEALTH':
      return { ...state, systemHealth: action.payload };
    
    case 'RESET_STATE':
      return initialState;
    
    default:
      return state;
  }
}

// Context
const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  actions: {
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    uploadDocument: (file: File) => Promise<void>;
    searchDocuments: (query: string) => Promise<any>;
    sendMessage: (message: string, sessionId?: string) => Promise<void>;
    loadDocuments: () => Promise<void>;
    loadSystemHealth: () => Promise<void>;
    createNewSession: () => void;
  };
} | null>(null);

// Provider component
export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Actions
  const actions = {
    setLoading: (loading: boolean) => dispatch({ type: 'SET_LOADING', payload: loading }),
    
    setError: (error: string | null) => dispatch({ type: 'SET_ERROR', payload: error }),
    
    uploadDocument: async (file: File) => {
      try {
        actions.setLoading(true);
        actions.setError(null);
        
        const response = await apiClient.uploadDocument(file);
        
        if (response.success) {
          // Create document object
          const newDocument: Document = {
            document_id: response.document_id,
            filename: response.filename,
            status: 'processing',
            upload_time: Date.now(),
            chunks_count: 0,
            file_size: file.size,
            content_type: file.type,
          };
          
          dispatch({ type: 'ADD_DOCUMENT', payload: newDocument });
        } else {
          throw new Error('Upload failed');
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Upload failed';
        actions.setError(errorMessage);
        console.error('Upload error:', error);
      } finally {
        actions.setLoading(false);
      }
    },
    
    searchDocuments: async (query: string) => {
      try {
        actions.setLoading(true);
        actions.setError(null);
        
        const response = await apiClient.searchDocuments(query);
        return response;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Search failed';
        actions.setError(errorMessage);
        console.error('Search error:', error);
        return null;
      } finally {
        actions.setLoading(false);
      }
    },
    
    sendMessage: async (message: string, sessionId?: string) => {
      try {
        actions.setLoading(true);
        actions.setError(null);
        
        // Create session ID if not provided
        const actualSessionId = sessionId || state.currentSession || `session_${Date.now()}`;
        
        const response = await apiClient.chat({
          message,
          session_id: actualSessionId,
          include_sources: true,
        });
        
        if (response.success) {
          // Create chat message
          const chatMessage: ChatMessage = {
            id: `msg_${Date.now()}`,
            message,
            response: response.response,
            sources: response.sources,
            session_id: response.session_id,
            timestamp: Date.now(),
            generation_time: response.generation_time,
          };
          
          dispatch({ type: 'ADD_CHAT_MESSAGE', payload: chatMessage });
          
          // Set current session if not set
          if (!state.currentSession) {
            dispatch({ type: 'SET_CURRENT_SESSION', payload: response.session_id });
          }
          
          return response;
        } else {
          throw new Error('Chat request failed');
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Chat failed';
        actions.setError(errorMessage);
        console.error('Chat error:', error);
        return null;
      } finally {
        actions.setLoading(false);
      }
    },
    
    loadDocuments: async () => {
      try {
        actions.setLoading(true);
        const response = await apiClient.listDocuments();
        dispatch({ type: 'SET_DOCUMENTS', payload: response.documents });
      } catch (error) {
        console.error('Load documents error:', error);
      } finally {
        actions.setLoading(false);
      }
    },
    
    loadSystemHealth: async () => {
      try {
        const response = await apiClient.getDetailedHealth();
        dispatch({ type: 'SET_SYSTEM_HEALTH', payload: response });
      } catch (error) {
        console.error('Load system health error:', error);
      }
    },
    
    createNewSession: () => {
      const newSessionId = `session_${Date.now()}`;
      dispatch({ type: 'SET_CURRENT_SESSION', payload: newSessionId });
      dispatch({ type: 'SET_CHAT_HISTORY', payload: [] });
    },
  };

  // Load initial data
  useEffect(() => {
    actions.loadDocuments();
    actions.loadSystemHealth();
  }, []);

  // Auto-refresh system health every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      actions.loadSystemHealth();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  return (
    <AppContext.Provider value={{ state, dispatch, actions }}>
      {children}
    </AppContext.Provider>
  );
}

// Hook to use the context
export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}

export default AppContext;
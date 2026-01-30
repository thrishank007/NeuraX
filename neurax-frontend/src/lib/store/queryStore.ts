import { create } from 'zustand';

interface QueryResult {
  results: any[];
  generated_response?: string;
  citations?: any[];
  processing_time?: number;
}

interface QueryStore {
  currentQuery: string;
  isQuerying: boolean;
  results: QueryResult | null;
  history: any[];
  setCurrentQuery: (query: string) => void;
  setIsQuerying: (isQuerying: boolean) => void;
  setResults: (results: QueryResult | null) => void;
  addToHistory: (query: string, results: QueryResult) => void;
}

export const useQueryStore = create<QueryStore>((set) => ({
  currentQuery: '',
  isQuerying: false,
  results: null,
  history: [],
  setCurrentQuery: (currentQuery) => set({ currentQuery }),
  setIsQuerying: (isQuerying) => set({ isQuerying }),
  setResults: (results) => set({ results }),
  addToHistory: (query, results) => 
    set((state) => ({ 
      history: [{ query, results, timestamp: new Date().toISOString() }, ...state.history] 
    })),
}));

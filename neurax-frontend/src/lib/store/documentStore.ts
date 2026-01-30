import { create } from 'zustand';

interface Document {
  id: string;
  name: string;
  type: string;
  size: number;
  status: 'processing' | 'completed' | 'error';
  timestamp: string;
}

interface DocumentStore {
  documents: Document[];
  uploading: boolean;
  setUploading: (uploading: boolean) => void;
  setDocuments: (documents: Document[]) => void;
  addDocument: (doc: Document) => void;
  updateDocumentStatus: (id: string, status: Document['status']) => void;
}

export const useDocumentStore = create<DocumentStore>((set) => ({
  documents: [],
  uploading: false,
  setUploading: (uploading) => set({ uploading }),
  setDocuments: (documents) => set({ documents }),
  addDocument: (doc) => set((state) => ({ documents: [doc, ...state.documents] })),
  updateDocumentStatus: (id, status) => 
    set((state) => ({
      documents: state.documents.map((d) => d.id === id ? { ...d, status } : d)
    })),
}));

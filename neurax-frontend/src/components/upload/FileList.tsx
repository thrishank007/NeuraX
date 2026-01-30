"use client";

import { useDocumentStore } from "@/lib/store/documentStore";
import { 
  FileText, 
  Image as ImageIcon, 
  Music, 
  Trash2, 
  MoreVertical,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2
} from "lucide-react";

export default function FileList() {
  const { documents } = useDocumentStore();

  const getFileIcon = (type: string) => {
    if (type.includes('pdf') || type.includes('word') || type.includes('text')) return <FileText size={18} />;
    if (type.includes('image')) return <ImageIcon size={18} />;
    if (type.includes('audio')) return <Music size={18} />;
    return <FileText size={18} />;
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (documents.length === 0) {
    return (
      <div className="text-center py-20 bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-2xl">
        <div className="inline-flex items-center justify-center p-4 bg-slate-50 dark:bg-slate-800 rounded-full text-slate-400 mb-4">
          <FileText size={32} />
        </div>
        <h3 className="text-lg font-medium text-slate-800 dark:text-slate-100">No documents yet</h3>
        <p className="text-slate-500 dark:text-slate-400">Upload documents to start analyzing them</p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-2xl overflow-hidden shadow-sm">
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="bg-slate-50 dark:bg-slate-800/50 border-b dark:border-slate-800">
            <tr>
              <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500">Name</th>
              <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500">Size</th>
              <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500">Status</th>
              <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500">Added</th>
              <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y dark:divide-slate-800">
            {documents.map((doc) => (
              <tr key={doc.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center">
                    <div className="p-2 bg-slate-100 dark:bg-slate-800 text-slate-500 rounded mr-3">
                      {getFileIcon(doc.type)}
                    </div>
                    <span className="text-sm font-medium text-slate-800 dark:text-slate-100 truncate max-w-xs">
                      {doc.name}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 text-sm text-slate-500 dark:text-slate-400">
                  {formatSize(doc.size)}
                </td>
                <td className="px-6 py-4">
                  {doc.status === 'completed' && (
                    <span className="inline-flex items-center px-2 py-1 bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 text-[10px] font-bold uppercase rounded">
                      <CheckCircle2 size={12} className="mr-1" /> Processed
                    </span>
                  )}
                  {doc.status === 'processing' && (
                    <span className="inline-flex items-center px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 text-[10px] font-bold uppercase rounded">
                      <Loader2 size={12} className="mr-1 animate-spin" /> Processing
                    </span>
                  )}
                  {doc.status === 'error' && (
                    <span className="inline-flex items-center px-2 py-1 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-[10px] font-bold uppercase rounded">
                      <AlertCircle size={12} className="mr-1" /> Error
                    </span>
                  )}
                </td>
                <td className="px-6 py-4 text-sm text-slate-500 dark:text-slate-400">
                  <div className="flex items-center uppercase text-[10px] font-bold">
                    <Clock size={12} className="mr-1" />
                    {new Date(doc.timestamp).toLocaleDateString()}
                  </div>
                </td>
                <td className="px-6 py-4 text-right">
                  <div className="flex items-center justify-end space-x-2">
                    <button className="p-2 text-slate-400 hover:text-red-500 transition-colors rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20">
                      <Trash2 size={18} />
                    </button>
                    <button className="p-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors">
                      <MoreVertical size={18} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, File, CheckCircle2, Loader2 } from "lucide-react";
import { documentApi } from "@/lib/api/client";
import { useDocumentStore } from "@/lib/store/documentStore";

export default function FileUploader() {
  const { addDocument, setUploading, uploading } = useDocumentStore();
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setPendingFiles(prev => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
      'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
      'audio/*': ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    }
  });

  const removeFile = (index: number) => {
    setPendingFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (pendingFiles.length === 0) return;
    
    setUploading(true);
    try {
      const results = await documentApi.upload(pendingFiles);
      
      results.forEach((res: any, idx: number) => {
        if (res.status === 'success') {
          addDocument({
            id: res.id || Math.random().toString(36).substr(2, 9),
            name: res.filename,
            type: pendingFiles[idx].type,
            size: pendingFiles[idx].size,
            status: 'completed',
            timestamp: new Date().toISOString()
          });
        }
      });
      
      setPendingFiles([]);
    } catch (error) {
      console.error("Upload failed", error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div 
        {...getRootProps()} 
        className={`
          border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer
          ${isDragActive 
            ? "border-blue-500 bg-blue-50 dark:bg-blue-900/10" 
            : "border-slate-300 dark:border-slate-800 hover:border-blue-400 dark:hover:border-blue-600"}
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center">
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 text-blue-600 rounded-full mb-4">
            <Upload size={32} />
          </div>
          <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            {isDragActive ? "Drop files here" : "Drag & drop files here"}
          </h3>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Support for PDF, Word, Images, and Audio (max 100MB per file)
          </p>
          <button className="mt-6 px-4 py-2 bg-white dark:bg-slate-800 border dark:border-slate-700 text-sm font-medium rounded-lg shadow-sm hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors">
            Select Files
          </button>
        </div>
      </div>

      {pendingFiles.length > 0 && (
        <div className="bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 flex justify-between items-center">
            <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
              {pendingFiles.length} files selected
            </span>
            <button 
              onClick={handleUpload}
              disabled={uploading}
              className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white text-xs font-bold uppercase tracking-wider rounded-md transition-all flex items-center"
            >
              {uploading ? (
                <>
                  <Loader2 size={14} className="mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                "Upload All"
              )}
            </button>
          </div>
          <ul className="divide-y dark:divide-slate-800">
            {pendingFiles.map((file, idx) => (
              <li key={idx} className="px-4 py-3 flex items-center justify-between">
                <div className="flex items-center">
                  <File size={18} className="text-slate-400 mr-3" />
                  <div>
                    <p className="text-sm font-medium text-slate-800 dark:text-slate-200 truncate max-w-[200px] md:max-w-md">
                      {file.name}
                    </p>
                    <p className="text-[10px] text-slate-500">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button 
                  onClick={() => removeFile(idx)}
                  className="text-slate-400 hover:text-red-500 transition-colors"
                >
                  <X size={18} />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

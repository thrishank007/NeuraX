'use client';

import React, { useState } from 'react';
import { 
  ChatBubbleLeftRightIcon, 
  DocumentTextIcon, 
  MagnifyingGlassIcon,
  ChartBarIcon,
  CloudArrowUpIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '@/hooks/useApp';
import { Document, SearchResult } from '@/types/api';
import { formatDistanceToNow } from 'date-fns';

interface FileUploadProps {
  onUpload: (file: File) => Promise<void>;
  isUploading: boolean;
}

export default function FileUpload({ onUpload, isUploading }: FileUploadProps) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
      'audio/*': ['.wav', '.mp3', '.m4a', '.flac', '.ogg'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    multiple: false,
    onDrop: async (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        await onUpload(acceptedFiles[0]);
      }
    },
  });

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200
          ${isDragActive 
            ? 'border-primary-400 bg-primary-50' 
            : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          }
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
        
        <p className="mt-2 text-sm text-gray-600">
          {isDragActive ? (
            "Drop the file here..."
          ) : (
            <>
              <span className="font-medium text-primary-600">Click to upload</span> or drag and drop
            </>
          )}
        </p>
        
        <p className="mt-1 text-xs text-gray-500">
          PDF, DOC, DOCX, TXT, images, or audio files (max. 100MB)
        </p>
        
        {isUploading && (
          <div className="mt-4 flex items-center justify-center">
            <ArrowPathIcon className="animate-spin h-5 w-5 text-primary-600" />
            <span className="ml-2 text-sm text-gray-600">Uploading...</span>
          </div>
        )}
      </div>
    </div>
  );
}

interface DocumentListProps {
  documents: Document[];
  onDelete?: (documentId: string) => void;
}

export function DocumentList({ documents, onDelete }: DocumentListProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'processed':
        return 'bg-green-100 text-green-800';
      case 'processing':
        return 'bg-yellow-100 text-yellow-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (documents.length === 0) {
    return (
      <div className="text-center py-8">
        <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-sm text-gray-600">No documents uploaded yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {documents.map((document) => (
        <motion.div
          key={document.document_id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow"
        >
          <div className="flex items-center justify-between">
            <div className="flex-1 min-w-0">
              <div className="flex items-center space-x-3">
                <DocumentTextIcon className="h-5 w-5 text-gray-400 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {document.filename}
                  </p>
                  <div className="flex items-center space-x-4 mt-1">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(document.status)}`}>
                      {document.status}
                    </span>
                    <span className="text-xs text-gray-500">
                      {formatDistanceToNow(new Date(document.upload_time), { addSuffix: true })}
                    </span>
                    <span className="text-xs text-gray-500">
                      {formatFileSize(document.file_size)}
                    </span>
                    {document.chunks_count > 0 && (
                      <span className="text-xs text-gray-500">
                        {document.chunks_count} chunks
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
            {onDelete && (
              <button
                onClick={() => onDelete(document.document_id)}
                className="ml-4 text-red-600 hover:text-red-800 transition-colors"
                title="Delete document"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            )}
          </div>
        </motion.div>
      ))}
    </div>
  );
}

interface SearchResultsProps {
  results: SearchResult[];
  query: string;
}

export function SearchResults({ results, query }: SearchResultsProps) {
  if (!query) return null;

  if (results.length === 0) {
    return (
      <div className="text-center py-8">
        <MagnifyingGlassIcon className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-sm text-gray-600">No results found for "{query}"</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-sm text-gray-600 mb-4">
        Found {results.length} result{results.length !== 1 ? 's' : ''} for "{query}"
      </div>
      
      {results.map((result, index) => (
        <motion.div
          key={`${result.chunk_id}-${index}`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow"
        >
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {Math.round(result.similarity * 100)}% match
              </span>
              <span className="text-xs text-gray-500 font-mono">
                {result.source}
              </span>
            </div>
          </div>
          
          <p className="text-sm text-gray-800 leading-relaxed">
            {result.content}
          </p>
        </motion.div>
      ))}
    </div>
  );
}

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
}

export function TabButton({ active, onClick, icon: Icon, children }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center space-x-2 px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200
        ${active 
          ? 'bg-primary-100 text-primary-700 border border-primary-200' 
          : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
        }
      `}
    >
      <Icon className="h-5 w-5" />
      <span>{children}</span>
    </button>
  );
}
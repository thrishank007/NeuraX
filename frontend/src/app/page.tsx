'use client';

import React, { useState } from 'react';
import { AppProvider, useApp } from '@/hooks/useApp';
import Navigation, { DashboardMetrics, SystemHealth } from '@/components/Navigation';
import FileUpload, { DocumentList, SearchResults } from '@/components/FileUpload';
import ChatInterface from '@/components/ChatInterface';
import { motion, AnimatePresence } from 'framer-motion';

export default function HomePage() {
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);

  return (
    <AppProvider>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <Navigation currentTab={currentTab} onTabChange={setCurrentTab} />

        {/* Main Content */}
        <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <AnimatePresence mode="wait">
            {currentTab === 'dashboard' && (
              <motion.div
                key="dashboard"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <DashboardContent />
              </motion.div>
            )}

            {currentTab === 'documents' && (
              <motion.div
                key="documents"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <DocumentsContent />
              </motion.div>
            )}

            {currentTab === 'search' && (
              <motion.div
                key="search"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <SearchContent query={searchQuery} results={searchResults} onQueryChange={setSearchQuery} onResultsChange={setSearchResults} />
              </motion.div>
            )}

            {currentTab === 'chat' && (
              <motion.div
                key="chat"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <ChatContent />
              </motion.div>
            )}

            {currentTab === 'metrics' && (
              <motion.div
                key="metrics"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <MetricsContent />
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        {/* Toast notifications */}
        <Toaster />
      </div>
    </AppProvider>
  );
}

// Dashboard Content Component
function DashboardContent() {
  const { state } = useApp();
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-600">
          Overview of your NeuraX system status and activity
        </p>
      </div>

      {/* Metrics Cards */}
      <DashboardMetrics health={state.systemHealth} isLoading={state.isLoading} />

      {/* System Health and Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SystemHealth health={state.systemHealth} />
        <RecentActivity />
      </div>
    </div>
  );
}

// Documents Content Component
function DocumentsContent() {
  const { state, actions } = useApp();
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Documents</h1>
        <p className="mt-1 text-sm text-gray-600">
          Upload, manage, and organize your documents
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Section */}
        <div className="lg:col-span-1">
          <div className="bg-white shadow rounded-lg p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Upload Documents</h3>
            <FileUpload onUpload={actions.uploadDocument} isUploading={state.isLoading} />
          </div>
        </div>

        {/* Documents List */}
        <div className="lg:col-span-2">
          <div className="bg-white shadow rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Document Library</h3>
              <span className="text-sm text-gray-500">
                {state.documents.length} document{state.documents.length !== 1 ? 's' : ''}
              </span>
            </div>
            <DocumentList documents={state.documents} />
          </div>
        </div>
      </div>
    </div>
  );
}

// Search Content Component
interface SearchContentProps {
  query: string;
  results: any[];
  onQueryChange: (query: string) => void;
  onResultsChange: (results: any[]) => void;
}

function SearchContent({ query, results, onQueryChange, onResultsChange }: SearchContentProps) {
  const { state, actions } = useApp();
  const [localQuery, setLocalQuery] = useState(query);

  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) return;
    
    onQueryChange(searchQuery);
    const response = await actions.searchDocuments(searchQuery);
    if (response) {
      onResultsChange(response.results);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSearch(localQuery);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Search</h1>
        <p className="mt-1 text-sm text-gray-600">
          Search through your uploaded documents
        </p>
      </div>

      {/* Search Form */}
      <div className="bg-white shadow rounded-lg p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="search" className="block text-sm font-medium text-gray-700">
              Search Query
            </label>
            <div className="mt-1 flex space-x-3">
              <input
                type="text"
                id="search"
                value={localQuery}
                onChange={(e) => setLocalQuery(e.target.value)}
                placeholder="Enter your search query..."
                className="flex-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-primary-500 focus:border-primary-500"
              />
              <button
                type="submit"
                disabled={!localQuery.trim() || state.isLoading}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {state.isLoading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>
        </form>
      </div>

      {/* Search Results */}
      {results.length > 0 && (
        <div className="bg-white shadow rounded-lg p-6">
          <SearchResults results={results} query={query} />
        </div>
      )}
    </div>
  );
}

// Chat Content Component
function ChatContent() {
  const { state, actions } = useApp();

  const handleSendMessage = async (message: string) => {
    await actions.sendMessage(message, state.currentSession);
  };

  const handleNewSession = () => {
    actions.createNewSession();
  };

  return (
    <div className="h-[calc(100vh-12rem)]">
      <ChatInterface
        messages={state.chatHistory}
        isLoading={state.isLoading}
        onSendMessage={handleSendMessage}
        onNewSession={handleNewSession}
      />
    </div>
  );
}

// Metrics Content Component
function MetricsContent() {
  const { state } = useApp();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Metrics</h1>
        <p className="mt-1 text-sm text-gray-600">
          System performance and evaluation metrics
        </p>
      </div>

      {/* Coming Soon */}
      <div className="bg-white shadow rounded-lg p-12 text-center">
        <div className="w-16 h-16 mx-auto bg-gray-100 rounded-full flex items-center justify-center mb-4">
          <svg className="w-8 h-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">Metrics Dashboard</h3>
        <p className="text-gray-600">
          Detailed performance metrics and evaluation data will be available here.
          This feature is currently under development.
        </p>
      </div>
    </div>
  );
}

// Recent Activity Component
function RecentActivity() {
  const { state } = useApp();

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h3>
      <div className="space-y-3">
        {state.chatHistory.slice(-3).reverse().map((message) => (
          <div key={message.id} className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                <span className="text-primary-600 font-semibold text-sm">AI</span>
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-900 truncate">
                {message.message}
              </p>
              <p className="text-xs text-gray-500">
                {new Date(message.timestamp).toLocaleString()}
              </p>
            </div>
          </div>
        ))}
        
        {state.chatHistory.length === 0 && (
          <p className="text-sm text-gray-500 text-center py-4">
            No recent activity
          </p>
        )}
      </div>
    </div>
  );
}
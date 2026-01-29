'use client'

import { useState, useCallback } from 'react'
import { Header } from '@/components/common/Header'
import { FileUploader } from '@/components/upload/FileUploader'
import { QueryInterface } from '@/components/query/QueryInterface'
import { ResultsDisplay } from '@/components/results/ResultsDisplay'
import { AnalyticsDashboard } from '@/components/analytics/AnalyticsDashboard'
import { QueryHistory } from '@/components/query/QueryHistory'
import { Settings } from '@/components/common/Settings'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { FileText, Search, BarChart3, History, Settings as SettingsIcon, Upload } from 'lucide-react'
import type { Query, SearchResult, FileUpload } from '@/types'

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<'query' | 'upload' | 'analytics' | 'history' | 'settings'>('query')
  const [currentQuery, setCurrentQuery] = useState<Query | null>(null)
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<FileUpload[]>([])

  const handleSearch = useCallback((query: Query) => {
    setCurrentQuery(query)
    // Search results will be set by the QueryInterface component
  }, [])

  const handleSearchResults = useCallback((results: SearchResult[]) => {
    setSearchResults(results)
  }, [])

  const handleFileUpload = useCallback((files: FileUpload[]) => {
    setUploadedFiles(prev => [...prev, ...files])
  }, [])

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'upload':
        return (
          <FileUploader
            onFileUpload={handleFileUpload}
            uploadedFiles={uploadedFiles}
          />
        )
      case 'analytics':
        return <AnalyticsDashboard />
      case 'history':
        return <QueryHistory />
      case 'settings':
        return <Settings />
      case 'query':
      default:
        return (
          <div className="space-y-6">
            <QueryInterface
              onSearch={handleSearch}
              onResults={handleSearchResults}
              uploadedFiles={uploadedFiles}
            />
            {searchResults.length > 0 && (
              <ResultsDisplay
                query={currentQuery}
                results={searchResults}
              />
            )}
          </div>
        )
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Header
        onSearch={() => setActiveTab('query')}
        onUpload={() => setActiveTab('upload')}
        onAnalytics={() => setActiveTab('analytics')}
        onHistory={() => setActiveTab('history')}
        onSettings={() => setActiveTab('settings')}
      />

      <main className="container mx-auto px-4 py-6">
        {/* Tab Navigation */}
        <div className="mb-6">
          <div className="flex space-x-1 rounded-lg bg-muted p-1">
            <Button
              variant={activeTab === 'query' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab('query')}
              className="gap-2"
            >
              <Search className="h-4 w-4" />
              Query & Search
            </Button>
            <Button
              variant={activeTab === 'upload' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab('upload')}
              className="gap-2"
            >
              <Upload className="h-4 w-4" />
              File Upload
            </Button>
            <Button
              variant={activeTab === 'analytics' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab('analytics')}
              className="gap-2"
            >
              <BarChart3 className="h-4 w-4" />
              Analytics
            </Button>
            <Button
              variant={activeTab === 'history' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab('history')}
              className="gap-2"
            >
              <History className="h-4 w-4" />
              History
            </Button>
            <Button
              variant={activeTab === 'settings' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab('settings')}
              className="gap-2"
            >
              <SettingsIcon className="h-4 w-4" />
              Settings
            </Button>
          </div>
        </div>

        {/* Tab Content */}
        <div className="animate-fade-in">
          {renderActiveTab()}
        </div>

        {/* Quick Stats Footer */}
        {activeTab === 'query' && (
          <Card className="mt-8">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-1">
                    <div className="h-2 w-2 bg-green-500 rounded-full" />
                    <span>{uploadedFiles.length} files uploaded</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="h-2 w-2 bg-blue-500 rounded-full" />
                    <span>{searchResults.length} results found</span>
                  </div>
                </div>
                <div>
                  System Status: Online
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  )
}
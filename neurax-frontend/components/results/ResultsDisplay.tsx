'use client'

import { useState, useCallback } from 'react'
import { 
  FileText, 
  Image, 
  Music, 
  Copy, 
  Download, 
  ExternalLink, 
  Star,
  StarOff,
  ThumbsUp,
  ThumbsDown,
  MessageSquare,
  Eye,
  Clock,
  Target
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn, formatBytes, getFileType, getFileIcon, formatSimilarityScore, copyToClipboard } from '@/lib/utils'
import { apiClient } from '@/lib/api/client'
import toast from 'react-hot-toast'
import type { Query, SearchResult } from '@/types'

interface ResultsDisplayProps {
  query: Query | null
  results: SearchResult[]
  onFeedback?: (resultId: string, feedback: any) => void
}

export function ResultsDisplay({ query, results, onFeedback }: ResultsDisplayProps) {
  const [favorites, setFavorites] = useState<Set<string>>(new Set())
  const [feedback, setFeedback] = useState<Record<string, 'positive' | 'negative' | null>>({})

  const toggleFavorite = useCallback((resultId: string) => {
    setFavorites(prev => {
      const newSet = new Set(prev)
      if (newSet.has(resultId)) {
        newSet.delete(resultId)
        toast.success('Removed from favorites')
      } else {
        newSet.add(resultId)
        toast.success('Added to favorites')
      }
      return newSet
    })
  }, [])

  const handleFeedback = useCallback((resultId: string, type: 'positive' | 'negative') => {
    setFeedback(prev => ({ ...prev, [resultId]: type }))
    onFeedback?.(resultId, { type, timestamp: new Date().toISOString() })
    
    toast.success(type === 'positive' ? 'Thanks for your feedback!' : 'Feedback recorded')
  }, [onFeedback])

  const copyToClipboardHandler = useCallback(async (text: string) => {
    try {
      await copyToClipboard(text)
      toast.success('Copied to clipboard')
    } catch (error) {
      toast.error('Failed to copy')
    }
  }, [])

  const exportResults = useCallback(() => {
    const exportData = {
      query: query?.text,
      timestamp: new Date().toISOString(),
      results: results.map(result => ({
        fileName: result.fileName,
        fileType: result.fileType,
        similarityScore: result.similarityScore,
        contentPreview: result.contentPreview,
        metadata: result.metadata
      }))
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `neurax-results-${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    toast.success('Results exported')
  }, [query, results])

  if (!results || results.length === 0) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="text-center py-12">
            <Target className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Results Found</h3>
            <p className="text-muted-foreground">
              Try adjusting your query or similarity threshold to find more results.
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Results Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Search Results
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                {query?.text && `Query: "${query.text}"`}
                {query?.type && ` • Type: ${query.type}`}
                {query?.processingTime && ` • Time: ${query.processingTime.toFixed(2)}s`}
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="secondary">
                {results.length} result{results.length !== 1 ? 's' : ''}
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={exportResults}
                className="gap-2"
              >
                <Download className="h-4 w-4" />
                Export
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Results List */}
      <div className="space-y-4">
        {results.map((result, index) => (
          <ResultCard
            key={result.id}
            result={result}
            index={index}
            isFavorite={favorites.has(result.id)}
            feedback={feedback[result.id]}
            onToggleFavorite={() => toggleFavorite(result.id)}
            onFeedback={(type) => handleFeedback(result.id, type)}
            onCopy={() => copyToClipboardHandler(result.contentPreview)}
          />
        ))}
      </div>

      {/* Export Options */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <h3 className="font-semibold">Export Options</h3>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  // Export as CSV
                  const csvContent = [
                    ['File Name', 'File Type', 'Similarity Score', 'Content Preview'],
                    ...results.map(r => [r.fileName, r.fileType, r.similarityScore.toString(), r.contentPreview])
                  ].map(row => row.map(cell => `"${cell}"`).join(',')).join('\n')
                  
                  const blob = new Blob([csvContent], { type: 'text/csv' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `neurax-results-${Date.now()}.csv`
                  document.body.appendChild(a)
                  a.click()
                  document.body.removeChild(a)
                  URL.revokeObjectURL(url)
                  toast.success('Results exported as CSV')
                }}
              >
                Export as CSV
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  // Export as PDF would require a library, for now export as JSON
                  exportResults()
                }}
              >
                Export as JSON
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

interface ResultCardProps {
  result: SearchResult
  index: number
  isFavorite: boolean
  feedback: 'positive' | 'negative' | null
  onToggleFavorite: () => void
  onFeedback: (type: 'positive' | 'negative') => void
  onCopy: () => void
}

function ResultCard({ 
  result, 
  index, 
  isFavorite, 
  feedback, 
  onToggleFavorite, 
  onFeedback, 
  onCopy 
}: ResultCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const similarityInfo = formatSimilarityScore(result.similarityScore)
  const fileTypeInfo = getFileType(result.fileName)

  const getFileTypeIcon = () => {
    switch (fileTypeInfo) {
      case 'document':
        return <FileText className="h-4 w-4" />
      case 'image':
        return <Image className="h-4 w-4" />
      case 'audio':
        return <Music className="h-4 w-4" />
      default:
        return <FileText className="h-4 w-4" />
    }
  }

  const getSimilarityBadgeVariant = () => {
    if (result.similarityScore >= 0.8) return 'default' // Green
    if (result.similarityScore >= 0.6) return 'secondary' // Yellow
    return 'destructive' // Red
  }

  return (
    <Card className="transition-all hover:shadow-md">
      <CardContent className="pt-6">
        <div className="space-y-4">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3 min-w-0 flex-1">
              <div className="flex-shrink-0 mt-1">
                {getFileTypeIcon()}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center space-x-2">
                  <h3 className="font-semibold truncate">{result.fileName}</h3>
                  <Badge variant={getSimilarityBadgeVariant()} className="text-xs">
                    {result.similarityScore.toFixed(3)}
                  </Badge>
                </div>
                <div className="flex items-center space-x-4 mt-1 text-sm text-muted-foreground">
                  <span className="flex items-center space-x-1">
                    <span>{getFileIcon(result.fileName)}</span>
                    <span>{fileTypeInfo}</span>
                  </span>
                  {result.pageNumber && (
                    <span>Page {result.pageNumber}</span>
                  )}
                  <span className="flex items-center space-x-1">
                    <Clock className="h-3 w-3" />
                    <span>{new Date(result.timestamp).toLocaleDateString()}</span>
                  </span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={onToggleFavorite}
                className={cn("h-8 w-8", isFavorite && "text-yellow-500")}
              >
                {isFavorite ? <Star className="h-4 w-4 fill-current" /> : <StarOff className="h-4 w-4" />}
              </Button>
            </div>
          </div>

          {/* Content Preview */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Content Preview</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsExpanded(!isExpanded)}
                className="gap-1"
              >
                <Eye className="h-3 w-3" />
                {isExpanded ? 'Show Less' : 'Show More'}
              </Button>
            </div>
            <div className={cn(
              "text-sm bg-muted/50 rounded-lg p-3 font-mono text-xs overflow-hidden",
              isExpanded ? "max-h-none" : "max-h-24"
            )}>
              {result.contentPreview}
              {!isExpanded && result.contentPreview.length > 200 && (
                <div className="mt-2 text-center">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsExpanded(true)}
                    className="text-xs"
                  >
                    Read more...
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* Metadata */}
          {result.metadata && Object.keys(result.metadata).length > 0 && (
            <div className="space-y-2">
              <span className="text-sm font-medium">Metadata</span>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                {Object.entries(result.metadata).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-muted-foreground">{key}:</span>
                    <span>{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between pt-2 border-t">
            <div className="flex items-center space-x-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={onFeedback}
                className={cn("gap-1", feedback === 'positive' && "text-green-600")}
                disabled={feedback !== null}
              >
                <ThumbsUp className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onFeedback('negative')}
                className={cn("gap-1", feedback === 'negative' && "text-red-600")}
                disabled={feedback !== null}
              >
                <ThumbsDown className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={onCopy}
                className="gap-1"
              >
                <Copy className="h-3 w-3" />
                Copy
              </Button>
            </div>
            
            <div className="flex items-center space-x-2">
              {result.confidence && (
                <Badge variant="outline" className="text-xs">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </Badge>
              )}
              <Button
                variant="ghost"
                size="sm"
                className="gap-1"
                onClick={() => {
                  // Open file preview (would need file viewer implementation)
                  toast.info('File preview feature coming soon')
                }}
              >
                <ExternalLink className="h-3 w-3" />
                Preview
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
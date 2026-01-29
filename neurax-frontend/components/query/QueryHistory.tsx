'use client'

import { useState, useEffect } from 'react'
import { 
  History, 
  Search, 
  FileText, 
  Image, 
  Music, 
  Clock, 
  Trash2, 
  Eye, 
  Play,
  RotateCcw,
  Star,
  StarOff
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import { apiClient } from '@/lib/api/client'
import toast from 'react-hot-toast'
import type { Query } from '@/types'

interface QueryHistoryProps {
  onQuerySelect?: (query: Query) => void
  onSearchFromHistory?: (queryText: string) => void
}

export function QueryHistory({ onQuerySelect, onSearchFromHistory }: QueryHistoryProps) {
  const [queries, setQueries] = useState<Query[]>([])
  const [filteredQueries, setFilteredQueries] = useState<Query[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchFilter, setSearchFilter] = useState('')
  const [typeFilter, setTypeFilter] = useState<'all' | 'text' | 'image' | 'voice' | 'multimodal'>('all')
  const [sortBy, setSortBy] = useState<'date' | 'relevance' | 'popularity'>('date')

  useEffect(() => {
    loadQueryHistory()
  }, [])

  useEffect(() => {
    filterAndSortQueries()
  }, [queries, searchFilter, typeFilter, sortBy])

  const loadQueryHistory = async () => {
    try {
      setIsLoading(true)
      const response = await apiClient.getQueryHistory(1, 100)
      if (response.success && response.data) {
        // Transform backend data to our Query type
        const transformedQueries: Query[] = response.data.map((item: any) => ({
          id: item.id || `query_${Date.now()}`,
          type: item.type || 'text',
          text: item.query || item.text,
          timestamp: item.timestamp || item.created_at,
          similarityThreshold: item.similarity_threshold || 0.5,
          status: item.status || 'completed',
          processingTime: item.processing_time || 0,
          results: item.results || []
        }))
        setQueries(transformedQueries)
      }
    } catch (error) {
      console.error('Failed to load query history:', error)
      toast.error('Failed to load query history')
      // Use localStorage as fallback
      const localHistory = localStorage.getItem('neurax_query_history')
      if (localHistory) {
        try {
          const localQueries = JSON.parse(localHistory)
          setQueries(localQueries)
        } catch (e) {
          console.error('Failed to parse local query history:', e)
        }
      }
    } finally {
      setIsLoading(false)
    }
  }

  const filterAndSortQueries = () => {
    let filtered = [...queries]

    // Apply search filter
    if (searchFilter) {
      filtered = filtered.filter(query => 
        query.text?.toLowerCase().includes(searchFilter.toLowerCase()) ||
        query.type.toLowerCase().includes(searchFilter.toLowerCase())
      )
    }

    // Apply type filter
    if (typeFilter !== 'all') {
      filtered = filtered.filter(query => query.type === typeFilter)
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'date':
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        case 'relevance':
          const avgScoreA = a.results?.reduce((sum, r) => sum + r.similarityScore, 0) / (a.results?.length || 1) || 0
          const avgScoreB = b.results?.reduce((sum, r) => sum + r.similarityScore, 0) / (b.results?.length || 1) || 0
          return avgScoreB - avgScoreA
        case 'popularity':
          // Sort by number of results (more results = more popular)
          return (b.results?.length || 0) - (a.results?.length || 0)
        default:
          return 0
      }
    })

    setFilteredQueries(filtered)
  }

  const clearHistory = async () => {
    try {
      // Clear from backend (if API exists)
      // await apiClient.clearQueryHistory()
      
      // Clear from localStorage
      localStorage.removeItem('neurax_query_history')
      setQueries([])
      setFilteredQueries([])
      toast.success('Query history cleared')
    } catch (error) {
      console.error('Failed to clear history:', error)
      toast.error('Failed to clear history')
    }
  }

  const deleteQuery = (queryId: string) => {
    setQueries(prev => prev.filter(q => q.id !== queryId))
    setFilteredQueries(prev => prev.filter(q => q.id !== queryId))
    
    // Update localStorage
    const updatedQueries = queries.filter(q => q.id !== queryId)
    localStorage.setItem('neurax_query_history', JSON.stringify(updatedQueries))
    
    toast.success('Query removed from history')
  }

  const runQueryAgain = (query: Query) => {
    if (onSearchFromHistory && query.text) {
      onSearchFromHistory(query.text)
    } else if (onQuerySelect) {
      onQuerySelect(query)
    }
  }

  const getQueryTypeIcon = (type: string) => {
    switch (type) {
      case 'text':
        return <FileText className="h-4 w-4" />
      case 'image':
        return <Image className="h-4 w-4" />
      case 'voice':
        return <Music className="h-4 w-4" />
      case 'multimodal':
        return <Search className="h-4 w-4" />
      default:
        return <Search className="h-4 w-4" />
    }
  }

  const getQueryTypeBadgeVariant = (type: string) => {
    switch (type) {
      case 'text':
        return 'default'
      case 'image':
        return 'secondary'
      case 'voice':
        return 'destructive'
      case 'multimodal':
        return 'outline'
      default:
        return 'outline'
    }
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                <div className="h-3 bg-muted rounded w-1/2" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5" />
              Query History
            </CardTitle>
            <Button
              variant="outline"
              size="sm"
              onClick={clearHistory}
              className="gap-2"
            >
              <Trash2 className="h-4 w-4" />
              Clear All
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {/* Filters */}
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <Input
                  placeholder="Search queries..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  className="w-full"
                />
              </div>
              <div className="flex gap-2">
                <select
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value as any)}
                  className="px-3 py-2 border rounded-md text-sm"
                >
                  <option value="all">All Types</option>
                  <option value="text">Text</option>
                  <option value="image">Image</option>
                  <option value="voice">Voice</option>
                  <option value="multimodal">Multimodal</option>
                </select>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  className="px-3 py-2 border rounded-md text-sm"
                >
                  <option value="date">Latest First</option>
                  <option value="relevance">Most Relevant</option>
                  <option value="popularity">Most Results</option>
                </select>
              </div>
            </div>

            {/* Results Count */}
            <div className="text-sm text-muted-foreground">
              Showing {filteredQueries.length} of {queries.length} queries
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Query List */}
      {filteredQueries.length === 0 ? (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center py-12">
              <History className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Query History</h3>
              <p className="text-muted-foreground mb-4">
                {queries.length === 0 
                  ? "You haven't made any queries yet. Start searching to build your history."
                  : "No queries match your current filters."
                }
              </p>
              {(searchFilter || typeFilter !== 'all') && (
                <Button
                  variant="outline"
                  onClick={() => {
                    setSearchFilter('')
                    setTypeFilter('all')
                  }}
                >
                  Clear Filters
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {filteredQueries.map((query) => (
            <QueryHistoryCard
              key={query.id}
              query={query}
              onRunAgain={() => runQueryAgain(query)}
              onDelete={() => deleteQuery(query.id)}
              onViewDetails={() => onQuerySelect?.(query)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

interface QueryHistoryCardProps {
  query: Query
  onRunAgain: () => void
  onDelete: () => void
  onViewDetails: () => void
}

function QueryHistoryCard({ query, onRunAgain, onDelete, onViewDetails }: QueryHistoryCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [isFavorite, setIsFavorite] = useState(false)

  const getQueryTypeIcon = (type: string) => {
    switch (type) {
      case 'text':
        return <FileText className="h-4 w-4" />
      case 'image':
        return <Image className="h-4 w-4" />
      case 'voice':
        return <Music className="h-4 w-4" />
      case 'multimodal':
        return <Search className="h-4 w-4" />
      default:
        return <Search className="h-4 w-4" />
    }
  }

  const getQueryTypeBadgeVariant = (type: string) => {
    switch (type) {
      case 'text':
        return 'default'
      case 'image':
        return 'secondary'
      case 'voice':
        return 'destructive'
      case 'multimodal':
        return 'outline'
      default:
        return 'outline'
    }
  }

  return (
    <Card className="transition-all hover:shadow-md">
      <CardContent className="pt-6">
        <div className="space-y-4">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3 min-w-0 flex-1">
              <div className="flex-shrink-0 mt-1">
                {getQueryTypeIcon(query.type)}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  <Badge variant={getQueryTypeBadgeVariant(query.type)} className="text-xs">
                    {query.type}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {new Date(query.timestamp).toLocaleDateString()}
                  </span>
                </div>
                <h3 className="font-medium truncate">
                  {query.text || `${query.type} query`}
                </h3>
                <div className="flex items-center space-x-4 mt-1 text-sm text-muted-foreground">
                  <span className="flex items-center space-x-1">
                    <Clock className="h-3 w-3" />
                    <span>{query.processingTime?.toFixed(2) || '0.00'}s</span>
                  </span>
                  {query.results && (
                    <span>{query.results.length} results</span>
                  )}
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsFavorite(!isFavorite)}
                className="h-8 w-8"
              >
                {isFavorite ? (
                  <Star className="h-4 w-4 fill-current text-yellow-500" />
                ) : (
                  <StarOff className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>

          {/* Query Details (expandable) */}
          {query.text && query.text.length > 100 && (
            <div className="space-y-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsExpanded(!isExpanded)}
                className="gap-1"
              >
                <Eye className="h-3 w-3" />
                {isExpanded ? 'Show Less' : 'Show More'}
              </Button>
              {isExpanded && (
                <div className="text-sm bg-muted/50 rounded-lg p-3">
                  {query.text}
                </div>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between pt-2 border-t">
            <div className="flex items-center space-x-1">
              <Button
                variant="outline"
                size="sm"
                onClick={onRunAgain}
                className="gap-1"
              >
                <Play className="h-3 w-3" />
                Run Again
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={onViewDetails}
                className="gap-1"
              >
                <Eye className="h-3 w-3" />
                Details
              </Button>
            </div>
            
            <Button
              variant="ghost"
              size="icon"
              onClick={onDelete}
              className="h-8 w-8 text-muted-foreground hover:text-destructive"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Citation } from "@/components/results/Citation"
import { DocumentPreview } from "@/components/results/DocumentPreview"
import { Copy, Download, ThumbsUp, ThumbsDown } from "lucide-react"
import type { QueryResponse } from "@/lib/types/api"
import toast from "react-hot-toast"
import { feedbackApi } from "@/lib/api/feedback"

interface ResultsDisplayProps {
  result: QueryResponse
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  const [expandedCitation, setExpandedCitation] = useState<number | null>(null)
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null)

  const handleCopyResponse = () => {
    if (result.response_text) {
      navigator.clipboard.writeText(result.response_text)
      toast.success("Response copied to clipboard")
    }
  }

  const handleExport = (format: "json" | "csv") => {
    // Export functionality
    toast.info(`Export as ${format.toUpperCase()} coming soon`)
  }

  const handleFeedback = async (rating: number) => {
    if (!result.response_text) return

    try {
      await feedbackApi.submitFeedback({
        query: result.query,
        response: result.response_text,
        rating,
      })
      toast.success("Thank you for your feedback!")
    } catch (error: any) {
      toast.error("Failed to submit feedback")
    }
  }

  return (
    <div className="space-y-6">
      {/* Response Section */}
      {result.response_text && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Response</CardTitle>
              <div className="flex gap-2">
                <Button variant="outline" size="icon" onClick={handleCopyResponse}>
                  <Copy className="h-4 w-4" />
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => handleExport("json")}
                >
                  <Download className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="prose dark:prose-invert max-w-none">
              <p className="whitespace-pre-wrap">{result.response_text}</p>
            </div>

            {/* Citations */}
            {result.citations.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-semibold">Citations</h4>
                <div className="flex flex-wrap gap-2">
                  {result.citations.map((citation) => (
                    <Citation
                      key={citation.citation_id}
                      citation={citation}
                      isExpanded={expandedCitation === citation.citation_id}
                      onToggle={() =>
                        setExpandedCitation(
                          expandedCitation === citation.citation_id
                            ? null
                            : citation.citation_id
                        )
                      }
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Feedback */}
            <div className="flex items-center gap-2 pt-4 border-t">
              <span className="text-sm text-muted-foreground">Was this helpful?</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => handleFeedback(5)}
              >
                <ThumbsUp className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => handleFeedback(1)}
              >
                <ThumbsDown className="h-4 w-4" />
              </Button>
            </div>

            {/* Metadata */}
            <div className="flex items-center gap-4 text-sm text-muted-foreground pt-2">
              <span>Processing time: {result.processing_time.toFixed(2)}s</span>
              {result.model_used && (
                <span>Model: {result.model_used}</span>
              )}
              <span>Results: {result.total_results}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Search Results */}
      {result.results.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Search Results ({result.results.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {result.results.map((item, index) => (
                <Card
                  key={item.document_id}
                  className="cursor-pointer hover:bg-accent transition-colors"
                  onClick={() => setSelectedDocument(item.document_id)}
                >
                  <CardContent className="pt-6">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{item.file_type}</Badge>
                        <span className="text-sm font-medium">
                          Similarity: {(item.similarity_score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      {item.file_path}
                    </p>
                    <p className="text-sm line-clamp-3">{item.content_preview}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Document Preview Modal */}
      {selectedDocument && (
        <DocumentPreview
          documentId={selectedDocument}
          onClose={() => setSelectedDocument(null)}
        />
      )}
    </div>
  )
}

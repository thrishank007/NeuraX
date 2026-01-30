"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react"
import type { Citation as CitationType } from "@/lib/types/api"

interface CitationProps {
  citation: CitationType
  isExpanded: boolean
  onToggle: () => void
}

export function Citation({ citation, isExpanded, onToggle }: CitationProps) {
  return (
    <div className="border rounded-lg p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge variant="secondary">#{citation.citation_id}</Badge>
          <span className="text-sm font-medium">{citation.source_document}</span>
        </div>
        <Button variant="ghost" size="icon" onClick={onToggle}>
          {isExpanded ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </Button>
      </div>
      {isExpanded && (
        <div className="space-y-2 pt-2 border-t">
          <p className="text-sm text-muted-foreground">{citation.content_snippet}</p>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Confidence: {(citation.confidence_score * 100).toFixed(1)}%</span>
            <Button variant="link" size="sm" className="h-auto p-0">
              <ExternalLink className="h-3 w-3 mr-1" />
              View Source
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

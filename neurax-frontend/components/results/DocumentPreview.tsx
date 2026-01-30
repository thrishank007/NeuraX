"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { X } from "lucide-react"

interface DocumentPreviewProps {
  documentId: string
  onClose: () => void
}

export function DocumentPreview({ documentId, onClose }: DocumentPreviewProps) {
  // In production, fetch document content by ID
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <Card className="w-full max-w-4xl max-h-[90vh] overflow-auto">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Document Preview</CardTitle>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Document preview for {documentId} - Implementation coming soon
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

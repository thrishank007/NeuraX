"use client"

import { FileUploader } from "@/components/upload/FileUploader"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function DocumentsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Document Management</h1>
        <p className="text-muted-foreground">
          Upload and manage your documents for indexing
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Upload Documents</CardTitle>
        </CardHeader>
        <CardContent>
          <FileUploader />
        </CardContent>
      </Card>
    </div>
  )
}

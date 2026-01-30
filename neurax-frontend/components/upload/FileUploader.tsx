"use client"

import { useCallback, useState } from "react"
import { useDropzone } from "react-dropzone"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Upload, X, File as FileIcon, Loader2 } from "lucide-react"
import { documentsApi } from "@/lib/api/documents"
import toast from "react-hot-toast"
import type { FileUploadResponse } from "@/lib/types/api"

const MAX_FILE_SIZE = parseInt(
  process.env.NEXT_PUBLIC_MAX_FILE_SIZE || "104857600"
)
const ALLOWED_TYPES = [
  ".pdf",
  ".docx",
  ".doc",
  ".txt",
  ".jpg",
  ".jpeg",
  ".png",
  ".bmp",
  ".tiff",
  ".webp",
  ".wav",
  ".mp3",
  ".m4a",
  ".flac",
  ".ogg",
]

interface FileUploaderProps {
  onUploadComplete?: (results: FileUploadResponse[]) => void
}

export function FileUploader({ onUploadComplete }: FileUploaderProps) {
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadResults, setUploadResults] = useState<FileUploadResponse[]>([])

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const validFiles = acceptedFiles.filter((file) => {
      const ext = `.${file.name.split(".").pop()?.toLowerCase()}`
      if (!ALLOWED_TYPES.includes(ext)) {
        toast.error(`File type ${ext} not supported`)
        return false
      }
      if (file.size > MAX_FILE_SIZE) {
        toast.error(`File ${file.name} exceeds maximum size`)
        return false
      }
      return true
    })

    setFiles((prev) => [...prev, ...validFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
        ".docx",
      ],
      "application/msword": [".doc"],
      "text/plain": [".txt"],
      "image/*": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
      "audio/*": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
    },
    maxSize: MAX_FILE_SIZE,
    multiple: true,
  })

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleUpload = async () => {
    if (files.length === 0) {
      toast.error("Please select files to upload")
      return
    }

    setUploading(true)
    setUploadProgress(0)

    try {
      // Show initial progress
      setUploadProgress(5)
      
      const results = await documentsApi.uploadFiles(files, (progress) => {
        // Clamp progress between 5% and 95% to account for backend processing
        const clampedProgress = Math.min(95, Math.max(5, progress))
        setUploadProgress(clampedProgress)
      })

      // Complete progress
      setUploadProgress(100)
      
      setUploadResults(results)
      
      // Check if all uploads succeeded
      const successCount = results.filter(r => r.status === "success").length
      const errorCount = results.filter(r => r.status === "error").length
      
      if (successCount > 0 && errorCount === 0) {
        toast.success(`Successfully uploaded ${successCount} file(s)`)
      } else if (successCount > 0 && errorCount > 0) {
        toast.success(`Uploaded ${successCount} file(s), ${errorCount} failed`)
      } else {
        toast.error(`Failed to upload ${errorCount} file(s)`)
      }
      
      if (onUploadComplete) {
        onUploadComplete(results)
      }

      // Clear files after upload (success or failure)
      setFiles([])
    } catch (error: any) {
      console.error("Upload error:", error)
      toast.error(error.message || "Failed to upload files. Check console for details.")
    } finally {
      setUploading(false)
      // Keep progress at 100 if successful, reset if error
      if (uploadResults.length === 0) {
        setUploadProgress(0)
      }
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="pt-6">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? "border-primary bg-primary/5"
                : "border-muted-foreground/25 hover:border-primary/50"
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            {isDragActive ? (
              <p className="text-lg font-medium">Drop files here...</p>
            ) : (
              <>
                <p className="text-lg font-medium mb-2">
                  Drag & drop files here, or click to select
                </p>
                <p className="text-sm text-muted-foreground">
                  Supported: PDF, DOCX, TXT, Images, Audio (Max {MAX_FILE_SIZE / 1024 / 1024}MB)
                </p>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {files.length > 0 && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">Selected Files ({files.length})</h3>
                <Button
                  onClick={handleUpload}
                  disabled={uploading}
                >
                  {uploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="mr-2 h-4 w-4" />
                      Upload Files
                    </>
                  )}
                </Button>
              </div>

              {uploading && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span>Uploading...</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}

              <div className="space-y-2 max-h-60 overflow-y-auto">
                {files.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-2 border rounded"
                  >
                    <div className="flex items-center gap-2">
                      <FileIcon className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm">{file.name}</span>
                      <span className="text-xs text-muted-foreground">
                        ({(file.size / 1024 / 1024).toFixed(2)} MB)
                      </span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => removeFile(index)}
                      disabled={uploading}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {uploadResults.length > 0 && (
        <Card>
          <CardContent className="pt-6">
            <h3 className="font-semibold mb-4">Upload Results</h3>
            <div className="space-y-2">
              {uploadResults.map((result, index) => (
                <div
                  key={index}
                  className={`p-3 border rounded ${
                    result.status === "success"
                      ? "border-green-500 bg-green-50 dark:bg-green-950"
                      : "border-red-500 bg-red-50 dark:bg-red-950"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{result.filename}</span>
                    <span
                      className={`text-xs ${
                        result.status === "success" ? "text-green-700" : "text-red-700"
                      }`}
                    >
                      {result.status}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {result.message}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

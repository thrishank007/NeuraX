'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, X, File, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { cn, formatBytes, getFileType, getFileIcon } from '@/lib/utils'
import { apiClient } from '@/lib/api/client'
import toast from 'react-hot-toast'
import type { FileUpload } from '@/types'

interface FileUploaderProps {
  onFileUpload: (files: FileUpload[]) => void
  uploadedFiles: FileUpload[]
}

export function FileUploader({ onFileUpload, uploadedFiles }: FileUploaderProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [dragActive, setDragActive] = useState(false)

  const allowedTypes = process.env.NEXT_PUBLIC_ALLOWED_FILE_TYPES?.split(',') || [
    '.pdf', '.docx', '.doc', '.txt', '.jpg', '.png', '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.bmp', '.tiff', '.webp'
  ]
  
  const maxFileSize = parseInt(process.env.NEXT_PUBLIC_MAX_FILE_SIZE || '104857600') // 100MB

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setIsUploading(true)
    setUploadProgress(0)

    try {
      // Validate files
      const validFiles = acceptedFiles.filter(file => {
        const fileType = getFileType(file.name)
        if (fileType === 'unknown') {
          toast.error(`Unsupported file type: ${file.name}`)
          return false
        }
        if (file.size > maxFileSize) {
          toast.error(`File too large: ${file.name} (${formatBytes(file.size)})`)
          return false
        }
        return true
      })

      if (validFiles.length === 0) {
        toast.error('No valid files to upload')
        setIsUploading(false)
        return
      }

      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90))
      }, 200)

      try {
        const uploadResponse = await apiClient.uploadFiles(validFiles)
        setUploadProgress(100)
        
        if (uploadResponse.success) {
          onFileUpload(uploadResponse.files)
          toast.success(`Successfully uploaded ${uploadResponse.totalFiles} files`)
        } else {
          toast.error('Upload failed')
        }
      } catch (error) {
        console.error('Upload error:', error)
        toast.error('Upload failed: ' + (error instanceof Error ? error.message : 'Unknown error'))
      } finally {
        clearInterval(progressInterval)
        setTimeout(() => {
          setIsUploading(false)
          setUploadProgress(0)
        }, 1000)
      }
    } catch (error) {
      console.error('Drop error:', error)
      setIsUploading(false)
      setUploadProgress(0)
    }
  }, [maxFileSize, onFileUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: allowedTypes.reduce((acc, type) => {
      acc[type] = []
      return acc
    }, {} as Record<string, string[]>),
    maxSize: maxFileSize,
    multiple: true,
    disabled: isUploading
  })

  const removeFile = async (fileId: string) => {
    try {
      await apiClient.deleteFile(fileId)
      toast.success('File removed')
    } catch (error) {
      toast.error('Failed to remove file')
    }
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            File Upload & Processing
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={cn(
              "border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer",
              isDragActive || dragActive
                ? "border-primary bg-primary/5"
                : "border-muted-foreground/25 hover:border-primary/50",
              isUploading && "pointer-events-none opacity-50"
            )}
          >
            <input {...getInputProps()} />
            
            {isUploading ? (
              <div className="space-y-4">
                <Loader2 className="h-12 w-12 mx-auto text-primary animate-spin" />
                <div className="space-y-2">
                  <p className="text-sm font-medium">Uploading files...</p>
                  <Progress value={uploadProgress} className="w-full max-w-xs mx-auto" />
                  <p className="text-xs text-muted-foreground">{uploadProgress}% complete</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <Upload className="h-12 w-12 mx-auto text-muted-foreground" />
                <div className="space-y-2">
                  <p className="text-lg font-medium">
                    {isDragActive ? "Drop files here" : "Upload files"}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Drag and drop files here, or click to browse
                  </p>
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p>Supported formats: PDF, DOCX, DOC, TXT, Images, Audio</p>
                    <p>Maximum file size: {formatBytes(maxFileSize)}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Uploaded Files List */}
      {uploadedFiles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Uploaded Files ({uploadedFiles.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {uploadedFiles.map((file) => (
                <FileCard key={file.id} file={file} onRemove={removeFile} />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Upload Guidelines */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <h3 className="font-semibold">Upload Guidelines</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="space-y-2">
                <h4 className="font-medium">📄 Documents</h4>
                <ul className="text-muted-foreground space-y-1">
                  <li>• PDF, DOCX, DOC, TXT</li>
                  <li>• Text extraction and indexing</li>
                  <li>• OCR for scanned documents</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium">🖼️ Images</h4>
                <ul className="text-muted-foreground space-y-1">
                  <li>• JPG, PNG, BMP, TIFF, WEBP</li>
                  <li>• OCR text extraction</li>
                  <li>• Visual similarity search</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium">🎵 Audio</h4>
                <ul className="text-muted-foreground space-y-1">
                  <li>• WAV, MP3, M4A, FLAC, OGG</li>
                  <li>• Speech-to-text transcription</li>
                  <li>• Audio content search</li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

interface FileCardProps {
  file: FileUpload
  onRemove: (fileId: string) => void
}

function FileCard({ file, onRemove }: FileCardProps) {
  const getStatusIcon = () => {
    switch (file.status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      case 'processing':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
      default:
        return <File className="h-4 w-4 text-muted-foreground" />
    }
  }

  const getStatusColor = () => {
    switch (file.status) {
      case 'completed':
        return 'text-green-600 dark:text-green-400'
      case 'error':
        return 'text-red-600 dark:text-red-400'
      case 'processing':
        return 'text-blue-600 dark:text-blue-400'
      default:
        return 'text-muted-foreground'
    }
  }

  return (
    <div className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 transition-colors">
      <div className="flex items-center space-x-3">
        <div className="flex-shrink-0">
          {getStatusIcon()}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center space-x-2">
            <span className="text-lg">{getFileIcon(file.fileName)}</span>
            <div className="min-w-0">
              <p className="text-sm font-medium truncate">{file.fileName}</p>
              <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                <span>{formatBytes(file.fileSize)}</span>
                <span>•</span>
                <span className={getStatusColor()}>{file.status}</span>
                {file.status === 'processing' && (
                  <>
                    <span>•</span>
                    <span>{file.progress}%</span>
                  </>
                )}
              </div>
            </div>
          </div>
          {file.error && (
            <p className="text-xs text-red-600 dark:text-red-400 mt-1">{file.error}</p>
          )}
        </div>
      </div>
      
      <Button
        variant="ghost"
        size="icon"
        onClick={() => onRemove(file.id)}
        className="h-8 w-8 text-muted-foreground hover:text-foreground"
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  )
}
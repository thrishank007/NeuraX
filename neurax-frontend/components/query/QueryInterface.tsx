'use client'

import { useState, useCallback, useRef } from 'react'
import { Search, Mic, Image, Type, X, Loader2, MicOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { apiClient } from '@/lib/api/client'
import toast from 'react-hot-toast'
import type { Query, SearchResult } from '@/types'

interface QueryInterfaceProps {
  onSearch: (query: Query) => void
  onResults: (results: SearchResult[]) => void
  uploadedFiles: any[]
}

type QueryType = 'text' | 'image' | 'voice' | 'multimodal'

export function QueryInterface({ onSearch, onResults, uploadedFiles }: QueryInterfaceProps) {
  const [queryType, setQueryType] = useState<QueryType>('text')
  const [textQuery, setTextQuery] = useState('')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [similarityThreshold, setSimilarityThreshold] = useState(0.5)
  const [isRecording, setIsRecording] = useState(false)
  const [transcription, setTranscription] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)

  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setImageFile(file)
    }
  }, [])

  const handleAudioUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setAudioFile(file)
    }
  }, [])

  const removeImage = useCallback(() => {
    setImageFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [])

  const removeAudio = useCallback(() => {
    setAudioFile(null)
    setTranscription('')
    if (audioInputRef.current) {
      audioInputRef.current.value = ''
    }
  }, [])

  const handleVoiceRecording = useCallback(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      toast.error('Voice recognition not supported in this browser')
      return
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    const recognition = new SpeechRecognition()

    recognition.continuous = false
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onstart = () => {
      setIsRecording(true)
      toast.success('Recording... Speak now')
    }

    recognition.onresult = (event) => {
      let finalTranscript = ''
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript
        }
      }
      
      if (finalTranscript) {
        setTextQuery(finalTranscript)
        setTranscription(finalTranscript)
      }
    }

    recognition.onerror = (event) => {
      setIsRecording(false)
      toast.error('Voice recognition error: ' + event.error)
    }

    recognition.onend = () => {
      setIsRecording(false)
    }

    recognition.start()
  }, [])

  const processQuery = useCallback(async () => {
    if (!textQuery.trim() && !imageFile && !audioFile) {
      toast.error('Please enter a query or upload files')
      return
    }

    setIsProcessing(true)

    try {
      let response
      
      switch (queryType) {
        case 'text':
          response = await apiClient.processTextQuery(
            textQuery,
            similarityThreshold,
            {
              includeImages: true,
              includeDocuments: true,
              includeAudio: true,
              maxResults: 10
            }
          )
          break

        case 'image':
          if (!imageFile) {
            toast.error('Please upload an image')
            return
          }
          response = await apiClient.processImageQuery(
            imageFile,
            similarityThreshold,
            {
              maxResults: 10,
              textQuery: textQuery || undefined
            }
          )
          break

        case 'voice':
          if (!audioFile) {
            toast.error('Please upload an audio file')
            return
          }
          response = await apiClient.processVoiceQuery(
            audioFile,
            similarityThreshold,
            {
              maxResults: 10,
              language: 'en'
            }
          )
          
          if (response.query?.type === 'voice' && response.query?.text) {
            setTextQuery(response.query.text)
            setTranscription(response.query.text)
          }
          break

        case 'multimodal':
          if (!textQuery.trim() && !imageFile) {
            toast.error('Please enter text or upload an image')
            return
          }
          response = await apiClient.processMultimodalQuery(
            textQuery,
            imageFile || undefined,
            similarityThreshold,
            { maxResults: 10 }
          )
          break

        default:
          throw new Error('Invalid query type')
      }

      // Create query object
      const query: Query = {
        id: `query_${Date.now()}`,
        type: queryType,
        text: textQuery,
        image: imageFile || undefined,
        audio: audioFile || undefined,
        timestamp: new Date().toISOString(),
        similarityThreshold,
        results: response.results,
        processingTime: response.processingTime,
        status: 'completed'
      }

      onSearch(query)
      onResults(response.results)
      toast.success(`Found ${response.results.length} results in ${response.processingTime.toFixed(2)}s`)

    } catch (error) {
      console.error('Query processing error:', error)
      toast.error('Query failed: ' + (error instanceof Error ? error.message : 'Unknown error'))
    } finally {
      setIsProcessing(false)
    }
  }, [queryType, textQuery, imageFile, audioFile, similarityThreshold, onSearch, onResults])

  const clearQuery = useCallback(() => {
    setTextQuery('')
    setImageFile(null)
    setAudioFile(null)
    setTranscription('')
    if (fileInputRef.current) fileInputRef.current.value = ''
    if (audioInputRef.current) audioInputRef.current.value = ''
  }, [])

  const renderQueryInput = () => {
    switch (queryType) {
      case 'text':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Search Query</label>
              <Input
                placeholder="Enter your search query (e.g., 'security protocols', 'network configuration')"
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && processQuery()}
                disabled={isProcessing}
              />
            </div>
          </div>
        )

      case 'image':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Optional Text Query</label>
              <Input
                placeholder="Describe what you're looking for..."
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                disabled={isProcessing}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Image</label>
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4">
                {imageFile ? (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Image className="h-4 w-4" />
                      <span className="text-sm">{imageFile.name}</span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={removeImage}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ) : (
                  <div className="text-center">
                    <Image className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                    <Button
                      variant="outline"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isProcessing}
                    >
                      Choose Image
                    </Button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        )

      case 'voice':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Voice Recording</label>
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4">
                {audioFile ? (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Mic className="h-4 w-4" />
                        <span className="text-sm">{audioFile.name}</span>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={removeAudio}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                    {transcription && (
                      <div className="p-2 bg-muted rounded text-sm">
                        <strong>Transcription:</strong> {transcription}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center space-y-2">
                    <Mic className="h-8 w-8 mx-auto text-muted-foreground" />
                    <div className="space-y-2">
                      <Button
                        variant="outline"
                        onClick={() => audioInputRef.current?.click()}
                        disabled={isProcessing}
                      >
                        Upload Audio File
                      </Button>
                      <input
                        ref={audioInputRef}
                        type="file"
                        accept="audio/*"
                        onChange={handleAudioUpload}
                        className="hidden"
                      />
                      {process.env.NEXT_PUBLIC_ENABLE_VOICE_INPUT === 'true' && (
                        <Button
                          variant={isRecording ? "destructive" : "outline"}
                          onClick={handleVoiceRecording}
                          disabled={isProcessing}
                          className="gap-2"
                        >
                          {isRecording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                          {isRecording ? 'Recording...' : 'Record Voice'}
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )

      case 'multimodal':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Text Description</label>
              <Input
                placeholder="Describe what you're looking for..."
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                disabled={isProcessing}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Reference Image (Optional)</label>
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-4">
                {imageFile ? (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Image className="h-4 w-4" />
                      <span className="text-sm">{imageFile.name}</span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={removeImage}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ) : (
                  <div className="text-center">
                    <Image className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                    <Button
                      variant="outline"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isProcessing}
                    >
                      Choose Reference Image
                    </Button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="h-5 w-5" />
          Multimodal Query Interface
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Query Type Selection */}
        <div className="flex flex-wrap gap-2">
          <Button
            variant={queryType === 'text' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setQueryType('text')}
            className="gap-2"
          >
            <Type className="h-4 w-4" />
            Text
          </Button>
          <Button
            variant={queryType === 'image' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setQueryType('image')}
            className="gap-2"
          >
            <Image className="h-4 w-4" />
            Image
          </Button>
          <Button
            variant={queryType === 'voice' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setQueryType('voice')}
            className="gap-2"
          >
            <Mic className="h-4 w-4" />
            Voice
          </Button>
          <Button
            variant={queryType === 'multimodal' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setQueryType('multimodal')}
            className="gap-2"
          >
            <Search className="h-4 w-4" />
            Multimodal
          </Button>
        </div>

        {/* Query Input */}
        {renderQueryInput()}

        {/* Advanced Settings */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">
              Similarity Threshold: {similarityThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              className="w-full"
              disabled={isProcessing}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Lenient</span>
              <span>Strict</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Status</label>
            <div className="flex items-center space-x-2 text-sm">
              {isProcessing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <div className="h-2 w-2 bg-green-500 rounded-full" />
                  <span>Ready</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-2">
          <Button
            onClick={processQuery}
            disabled={isProcessing || (!textQuery.trim() && !imageFile && !audioFile)}
            className="gap-2"
          >
            {isProcessing ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Search className="h-4 w-4" />
            )}
            {isProcessing ? 'Processing...' : 'Search'}
          </Button>
          
          <Button
            variant="outline"
            onClick={clearQuery}
            disabled={isProcessing}
          >
            Clear
          </Button>
        </div>

        {/* Uploaded Files Info */}
        {uploadedFiles.length > 0 && (
          <div className="p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">
              📁 Searching through {uploadedFiles.length} uploaded files
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
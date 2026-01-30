"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent } from "@/components/ui/card"
import { Send, Mic, Image as ImageIcon, Loader2 } from "lucide-react"
import { queryApi } from "@/lib/api/queries"
import toast from "react-hot-toast"
import type { QueryResponse } from "@/lib/types/api"

interface QueryInputProps {
  onQuery: (result: QueryResponse) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export function QueryInput({ onQuery, isLoading, setIsLoading }: QueryInputProps) {
  const [query, setQuery] = useState("")
  const [isRecording, setIsRecording] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || isLoading) return

    setIsLoading(true)
    try {
      const result = await queryApi.processQuery({
        query: query.trim(),
        query_type: "text",
        generate_response: true,
      })
      onQuery(result)
      setQuery("")
      toast.success("Query processed successfully")
    } catch (error: any) {
      toast.error(error.message || "Failed to process query")
    } finally {
      setIsLoading(false)
    }
  }

  const handleVoiceInput = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      toast.error("Voice input not supported in your browser")
      return
    }

    setIsRecording(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      const audioChunks: Blob[] = []

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data)
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" })
        const audioFile = new File([audioBlob], "voice-query.wav", {
          type: "audio/wav",
        })

        setIsLoading(true)
        try {
          const result = await queryApi.processVoiceQuery(audioFile)
          onQuery(result)
          toast.success("Voice query processed successfully")
        } catch (error: any) {
          toast.error(error.message || "Failed to process voice query")
        } finally {
          setIsLoading(false)
        }
      }

      mediaRecorder.start()
      setTimeout(() => {
        mediaRecorder.stop()
        stream.getTracks().forEach((track) => track.stop())
        setIsRecording(false)
      }, 5000) // Record for 5 seconds
    } catch (error: any) {
      toast.error(error.message || "Failed to start voice recording")
      setIsRecording(false)
    }
  }

  const handleImageUpload = () => {
    fileInputRef.current?.click()
  }

  const handleImageFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.type.startsWith("image/")) {
      toast.error("Please select an image file")
      return
    }

    // For now, multimodal queries via image upload would need backend support
    toast.info("Image queries coming soon")
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="relative">
        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your documents..."
          className="min-h-[120px] pr-24"
          disabled={isLoading}
        />
        <div className="absolute bottom-2 right-2 flex gap-2">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={handleVoiceInput}
            disabled={isLoading || isRecording}
            title="Voice input"
          >
            {isRecording ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Mic className="h-4 w-4" />
            )}
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={handleImageUpload}
            disabled={isLoading}
            title="Image input"
          >
            <ImageIcon className="h-4 w-4" />
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageFile}
            className="hidden"
          />
        </div>
      </div>
      <div className="flex justify-end">
        <Button type="submit" disabled={!query.trim() || isLoading}>
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Send className="mr-2 h-4 w-4" />
              Submit Query
            </>
          )}
        </Button>
      </div>
    </form>
  )
}

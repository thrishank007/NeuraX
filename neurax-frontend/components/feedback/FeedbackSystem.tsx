'use client'

import { useState } from 'react'
import { MessageSquare, ThumbsUp, ThumbsDown, Star, Send } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { apiClient } from '@/lib/api/client'
import toast from 'react-hot-toast'

interface FeedbackSystemProps {
  queryId?: string
  responseId?: string
  onFeedbackSubmitted?: () => void
}

export function FeedbackSystem({ queryId, responseId, onFeedbackSubmitted }: FeedbackSystemProps) {
  const [rating, setRating] = useState(0)
  const [comments, setComments] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async () => {
    if (rating === 0) {
      toast.error('Please provide a rating')
      return
    }

    setIsSubmitting(true)
    try {
      await apiClient.submitFeedback({
        queryId: queryId || '',
        responseId: responseId,
        rating,
        comments: comments.trim() || undefined,
        isHelpful: rating >= 4,
        metadata: {
          userAgent: navigator.userAgent,
          timestamp: new Date().toISOString()
        }
      })

      setSubmitted(true)
      onFeedbackSubmitted?.()
      toast.success('Thank you for your feedback!')
    } catch (error) {
      console.error('Feedback submission error:', error)
      toast.error('Failed to submit feedback')
    } finally {
      setIsSubmitting(false)
    }
  }

  const resetForm = () => {
    setRating(0)
    setComments('')
    setSubmitted(false)
  }

  if (submitted) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="text-center py-4">
            <ThumbsUp className="h-8 w-8 mx-auto text-green-500 mb-2" />
            <p className="text-sm font-medium mb-2">Thank you for your feedback!</p>
            <p className="text-xs text-muted-foreground mb-4">
              Your input helps us improve NeuraX for everyone.
            </p>
            <Button variant="outline" size="sm" onClick={resetForm}>
              Submit Another Feedback
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <MessageSquare className="h-5 w-5" />
          Share Your Feedback
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Rating */}
        <div className="space-y-2">
          <label className="text-sm font-medium">How would you rate this result?</label>
          <div className="flex space-x-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => setRating(star)}
                className={`p-1 rounded transition-colors ${
                  star <= rating 
                    ? 'text-yellow-500 hover:text-yellow-600' 
                    : 'text-muted-foreground hover:text-yellow-400'
                }`}
                disabled={isSubmitting}
              >
                <Star className={`h-6 w-6 ${star <= rating ? 'fill-current' : ''}`} />
              </button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            {rating === 0 && 'Click to rate'}
            {rating === 1 && 'Very poor'}
            {rating === 2 && 'Poor'}
            {rating === 3 && 'Average'}
            {rating === 4 && 'Good'}
            {rating === 5 && 'Excellent'}
          </p>
        </div>

        {/* Comments */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Additional comments (optional)</label>
          <textarea
            placeholder="Tell us more about your experience..."
            value={comments}
            onChange={(e) => setComments(e.target.value)}
            disabled={isSubmitting}
            className="w-full min-h-[80px] px-3 py-2 border border-input bg-background rounded-md text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none"
            rows={3}
          />
        </div>

        {/* Quick feedback buttons */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Quick feedback</label>
          <div className="flex space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setRating(5)
                setComments('Very helpful result')
              }}
              disabled={isSubmitting}
              className="gap-2"
            >
              <ThumbsUp className="h-4 w-4" />
              Helpful
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setRating(2)
                setComments('Not very helpful')
              }}
              disabled={isSubmitting}
              className="gap-2"
            >
              <ThumbsDown className="h-4 w-4" />
              Not Helpful
            </Button>
          </div>
        </div>

        {/* Submit */}
        <div className="flex justify-end space-x-2">
          <Button
            variant="outline"
            onClick={resetForm}
            disabled={isSubmitting}
          >
            Reset
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={rating === 0 || isSubmitting}
            className="gap-2"
          >
            {isSubmitting ? (
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            Submit Feedback
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
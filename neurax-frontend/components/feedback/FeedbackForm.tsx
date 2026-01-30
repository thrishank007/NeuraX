"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Star } from "lucide-react"
import { feedbackApi } from "@/lib/api/feedback"
import toast from "react-hot-toast"

interface FeedbackFormProps {
  query: string
  response: string
  onSubmitted?: () => void
}

export function FeedbackForm({ query, response, onSubmitted }: FeedbackFormProps) {
  const [rating, setRating] = useState(0)
  const [comments, setComments] = useState("")
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (rating === 0) {
      toast.error("Please select a rating")
      return
    }

    setSubmitting(true)
    try {
      await feedbackApi.submitFeedback({
        query,
        response,
        rating,
        comments: comments || undefined,
      })
      toast.success("Thank you for your feedback!")
      setRating(0)
      setComments("")
      if (onSubmitted) {
        onSubmitted()
      }
    } catch (error: any) {
      toast.error(error.message || "Failed to submit feedback")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Provide Feedback</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">Rating</label>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setRating(value)}
                  className={`p-1 ${
                    rating >= value
                      ? "text-yellow-400"
                      : "text-gray-300 dark:text-gray-600"
                  }`}
                >
                  <Star
                    className={`h-6 w-6 ${
                      rating >= value ? "fill-current" : ""
                    }`}
                  />
                </button>
              ))}
            </div>
          </div>

          <div>
            <label htmlFor="comments" className="text-sm font-medium mb-2 block">
              Comments (Optional)
            </label>
            <Textarea
              id="comments"
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              placeholder="Share your thoughts..."
              rows={4}
            />
          </div>

          <Button type="submit" disabled={submitting || rating === 0}>
            {submitting ? "Submitting..." : "Submit Feedback"}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

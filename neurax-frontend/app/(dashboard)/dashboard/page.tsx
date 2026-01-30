"use client"

import { useState } from "react"
import { QueryInput } from "@/components/query/QueryInput"
import { ResultsDisplay } from "@/components/results/ResultsDisplay"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { QueryResponse } from "@/lib/types/api"

export default function DashboardPage() {
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleQuery = async (result: QueryResponse) => {
    setQueryResult(result)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Query Interface</h1>
        <p className="text-muted-foreground">
          Ask questions about your documents using text, voice, or images
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Enter Your Query</CardTitle>
        </CardHeader>
        <CardContent>
          <QueryInput
            onQuery={handleQuery}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        </CardContent>
      </Card>

      {queryResult && (
        <ResultsDisplay result={queryResult} />
      )}
    </div>
  )
}

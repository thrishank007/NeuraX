"use client"

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface MetricsChartProps {
  metrics: any
}

export function MetricsChart({ metrics }: MetricsChartProps) {
  if (!metrics) {
    return <p className="text-sm text-muted-foreground">No metrics data available</p>
  }

  // Transform metrics data for chart
  const chartData = [
    {
      name: "Retrieval",
      value: metrics.metrics?.retrieval?.mrr || 0,
    },
    {
      name: "Generation",
      value: metrics.metrics?.generation?.grounding_score || 0,
    },
    {
      name: "Latency",
      value: metrics.metrics?.latency?.average_total_latency_ms || 0,
    },
  ]

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="value" stroke="#8884d8" />
      </LineChart>
    </ResponsiveContainer>
  )
}

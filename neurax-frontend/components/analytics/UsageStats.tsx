"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface UsageStatsProps {
  stats: any
}

export function UsageStats({ stats }: UsageStatsProps) {
  if (!stats) {
    return <p className="text-sm text-muted-foreground">No usage data available</p>
  }

  const chartData = [
    {
      name: "Queries",
      value: stats.queries_per_day || 0,
    },
    {
      name: "Uploads",
      value: stats.file_uploads || 0,
    },
  ]

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="value" fill="#8884d8" />
      </BarChart>
    </ResponsiveContainer>
  )
}

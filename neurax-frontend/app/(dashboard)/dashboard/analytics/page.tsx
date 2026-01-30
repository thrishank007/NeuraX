"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { MetricsChart } from "@/components/analytics/MetricsChart"
import { UsageStats } from "@/components/analytics/UsageStats"
import { SecurityAlerts } from "@/components/analytics/SecurityAlerts"
import { analyticsApi } from "@/lib/api/analytics"
import toast from "react-hot-toast"
import { Loader2 } from "lucide-react"

export default function AnalyticsPage() {
  const [loading, setLoading] = useState(true)
  const [metrics, setMetrics] = useState<any>(null)
  const [usageStats, setUsageStats] = useState<any>(null)
  const [securityEvents, setSecurityEvents] = useState<any>(null)
  const [timeRange, setTimeRange] = useState(24)

  useEffect(() => {
    loadAnalytics()
  }, [timeRange])

  const loadAnalytics = async () => {
    setLoading(true)
    try {
      const [metricsData, usageData, securityData] = await Promise.all([
        analyticsApi.getMetrics(timeRange),
        analyticsApi.getUsageStats(timeRange),
        analyticsApi.getSecurityEvents(50),
      ])
      setMetrics(metricsData)
      setUsageStats(usageData)
      setSecurityEvents(securityData)
    } catch (error: any) {
      toast.error(error.message || "Failed to load analytics")
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Analytics Dashboard</h1>
          <p className="text-muted-foreground">
            System performance metrics and usage statistics
          </p>
        </div>
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(Number(e.target.value))}
          className="px-3 py-2 border rounded-md"
        >
          <option value={1}>Last Hour</option>
          <option value={24}>Last 24 Hours</option>
          <option value={168}>Last Week</option>
          <option value={720}>Last Month</option>
        </select>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <MetricsChart metrics={metrics} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Usage Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <UsageStats stats={usageStats} />
          </CardContent>
        </Card>

        <Card className="md:col-span-2 lg:col-span-3">
          <CardHeader>
            <CardTitle>Security Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <SecurityAlerts events={securityEvents} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

'use client'

import { useState, useEffect } from 'react'
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Database, 
  Clock, 
  Users,
  FileText,
  Image,
  Music,
  Shield,
  AlertTriangle,
  CheckCircle,
  Download
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { apiClient } from '@/lib/api/client'
import toast from 'react-hot-toast'
import type { Analytics, SecurityEvent } from '@/types'

export function AnalyticsDashboard() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [securityEvents, setSecurityEvents] = useState<SecurityEvent[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d' | '90d'>('7d')

  useEffect(() => {
    loadAnalytics()
    loadSecurityEvents()
  }, [timeRange])

  const loadAnalytics = async () => {
    try {
      setIsLoading(true)
      const end = new Date()
      const start = new Date()
      
      switch (timeRange) {
        case '24h':
          start.setDate(start.getDate() - 1)
          break
        case '7d':
          start.setDate(start.getDate() - 7)
          break
        case '30d':
          start.setDate(start.getDate() - 30)
          break
        case '90d':
          start.setDate(start.getDate() - 90)
          break
      }

      const data = await apiClient.getAnalytics({
        start: start.toISOString(),
        end: end.toISOString()
      })
      setAnalytics(data)
    } catch (error) {
      console.error('Failed to load analytics:', error)
      toast.error('Failed to load analytics data')
    } finally {
      setIsLoading(false)
    }
  }

  const loadSecurityEvents = async () => {
    try {
      const events = await apiClient.getSecurityEvents(1, 10)
      if (events.success && events.data) {
        setSecurityEvents(events.data)
      }
    } catch (error) {
      console.error('Failed to load security events:', error)
    }
  }

  const exportAnalytics = async () => {
    try {
      const blob = await apiClient.exportAnalytics('json', {
        timeRange,
        timestamp: new Date().toISOString()
      })
      
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `neurax-analytics-${timeRange}-${Date.now()}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast.success('Analytics exported successfully')
    } catch (error) {
      console.error('Export failed:', error)
      toast.error('Failed to export analytics')
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardContent className="pt-6">
                <div className="animate-pulse">
                  <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                  <div className="h-8 bg-muted rounded w-1/2" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  if (!analytics) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="text-center py-12">
            <AlertTriangle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Analytics Data</h3>
            <p className="text-muted-foreground mb-4">
              Analytics data is not available. The system may not have been running long enough to generate meaningful statistics.
            </p>
            <Button onClick={loadAnalytics}>Retry</Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Analytics Dashboard</h2>
          <p className="text-muted-foreground">System performance and usage analytics</p>
        </div>
        <div className="flex items-center space-x-2">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="px-3 py-1 border rounded-md text-sm"
          >
            <option value="24h">Last 24 hours</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
          <Button onClick={exportAnalytics} variant="outline" size="sm" className="gap-2">
            <Download className="h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Queries"
          value={analytics.queryStats.totalQueries.toString()}
          icon={<BarChart3 className="h-5 w-5" />}
          trend={analytics.queryStats.successRate > 80 ? 'up' : 'down'}
          trendValue={`${analytics.queryStats.successRate.toFixed(1)}% success rate`}
        />
        <StatCard
          title="Files Processed"
          value={analytics.fileStats.totalFiles.toString()}
          icon={<FileText className="h-5 w-5" />}
          trend="up"
          trendValue={`${(analytics.fileStats.totalSize / 1024 / 1024).toFixed(1)} MB total`}
        />
        <StatCard
          title="Avg Response Time"
          value={`${analytics.queryStats.avgProcessingTime.toFixed(2)}s`}
          icon={<Clock className="h-5 w-5" />}
          trend={analytics.queryStats.avgProcessingTime < 2 ? 'up' : 'down'}
          trendValue={analytics.queryStats.avgProcessingTime < 2 ? 'Fast' : 'Needs optimization'}
        />
        <StatCard
          title="System Uptime"
          value={`${Math.floor(analytics.systemStats.uptime / 3600)}h`}
          icon={<Activity className="h-5 w-5" />}
          trend="up"
          trendValue={`${((analytics.systemStats.uptime / 86400) * 100).toFixed(1)}% of time`}
        />
      </div>

      {/* Query Statistics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Query Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  <span className="text-sm">Text Queries</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">{analytics.queryStats.textQueries}</span>
                  <div className="w-20 bg-muted rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${(analytics.queryStats.textQueries / analytics.queryStats.totalQueries) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Image className="h-4 w-4" />
                  <span className="text-sm">Image Queries</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">{analytics.queryStats.imageQueries}</span>
                  <div className="w-20 bg-muted rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${(analytics.queryStats.imageQueries / analytics.queryStats.totalQueries) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Music className="h-4 w-4" />
                  <span className="text-sm">Voice Queries</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">{analytics.queryStats.voiceQueries}</span>
                  <div className="w-20 bg-muted rounded-full h-2">
                    <div 
                      className="bg-purple-500 h-2 rounded-full"
                      style={{ width: `${(analytics.queryStats.voiceQueries / analytics.queryStats.totalQueries) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  <span className="text-sm">Multimodal Queries</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">{analytics.queryStats.multimodalQueries}</span>
                  <div className="w-20 bg-muted rounded-full h-2">
                    <div 
                      className="bg-orange-500 h-2 rounded-full"
                      style={{ width: `${(analytics.queryStats.multimodalQueries / analytics.queryStats.totalQueries) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              File Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  <span className="text-sm">Documents</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">{analytics.fileStats.documentFiles}</span>
                  <Badge variant="secondary">{((analytics.fileStats.documentFiles / analytics.fileStats.totalFiles) * 100).toFixed(1)}%</Badge>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Image className="h-4 w-4" />
                  <span className="text-sm">Images</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">{analytics.fileStats.imageFiles}</span>
                  <Badge variant="secondary">{((analytics.fileStats.imageFiles / analytics.fileStats.totalFiles) * 100).toFixed(1)}%</Badge>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Music className="h-4 w-4" />
                  <span className="text-sm">Audio</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">{analytics.fileStats.audioFiles}</span>
                  <Badge variant="secondary">{((analytics.fileStats.audioFiles / analytics.fileStats.totalFiles) * 100).toFixed(1)}%</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <PerformanceMetric
                label="CPU Usage"
                value={analytics.systemStats.cpuUsage}
                unit="%"
                color={analytics.systemStats.cpuUsage > 80 ? 'red' : analytics.systemStats.cpuUsage > 60 ? 'yellow' : 'green'}
              />
              <PerformanceMetric
                label="Memory Usage"
                value={analytics.systemStats.memoryUsage}
                unit="%"
                color={analytics.systemStats.memoryUsage > 80 ? 'red' : analytics.systemStats.memoryUsage > 60 ? 'yellow' : 'green'}
              />
              <PerformanceMetric
                label="Disk Usage"
                value={analytics.systemStats.diskUsage}
                unit="%"
                color={analytics.systemStats.diskUsage > 90 ? 'red' : analytics.systemStats.diskUsage > 75 ? 'yellow' : 'green'}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Usage Trends
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {analytics.usageTrends.slice(-7).map((trend, index) => (
                <div key={index} className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">
                    {new Date(trend.date).toLocaleDateString()}
                  </span>
                  <div className="flex items-center space-x-2">
                    <span>{trend.queries} queries</span>
                    <span>{trend.uploads} uploads</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Popular Queries
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {analytics.popularQueries.slice(0, 5).map((query, index) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-sm truncate flex-1" title={query.query}>
                    {query.query}
                  </span>
                  <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                    <span>{query.count}</span>
                    <span>•</span>
                    <span>{query.avgRating.toFixed(1)}★</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Security Events */}
      {securityEvents.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Recent Security Events
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {securityEvents.map((event) => (
                <div key={event.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className={cn(
                      "h-2 w-2 rounded-full",
                      event.severity === 'critical' ? 'bg-red-500' :
                      event.severity === 'high' ? 'bg-orange-500' :
                      event.severity === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                    )} />
                    <div>
                      <p className="text-sm font-medium">{event.title}</p>
                      <p className="text-xs text-muted-foreground">{event.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="text-xs">
                      {event.severity}
                    </Badge>
                    <Badge variant={event.resolved ? 'default' : 'destructive'} className="text-xs">
                      {event.resolved ? 'Resolved' : 'Active'}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

interface StatCardProps {
  title: string
  value: string
  icon: React.ReactNode
  trend: 'up' | 'down'
  trendValue: string
}

function StatCard({ title, value, icon, trend, trendValue }: StatCardProps) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold">{value}</p>
            <div className="flex items-center space-x-1 mt-1">
              {trend === 'up' ? (
                <TrendingUp className="h-3 w-3 text-green-500" />
              ) : (
                <TrendingDown className="h-3 w-3 text-red-500" />
              )}
              <span className="text-xs text-muted-foreground">{trendValue}</span>
            </div>
          </div>
          <div className="text-muted-foreground">
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

interface PerformanceMetricProps {
  label: string
  value: number
  unit: string
  color: 'green' | 'yellow' | 'red'
}

function PerformanceMetric({ label, value, unit, color }: PerformanceMetricProps) {
  const colorClasses = {
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500'
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span>{label}</span>
        <span>{value.toFixed(1)}{unit}</span>
      </div>
      <div className="w-full bg-muted rounded-full h-2">
        <div 
          className={cn("h-2 rounded-full transition-all", colorClasses[color])}
          style={{ width: `${Math.min(value, 100)}%` }}
        />
      </div>
    </div>
  )
}
"use client"

import { Badge } from "@/components/ui/badge"
import { AlertTriangle, Shield, Info } from "lucide-react"

interface SecurityAlertsProps {
  events: any
}

export function SecurityAlerts({ events }: SecurityAlertsProps) {
  if (!events || !events.anomalies || events.anomalies.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-muted-foreground">
        <Shield className="h-8 w-8 mr-2" />
        <span>No security alerts</span>
      </div>
    )
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case "high":
        return <AlertTriangle className="h-4 w-4 text-red-500" />
      case "medium":
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      default:
        return <Info className="h-4 w-4 text-blue-500" />
    }
  }

  const getSeverityBadge = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case "high":
        return <Badge variant="destructive">High</Badge>
      case "medium":
        return <Badge className="bg-yellow-500">Medium</Badge>
      default:
        return <Badge variant="secondary">Low</Badge>
    }
  }

  return (
    <div className="space-y-2">
      {events.anomalies.map((anomaly: any, index: number) => (
        <div
          key={index}
          className="flex items-start gap-3 p-3 border rounded-lg hover:bg-accent transition-colors"
        >
          {getSeverityIcon(anomaly.severity)}
          <div className="flex-1 space-y-1">
            <div className="flex items-center justify-between">
              <span className="font-medium text-sm">{anomaly.type}</span>
              {getSeverityBadge(anomaly.severity)}
            </div>
            <p className="text-sm text-muted-foreground">{anomaly.description}</p>
            <p className="text-xs text-muted-foreground">
              {anomaly.timestamp ? new Date(anomaly.timestamp).toLocaleString() : ""}
            </p>
          </div>
        </div>
      ))}
    </div>
  )
}

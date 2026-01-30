"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { configApi } from "@/lib/api/config"
import toast from "react-hot-toast"
import { Loader2, Save } from "lucide-react"
import type { ConfigResponse } from "@/lib/types/api"

export default function SettingsPage() {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [config, setConfig] = useState<Partial<ConfigResponse>>({
    lm_studio_url: "",
    similarity_threshold: 0.5,
    max_results: 10,
    model_preference: "auto",
  })

  useEffect(() => {
    loadConfig()
  }, [])

  const loadConfig = async () => {
    try {
      const data = await configApi.getConfig()
      setConfig(data)
    } catch (error: any) {
      toast.error(error.message || "Failed to load configuration")
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      // Validate first
      const validation = await configApi.validateConfig(config)
      if (!validation.valid) {
        toast.error(`Validation failed: ${validation.errors.join(", ")}`)
        return
      }

      await configApi.updateConfig(config)
      toast.success("Configuration saved successfully")
    } catch (error: any) {
      toast.error(error.message || "Failed to save configuration")
    } finally {
      setSaving(false)
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
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Configure system settings and preferences
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>LM Studio Configuration</CardTitle>
          <CardDescription>
            Configure connection to LM Studio server
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="lm_studio_url">LM Studio Base URL</Label>
            <Input
              id="lm_studio_url"
              value={config.lm_studio_url || ""}
              onChange={(e) =>
                setConfig({ ...config, lm_studio_url: e.target.value })
              }
              placeholder="http://localhost:1234/v1"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Search Configuration</CardTitle>
          <CardDescription>
            Configure search and retrieval parameters
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="similarity_threshold">Similarity Threshold</Label>
            <Input
              id="similarity_threshold"
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={config.similarity_threshold || 0.5}
              onChange={(e) =>
                setConfig({
                  ...config,
                  similarity_threshold: parseFloat(e.target.value),
                })
              }
            />
            <p className="text-xs text-muted-foreground">
              Minimum similarity score for results (0.0 - 1.0)
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="max_results">Max Results</Label>
            <Input
              id="max_results"
              type="number"
              min="1"
              max="100"
              value={config.max_results || 10}
              onChange={(e) =>
                setConfig({
                  ...config,
                  max_results: parseInt(e.target.value),
                })
              }
            />
            <p className="text-xs text-muted-foreground">
              Maximum number of results to return (1 - 100)
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="model_preference">Model Preference</Label>
            <select
              id="model_preference"
              value={config.model_preference || "auto"}
              onChange={(e) =>
                setConfig({ ...config, model_preference: e.target.value })
              }
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            >
              <option value="auto">Auto (Recommended)</option>
              <option value="gemma">Gemma (Multimodal)</option>
              <option value="qwen">Qwen (Reasoning)</option>
            </select>
            <p className="text-xs text-muted-foreground">
              Preferred model for generation
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={saving}>
          {saving ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Saving...
            </>
          ) : (
            <>
              <Save className="mr-2 h-4 w-4" />
              Save Changes
            </>
          )}
        </Button>
      </div>
    </div>
  )
}

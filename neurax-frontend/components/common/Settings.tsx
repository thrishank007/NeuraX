'use client'

import { useState, useEffect } from 'react'
import { 
  Settings as SettingsIcon, 
  Save, 
  RotateCcw, 
  Upload, 
  Database, 
  Shield, 
  Bell, 
  Monitor,
  Palette,
  Globe,
  HardDrive,
  Cpu,
  MemoryStick
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { apiClient } from '@/lib/api/client'
import toast from 'react-hot-toast'
import type { SystemConfig } from '@/types'

interface SettingsProps {
  onConfigChange?: (config: Partial<SystemConfig>) => void
}

export function Settings({ onConfigChange }: SettingsProps) {
  const [config, setConfig] = useState<Partial<SystemConfig>>({})
  const [originalConfig, setOriginalConfig] = useState<Partial<SystemConfig>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)

  useEffect(() => {
    loadConfig()
  }, [])

  useEffect(() => {
    // Check if there are changes compared to original config
    const hasChanges = JSON.stringify(config) !== JSON.stringify(originalConfig)
    setHasChanges(hasChanges)
  }, [config, originalConfig])

  const loadConfig = async () => {
    try {
      setIsLoading(true)
      
      // Try to load from backend API
      try {
        const backendConfig = await apiClient.getSystemConfig()
        setConfig(backendConfig)
        setOriginalConfig(backendConfig)
      } catch (apiError) {
        // Fallback to environment variables and defaults
        const fallbackConfig: Partial<SystemConfig> = {
          apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
          wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
          lmStudioUrl: process.env.NEXT_PUBLIC_LM_STUDIO_URL || 'http://localhost:1234',
          maxFileSize: parseInt(process.env.NEXT_PUBLIC_MAX_FILE_SIZE || '104857600'),
          allowedFileTypes: (process.env.NEXT_PUBLIC_ALLOWED_FILE_TYPES || '.pdf,.docx,.doc,.txt,.jpg,.png,.mp3,.wav,.m4a,.flac,.ogg,.bmp,.tiff,.webp').split(','),
          enableAnalytics: process.env.NEXT_PUBLIC_ENABLE_ANALYTICS === 'true',
          enableDarkMode: process.env.NEXT_PUBLIC_ENABLE_DARK_MODE === 'true',
          defaultSimilarityThreshold: parseFloat(process.env.NEXT_PUBLIC_DEFAULT_SIMILARITY_THRESHOLD || '0.5'),
          maxQueryHistory: parseInt(process.env.NEXT_PUBLIC_MAX_QUERY_HISTORY || '50'),
          enableVoiceInput: process.env.NEXT_PUBLIC_ENABLE_VOICE_INPUT === 'true',
          models: {
            primary: 'gemma-3n',
            fallback: 'qwen3-4b'
          },
          performance: {
            batchSize: 10,
            maxConcurrency: 4,
            cacheEnabled: true,
            cacheTimeout: 3600
          },
          security: {
            auditLogging: true,
            anomalyDetection: true,
            rateLimiting: true,
            maxUploadsPerHour: 100
          }
        }
        setConfig(fallbackConfig)
        setOriginalConfig(fallbackConfig)
      }
    } catch (error) {
      console.error('Failed to load configuration:', error)
      toast.error('Failed to load configuration')
    } finally {
      setIsLoading(false)
    }
  }

  const saveConfig = async () => {
    try {
      setIsSaving(true)
      
      // Validate configuration
      const validationResponse = await apiClient.validateConfig(config)
      if (!validationResponse.success) {
        toast.error('Configuration validation failed: ' + validationResponse.error)
        return
      }
      
      // Save to backend
      try {
        const savedConfig = await apiClient.updateSystemConfig(config)
        setConfig(savedConfig)
        setOriginalConfig(savedConfig)
        onConfigChange?.(savedConfig)
        toast.success('Configuration saved successfully')
      } catch (apiError) {
        // Save to localStorage as fallback
        localStorage.setItem('neurax_config', JSON.stringify(config))
        setOriginalConfig(config)
        onConfigChange?.(config)
        toast.success('Configuration saved locally (backend not available)')
      }
    } catch (error) {
      console.error('Failed to save configuration:', error)
      toast.error('Failed to save configuration')
    } finally {
      setIsSaving(false)
    }
  }

  const resetConfig = () => {
    setConfig(originalConfig)
    toast.info('Configuration reset to original values')
  }

  const updateConfig = (path: string, value: any) => {
    setConfig(prev => {
      const newConfig = { ...prev }
      const keys = path.split('.')
      let current: any = newConfig
      
      for (let i = 0; i < keys.length - 1; i++) {
        if (!current[keys[i]]) {
          current[keys[i]] = {}
        }
        current = current[keys[i]]
      }
      
      current[keys[keys.length - 1]] = value
      return newConfig
    })
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-muted rounded w-1/4 mb-2" />
                <div className="h-10 bg-muted rounded" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <SettingsIcon className="h-5 w-5" />
                System Settings
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                Configure NeuraX system preferences and behavior
              </p>
            </div>
            <div className="flex items-center space-x-2">
              {hasChanges && (
                <Badge variant="destructive">Unsaved Changes</Badge>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={resetConfig}
                disabled={!hasChanges}
                className="gap-2"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
              <Button
                onClick={saveConfig}
                disabled={!hasChanges || isSaving}
                className="gap-2"
              >
                {isSaving ? (
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
                {isSaving ? 'Saving...' : 'Save Changes'}
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* General Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Monitor className="h-5 w-5" />
              General Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Default Similarity Threshold</label>
              <div className="space-y-2">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={config.defaultSimilarityThreshold || 0.5}
                  onChange={(e) => updateConfig('defaultSimilarityThreshold', parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Lenient (0.0)</span>
                  <span className="font-medium">{(config.defaultSimilarityThreshold || 0.5).toFixed(2)}</span>
                  <span>Strict (1.0)</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Max Query History</label>
              <Input
                type="number"
                value={config.maxQueryHistory || 50}
                onChange={(e) => updateConfig('maxQueryHistory', parseInt(e.target.value))}
                min="10"
                max="1000"
              />
              <p className="text-xs text-muted-foreground">
                Number of queries to keep in history
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Max File Size (MB)</label>
              <Input
                type="number"
                value={(config.maxFileSize || 104857600) / 1024 / 1024}
                onChange={(e) => updateConfig('maxFileSize', parseInt(e.target.value) * 1024 * 1024)}
                min="1"
                max="1000"
              />
              <p className="text-xs text-muted-foreground">
                Maximum file size for uploads
              </p>
            </div>

            <div className="space-y-3">
              <label className="text-sm font-medium">Features</label>
              <div className="space-y-2">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.enableAnalytics || false}
                    onChange={(e) => updateConfig('enableAnalytics', e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm">Enable Analytics</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.enableDarkMode || false}
                    onChange={(e) => updateConfig('enableDarkMode', e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm">Enable Dark Mode</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.enableVoiceInput || false}
                    onChange={(e) => updateConfig('enableVoiceInput', e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm">Enable Voice Input</span>
                </label>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Model Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Model Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Primary Model</label>
              <select
                value={config.models?.primary || 'gemma-3n'}
                onChange={(e) => updateConfig('models.primary', e.target.value)}
                className="w-full px-3 py-2 border rounded-md"
              >
                <option value="gemma-3n">Gemma 3n (Multimodal)</option>
                <option value="qwen3-4b">Qwen3 4b (Reasoning)</option>
              </select>
              <p className="text-xs text-muted-foreground">
                Main model for most queries
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Fallback Model</label>
              <select
                value={config.models?.fallback || 'qwen3-4b'}
                onChange={(e) => updateConfig('models.fallback', e.target.value)}
                className="w-full px-3 py-2 border rounded-md"
              >
                <option value="gemma-3n">Gemma 3n (Multimodal)</option>
                <option value="qwen3-4b">Qwen3 4b (Reasoning)</option>
              </select>
              <p className="text-xs text-muted-foreground">
                Backup model if primary fails
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">LM Studio URL</label>
              <Input
                type="url"
                value={config.lmStudioUrl || ''}
                onChange={(e) => updateConfig('lmStudioUrl', e.target.value)}
                placeholder="http://localhost:1234"
              />
              <p className="text-xs text-muted-foreground">
                URL where LM Studio is running
              </p>
            </div>

            <div className="space-y-3">
              <label className="text-sm font-medium">Performance Settings</label>
              <div className="space-y-2">
                <div>
                  <label className="text-sm">Batch Size</label>
                  <Input
                    type="number"
                    value={config.performance?.batchSize || 10}
                    onChange={(e) => updateConfig('performance.batchSize', parseInt(e.target.value))}
                    min="1"
                    max="50"
                  />
                </div>
                <div>
                  <label className="text-sm">Max Concurrency</label>
                  <Input
                    type="number"
                    value={config.performance?.maxConcurrency || 4}
                    onChange={(e) => updateConfig('performance.maxConcurrency', parseInt(e.target.value))}
                    min="1"
                    max="16"
                  />
                </div>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.performance?.cacheEnabled || false}
                    onChange={(e) => updateConfig('performance.cacheEnabled', e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm">Enable Caching</span>
                </label>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Security Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Security & Privacy
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <label className="text-sm font-medium">Security Features</label>
              <div className="space-y-2">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.security?.auditLogging || false}
                    onChange={(e) => updateConfig('security.auditLogging', e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm">Audit Logging</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.security?.anomalyDetection || false}
                    onChange={(e) => updateConfig('security.anomalyDetection', e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm">Anomaly Detection</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.security?.rateLimiting || false}
                    onChange={(e) => updateConfig('security.rateLimiting', e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm">Rate Limiting</span>
                </label>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Max Uploads Per Hour</label>
              <Input
                type="number"
                value={config.security?.maxUploadsPerHour || 100}
                onChange={(e) => updateConfig('security.maxUploadsPerHour', parseInt(e.target.value))}
                min="10"
                max="1000"
              />
              <p className="text-xs text-muted-foreground">
                Rate limit for file uploads
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Allowed File Types</label>
              <div className="space-y-2">
                <div className="flex flex-wrap gap-2">
                  {(config.allowedFileTypes || []).map((type, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      {type}
                    </Badge>
                  ))}
                </div>
                <Input
                  placeholder="Enter file extensions (e.g., .pdf,.docx)"
                  value={(config.allowedFileTypes || []).join(',')}
                  onChange={(e) => updateConfig('allowedFileTypes', e.target.value.split(',').map(t => t.trim()).filter(Boolean))}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* System Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <HardDrive className="h-5 w-5" />
              System Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Browser</span>
                </div>
                <span className="text-sm text-muted-foreground">
                  {typeof navigator !== 'undefined' ? navigator.userAgent.split(' ')[0] : 'Unknown'}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MemoryStick className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Memory</span>
                </div>
                <span className="text-sm text-muted-foreground">
                  {typeof navigator !== 'undefined' && 'memory' in performance ? 
                    `${Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024)}MB` : 
                    'N/A'
                  }
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Globe className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Connection</span>
                </div>
                <span className="text-sm text-muted-foreground">
                  {typeof navigator !== 'undefined' && 'connection' in navigator ? 
                    (navigator as any).connection.effectiveType : 
                    'Unknown'
                  }
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Monitor className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Screen</span>
                </div>
                <span className="text-sm text-muted-foreground">
                  {typeof window !== 'undefined' ? `${window.screen.width}x${window.screen.height}` : 'N/A'}
                </span>
              </div>
            </div>

            <div className="pt-4 border-t space-y-2">
              <h4 className="text-sm font-medium">Environment</h4>
              <div className="space-y-1 text-xs text-muted-foreground">
                <div>API URL: {config.apiUrl || 'Not configured'}</div>
                <div>WebSocket URL: {config.wsUrl || 'Not configured'}</div>
                <div>LM Studio: {config.lmStudioUrl || 'Not configured'}</div>
              </div>
            </div>

            <div className="pt-4 border-t">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  // Export configuration
                  const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `neurax-config-${Date.now()}.json`
                  document.body.appendChild(a)
                  a.click()
                  document.body.removeChild(a)
                  URL.revokeObjectURL(url)
                  toast.success('Configuration exported')
                }}
                className="w-full gap-2"
              >
                <Upload className="h-4 w-4" />
                Export Configuration
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
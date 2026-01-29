'use client'

import { useState, useEffect } from 'react'
import { Moon, Sun, Settings, Search, Upload, FileText, BarChart3, History, Shield } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface HeaderProps {
  className?: string
  onThemeToggle?: () => void
  onSearch?: () => void
  onUpload?: () => void
  onAnalytics?: () => void
  onHistory?: () => void
  onSettings?: () => void
}

export function Header({
  className,
  onThemeToggle,
  onSearch,
  onUpload,
  onAnalytics,
  onHistory,
  onSettings
}: HeaderProps) {
  const [isDarkMode, setIsDarkMode] = useState(false)

  useEffect(() => {
    // Check system preference or stored preference
    const savedTheme = localStorage.getItem('theme')
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    
    setIsDarkMode(savedTheme === 'dark' || (!savedTheme && systemPrefersDark))
  }, [])

  useEffect(() => {
    // Apply theme to document
    if (isDarkMode) {
      document.documentElement.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      document.documentElement.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }, [isDarkMode])

  const handleThemeToggle = () => {
    setIsDarkMode(!isDarkMode)
    onThemeToggle?.()
  }

  return (
    <header className={cn(
      "sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60",
      className
    )}>
      <div className="container flex h-14 items-center">
        {/* Logo and Title */}
        <div className="mr-4 flex items-center space-x-2">
          <Shield className="h-6 w-6 text-primary" />
          <h1 className="text-xl font-bold">NeuraX</h1>
          <span className="text-sm text-muted-foreground">RAG System</span>
        </div>

        {/* Navigation */}
        <nav className="flex items-center space-x-1 flex-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={onSearch}
            className="gap-2"
          >
            <Search className="h-4 w-4" />
            Search
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={onUpload}
            className="gap-2"
          >
            <Upload className="h-4 w-4" />
            Upload
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={onAnalytics}
            className="gap-2"
          >
            <BarChart3 className="h-4 w-4" />
            Analytics
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={onHistory}
            className="gap-2"
          >
            <History className="h-4 w-4" />
            History
          </Button>
        </nav>

        {/* Right side controls */}
        <div className="flex items-center space-x-2">
          {/* System Status Indicator */}
          <div className="flex items-center space-x-1 text-sm text-muted-foreground">
            <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse" />
            <span className="hidden sm:inline">Online</span>
          </div>

          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={handleThemeToggle}
            className="h-9 w-9"
          >
            {isDarkMode ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
            <span className="sr-only">Toggle theme</span>
          </Button>

          {/* Settings */}
          <Button
            variant="ghost"
            size="icon"
            onClick={onSettings}
            className="h-9 w-9"
          >
            <Settings className="h-4 w-4" />
            <span className="sr-only">Settings</span>
          </Button>
        </div>
      </div>
    </header>
  )
}
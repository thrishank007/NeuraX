import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 Bytes'

  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']

  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = Math.round(seconds % 60)
    return `${minutes}m ${remainingSeconds}s`
  } else {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const remainingSeconds = Math.round(seconds % 60)
    return `${hours}h ${minutes}m ${remainingSeconds}s`
  }
}

export function formatSimilarityScore(score: number): { label: string; color: string } {
  if (score >= 0.8) {
    return { label: 'Excellent', color: 'text-green-600 dark:text-green-400' }
  } else if (score >= 0.6) {
    return { label: 'Good', color: 'text-yellow-600 dark:text-yellow-400' }
  } else {
    return { label: 'Fair', color: 'text-red-600 dark:text-red-400' }
  }
}

export function formatFileType(fileType: string): string {
  const typeMap: Record<string, string> = {
    'document': '📄 Document',
    'image': '🖼️ Image',
    'audio': '🎵 Audio',
    'unknown': '❓ Unknown'
  }
  return typeMap[fileType] || '❓ Unknown'
}

export function getFileIcon(fileName: string): string {
  const extension = fileName.split('.').pop()?.toLowerCase()
  
  const iconMap: Record<string, string> = {
    // Documents
    'pdf': '📄',
    'doc': '📝',
    'docx': '📝',
    'txt': '📄',
    'rtf': '📄',
    'odt': '📄',
    
    // Images
    'jpg': '🖼️',
    'jpeg': '🖼️',
    'png': '🖼️',
    'gif': '🖼️',
    'bmp': '🖼️',
    'tiff': '🖼️',
    'webp': '🖼️',
    'svg': '🎨',
    
    // Audio
    'mp3': '🎵',
    'wav': '🎵',
    'm4a': '🎵',
    'flac': '🎵',
    'ogg': '🎵',
    'aac': '🎵',
    
    // Archives
    'zip': '📦',
    'rar': '📦',
    '7z': '📦',
    'tar': '📦',
    'gz': '📦'
  }
  
  return iconMap[extension || ''] || '📄'
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null
  
  return (...args: Parameters<T>) => {
    if (timeout) {
      clearTimeout(timeout)
    }
    
    timeout = setTimeout(() => {
      func(...args)
    }, wait)
  }
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

export function generateId(): string {
  return Math.random().toString(36).substr(2, 9)
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + '...'
}

export function highlightSearchTerms(text: string, searchTerm: string): string {
  if (!searchTerm) return text
  
  const regex = new RegExp(`(${searchTerm})`, 'gi')
  return text.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-800">$1</mark>')
}

export function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

export function validateFileType(fileName: string, allowedTypes: string[]): boolean {
  const extension = fileName.split('.').pop()?.toLowerCase()
  return extension ? allowedTypes.includes(`.${extension}`) : false
}

export function validateFileSize(fileSize: number, maxSize: number): boolean {
  return fileSize <= maxSize
}

export function isImageFile(fileName: string): boolean {
  const imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'svg']
  const extension = fileName.split('.').pop()?.toLowerCase()
  return extension ? imageExtensions.includes(extension) : false
}

export function isAudioFile(fileName: string): boolean {
  const audioExtensions = ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac']
  const extension = fileName.split('.').pop()?.toLowerCase()
  return extension ? audioExtensions.includes(extension) : false
}

export function isDocumentFile(fileName: string): boolean {
  const documentExtensions = ['pdf', 'doc', 'docx', 'txt', 'rtf', 'odt']
  const extension = fileName.split('.').pop()?.toLowerCase()
  return extension ? documentExtensions.includes(extension) : false
}

export function getFileType(fileName: string): 'document' | 'image' | 'audio' | 'unknown' {
  if (isImageFile(fileName)) return 'image'
  if (isAudioFile(fileName)) return 'audio'
  if (isDocumentFile(fileName)) return 'document'
  return 'unknown'
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text)
  } else {
    // Fallback for older browsers
    const textArea = document.createElement('textarea')
    textArea.value = text
    textArea.style.position = 'absolute'
    textArea.style.left = '-999999px'
    
    document.body.prepend(textArea)
    textArea.select()
    
    try {
      document.execCommand('copy')
    } finally {
      textArea.remove()
    }
    
    return Promise.resolve()
  }
}

export function shareContent(data: {
  title?: string;
  text?: string;
  url?: string;
}): Promise<void> {
  if (navigator.share) {
    return navigator.share(data)
  } else {
    // Fallback to clipboard
    const shareText = [data.title, data.text, data.url]
      .filter(Boolean)
      .join('\n')
    
    return copyToClipboard(shareText)
  }
}
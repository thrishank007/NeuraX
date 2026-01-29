import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'NeuraX - Multimodal RAG System',
  description: 'Secure, offline multimodal retrieval-augmented generation system with document intelligence and analytics.',
  keywords: 'RAG, multimodal, document intelligence, AI, security, offline',
  authors: [{ name: 'NeuraX Team' }],
  viewport: 'width=device-width, initial-scale=1',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <div className="relative flex min-h-screen flex-col">
          {children}
        </div>
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'hsl(var(--background))',
              color: 'hsl(var(--foreground))',
              border: '1px solid hsl(var(--border))',
            },
            success: {
              className: 'toast-success',
            },
            error: {
              className: 'toast-error',
            },
            loading: {
              className: 'toast-info',
            },
          }}
        />
      </body>
    </html>
  )
}
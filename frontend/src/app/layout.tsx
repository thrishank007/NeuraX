import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import '@/styles/globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'NeuraX - Offline Multimodal RAG System',
  description: 'Secure, air-gapped document intelligence with advanced multimodal capabilities',
  keywords: ['RAG', 'AI', 'Document Intelligence', 'Offline', 'Multimodal'],
  authors: [{ name: 'NeuraX Team' }],
  viewport: 'width=device-width, initial-scale=1',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div id="root">
          {children}
        </div>
      </body>
    </html>
  )
}
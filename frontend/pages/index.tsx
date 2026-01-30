import Head from 'next/head'
import { useState } from 'react'

export default function Home() {
  const [files, setFiles] = useState<File[]>([])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files))
    }
  }

  const handleUpload = async () => {
    if (files.length === 0) return
    
    setIsLoading(true)
    setError('')
    
    try {
      const formData = new FormData()
      files.forEach(file => {
        formData.append('files', file)
      })

      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const data = await response.json()
      alert(`Upload successful! ${data.files.length} files processed.`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
      console.error('Upload error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleQuery = async () => {
    if (!query.trim()) return
    
    setIsLoading(true)
    setError('')
    
    try {
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: query })
      })

      if (!response.ok) {
        throw new Error('Query failed')
      }

      const data = await response.json()
      setResults(data.results || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query failed')
      console.error('Query error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <Head>
        <title>NeuraX - Multimodal RAG System</title>
        <meta name="description" content="Offline multimodal RAG system" />
      </Head>
      
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center text-blue-600">NeuraX</h1>
        <p className="text-center text-gray-600 mb-8">Offline Multimodal RAG System</p>
        
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            Error: {error}
          </div>
        )}

        <div className="bg-white p-6 rounded-lg shadow mb-6">
          <h2 className="text-xl font-semibold mb-4">File Upload</h2>
          <div className="mb-4">
            <input 
              type="file" 
              multiple 
              onChange={handleFileUpload} 
              className="border p-2 rounded w-full"
              accept=".pdf,.docx,.doc,.txt,.jpg,.jpeg,.png,.bmp,.tiff,.webp,.wav,.mp3,.m4a,.flac,.ogg"
            />
          </div>
          <button 
            onClick={handleUpload} 
            disabled={files.length === 0 || isLoading}
            className={`bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 ${(files.length === 0 || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isLoading ? 'Uploading...' : `Upload ${files.length} file(s)`}
          </button>
          
          {files.length > 0 && (
            <div className="mt-4">
              <h3 className="font-medium">Selected Files:</h3>
              <ul className="list-disc pl-5">
                {files.map((file, index) => (
                  <li key={index} className="text-sm text-gray-600">{file.name} ({Math.round(file.size / 1024)} KB)</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="bg-white p-6 rounded-lg shadow mb-6">
          <h2 className="text-xl font-semibold mb-4">Query</h2>
          <textarea 
            value={query} 
            onChange={(e) => setQuery(e.target.value)} 
            className="border p-2 rounded w-full h-32 mb-4"
            placeholder="Enter your query..."
          />
          <button 
            onClick={handleQuery} 
            disabled={!query.trim() || isLoading}
            className={`bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 ${(!query.trim() || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {results.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Results</h2>
            <div className="space-y-4">
              {results.map((result, index) => (
                <div key={index} className="border p-4 rounded bg-gray-50">
                  <p className="mb-2">{result.text || 'No text available'}</p>
                  {result.citations && (
                    <div className="mt-2 text-sm text-gray-600">
                      <strong>Citations:</strong> {result.citations.join(', ')}
                    </div>
                  )}
                  {result.confidence && (
                    <div className="mt-1 text-xs text-gray-500">
                      Confidence: {(result.confidence * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
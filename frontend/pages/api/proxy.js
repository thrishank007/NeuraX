// Next.js API route for proxying requests to backend
export default function handler(req, res) {
  const { method, body, query } = req
  
  // Proxy to backend API
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const targetUrl = `${backendUrl}${req.url.replace('/api/proxy', '')}`
  
  // Forward request to backend
  fetch(targetUrl, {
    method: method,
    headers: {
      'Content-Type': 'application/json',
      ...req.headers
    },
    body: method !== 'GET' ? JSON.stringify(body) : undefined
  })
  .then(response => {
    // Forward response status and headers
    const headers = {}
    response.headers.forEach((value, name) => {
      headers[name] = value
    })
    
    res.status(response.status).json(headers)
  })
  .catch(error => {
    console.error('Proxy error:', error)
    res.status(500).json({ error: 'Proxy request failed' })
  })
}
import { useState } from 'react'

const API_BASE_URL = 'http://localhost:8080'

function App() {
  const [inputText, setInputText] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [chartUrl, setChartUrl] = useState('')
  const [wordcloudUrl, setWordcloudUrl] = useState('')
  const [error, setError] = useState('')

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to analyze')
      return
    }
    
    setError('')
    setLoading(true)
    
    try {
      const comments = inputText.split('\n').filter(c => c.trim())
      
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comments })
      })
      
      if (!response.ok) throw new Error('Prediction failed')
      
      const data = await response.json()
      setResults(data)

      // Calculate sentiment counts
      const counts = { '1': 0, '0': 0, '-1': 0 }
      data.forEach(item => {
        counts[item.sentiment.toString()] += 1
      })

      // Fetch pie chart
      const chartRes = await fetch(`${API_BASE_URL}/generate_chart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentiment_counts: counts })
      })
      if (chartRes.ok) {
        const blob = await chartRes.blob()
        setChartUrl(URL.createObjectURL(blob))
      }

      // Fetch wordcloud
      const wcRes = await fetch(`${API_BASE_URL}/generate_wordcloud`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comments })
      })
      if (wcRes.ok) {
        const blob = await wcRes.blob()
        setWordcloudUrl(URL.createObjectURL(blob))
      }

    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const getSentimentLabel = (score) => {
    switch (parseInt(score)) {
      case 1: return { text: 'Positive', class: 'positive' }
      case -1: return { text: 'Negative', class: 'negative' }
      default: return { text: 'Neutral', class: 'neutral' }
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>SentimentPulse</h1>
        <p>AI-powered sentiment analysis for your YouTube comments</p>
      </header>

      <main>
        <section className="glass-card mb-4">
          <div className="input-group">
            <textarea 
              placeholder="Paste comments here, one per line..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
            {error && <div style={{color: 'var(--danger)', fontSize: '0.9rem'}}>{error}</div>}
            <button className="btn" onClick={handleAnalyze} disabled={loading}>
              {loading ? <span className="loader"></span> : 'Analyze Sentiment'}
            </button>
          </div>
        </section>

        {results.length > 0 && (
          <div className="results-grid">
            <div className="glass-card">
              <h2 style={{marginBottom: '1rem'}}>Analysis Results</h2>
              <ul className="sentiment-list">
                {results.map((res, i) => {
                  const label = getSentimentLabel(res.sentiment)
                  return (
                    <li key={i} className={`sentiment-item ${label.class}`}>
                      <p style={{fontSize: '0.95rem'}}>{res.comment}</p>
                      <div>
                        <span className={`sentiment-badge ${label.class}`}>{label.text}</span>
                      </div>
                    </li>
                  )
                })}
              </ul>
            </div>

            <div className="glass-card chart-container">
              <h2 style={{marginBottom: '1rem'}}>Sentiment Distribution</h2>
              {chartUrl ? <img src={chartUrl} alt="Sentiment Pie Chart" /> : <p>Loading chart...</p>}
            </div>

            {wordcloudUrl && (
              <div className="glass-card chart-container" style={{gridColumn: '1 / -1'}}>
                <h2 style={{marginBottom: '1rem'}}>Word Cloud</h2>
                <img src={wordcloudUrl} alt="Word Cloud" style={{maxWidth: '100%'}} />
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App

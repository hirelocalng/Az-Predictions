import { useState, useEffect, useCallback } from 'react'
import PredictionCard from './PredictionCard.jsx'

export default function DailyTipsSection() {
  const [tips,      setTips]      = useState([])
  const [loading,   setLoading]   = useState(true)
  const [error,     setError]     = useState(false)
  const [fetchedAt, setFetchedAt] = useState(null)

  const load = useCallback(() => {
    setLoading(true)
    setError(false)
    fetch('/api/daily-tips')
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json() })
      .then(d  => { setTips(Array.isArray(d) ? d : []); setFetchedAt(new Date()); setLoading(false) })
      .catch(() => { setError(true); setLoading(false) })
  }, [])

  useEffect(() => { load() }, [load])

  return (
    <section className="section" id="daily-tips">
      <div className="section-header">
        <div className="section-label">Live · Auto-updated</div>
        <h2 className="section-title">
          Today's <span className="green">Best</span> Predictions
        </h2>
        <p className="section-sub">
          Live fixtures from World Cup 2026, Champions League, Premier League, La Liga, Bundesliga,
          Serie A, Ligue 1, Brazil Série A, and international warm-up matches.
          Only predictions with ≥60% model confidence shown, ranked highest first.
        </p>
      </div>

      {!loading && fetchedAt && (
        <div style={{ textAlign: 'center', marginBottom: 20, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            Updated {fetchedAt.toLocaleTimeString()}
          </span>
          <button className="filter-btn" onClick={load} style={{ padding: '4px 12px' }}>
            ↻ Refresh
          </button>
        </div>
      )}

      {loading && (
        <div className="grid-2">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="pred-card">
              <div className="skeleton" style={{ height: 360 }} />
            </div>
          ))}
        </div>
      )}

      {error && (
        <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0' }}>
          Could not load today's fixtures — make sure the Flask server is running on port 5000.
        </p>
      )}

      {!loading && !error && tips.length === 0 && (
        <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0' }}>
          No fixtures scheduled today in the tracked competitions.
        </p>
      )}

      {!loading && !error && tips.length > 0 && (
        <div className="grid-2">
          {tips.map(tip => (
            <PredictionCard key={tip.id} tip={tip} />
          ))}
        </div>
      )}

      <p style={{ textAlign: 'center', marginTop: 36, fontSize: '0.7rem', color: 'var(--text-subtle)', lineHeight: 1.9 }}>
        Stats derived from historical match data · No live odds used · For entertainment purposes only.
        Please gamble responsibly.
      </p>
    </section>
  )
}

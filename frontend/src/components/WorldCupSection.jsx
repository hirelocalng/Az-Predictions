import { useState, useEffect } from 'react'
import PredictionCard from './PredictionCard.jsx'
import CountdownTimer from './CountdownTimer.jsx'

export default function WorldCupSection() {
  const [fixtures, setFixtures] = useState([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(false)

  useEffect(() => {
    fetch('/api/worldcup/fixtures')
      .then(r => r.json())
      .then(d => { setFixtures(d); setLoading(false) })
      .catch(() => { setError(true); setLoading(false) })
  }, [])

  return (
    <section className="section" id="worldcup">
      <div className="section-header">
        <div className="section-label">FIFA World Cup 2026</div>
        <h2 className="section-title">
          Group Stage <span className="amber">Predictions</span>
        </h2>
        <p className="section-sub">
          Powered by 49,000+ international matches. Rolling form, H2H records,
          and tournament importance weighting.
        </p>
      </div>

      <CountdownTimer />

      {loading && (
        <div className="grid-3">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="pred-card">
              <div className="skeleton" style={{ height: 260 }} />
            </div>
          ))}
        </div>
      )}

      {error && (
        <p style={{ textAlign:'center', color:'var(--text-muted)', padding:'40px 0' }}>
          Could not load fixtures — ensure the Flask server is running on port 5000.
        </p>
      )}

      {!loading && !error && (
        <div className="grid-3">
          {fixtures.map(f => (
            <PredictionCard key={f.id} tip={f} isWC />
          ))}
        </div>
      )}
    </section>
  )
}

import { useState, useEffect } from 'react'
import PredictionCard from './PredictionCard.jsx'

export default function IntlSection() {
  const [fixtures, setFixtures] = useState([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(false)

  useEffect(() => {
    fetch('/api/intl/fixtures')
      .then(r => r.json())
      .then(d => { setFixtures(d); setLoading(false) })
      .catch(() => { setError(true); setLoading(false) })
  }, [])

  return (
    <section className="section" id="intl">
      <div className="section-header">
        <div className="section-label">Today's International Matches</div>
        <h2 className="section-title">
          International <span className="green">Predictions</span>
        </h2>
        <p className="section-sub">
          All of today's international fixtures — friendlies, qualifiers, and
          tournaments — run through XGBoost models trained on 49,000+ matches.
        </p>
      </div>

      {loading && (
        <div className="grid-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="pred-card">
              <div className="skeleton" style={{ height: 260 }} />
            </div>
          ))}
        </div>
      )}

      {error && (
        <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0' }}>
          Could not load fixtures — ensure the Flask server is running.
        </p>
      )}

      {!loading && !error && fixtures.length === 0 && (
        <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0' }}>
          No international matches today — check back tomorrow.
        </p>
      )}

      {!loading && !error && fixtures.length > 0 && (
        <div className="grid-3">
          {fixtures.map(f => (
            <PredictionCard key={f.id} tip={f} />
          ))}
        </div>
      )}
    </section>
  )
}

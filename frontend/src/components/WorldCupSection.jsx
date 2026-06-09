import { useState, useEffect } from 'react'
import PredictionCard from './PredictionCard.jsx'
import CountdownTimer from './CountdownTimer.jsx'
import PremiumGate from './PremiumGate.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'

export default function WorldCupSection() {
  const { auth }   = useAuth()
  const isPremium  = Boolean(auth?.user?.is_premium)

  const [fixtures, setFixtures] = useState([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(false)

  useEffect(() => {
    if (!isPremium) { setLoading(false); return }
    fetch('/api/worldcup/fixtures')
      .then(r => r.json())
      .then(d => { setFixtures(d); setLoading(false) })
      .catch(() => { setError(true); setLoading(false) })
  }, [isPremium])

  return (
    <section className="section" id="worldcup">
      <div className="section-header">
        <div className="section-label">FIFA World Cup 2026</div>
        <h2 className="section-title">
          Group Stage <span className="amber">Predictions</span>
        </h2>
      </div>

      <CountdownTimer />

      {!isPremium ? <PremiumGate /> : (
        <>
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
              Could not load fixtures.
            </p>
          )}
          {!loading && !error && fixtures.length === 0 && (
            <p style={{ textAlign:'center', color:'var(--text-muted)', padding:'40px 0' }}>
              No upcoming fixtures for this round — check back soon.
            </p>
          )}
          {!loading && !error && fixtures.length > 0 && (
            <div className="grid-3">
              {fixtures.map(f => <PredictionCard key={f.id} tip={f} isWC />)}
            </div>
          )}
        </>
      )}
    </section>
  )
}

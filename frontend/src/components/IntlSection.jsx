import { useState, useEffect } from 'react'
import PredictionCard from './PredictionCard.jsx'
import PremiumGate from './PremiumGate.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'

const FREE_LIMIT = 5

export default function IntlSection() {
  const { auth }  = useAuth()
  const isPremium = Boolean(auth?.user?.is_premium)

  const [fixtures, setFixtures] = useState([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(false)

  useEffect(() => {
    fetch('/api/intl/fixtures')
      .then(r => r.json())
      .then(d => { setFixtures(d); setLoading(false) })
      .catch(() => { setError(true); setLoading(false) })
  }, [])

  const isFinished = f => f.utc_kickoff
    ? Date.now() > new Date(f.utc_kickoff).getTime() + 2 * 60 * 60 * 1000
    : false
  const upcoming = fixtures.filter(f => !isFinished(f))
  const free   = upcoming.slice(0, FREE_LIMIT)
  const locked = upcoming.slice(FREE_LIMIT)

  return (
    <section className="section" id="intl">
      <div className="section-header">
        <div className="section-label">Today's International Matches</div>
        <h2 className="section-title">
          International <span className="green">Friendlies</span> &amp; Fixtures
        </h2>
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
        <p style={{ textAlign:'center', color:'var(--text-muted)', padding:'40px 0' }}>
          Could not load fixtures — ensure the Flask server is running.
        </p>
      )}

      {!loading && !error && upcoming.length === 0 && (
        <p style={{ textAlign:'center', color:'var(--text-muted)', padding:'40px 0' }}>
          {fixtures.length > 0
            ? "All international fixtures today have finished — check History for results."
            : "No international matches today — check back tomorrow."}
        </p>
      )}

      {!loading && !error && upcoming.length > 0 && (
        <>
          <div className="grid-3">
            {free.map(f => <PredictionCard key={f.id} tip={f} />)}
          </div>

          {locked.length > 0 && !isPremium && (
            <div style={{ marginTop: 32 }}>
              <PremiumGate lockedCount={locked.length} />
            </div>
          )}

          {locked.length > 0 && isPremium && (
            <div className="grid-3" style={{ marginTop: 20 }}>
              {locked.map(f => <PredictionCard key={f.id} tip={f} />)}
            </div>
          )}
        </>
      )}
    </section>
  )
}

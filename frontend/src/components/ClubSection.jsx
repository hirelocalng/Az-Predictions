import { useState, useEffect } from 'react'
import PredictionCard from './PredictionCard.jsx'
import PremiumGate from './PremiumGate.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'

const FREE_LIMIT = 5

export default function ClubSection() {
  const { auth }  = useAuth()
  const isPremium = Boolean(auth?.user?.is_premium)

  const [tips,    setTips]    = useState([])
  const [loading, setLoading] = useState(true)
  const [filter,  setFilter]  = useState('All')
  const [error,   setError]   = useState(false)

  useEffect(() => {
    fetch('/api/club/tips')
      .then(r => r.json())
      .then(d => { setTips(d); setLoading(false) })
      .catch(() => { setError(true); setLoading(false) })
  }, [])

  const isFinished = t => t.utc_kickoff
    ? Date.now() > new Date(t.utc_kickoff).getTime() + 2 * 60 * 60 * 1000
    : false
  const leagues = ['All', ...new Set(tips.map(t => t.league))]
  const filtered = (filter === 'All' ? tips : tips.filter(t => t.league === filter))
    .filter(t => !isFinished(t))
  const free     = filtered.slice(0, FREE_LIMIT)
  const locked   = filtered.slice(FREE_LIMIT)

  return (
    <section className="section" id="club" style={{ paddingTop: 0 }}>
      <div className="section-header">
        <div className="section-label">Club Football</div>
        <h2 className="section-title">
          Daily <span className="green">Tips</span>
        </h2>
      </div>

      {!loading && tips.length > 0 && (
        <div className="filter-row">
          {leagues.map(l => (
            <button key={l} className={`filter-btn${filter === l ? ' active' : ''}`}
                    onClick={() => setFilter(l)}>{l}</button>
          ))}
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
        <p style={{ textAlign:'center', color:'var(--text-muted)', padding:'40px 0' }}>
          Could not load tips.
        </p>
      )}

      {!loading && !error && (
        <>
          <div className="grid-2">
            {free.map(tip => <PredictionCard key={tip.id} tip={tip} />)}
          </div>

          {locked.length > 0 && !isPremium && (
            <div style={{ marginTop: 32 }}>
              <PremiumGate lockedCount={locked.length} />
            </div>
          )}

          {locked.length > 0 && isPremium && (
            <div className="grid-2" style={{ marginTop: 20 }}>
              {locked.map(tip => <PredictionCard key={tip.id} tip={tip} />)}
            </div>
          )}
        </>
      )}

      <p style={{ textAlign:'center', marginTop:36, fontSize:'0.7rem', color:'var(--text-subtle)', lineHeight:1.9 }}>
        Model outputs only · Not financial advice · Accuracy: Result ~50% · Goals ~56% · Corners ~58%
      </p>
    </section>
  )
}

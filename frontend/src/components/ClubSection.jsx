import { useState, useEffect } from 'react'
import PredictionCard from './PredictionCard.jsx'

export default function ClubSection() {
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

  const leagues = ['All', ...new Set(tips.map(t => t.league))]
  const visible = filter === 'All' ? tips : tips.filter(t => t.league === filter)

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
            <button
              key={l}
              className={`filter-btn${filter === l ? ' active' : ''}`}
              onClick={() => setFilter(l)}
            >
              {l}
            </button>
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
          Could not load tips — ensure the Flask server is running on port 5000.
        </p>
      )}

      {!loading && !error && (
        <div className="grid-2">
          {visible.map(tip => (
            <PredictionCard key={tip.id} tip={tip} />
          ))}
        </div>
      )}

      <p style={{ textAlign:'center', marginTop:36, fontSize:'0.7rem', color:'var(--text-subtle)', lineHeight:1.9 }}>
        Model outputs only — not financial advice. Accuracy: Match Result ~50% · Goals ~56% · Corners ~58%.
        Please gamble responsibly.
      </p>
    </section>
  )
}

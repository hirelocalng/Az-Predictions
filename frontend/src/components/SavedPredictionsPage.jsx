import { useState, useEffect } from 'react'

function AuthHeader({ navigate }) {
  return (
    <header className="auth-header">
      <button className="auth-back" onClick={() => navigate('/')}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <path d="M19 12H5M12 5l-7 7 7 7" />
        </svg>
        Back
      </button>
      <div className="auth-logo">⚽ Az-Predictions</div>
    </header>
  )
}

function StatusBadge({ status }) {
  if (status === 'won')  return <span className="saved-badge saved-badge--won">✓ WON</span>
  if (status === 'lost') return <span className="saved-badge saved-badge--lost">✗ LOST</span>
  return <span className="saved-badge saved-badge--pending">⏳ Pending</span>
}

function PredictionRow({ prediction }) {
  const {
    home_team, away_team, competition, match_date, match_time,
    predicted_winner, predicted_goals, predicted_btts,
    status, match_result, saved_at,
  } = prediction

  return (
    <div className={`saved-row saved-row--${status}`}>
      <div className="saved-row-top">
        <span className="saved-row-league">{competition || 'Football'}</span>
        <StatusBadge status={status} />
      </div>
      <div className="saved-row-teams">
        {home_team} <span className="saved-row-vs">vs</span> {away_team}
      </div>
      {match_result && status !== 'pending' && (
        <div className="saved-row-result">Final: {match_result}</div>
      )}
      <div className="saved-row-meta">
        <span>{match_date}{match_time ? ` · ${match_time}` : ''}</span>
        {predicted_winner && <span>Pick: <strong>{predicted_winner}</strong></span>}
        {predicted_goals  && <span>{predicted_goals}</span>}
        {predicted_btts   && <span>BTTS {predicted_btts}</span>}
      </div>
    </div>
  )
}

const FILTERS = ['All', 'Pending', 'Won', 'Lost']

export default function SavedPredictionsPage({ navigate, auth }) {
  const [data,    setData]    = useState(null)
  const [filter,  setFilter]  = useState('All')
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  useEffect(() => {
    if (!auth?.token) { navigate('/login'); return }
    fetch('/api/user/saved-predictions', {
      headers: { Authorization: `Bearer ${auth.token}` },
    })
      .then(r => r.json())
      .then(d => {
        if (d.error) setError(d.error)
        else setData(d)
      })
      .catch(() => setError('Failed to load saved predictions.'))
      .finally(() => setLoading(false))
  }, [auth?.token])

  const filtered = data?.predictions?.filter(p => {
    if (filter === 'All')     return true
    if (filter === 'Pending') return p.status === 'pending'
    if (filter === 'Won')     return p.status === 'won'
    if (filter === 'Lost')    return p.status === 'lost'
    return true
  }) ?? []

  const hasLost = filtered.some(p => p.status === 'lost')

  return (
    <div className="auth-page">
      <AuthHeader navigate={navigate} />
      <div className="saved-page">
        <h1 className="saved-title">Your Saved Predictions</h1>

        {loading && <div className="saved-loading">Loading…</div>}
        {error   && <div className="auth-error">⚠ {error}</div>}

        {data && (
          <>
            {/* Stats row */}
            <div className="saved-stats">
              <div className="saved-stat">
                <div className="saved-stat-num">{data.stats.total}</div>
                <div className="saved-stat-label">Saved</div>
              </div>
              <div className="saved-stat saved-stat--green">
                <div className="saved-stat-num">{data.stats.win_rate}%</div>
                <div className="saved-stat-label">Win rate</div>
              </div>
              <div className="saved-stat">
                <div className="saved-stat-num green">{data.stats.won}</div>
                <div className="saved-stat-label">Won</div>
              </div>
              <div className="saved-stat">
                <div className="saved-stat-num" style={{ color: 'var(--red)' }}>{data.stats.lost}</div>
                <div className="saved-stat-label">Lost</div>
              </div>
              <div className="saved-stat">
                <div className="saved-stat-num" style={{ color: 'var(--text-muted)' }}>{data.stats.pending}</div>
                <div className="saved-stat-label">Pending</div>
              </div>
            </div>

            {/* Filter tabs */}
            <div className="saved-tabs">
              {FILTERS.map(f => (
                <button
                  key={f}
                  className={`saved-tab${filter === f ? ' saved-tab--active' : ''}`}
                  onClick={() => setFilter(f)}
                >
                  {f}
                </button>
              ))}
            </div>

            {/* List */}
            {filtered.length === 0 ? (
              <div className="saved-empty">
                {filter === 'All'
                  ? "You haven't saved any predictions yet. Tap the heart icon on any prediction card."
                  : `No ${filter.toLowerCase()} predictions.`}
              </div>
            ) : (
              <div className="saved-list">
                {filtered.map(p => (
                  <PredictionRow key={p.id} prediction={p} />
                ))}
              </div>
            )}

            {/* Premium banner */}
            {hasLost && (
              <div className="saved-premium-banner">
                <div className="saved-premium-icon">👑</div>
                <div>
                  <div className="saved-premium-title">Get alerts 15 min before your saved bets start</div>
                  <div className="saved-premium-sub">Never miss a kick-off · ₦5,000/month</div>
                </div>
                <button className="btn-primary" onClick={() => navigate('/subscribe')}>
                  Go Premium
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

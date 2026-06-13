import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext.jsx'

function StatCard({ value, label, color }) {
  return (
    <div className="res-stat-card">
      <div className={`res-stat-num${color ? ' ' + color : ''}`}>{value}</div>
      <div className="res-stat-label">{label}</div>
    </div>
  )
}

function SubChip({ label, value, status }) {
  const cls  = status === 'WON' ? 'won' : status === 'LOST' ? 'lost' : 'na'
  const icon = status === 'WON' ? '✅' : status === 'LOST' ? '❌' : null
  return (
    <span className={`res-sub-chip ${cls}`}>
      <span className="res-sub-label">{label}</span>
      <span className="res-sub-val">{value}</span>
      {icon
        ? <span className="res-sub-icon">{icon}</span>
        : <span className="res-sub-nodata">No data</span>}
    </span>
  )
}

function ResultRow({ pred, liveOverlay }) {
  const status   = pred.result_status
  const isLive   = status === 'LIVE'
  const won      = status === 'WON'
  const resolved = status === 'WON' || status === 'LOST'
  const sub      = pred.sub_results

  const date = pred.match_date
    ? new Date(pred.match_date + 'T12:00:00Z').toLocaleDateString('en-GB', {
        day: 'numeric', month: 'short', timeZone: 'UTC',
      })
    : '—'

  // Live overlay takes priority, then actual score, then em-dash
  const live   = liveOverlay || {}
  const liveHs = live.live_home_score ?? pred.live_home_score
  const liveAs = live.live_away_score ?? pred.live_away_score
  const liveMin = live.live_minute   ?? pred.live_minute

  let scoreLabel, scoreClass
  if (isLive && liveHs != null && liveAs != null) {
    const min = liveMin ? ` ${liveMin}'` : ''
    scoreLabel = `${liveHs}–${liveAs}${min}`
    scoreClass = 'res-score-live'
  } else if (pred.actual_home_score != null && pred.actual_away_score != null) {
    scoreLabel = `${pred.actual_home_score}–${pred.actual_away_score}`
    scoreClass = ''
  } else {
    scoreLabel = '—'
    scoreClass = ''
  }

  return (
    <div className={`res-row ${won ? 'res-row-won' : resolved ? 'res-row-lost' : isLive ? 'res-row-live' : 'res-row-pending'}`}>
      <div className="res-row-badge">
        {won      ? <span className="res-badge won">✅ WON</span>
         : resolved ? <span className="res-badge lost">❌ LOST</span>
         : isLive  ? <span className="res-badge live">🔴 LIVE</span>
                   : <span className="res-badge pending">⏳ PENDING</span>}
      </div>

      <div className="res-row-teams">
        <span className="res-home">{pred.home_team}</span>
        <span className="res-vs">vs</span>
        <span className="res-away">{pred.away_team}</span>
      </div>

      <div className="res-row-comp">
        {(pred.sport === 'nba' || pred.sport === 'wnba')
          ? <span className="res-sport-badge">🏀 {pred.competition}</span>
          : pred.competition}
      </div>

      <div className="res-row-pred">
        <span className="res-pred-label">Best Bet</span>
        <span className="res-pred-val">{pred.predicted_winner || '—'}</span>
      </div>

      <div className="res-row-score">
        <span className="res-score-label">Score</span>
        <span className={`res-score-val ${scoreClass}`}>{scoreLabel}</span>
      </div>

      <div className="res-row-date">{date}</div>

      {resolved && sub && (
        <div className="res-sub-row">
          {sub.result !== undefined && pred.predicted_winner && (
            <SubChip label="Result" value={pred.predicted_winner} status={sub.result} />
          )}
          {sub.goals !== undefined && pred.predicted_goals && (
            <SubChip
              label={pred.sport === 'nba' || pred.sport === 'wnba' ? 'O/U' : 'Goals'}
              value={pred.predicted_goals}
              status={sub.goals}
            />
          )}
          {sub.btts !== undefined && pred.predicted_btts && (
            <SubChip label="BTTS" value={`BTTS ${pred.predicted_btts}`} status={sub.btts} />
          )}
          {sub.corners !== undefined && pred.predicted_corners && (
            <SubChip label="Corners" value={pred.predicted_corners} status={sub.corners} />
          )}
        </div>
      )}
    </div>
  )
}

function Skeleton() {
  return (
    <div className="res-skeleton">
      {[1, 2, 3, 4, 5].map(i => (
        <div key={i} className="res-row-skel shimmer" />
      ))}
    </div>
  )
}

const FULL_REFRESH_MS  = 60_000   // re-fetch entire results list every 60 s
const LIVE_REFRESH_MS  = 10_000   // poll /api/live-scores every 10 s when a LIVE game exists

export default function ResultsSection() {
  const { auth } = useAuth()
  const [data,      setData]      = useState(null)
  const [liveMap,   setLiveMap]   = useState({})   // match_id → {live_home_score, live_away_score, live_minute}
  const [error,     setError]     = useState(null)
  const [loading,   setLoading]   = useState(true)
  const [showList,  setShowList]  = useState(false)
  const [lastAt,    setLastAt]    = useState(null)

  const authRef = useRef(auth)
  useEffect(() => { authRef.current = auth }, [auth])

  const fetchFull = useCallback(() => {
    const headers = authRef.current?.token
      ? { Authorization: `Bearer ${authRef.current.token}` }
      : {}
    fetch('/api/results', { headers })
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json() })
      .then(d => { setData(d); setError(null); setLoading(false); setLastAt(new Date()) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  const fetchLive = useCallback(() => {
    fetch('/api/live-scores')
      .then(r => r.json())
      .then(d => { if (d.live) setLiveMap(d.live) })
      .catch(() => {})
  }, [])

  // Full load on mount + every 60 s
  useEffect(() => {
    fetchFull()
    const id = setInterval(fetchFull, FULL_REFRESH_MS)
    return () => clearInterval(id)
  }, [fetchFull])

  // Live-score overlay every 10 s (only when at least one LIVE game is present)
  const hasLive = (data?.predictions || []).some(p => p.result_status === 'LIVE')
  useEffect(() => {
    if (!hasLive) return
    fetchLive()
    const id = setInterval(fetchLive, LIVE_REFRESH_MS)
    return () => clearInterval(id)
  }, [hasLive, fetchLive])

  const stats  = data?.stats  || {}
  const preds  = data?.predictions || []
  const streak = stats.streak > 0
    ? `${stats.streak}${stats.streak_type === 'WON' ? 'W' : 'L'} streak`
    : null

  return (
    <section className="section" id="history">
      <div className="section-header">
        <div className="section-label">Track Record</div>
        <h2 className="section-title">
          Prediction <span className="green">History</span>
        </h2>
        <p className="section-sub">
          Every prediction is saved automatically and verified against real results.
        </p>
      </div>

      {/* Stats bar */}
      <div className="res-stats-row">
        <StatCard
          value={stats.win_rate != null ? `${stats.win_rate}%` : '—'}
          label="Win Rate"
          color={stats.win_rate >= 55 ? 'green' : stats.win_rate >= 40 ? 'amber' : ''}
        />
        <StatCard value={stats.total ?? 0} label="Total Bets" />
        <StatCard value={stats.won  ?? 0}  label="Won" color="green" />
        {streak && (
          <StatCard
            value={streak}
            label="Current Streak"
            color={stats.streak_type === 'WON' ? 'green' : 'red'}
          />
        )}
      </div>

      {/* Live indicator + last-updated timestamp */}
      {!loading && (
        <div style={{ textAlign: 'center', marginBottom: 12, display: 'flex',
                      alignItems: 'center', justifyContent: 'center', gap: 12 }}>
          {hasLive && (
            <span style={{ fontSize: '0.7rem', color: 'var(--red)',
                           display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 7, height: 7, borderRadius: '50%',
                             background: 'var(--red)', display: 'inline-block',
                             animation: 'pulse 1.4s infinite' }} />
              LIVE — scores update every 10 s
            </span>
          )}
          {lastAt && (
            <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
              Updated {lastAt.toLocaleTimeString()}
            </span>
          )}
          <button className="filter-btn" onClick={fetchFull}
                  style={{ padding: '3px 10px', fontSize: '0.72rem' }}>
            ↻ Refresh
          </button>
        </div>
      )}

      {loading && <Skeleton />}

      {error && (
        <div className="res-empty">
          <div className="res-empty-icon">⚠</div>
          <p>Could not load history — {error}</p>
        </div>
      )}

      {!loading && !error && preds.length === 0 && (
        <div className="res-empty">
          <div className="res-empty-icon">📋</div>
          <p>No completed predictions yet.</p>
          <p className="res-empty-sub">
            Results appear here automatically once matches finish.
          </p>
        </div>
      )}

      {!loading && !error && preds.length > 0 && (
        <button className="res-mobile-toggle" onClick={() => setShowList(v => !v)}>
          {showList ? 'Hide predictions ▲' : `View ${preds.length} predictions ▼`}
        </button>
      )}

      {!loading && preds.length > 0 && (
        <div className={`res-list${showList ? ' res-list-open' : ''}`}>
          {preds.map(p => (
            <ResultRow
              key={p.match_id}
              pred={p}
              liveOverlay={liveMap[p.match_id] || null}
            />
          ))}
        </div>
      )}

      <p className="footer-text">
        Scores from TheSportsDB &amp; ESPN · Live scores refresh every 10 s · Results auto-resolve every 2 min
      </p>
    </section>
  )
}

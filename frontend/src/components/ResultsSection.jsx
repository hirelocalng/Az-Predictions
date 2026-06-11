import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext.jsx'

function StatCard({ value, label, sub, color }) {
  return (
    <div className="res-stat-card">
      <div className={`res-stat-num${color ? ' ' + color : ''}`}>{value}</div>
      <div className="res-stat-label">{label}</div>
      {sub && <div className="res-stat-sub">{sub}</div>}
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

function ResultRow({ pred }) {
  const status   = pred.result_status
  const won      = status === 'WON'
  const resolved = status === 'WON' || status === 'LOST'
  const sub      = pred.sub_results   // { result, goals, btts, corners } or null

  const date = pred.match_date
    ? new Date(pred.match_date + 'T12:00:00Z').toLocaleDateString('en-GB', {
        day: 'numeric', month: 'short', timeZone: 'UTC',
      })
    : '—'

  const score = pred.actual_home_score != null && pred.actual_away_score != null
    ? `${pred.actual_home_score}–${pred.actual_away_score}`
    : '—'

  return (
    <div className={`res-row ${won ? 'res-row-won' : resolved ? 'res-row-lost' : 'res-row-pending'}`}>
      {/* Top row */}
      <div className="res-row-badge">
        {won      ? <span className="res-badge won">✅ WON</span>
         : resolved ? <span className="res-badge lost">❌ LOST</span>
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
        <span className="res-score-val">{score}</span>
      </div>

      <div className="res-row-date">{date}</div>

      {/* Per-market breakdown — only for finished matches with score data */}
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

export default function ResultsSection() {
  const { auth } = useAuth()
  const [data, setData]         = useState(null)
  const [error, setError]       = useState(null)
  const [loading, setLoading]   = useState(true)
  const [showList, setShowList] = useState(false)

  useEffect(() => {
    const headers = auth?.token
      ? { Authorization: `Bearer ${auth.token}` }
      : {}
    fetch('/api/results', { headers })
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [auth?.token])

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
        <StatCard value={stats.total ?? 0}  label="Total Bets" />
        <StatCard value={stats.won  ?? 0}   label="Won"   color="green" />
        {streak && (
          <StatCard
            value={streak}
            label="Current Streak"
            color={stats.streak_type === 'WON' ? 'green' : 'red'}
          />
        )}
      </div>

      {/* Results list */}
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

      {/* Mobile toggle button — only visible on small screens */}
      {!loading && !error && preds.length > 0 && (
        <button
          className="res-mobile-toggle"
          onClick={() => setShowList(v => !v)}
        >
          {showList ? 'Hide predictions ▲' : `View ${preds.length} predictions ▼`}
        </button>
      )}

      {!loading && preds.length > 0 && (
        <div className={`res-list${showList ? ' res-list-open' : ''}`}>
          {preds.map(p => <ResultRow key={p.match_id} pred={p} />)}
        </div>
      )}

      <p className="footer-text">
        Results fetched automatically from TheSportsDB &amp; ESPN · Live updates every 2 minutes
      </p>
    </section>
  )
}

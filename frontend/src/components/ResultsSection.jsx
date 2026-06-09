import { useState, useEffect } from 'react'

function StatCard({ value, label, sub, color }) {
  return (
    <div className="res-stat-card">
      <div className={`res-stat-num${color ? ' ' + color : ''}`}>{value}</div>
      <div className="res-stat-label">{label}</div>
      {sub && <div className="res-stat-sub">{sub}</div>}
    </div>
  )
}

function ResultRow({ pred }) {
  const status   = pred.result_status
  const won      = status === 'WON'
  const resolved = status === 'WON' || status === 'LOST'
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

      <div className="res-row-comp">{pred.competition}</div>

      <div className="res-row-pred">
        <span className="res-pred-label">Prediction</span>
        <span className="res-pred-val">{pred.predicted_winner || '—'}</span>
      </div>

      <div className="res-row-score">
        <span className="res-score-label">Score</span>
        <span className="res-score-val">{score}</span>
      </div>

      <div className="res-row-date">{date}</div>
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
  const [data, setData]     = useState(null)
  const [error, setError]   = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/results')
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

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
        <StatCard value={stats.lost ?? 0}   label="Lost"  color="red"   />
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

      {!loading && preds.length > 0 && (
        <div className="res-list">
          {preds.map(p => <ResultRow key={p.match_id} pred={p} />)}
        </div>
      )}

      <p className="footer-text">
        Results fetched automatically from TheSportsDB &amp; ESPN · Live updates every 2 minutes
      </p>
    </section>
  )
}

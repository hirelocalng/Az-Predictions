import { useLiveScores } from '../contexts/LiveScoresContext.jsx'
import SavePredictionButton from './SavePredictionButton.jsx'

function pct(v) { return v != null ? `${(v * 100).toFixed(1)}%` : '—' }

function StatBar({ label, value, color = 'green', extra }) {
  const pctVal = Math.round((value ?? 0) * 100)
  return (
    <div className="stat-row">
      <span className="stat-label">{label}</span>
      <div className="stat-bar-wrap">
        <div
          className="stat-bar-fill"
          style={{ width: `${pctVal}%` }}
          data-color={color}
        />
      </div>
      <span className="stat-val">{pct(value)}{extra ? ` · ${extra}` : ''}</span>
    </div>
  )
}

function ResultBar({ home, draw, away, homeTeam, awayTeam }) {
  const hPct = Math.round((home ?? 0.33) * 100)
  const dPct = Math.round((draw ?? 0.33) * 100)
  const aPct = 100 - hPct - dPct
  return (
    <div title={`${homeTeam} ${hPct}% · Draw ${dPct}% · ${awayTeam} ${aPct}%`}>
      <div style={{ display: 'flex', gap: 2, height: 6, borderRadius: 6, overflow: 'hidden', marginBottom: 4 }}>
        <div style={{ width: `${hPct}%`, background: 'var(--green)', borderRadius: '6px 0 0 6px' }} />
        <div style={{ width: `${dPct}%`, background: 'rgba(255,255,255,0.18)' }} />
        <div style={{ width: `${aPct}%`, background: 'var(--amber)', borderRadius: '0 6px 6px 0' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--text-subtle)' }}>
        <span style={{ color: 'var(--green)', fontWeight: 600 }}>{homeTeam} {hPct}%</span>
        <span>Draw {dPct}%</span>
        <span style={{ color: 'var(--amber)', fontWeight: 600 }}>{awayTeam} {aPct}%</span>
      </div>
    </div>
  )
}

export default function PredictionCard({ tip, isWC }) {
  const liveScores = useLiveScores()

  const home    = tip.home_team || tip.home || ''
  const away    = tip.away_team || tip.away || ''
  const league  = tip.league || tip.competition || ''
  const result  = tip.result || {}
  const homePr  = result.home ?? 0
  const drawPr  = result.draw ?? 0
  const awayPr  = result.away ?? 0
  const goals   = tip.over_goals ?? 0.5
  const btts    = tip.btts ?? 0.5
  const corners = tip.over_corners ?? 0.5
  const bb      = tip.best_bet || {}
  const bbLabel = bb.label || ''
  const bbConf  = bb.confidence ?? bb.prob ?? 0

  // Kickoff
  const kickoffMs = tip.utc_kickoff ? new Date(tip.utc_kickoff).getTime() : null
  const now = Date.now()

  // Live data
  const matchDate = (tip.utc_kickoff ?? '').slice(0, 10) || tip.date || ''
  const matchId   = matchDate && home && away
    ? `${matchDate}/${home.toLowerCase().trim()}/${away.toLowerCase().trim()}`
    : null
  const liveData  = matchId ? liveScores[matchId] : null
  const isLive    = Boolean(liveData) || tip.status === 'IN_PLAY' || tip.status === 'PAUSED'
    || Boolean(kickoffMs && now >= kickoffMs && now < kickoffMs + 120 * 60_000)

  // Time display
  let timeDisplay = '—'
  if (tip.utc_kickoff) {
    const d = new Date(tip.utc_kickoff)
    timeDisplay = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } else if (tip.time) {
    timeDisplay = tip.time
  }

  // Best bet chip color
  const bbIsGoals   = /goal/i.test(bbLabel)
  const bbIsCorners = /corner/i.test(bbLabel)
  const chipClass   = bbIsGoals || bbIsCorners ? 'best-bet-chip amber-chip' : 'best-bet-chip'

  return (
    <div className="pred-card">
      {/* Header */}
      <div className="card-header">
        <span className="card-league" title={league}>{league || 'Football'}</span>
        <div className="card-header-right">
          {isLive
            ? <div className="live-badge">
                <span className="live-pulse" />
                <span className="live-label">LIVE</span>
                {liveData?.live_home_score != null &&
                  <span className="live-score-inline">{liveData.live_home_score}–{liveData.live_away_score}</span>}
                {liveData?.live_minute && <span className="live-minute-inline">{liveData.live_minute}'</span>}
              </div>
            : <span className="card-time">{timeDisplay}</span>
          }
          <SavePredictionButton matchId={matchId} />
        </div>
      </div>

      {/* Teams */}
      <div className="card-teams">
        <span className="team-name" title={home}>{home}</span>
        <span className="card-vs">
          {liveData?.live_home_score != null
            ? <span className="live-score">{liveData.live_home_score} – {liveData.live_away_score}</span>
            : 'vs'
          }
        </span>
        <span className="team-name" style={{ textAlign: 'right' }} title={away}>{away}</span>
      </div>

      {/* Result probability bar */}
      <ResultBar home={homePr} draw={drawPr} away={awayPr} homeTeam={home} awayTeam={away} />

      {/* Stats */}
      <div className="card-stats">
        <StatBar label="Over 2.5" value={goals}   color={goals >= 0.5 ? 'green' : 'red'} />
        <StatBar label="BTTS"     value={btts}    color={btts  >= 0.5 ? 'green' : 'red'} />
        <StatBar label="Corners"  value={corners} color="blue" />
      </div>

      {/* Best bet chip */}
      {bbLabel && (
        <div className={chipClass}>
          <span className="chip-label">Best Bet</span>
          <span className="chip-pick">{bbLabel}</span>
          <span className="chip-conf">{pct(bbConf)}</span>
        </div>
      )}
    </div>
  )
}

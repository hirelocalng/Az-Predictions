import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext.jsx'

// ── Helpers ───────────────────────────────────────────────────────────────────

function dateLabel(dateStr) {
  const now     = new Date()
  const todayMs = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())
  const gameMs  = new Date(dateStr + 'T00:00:00Z').getTime()
  const diff    = Math.round((gameMs - todayMs) / 86_400_000)
  if (diff === 0) return 'Today'
  if (diff === 1) return 'Tomorrow'
  const d = new Date(dateStr + 'T00:00:00Z')
  const weekday = d.toLocaleDateString('en-US', { weekday: 'long', timeZone: 'UTC' })
  const month   = d.toLocaleDateString('en-US', { month: 'short',  timeZone: 'UTC' })
  return `${weekday} ${month} ${d.getUTCDate()}`
}

function groupByDate(games) {
  const map = {}
  for (const g of games) {
    const d = g.date || 'unknown'
    if (!map[d]) map[d] = []
    map[d].push(g)
  }
  return Object.entries(map).sort(([a], [b]) => a.localeCompare(b))
}

function localKickoff(dateStr, timeStr) {
  if (!timeStr || typeof timeStr !== 'string') return null
  const m = timeStr.match(/^(\d{1,2}):(\d{2})/)
  if (!m) return null
  if (m[1] === '0' && m[2] === '00') return null
  if (m[1] === '00' && m[2] === '00') return null
  try {
    const dt = new Date(`${dateStr}T${m[1].padStart(2, '0')}:${m[2]}:00Z`)
    if (isNaN(dt.getTime())) return null
    return new Intl.DateTimeFormat([], {
      hour: 'numeric', minute: '2-digit', timeZoneName: 'short',
    }).format(dt)
  } catch { return null }
}

// ── Probability bar ───────────────────────────────────────────────────────────

function WinBar({ homeTeam, awayTeam, homePct, awayPct }) {
  const homeW = Math.round(homePct)
  const awayW = Math.round(awayPct)
  return (
    <div className="bball-bar-wrap">
      <div className="bball-bar">
        <div className="bball-bar-home" style={{ width: `${homeW}%` }} />
        <div className="bball-bar-away" style={{ width: `${awayW}%` }} />
      </div>
      <div className="bball-bar-labels">
        <span className="bball-bar-home-lbl">
          <span className="bball-bar-pct">{homeW}%</span>
          <span className="bball-bar-team">{homeTeam.split(' ').pop()}</span>
        </span>
        <span className="bball-bar-away-lbl">
          <span className="bball-bar-team">{awayTeam.split(' ').pop()}</span>
          <span className="bball-bar-pct">{awayW}%</span>
        </span>
      </div>
    </div>
  )
}

// ── Single game card ──────────────────────────────────────────────────────────

function GameCard({ game }) {
  const homePct    = game.result?.home * 100 || game.home_win_pct || 50
  const awayPct    = game.result?.away * 100 || game.away_win_pct || 50
  const overPct    = game.over_total * 100 || game.over_pct || 50
  const ouLine     = game.ou_line || (game.sport === 'wnba' ? 155.5 : 220.5)
  const ouLabel    = overPct > 50 ? `Over ${ouLine}` : `Under ${ouLine}`
  const ouConf     = overPct > 50 ? overPct : (100 - overPct)
  const bestBet    = game.best_bet || game.predicted_winner || ''
  const bestBetPct = game.best_bet_type === 'ou' ? ouConf : Math.max(homePct, awayPct)

  const status  = (game.status || '').toLowerCase()
  const isLive  = status.includes('progress') || status.includes('half') || status === 'live' || status === 'inprogress'
  const isDone  = status === 'final' || status === 'ft' || status.includes('final')
  const hasSco  = game.home_score != null && game.away_score != null
  const kickoff = localKickoff(game.date, game.time)

  function Logo({ src, abbr, alt }) {
    const [err, setErr] = useState(false)
    if (err || !src) {
      return <div className="bball-logo-fallback">{abbr?.slice(0, 3) || alt?.slice(0, 3) || '—'}</div>
    }
    return (
      <img src={src} alt={alt} className="bball-logo"
           onError={() => setErr(true)} loading="lazy" />
    )
  }

  return (
    <div className={`bball-card${isDone ? ' bball-done' : isLive ? ' bball-live' : ''}`}>
      <div className="bball-card-header">
        <span className="bball-league">{game.competition}</span>
        <div className="bball-card-header-right">
          {kickoff && !isLive && !isDone && (
            <span className="bball-card-time">{kickoff}</span>
          )}
          {isLive && <span className="bball-live-badge">LIVE</span>}
          {isDone && <span className="bball-done-badge">FINAL</span>}
        </div>
      </div>

      <div className="bball-teams">
        <div className="bball-team home">
          <Logo src={game.home_logo} abbr={game.home_abbr} alt={game.home_team} />
          <span className="bball-team-name">{game.home_team}</span>
          {hasSco && <span className="bball-score">{game.home_score}</span>}
        </div>
        <div className="bball-vs">vs</div>
        <div className="bball-team away">
          {hasSco && <span className="bball-score">{game.away_score}</span>}
          <span className="bball-team-name">{game.away_team}</span>
          <Logo src={game.away_logo} abbr={game.away_abbr} alt={game.away_team} />
        </div>
      </div>

      <WinBar homeTeam={game.home_team} awayTeam={game.away_team}
              homePct={homePct} awayPct={awayPct} />

      <div className="bball-preds">
        <div className="bball-pred-item">
          <span className="bball-pred-label">Winner</span>
          <span className="bball-pred-val">{game.predicted_winner}</span>
          <span className="bball-pred-pct">{Math.max(homePct, awayPct).toFixed(1)}%</span>
        </div>
        <div className="bball-pred-sep" />
        <div className="bball-pred-item">
          <span className="bball-pred-label">Total Points</span>
          <span className="bball-pred-val">{ouLabel}</span>
          <span className="bball-pred-pct">{ouConf.toFixed(1)}%</span>
        </div>
      </div>

      {bestBet && (
        <div className="bball-best-bet">
          <span className="bball-bb-star">★</span>
          <span className="bball-bb-label">Best Bet</span>
          <span className="bball-bb-pick">{bestBet}</span>
          <span className="bball-bb-conf">{bestBetPct.toFixed(1)}%</span>
        </div>
      )}
    </div>
  )
}

// ── Premium teaser (replaces locked cards) ────────────────────────────────────

function PremiumTeaser({ count, onUpgrade }) {
  return (
    <div className="bball-premium-teaser">
      <span className="bball-pt-icon">👑</span>
      <div className="bball-pt-body">
        <span className="bball-pt-title">
          {count} more prediction{count !== 1 ? 's' : ''} hidden
        </span>
        <span className="bball-pt-sub">Upgrade to Premium to unlock all daily picks</span>
      </div>
      <button className="bball-locked-btn" onClick={onUpgrade}>Go Premium</button>
    </div>
  )
}

// ── Skeletons / empty ─────────────────────────────────────────────────────────

function BballSkeleton() {
  return (
    <div className="bball-grid">
      {[1, 2, 3].map(i => <div key={i} className="bball-card bball-skel shimmer" />)}
    </div>
  )
}

function Empty({ title }) {
  return (
    <div className="bball-empty">
      <div className="bball-empty-icon">🏀</div>
      <p>No {title} games in the next 2 days — check back soon.</p>
    </div>
  )
}

// ── Sport section ─────────────────────────────────────────────────────────────

function SportSection({ id, title, games, err, isPremium, onUpgrade }) {
  const loading = games === null && !err

  // Split free vs premium-only — free_tier undefined treated as free
  const visibleGames = isPremium
    ? (games || [])
    : (games || []).filter(g => g.free_tier !== false)
  const lockedCount = isPremium
    ? 0
    : (games || []).filter(g => g.free_tier === false).length

  return (
    <div className="bball-sport-section" id={id}>
      <div className="bball-sport-header">
        <span className="bball-sport-name">🏀 {title}</span>
        {visibleGames.length > 0 && (
          <span className="bball-sport-count">
            {visibleGames.length} game{visibleGames.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {loading && <BballSkeleton />}

      {err && (
        <div className="bball-empty">
          <div className="bball-empty-icon">⚠</div>
          <p>Could not load {title} games — {err}</p>
        </div>
      )}

      {!loading && !err && visibleGames.length === 0 && lockedCount === 0 && (
        <Empty title={title} />
      )}

      {!loading && !err && visibleGames.length > 0 && (
        <div className="bball-days">
          {groupByDate(visibleGames).map(([date, dayGames]) => (
            <div key={date} className="bball-day-section">
              <div className="bball-day-header">{dateLabel(date)}</div>
              <div className="bball-grid">
                {dayGames.map((g, i) => <GameCard key={g.id || i} game={g} />)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Single teaser — no individual locked cards shown */}
      {!loading && !isPremium && lockedCount > 0 && (
        <PremiumTeaser count={lockedCount} onUpgrade={onUpgrade} />
      )}
    </div>
  )
}

// ── Main section ──────────────────────────────────────────────────────────────

export default function BasketballSection() {
  const { isPremium, navigate } = useAuth()
  const [nbaGames,  setNbaGames]  = useState(null)
  const [wnbaGames, setWnbaGames] = useState(null)
  const [nbaErr,    setNbaErr]    = useState(null)
  const [wnbaErr,   setWnbaErr]   = useState(null)

  const isFinished = g => {
    const s = (g.status || '').toLowerCase()
    return s === 'final' || s === 'ft' || s === 'finished' || s.includes('final')
  }

  useEffect(() => {
    fetch('/api/nba/fixtures')
      .then(r => r.json())
      .then(d => setNbaGames(Array.isArray(d) ? d.filter(g => !isFinished(g)) : []))
      .catch(e => setNbaErr(e.message))
  }, [])

  useEffect(() => {
    fetch('/api/wnba/fixtures')
      .then(r => r.json())
      .then(d => setWnbaGames(Array.isArray(d) ? d.filter(g => !isFinished(g)) : []))
      .catch(e => setWnbaErr(e.message))
  }, [])

  const scrollTo = id => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  const goUpgrade = () => navigate('/subscribe')

  return (
    <section className="section" id="basketball">
      <div className="section-header">
        <div className="section-label">Basketball Predictions</div>
        <h2 className="section-title">
          NBA &amp; <span className="green">WNBA</span>
        </h2>
        {!isPremium && (
          <p className="bball-free-note">
            Free: 2 picks/day · <button className="bball-free-upgrade" onClick={goUpgrade}>Go Premium</button> for all picks
          </p>
        )}
      </div>

      <div className="bball-tabs">
        <button className="bball-tab" onClick={() => scrollTo('nba-section')}>🏀 NBA</button>
        <button className="bball-tab" onClick={() => scrollTo('wnba-section')}>🏀 WNBA</button>
      </div>

      <SportSection id="nba-section"  title="NBA"  games={nbaGames}  err={nbaErr}
                    isPremium={isPremium} onUpgrade={goUpgrade} />

      <div className="bball-sport-divider" />

      <SportSection id="wnba-section" title="WNBA" games={wnbaGames} err={wnbaErr}
                    isPremium={isPremium} onUpgrade={goUpgrade} />

      <p className="footer-text">
        NBA via TheSportsDB · WNBA via ESPN · predictions updated every 10 minutes
      </p>
    </section>
  )
}

import { useState, useEffect } from 'react'

// ── Probability bar (home/away, no draw) ──────────────────────────────────────

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
  const homePct   = game.result?.home * 100 || game.home_win_pct || 50
  const awayPct   = game.result?.away * 100 || game.away_win_pct || 50
  const overPct   = game.over_total * 100 || game.over_pct || 50
  const ouLine    = game.ou_line || (game.sport === 'wnba' ? 170.5 : 220.5)
  const ouLabel   = overPct > 50 ? `Over ${ouLine}` : `Under ${ouLine}`
  const ouConf    = overPct > 50 ? overPct : (100 - overPct)
  const bestBet   = game.best_bet || game.predicted_winner || ''
  const bestBetPct = game.best_bet_type === 'ou' ? ouConf : Math.max(homePct, awayPct)

  const status  = (game.status || '').toLowerCase()
  const isLive  = status.includes('progress') || status.includes('half') || status === 'live'
  const isDone  = status === 'final' || status === 'ft' || status.includes('final')
  const hasSco  = game.home_score != null && game.away_score != null

  // Logo fallback: render initials if image errors
  function Logo({ src, abbr, alt }) {
    const [err, setErr] = useState(false)
    if (err || !src) {
      return (
        <div className="bball-logo-fallback">
          {abbr?.slice(0, 3) || alt?.slice(0, 3) || '—'}
        </div>
      )
    }
    return (
      <img
        src={src} alt={alt}
        className="bball-logo"
        onError={() => setErr(true)}
        loading="lazy"
      />
    )
  }

  return (
    <div className={`bball-card${isDone ? ' bball-done' : isLive ? ' bball-live' : ''}`}>
      {/* Header row */}
      <div className="bball-card-header">
        <span className="bball-league">{game.competition}</span>
        {isLive && <span className="bball-live-badge">LIVE</span>}
        {isDone && <span className="bball-done-badge">FINAL</span>}
      </div>

      {/* Teams row */}
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

      {/* Win probability bar */}
      <WinBar
        homeTeam={game.home_team} awayTeam={game.away_team}
        homePct={homePct} awayPct={awayPct}
      />

      {/* Predictions row */}
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

      {/* Best bet badge */}
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

// ── Loading skeleton ──────────────────────────────────────────────────────────

function BballSkeleton() {
  return (
    <div className="bball-grid">
      {[1, 2, 3].map(i => (
        <div key={i} className="bball-card bball-skel shimmer" />
      ))}
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────

function Empty({ sport }) {
  const label = sport === 'nba' ? 'NBA' : 'WNBA'
  return (
    <div className="bball-empty">
      <div className="bball-empty-icon">🏀</div>
      <p>No {label} games today — check back tomorrow.</p>
    </div>
  )
}

// ── Main section ──────────────────────────────────────────────────────────────

export default function BasketballSection({ activeNav }) {
  const [tab,      setTab]      = useState('nba')

  useEffect(() => {
    if (activeNav === 'nba' || activeNav === 'wnba') setTab(activeNav)
  }, [activeNav])

  const [nbaGames, setNbaGames] = useState(null)
  const [wnbaGames,setWnbaGames]= useState(null)
  const [nbaErr,   setNbaErr]   = useState(null)
  const [wnbaErr,  setWnbaErr]  = useState(null)

  useEffect(() => {
    fetch('/api/nba/fixtures')
      .then(r => r.json())
      .then(d => setNbaGames(Array.isArray(d) ? d : []))
      .catch(e => setNbaErr(e.message))
  }, [])

  useEffect(() => {
    fetch('/api/wnba/fixtures')
      .then(r => r.json())
      .then(d => setWnbaGames(Array.isArray(d) ? d : []))
      .catch(e => setWnbaErr(e.message))
  }, [])

  const games   = tab === 'nba' ? nbaGames : wnbaGames
  const err     = tab === 'nba' ? nbaErr   : wnbaErr
  const loading = games === null && !err

  return (
    <section className="section" id="basketball">
      <div className="section-header">
        <div className="section-label">Basketball Predictions</div>
        <h2 className="section-title">
          NBA &amp; <span className="green">WNBA</span>
        </h2>
      </div>

      {/* Sport tabs */}
      <div className="bball-tabs">
        <button
          className={`bball-tab${tab === 'nba' ? ' active' : ''}`}
          onClick={() => setTab('nba')}
        >
          🏀 NBA
        </button>
        <button
          className={`bball-tab${tab === 'wnba' ? ' active' : ''}`}
          onClick={() => setTab('wnba')}
        >
          🏀 WNBA
        </button>
      </div>



      {loading && <BballSkeleton />}

      {err && (
        <div className="bball-empty">
          <div className="bball-empty-icon">⚠</div>
          <p>Could not load {tab.toUpperCase()} games — {err}</p>
        </div>
      )}

      {!loading && !err && games?.length === 0 && <Empty sport={tab} />}

      {!loading && !err && games?.length > 0 && (
        <div className="bball-grid">
          {games.map((g, i) => (
            <GameCard key={g.id || i} game={g} />
          ))}
        </div>
      )}

      <p className="footer-text">
        {tab === 'nba'
          ? 'NBA data via BallDontLie API · predictions updated every 10 minutes'
          : 'WNBA fixtures via TheSportsDB · predictions updated every 10 minutes'}
      </p>
    </section>
  )
}

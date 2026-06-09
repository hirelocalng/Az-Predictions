import { useState, useEffect } from 'react'
import WorldCupSection from './components/WorldCupSection.jsx'
import IntlSection from './components/IntlSection.jsx'
import ClubSection from './components/ClubSection.jsx'
import DailyTipsSection from './components/DailyTipsSection.jsx'
import ResultsSection from './components/ResultsSection.jsx'
import { LiveScoresContext } from './contexts/LiveScoresContext.jsx'

const LogoIcon = () => (
  <svg viewBox="0 0 24 16" fill="none">
    {/* Glow halo behind prediction dot */}
    <circle cx="22" cy="1.5" r="5" fill="var(--green)" opacity="0.10" />
    {/* Historical trend — solid */}
    <path
      d="M1 14.5 L6.5 8.5 L11 11 L16 4"
      stroke="var(--green)"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    {/* Prediction projection — dashed */}
    <path
      d="M16 4 L22 1.5"
      stroke="var(--green)"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeDasharray="2.8 2.2"
      opacity="0.5"
    />
    {/* Prediction endpoint dot — animated */}
    <circle cx="22" cy="1.5" r="2.4" fill="var(--green)" className="logo-dot-pulse" />
  </svg>
)

function Header({ active, setActive }) {
  const scrollTo = id => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  return (
    <header className="header">
      <div className="header-logo">
        <div className="logo-mark"><LogoIcon /></div>
        <span className="logo-word">Az</span><span className="logo-sep">·</span><span className="logo-accent">Predictions</span>
      </div>

      <nav className="header-nav">
        {[['daily-tips',"Today's Tips"],['worldcup','World Cup 2026'],['intl','Live Internationals'],['club','Club Tips'],['history','History']].map(([id, label]) => (
          <button
            key={id}
            className={`nav-btn${active === id ? ' active' : ''}`}
            onClick={() => { setActive(id); scrollTo(id) }}
          >
            {label}
          </button>
        ))}
      </nav>

      <div className="header-status">
        <div className="status-dot" />
        Live
      </div>
    </header>
  )
}

function Hero({ onDailyTips, onWC, onIntl, onClub }) {
  return (
    <section className="hero">
      <div className="hero-eyebrow">AI-Powered Match Predictions</div>

      <h1 className="hero-title">
        The Smartest<br />
        <span className="accent-green">Football</span>{' '}
        <span className="accent-amber">Predictor</span>
      </h1>

      <p className="hero-desc">
        XGBoost models trained on 280,000+ matches across 40 leagues.
        World Cup 2026 forecasts, live internationals, and daily club tips.
      </p>

      <div className="hero-actions">
        <button className="btn-primary" onClick={onDailyTips}>Today's Top 5</button>
        <button className="btn-outline" onClick={onWC}>World Cup 2026</button>
        <button className="btn-outline" onClick={onIntl}>Live Internationals</button>
        <button className="btn-outline" onClick={onClub}>Club Tips</button>
      </div>

      <div className="hero-metrics">
        {[
          ['280k+', 'Matches trained', 'green'],
          ['40',    'Leagues covered', ''],
          ['57.7%', 'Corners accuracy', 'green'],
          ['59.2%', 'WC major accuracy', 'amber'],
        ].map(([num, label, cls]) => (
          <div className="metric" key={label}>
            <div className={`metric-num${cls ? ' ' + cls : ''}`}>{num}</div>
            <div className="metric-label">{label}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

export default function App() {
  const [active, setActive] = useState('daily-tips')
  const [liveScores, setLiveScores] = useState({})
  const scrollTo = id => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })

  useEffect(() => {
    const poll = () =>
      fetch('/api/live-scores')
        .then(r => r.json())
        .then(d => setLiveScores(d.live || {}))
        .catch(() => {})
    poll()
    const t = setInterval(poll, 60_000)
    return () => clearInterval(t)
  }, [])

  return (
    <LiveScoresContext.Provider value={liveScores}>
    <div className="app">
      <Header active={active} setActive={setActive} />

      <main style={{ paddingTop: 60 }}>
        <Hero
          onDailyTips={() => { setActive('daily-tips'); scrollTo('daily-tips') }}
          onWC={() => { setActive('worldcup'); scrollTo('worldcup') }}
          onIntl={() => { setActive('intl'); scrollTo('intl') }}
          onClub={() => { setActive('club'); scrollTo('club') }}
        />
        <div className="divider" />
        <DailyTipsSection />
        <div className="divider" />
        <WorldCupSection />
        <div className="divider" />
        <IntlSection />
        <div className="divider" />
        <ClubSection />
        <div className="divider" />
        <ResultsSection />
      </main>

      <footer className="footer">
        <strong>Az-Predictions</strong> — AI Football Predictions &nbsp;·&nbsp;
        Models trained on public match data &nbsp;·&nbsp;
        For entertainment purposes only<br />
        Please gamble responsibly. 18+
      </footer>
    </div>
    </LiveScoresContext.Provider>
  )
}

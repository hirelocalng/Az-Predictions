import { useState } from 'react'
import WorldCupSection from './components/WorldCupSection.jsx'
import ClubSection from './components/ClubSection.jsx'
import DailyTipsSection from './components/DailyTipsSection.jsx'

const LogoIcon = () => (
  <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
    <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <polygon points="8,3 10,7 14,7 11,10 12,14 8,11 4,14 5,10 2,7 6,7" fill="currentColor" opacity="0.8"/>
  </svg>
)

function Header({ active, setActive }) {
  const scrollTo = id => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  return (
    <header className="header">
      <div className="header-logo">
        <div className="logo-mark"><LogoIcon /></div>
        Az-<span className="logo-accent">Predictions</span>
      </div>

      <nav className="header-nav">
        {[['daily-tips',"Today's Tips"],['worldcup','World Cup 2026'],['club','Club Tips']].map(([id, label]) => (
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

function Hero({ onDailyTips, onWC, onClub }) {
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
        World Cup 2026 group-stage forecasts and daily club tips.
      </p>

      <div className="hero-actions">
        <button className="btn-primary" onClick={onDailyTips}>Today's Top 5</button>
        <button className="btn-outline" onClick={onWC}>World Cup 2026</button>
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
  const scrollTo = id => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })

  return (
    <div className="app">
      <Header active={active} setActive={setActive} />

      <main style={{ paddingTop: 60 }}>
        <Hero
          onDailyTips={() => { setActive('daily-tips'); scrollTo('daily-tips') }}
          onWC={() => { setActive('worldcup'); scrollTo('worldcup') }}
          onClub={() => { setActive('club'); scrollTo('club') }}
        />
        <div className="divider" />
        <DailyTipsSection />
        <div className="divider" />
        <WorldCupSection />
        <div className="divider" />
        <ClubSection />
      </main>

      <footer className="footer">
        <strong>Az-Predictions</strong> — AI Football Predictions &nbsp;·&nbsp;
        Models trained on public match data &nbsp;·&nbsp;
        For entertainment purposes only<br />
        Please gamble responsibly. 18+
      </footer>
    </div>
  )
}

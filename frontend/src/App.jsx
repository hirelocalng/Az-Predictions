import { useState, useEffect } from 'react'
import WorldCupSection from './components/WorldCupSection.jsx'
import IntlSection from './components/IntlSection.jsx'
import ClubSection from './components/ClubSection.jsx'
import BestBetSection from './components/BestBetSection.jsx'
import DailyTipsSection from './components/DailyTipsSection.jsx'
import ResultsSection from './components/ResultsSection.jsx'
import BasketballSection from './components/BasketballSection.jsx'
import LoginPage from './components/LoginPage.jsx'
import RegisterPage from './components/RegisterPage.jsx'
import SubscribePage from './components/SubscribePage.jsx'
import { LiveScoresContext } from './contexts/LiveScoresContext.jsx'
import { AuthContext } from './contexts/AuthContext.jsx'

// ── Routing helpers ───────────────────────────────────────────────────────────

const pathToPage = p => {
  if (p === '/login')     return 'login'
  if (p === '/register')  return 'register'
  if (p === '/subscribe') return 'subscribe'
  return 'home'
}

const getStoredAuth = () => {
  try { return JSON.parse(localStorage.getItem('az_auth') || 'null') } catch { return null }
}

// ── Telegram ─────────────────────────────────────────────────────────────────

const TelegramIcon = ({ size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M11.944 0A12 12 0 1 0 24 12 12 12 0 0 0 11.944 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.831-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/>
  </svg>
)

function TelegramFloat() {
  return (
    <a href="https://t.me/AZPREDICTS" target="_blank" rel="noopener noreferrer"
       className="tg-float" aria-label="Join AZPREDICTS Telegram channel">
      <TelegramIcon />
      <span>📢 Join Telegram · @AZPREDICTS</span>
    </a>
  )
}

// ── Header ────────────────────────────────────────────────────────────────────

function Header({ auth, onLogout, navigate }) {
  const isPremium = Boolean(auth?.user?.is_premium)

  return (
    <header className="header">
      <div className="header-logo" onClick={() => navigate('/')}>
        <div className="logo-wordmark">
          <span className="logo-az">AZ</span>
          <div className="logo-divider" />
          <span className="logo-prediction">PREDICTION</span>
        </div>
      </div>

      <div className="header-right">
        {auth ? (
          <div className="user-area">
            {isPremium
              ? <span className="premium-badge" title="Premium member">👑 Premium</span>
              : <button className="nav-btn nav-upgrade" onClick={() => navigate('/subscribe')}>Upgrade</button>
            }
            <button className="nav-btn" onClick={onLogout}>Log out</button>
          </div>
        ) : (
          <div className="user-area">
            <button className="nav-btn" onClick={() => navigate('/login')}>Log in</button>
            <button className="nav-btn nav-cta" onClick={() => navigate('/register')}>Sign up</button>
          </div>
        )}
      </div>
    </header>
  )
}

// ── Hero ──────────────────────────────────────────────────────────────────────

function Hero({ onBestBet, onDailyTips, onWC, onIntl, onClub, onNBA, onWNBA, navigate }) {
  return (
    <section className="hero">
      <div className="hero-eyebrow">AI-Powered Match Predictions</div>
      <h1 className="hero-title">
        The Smartest<br />
        <span className="accent-green">Football</span>{' '}
        <span className="accent-amber">Predictor</span>
      </h1>
      <div className="hero-actions">
        <button className="btn-primary hero-bb-btn" onClick={onBestBet}>⭐ Best Bet Today</button>
        <button className="btn-outline" onClick={onDailyTips}>Today's Tips</button>
        <button className="btn-outline" onClick={onWC}>World Cup 2026</button>
        <button className="btn-outline" onClick={onIntl}>Internationals</button>
        <button className="btn-outline" onClick={onClub}>Club Tips</button>
        <button className="btn-outline" onClick={onNBA}>🏀 NBA</button>
        <button className="btn-outline" onClick={onWNBA}>🏀 WNBA</button>
        <button className="btn-outline" onClick={() => navigate('/subscribe')}
                style={{ borderColor: 'rgba(201,146,42,0.5)', color: 'var(--amber)' }}>
          👑 Go Premium
        </button>
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

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [page,       setPage]       = useState(() => pathToPage(window.location.pathname))
  const [active,     setActive]     = useState('daily-tips')
  const [auth,       setAuthState]  = useState(getStoredAuth)
  const [liveScores, setLiveScores] = useState({})

  const navigate = path => {
    window.history.pushState({}, '', path)
    setPage(pathToPage(path))
  }

  const saveAuth = data => {
    localStorage.setItem('az_auth', JSON.stringify(data))
    setAuthState(data)
  }

  const logout = () => {
    localStorage.removeItem('az_auth')
    setAuthState(null)
    navigate('/')
  }

  // Browser back/forward
  useEffect(() => {
    const handlePop = () => setPage(pathToPage(window.location.pathname))
    window.addEventListener('popstate', handlePop)
    return () => window.removeEventListener('popstate', handlePop)
  }, [])

  // Verify stored token on mount
  useEffect(() => {
    if (!auth?.token) return
    fetch('/api/auth/me', { headers: { Authorization: `Bearer ${auth.token}` } })
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (d?.user) saveAuth({ ...auth, user: d.user })
        else logout()
      })
      .catch(() => {})
  }, [])

  // Live scores polling
  useEffect(() => {
    const poll = () =>
      fetch('/api/live-scores').then(r => r.json()).then(d => setLiveScores(d.live || {})).catch(() => {})
    poll()
    const t = setInterval(poll, 60_000)
    return () => clearInterval(t)
  }, [])

  const scrollTo = id => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })

  const authCtx = {
    auth,
    isPremium: Boolean(auth?.user?.is_premium),
    navigate,
    logout,
  }

  // Auth pages (full-page, no shell)
  if (page === 'login')
    return <LoginPage navigate={navigate} onAuth={saveAuth} />
  if (page === 'register')
    return <RegisterPage navigate={navigate} onAuth={saveAuth} />
  if (page === 'subscribe')
    return <SubscribePage navigate={navigate} auth={auth} onAuth={saveAuth} />

  // Main app
  return (
    <AuthContext.Provider value={authCtx}>
      <LiveScoresContext.Provider value={liveScores}>
        <div className="app">
          <Header auth={auth} onLogout={logout} navigate={navigate} />

          <main style={{ paddingTop: 64 }}>
            <Hero
              onBestBet={() => { setActive('best-bet');    scrollTo('best-bet')    }}
              onDailyTips={() => { setActive('daily-tips'); scrollTo('daily-tips') }}
              onWC={()        => { setActive('worldcup');   scrollTo('worldcup')   }}
              onIntl={()      => { setActive('intl');       scrollTo('intl')       }}
              onClub={()      => { setActive('club');       scrollTo('club')       }}
              onNBA={()       => { setActive('nba');        scrollTo('nba-section')  }}
              onWNBA={()      => { setActive('wnba');       scrollTo('wnba-section') }}
              navigate={navigate}
            />
            <BestBetSection />
            <div className="divider" />
            <DailyTipsSection />
            <div className="divider" />
            <WorldCupSection />
            <div className="divider" />
            <IntlSection />
            <div className="divider" />
            <BasketballSection />
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
            <div className="footer-tg">
              <a href="https://t.me/AZPREDICTS" target="_blank" rel="noopener noreferrer"
                 className="tg-footer-link">
                <TelegramIcon size={15} />&ensp;@AZPREDICTS — Free Daily Tips on Telegram
              </a>
            </div>
          </footer>
          <TelegramFloat />
        </div>
      </LiveScoresContext.Provider>
    </AuthContext.Provider>
  )
}

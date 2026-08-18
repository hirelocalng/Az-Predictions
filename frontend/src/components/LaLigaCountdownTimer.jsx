import { useState, useEffect } from 'react'

const pad = n => String(Math.max(0, n)).padStart(2, '0')

function Crest({ src, name }) {
  const [err, setErr] = useState(false)
  if (!src || err) {
    return (
      <div className="countdown-flag-fallback">
        {(name || '??').slice(0, 3).toUpperCase()}
      </div>
    )
  }
  return (
    <div className="countdown-flag">
      <img src={src} alt={name} onError={() => setErr(true)} style={{ objectFit: 'contain', background: '#fff' }} />
    </div>
  )
}

export default function LaLigaCountdownTimer() {
  const [data, setData]   = useState(null)
  const [secs, setSecs]   = useState(0)
  const [tried, setTried] = useState(false)

  useEffect(() => {
    fetch('/api/laliga/countdown')
      .then(r => r.json())
      .then(d => { setData(d); setSecs(d.seconds_remaining ?? 0); setTried(true) })
      .catch(() => setTried(true))
  }, [])

  useEffect(() => {
    if (secs <= 0) return
    const id = setInterval(() => setSecs(s => Math.max(0, s - 1)), 1000)
    return () => clearInterval(id)
  }, [secs > 0])

  if (!tried) return (
    <div className="countdown-card fade-up" style={{ minHeight: 140, justifyContent: 'center' }}>
      <div className="skeleton" style={{ width: 280, height: 24 }} />
    </div>
  )

  if (!data || !data.fixture) return (
    <div className="countdown-card fade-up" style={{ minHeight: 140, justifyContent: 'center' }}>
      <div className="countdown-kicker">Next La Liga Match</div>
      <p style={{ color: 'var(--text-muted)', marginTop: 8 }}>Fixture not available yet — check back soon.</p>
    </div>
  )

  const f = data.fixture
  const days  = Math.floor(secs / 86400)
  const hours = Math.floor((secs % 86400) / 3600)
  const mins  = Math.floor((secs % 3600) / 60)
  const ss    = secs % 60

  let localTime = ''
  try {
    localTime = f.utc_kickoff
      ? new Intl.DateTimeFormat([], { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' })
          .format(new Date(f.utc_kickoff))
      : ''
  } catch { /* noop */ }

  return (
    <div className="countdown-card fade-up">
      <div className="countdown-kicker">Next La Liga Match</div>

      <div className="countdown-fixture">
        <div className="countdown-team">
          <Crest src={f.home_crest} name={f.home} />
          <div className="countdown-team-name">{f.home}</div>
        </div>
        <div className="countdown-vs">vs</div>
        <div className="countdown-team">
          <Crest src={f.away_crest} name={f.away} />
          <div className="countdown-team-name">{f.away}</div>
        </div>
      </div>

      <div className="countdown-timer">
        {[['Days', pad(days)], ['Hours', pad(hours)], ['Min', pad(mins)], ['Sec', pad(ss)]].map(([label, val], i) => (
          <>
            {i > 0 && <div className="timer-sep" key={`sep${i}`}>:</div>}
            <div className="timer-unit" key={label}>
              <div className="timer-num">{val}</div>
              <div className="timer-label">{label}</div>
            </div>
          </>
        ))}
      </div>

      <div className="countdown-meta">
        {f.venue && <><span>{f.venue}</span><span className="meta-sep">·</span></>}
        {localTime && <><span>{localTime} Local</span><span className="meta-sep">·</span></>}
        <span>La Liga</span>
      </div>
    </div>
  )
}

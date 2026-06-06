import { useState, useEffect } from 'react'

const pad = n => String(Math.max(0, n)).padStart(2, '0')

const FLAG_CDN = code =>
  code ? `https://flagcdn.com/w80/${code.toLowerCase()}.png` : null

function Flag({ code, name }) {
  const [err, setErr] = useState(false)
  const src = FLAG_CDN(code)
  if (!src || err) {
    return (
      <div className="countdown-flag-fallback">
        {(code || '??').toUpperCase()}
      </div>
    )
  }
  return (
    <div className="countdown-flag">
      <img src={src} alt={name} onError={() => setErr(true)} />
    </div>
  )
}

export default function CountdownTimer() {
  const [data, setData]   = useState(null)
  const [secs, setSecs]   = useState(0)

  useEffect(() => {
    fetch('/api/countdown')
      .then(r => r.json())
      .then(d => { setData(d); setSecs(d.seconds_remaining ?? 0) })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (secs <= 0) return
    const id = setInterval(() => setSecs(s => Math.max(0, s - 1)), 1000)
    return () => clearInterval(id)
  }, [secs > 0])

  if (!data) return (
    <div className="countdown-card fade-up" style={{ minHeight: 140, justifyContent: 'center' }}>
      <div className="skeleton" style={{ width: 280, height: 24 }} />
    </div>
  )

  const f = data.fixture
  const days  = Math.floor(secs / 86400)
  const hours = Math.floor((secs % 86400) / 3600)
  const mins  = Math.floor((secs % 3600) / 60)
  const ss    = secs % 60

  return (
    <div className="countdown-card fade-up">
      <div className="countdown-kicker">Next World Cup Match</div>

      <div className="countdown-fixture">
        <div className="countdown-team">
          <Flag code={f.home_code} name={f.home} />
          <div className="countdown-team-name">{f.home}</div>
        </div>
        <div className="countdown-vs">vs</div>
        <div className="countdown-team">
          <Flag code={f.away_code} name={f.away} />
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
        <span>{f.venue}</span>
        <span className="meta-sep">·</span>
        <span>{f.time} Local</span>
        <span className="meta-sep">·</span>
        <span>Group {f.group}</span>
      </div>
    </div>
  )
}

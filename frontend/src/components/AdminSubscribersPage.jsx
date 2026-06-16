import { useState, useEffect, useCallback } from 'react'

// ── Tiny SVG line chart ────────────────────────────────────────────────────────
function LineChart({ data, width = 560, height = 180 }) {
  if (!data || data.length < 2) {
    return <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 40 }}>Not enough data yet</div>
  }
  const pad = { top: 12, right: 16, bottom: 36, left: 44 }
  const W = width - pad.left - pad.right
  const H = height - pad.top - pad.bottom

  const vals = data.map(d => d.cumulative)
  const minV = Math.min(...vals)
  const maxV = Math.max(...vals)
  const range = maxV - minV || 1

  const px = i => (i / (data.length - 1)) * W
  const py = v => H - ((v - minV) / range) * H

  const points = data.map((d, i) => `${px(i)},${py(d.cumulative)}`).join(' ')
  const area   = `M0,${H} ` + data.map((d, i) => `L${px(i)},${py(d.cumulative)}`).join(' ') + ` L${W},${H} Z`

  // x-axis labels: show ~6 evenly spaced
  const step = Math.max(1, Math.floor(data.length / 6))
  const xLabels = data.filter((_, i) => i % step === 0 || i === data.length - 1)

  // y-axis ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(t => ({
    v: Math.round(minV + t * range),
    y: py(minV + t * range),
  }))

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', maxWidth: width, display: 'block' }}>
      <defs>
        <linearGradient id="lineGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#22c55e" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#22c55e" stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <g transform={`translate(${pad.left},${pad.top})`}>
        {/* grid lines */}
        {yTicks.map(t => (
          <line key={t.v} x1={0} y1={t.y} x2={W} y2={t.y}
            stroke="var(--border)" strokeDasharray="3 3" strokeWidth={0.5} />
        ))}
        {/* area fill */}
        <path d={area} fill="url(#lineGrad)" />
        {/* line */}
        <polyline points={points} fill="none" stroke="#22c55e" strokeWidth={2} strokeLinejoin="round" />
        {/* y-axis labels */}
        {yTicks.map(t => (
          <text key={t.v} x={-6} y={t.y + 4} textAnchor="end"
            fontSize={9} fill="var(--text-muted)">{t.v}</text>
        ))}
        {/* x-axis labels */}
        {xLabels.map((d, i) => {
          const idx = data.indexOf(d)
          return (
            <text key={i} x={px(idx)} y={H + 20} textAnchor="middle"
              fontSize={9} fill="var(--text-muted)">
              {d.date.slice(5)} {/* MM-DD */}
            </text>
          )
        })}
        {/* axes */}
        <line x1={0} y1={0} x2={0} y2={H} stroke="var(--border)" strokeWidth={1} />
        <line x1={0} y1={H} x2={W} y2={H} stroke="var(--border)" strokeWidth={1} />
      </g>
    </svg>
  )
}

// ── Tiny SVG pie chart ────────────────────────────────────────────────────────
function PieChart({ free, premium }) {
  const total = free + premium
  if (total === 0) return <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 40 }}>No subscribers yet</div>

  const R = 70
  const cx = 90
  const cy = 90

  const slice = (startAngle, value) => {
    const angle = (value / total) * 2 * Math.PI
    const x1 = cx + R * Math.sin(startAngle)
    const y1 = cy - R * Math.cos(startAngle)
    const x2 = cx + R * Math.sin(startAngle + angle)
    const y2 = cy - R * Math.cos(startAngle + angle)
    const large = angle > Math.PI ? 1 : 0
    return `M${cx},${cy} L${x1},${y1} A${R},${R} 0 ${large},1 ${x2},${y2} Z`
  }

  const premiumPct = Math.round(premium / total * 100)
  const freePct = 100 - premiumPct

  return (
    <svg viewBox="0 0 220 180" style={{ width: '100%', maxWidth: 220, display: 'block' }}>
      {premium === 0 ? (
        <circle cx={cx} cy={cy} r={R} fill="#374151" />
      ) : free === 0 ? (
        <circle cx={cx} cy={cy} r={R} fill="#f59e0b" />
      ) : (
        <>
          <path d={slice(0, free)} fill="#374151" />
          <path d={slice((free / total) * 2 * Math.PI, premium)} fill="#f59e0b" />
        </>
      )}
      {/* inner circle for donut */}
      <circle cx={cx} cy={cy} r={R * 0.52} fill="var(--card-bg)" />
      {/* center label */}
      <text x={cx} y={cy - 6} textAnchor="middle" fontSize={15} fontWeight="700" fill="var(--text)">{premiumPct}%</text>
      <text x={cx} y={cy + 12} textAnchor="middle" fontSize={9} fill="var(--text-muted)">premium</text>
      {/* legend */}
      <rect x={0} y={160} width={10} height={10} rx={2} fill="#374151" />
      <text x={14} y={169} fontSize={10} fill="var(--text-muted)">Free {freePct}%</text>
      <rect x={80} y={160} width={10} height={10} rx={2} fill="#f59e0b" />
      <text x={94} y={169} fontSize={10} fill="var(--text-muted)">Premium {premiumPct}%</text>
    </svg>
  )
}

// ── Metric card ───────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, color = 'var(--green)' }) {
  return (
    <div style={{
      background: 'var(--card-bg)', border: '1px solid var(--border)', borderRadius: 12,
      padding: '20px 24px', flex: '1 1 140px', minWidth: 130,
    }}>
      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: '1.9rem', fontWeight: 800, color, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>{sub}</div>}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function AdminSubscribersPage({ navigate }) {
  const [pw,      setPw]      = useState(() => sessionStorage.getItem('az_admin_pw') || '')
  const [authed,  setAuthed]  = useState(false)
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [msg,     setMsg]     = useState('')

  const load = useCallback(async (password) => {
    const r = await fetch('/admin/subscribers', {
      headers: { 'X-Admin-Password': password },
    })
    if (r.status === 403 || r.status === 503) {
      setAuthed(false)
      sessionStorage.removeItem('az_admin_pw')
      setMsg('Wrong password or admin not configured.')
      return
    }
    const d = await r.json()
    if (d.error) { setMsg(d.error); return }
    setData(d)
    setAuthed(true)
    sessionStorage.setItem('az_admin_pw', password)
    setMsg('')
  }, [])

  useEffect(() => {
    if (pw) load(pw)
  }, [])

  const login = async e => {
    e.preventDefault()
    setLoading(true)
    setMsg('')
    await load(pw)
    setLoading(false)
  }

  if (!authed) {
    return (
      <div className="auth-page">
        <header className="auth-header">
          <button className="auth-back" onClick={() => navigate('/admin')}>← Admin</button>
          <div className="auth-logo">⚽ Az-Predictions</div>
        </header>
        <div className="auth-card">
          <h2 className="auth-title">Admin Access</h2>
          <p className="auth-sub">Enter the admin password to view subscriber analytics.</p>
          <form onSubmit={login} className="auth-form">
            <div className="field">
              <label>Admin Password</label>
              <input type="password" value={pw} onChange={e => setPw(e.target.value)}
                required placeholder="••••••••" autoFocus />
            </div>
            {msg && <div className="auth-error">⚠ {msg}</div>}
            <button type="submit" className="btn-primary btn-full" disabled={loading}>
              {loading ? 'Checking…' : 'Enter'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  const { totals, new_this_week, new_this_month, recent, growth } = data || {}
  const t = totals || {}

  const fmtDate = iso => {
    if (!iso) return '—'
    return new Date(iso).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })
  }
  const fmtTime = iso => {
    if (!iso) return '—'
    const d = new Date(iso)
    return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }) + ' ' +
           d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh', padding: '80px 20px 60px' }}>
      <div style={{ maxWidth: 1000, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 32, flexWrap: 'wrap', gap: 12 }}>
          <div>
            <h1 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--green)', margin: 0 }}>
              Subscriber Analytics
            </h1>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 4 }}>
              Live data from the users table
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button className="nav-btn" onClick={() => navigate('/admin')}>← Odds Tips</button>
            <button className="nav-btn" onClick={() => load(pw)} style={{ color: 'var(--green)' }}>↺ Refresh</button>
            <button className="nav-btn" onClick={() => navigate('/')}>← Site</button>
          </div>
        </div>

        {/* Metric cards */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 28 }}>
          <StatCard label="Total Subscribers" value={t.total ?? '—'} />
          <StatCard label="Free" value={t.free ?? '—'} color="var(--text-2)" />
          <StatCard label="Premium" value={t.premium ?? '—'} color="var(--amber)" />
          <StatCard label="Conversion Rate" value={t.conversion_rate != null ? `${t.conversion_rate}%` : '—'} color="var(--amber)" sub="free → premium" />
          <StatCard label="New This Week" value={new_this_week ?? '—'} color="var(--green)" />
          <StatCard label="New This Month" value={new_this_month ?? '—'} color="var(--green)" />
        </div>

        {/* Charts row */}
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 28 }}>
          {/* Growth chart */}
          <div style={{
            flex: '3 1 380px', background: 'var(--card-bg)', border: '1px solid var(--border)',
            borderRadius: 12, padding: '20px 24px',
          }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 16 }}>
              Subscriber Growth — Last 60 Days
            </div>
            <LineChart data={growth} />
          </div>

          {/* Pie chart */}
          <div style={{
            flex: '1 1 200px', background: 'var(--card-bg)', border: '1px solid var(--border)',
            borderRadius: 12, padding: '20px 24px', display: 'flex', flexDirection: 'column', alignItems: 'center',
          }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 16, alignSelf: 'flex-start' }}>
              Free vs Premium
            </div>
            <PieChart free={t.free || 0} premium={t.premium || 0} />
          </div>
        </div>

        {/* Recent subscribers table */}
        <div style={{
          background: 'var(--card-bg)', border: '1px solid var(--border)',
          borderRadius: 12, overflow: 'hidden',
        }}>
          <div style={{ padding: '18px 24px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontWeight: 700, color: 'var(--text-2)' }}>Recent Subscribers</span>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Last 30</span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Name', 'Email', 'Type', 'Joined', 'Last Login'].map(h => (
                    <th key={h} style={{ padding: '10px 16px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(recent || []).map(u => (
                  <tr key={u.id} style={{ borderBottom: '1px solid var(--border)' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--card-hover, rgba(255,255,255,0.03))'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                    <td style={{ padding: '10px 16px', color: 'var(--text-2)', fontWeight: 500 }}>{u.name}</td>
                    <td style={{ padding: '10px 16px', color: 'var(--text-muted)' }}>{u.email}</td>
                    <td style={{ padding: '10px 16px' }}>
                      <span style={{
                        display: 'inline-block', padding: '2px 10px', borderRadius: 20,
                        fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase',
                        background: u.is_premium ? 'rgba(245,158,11,0.15)' : 'rgba(100,116,139,0.15)',
                        color: u.is_premium ? 'var(--amber)' : 'var(--text-muted)',
                      }}>
                        {u.is_premium ? 'Premium' : 'Free'}
                      </span>
                    </td>
                    <td style={{ padding: '10px 16px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{fmtDate(u.joined)}</td>
                    <td style={{ padding: '10px 16px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{fmtTime(u.last_login)}</td>
                  </tr>
                ))}
                {(!recent || recent.length === 0) && (
                  <tr>
                    <td colSpan={5} style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>No subscribers yet</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  )
}

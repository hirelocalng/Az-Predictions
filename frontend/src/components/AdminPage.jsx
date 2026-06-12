import { useState, useEffect, useCallback } from 'react'

const TIERS    = ['free', 'vip']
const STATUS_EMOJI = { won: '✅', lost: '❌', pending: '⏳' }

const emptyForm = {
  date: new Date().toISOString().slice(0, 10),
  time: '', league: '', team_home: '', team_away: '',
  score: '', tip_description: '', odd_value: '', tier: 'free', status: 'pending',
}

export default function AdminPage({ navigate }) {
  const [pw,      setPw]      = useState(() => sessionStorage.getItem('az_admin_pw') || '')
  const [authed,  setAuthed]  = useState(false)
  const [tips,    setTips]    = useState([])
  const [form,    setForm]    = useState(emptyForm)
  const [msg,     setMsg]     = useState('')
  const [loading, setLoading] = useState(false)

  const headers = useCallback(() => ({
    'Content-Type': 'application/json',
    'X-Admin-Password': pw,
  }), [pw])

  const loadTips = useCallback(async () => {
    const r = await fetch(`/admin/odds-tips?admin_pw=${encodeURIComponent(pw)}`)
    if (r.status === 403 || r.status === 503) {
      setAuthed(false)
      sessionStorage.removeItem('az_admin_pw')
      return false
    }
    const d = await r.json()
    setTips(d.tips || [])
    setAuthed(true)
    sessionStorage.setItem('az_admin_pw', pw)
    return true
  }, [pw])

  useEffect(() => {
    if (pw) loadTips()
  }, [])

  const login = async e => {
    e.preventDefault()
    setLoading(true)
    setMsg('')
    try {
      const ok = await loadTips()
      if (!ok) setMsg('Wrong password or admin not configured.')
    } catch { setMsg('Connection error') }
    setLoading(false)
  }

  const submit = async e => {
    e.preventDefault()
    setMsg('')
    const body = {
      ...form,
      odd_value: form.odd_value !== '' ? parseFloat(form.odd_value) : null,
    }
    const r = await fetch('/admin/odds-tips', {
      method: 'POST', headers: headers(), body: JSON.stringify(body),
    })
    const d = await r.json()
    if (!r.ok) { setMsg('Error: ' + (d.error || r.status)); return }
    setMsg(`✅ Tip #${d.id} added successfully`)
    setForm(f => ({ ...emptyForm, date: f.date, tier: f.tier }))
    loadTips()
  }

  const update = async (id, patch) => {
    await fetch(`/admin/odds-tips/${id}`, {
      method: 'PUT', headers: headers(), body: JSON.stringify(patch),
    })
    loadTips()
  }

  const del = async id => {
    if (!window.confirm('Delete this tip?')) return
    await fetch(`/admin/odds-tips/${id}`, { method: 'DELETE', headers: headers() })
    loadTips()
  }

  const field = key => e => setForm(f => ({ ...f, [key]: e.target.value }))

  if (!authed) {
    return (
      <div className="auth-page">
        <header className="auth-header">
          <button className="auth-back" onClick={() => navigate('/')}>← Back</button>
          <div className="auth-logo">⚽ Az-Predictions</div>
        </header>
        <div className="auth-card">
          <h2 className="auth-title">Admin Access</h2>
          <p className="auth-sub">Enter the admin password to manage odds tips.</p>
          <form onSubmit={login} className="auth-form">
            <div className="field">
              <label>Admin Password</label>
              <input type="password" value={pw} onChange={e => setPw(e.target.value)}
                required placeholder="••••••••" autoFocus />
            </div>
            {msg && <div className="auth-error">⚠ {msg}</div>}
            <button type="submit" className="btn-primary btn-full" disabled={loading}>
              {loading ? 'Checking…' : 'Enter Admin'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  const freeTips = tips.filter(t => t.tier === 'free')
  const vipTips  = tips.filter(t => t.tier === 'vip')

  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh', padding: '80px 20px 60px' }}>
      <div style={{ maxWidth: 900, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 32, flexWrap: 'wrap', gap: 12 }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--green)' }}>
            Admin — Odds Tips
          </h1>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <a href={`/admin/analytics?admin_pw=${encodeURIComponent(pw)}`}
               target="_blank" rel="noopener noreferrer"
               style={{ color: 'var(--amber)', fontSize: '0.85rem', textDecoration: 'none' }}>
              📊 Analytics →
            </a>
            <button className="nav-btn" onClick={() => navigate('/')}>← Site</button>
            <button className="nav-btn" onClick={() => { setAuthed(false); sessionStorage.removeItem('az_admin_pw') }}>
              Log out
            </button>
          </div>
        </div>

        {/* Add tip form */}
        <div className="admin-card">
          <h3 style={{ marginBottom: 20, color: 'var(--text-2)', fontSize: '1rem', fontWeight: 700 }}>
            Add New Tip
          </h3>
          <form onSubmit={submit} className="admin-form">
            <div className="admin-row">
              <div className="field">
                <label>Date</label>
                <input type="date" value={form.date} onChange={field('date')} required />
              </div>
              <div className="field">
                <label>Time (UTC)</label>
                <input type="time" value={form.time} onChange={field('time')} />
              </div>
              <div className="field">
                <label>Tier</label>
                <select value={form.tier} onChange={field('tier')}>
                  {TIERS.map(t => <option key={t} value={t}>{t.toUpperCase()}</option>)}
                </select>
              </div>
            </div>

            <div className="field">
              <label>League</label>
              <input placeholder="e.g. World: Friendly International" value={form.league} onChange={field('league')} />
            </div>

            <div className="admin-row">
              <div className="field">
                <label>Home Team</label>
                <input placeholder="Netherlands" value={form.team_home} onChange={field('team_home')} required />
              </div>
              <div className="field">
                <label>Away Team</label>
                <input placeholder="France" value={form.team_away} onChange={field('team_away')} required />
              </div>
            </div>

            <div className="admin-row">
              <div className="field" style={{ flex: 2 }}>
                <label>Tip Description</label>
                <input placeholder="Netherlands Over+ 1.5 Goals" value={form.tip_description} onChange={field('tip_description')} />
              </div>
              <div className="field">
                <label>Odd Value</label>
                <input type="number" step="0.01" min="1" placeholder="1.40"
                  value={form.odd_value} onChange={field('odd_value')} />
              </div>
            </div>

            <button type="submit" className="btn-primary" style={{ alignSelf: 'flex-start', marginTop: 4 }}>
              + Add Tip
            </button>
            {msg && (
              <p style={{ color: msg.startsWith('✅') ? 'var(--green)' : 'var(--red)', fontSize: '0.85rem', marginTop: 4 }}>
                {msg}
              </p>
            )}
          </form>
        </div>

        {/* Tips list */}
        {[['🆓 Free Tips', freeTips], ['👑 VIP Tips', vipTips]].map(([label, list]) => (
          <div key={label} className="admin-card" style={{ marginTop: 20 }}>
            <h3 style={{ marginBottom: 14, color: 'var(--text-2)', fontSize: '0.95rem', fontWeight: 700 }}>
              {label} ({list.length})
            </h3>
            {list.length === 0 ? (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No tips yet.</p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {list.map(t => (
                  <div key={t.id} className="admin-tip-row">
                    <div className="admin-tip-info">
                      <span className="admin-tip-date">{t.date} {t.time || ''}</span>
                      <span className="admin-tip-teams">{t.team_home} vs {t.team_away}</span>
                      {t.tip_description && (
                        <span className="admin-tip-desc">{t.tip_description}</span>
                      )}
                      {t.odd_value != null && (
                        <span className="admin-tip-odd">@{Number(t.odd_value).toFixed(2)}</span>
                      )}
                      {t.score && (
                        <span style={{ color: 'var(--amber)', fontSize: '0.78rem' }}>({t.score})</span>
                      )}
                    </div>
                    <div className="admin-tip-actions">
                      <span style={{ fontSize: '1.05rem' }}>{STATUS_EMOJI[t.status] || '⏳'}</span>
                      {t.status !== 'won' && (
                        <button className="admin-btn admin-btn-won"
                          onClick={() => update(t.id, { status: 'won' })}>Won</button>
                      )}
                      {t.status !== 'lost' && (
                        <button className="admin-btn admin-btn-lost"
                          onClick={() => update(t.id, { status: 'lost' })}>Lost</button>
                      )}
                      {t.status !== 'pending' && (
                        <button className="admin-btn"
                          onClick={() => update(t.id, { status: 'pending' })}>Reset</button>
                      )}
                      <input
                        placeholder="Score"
                        style={{ width: 68, background: 'rgba(255,255,255,.05)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text)', padding: '3px 8px', fontSize: '0.78rem' }}
                        defaultValue={t.score || ''}
                        onBlur={e => { if (e.target.value !== (t.score || '')) update(t.id, { score: e.target.value }) }}
                      />
                      <button className="admin-btn" style={{ color: 'var(--red)' }}
                        onClick={() => del(t.id)}>✕</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

      </div>
    </div>
  )
}

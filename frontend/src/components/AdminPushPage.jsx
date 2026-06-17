import { useState, useEffect, useCallback } from 'react'

const RECIPIENT_TYPES = [
  { value: 'all',      label: 'All Users (subscribed)' },
  { value: 'free',     label: 'Free Users Only' },
  { value: 'premium',  label: 'Premium Users Only' },
  { value: 'specific', label: 'Specific User (by ID)' },
]

function MobilePreview({ title, message }) {
  return (
    <div className="push-phone">
      <div className="push-phone-bar">9:41</div>
      <div className="push-notif">
        <div className="push-notif-top">
          <span className="push-notif-app">⚽ Az-Predictions</span>
          <span className="push-notif-time">now</span>
        </div>
        <div className="push-notif-title">{title || 'Notification title'}</div>
        <div className="push-notif-body">{message || 'Your message will appear here...'}</div>
      </div>
      <div className="push-phone-label">Preview</div>
    </div>
  )
}

export default function AdminPushPage({ navigate }) {
  const [pw,      setPw]      = useState(() => sessionStorage.getItem('az_admin_pw') || '')
  const [authed,  setAuthed]  = useState(false)
  const [loginMsg, setLoginMsg] = useState('')
  const [form,    setForm]    = useState({ title: '', message: '', recipientType: 'all', userId: '' })
  const [sendMsg, setSendMsg] = useState('')
  const [sending, setSending] = useState(false)
  const [history, setHistory] = useState([])

  const loadHistory = useCallback(async (password) => {
    const r = await fetch('/api/admin/notification-history', {
      headers: { 'X-Admin-Password': password },
    })
    if (r.status === 403 || r.status === 503) {
      setAuthed(false)
      sessionStorage.removeItem('az_admin_pw')
      setLoginMsg('Wrong password or admin not configured.')
      return false
    }
    const d = await r.json()
    if (d.error) { setLoginMsg(d.error); return false }
    setHistory(d.notifications || [])
    setAuthed(true)
    sessionStorage.setItem('az_admin_pw', password)
    setLoginMsg('')
    return true
  }, [])

  useEffect(() => { if (pw) loadHistory(pw) }, [])

  const login = async e => {
    e.preventDefault()
    setLoginMsg('')
    await loadHistory(pw)
  }

  const field = k => e => setForm(f => ({ ...f, [k]: e.target.value }))

  const send = async e => {
    e.preventDefault()
    setSendMsg('')
    setSending(true)
    try {
      const body = { password: pw, title: form.title, message: form.message, recipientType: form.recipientType }
      if (form.recipientType === 'specific') body.userId = parseInt(form.userId, 10)
      const r = await fetch('/api/admin/send-notification', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Admin-Password': pw },
        body: JSON.stringify(body),
      })
      const d = await r.json()
      if (!r.ok) { setSendMsg('❌ ' + (d.error || `Error ${r.status}`)); return }
      const count = d.sent_count < 0 ? 'all subscribers' : `${d.sent_count} device(s)`
      setSendMsg(`✅ Sent to ${count}`)
      setForm(f => ({ ...f, title: '', message: '' }))
      loadHistory(pw)
    } catch { setSendMsg('❌ Network error') }
    finally { setSending(false) }
  }

  const recipientLabel = {
    all:      'All subscribers',
    free:     'Free users',
    premium:  'Premium users',
    specific: 'Specific user',
  }

  /* ── Password gate ──────────────────────────────────────────────────────── */
  if (!authed) {
    return (
      <div className="auth-page">
        <header className="auth-header">
          <button className="auth-back" onClick={() => navigate('/admin')}>← Admin</button>
          <div className="auth-logo">⚽ Az-Predictions</div>
        </header>
        <div className="auth-card">
          <h2 className="auth-title">Admin Access</h2>
          <p className="auth-sub">Enter the admin password to manage push notifications.</p>
          <form onSubmit={login} className="auth-form">
            <div className="field">
              <label>Admin Password</label>
              <input type="password" value={pw} onChange={e => setPw(e.target.value)}
                required placeholder="••••••••" autoFocus />
            </div>
            {loginMsg && <div className="auth-error">⚠ {loginMsg}</div>}
            <button type="submit" className="btn-primary btn-full">Enter Admin</button>
          </form>
        </div>
      </div>
    )
  }

  /* ── Main page ──────────────────────────────────────────────────────────── */
  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh', padding: '80px 20px 60px' }}>
      <div style={{ maxWidth: 960, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 32, flexWrap: 'wrap', gap: 12 }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--green)' }}>
            🔔 Push Notifications
          </h1>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button className="nav-btn" onClick={() => navigate('/admin')}
              style={{ color: 'var(--green)', borderColor: 'var(--green)' }}>
              ← Odds Tips
            </button>
            <button className="nav-btn" onClick={() => navigate('/admin/subscribers')}
              style={{ color: 'var(--green)', borderColor: 'var(--green)' }}>
              👥 Subscribers
            </button>
            <button className="nav-btn" onClick={() => navigate('/')}>← Site</button>
            <button className="nav-btn" onClick={() => {
              setAuthed(false)
              sessionStorage.removeItem('az_admin_pw')
            }}>Log out</button>
          </div>
        </div>

        {/* Compose + Preview side-by-side */}
        <div className="push-compose-grid">
          <div className="admin-card">
            <h3 style={{ marginBottom: 20, color: 'var(--text-2)', fontSize: '1rem', fontWeight: 700 }}>
              Compose Notification
            </h3>
            <form onSubmit={send} className="admin-form">
              <div className="field">
                <label>Title</label>
                <input
                  value={form.title}
                  onChange={field('title')}
                  placeholder="e.g. ✅ New WC 2026 predictions live!"
                  maxLength={100}
                  required
                />
              </div>

              <div className="field">
                <label>Message</label>
                <textarea
                  value={form.message}
                  onChange={field('message')}
                  placeholder="e.g. 14 World Cup matches analysed — check your saved bets"
                  rows={3}
                  maxLength={300}
                  style={{ resize: 'vertical', fontFamily: 'inherit', fontSize: '0.88rem' }}
                  required
                />
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  {form.message.length}/300
                </span>
              </div>

              <div className="field">
                <label>Recipients</label>
                <select value={form.recipientType} onChange={field('recipientType')}>
                  {RECIPIENT_TYPES.map(r => (
                    <option key={r.value} value={r.value}>{r.label}</option>
                  ))}
                </select>
              </div>

              {form.recipientType === 'specific' && (
                <div className="field">
                  <label>User ID</label>
                  <input
                    type="number"
                    value={form.userId}
                    onChange={field('userId')}
                    placeholder="e.g. 42"
                    min="1"
                    required
                  />
                </div>
              )}

              <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                <button type="submit" className="btn-primary" disabled={sending}>
                  {sending ? 'Sending…' : '🔔 Send Now'}
                </button>
                {sendMsg && (
                  <p style={{ fontSize: '0.85rem', margin: 0,
                    color: sendMsg.startsWith('✅') ? 'var(--green)' : 'var(--red)' }}>
                    {sendMsg}
                  </p>
                )}
              </div>

              {/* Quick templates */}
              <details style={{ marginTop: 16 }}>
                <summary style={{ fontSize: '0.8rem', color: 'var(--text-muted)', cursor: 'pointer' }}>
                  Quick templates
                </summary>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
                  {[
                    ['New WC predictions', '✅ New WC 2026 predictions live!', "Today's World Cup matches are ready — 14 games analysed"],
                    ['Premium deal',       '👑 Premium 50% off — today only!', 'Unlock VIP tips, daily accumulators & match alerts for ₦2,500'],
                    ['Maintenance',        '🔧 Brief maintenance tonight',     'Az-Predictions will be offline 02:00–03:00 UTC for upgrades'],
                    ['Feature launch',     '🚀 Save predictions — new feature!', 'Tap ❤️ on any prediction to bookmark it and track your wins'],
                  ].map(([name, t, m]) => (
                    <button key={name} type="button"
                      style={{ textAlign: 'left', background: 'rgba(255,255,255,.04)', border: '1px solid var(--border)', borderRadius: 7, padding: '6px 10px', cursor: 'pointer', color: 'var(--text-2)', fontSize: '0.78rem' }}
                      onClick={() => setForm(f => ({ ...f, title: t, message: m }))}>
                      <strong style={{ color: 'var(--text)' }}>{name}</strong> — {t}
                    </button>
                  ))}
                </div>
              </details>
            </form>
          </div>

          <MobilePreview title={form.title} message={form.message} />
        </div>

        {/* History table */}
        <div className="admin-card" style={{ marginTop: 20 }}>
          <h3 style={{ marginBottom: 16, color: 'var(--text-2)', fontSize: '1rem', fontWeight: 700 }}>
            Notification History
          </h3>
          {history.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No notifications sent yet.</p>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: '0.78rem', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', textAlign: 'left', borderBottom: '1px solid var(--border)' }}>
                    {['Title', 'Message', 'Recipients', 'Sent at', 'Devices'].map(h => (
                      <th key={h} style={{ padding: '7px 10px', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {history.map((n, i) => (
                    <tr key={n.id} style={{
                      borderBottom: '1px solid rgba(255,255,255,.04)',
                      background: i % 2 === 0 ? 'rgba(255,255,255,.02)' : 'transparent',
                    }}>
                      <td style={{ padding: '8px 10px', color: 'var(--text)', fontWeight: 600, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {n.title}
                      </td>
                      <td style={{ padding: '8px 10px', color: 'var(--text-2)', maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {n.message}
                      </td>
                      <td style={{ padding: '8px 10px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                        {n.recipient_type === 'specific'
                          ? `User: ${n.recipient_user_email || n.recipient_user_id || '—'}`
                          : recipientLabel[n.recipient_type] || n.recipient_type}
                      </td>
                      <td style={{ padding: '8px 10px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                        {n.sent_at ? new Date(n.sent_at).toLocaleString() : '—'}
                      </td>
                      <td style={{ padding: '8px 10px', color: 'var(--green)', fontWeight: 700 }}>
                        {n.sent_count > 0 ? n.sent_count : n.recipient_type === 'all' ? 'all' : '0'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}

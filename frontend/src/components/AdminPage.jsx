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
            <a href="/admin/analytics"
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

        {/* DB Corrections */}
        <DbCorrections headers={headers} />

        {/* Recalc Pending */}
        <RecalcPending headers={headers} />

        {/* Backfill Corners */}
        <BackfillCorners headers={headers} />

        {/* Deduplicate */}
        <DeduplicatePanel headers={headers} />

        {/* Audit + Re-resolve */}
        <AuditPanel headers={headers} />

        {/* Re-derive WON/LOST statuses */}
        <RederivePanel headers={headers} />

      </div>
    </div>
  )
}

function DeduplicatePanel({ headers }) {
  const [status, setStatus] = useState('')
  const [result, setResult] = useState(null)
  const [busy,   setBusy]   = useState(false)

  const run = async (apply) => {
    if (apply && !window.confirm(
      'This will permanently delete the less-complete duplicate row for each pair found. Continue?'
    )) return
    setBusy(true); setStatus(''); setResult(null)
    try {
      const r = await fetch('/admin/deduplicate', {
        method: apply ? 'POST' : 'GET',
        headers: headers(),
      })
      const d = await r.json()
      setResult(d)
      if (!r.ok) { setStatus('Error: ' + (d.error || r.status)); return }
      setStatus(apply
        ? `✅ Deleted ${d.deleted} duplicate row(s)`
        : `Preview — ${d.duplicate_pairs} duplicate pair(s) found`)
    } catch { setStatus('Connection error') }
    setBusy(false)
  }

  return (
    <div className="admin-card" style={{ marginTop: 20 }}>
      <h3 style={{ marginBottom: 6, color: 'var(--amber)', fontSize: '0.95rem', fontWeight: 700 }}>
        🔁 Deduplicate History
      </h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: 14 }}>
        Finds prediction rows where the same two teams appear with match dates ≤2 days apart
        (UTC midnight-crossover duplicates). Preview first, then delete the less-complete copy.
      </p>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <button className="admin-btn" onClick={() => run(false)} disabled={busy}>Preview</button>
        <button className="admin-btn admin-btn-lost" onClick={() => run(true)} disabled={busy}>
          Delete Duplicates
        </button>
      </div>
      {status && (
        <p style={{ marginTop: 10, fontSize: '0.82rem',
          color: status.startsWith('✅') ? 'var(--green)'
               : status.startsWith('Preview') ? 'var(--amber)' : 'var(--red)' }}>
          {status}
        </p>
      )}
      {result && result.pairs && result.pairs.length > 0 && (
        <div style={{ marginTop: 14, overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '0.72rem', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Match</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>ID 1</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Date 1</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Status 1</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>ID 2</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Date 2</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Status 2</th>
              </tr>
            </thead>
            <tbody>
              {result.pairs.map((p, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? 'rgba(255,255,255,.03)' : 'transparent' }}>
                  <td style={{ padding: '4px 8px', color: 'var(--text)' }}>{p.home_team} vs {p.away_team}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)', fontSize: '0.65rem' }}>{p.id1}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{p.date1}</td>
                  <td style={{ padding: '4px 8px' }}>{p.status1}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)', fontSize: '0.65rem' }}>{p.id2}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{p.date2}</td>
                  <td style={{ padding: '4px 8px' }}>{p.status2}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {result && result.records && result.records.length > 0 && (
        <ul style={{ marginTop: 12, fontSize: '0.78rem', color: 'var(--green)', paddingLeft: 16 }}>
          {result.records.map((r, i) => (
            <li key={i}>Deleted <code style={{ fontSize: '0.68rem' }}>{r.deleted}</code> → kept <code style={{ fontSize: '0.68rem' }}>{r.kept}</code> ({r.match})</li>
          ))}
        </ul>
      )}
      {result && result.duplicate_pairs === 0 && (
        <p style={{ marginTop: 10, fontSize: '0.82rem', color: 'var(--green)' }}>
          ✅ No duplicates found — history is clean.
        </p>
      )}
    </div>
  )
}

function RecalcPending({ headers }) {
  const [status,  setStatus]  = useState('')
  const [result,  setResult]  = useState(null)
  const [busy,    setBusy]    = useState(false)

  const run = async (apply) => {
    if (apply && !window.confirm(
      'This will re-run every PENDING/LIVE match through the shared prediction function and overwrite predicted_winner / predicted_goals / predicted_btts / predicted_corners in the DB. Continue?'
    )) return
    setBusy(true); setStatus(''); setResult(null)
    try {
      const r = await fetch('/admin/recalc-pending', {
        method: apply ? 'POST' : 'GET',
        headers: headers(),
      })
      const d = await r.json()
      setResult(d)
      if (!r.ok) { setStatus('Error: ' + (d.error || r.status)); return }
      setStatus(apply
        ? `✅ Applied — ${d.updated} updated, ${d.errors} errors`
        : `Preview — ${d.total_pending} pending, ${d.updated} would change, ${d.errors} errors`)
    } catch { setStatus('Connection error') }
    setBusy(false)
  }

  const fmtPW = v => v || '—'

  return (
    <div className="admin-card" style={{ marginTop: 20 }}>
      <h3 style={{ marginBottom: 6, color: 'var(--amber)', fontSize: '0.95rem', fontWeight: 700 }}>
        ♻️ Recalculate Pending Predictions
      </h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: 14 }}>
        Re-runs every PENDING/LIVE match through the shared prediction function.
        Updates <code>predicted_winner</code>, <code>predicted_goals</code>,{' '}
        <code>predicted_btts</code>, <code>predicted_corners</code>.
        <br />
        <em>win_probability, ou_probability, best_bet_type are live-computed — not stored.</em>
      </p>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <button className="admin-btn" onClick={() => run(false)} disabled={busy}>
          Preview
        </button>
        <button className="admin-btn admin-btn-won" onClick={() => run(true)} disabled={busy}>
          Apply Recalc
        </button>
      </div>
      {status && (
        <p style={{ marginTop: 10, fontSize: '0.82rem',
          color: status.startsWith('✅') ? 'var(--green)' : status.startsWith('Preview') ? 'var(--amber)' : 'var(--red)' }}>
          {status}
        </p>
      )}
      {result && result.rows && result.rows.length > 0 && (
        <div style={{ marginTop: 14, overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '0.72rem', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Match</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Date</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Sport</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Before Best Bet</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>After Best Bet</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Before Goals</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>After Goals</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Δ</th>
              </tr>
            </thead>
            <tbody>
              {result.rows.map((r, i) => (
                <tr key={i} style={{
                  background: r.error ? 'rgba(239,68,68,.08)' : r.changed ? 'rgba(16,185,129,.06)' : 'transparent',
                }}>
                  <td style={{ padding: '4px 8px', color: 'var(--text)' }}>
                    {r.home_team} vs {r.away_team}
                  </td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{r.match_date}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{r.sport}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{fmtPW(r.before.predicted_winner)}</td>
                  <td style={{ padding: '4px 8px', color: r.changed ? 'var(--green)' : 'var(--text-muted)' }}>
                    {r.error ? <span style={{ color: 'var(--red)' }}>ERR</span> : fmtPW(r.after.predicted_winner)}
                  </td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{fmtPW(r.before.predicted_goals)}</td>
                  <td style={{ padding: '4px 8px', color: r.changed ? 'var(--green)' : 'var(--text-muted)' }}>
                    {r.error ? '—' : fmtPW(r.after.predicted_goals)}
                  </td>
                  <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                    {r.error ? '⚠️' : r.changed ? '✏️' : '✓'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {result.rows.some(r => r.error) && (
            <details style={{ marginTop: 10 }}>
              <summary style={{ fontSize: '0.75rem', color: 'var(--text-muted)', cursor: 'pointer' }}>Error details</summary>
              <pre style={{ fontSize: '0.68rem', color: 'var(--red)', whiteSpace: 'pre-wrap', marginTop: 6 }}>
                {result.rows.filter(r => r.error).map(r =>
                  `${r.home_team} vs ${r.away_team}: ${r.error}`
                ).join('\n')}
              </pre>
            </details>
          )}
        </div>
      )}
      {result && result.rows && result.rows.length === 0 && (
        <p style={{ marginTop: 10, fontSize: '0.82rem', color: 'var(--text-muted)' }}>
          No PENDING or LIVE predictions in the database.
        </p>
      )}
    </div>
  )
}

function BackfillCorners({ headers }) {
  const [status,  setStatus]  = useState('')
  const [result,  setResult]  = useState(null)
  const [busy,    setBusy]    = useState(false)

  const run = async (apply) => {
    if (apply && !window.confirm(
      'This will attempt to fetch missing corner data for all finished football games. Continue?'
    )) return
    setBusy(true); setStatus(''); setResult(null)
    try {
      const r = await fetch('/admin/backfill-corners', {
        method: apply ? 'POST' : 'GET',
        headers: headers(),
      })
      const d = await r.json()
      setResult(d)
      if (!r.ok) { setStatus('Error: ' + (d.error || r.status)); return }
      if (apply) {
        setStatus(`✅ Done — ${d.filled} filled, ${d.failed} failed, ${d.skipped} skipped`)
      } else {
        setStatus(`Preview — ${d.missing_count} games missing corner data`)
      }
    } catch { setStatus('Connection error') }
    setBusy(false)
  }

  return (
    <div className="admin-card" style={{ marginTop: 20 }}>
      <h3 style={{ marginBottom: 6, color: 'var(--amber)', fontSize: '0.95rem', fontWeight: 700 }}>
        📐 Backfill Corner Data
      </h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: 14 }}>
        Finds finished football games with missing <code>actual_corners</code> and fetches them
        from TheSportsDB + ESPN. Run Preview first to see how many need filling.
      </p>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <button className="admin-btn" onClick={() => run(false)} disabled={busy}>
          Preview
        </button>
        <button className="admin-btn admin-btn-won" onClick={() => run(true)} disabled={busy}>
          Apply Backfill
        </button>
      </div>
      {busy && (
        <p style={{ marginTop: 10, fontSize: '0.82rem', color: 'var(--amber)' }}>
          Fetching corner data from external APIs…
        </p>
      )}
      {status && (
        <p style={{ marginTop: 10, fontSize: '0.82rem',
          color: status.startsWith('✅') ? 'var(--green)'
               : status.startsWith('Preview') ? 'var(--amber)' : 'var(--red)' }}>
          {status}
        </p>
      )}
      {result && result.rows && result.rows.length > 0 && (
        <div style={{ marginTop: 14, overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '0.72rem', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Match</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Date</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Corners</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {result.rows.map((r, i) => (
                <tr key={i} style={{
                  background: r.status === 'filled' ? 'rgba(16,185,129,.06)'
                            : r.status === 'failed'  ? 'rgba(239,68,68,.08)' : 'transparent',
                }}>
                  <td style={{ padding: '4px 8px', color: 'var(--text)' }}>
                    {r.home_team} vs {r.away_team}
                  </td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{r.match_date}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--green)' }}>
                    {r.corners != null ? r.corners : '—'}
                  </td>
                  <td style={{ padding: '4px 8px' }}>
                    {r.status === 'filled' ? '✅ Filled'
                     : r.status === 'failed' ? <span style={{ color: 'var(--red)' }}>❌ Not found</span>
                     : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function AuditPanel({ headers }) {
  const [audit,   setAudit]   = useState(null)
  const [rr,      setRr]      = useState(null)
  const [status,  setStatus]  = useState('')
  const [busy,    setBusy]    = useState(false)

  const runAudit = async () => {
    setBusy(true); setStatus(''); setAudit(null)
    try {
      const r = await fetch('/admin/audit-history', { headers: headers() })
      const d = await r.json()
      if (!r.ok) { setStatus('Error: ' + (d.error || r.status)); setBusy(false); return }
      setAudit(d)
      setStatus(`Audit: ${d.total_won_lost} WON/LOST, ${d.stale_pending} stale PENDING, ${d.remaining_dups} duplicate pairs`)
    } catch { setStatus('Connection error') }
    setBusy(false)
  }

  const runResolve = async () => {
    if (!window.confirm('Re-fetch ESPN results for all past-date PENDING records and update scores + status?')) return
    setBusy(true); setStatus(''); setRr(null)
    try {
      const r = await fetch('/admin/re-resolve-stale', { method: 'POST', headers: headers() })
      const d = await r.json()
      if (!r.ok) { setStatus('Error: ' + (d.error || r.status)); setBusy(false); return }
      setRr(d)
      setStatus(`✅ Re-resolved ${d.resolved} records, ${d.failed} not found`)
    } catch { setStatus('Connection error') }
    setBusy(false)
  }

  const cell = s => ({ padding: '4px 8px', color: s || 'var(--text-muted)', borderBottom: '1px solid rgba(255,255,255,.04)' })

  return (
    <div className="admin-card" style={{ marginTop: 20 }}>
      <h3 style={{ marginBottom: 6, color: 'var(--amber)', fontSize: '0.95rem', fontWeight: 700 }}>
        🔍 History Audit &amp; Re-resolve
      </h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: 14 }}>
        Audit shows all WON/LOST records, stale PENDING matches, remaining duplicates, and
        live bet-outcome totals so you can spot rogue or missing records.
        Re-resolve fetches ESPN results for every past-date PENDING football match.
      </p>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <button className="admin-btn" onClick={runAudit} disabled={busy}>Run Audit</button>
        <button className="admin-btn admin-btn-won" onClick={runResolve} disabled={busy}>
          Re-resolve Stale PENDING
        </button>
      </div>
      {status && (
        <p style={{ marginTop: 10, fontSize: '0.82rem',
          color: status.startsWith('✅') ? 'var(--green)' : 'var(--amber)' }}>
          {status}
        </p>
      )}

      {audit && (
        <div style={{ marginTop: 14 }}>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
            {Object.entries(audit.status_counts || {}).map(([k,v]) => (
              <div key={k} style={{ background: 'rgba(255,255,255,.04)', borderRadius: 8, padding: '8px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.2rem', fontWeight: 800, color: k==='WON'?'var(--green)':k==='LOST'?'var(--red)':'var(--amber)' }}>{v}</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{k}</div>
              </div>
            ))}
            <div style={{ background: 'rgba(255,255,255,.04)', borderRadius: 8, padding: '8px 16px', textAlign: 'center' }}>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--green)' }}>{audit.bet_outcomes_won}/{audit.bet_outcomes_total}</div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Outcomes Won/Total</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,.04)', borderRadius: 8, padding: '8px 16px', textAlign: 'center' }}>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--green)' }}>{audit.win_rate}%</div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Win Rate</div>
            </div>
          </div>

          {audit.stale_pending > 0 && (
            <details style={{ marginBottom: 12 }}>
              <summary style={{ fontSize: '0.8rem', color: 'var(--amber)', cursor: 'pointer' }}>
                ⚠️ {audit.stale_pending} stale PENDING records (past-date, unresolved)
              </summary>
              <table style={{ width: '100%', fontSize: '0.7rem', borderCollapse: 'collapse', marginTop: 8 }}>
                <thead><tr style={{ color: 'var(--text-muted)' }}>
                  <th style={cell()}>Match</th><th style={cell()}>Date</th>
                  <th style={cell()}>Sport</th><th style={cell()}>Best Bet</th>
                </tr></thead>
                <tbody>
                  {audit.stale_pending_records.map((r,i) => (
                    <tr key={i}>
                      <td style={cell('var(--text)')}>{r.home_team} vs {r.away_team}</td>
                      <td style={cell()}>{r.match_date}</td>
                      <td style={cell()}>{r.sport}</td>
                      <td style={cell()}>{r.predicted_winner}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          )}

          {audit.remaining_dups > 0 && (
            <p style={{ fontSize: '0.8rem', color: 'var(--red)' }}>
              ⚠️ {audit.remaining_dups} duplicate pair(s) still in DB — run Deduplicate again.
            </p>
          )}

          <details>
            <summary style={{ fontSize: '0.8rem', color: 'var(--text-muted)', cursor: 'pointer' }}>
              All {audit.total_won_lost} WON/LOST records
            </summary>
            <table style={{ width: '100%', fontSize: '0.68rem', borderCollapse: 'collapse', marginTop: 8 }}>
              <thead><tr style={{ color: 'var(--text-muted)' }}>
                <th style={cell()}>Match</th><th style={cell()}>Date</th>
                <th style={cell()}>Status</th><th style={cell()}>Score</th>
                <th style={cell()}>Corners</th><th style={cell()}>Best Bet</th>
              </tr></thead>
              <tbody>
                {audit.won_lost_records.map((r,i) => (
                  <tr key={i} style={{ background: r.result_status==='WON'?'rgba(16,185,129,.04)':'transparent' }}>
                    <td style={cell('var(--text)')}>{r.home_team} vs {r.away_team}</td>
                    <td style={cell()}>{r.match_date}</td>
                    <td style={{ padding:'4px 8px', color: r.result_status==='WON'?'var(--green)':'var(--red)' }}>{r.result_status}</td>
                    <td style={cell()}>{r.actual_home_score != null ? `${r.actual_home_score}–${r.actual_away_score}` : '—'}</td>
                    <td style={cell()}>{r.actual_corners ?? '—'}</td>
                    <td style={cell()}>{r.predicted_winner}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </div>
      )}

      {rr && (
        <div style={{ marginTop: 14 }}>
          <p style={{ fontSize: '0.8rem', color: 'var(--green)', marginBottom: 8 }}>
            ✅ Resolved {rr.resolved} · Failed {rr.failed}
          </p>
          {rr.resolved_records.length > 0 && (
            <table style={{ width: '100%', fontSize: '0.7rem', borderCollapse: 'collapse' }}>
              <thead><tr style={{ color: 'var(--text-muted)' }}>
                <th style={cell()}>Match</th><th style={cell()}>Date</th>
                <th style={cell()}>Score</th><th style={cell()}>Corners</th><th style={cell()}>Status</th>
              </tr></thead>
              <tbody>
                {rr.resolved_records.map((r,i) => (
                  <tr key={i}>
                    <td style={cell('var(--text)')}>{r.teams}</td>
                    <td style={cell()}>{r.date}</td>
                    <td style={cell()}>{r.score}</td>
                    <td style={cell()}>{r.corners ?? '—'}</td>
                    <td style={{ padding:'4px 8px', color: r.status==='WON'?'var(--green)':'var(--red)' }}>{r.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {rr.failed_records.length > 0 && (
            <details style={{ marginTop: 8 }}>
              <summary style={{ fontSize: '0.75rem', color: 'var(--text-muted)', cursor: 'pointer' }}>
                {rr.failed} not found
              </summary>
              <ul style={{ fontSize: '0.7rem', color: 'var(--text-muted)', paddingLeft: 16, marginTop: 6 }}>
                {rr.failed_records.map((r,i) => (
                  <li key={i}>{r.teams} ({r.date}) — {r.reason}</li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  )
}

function RederivePanel({ headers }) {
  const [status,  setStatus]  = useState('')
  const [result,  setResult]  = useState(null)
  const [busy,    setBusy]    = useState(false)

  const run = async (apply) => {
    if (apply && !window.confirm(
      `Apply fixes to ${result?.mismatches ?? '?'} mismatched record(s)? This updates result_status in the DB.`
    )) return
    setBusy(true); setStatus(''); if (!apply) setResult(null)
    try {
      const r = await fetch('/admin/rederive-statuses', {
        method: apply ? 'POST' : 'GET',
        headers: headers(),
      })
      const d = await r.json()
      setResult(d)
      if (!r.ok) { setStatus('Error: ' + (d.error || r.status)); setBusy(false); return }
      setStatus(apply
        ? `✅ Fixed ${d.fixed} record(s) out of ${d.checked} checked`
        : `Preview — ${d.mismatches} mismatch(es) out of ${d.checked} records`)
    } catch { setStatus('Connection error') }
    setBusy(false)
  }

  const statusColor = s =>
    s === 'WON' ? 'var(--green)' : s === 'LOST' ? 'var(--red)' : 'var(--text-muted)'

  return (
    <div className="admin-card" style={{ marginTop: 20 }}>
      <h3 style={{ marginBottom: 6, color: 'var(--amber)', fontSize: '0.95rem', fontWeight: 700 }}>
        ✏️ Re-derive WON/LOST Statuses
      </h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: 14 }}>
        Re-computes the correct <code>result_status</code> for every finished record using the
        updated logic (BTTS branch + corner O/U with actual corner data). Preview first to see
        mismatches, then apply.
      </p>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <button className="admin-btn" onClick={() => run(false)} disabled={busy}>
          Preview Mismatches
        </button>
        <button
          className="admin-btn admin-btn-won"
          onClick={() => run(true)}
          disabled={busy || !result || result.mismatches === 0}
        >
          Apply Fixes
        </button>
      </div>
      {status && (
        <p style={{ marginTop: 10, fontSize: '0.82rem',
          color: status.startsWith('✅') ? 'var(--green)'
               : status.startsWith('Preview') ? 'var(--amber)' : 'var(--red)' }}>
          {status}
        </p>
      )}
      {result && result.records && result.records.length > 0 && (
        <div style={{ marginTop: 14, overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '0.72rem', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Match</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Date</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Best Bet</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Score</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Corners</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Was</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>Should Be</th>
              </tr>
            </thead>
            <tbody>
              {result.records.map((r, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? 'rgba(255,255,255,.03)' : 'transparent' }}>
                  <td style={{ padding: '4px 8px', color: 'var(--text)' }}>{r.teams}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{r.match_date}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{r.best_bet}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{r.score}</td>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{r.corners ?? '—'}</td>
                  <td style={{ padding: '4px 8px', color: statusColor(r.was), fontWeight: 700 }}>{r.was}</td>
                  <td style={{ padding: '4px 8px', color: statusColor(r.should_be), fontWeight: 700 }}>{r.should_be}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {result && result.mismatches === 0 && (
        <p style={{ marginTop: 10, fontSize: '0.82rem', color: 'var(--green)' }}>
          ✅ All {result.checked} records have correct statuses — no fixes needed.
        </p>
      )}
    </div>
  )
}

function DbCorrections({ headers }) {
  const [status, setStatus] = useState('')
  const [records, setRecords] = useState(null)
  const [busy, setBusy] = useState(false)

  const preview = async () => {
    setBusy(true); setStatus(''); setRecords(null)
    try {
      const r = await fetch('/admin/fix-history-records', { headers: headers() })
      const d = await r.json()
      setRecords(d.records || [])
      setStatus(r.ok ? 'preview' : 'Error: ' + (d.error || r.status))
    } catch { setStatus('Connection error') }
    setBusy(false)
  }

  const apply = async () => {
    if (!window.confirm('Apply DB corrections? This will update predicted_winner and result_status for the targeted records.')) return
    setBusy(true); setStatus('')
    try {
      const r = await fetch('/admin/fix-history-records', { method: 'POST', headers: headers() })
      const d = await r.json()
      setRecords(d.records || [])
      setStatus(r.ok ? '✅ Corrections applied' : 'Error: ' + (d.error || r.status))
    } catch { setStatus('Connection error') }
    setBusy(false)
  }

  return (
    <div className="admin-card" style={{ marginTop: 20 }}>
      <h3 style={{ marginBottom: 14, color: 'var(--amber)', fontSize: '0.95rem', fontWeight: 700 }}>
        🔧 DB Corrections
      </h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: 14 }}>
        Fix mis-classified prediction records (Washington Mystics vs Toronto Tempo, USA vs Paraguay, GSV vs Seattle Storm).
      </p>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <button className="admin-btn" onClick={preview} disabled={busy}>
          Preview
        </button>
        <button className="admin-btn admin-btn-won" onClick={apply} disabled={busy}>
          Apply Fixes
        </button>
      </div>
      {status && (
        <p style={{ marginTop: 10, fontSize: '0.82rem', color: status.startsWith('✅') ? 'var(--green)' : 'var(--red)' }}>
          {status}
        </p>
      )}
      {records && (
        <pre style={{ marginTop: 10, fontSize: '0.72rem', color: 'var(--text-muted)', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
          {JSON.stringify(records, null, 2)}
        </pre>
      )}
    </div>
  )
}

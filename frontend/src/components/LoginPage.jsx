import { useState } from 'react'

function AuthHeader({ navigate }) {
  return (
    <header className="auth-header">
      <button className="auth-back" onClick={() => navigate('/')}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M19 12H5M12 5l-7 7 7 7"/></svg>
        Back
      </button>
      <div className="auth-logo">⚽ Az-Predictions</div>
    </header>
  )
}

export default function LoginPage({ navigate, onAuth }) {
  const [form, setForm]     = useState({ email: '', password: '' })
  const [error, setError]   = useState('')
  const [loading, setLoading] = useState(false)

  const set = k => e => setForm(f => ({ ...f, [k]: e.target.value }))

  const submit = async e => {
    e.preventDefault()
    setLoading(true); setError('')
    try {
      const r = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      let data = {}
      try { data = await r.json() } catch { /* non-JSON response */ }
      if (!r.ok) {
        setError(data.error || `Server error (${r.status}) — please try again`)
        return
      }
      onAuth(data)
      navigate('/')
    } catch (err) { setError('Network error — check your connection and try again') }
    finally { setLoading(false) }
  }

  return (
    <div className="auth-page">
      <AuthHeader navigate={navigate} />
      <div className="auth-card">
        <h2 className="auth-title">Welcome back</h2>
        <p className="auth-sub">Sign in to access your predictions</p>
        <form onSubmit={submit} className="auth-form">
          <div className="field">
            <label>Email</label>
            <input type="email" value={form.email} onChange={set('email')} required placeholder="you@email.com" autoFocus />
          </div>
          <div className="field">
            <label>Password</label>
            <input type="password" value={form.password} onChange={set('password')} required placeholder="Your password" />
          </div>
          {error && <div className="auth-error">⚠ {error}</div>}
          <button type="submit" className="btn-primary btn-full" disabled={loading}>
            {loading ? <span className="btn-spinner" /> : null}
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
        <p className="auth-switch">
          No account?{' '}
          <button className="auth-link" onClick={() => navigate('/register')}>Create one free</button>
        </p>
      </div>
    </div>
  )
}

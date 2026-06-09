import { useState } from 'react'

function AuthHeader({ navigate }) {
  return (
    <header className="auth-header">
      <button className="auth-back" onClick={() => navigate('/')}>← Back</button>
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
      const data = await r.json()
      if (!r.ok) { setError(data.error || 'Login failed'); return }
      onAuth(data)
      navigate('/')
    } catch { setError('Network error — please try again') }
    finally { setLoading(false) }
  }

  return (
    <div className="auth-page">
      <AuthHeader navigate={navigate} />
      <div className="auth-card">
        <h2 className="auth-title">Welcome back</h2>
        <p className="auth-sub">Log in to access your predictions</p>
        <form onSubmit={submit} className="auth-form">
          <div className="field">
            <label>Email</label>
            <input type="email" value={form.email} onChange={set('email')} required placeholder="you@email.com" />
          </div>
          <div className="field">
            <label>Password</label>
            <input type="password" value={form.password} onChange={set('password')} required placeholder="••••••••" />
          </div>
          {error && <div className="auth-error">{error}</div>}
          <button type="submit" className="btn-primary btn-full" disabled={loading}>
            {loading ? 'Logging in…' : 'Log in'}
          </button>
        </form>
        <p className="auth-switch">
          No account?{' '}
          <button className="auth-link" onClick={() => navigate('/register')}>Sign up free</button>
        </p>
      </div>
    </div>
  )
}

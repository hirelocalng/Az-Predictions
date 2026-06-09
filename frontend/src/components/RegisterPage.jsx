import { useState } from 'react'

function AuthHeader({ navigate }) {
  return (
    <header className="auth-header">
      <button className="auth-back" onClick={() => navigate('/')}>← Back</button>
      <div className="auth-logo">⚽ Az-Predictions</div>
    </header>
  )
}

export default function RegisterPage({ navigate, onAuth }) {
  const [form, setForm]       = useState({ name: '', email: '', password: '', confirm: '' })
  const [error, setError]     = useState('')
  const [loading, setLoading] = useState(false)

  const set = k => e => setForm(f => ({ ...f, [k]: e.target.value }))

  const submit = async e => {
    e.preventDefault()
    if (form.password !== form.confirm) { setError('Passwords do not match'); return }
    if (form.password.length < 6) { setError('Password must be at least 6 characters'); return }
    setLoading(true); setError('')
    try {
      const r = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: form.name, email: form.email, password: form.password }),
      })
      const data = await r.json()
      if (!r.ok) { setError(data.error || 'Registration failed'); return }
      onAuth(data)
      navigate('/')
    } catch { setError('Network error — please try again') }
    finally { setLoading(false) }
  }

  return (
    <div className="auth-page">
      <AuthHeader navigate={navigate} />
      <div className="auth-card">
        <h2 className="auth-title">Create your account</h2>
        <p className="auth-sub">Get 5 free daily picks. Upgrade for unlimited access.</p>
        <form onSubmit={submit} className="auth-form">
          <div className="field">
            <label>Your name</label>
            <input type="text" value={form.name} onChange={set('name')} required placeholder="Full name" />
          </div>
          <div className="field">
            <label>Email</label>
            <input type="email" value={form.email} onChange={set('email')} required placeholder="you@email.com" />
          </div>
          <div className="field">
            <label>Password</label>
            <input type="password" value={form.password} onChange={set('password')} required placeholder="Min 6 characters" />
          </div>
          <div className="field">
            <label>Confirm password</label>
            <input type="password" value={form.confirm} onChange={set('confirm')} required placeholder="Repeat password" />
          </div>
          {error && <div className="auth-error">{error}</div>}
          <button type="submit" className="btn-primary btn-full" disabled={loading}>
            {loading ? 'Creating account…' : 'Create account'}
          </button>
        </form>
        <p className="auth-switch">
          Have an account?{' '}
          <button className="auth-link" onClick={() => navigate('/login')}>Log in</button>
        </p>
      </div>
    </div>
  )
}

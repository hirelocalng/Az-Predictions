import { useState, useEffect } from 'react'

const PLANS = [
  {
    id: '1week', name: '1 WEEK', price: '₦1,500', usd: '≈ $1 · perfect for a matchweek',
    desc: 'Full access for 7 days',
  },
  {
    id: 'monthly', name: 'Monthly', price: '₦5,000', usd: '≈ $5 / month',
    desc: 'Full access for 1 month',
  },
  {
    id: '3month', name: '3 Months', price: '₦15,000', usd: '≈ $12 total · save 20%',
    desc: 'Best value — pay once, enjoy 3 months', popular: true,
  },
]

const FEATURES = [
  'All World Cup 2026 group stage fixtures',
  'Live international matches & scores',
  'Daily club football tips',
  'Unlimited daily predictions (no 5-pick limit)',
  'Real-time live score tracking',
  'Prediction history & win rate stats',
]

function AuthHeader({ navigate }) {
  return (
    <header className="auth-header">
      <button className="auth-back" onClick={() => navigate('/')}>← Back</button>
      <div className="auth-logo">⚽ Az-Predictions</div>
    </header>
  )
}

export default function SubscribePage({ navigate, auth, onAuth }) {
  const [paying, setPaying]   = useState(null)
  const [message, setMessage] = useState('')
  const [msgOk, setMsgOk]     = useState(false)

  useEffect(() => {
    // Korapay redirects back to /subscribe?reference=... after checkout
    const params = new URLSearchParams(window.location.search)
    const ref = params.get('reference') || params.get('trxref')
    if (ref && auth?.token) verifyPayment(ref)
  }, [])

  const verifyPayment = async ref => {
    setMessage('Verifying payment…'); setMsgOk(false)
    try {
      const r = await fetch(`/api/payment/verify?reference=${ref}`, {
        headers: { Authorization: `Bearer ${auth.token}` },
      })
      const data = await r.json()
      if (data.success && data.user) {
        onAuth({ ...auth, user: data.user })
        setMessage('🎉 Payment successful! Welcome to Premium.')
        setMsgOk(true)
        // Clean the reference from the URL without a reload
        window.history.replaceState({}, '', '/subscribe')
        setTimeout(() => navigate('/'), 2500)
      } else {
        setMessage(data.error || 'Payment could not be verified.')
      }
    } catch { setMessage('Verification error — please contact support.') }
  }

  const pay = async plan => {
    if (!auth) { navigate('/login'); return }
    setPaying(plan.id)
    setMessage('')
    try {
      const r = await fetch('/api/payment/initialize', {
        method:  'POST',
        headers: {
          'Content-Type':  'application/json',
          'Authorization': `Bearer ${auth.token}`,
        },
        body: JSON.stringify({ plan: plan.id }),
      })
      const data = await r.json()
      if (data.checkout_url) {
        window.location.href = data.checkout_url
      } else {
        setMessage(data.error || 'Could not start payment. Please try again.')
        setPaying(null)
      }
    } catch {
      setMessage('Payment system unavailable. Contact support.')
      setPaying(null)
    }
  }

  if (auth?.user?.is_premium) {
    return (
      <div className="auth-page">
        <AuthHeader navigate={navigate} />
        <div className="auth-card" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '3.5rem', margin: '12px 0' }}>👑</div>
          <h2 className="auth-title">You're Premium!</h2>
          <p className="auth-sub">You already have full access to all predictions.</p>
          {auth.user.premium_until && (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: 8 }}>
              Active until {new Date(auth.user.premium_until).toLocaleDateString()}
            </p>
          )}
          <button className="btn-primary" style={{ marginTop: 24 }} onClick={() => navigate('/')}>
            View Predictions →
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="auth-page sub-page">
      <AuthHeader navigate={navigate} />
      <div className="sub-container">
        <div className="sub-header">
          <h2 className="auth-title" style={{ fontSize: '1.9rem' }}>Unlock Premium Access</h2>
          <p className="auth-sub">World Cup · Internationals · Club tips · No limits</p>
        </div>

        {message && (
          <div className={`auth-message${msgOk ? ' ok' : ''}`}>{message}</div>
        )}

        <div className="plan-grid">
          {PLANS.map(plan => (
            <div key={plan.id} className={`plan-card${plan.popular ? ' popular' : ''}`}>
              {plan.popular && <div className="plan-badge">Best Value</div>}
              <div className="plan-name">{plan.name}</div>
              <div className="plan-price">{plan.price}</div>
              <div className="plan-usd">{plan.usd}</div>
              <div className="plan-desc">{plan.desc}</div>
              <ul className="plan-features">
                {FEATURES.map(f => <li key={f}>✅ {f}</li>)}
              </ul>
              <button
                className="btn-primary btn-full"
                style={{ marginTop: 'auto' }}
                onClick={() => pay(plan)}
                disabled={paying === plan.id}
              >
                {paying === plan.id ? 'Redirecting…' : `Pay ${plan.price}`}
              </button>
            </div>
          ))}
        </div>

        {!auth && (
          <p className="auth-switch" style={{ textAlign: 'center', marginTop: 28 }}>
            Already subscribed?{' '}
            <button className="auth-link" onClick={() => navigate('/login')}>Log in to restore access</button>
          </p>
        )}
        <p style={{ textAlign:'center', marginTop:32, fontSize:'0.68rem', color:'var(--text-subtle)' }}>
          Secured by Korapay · Nigerian payments (NGN) · Cancel anytime
        </p>
      </div>
    </div>
  )
}

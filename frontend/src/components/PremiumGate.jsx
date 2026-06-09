import { useAuth } from '../contexts/AuthContext.jsx'

export default function PremiumGate() {
  const { auth, navigate } = useAuth()
  return (
    <div className="premium-gate">
      <div className="gate-crown">👑</div>
      <h3 className="gate-title">Premium Feature</h3>
      <p className="gate-desc">
        Unlock all World Cup fixtures, live internationals, and club football tips.
      </p>
      <div className="gate-pricing">
        <span>Monthly — ₦5,000</span>
        <span className="gate-sep">·</span>
        <span>3 Months — ₦15,000</span>
      </div>
      <div className="gate-actions">
        <button className="btn-primary" onClick={() => navigate('/subscribe')}>
          Upgrade to Premium
        </button>
        {!auth && (
          <button className="btn-outline" style={{ marginLeft: 10 }} onClick={() => navigate('/login')}>
            Log in
          </button>
        )}
      </div>
    </div>
  )
}

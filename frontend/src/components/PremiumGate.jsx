import { useAuth } from '../contexts/AuthContext.jsx'

export default function PremiumGate({ lockedCount }) {
  const { auth, navigate } = useAuth()
  const countText = lockedCount ? `${lockedCount} more prediction${lockedCount === 1 ? '' : 's'}` : 'more predictions'
  return (
    <div className="premium-gate">
      <div className="gate-crown">👑</div>
      <h3 className="gate-title">Unlock {countText}</h3>
      <p className="gate-desc">
        Upgrade to Premium to see all predictions with no limits across every section.
      </p>
      <div className="gate-pricing">
        <span>1 Week — ₦1,500</span>
        <span className="gate-sep">·</span>
        <span>Monthly — ₦5,000</span>
        <span className="gate-sep">·</span>
        <span>3 Months — ₦15,000</span>
      </div>
      <div className="gate-actions">
        <button className="btn-primary" onClick={() => navigate('/subscribe')}>
          Upgrade to Premium
        </button>
        {!auth && (
          <button className="btn-outline" onClick={() => navigate('/login')}>
            Log in
          </button>
        )}
      </div>
    </div>
  )
}

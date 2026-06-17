import { useEffect } from 'react'

export default function SaveLoginModal({ onClose, navigate }) {
  useEffect(() => {
    const close = e => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', close)
    return () => document.removeEventListener('keydown', close)
  }, [onClose])

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        <div className="modal-icon">🔖</div>
        <h2 className="modal-title">Save your predictions</h2>
        <p className="modal-body">
          Create a free account to bookmark predictions and track your personal win rate.
        </p>
        <div className="modal-actions">
          <button
            className="btn-primary btn-full"
            onClick={() => { onClose(); navigate('/register') }}
          >
            Sign up free
          </button>
          <button
            className="btn-outline btn-full"
            onClick={() => { onClose(); navigate('/login') }}
          >
            Log in
          </button>
        </div>
      </div>
    </div>
  )
}

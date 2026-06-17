import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext.jsx'
import SaveLoginModal from './SaveLoginModal.jsx'

const HeartIcon = ({ filled }) => (
  <svg
    width="16" height="16" viewBox="0 0 24 24"
    fill={filled ? 'var(--red)' : 'none'}
    stroke={filled ? 'var(--red)' : 'currentColor'}
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
  </svg>
)

export default function SavePredictionButton({ matchId }) {
  const { auth, navigate } = useAuth()
  const [saved,     setSaved]     = useState(false)
  const [loading,   setLoading]   = useState(false)
  const [showModal, setShowModal] = useState(false)

  useEffect(() => {
    if (!auth?.token || !matchId) return
    fetch(`/api/predictions/${encodeURIComponent(matchId)}/is-saved`, {
      headers: { Authorization: `Bearer ${auth.token}` },
    })
      .then(r => r.json())
      .then(d => setSaved(Boolean(d.saved)))
      .catch(() => {})
  }, [auth?.token, matchId])

  const toggle = async e => {
    e.stopPropagation()
    if (!auth?.token) { setShowModal(true); return }
    if (loading || !matchId) return
    setLoading(true)
    try {
      const r = await fetch(`/api/predictions/${encodeURIComponent(matchId)}/save`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${auth.token}` },
      })
      if (r.ok) {
        const d = await r.json()
        setSaved(Boolean(d.saved))
      }
    } catch { /* ignore */ } finally {
      setLoading(false)
    }
  }

  if (!matchId) return null

  return (
    <>
      <button
        className={`save-btn${saved ? ' save-btn--saved' : ''}`}
        onClick={toggle}
        disabled={loading}
        aria-label={saved ? 'Unsave prediction' : 'Save prediction'}
        title={saved ? 'Saved' : 'Save prediction'}
      >
        <HeartIcon filled={saved} />
      </button>
      {showModal && (
        <SaveLoginModal onClose={() => setShowModal(false)} navigate={navigate} />
      )}
    </>
  )
}

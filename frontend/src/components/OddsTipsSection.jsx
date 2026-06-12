import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext.jsx'

const STATUS_BADGE = {
  won:     { label: '✅ WON',     cls: 'odds-badge-won'     },
  lost:    { label: '❌ LOST',    cls: 'odds-badge-lost'    },
  pending: { label: '⏳ PENDING', cls: 'odds-badge-pending' },
}

function TipCard({ tip }) {
  const badge = STATUS_BADGE[tip.status] || STATUS_BADGE.pending
  return (
    <div className={`odds-tip-card odds-tip-${tip.status}`}>
      <div className="odds-tip-header">
        <span className="odds-tip-time">{tip.time || '--:--'}</span>
        <span className="odds-tip-league">{tip.league || 'Match'}</span>
        <span className={`odds-badge ${badge.cls}`}>{badge.label}</span>
      </div>
      <div className="odds-tip-teams">
        <span className="odds-tip-home">{tip.team_home}</span>
        {tip.score
          ? <span className="odds-tip-score">{tip.score}</span>
          : <span className="odds-tip-vs">vs</span>
        }
        <span className="odds-tip-away">{tip.team_away}</span>
      </div>
      <div className="odds-tip-footer">
        <span className="odds-tip-desc">{tip.tip_description}</span>
        {tip.odd_value != null && (
          <span className="odds-tip-odd">@{Number(tip.odd_value).toFixed(2)}</span>
        )}
      </div>
    </div>
  )
}

function VipLockedPlaceholder({ navigate, auth }) {
  return (
    <div className="odds-vip-locked">
      <div className="odds-vip-blur-wrap" aria-hidden="true">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="odds-tip-card odds-tip-blur">
            <div className="odds-tip-header">
              <span className="odds-tip-time">??:??</span>
              <span className="odds-tip-league">VIP Match</span>
              <span className="odds-badge odds-badge-pending">⏳ PENDING</span>
            </div>
            <div className="odds-tip-teams">
              <span className="odds-tip-home">Team A</span>
              <span className="odds-tip-vs">vs</span>
              <span className="odds-tip-away">Team B</span>
            </div>
            <div className="odds-tip-footer">
              <span className="odds-tip-desc">VIP Tip — Hidden</span>
              <span className="odds-tip-odd">@?.??</span>
            </div>
          </div>
        ))}
      </div>
      <div className="odds-vip-gate">
        <div className="gate-crown">👑</div>
        <h3 className="gate-title">VIP Tips Locked</h3>
        <p className="gate-desc">Upgrade to Premium to unlock today's expert VIP odds tips.</p>
        <div className="gate-pricing">
          <span>Monthly — ₦5,000</span>
          <span className="gate-sep">·</span>
          <span>3 Months — ₦15,000</span>
        </div>
        <div className="gate-actions">
          <button className="btn-primary" onClick={() => navigate('/subscribe')}>Upgrade to VIP</button>
          {!auth && (
            <button className="btn-outline" onClick={() => navigate('/login')}>Log in</button>
          )}
        </div>
      </div>
    </div>
  )
}

export default function OddsTipsSection() {
  const { auth, navigate } = useAuth()
  const [freeTips,  setFreeTips]  = useState([])
  const [vipTips,   setVipTips]   = useState([])
  const [vipLocked, setVipLocked] = useState(true)
  const [loading,   setLoading]   = useState(true)

  useEffect(() => {
    const headers = auth?.token ? { Authorization: `Bearer ${auth.token}` } : {}
    Promise.all([
      fetch('/api/odds-tips/free').then(r => r.json()),
      fetch('/api/odds-tips/vip', { headers }).then(r => r.json()),
    ])
      .then(([free, vip]) => {
        setFreeTips(free.tips  || [])
        setVipTips(vip.tips    || [])
        setVipLocked(vip.locked !== false)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [auth?.token])

  const accumOdd = freeTips.length > 0
    ? freeTips.reduce((acc, t) => acc * (t.odd_value || 1), 1)
    : null

  return (
    <section className="section odds-section" id="odds-tips">
      <div className="section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
        <div>
          <div className="section-label">⚽ Betting Tips</div>
          <h2 className="section-title">Daily 5+ Odds</h2>
          <p className="section-sub">Today's handpicked tips — free picks &amp; VIP selections</p>
        </div>
        {accumOdd !== null && (
          <div className="odds-accum">
            <span className="odds-accum-label">Free Accumulator</span>
            <span className="odds-accum-val">@{accumOdd.toFixed(2)}</span>
          </div>
        )}
      </div>

      {loading ? (
        <div className="odds-loading">Loading today's tips…</div>
      ) : (
        <>
          <div className="odds-tier-row">
            <span className="odds-tier-badge odds-tier-free">🆓 Free Tips</span>
          </div>
          {freeTips.length === 0 ? (
            <div className="odds-empty">
              <span>⚽</span>
              <span>No free tips posted yet for today. Check back soon!</span>
            </div>
          ) : (
            <div className="odds-grid">
              {freeTips.map(t => <TipCard key={t.id} tip={t} />)}
            </div>
          )}

          <div className="odds-divider" />

          <div className="odds-tier-row">
            <span className="odds-tier-badge odds-tier-vip">👑 VIP Tips</span>
          </div>
          {vipLocked ? (
            <VipLockedPlaceholder navigate={navigate} auth={auth} />
          ) : vipTips.length === 0 ? (
            <div className="odds-empty">
              <span>👑</span>
              <span>No VIP tips posted yet for today.</span>
            </div>
          ) : (
            <div className="odds-grid">
              {vipTips.map(t => <TipCard key={t.id} tip={t} />)}
            </div>
          )}
        </>
      )}
    </section>
  )
}

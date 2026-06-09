import { useState, useEffect } from 'react'
import PremiumGate from './PremiumGate.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'

export default function BestBetSection() {
  const { auth }   = useAuth()
  const isPremium  = Boolean(auth?.user?.is_premium)
  const [data,     setData]    = useState(null)
  const [loading,  setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/best-bet')
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  return (
    <section className="section" id="best-bet">
      {/* ── Best Bet (FREE) ─────────────────────────────────────────────── */}
      <div className="section-header">
        <div className="section-label" style={{ color: 'var(--amber)' }}>⭐ Today's Best Pick · Free</div>
        <h2 className="section-title">Best Bet <span className="amber">of the Day</span></h2>
      </div>

      {loading && (
        <div className="skeleton" style={{ height: 180, borderRadius: 20, maxWidth: 700, margin: '0 auto' }} />
      )}

      {!loading && !data?.best_bet && (
        <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0' }}>
          No fixtures analysed yet — check back soon.
        </p>
      )}

      {!loading && data?.best_bet && (
        <div className="best-bet-card">
          <div className="bb-star-badge">⭐ BEST BET</div>
          <div className="bb-body">
            <div className="bb-left">
              <div className="bb-league">{data.best_bet.league || 'Football'}</div>
              <div className="bb-match">{data.best_bet.match}</div>
              <div className="bb-pick">{data.best_bet.pick}</div>
            </div>
            <div className="bb-right">
              <div className="bb-conf-label">Confidence</div>
              <div className="bb-conf-num">{(data.best_bet.prob * 100).toFixed(1)}%</div>
              <div className="bb-conf-bar">
                <div className="bb-conf-fill" style={{ width: `${Math.round(data.best_bet.prob * 100)}%` }} />
              </div>
            </div>
          </div>
          <div className="bb-footer">AI model output · Not financial advice · Gamble responsibly</div>
        </div>
      )}

      {/* ── Daily Accumulator (PREMIUM) ─────────────────────────────────── */}
      <div className="section-header" style={{ marginTop: 64 }}>
        <div className="section-label" style={{ color: 'var(--amber)' }}>🎯 Daily Accumulator · Premium</div>
        <h2 className="section-title">Daily <span className="amber">Accumulator</span></h2>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: 8 }}>
          Top 3 picks from different matches — combined into one slip
        </p>
      </div>

      {!isPremium
        ? <PremiumGate />
        : loading
          ? <div className="skeleton" style={{ height: 220, borderRadius: 16, maxWidth: 700, margin: '0 auto' }} />
          : !data?.accumulator?.length
            ? <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px 0' }}>No accumulator available today.</p>
            : (
              <div className="acca-card">
                {data.accumulator.map((pick, i) => (
                  <div key={i} className="acca-row">
                    <div className="acca-num">{i + 1}</div>
                    <div className="acca-details">
                      <div className="acca-match">{pick.match}</div>
                      <div className="acca-pick-label">{pick.pick}</div>
                      {pick.league && <div className="acca-league">{pick.league}</div>}
                    </div>
                    <div className="acca-conf">{(pick.prob * 100).toFixed(1)}%</div>
                  </div>
                ))}
                <div className="acca-combined">
                  <span className="acca-combined-label">Combined confidence</span>
                  <span className="acca-combined-num">{(data.combined_prob * 100).toFixed(1)}%</span>
                </div>
              </div>
            )
      }
    </section>
  )
}

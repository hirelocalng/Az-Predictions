import { useState, useEffect, useRef } from 'react'
import PremiumGate from './PremiumGate.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'

const isFinished = utc_kickoff =>
  utc_kickoff
    ? Date.now() > new Date(utc_kickoff).getTime() + 2 * 60 * 60 * 1000
    : false

export default function BestBetSection() {
  const { auth }   = useAuth()
  const isPremium  = Boolean(auth?.user?.is_premium)
  const [data,     setData]    = useState(null)
  const [loading,  setLoading] = useState(true)
  const retryRef   = useRef(null)

  const fetchBestBet = () =>
    fetch('/api/best-bet')
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))

  // Initial fetch + poll every 3 minutes
  useEffect(() => {
    fetchBestBet()
    const t = setInterval(fetchBestBet, 3 * 60 * 1000)
    return () => { clearInterval(t); clearTimeout(retryRef.current) }
  }, [])

  const bestBet  = data?.best_bet
  const bestDone = bestBet && isFinished(bestBet.utc_kickoff)

  // If the cached pick just finished, refetch after 10 s so backend serves the next one
  useEffect(() => {
    if (bestDone) {
      clearTimeout(retryRef.current)
      retryRef.current = setTimeout(fetchBestBet, 10_000)
    }
  }, [bestDone])

  // Filter accumulator picks whose match hasn't finished yet
  const acca = (data?.accumulator || []).filter(p => !isFinished(p.utc_kickoff))
  const combinedProb = acca.reduce((acc, p) => acc * p.prob, 1.0)

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

      {/* bestDone: cached pick is finished but next one is loading */}
      {!loading && bestDone && (
        <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0' }}>
          Finding the next best pick…
        </p>
      )}

      {/* No picks at all */}
      {!loading && !bestBet && (
        <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0' }}>
          No upcoming fixtures to analyse yet — check back soon.
        </p>
      )}

      {!loading && bestBet && !bestDone && (
        <div className="best-bet-card">
          <div className="bb-star-badge">⭐ BEST BET</div>
          <div className="bb-body">
            <div className="bb-left">
              <div className="bb-league">{bestBet.league || 'Football'}</div>
              <div className="bb-match">{bestBet.match}</div>
              <div className="bb-pick">{bestBet.pick}</div>
            </div>
            <div className="bb-right">
              <div className="bb-conf-label">Confidence</div>
              <div className="bb-conf-num">{(bestBet.prob * 100).toFixed(1)}%</div>
              <div className="bb-conf-bar">
                <div className="bb-conf-fill" style={{ width: `${Math.round(bestBet.prob * 100)}%` }} />
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
          : acca.length === 0
            ? <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px 0' }}>
                No accumulator picks remaining today.
              </p>
            : (
              <div className="acca-card">
                {acca.map((pick, i) => (
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
                  <span className="acca-combined-num">{(combinedProb * 100).toFixed(1)}%</span>
                </div>
              </div>
            )
      }
    </section>
  )
}

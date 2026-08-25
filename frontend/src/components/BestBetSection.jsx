import { useState, useEffect, useRef } from 'react'
import PremiumGate from './PremiumGate.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'

// A pick is no longer selectable once its match has kicked off -- not
// once it's presumed finished. The old version added a 2-hour grace
// period on top of kickoff, which let an already-started (or already
// long-finished) match keep displaying as "today's pick" for up to two
// hours after the backend itself would have excluded it (2026-08-24 fix).
const hasKickedOff = utc_kickoff =>
  utc_kickoff ? Date.now() > new Date(utc_kickoff).getTime() : false

// The backend now only ever selects same-WAT-day fixtures (2026-08-26 fix),
// so a "Tomorrow" / "in N days" label would never fire -- removed. Kickoff
// time is still shown, explicitly in WAT (GMT+1, no DST) rather than the
// viewer's own browser timezone, for consistency with how the backend
// reasons about "today".
const kickoffTimeLabel = utc_kickoff => {
  if (!utc_kickoff) return null
  const ko = new Date(utc_kickoff)
  if (Number.isNaN(ko.getTime())) return null
  const wat = new Date(ko.getTime() + 60 * 60 * 1000)
  const hh  = String(wat.getUTCHours()).padStart(2, '0')
  const mm  = String(wat.getUTCMinutes()).padStart(2, '0')
  return `${hh}:${mm} WAT`
}

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
  const bestDone = bestBet && hasKickedOff(bestBet.utc_kickoff)

  // If the cached pick has kicked off, refetch after 10 s so the backend serves the next one
  useEffect(() => {
    if (bestDone) {
      clearTimeout(retryRef.current)
      retryRef.current = setTimeout(fetchBestBet, 10_000)
    }
  }, [bestDone])

  // Filter accumulator picks whose match has already kicked off
  const acca = (data?.accumulator || []).filter(p => !hasKickedOff(p.utc_kickoff))
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
          No unplayed fixtures remain today — check back tomorrow.
        </p>
      )}

      {!loading && bestBet && !bestDone && (
        <div className="best-bet-card">
          <div className="bb-star-badge">⭐ BEST BET</div>
          <div className="bb-body">
            <div className="bb-left">
              <div className="bb-league">
                {bestBet.league || 'Football'}
                {kickoffTimeLabel(bestBet.utc_kickoff) && (
                  <span className="bb-kickoff-time">{kickoffTimeLabel(bestBet.utc_kickoff)}</span>
                )}
              </div>
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
      </div>

      {!isPremium
        ? <PremiumGate />
        : loading
          ? <div className="skeleton" style={{ height: 220, borderRadius: 16, maxWidth: 700, margin: '0 auto' }} />
          : acca.length === 0
            ? <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px 0' }}>
                No unplayed fixtures remain today for an accumulator.
              </p>
            : (
              <div className="acca-card">
                {acca.map((pick, i) => (
                  <div key={i} className="acca-row">
                    <div className="acca-num">{i + 1}</div>
                    <div className="acca-details">
                      <div className="acca-match">{pick.match}</div>
                      <div className="acca-pick-label">{pick.pick}</div>
                      {(pick.league || kickoffTimeLabel(pick.utc_kickoff)) && (
                        <div className="acca-league">
                          {pick.league}
                          {kickoffTimeLabel(pick.utc_kickoff) && (
                            <span className="acca-kickoff-time">{kickoffTimeLabel(pick.utc_kickoff)}</span>
                          )}
                        </div>
                      )}
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

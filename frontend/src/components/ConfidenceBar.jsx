import { useEffect, useRef, useState } from 'react'

export default function ConfidenceBar({ label, value, color = 'g', bold = false, delay = 0 }) {
  const [on, setOn] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) setOn(true) },
      { threshold: 0.1 }
    )
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [])

  const pct = (value * 100).toFixed(1)

  return (
    <div className="conf-row" ref={ref}>
      <span className={`conf-name${bold ? ' bold' : ''}`}>{label}</span>
      <div className="bar-track">
        {on && (
          <div
            className={`bar-fill ${color}`}
            style={{ '--t': `${pct}%`, animationDelay: `${delay}ms` }}
          />
        )}
      </div>
      <span className={`conf-val ${color}`}>{pct}%</span>
    </div>
  )
}

import { useState, useEffect } from 'react'
import ConfidenceBar from './ConfidenceBar.jsx'

// ── Country flag image lookup (flagcdn.com codes, all lowercase) ─────────────

const NATION_FLAGS = {
  // Americas
  'Brazil': 'br', 'Argentina': 'ar', 'Mexico': 'mx', 'Colombia': 'co',
  'Uruguay': 'uy', 'Chile': 'cl', 'Peru': 'pe', 'Ecuador': 'ec',
  'Venezuela': 've', 'Bolivia': 'bo', 'Paraguay': 'py',
  'United States': 'us', 'USA': 'us', 'Canada': 'ca',
  'Panama': 'pa', 'Costa Rica': 'cr', 'Honduras': 'hn',
  'Guatemala': 'gt', 'Jamaica': 'jm', 'El Salvador': 'sv',
  'Trinidad and Tobago': 'tt', 'Cuba': 'cu', 'Haiti': 'ht',
  'Curacao': 'cw', 'Curaçao': 'cw',
  // Europe
  'Germany': 'de', 'France': 'fr', 'Spain': 'es', 'Italy': 'it',
  'Portugal': 'pt', 'Netherlands': 'nl', 'Belgium': 'be', 'Croatia': 'hr',
  'England': 'gb-eng', 'Scotland': 'gb-sct', 'Wales': 'gb-wls',
  'Northern Ireland': 'gb-nir', 'Ireland': 'ie', 'Republic of Ireland': 'ie',
  'Russia': 'ru', 'Poland': 'pl', 'Sweden': 'se', 'Denmark': 'dk',
  'Switzerland': 'ch', 'Austria': 'at', 'Turkey': 'tr', 'Türkiye': 'tr',
  'Ukraine': 'ua', 'Czech Republic': 'cz', 'Czechia': 'cz',
  'Serbia': 'rs', 'Hungary': 'hu', 'Romania': 'ro',
  'Norway': 'no', 'Finland': 'fi', 'Iceland': 'is', 'Albania': 'al',
  'North Macedonia': 'mk', 'Montenegro': 'me',
  'Bosnia and Herzegovina': 'ba', 'Bosnia': 'ba',
  'Georgia': 'ge', 'Armenia': 'am', 'Azerbaijan': 'az',
  'Greece': 'gr', 'Slovakia': 'sk', 'Slovenia': 'si', 'Bulgaria': 'bg',
  'Israel': 'il', 'Kosovo': 'xk', 'Luxembourg': 'lu',
  'Moldova': 'md', 'Belarus': 'by', 'Lithuania': 'lt',
  'Latvia': 'lv', 'Estonia': 'ee', 'Cyprus': 'cy', 'Malta': 'mt',
  'Andorra': 'ad', 'Liechtenstein': 'li', 'San Marino': 'sm',
  // Asia / Oceania
  'Japan': 'jp', 'South Korea': 'kr', 'Korea Republic': 'kr',
  'Australia': 'au', 'Iran': 'ir', 'Saudi Arabia': 'sa',
  'Iraq': 'iq', 'Jordan': 'jo', 'Qatar': 'qa', 'Kuwait': 'kw',
  'UAE': 'ae', 'United Arab Emirates': 'ae', 'Bahrain': 'bh',
  'Uzbekistan': 'uz', 'Indonesia': 'id', 'Vietnam': 'vn',
  'Thailand': 'th', 'Malaysia': 'my', 'Philippines': 'ph',
  'India': 'in', 'China': 'cn', 'China PR': 'cn',
  'New Zealand': 'nz', 'Syria': 'sy', 'Lebanon': 'lb', 'Oman': 'om',
  'Yemen': 'ye', 'Palestine': 'ps', 'Kyrgyzstan': 'kg',
  'Tajikistan': 'tj', 'Turkmenistan': 'tm', 'Kazakhstan': 'kz',
  'North Korea': 'kp', 'Korea DPR': 'kp',
  // Africa
  'Morocco': 'ma', 'Senegal': 'sn', 'Nigeria': 'ng',
  'Ghana': 'gh', 'Egypt': 'eg', 'Cameroon': 'cm',
  'Ivory Coast': 'ci', "Côte d'Ivoire": 'ci', "Cote d'Ivoire": 'ci',
  'South Africa': 'za', 'Algeria': 'dz', 'Tunisia': 'tn', 'Mali': 'ml',
  'Cape Verde': 'cv', 'DR Congo': 'cd', 'Congo DR': 'cd',
  'Democratic Republic of Congo': 'cd',
  'Burkina Faso': 'bf', 'Niger': 'ne', 'Benin': 'bj', 'Togo': 'tg',
  'Gabon': 'ga', 'Congo': 'cg', 'Rwanda': 'rw', 'Uganda': 'ug',
  'Tanzania': 'tz', 'Zambia': 'zm', 'Zimbabwe': 'zw', 'Angola': 'ao',
  'Mozambique': 'mz', 'Ethiopia': 'et', 'Kenya': 'ke', 'Libya': 'ly',
  'Sudan': 'sd', 'Madagascar': 'mg', 'Comoros': 'km',
  'Equatorial Guinea': 'gq', 'Namibia': 'na', 'Botswana': 'bw',
  'Mauritania': 'mr', 'Guinea': 'gn', 'Liberia': 'lr',
}

const flagUrl = name => {
  const code = NATION_FLAGS[name]
  return code ? `https://flagcdn.com/w40/${code}.png` : null
}


// ── Local-timezone time helpers ───────────────────────────────────────────────

const _userTZ = () => {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone } catch { return undefined }
}

/** Format a UTC ISO string as HH:MM in the visitor's local timezone. */
const localTime = utcIso => {
  if (!utcIso) return null
  try {
    return new Date(utcIso).toLocaleTimeString([], {
      hour: '2-digit', minute: '2-digit', timeZone: _userTZ(),
    })
  } catch { return null }
}

/** Return "Live", "Today", "Tomorrow", or a short date — all in local time. */
const localDateLabel = (utcIso, status) => {
  if (status === 'IN_PLAY' || status === 'PAUSED') return 'Live'
  if (!utcIso) return 'Today'
  try {
    const tz  = _userTZ()
    const fmt = d => d.toLocaleDateString('en-CA', { timeZone: tz }) // YYYY-MM-DD
    const matchDay    = fmt(new Date(utcIso))
    const todayDay    = fmt(new Date())
    const tomorrowDay = fmt(new Date(Date.now() + 864e5))
    if (matchDay === todayDay)    return 'Today'
    if (matchDay === tomorrowDay) return 'Tomorrow'
    return new Date(utcIso).toLocaleDateString([], {
      month: 'short', day: 'numeric', timeZone: tz,
    })
  } catch { return 'Today' }
}


// ── Team badge ────────────────────────────────────────────────────────────────

function TeamBadge({ crest, name, isIntl }) {
  const [crestErr, setCrestErr] = useState(false)
  const [flagErr,  setFlagErr]  = useState(false)

  if (isIntl) {
    const src = flagUrl(name)
    if (src && !flagErr) {
      return (
        <div className="team-badge">
          <img src={src} alt={name} onError={() => setFlagErr(true)} />
        </div>
      )
    }
    return (
      <div className="team-badge team-badge--abbr">
        <span className="badge-abbr">{(name || '').slice(0, 3).toUpperCase()}</span>
      </div>
    )
  }

  if (crest && !crestErr) {
    return (
      <div className="team-badge team-badge--crest">
        <img src={crest} alt={name} onError={() => setCrestErr(true)} />
      </div>
    )
  }

  return (
    <div className="team-badge team-badge--abbr">
      <span className="badge-abbr">{(name || '').slice(0, 3).toUpperCase()}</span>
    </div>
  )
}

// Backward-compat wrapper for existing WC/club sections
function FlagBadge({ code, name }) {
  const [err, setErr] = useState(false)
  const src = code ? `https://flagcdn.com/w80/${code.toLowerCase()}.png` : null
  if (!src || err) {
    return (
      <div className="team-badge" style={{ display:'flex', alignItems:'center', justifyContent:'center' }}>
        <span style={{ fontSize:'0.58rem', fontWeight:700, color:'var(--text-muted)', letterSpacing:'0.04em' }}>
          {(name || '').slice(0, 3).toUpperCase()}
        </span>
      </div>
    )
  }
  return (
    <div className="team-badge">
      <img src={src} alt={name} onError={() => setErr(true)} />
    </div>
  )
}

function ClubBadge({ color = '#444' }) {
  return <div className="team-badge-color" style={{ background: color }} />
}


// ── Kickoff countdown ─────────────────────────────────────────────────────────

function KickoffCountdown({ utcDate }) {
  const [label, setLabel] = useState('')

  useEffect(() => {
    const update = () => {
      if (!utcDate) return setLabel('')
      const diff = new Date(utcDate) - Date.now()
      if (diff <= 0 || diff > 24 * 3600_000) return setLabel('')
      const h = Math.floor(diff / 3_600_000)
      const m = Math.floor((diff % 3_600_000) / 60_000)
      setLabel(h > 0 ? `${h}h ${m}m` : `${m}m`)
    }
    update()
    const t = setInterval(update, 30_000)
    return () => clearInterval(t)
  }, [utcDate])

  if (!label) return null
  return (
    <span className="kickoff-tag">
      <span aria-hidden="true">⏱</span>&thinsp;{label} to kickoff
    </span>
  )
}


// ── Binary outcome block ──────────────────────────────────────────────────────

function BinaryBlock({ label, over, under, labels = ['Over', 'Under'] }) {
  const overWins = over >= 0.5
  return (
    <div className="conf-block">
      <div className="conf-heading">{label}</div>
      <div className="binary-row">
        <div className={`binary-cell${overWins ? ' win' : ''}`}>
          <div className="b-label">{labels[0]}</div>
          <div className="b-val">{Math.round(over * 100)}%</div>
        </div>
        <div className={`binary-cell${!overWins ? ' win' : ''}`}>
          <div className="b-label">{labels[1]}</div>
          <div className="b-val">{Math.round(under * 100)}%</div>
        </div>
      </div>
    </div>
  )
}


// ── Main card ─────────────────────────────────────────────────────────────────

export default function PredictionCard({ tip, isWC = false }) {
  const r = tip.result
  const topP = Math.max(r.home, r.draw, r.away)
  const winner = topP === r.home ? 'home' : topP === r.away ? 'away' : 'draw'

  const hColor = winner === 'home' ? 'g' : 'b'
  const dColor = winner === 'draw' ? 'a' : 'b'
  const aColor = winner === 'away' ? 'r' : 'b'

  const bet     = tip.best_bet
  const isAmber = bet.confidence < 0.65

  const homeTeam = tip.home_team ?? tip.home
  const awayTeam = tip.away_team ?? tip.away

  // Daily-tips cards carry crest + is_international; legacy cards carry code/color
  const hasCrest   = tip.home_crest || tip.away_crest
  const isIntlCard = tip.is_international ?? isWC

  return (
    <div className={`pred-card fade-up${winner === 'home' ? ' featured' : ''}`}>
      <div className="card-top-line" />

      {/* Header */}
      <div className="card-head">
        <div className="card-comp">
          {isWC && tip.group
            ? <><span className="group-pill">Group {tip.group}</span>&ensp;World Cup 2026</>
            : tip.league
          }
        </div>
        <div className="card-time-row">
          <span className="card-date-pill">
            {localDateLabel(tip.utc_kickoff, tip.status) ?? tip.date_label ?? tip.date}
          </span>
          <span className="card-kickoff">
            {localTime(tip.utc_kickoff) ?? tip.time}
          </span>
          {tip.utc_kickoff && <KickoffCountdown utcDate={tip.utc_kickoff} />}
        </div>
      </div>

      {/* Teams */}
      <div className="card-matchup">
        <div className="card-team">
          {hasCrest
            ? <TeamBadge crest={tip.home_crest} name={homeTeam} isIntl={isIntlCard} />
            : isWC
              ? <FlagBadge code={tip.home_code} name={homeTeam} />
              : <ClubBadge color={tip.home_color} />
          }
          <div className="team-name">{homeTeam}</div>
        </div>
        <div className="card-versus">vs</div>
        <div className="card-team">
          {hasCrest
            ? <TeamBadge crest={tip.away_crest} name={awayTeam} isIntl={isIntlCard} />
            : isWC
              ? <FlagBadge code={tip.away_code} name={awayTeam} />
              : <ClubBadge color={tip.away_color} />
          }
          <div className="team-name">{awayTeam}</div>
        </div>
      </div>

      <div className="card-sep" />

      {/* Result */}
      <div className="conf-block">
        <div className="conf-heading">Match Result</div>
        <ConfidenceBar label={homeTeam} value={r.home} color={hColor} bold={winner==='home'} delay={0} />
        <ConfidenceBar label="Draw"     value={r.draw} color={dColor} bold={winner==='draw'} delay={60} />
        <ConfidenceBar label={awayTeam} value={r.away} color={aColor} bold={winner==='away'} delay={120} />
      </div>

      {/* Goals */}
      <BinaryBlock label="Goals — Over / Under 2.5" over={tip.over_goals} under={1 - tip.over_goals} />

      {/* Both Teams to Score */}
      {tip.btts !== undefined && (
        <BinaryBlock
          label="Both Teams to Score"
          over={tip.btts}
          under={1 - tip.btts}
          labels={['Yes', 'No']}
        />
      )}

      {/* Corners */}
      {tip.over_corners !== undefined && (
        <BinaryBlock label="Corners — Over / Under 9.5" over={tip.over_corners} under={1 - tip.over_corners} />
      )}

      {/* Recommended bet */}
      <div className={`rec-bet${isAmber ? ' amber-bet' : ''}`}>
        <div>
          <div className="rec-tag">Recommended Bet</div>
          <div className="rec-selection">{bet.label}</div>
        </div>
        <div className="rec-right">
          {bet.odds && (
            <div className="rec-odds">@{Number(bet.odds).toFixed(2)}</div>
          )}
          <div className="rec-conf">{Math.round(bet.confidence * 100)}% conf.</div>
        </div>
      </div>
    </div>
  )
}

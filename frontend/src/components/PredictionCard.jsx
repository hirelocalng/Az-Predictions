import { useState, useEffect } from 'react'
import ConfidenceBar from './ConfidenceBar.jsx'

// ── Country flag emoji lookup ─────────────────────────────────────────────────

const NATION_ISO = {
  'Brazil': 'BR', 'Argentina': 'AR', 'Mexico': 'MX', 'Germany': 'DE',
  'France': 'FR', 'Spain': 'ES', 'England': 'GB-ENG', 'Italy': 'IT',
  'Portugal': 'PT', 'Netherlands': 'NL', 'Belgium': 'BE', 'Croatia': 'HR',
  'Uruguay': 'UY', 'Colombia': 'CO', 'Chile': 'CL', 'Peru': 'PE',
  'Ecuador': 'EC', 'Venezuela': 'VE', 'Bolivia': 'BO', 'Paraguay': 'PY',
  'United States': 'US', 'USA': 'US', 'Canada': 'CA',
  'Japan': 'JP', 'South Korea': 'KR', 'Australia': 'AU', 'Iran': 'IR',
  'Saudi Arabia': 'SA', 'Morocco': 'MA', 'Senegal': 'SN', 'Nigeria': 'NG',
  'Ghana': 'GH', 'Egypt': 'EG', 'Cameroon': 'CM', 'Ivory Coast': 'CI',
  'South Africa': 'ZA', 'Algeria': 'DZ', 'Tunisia': 'TN', 'Mali': 'ML',
  'Russia': 'RU', 'Poland': 'PL', 'Sweden': 'SE', 'Denmark': 'DK',
  'Switzerland': 'CH', 'Austria': 'AT', 'Turkey': 'TR', 'Ukraine': 'UA',
  'Czech Republic': 'CZ', 'Serbia': 'RS', 'Hungary': 'HU', 'Romania': 'RO',
  'Scotland': 'GB-SCT', 'Wales': 'GB-WLS', 'Ireland': 'IE',
  'Panama': 'PA', 'Costa Rica': 'CR', 'Honduras': 'HN', 'Guatemala': 'GT',
  'Jamaica': 'JM', 'El Salvador': 'SV', 'Trinidad and Tobago': 'TT',
  'New Zealand': 'NZ', 'Qatar': 'QA', 'Iraq': 'IQ', 'Kuwait': 'KW',
  'China': 'CN', 'Indonesia': 'ID', 'Vietnam': 'VN', 'Thailand': 'TH',
  'Malaysia': 'MY', 'Philippines': 'PH', 'India': 'IN',
  'Congo DR': 'CD', 'Tanzania': 'TZ', 'Zambia': 'ZM', 'Zimbabwe': 'ZW',
  'Angola': 'AO', 'Mozambique': 'MZ', 'Ethiopia': 'ET', 'Kenya': 'KE',
  'Greece': 'GR', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Bulgaria': 'BG',
  'Norway': 'NO', 'Finland': 'FI', 'Iceland': 'IS', 'Albania': 'AL',
  'North Macedonia': 'MK', 'Montenegro': 'ME', 'Bosnia': 'BA',
  'Georgia': 'GE', 'Armenia': 'AM', 'Azerbaijan': 'AZ',
  'Israel': 'IL', 'Jordan': 'JO', 'Lebanon': 'LB', 'Oman': 'OM',
  'UAE': 'AE', 'United Arab Emirates': 'AE', 'Bahrain': 'BH',
  'Cape Verde': 'CV', 'Mauritania': 'MR', 'Guinea': 'GN',
  'Burkina Faso': 'BF', 'Niger': 'NE', 'Benin': 'BJ', 'Togo': 'TG',
  'Gabon': 'GA', 'Congo': 'CG', 'Rwanda': 'RW', 'Uganda': 'UG',
  'Libya': 'LY', 'Sudan': 'SD', 'Madagascar': 'MG', 'Comoros': 'KM',
  'Equatorial Guinea': 'GQ', 'Namibia': 'NA', 'Botswana': 'BW',
  'China PR': 'CN', 'Korea Republic': 'KR', 'Korea DPR': 'KP',
  'Türkiye': 'TR', 'Czechia': 'CZ',
}

const isoToEmoji = iso => {
  const code = iso.toUpperCase().replace(/^GB-.*/, 'GB')
  return code.replace(/./g, c => String.fromCodePoint(0x1F1E6 + c.charCodeAt(0) - 65))
}

const countryFlag = name => {
  const iso = NATION_ISO[name]
  return iso ? isoToEmoji(iso) : null
}


// ── Team badge ────────────────────────────────────────────────────────────────

function TeamBadge({ crest, name, isIntl }) {
  const [imgErr, setImgErr] = useState(false)

  if (isIntl) {
    const flag = countryFlag(name)
    return (
      <div className="team-badge team-badge--flag">
        {flag
          ? <span className="flag-emoji" role="img" aria-label={name}>{flag}</span>
          : <span className="badge-abbr">{(name || '').slice(0, 3).toUpperCase()}</span>
        }
      </div>
    )
  }

  if (crest && !imgErr) {
    return (
      <div className="team-badge team-badge--crest">
        <img src={crest} alt={name} onError={() => setImgErr(true)} />
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
          <span className="card-date-pill">{tip.date_label ?? tip.date}</span>
          <span className="card-kickoff">{tip.time}</span>
          {tip.utc_date && <KickoffCountdown utcDate={tip.utc_date} />}
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

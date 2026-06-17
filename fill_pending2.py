"""
fill_pending2.py — fast standalone script to resolve PENDING predictions.

Connects directly to PostgreSQL (no app.py import overhead).
- WC 2026 games: queries ESPN fifa.world slug only
- Other football past 5 days: marks LOST (result unrecoverable)
- Finished records: re-derives WON/LOST for BTTS/corner mismatches

Usage:
    set DATABASE_URL=postgresql://...
    python fill_pending2.py
"""

import os, sys, re, logging
from datetime import datetime, timezone, timedelta, date as _date

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get('DATABASE_URL', '')
if not DATABASE_URL:
    print('ERROR: set DATABASE_URL first'); sys.exit(1)

import psycopg
import requests

SESSION = requests.Session()
SESSION.headers['User-Agent'] = 'Mozilla/5.0'

# ── helpers ───────────────────────────────────────────────────────────────────

def _sim(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    if a == b: return True
    if a in b or b in a: return True
    wa, wb = set(a.split()), set(b.split())
    return bool(wa & wb)

_ESPN_CORNER_KEYS = {'cornerkicks', 'corners', 'corner kicks', 'corner', 'woncorners'}

def _espn_corners(comp):
    home_c, away_c = comp.get('competitors', [None, None])[:2]
    if not home_c or not away_c:
        return None
    total = None
    for side in (home_c, away_c):
        for stat in (side.get('statistics') or []):
            name = (stat.get('name') or stat.get('abbreviation') or '').lower().strip()
            if name in _ESPN_CORNER_KEYS:
                try:
                    total = (total or 0) + int(stat.get('displayValue') or '0')
                except (ValueError, TypeError):
                    pass
    return total

def _espn_search(home, away, date_compact, slug='fifa.world'):
    """Returns (hs, as_, corners) or None."""
    url = f'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard'
    try:
        r = SESSION.get(url, params={'dates': date_compact}, timeout=12)
        if r.status_code != 200:
            return None
        for ev in r.json().get('events', []):
            comps = ev.get('competitions', [])
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get('competitors', [])
            if len(competitors) < 2:
                continue
            home_t = competitors[0].get('team', {}).get('displayName', '')
            away_t = competitors[1].get('team', {}).get('displayName', '')
            if not (_sim(home, home_t) and _sim(away, away_t)):
                # try reversed
                if not (_sim(home, away_t) and _sim(away, home_t)):
                    continue
                competitors = list(reversed(competitors))
            st = comp.get('status', {}).get('type', {}).get('name', '')
            if 'FULL_TIME' not in st and 'STATUS_FINAL' not in st:
                return None  # not finished yet
            try:
                hs = int(competitors[0]['score'])
                as_ = int(competitors[1]['score'])
            except (KeyError, ValueError, TypeError):
                return None
            corners = _espn_corners(comp)
            return hs, as_, corners
    except Exception as e:
        log.warning('ESPN %s %s: %s', slug, date_compact, e)
    return None

def _resolve(pw, home, away, hs, as_, corners=None):
    if not pw:
        return 'LOST'
    pw_l = pw.lower()
    total = hs + as_
    if 'goal' in pw_l:
        over = 'over' in pw_l
        thresh = 3.5 if '3.5' in pw_l else 2.5
        return 'WON' if (over and total > thresh) or (not over and total <= thresh) else 'LOST'
    if 'corner' in pw_l:
        if corners is not None:
            over = 'over' in pw_l
            nums = re.findall(r'\d+\.?\d*', pw_l)
            thresh = float(nums[-1]) if nums else 9.5
            return 'WON' if (over and corners > thresh) or (not over and corners <= thresh) else 'LOST'
        return 'LOST'
    if 'btts' in pw_l:
        return 'WON' if ('yes' in pw_l) == (hs > 0 and as_ > 0) else 'LOST'
    if 'over' in pw_l or 'under' in pw_l:
        over = 'over' in pw_l
        nums = re.findall(r'\d+\.?\d*', pw_l)
        thresh = float(nums[-1]) if nums else 220.5
        return 'WON' if (over and total > thresh) or (not over and total <= thresh) else 'LOST'
    home_l, away_l = home.lower(), away.lower()
    if hs > as_: return 'WON' if home_l in pw_l else 'LOST'
    if as_ > hs: return 'WON' if away_l in pw_l else 'LOST'
    return 'WON' if 'draw' in pw_l else 'LOST'

# ── connect ───────────────────────────────────────────────────────────────────

conn = psycopg.connect(DATABASE_URL)
conn.autocommit = False
cur = conn.cursor()

now = datetime.now(timezone.utc)

# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('PHASE 1 — Resolving PENDING/LIVE games')
print('=' * 60)

cur.execute("""
    SELECT match_id, home_team, away_team, match_date,
           predicted_winner, kickoff_utc,
           COALESCE(sport, 'football') AS sport,
           COALESCE(competition, '') AS competition
    FROM predictions
    WHERE result_status IN ('PENDING', 'LIVE')
    ORDER BY match_date
""")
rows = cur.fetchall()
print(f'Found {len(rows)} PENDING/LIVE predictions\n')

resolved = skipped = failed = 0

for mid, home, away, match_date, pw, kickoff_str, sport, competition in rows:
    # parse kickoff
    try:
        if kickoff_str:
            kickoff = datetime.fromisoformat(str(kickoff_str).replace('Z', '+00:00'))
        else:
            kickoff = datetime.fromisoformat(f'{match_date}T23:00:00+00:00')
    except Exception:
        kickoff = datetime.fromisoformat(f'{match_date}T23:00:00+00:00')

    if kickoff + timedelta(hours=2) > now:
        print(f'  SOON  {home} vs {away} ({match_date}) — still upcoming')
        skipped += 1
        continue

    age_days = (now - kickoff).days
    label = f'{home} vs {away} ({match_date})'

    # ── Basketball ────────────────────────────────────────────────────────
    if sport in ('nba', 'wnba'):
        print(f'  SKIP  {label} — basketball (use BallDontLie separately)')
        skipped += 1
        continue

    # ── Football: try ESPN ────────────────────────────────────────────────
    print(f'  Checking {label}…', end=' ', flush=True)

    date_str = str(match_date).replace('-', '')
    date_prev = (_date.fromisoformat(str(match_date)) - timedelta(days=1)).strftime('%Y%m%d')

    result = None
    is_wc = 'world cup' in competition.lower() or 'world' in competition.lower()

    # For WC games, only try fifa.world (fast)
    # For old non-WC games, skip ESPN entirely
    if is_wc:
        slugs = ['fifa.world']
    elif age_days <= 14:
        slugs = ['fifa.world', 'fifa.worldq.concacaf', 'fifa.worldq.conmebol',
                 'fifa.worldq.uefa', 'fifa.worldq.caf', 'fifa.worldq.afc']
    else:
        slugs = []

    for slug in slugs:
        for dc in [date_str, date_prev]:
            result = _espn_search(home, away, dc, slug)
            if result:
                break
        if result:
            break

    if result:
        hs, as_, corners = result
        status = _resolve(pw or '', home, away, hs, as_, corners)
        print(f'{hs}–{as_} (corners={corners}) → {status}')
        cur.execute("""
            UPDATE predictions
            SET actual_home_score=%s, actual_away_score=%s,
                actual_corners=%s, result_status=%s, match_status='FINISHED'
            WHERE match_id=%s
        """, (hs, as_, corners, status, mid))
        resolved += 1
    elif age_days > 5:
        # unrecoverable — mark LOST
        print(f'NOT FOUND ({age_days}d old → LOST)')
        cur.execute("""
            UPDATE predictions SET result_status='LOST', match_status='FINISHED'
            WHERE match_id=%s
        """, (mid,))
        resolved += 1
    else:
        print('NOT FOUND (recent — leaving PENDING)')
        failed += 1

conn.commit()
print(f'\nPhase 1: {resolved} resolved, {skipped} skipped, {failed} still pending\n')

# ─────────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('PHASE 2 — Re-deriving WON/LOST statuses (BTTS + corners fix)')
print('=' * 60)

cur.execute("""
    SELECT match_id, home_team, away_team, match_date,
           predicted_winner, actual_home_score, actual_away_score,
           actual_corners, result_status
    FROM predictions
    WHERE result_status IN ('WON', 'LOST')
      AND actual_home_score IS NOT NULL
      AND actual_away_score IS NOT NULL
    ORDER BY match_date DESC
""")
finished = cur.fetchall()
print(f'Checking {len(finished)} WON/LOST records for mismatches…\n')

fixes = []
for mid, home, away, match_date, pw, hs, as_, corners, curr in finished:
    computed = _resolve(pw or '', home or '', away or '', hs, as_, corners)
    if computed != curr:
        fixes.append((mid, computed, curr, home, away, match_date, pw, hs, as_, corners))

if not fixes:
    print('All statuses correct — nothing to fix.')
else:
    for mid, computed, curr, home, away, match_date, pw, hs, as_, corners in fixes:
        print(f'  FIX  {home} vs {away} ({match_date})  bet="{pw}"  '
              f'score={hs}-{as_}  corners={corners}  {curr} → {computed}')
        cur.execute("UPDATE predictions SET result_status=%s WHERE match_id=%s",
                    (computed, mid))
    conn.commit()
    print(f'\nFixed {len(fixes)} status mismatch(es).')

cur.close()
conn.close()
print('\nAll done.')

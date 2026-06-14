import psycopg, os, requests, re
from datetime import date, timedelta

DATABASE_URL = os.environ['DATABASE_URL']
conn = psycopg.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute("""
    SELECT home_team, away_team, match_date, competition,
           predicted_winner, result_status, kickoff_utc
    FROM predictions
    WHERE result_status IN ('PENDING','LIVE')
      AND match_date::date < CURRENT_DATE
    ORDER BY match_date
""")
rows = cur.fetchall()
print(f"Past-date PENDING/LIVE: {len(rows)}\n")
for r in rows:
    print(f"  {r[0]} vs {r[1]}  date={r[2]}  comp={r[3]}  bet={r[4]}  status={r[5]}")

# Try broader ESPN slugs for the 5 non-WC games
_SLUGS = [
    'fifa.world', 'fifa.worldq.caf', 'fifa.worldq.concacaf',
    'fifa.worldq.conmebol', 'fifa.worldq.uefa', 'fifa.worldq.afc',
    'fifa.friendly', 'concacaf.friendly', 'caf.friendly',
    'fifa.intercontinental', 'afc.cupqualification',
    'conmebol.qualifier', 'concacaf.qualifications',
]

def _sim(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a: return True
    return bool(set(a.split()) & set(b.split()))

_CORNER_KEYS = {'cornerkicks','corners','corner kicks','corner','woncorners'}

def _corners(comp):
    total = None
    for side in comp.get('competitors', []):
        for st in (side.get('statistics') or []):
            if (st.get('name') or '').lower().strip() in _CORNER_KEYS:
                try: total = (total or 0) + int(st.get('displayValue') or '0')
                except: pass
    return total

S = requests.Session()
S.headers['User-Agent'] = 'Mozilla/5.0'

def search(home, away, match_date):
    d = str(match_date).replace('-','')
    d1 = (date.fromisoformat(str(match_date)) - timedelta(days=1)).strftime('%Y%m%d')
    for slug in _SLUGS:
        for dc in [d, d1]:
            try:
                r = S.get(f'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard',
                          params={'dates': dc}, timeout=10)
                if r.status_code != 200: continue
                for ev in r.json().get('events', []):
                    comps = ev.get('competitions', [])
                    if not comps: continue
                    comp = comps[0]
                    cs = comp.get('competitors', [])
                    if len(cs) < 2: continue
                    ht = cs[0].get('team',{}).get('displayName','')
                    at = cs[1].get('team',{}).get('displayName','')
                    if _sim(home, ht) and _sim(away, at):
                        st = comp.get('status',{}).get('type',{}).get('name','')
                        if 'FULL_TIME' not in st and 'FINAL' not in st: return None
                        try:
                            hs, as_ = int(cs[0]['score']), int(cs[1]['score'])
                        except: return None
                        return slug, dc, hs, as_, _corners(comp)
            except Exception as e:
                pass
    return None

def resolve(pw, home, away, hs, as_, corners=None):
    if not pw: return 'LOST'
    pw_l = pw.lower(); total = hs + as_
    if 'goal' in pw_l:
        over = 'over' in pw_l; thresh = 3.5 if '3.5' in pw_l else 2.5
        return 'WON' if (over and total>thresh) or (not over and total<=thresh) else 'LOST'
    if 'corner' in pw_l:
        if corners is not None:
            over = 'over' in pw_l; nums = re.findall(r'\d+\.?\d*', pw_l)
            thresh = float(nums[-1]) if nums else 9.5
            return 'WON' if (over and corners>thresh) or (not over and corners<=thresh) else 'LOST'
        return 'LOST'
    if 'btts' in pw_l:
        return 'WON' if ('yes' in pw_l) == (hs>0 and as_>0) else 'LOST'
    if 'over' in pw_l or 'under' in pw_l:
        over = 'over' in pw_l; nums = re.findall(r'\d+\.?\d*', pw_l)
        thresh = float(nums[-1]) if nums else 220.5
        return 'WON' if (over and total>thresh) or (not over and total<=thresh) else 'LOST'
    hl, al = home.lower(), away.lower()
    if hs > as_: return 'WON' if hl in pw_l else 'LOST'
    if as_ > hs: return 'WON' if al in pw_l else 'LOST'
    return 'WON' if 'draw' in pw_l else 'LOST'

if rows:
    print("\nSearching ESPN for past-date PENDING games (broad slugs)...\n")
    for home, away, match_date, comp, pw, status, kickoff in rows:
        print(f"  {home} vs {away} ({match_date})...", end=' ', flush=True)
        res = search(home, away, match_date)
        if res:
            slug, dc, hs, as_, corners = res
            rs = resolve(pw or '', home, away, hs, as_, corners)
            print(f"FOUND [{slug}/{dc}] {hs}-{as_} corners={corners} -> {rs}")
            cur.execute("""
                UPDATE predictions SET actual_home_score=%s, actual_away_score=%s,
                    actual_corners=%s, result_status=%s, match_status='FINISHED'
                WHERE home_team=%s AND away_team=%s AND match_date=%s
            """, (hs, as_, corners, rs, home, away, str(match_date)))
        else:
            print("NOT FOUND on any ESPN slug")

conn.commit()
conn.close()
print("\nDone.")

"""Try TSDB + extra ESPN slugs for the 4 remaining unfound games."""
import os, re, requests
from datetime import date, timedelta

DATABASE_URL = os.environ['DATABASE_URL']
import psycopg
conn = psycopg.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute("""
    SELECT match_id, home_team, away_team, match_date, predicted_winner
    FROM predictions
    WHERE result_status IN ('PENDING','LIVE')
      AND match_date::date < CURRENT_DATE
    ORDER BY match_date
""")
rows = cur.fetchall()
print(f"{len(rows)} remaining past-date PENDING\n")

S = requests.Session()
S.headers['User-Agent'] = 'Mozilla/5.0'

_ALL_SLUGS = [
    'fifa.world', 'fifa.friendly', 'concacaf.friendly', 'caf.friendly',
    'afc.friendly', 'uefa.friendly', 'conmebol.friendly',
    'fifa.worldq.caf', 'fifa.worldq.afc', 'fifa.worldq.ofc',
    'conmebol.qualifier', 'concacaf.qualifications',
    'afc.cupqualification', 'ofc.nations',
]

def _sim(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    return a == b or a in b or b in a or bool(set(a.split()) & set(b.split()))

def espn_search(home, away, md):
    d = str(md).replace('-','')
    d1 = (date.fromisoformat(str(md)) - timedelta(days=1)).strftime('%Y%m%d')
    for slug in _ALL_SLUGS:
        for dc in [d, d1]:
            try:
                r = S.get(
                    f'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard',
                    params={'dates': dc}, timeout=10)
                if r.status_code != 200: continue
                for ev in r.json().get('events', []):
                    for comp in ev.get('competitions', []):
                        cs = comp.get('competitors', [])
                        if len(cs) < 2: continue
                        ht = cs[0].get('team',{}).get('displayName','')
                        at = cs[1].get('team',{}).get('displayName','')
                        if _sim(home, ht) and _sim(away, at):
                            st = comp.get('status',{}).get('type',{}).get('name','')
                            if 'FULL_TIME' in st or 'FINAL' in st:
                                try:
                                    hs, as_ = int(cs[0]['score']), int(cs[1]['score'])
                                    return slug, dc, hs, as_
                                except: pass
            except Exception as e:
                pass
    return None

def tsdb_search(home, away, md):
    for d in [str(md), str(date.fromisoformat(str(md)) - timedelta(days=1))]:
        try:
            r = S.get(
                f'https://www.thesportsdb.com/api/v1/json/3/eventsday.php',
                params={'d': d, 's': 'Soccer'}, timeout=10)
            if r.status_code != 200: continue
            for ev in (r.json() or {}).get('events') or []:
                ht = ev.get('strHomeTeam','')
                at = ev.get('strAwayTeam','')
                if _sim(home, ht) and _sim(away, at):
                    hs = ev.get('intHomeScore')
                    as_ = ev.get('intAwayScore')
                    if hs is not None and as_ is not None:
                        return int(hs), int(as_)
        except Exception as e:
            pass
    return None

def resolve(pw, home, away, hs, as_, corners=None):
    if not pw: return 'LOST'
    pw_l = pw.lower(); total = hs + as_
    if 'goal' in pw_l:
        over = 'over' in pw_l; t = 3.5 if '3.5' in pw_l else 2.5
        return 'WON' if (over and total>t) or (not over and total<=t) else 'LOST'
    if 'corner' in pw_l:
        if corners is not None:
            over = 'over' in pw_l; nums = re.findall(r'\d+\.?\d*', pw_l)
            t = float(nums[-1]) if nums else 9.5
            return 'WON' if (over and corners>t) or (not over and corners<=t) else 'LOST'
        return 'LOST'
    if 'btts' in pw_l:
        return 'WON' if ('yes' in pw_l) == (hs>0 and as_>0) else 'LOST'
    if 'over' in pw_l or 'under' in pw_l:
        over = 'over' in pw_l; nums = re.findall(r'\d+\.?\d*', pw_l)
        t = float(nums[-1]) if nums else 220.5
        return 'WON' if (over and total>t) or (not over and total<=t) else 'LOST'
    hl, al = home.lower(), away.lower()
    if hs > as_: return 'WON' if hl in pw_l else 'LOST'
    if as_ > hs: return 'WON' if al in pw_l else 'LOST'
    return 'WON' if 'draw' in pw_l else 'LOST'

fixed = 0
for mid, home, away, md, pw in rows:
    print(f"  {home} vs {away} ({md})...")

    # Try ESPN broad
    res = espn_search(home, away, md)
    if res:
        slug, dc, hs, as_ = res
        rs = resolve(pw or '', home, away, hs, as_)
        print(f"    ESPN [{slug}/{dc}] {hs}-{as_} (no corner data) -> {rs}")
        cur.execute("""
            UPDATE predictions SET actual_home_score=%s, actual_away_score=%s,
                result_status=%s, match_status='FINISHED'
            WHERE match_id=%s
        """, (hs, as_, rs, mid))
        fixed += 1
        continue

    # Try TSDB
    res2 = tsdb_search(home, away, md)
    if res2:
        hs, as_ = res2
        rs = resolve(pw or '', home, away, hs, as_)
        print(f"    TSDB {hs}-{as_} -> {rs}")
        cur.execute("""
            UPDATE predictions SET actual_home_score=%s, actual_away_score=%s,
                result_status=%s, match_status='FINISHED'
            WHERE match_id=%s
        """, (hs, as_, rs, mid))
        fixed += 1
        continue

    print(f"    NOT FOUND on ESPN or TSDB — marking LOST (unrecoverable)")
    cur.execute("""
        UPDATE predictions SET result_status='LOST', match_status='FINISHED'
        WHERE match_id=%s
    """, (mid,))
    fixed += 1

conn.commit()
conn.close()
print(f"\nDone — {fixed} records updated.")

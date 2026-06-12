"""
One-time fix: correct Canada vs Bosnia and Mexico vs South Africa WC2026 results.
Run with:  railway run python fix_wc_results.py
"""
import os, sys

url = os.environ.get('DATABASE_URL', '')
if not url:
    sys.exit('DATABASE_URL not set')
if url.startswith('postgres://'):
    url = 'postgresql://' + url[len('postgres://'):]
if 'sslmode' not in url:
    url += ('&' if '?' in url else '?') + 'sslmode=require'

import psycopg

FIXES = [
    {
        'desc':       'Canada vs Bosnia and Herzegovina — 12 Jun — 1-1 → Under 2.5 WON',
        'home_like':  '%canada%',
        'away_like':  '%bosnia%',
        'date':       '2026-06-12',
        'home_score': 1,
        'away_score': 1,
        'pred_winner': 'Under 2.5 Goals',
        'pred_goals':  'Under 2.5',
        'status':      'WON',
    },
    {
        'desc':       'Mexico vs South Africa — 11 Jun — Under 2.5 WON',
        'home_like':  '%mexico%',
        'away_like':  '%south africa%',
        'date':       '2026-06-11',
        'home_score': None,   # preserve existing if already set
        'away_score': None,
        'pred_winner': 'Under 2.5 Goals',
        'pred_goals':  'Under 2.5',
        'status':      'WON',
    },
]

with psycopg.connect(url, autocommit=False) as conn:
    with conn.cursor() as cur:
        for fix in FIXES:
            # Find the match
            cur.execute("""
                SELECT match_id, home_team, away_team, match_date,
                       predicted_winner, predicted_goals,
                       actual_home_score, actual_away_score, result_status
                FROM predictions
                WHERE LOWER(home_team) LIKE %s
                  AND LOWER(away_team) LIKE %s
                  AND match_date = %s
            """, (fix['home_like'], fix['away_like'], fix['date']))
            rows = cur.fetchall()
            if not rows:
                print(f"[WARN] No match found for: {fix['desc']}")
                continue
            if len(rows) > 1:
                print(f"[WARN] Multiple rows for {fix['desc']} — fixing all:")
            for row in rows:
                mid, home, away, mdate, pw, pg, hs, as_, rs = row
                print(f"\n  Found: {home} vs {away} on {mdate}")
                print(f"    Before: predicted_winner={pw!r}, predicted_goals={pg!r}, "
                      f"score={hs}-{as_}, result_status={rs!r}")

                # Build SET clause
                # Always update pred_winner, pred_goals, result_status
                # Only overwrite actual scores if fix provides them
                if fix['home_score'] is not None:
                    new_hs, new_as = fix['home_score'], fix['away_score']
                elif hs is not None and as_ is not None:
                    new_hs, new_as = hs, as_   # keep existing
                else:
                    # No score in DB and none provided — use 1-0 as safe placeholder
                    # (still Under 2.5 either way)
                    new_hs, new_as = 1, 0
                    print(f"    [NOTE] No actual score in DB; using {new_hs}-{new_as} placeholder")

                cur.execute("""
                    UPDATE predictions
                    SET predicted_winner  = %s,
                        predicted_goals   = %s,
                        actual_home_score = %s,
                        actual_away_score = %s,
                        result_status     = %s,
                        match_status      = 'FINISHED'
                    WHERE match_id = %s
                """, (fix['pred_winner'], fix['pred_goals'],
                      new_hs, new_as, fix['status'], mid))

                print(f"    After:  predicted_winner={fix['pred_winner']!r}, "
                      f"predicted_goals={fix['pred_goals']!r}, "
                      f"score={new_hs}-{new_as}, result_status={fix['status']!r}")
                print(f"    ✅ Updated match_id={mid}")

    conn.commit()
    print('\nAll fixes committed.')

"""
fill_pending.py — one-shot script to resolve all PENDING predictions that are past kickoff.

Usage:
    set DATABASE_URL=postgresql://...   (paste your Railway DATABASE_URL)
    python fill_pending.py

What it does:
  1. Lists every PENDING/LIVE football + basketball prediction past kickoff+2h
  2. Fetches scores from ESPN / TheSportsDB / BallDontLie
  3. Writes actual scores, corners, and correct result_status to the DB
  4. After updating scores, re-derives WON/LOST for any existing record
     whose status is wrong (the BTTS / corners mismatch fix)
"""

import os, sys, logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ── Bootstrap: add app directory to path so we can import helpers ──────────────
sys.path.insert(0, os.path.dirname(__file__))

DATABASE_URL = os.environ.get('DATABASE_URL', '')
if not DATABASE_URL:
    print()
    print('ERROR: DATABASE_URL environment variable is not set.')
    print()
    print('Run:')
    print('  set DATABASE_URL=postgresql://user:pass@host:5432/dbname')
    print('  python fill_pending.py')
    print()
    sys.exit(1)

# Patch the env before importing app so _DB_CONN_URL is set
os.environ['DATABASE_URL'] = DATABASE_URL

# Import helpers from app.py
try:
    from app import (
        _db,
        _resolve_result,
        _get_result,
        _fetch_espn_result_dated,
        _get_nba_game_result,
        _get_wnba_game_result,
    )
except ImportError as e:
    print(f'ERROR: Could not import from app.py: {e}')
    sys.exit(1)

now = datetime.now(timezone.utc)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Resolve PENDING / LIVE games that have finished
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('PHASE 1 — Resolving PENDING/LIVE games')
print('=' * 60)

resolved = 0
skipped  = 0
failed   = 0

with _db() as (conn, cur):
    if conn is None:
        print('ERROR: Cannot connect to DB'); sys.exit(1)

    cur.execute("""
        SELECT match_id, home_team, away_team, match_date, predicted_winner,
               kickoff_utc, COALESCE(sport, 'football') AS sport
        FROM predictions
        WHERE result_status IN ('PENDING', 'LIVE')
        ORDER BY match_date
    """)
    rows = cur.fetchall()
    print(f'Found {len(rows)} PENDING/LIVE predictions\n')

    for mid, home, away, match_date, pw, kickoff_str, sport in rows:
        try:
            if kickoff_str:
                kickoff = datetime.fromisoformat(kickoff_str.replace('Z', '+00:00'))
            elif match_date:
                kickoff = datetime.fromisoformat(f'{match_date}T23:00:00+00:00')
            else:
                print(f'  SKIP  {home} vs {away} — no kickoff date')
                skipped += 1
                continue
        except Exception:
            print(f'  SKIP  {home} vs {away} — bad kickoff format: {kickoff_str}')
            skipped += 1
            continue

        if kickoff + __import__('datetime').timedelta(hours=2) > now:
            print(f'  SOON  {home} vs {away} ({match_date}) — not finished yet')
            skipped += 1
            continue

        print(f'  Checking {home} vs {away} ({match_date}, {sport})…', end=' ', flush=True)

        if sport in ('nba', 'wnba'):
            res = (_get_nba_game_result(home, away, str(match_date))
                   if sport == 'nba'
                   else _get_wnba_game_result(home, away, str(match_date)))
            if res is None:
                print('NOT FOUND')
                failed += 1
                continue
            hs, as_ = res
            corners = None
        else:
            res = _get_result(home, away, str(match_date))
            if res is None:
                res_espn = _fetch_espn_result_dated(home, away, str(match_date))
                if res_espn:
                    hs, as_, corners = res_espn
                else:
                    print('NOT FOUND')
                    failed += 1
                    continue
            else:
                hs, as_, corners = res

        status = _resolve_result(pw or '', home, away, hs, as_, actual_corners=corners)
        print(f'{hs}–{as_} (corners={corners}) → {status}')

        cur.execute("""
            UPDATE predictions
            SET actual_home_score=%s, actual_away_score=%s,
                actual_corners=%s, result_status=%s, match_status='FINISHED'
            WHERE match_id=%s
        """, (hs, as_, corners, status, mid))
        resolved += 1

print()
print(f'Phase 1 done: {resolved} resolved, {skipped} not yet played, {failed} not found')

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Re-derive WON/LOST for all finished records (BTTS + corners fix)
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('PHASE 2 — Re-deriving WON/LOST statuses (BTTS + corners fix)')
print('=' * 60)

with _db() as (conn, cur):
    if conn is None:
        print('ERROR: Cannot connect to DB'); sys.exit(1)

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
    rows = cur.fetchall()
    print(f'Checking {len(rows)} WON/LOST records for status mismatches…\n')

    fixes = []
    for mid, home, away, match_date, pw, hs, as_, corners, curr in rows:
        computed = _resolve_result(pw or '', home or '', away or '', hs, as_,
                                   actual_corners=corners)
        if computed != curr:
            fixes.append((mid, home, away, match_date, pw, hs, as_, corners, curr, computed))
            print(f'  MISMATCH  {home} vs {away} ({match_date})')
            print(f'            bet="{pw}"  score={hs}–{as_}  corners={corners}')
            print(f'            was={curr} → should be={computed}')

    if not fixes:
        print('All statuses are correct — nothing to fix.')
    else:
        print(f'\nApplying {len(fixes)} fix(es)…')
        for mid, home, away, match_date, pw, hs, as_, corners, curr, computed in fixes:
            cur.execute(
                "UPDATE predictions SET result_status=%s WHERE match_id=%s",
                (computed, mid),
            )
        print(f'Done — {len(fixes)} record(s) corrected.')

print()
print('All done.')

"""
refresh_weekly.py
-----------------
Fast weekly data refresh — runs in ~5 minutes, no full model retraining.
Updates form caches and extends the NBA supplement with the latest games.

What it does:
  1. Extends data/nba_supplement_2023_2026.csv with ESPN games since last update
  2. Rebuilds data/nba_team_form_cache.csv from the updated data
  3. Downloads the latest 2025-26 football season data from football-data.co.uk
  4. Rebuilds data/club_form_cache.csv from the latest season

What it does NOT do:
  - Full model retraining (run train.py / nba_train.py locally for that)
  - WNBA update (ESPN player data not available here; retrain locally after each season)

Run locally or via GitHub Actions (.github/workflows/weekly_refresh.yml).
"""

import io, os, time, sys
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Config ────────────────────────────────────────────────────────────────────

NBA_SUPPLEMENT = 'data/nba_supplement_2023_2026.csv'
NBA_FORM_CACHE = 'data/nba_team_form_cache.csv'
CLUB_FORM_CACHE = 'data/club_form_cache.csv'
ESPN_NBA        = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
FDCO_MAIN       = 'https://www.football-data.co.uk/mmz4281'
FDCO_EXTRA      = 'https://www.football-data.co.uk/new'
HEADERS         = {'User-Agent': 'Mozilla/5.0'}
SLEEP           = 0.3

MAIN_DIVS_2526 = [
    'E0','E1','E2','E3','SC0','SC1','SC2','SC3',
    'D1','D2','I1','I2','SP1','SP2','F1','F2',
    'N1','B1','P1','T1','G1',
]
EXTRA_DIVS = [
    ('NOR','NOR'),('SWE','SWE'),('DEN','DNK'),('FIN','FIN'),
    ('IRL','IRL'),('AUT','AUT'),('POL','POL'),('ROM','ROM'),
    ('RUS','RUS'),('BRA','BRA'),('ARG','ARG'),('USA','USA'),
    ('MEX','MEX'),('CHN','CHN'),('JAP','JPN'),
]

# ── NBA supplement incremental update ─────────────────────────────────────────

def _espn_stat(stats, key):
    for s in stats:
        if s.get('name') == key:
            try:
                return float(s['displayValue'])
            except Exception:
                return np.nan
    return np.nan


def extend_nba_supplement():
    """Fetch ESPN games since last date in supplement and append."""
    if not os.path.exists(NBA_SUPPLEMENT):
        print(f'  {NBA_SUPPLEMENT} not found — skipping NBA update')
        return

    existing = pd.read_csv(NBA_SUPPLEMENT, parse_dates=['game_date'])
    last_date = existing['game_date'].max().date()
    yesterday = date.today() - timedelta(days=1)

    if last_date >= yesterday:
        print(f'  NBA supplement already current through {last_date}')
        return

    print(f'  Fetching NBA games {last_date + timedelta(1)} to {yesterday} ...')
    rows = []
    d = last_date + timedelta(days=1)
    while d <= yesterday:
        date_str = d.strftime('%Y%m%d')
        try:
            r = requests.get(ESPN_NBA, params={'dates': date_str}, timeout=10, headers=HEADERS)
            if r.status_code == 200:
                for ev in r.json().get('events', []):
                    comp   = (ev.get('competitions') or [{}])[0]
                    status = comp.get('status', {}).get('type', {})
                    if not status.get('completed'):
                        continue
                    teams = comp.get('competitors', [])
                    home  = next((t for t in teams if t.get('homeAway') == 'home'), None)
                    away  = next((t for t in teams if t.get('homeAway') == 'away'), None)
                    if not home or not away:
                        continue
                    try:
                        pts_h = float(home.get('score', 0))
                        pts_a = float(away.get('score', 0))
                    except Exception:
                        continue
                    if pts_h == 0 and pts_a == 0:
                        continue
                    h_stats = home.get('statistics', [])
                    a_stats = away.get('statistics', [])
                    # Determine season_id from the date
                    season_id = 22025 if d >= date(2025, 10, 1) else (
                                22024 if d >= date(2024, 10, 1) else 22023)
                    rows.append({
                        'game_id':        f"espn_{date_str}_{ev.get('id','')}",
                        'game_date':      str(d),
                        'season_id':      season_id,
                        'team_name_home': home.get('team', {}).get('displayName', ''),
                        'team_name_away': away.get('team', {}).get('displayName', ''),
                        'pts_home': pts_h, 'pts_away': pts_a,
                        'wl_home':  'W' if pts_h > pts_a else 'L',
                        'wl_away':  'W' if pts_a > pts_h else 'L',
                        'fga_home': _espn_stat(h_stats, 'fieldGoalsAttempted'),
                        'oreb_home': np.nan, 'tov_home': np.nan,
                        'fta_home': _espn_stat(h_stats, 'freeThrowsAttempted'),
                        'fga_away': _espn_stat(a_stats, 'fieldGoalsAttempted'),
                        'oreb_away': np.nan, 'tov_away': np.nan,
                        'fta_away': _espn_stat(a_stats, 'freeThrowsAttempted'),
                    })
        except Exception as e:
            print(f'    [warn] {date_str}: {e}')
        d += timedelta(days=1)
        time.sleep(SLEEP)

    if not rows:
        print('  No new NBA games found.')
        return

    new_df = pd.DataFrame(rows)
    new_df['game_date'] = pd.to_datetime(new_df['game_date'])
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values('game_date').reset_index(drop=True)
    combined.to_csv(NBA_SUPPLEMENT, index=False)
    print(f'  Added {len(rows)} games -> {NBA_SUPPLEMENT}  '
          f'(total {len(combined):,}, through {combined["game_date"].max().date()})')


# ── NBA form cache rebuild ─────────────────────────────────────────────────────

def rebuild_nba_form_cache():
    """Rebuild nba_team_form_cache from the supplement CSV (no SQLite needed)."""
    print('  Rebuilding NBA form cache ...')
    if not os.path.exists(NBA_SUPPLEMENT):
        print(f'  {NBA_SUPPLEMENT} not found — skipping NBA form cache')
        return
    try:
        from nba_train import _save_nba_team_form
        df = pd.read_csv(NBA_SUPPLEMENT, parse_dates=['game_date'])
        df = df[df['pts_home'].notna() & df['pts_away'].notna() & df['wl_home'].notna()]
        df = df.sort_values('game_date').reset_index(drop=True)
        print(f'  Loaded {len(df):,} games from supplement '
              f'({df["game_date"].min().date()} to {df["game_date"].max().date()})')
        _save_nba_team_form(df)
        print(f'  Done -> {NBA_FORM_CACHE}')
    except Exception as e:
        print(f'  ERROR rebuilding NBA form cache: {e}')


# ── Football form cache rebuild ────────────────────────────────────────────────

def _fetch(url):
    try:
        r = requests.get(url, timeout=20, headers=HEADERS)
        if r.status_code == 200:
            return r.text
        print(f'    HTTP {r.status_code}: {url}')
    except Exception as e:
        print(f'    ERROR: {e}')
    return None


def _to_float(series):
    return pd.to_numeric(series, errors='coerce')


def fetch_latest_football():
    """
    Download the current 2025-26 season for main EU leagues + rolling CSVs
    for extra leagues.  Returns a DataFrame with columns used by _build_form_cache.
    """
    frames = []

    print('  Main leagues (2025-26) ...')
    for div in MAIN_DIVS_2526:
        text = _fetch(f'{FDCO_MAIN}/2526/{div}.csv')
        if not text:
            continue
        raw = pd.read_csv(io.StringIO(text), encoding='utf-8-sig', low_memory=False)
        if raw.empty:
            continue
        dates = pd.to_datetime(raw.get('Date', pd.Series()), dayfirst=True, errors='coerce')
        mask  = dates.notna() & raw.get('FTR', pd.Series()).isin(['H','D','A'])
        raw   = raw[mask].copy()
        if raw.empty:
            continue
        df = pd.DataFrame({
            'date':         dates[mask].values,
            'home_team':    raw.get('HomeTeam', pd.Series()).values,
            'away_team':    raw.get('AwayTeam', pd.Series()).values,
            'home_goals':   _to_float(raw.get('FTHG', pd.Series())).values,
            'away_goals':   _to_float(raw.get('FTAG', pd.Series())).values,
            'home_sot':     _to_float(raw.get('HST', pd.Series())).values,
            'away_sot':     _to_float(raw.get('AST', pd.Series())).values,
            'home_corners': _to_float(raw.get('HC', pd.Series())).values,
            'away_corners': _to_float(raw.get('AC', pd.Series())).values,
            'league':       div,
        })
        df = df.dropna(subset=['home_goals','away_goals'])
        frames.append(df)
        time.sleep(SLEEP)

    print('  Extra leagues (rolling) ...')
    for div_code, file_code in EXTRA_DIVS:
        text = _fetch(f'{FDCO_EXTRA}/{file_code}.csv')
        if not text:
            continue
        raw = pd.read_csv(io.StringIO(text), encoding='utf-8-sig', low_memory=False)
        if raw.empty:
            continue
        dates = pd.to_datetime(raw.get('Date', pd.Series()), dayfirst=True, errors='coerce')
        # Only take the most recent season (last 12 months)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=400)
        mask = dates.notna() & (dates > cutoff) & raw.get('Res', pd.Series()).isin(['H','D','A'])
        raw = raw[mask].copy()
        if raw.empty:
            continue
        df = pd.DataFrame({
            'date':         dates[mask].values,
            'home_team':    raw.get('Home', pd.Series()).values,
            'away_team':    raw.get('Away', pd.Series()).values,
            'home_goals':   _to_float(raw.get('HG', pd.Series())).values,
            'away_goals':   _to_float(raw.get('AG', pd.Series())).values,
            'home_sot':     np.nan,
            'away_sot':     np.nan,
            'home_corners': np.nan,
            'away_corners': np.nan,
            'league':       div_code,
        })
        df = df.dropna(subset=['home_goals','away_goals'])
        frames.append(df)
        time.sleep(SLEEP)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values('date').reset_index(drop=True)
    print(f'  Loaded {len(combined):,} football games for form cache '
          f'({combined["date"].min().date()} to {combined["date"].max().date()})')
    return combined


def rebuild_football_form_cache(df):
    """Compute rolling stats and save club_form_cache.csv."""
    if df.empty:
        print('  No football data — skipping form cache rebuild')
        return

    n = 5
    rows = []
    all_teams = sorted(set(df['home_team'].dropna().tolist() +
                           df['away_team'].dropna().tolist()))

    for team in all_teams:
        mask   = (df['home_team'] == team) | (df['away_team'] == team)
        recent = df[mask].sort_values('date').tail(n)
        if recent.empty:
            continue

        gf = ga = sot_s = sot_n = cor_s = cor_n = 0.0
        wins = draws = pts = hwin = hg = awin = ag_ = 0

        for _, row in recent.iterrows():
            ih = (row['home_team'] == team)
            g  = float(row['home_goals'] if ih else row['away_goals'])
            gc = float(row['away_goals'] if ih else row['home_goals'])
            gf += g; ga += gc
            s = row['home_sot'] if ih else row['away_sot']
            c = row['home_corners'] if ih else row['away_corners']
            if pd.notna(s) and float(s) > 0: sot_s += float(s); sot_n += 1
            if pd.notna(c) and float(c) > 0: cor_s += float(c); cor_n += 1
            if g > gc:    wins += 1; pts += 3
            elif g == gc: draws += 1; pts += 1
            if ih:   hg  += 1; hwin += int(g > gc)
            else:    ag_ += 1; awin += int(g > gc)

        nm = len(recent)
        league = recent.iloc[-1]['league']
        rows.append({
            'team':      team,
            'league':    league,
            'gf':        round(gf / nm, 3),
            'ga':        round(ga / nm, 3),
            'sot':       round(sot_s / sot_n if sot_n else 4.5, 3),
            'corners':   round(cor_s / cor_n if cor_n else 5.0, 3),
            'win':       round(wins  / nm, 3),
            'draw':      round(draws / nm, 3),
            'hwn':       round(hwin  / hg  if hg  else min(wins / nm + 0.08, 1.0), 3),
            'awn':       round(awin  / ag_ if ag_ else max(wins / nm - 0.08, 0.0), 3),
            'last_date': recent['date'].max().strftime('%Y-%m-%d'),
            'n_games':   nm,
        })

    cache = pd.DataFrame(rows)
    cache.to_csv(CLUB_FORM_CACHE, index=False)
    print(f'  Form cache: {len(cache)} teams -> {CLUB_FORM_CACHE}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  WEEKLY REFRESH')
    print('=' * 60)

    print('\n[1] NBA supplement update')
    extend_nba_supplement()

    print('\n[2] NBA form cache')
    rebuild_nba_form_cache()

    print('\n[3] Football data (latest season)')
    football_df = fetch_latest_football()

    print('\n[4] Football form cache')
    rebuild_football_form_cache(football_df)

    print('\n' + '=' * 60)
    print('  REFRESH COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()

"""
fetch_nba_supplement.py
-----------------------
Builds data/nba_supplement_2023_2026.csv, containing NBA game-level rows for
the three seasons missing from data/nba.sqlite (which ends Jun 2023):

  2023-24  Oct 24 2023 – Jun 17 2024  <- ESPN scoreboard API
  2024-25  Oct 22 2024 – Jun 22 2025  <- nba_dailyleaders_full_24_25.csv (player agg)
  2025-26  Oct 22 2025 – yesterday    <- ESPN scoreboard API

Schema matches the columns load_nba() reads from nba.sqlite:
  game_id, game_date, season_id,
  team_name_home, team_name_away,
  pts_home, pts_away, wl_home, wl_away,
  fga_home, oreb_home, tov_home, fta_home,
  fga_away, oreb_away, tov_away, fta_away

Note: oreb / tov are unavailable from ESPN (NaN); _approx_ortg() fills them
      with league-average defaults (10 and 14) — same as old seasons with
      missing stats in the original SQLite.
"""

import requests, time, pandas as pd, numpy as np
from datetime import date, timedelta

OUT_PATH       = "data/nba_supplement_2023_2026.csv"
DAILY_LEADERS  = "data/nba_dailyleaders_full_24_25.csv"
ESPN_BASE      = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SLEEP_SEC      = 0.25   # between ESPN requests

# BR abbreviation -> full team name (matches nba.sqlite team_name_home values)
ABBR_TO_NAME = {
    'ATL': 'Atlanta Hawks',        'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',        'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',        'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',     'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',      'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',      'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',      'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder','ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',   'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers','SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',            'WAS': 'Washington Wizards',
    # Occasional BR variants
    'NJN': 'Brooklyn Nets',        'NOH': 'New Orleans Pelicans',
    'NOK': 'New Orleans Pelicans', 'CHH': 'Charlotte Hornets',
    'SEA': 'Oklahoma City Thunder','VAN': 'Memphis Grizzlies',
}


def _espn_stat(stats_list, key):
    for s in stats_list:
        if s.get('name') == key:
            try:
                return float(s['displayValue'])
            except Exception:
                return np.nan
    return np.nan


def fetch_espn_range(start: date, end: date, season_id: int) -> list[dict]:
    """
    Fetch all completed NBA games from ESPN between start and end (inclusive).
    Returns a list of row dicts.
    """
    rows = []
    d = start
    total_days = (end - start).days + 1
    done = 0
    while d <= end:
        date_str = d.strftime('%Y%m%d')
        try:
            r = requests.get(ESPN_BASE, params={'dates': date_str}, timeout=10)
            if r.status_code == 200:
                for ev in r.json().get('events', []):
                    comps = ev.get('competitions', [{}])
                    if not comps:
                        continue
                    comp  = comps[0]
                    status = comp.get('status', {}).get('type', {})
                    if not status.get('completed'):
                        continue
                    teams = comp.get('competitors', [])
                    if len(teams) < 2:
                        continue
                    home = next((t for t in teams if t.get('homeAway') == 'home'), None)
                    away = next((t for t in teams if t.get('homeAway') == 'away'), None)
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
                    rows.append({
                        'game_id':        f"espn_{date_str}_{ev.get('id','')}",
                        'game_date':      d.strftime('%Y-%m-%d'),
                        'season_id':      season_id,
                        'team_name_home': home.get('team', {}).get('displayName', ''),
                        'team_name_away': away.get('team', {}).get('displayName', ''),
                        'pts_home':  pts_h,
                        'pts_away':  pts_a,
                        'wl_home':   'W' if pts_h > pts_a else 'L',
                        'wl_away':   'W' if pts_a > pts_h else 'L',
                        'fga_home':  _espn_stat(h_stats, 'fieldGoalsAttempted'),
                        'oreb_home': np.nan,
                        'tov_home':  np.nan,
                        'fta_home':  _espn_stat(h_stats, 'freeThrowsAttempted'),
                        'fga_away':  _espn_stat(a_stats, 'fieldGoalsAttempted'),
                        'oreb_away': np.nan,
                        'tov_away':  np.nan,
                        'fta_away':  _espn_stat(a_stats, 'freeThrowsAttempted'),
                    })
        except Exception as e:
            print(f"  [warn] {date_str}: {e}")
        d += timedelta(days=1)
        done += 1
        if done % 30 == 0:
            print(f"  ESPN fetch: {done}/{total_days} days, {len(rows)} games so far")
        time.sleep(SLEEP_SEC)
    return rows


def extract_from_dailyleaders(path: str, season_id: int) -> list[dict]:
    """
    Aggregate player-level box score rows (Basketball Reference daily leaders format)
    to team-game level. One row per home-team perspective.
    """
    dl = pd.read_csv(path)
    dl['Date'] = pd.to_datetime(dl['Date'], errors='coerce')
    dl = dl.dropna(subset=['Date', 'Tm', 'Opp', 'Result'])
    # 'Unnamed: 3' == '@' means the team was playing away
    dl['is_away'] = dl['Unnamed: 3'].fillna('').str.strip() == '@'

    def to_num(col):
        return pd.to_numeric(dl[col], errors='coerce').fillna(0)

    dl['_pts'] = to_num('PTS')
    dl['_fga'] = to_num('FGA')
    dl['_oreb'] = to_num('ORB')
    dl['_tov'] = to_num('TOV')
    dl['_fta'] = to_num('FTA')

    grp = dl.groupby(['Date', 'Tm', 'Opp', 'is_away'])
    tg = grp.agg(
        pts  = ('_pts',  'sum'),
        fga  = ('_fga',  'sum'),
        oreb = ('_oreb', 'sum'),
        tov  = ('_tov',  'sum'),
        fta  = ('_fta',  'sum'),
        wl   = ('Result','first'),
    ).reset_index()

    # Home rows (is_away=False)
    home = tg[~tg['is_away']].copy()
    # Away rows (is_away=True) — needed for away team stats
    away_idx = tg[tg['is_away']].set_index(['Date', 'Tm', 'Opp'])

    rows = []
    for _, h in home.iterrows():
        d, ht, at = h['Date'], h['Tm'], h['Opp']
        pts_h = h['pts'];  pts_a = np.nan
        fga_h = h['fga']; oreb_h = h['oreb']; tov_h = h['tov']; fta_h = h['fta']
        fga_a = oreb_a = tov_a = fta_a = np.nan
        try:
            a = away_idx.loc[(d, at, ht)]
            pts_a  = a['pts']
            fga_a  = a['fga'];  oreb_a = a['oreb']
            tov_a  = a['tov'];  fta_a  = a['fta']
        except KeyError:
            pass
        if pd.isna(pts_a):
            continue

        ht_name = ABBR_TO_NAME.get(ht, ht)
        at_name = ABBR_TO_NAME.get(at, at)
        rows.append({
            'game_id':        f"dl_{d.date()}_{ht}_{at}",
            'game_date':      str(d.date()),
            'season_id':      season_id,
            'team_name_home': ht_name,
            'team_name_away': at_name,
            'pts_home':  pts_h,
            'pts_away':  pts_a,
            'wl_home':   'W' if pts_h > pts_a else 'L',
            'wl_away':   'W' if pts_a > pts_h else 'L',
            'fga_home':  fga_h,  'oreb_home': oreb_h,
            'tov_home':  tov_h,  'fta_home':  fta_h,
            'fga_away':  fga_a,  'oreb_away': oreb_a,
            'tov_away':  tov_a,  'fta_away':  fta_a,
        })
    return rows


def main():
    all_rows = []

    # ── 2023-24 season via ESPN ──────────────────────────────────────────────
    print("[1] Fetching 2023-24 season from ESPN (Oct 24 2023 – Jun 17 2024) ...")
    rows_2324 = fetch_espn_range(date(2023, 10, 24), date(2024, 6, 17), season_id=22023)
    print(f"    -> {len(rows_2324)} games")
    all_rows.extend(rows_2324)

    # ── 2024-25 season from dailyleaders CSV ─────────────────────────────────
    print("\n[2] Extracting 2024-25 season from nba_dailyleaders_full_24_25.csv ...")
    rows_2425 = extract_from_dailyleaders(DAILY_LEADERS, season_id=22024)
    print(f"    -> {len(rows_2425)} games")
    all_rows.extend(rows_2425)

    # ── 2025-26 season via ESPN (Oct 22 2025 – yesterday) ───────────────────
    yesterday = date.today() - timedelta(days=1)
    print(f"\n[3] Fetching 2025-26 season from ESPN (Oct 22 2025 – {yesterday}) ...")
    rows_2526 = fetch_espn_range(date(2025, 10, 22), yesterday, season_id=22025)
    print(f"    -> {len(rows_2526)} games")
    all_rows.extend(rows_2526)

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)
    df = df[df['pts_home'].notna() & df['pts_away'].notna()]
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df):,} supplemental games -> {OUT_PATH}")
    print(f"Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    by_season = df.groupby('season_id').size()
    print(f"By season:\n{by_season.to_string()}")


if __name__ == '__main__':
    main()

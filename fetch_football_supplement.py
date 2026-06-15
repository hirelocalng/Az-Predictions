"""
fetch_football_supplement.py
----------------------------
Downloads missing match data from football-data.co.uk and appends to
data/Matches.csv.

Two sources:
  1. Main European leagues (mmz4281/{season}/{div}.csv)
     - 2025-26 season: E0,E1,E2,E3,SC0,SC1,SC2,SC3,D1,D2,I1,I2,SP1,SP2,F1,F2,N1,B1,P1,T1,G1
     - Columns include shots, corners, B365 odds
  2. Extra leagues (new/{code}.csv)
     - Full rolling historical CSV; we filter to rows after the cutoff already
       present in Matches.csv for each division
     - BRA,ARG,USA,MEX,NOR,SWE,DIN,FIN,IRL,ROM,RUS,AUT,POL,CHN,DEN,JAP
     - Columns: Home/Away team, HG/AG, Res, PSCH/PSCD/PSCA (Pinnacle odds)
     - No shots/corners in these files

Column mapping to Matches.csv schema:
  Main leagues: Div->Division, Date->MatchDate, Time->MatchTime,
    HomeTeam->HomeTeam, AwayTeam->AwayTeam, FTHG->FTHome, FTAG->FTAway,
    FTR->FTResult, HTHG->HTHome, HTAG->HTAway, HTR->HTResult,
    HS->HomeShots, AS->AwayShots, HST->HomeTarget, AST->AwayTarget,
    HF->HomeFouls, AF->AwayFouls, HC->HomeCorners, AC->AwayCorners,
    HY->HomeYellow, AY->AwayYellow, HR->HomeRed, AR->AwayRed,
    B365H->OddHome, B365D->OddDraw, B365A->OddAway

  Extra leagues: Home->HomeTeam, Away->AwayTeam, HG->FTHome, AG->FTAway,
    Res->FTResult, PSCH->OddHome, PSCD->OddDraw, PSCA->OddAway
    (no shots/corners -- those columns become NaN)

Usage:
    python fetch_football_supplement.py
"""

import io
import time
import requests
import pandas as pd
import numpy as np

MATCHES_PATH = 'data/Matches.csv'
OUT_PATH     = MATCHES_PATH   # update in-place
BASE_URL_MAIN  = 'https://www.football-data.co.uk/mmz4281'
BASE_URL_EXTRA = 'https://www.football-data.co.uk/new'
SLEEP_SEC    = 0.5
HEADERS      = {'User-Agent': 'Mozilla/5.0'}

# Main European leagues to download for 2025-26 season
MAIN_DIVS = [
    'E0', 'E1', 'E2', 'E3',
    'SC0', 'SC1', 'SC2', 'SC3',
    'D1', 'D2',
    'I1', 'I2',
    'SP1', 'SP2',
    'F1', 'F2',
    'N1', 'B1', 'P1', 'T1', 'G1',
]
MAIN_SEASON = '2526'   # 2025-26

# Extra league: (Matches.csv Division code, football-data.co.uk filename)
EXTRA_DIVS = [
    ('NOR', 'NOR'),
    ('SWE', 'SWE'),
    ('DEN', 'DNK'),
    ('FIN', 'FIN'),
    ('IRL', 'IRL'),
    ('AUT', 'AUT'),
    ('POL', 'POL'),
    ('ROM', 'ROM'),
    ('RUS', 'RUS'),
    ('BRA', 'BRA'),
    ('ARG', 'ARG'),
    ('USA', 'USA'),
    ('MEX', 'MEX'),
    ('CHN', 'CHN'),
    ('JAP', 'JPN'),
]

# All columns that Matches.csv expects
MATCHES_COLS = [
    'Division', 'MatchDate', 'MatchTime', 'HomeTeam', 'AwayTeam',
    'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
    'FTHome', 'FTAway', 'FTResult',
    'HTHome', 'HTAway', 'HTResult',
    'HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget',
    'HomeFouls', 'AwayFouls', 'HomeCorners', 'AwayCorners',
    'HomeYellow', 'AwayYellow', 'HomeRed', 'AwayRed',
    'OddHome', 'OddDraw', 'OddAway',
    'MaxHome', 'MaxDraw', 'MaxAway',
    'Over25', 'Under25', 'MaxOver25', 'MaxUnder25',
    'HandiSize', 'HandiHome', 'HandiAway',
    'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB',
]


def _fetch(url):
    try:
        r = requests.get(url, timeout=20, headers=HEADERS)
        if r.status_code == 200:
            return r.text
        print(f'  HTTP {r.status_code}: {url}')
    except Exception as e:
        print(f'  ERROR fetching {url}: {e}')
    return None


def _parse_date(series, dayfirst=True):
    return pd.to_datetime(series, dayfirst=dayfirst, errors='coerce')


def _to_float(series):
    return pd.to_numeric(series, errors='coerce')


def _blank(n):
    return pd.Series(np.nan, index=range(n))


def fetch_main_season(div, season):
    """Download one main-league season CSV and return a Matches.csv-schema df."""
    url = f'{BASE_URL_MAIN}/{season}/{div}.csv'
    text = _fetch(url)
    if text is None:
        return pd.DataFrame()

    raw = pd.read_csv(io.StringIO(text), encoding='utf-8-sig', low_memory=False)
    if raw.empty:
        return pd.DataFrame()

    n = len(raw)

    # Parse date (football-data.co.uk uses DD/MM/YYYY)
    dates = _parse_date(raw.get('Date', pd.Series()), dayfirst=True)

    df = pd.DataFrame(index=range(n))
    df['Division']    = div
    df['MatchDate']   = dates.dt.strftime('%Y-%m-%d')
    df['MatchTime']   = raw.get('Time', pd.Series(np.nan, index=range(n)))
    df['HomeTeam']    = raw.get('HomeTeam', pd.Series(np.nan, index=range(n)))
    df['AwayTeam']    = raw.get('AwayTeam', pd.Series(np.nan, index=range(n)))
    # ELO / form — not in this source
    df['HomeElo']     = np.nan; df['AwayElo']   = np.nan
    df['Form3Home']   = np.nan; df['Form5Home'] = np.nan
    df['Form3Away']   = np.nan; df['Form5Away'] = np.nan
    # Full-time / half-time scores
    df['FTHome']      = _to_float(raw.get('FTHG', pd.Series()))
    df['FTAway']      = _to_float(raw.get('FTAG', pd.Series()))
    df['FTResult']    = raw.get('FTR', pd.Series())
    df['HTHome']      = _to_float(raw.get('HTHG', pd.Series()))
    df['HTAway']      = _to_float(raw.get('HTAG', pd.Series()))
    df['HTResult']    = raw.get('HTR', pd.Series())
    # Match stats
    df['HomeShots']   = _to_float(raw.get('HS',  pd.Series()))
    df['AwayShots']   = _to_float(raw.get('AS',  pd.Series()))
    df['HomeTarget']  = _to_float(raw.get('HST', pd.Series()))
    df['AwayTarget']  = _to_float(raw.get('AST', pd.Series()))
    df['HomeFouls']   = _to_float(raw.get('HF',  pd.Series()))
    df['AwayFouls']   = _to_float(raw.get('AF',  pd.Series()))
    df['HomeCorners'] = _to_float(raw.get('HC',  pd.Series()))
    df['AwayCorners'] = _to_float(raw.get('AC',  pd.Series()))
    df['HomeYellow']  = _to_float(raw.get('HY',  pd.Series()))
    df['AwayYellow']  = _to_float(raw.get('AY',  pd.Series()))
    df['HomeRed']     = _to_float(raw.get('HR',  pd.Series()))
    df['AwayRed']     = _to_float(raw.get('AR',  pd.Series()))
    # Odds (B365 as primary)
    df['OddHome']     = _to_float(raw.get('B365H', pd.Series()))
    df['OddDraw']     = _to_float(raw.get('B365D', pd.Series()))
    df['OddAway']     = _to_float(raw.get('B365A', pd.Series()))
    # Optional odds columns — NaN
    for c in ['MaxHome','MaxDraw','MaxAway','Over25','Under25',
              'MaxOver25','MaxUnder25','HandiSize','HandiHome','HandiAway',
              'C_LTH','C_LTA','C_VHD','C_VAD','C_HTB','C_PHB']:
        df[c] = np.nan

    # Drop rows with no valid date or result
    df = df[dates.notna() & df['FTResult'].isin(['H','D','A'])].copy()
    df['_date_parsed'] = dates[df.index]
    return df


def fetch_extra_league(div_code, file_code, cutoff_date):
    """
    Download the extra-league rolling CSV and return rows strictly after cutoff_date.
    These files use: Home/Away, HG/AG, Res, PSCH/PSCD/PSCA
    """
    url = f'{BASE_URL_EXTRA}/{file_code}.csv'
    text = _fetch(url)
    if text is None:
        return pd.DataFrame()

    raw = pd.read_csv(io.StringIO(text), encoding='utf-8-sig', low_memory=False)
    if raw.empty:
        return pd.DataFrame()

    dates = _parse_date(raw.get('Date', pd.Series()), dayfirst=True)
    mask  = dates.notna() & (dates > cutoff_date)
    raw   = raw[mask].copy()
    dates = dates[mask]
    if raw.empty:
        return pd.DataFrame()

    n = len(raw)
    idx = raw.index

    df = pd.DataFrame(index=range(n))
    df['Division']    = div_code
    df['MatchDate']   = dates.dt.strftime('%Y-%m-%d').values
    df['MatchTime']   = raw.get('Time', pd.Series(np.nan, index=idx)).values
    df['HomeTeam']    = raw.get('Home', pd.Series(np.nan, index=idx)).values
    df['AwayTeam']    = raw.get('Away', pd.Series(np.nan, index=idx)).values
    # ELO / form — not available
    for c in ['HomeElo','AwayElo','Form3Home','Form5Home','Form3Away','Form5Away']:
        df[c] = np.nan
    # Scores
    df['FTHome']   = pd.to_numeric(raw.get('HG', pd.Series(index=idx)), errors='coerce').values
    df['FTAway']   = pd.to_numeric(raw.get('AG', pd.Series(index=idx)), errors='coerce').values
    # Res column in these files: H / D / A (same convention)
    df['FTResult'] = raw.get('Res', pd.Series(np.nan, index=idx)).values
    # No half-time or match stats
    for c in ['HTHome','HTAway','HTResult',
              'HomeShots','AwayShots','HomeTarget','AwayTarget',
              'HomeFouls','AwayFouls','HomeCorners','AwayCorners',
              'HomeYellow','AwayYellow','HomeRed','AwayRed']:
        df[c] = np.nan
    # Odds — Pinnacle closing odds as primary
    df['OddHome'] = pd.to_numeric(raw.get('PSCH', pd.Series(index=idx)), errors='coerce').values
    df['OddDraw'] = pd.to_numeric(raw.get('PSCD', pd.Series(index=idx)), errors='coerce').values
    df['OddAway'] = pd.to_numeric(raw.get('PSCA', pd.Series(index=idx)), errors='coerce').values
    # Optional
    for c in ['MaxHome','MaxDraw','MaxAway','Over25','Under25',
              'MaxOver25','MaxUnder25','HandiSize','HandiHome','HandiAway',
              'C_LTH','C_LTA','C_VHD','C_VAD','C_HTB','C_PHB']:
        df[c] = np.nan

    df = df[pd.Series(df['FTResult']).isin(['H','D','A'])].copy()
    df['_date_parsed'] = dates[mask].values
    return df


def main():
    print('Loading existing Matches.csv ...')
    existing = pd.read_csv(MATCHES_PATH, low_memory=False)
    existing['_date_parsed'] = pd.to_datetime(existing['MatchDate'], errors='coerce')
    print(f'  Existing: {len(existing):,} rows  ({existing["_date_parsed"].min().date()} '
          f'to {existing["_date_parsed"].max().date()})')

    # Last date per division (for cutoff filtering)
    cutoffs = (existing.groupby('Division')['_date_parsed'].max()
               .fillna(pd.Timestamp('2000-01-01')))

    new_frames = []

    # ── 1. Main European leagues (2025-26 full season) ────────────────────────
    print(f'\nFetching main leagues (season {MAIN_SEASON}) ...')
    for div in MAIN_DIVS:
        cutoff = cutoffs.get(div, pd.Timestamp('2000-01-01'))
        df = fetch_main_season(div, MAIN_SEASON)
        if df.empty:
            print(f'  {div}: no data')
            continue
        # Filter to rows after the cutoff already in Matches.csv
        df = df[df['_date_parsed'] > cutoff].copy()
        if df.empty:
            print(f'  {div}: already up-to-date (last: {cutoff.date()})')
        else:
            print(f'  {div}: +{len(df)} rows  ({df["_date_parsed"].min().date()} '
                  f'to {df["_date_parsed"].max().date()})')
            new_frames.append(df)
        time.sleep(SLEEP_SEC)

    # ── 2. Extra leagues ──────────────────────────────────────────────────────
    print('\nFetching extra leagues ...')
    for div_code, file_code in EXTRA_DIVS:
        cutoff = cutoffs.get(div_code, pd.Timestamp('2000-01-01'))
        df = fetch_extra_league(div_code, file_code, cutoff)
        if df.empty:
            print(f'  {div_code}: no new rows after {cutoff.date()}')
        else:
            print(f'  {div_code}: +{len(df)} rows  ({df["_date_parsed"].min().date()} '
                  f'to {df["_date_parsed"].max().date()})')
            new_frames.append(df)
        time.sleep(SLEEP_SEC)

    if not new_frames:
        print('\nNo new data found — Matches.csv is already current.')
        return

    # ── Merge ─────────────────────────────────────────────────────────────────
    print('\nMerging ...')
    supplement = pd.concat(new_frames, ignore_index=True)

    # Dedup: drop rows where (Division, MatchDate, HomeTeam, AwayTeam) already exist
    existing_keys = set(zip(
        existing['Division'].astype(str),
        existing['MatchDate'].astype(str),
        existing['HomeTeam'].astype(str),
        existing['AwayTeam'].astype(str),
    ))
    before = len(supplement)
    sup_keys = list(zip(
        supplement['Division'].astype(str),
        supplement['MatchDate'].astype(str),
        supplement['HomeTeam'].astype(str),
        supplement['AwayTeam'].astype(str),
    ))
    supplement = supplement[[k not in existing_keys for k in sup_keys]].copy()
    print(f'  Supplement rows before dedup: {before}  after: {len(supplement)}')

    # Drop internal helper column
    existing.drop(columns=['_date_parsed'], inplace=True)
    supplement.drop(columns=['_date_parsed'], inplace=True)

    # Align columns exactly to Matches.csv schema
    for c in MATCHES_COLS:
        if c not in existing.columns:
            existing[c] = np.nan
        if c not in supplement.columns:
            supplement[c] = np.nan

    combined = pd.concat([existing[MATCHES_COLS], supplement[MATCHES_COLS]], ignore_index=True)
    combined['_d'] = pd.to_datetime(combined['MatchDate'], errors='coerce')
    combined = combined.sort_values('_d').drop(columns=['_d']).reset_index(drop=True)

    combined.to_csv(OUT_PATH, index=False)
    print(f'\nSaved {len(combined):,} rows -> {OUT_PATH}')
    print(f'  Added {len(supplement):,} new rows')
    by_div = supplement.groupby('Division').size().sort_values(ascending=False)
    print(f'  By division:\n{by_div.to_string()}')


if __name__ == '__main__':
    main()

"""
nba_predict.py — NBA and WNBA match prediction using trained XGBoost models.

Data sources
------------
NBA fixtures  : TheSportsDB API (league id 4387); BallDontLie fallback for form
NBA team form : balldontlie.io v1 API (free, no key required)
WNBA fixtures : TheSportsDB API (league id 4328)
WNBA team form: data/wnba_team_form_cache.csv (up to 2020 season)
H2H           : data/nba.sqlite (historical)
"""

import os, pickle, sqlite3, time, traceback
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─── Paths ────────────────────────────────────────────────────────────────────
NBA_RESULT_PATH  = 'nba_result_model.pkl'
NBA_OU_PATH      = 'nba_ou_model.pkl'
WNBA_RESULT_PATH = 'wnba_result_model.pkl'
WNBA_OU_PATH     = 'wnba_ou_model.pkl'
NBA_DB           = 'data/nba.sqlite'
NBA_FORM_CACHE   = 'data/nba_team_form_cache.csv'
WNBA_FORM_CACHE  = 'data/wnba_team_form_cache.csv'

NBA_OU_LINE  = 220.5
WNBA_OU_LINE = 163.5  # fallback when ESPN line unavailable; recent seasons median ~163

BDL_BASE      = 'https://www.balldontlie.io/api/v1'
BDL_WNBA_BASE = 'https://api.balldontlie.io/wnba/v1'
TSDB_BASE     = 'https://www.thesportsdb.com/api/v1/json/3'
WNBA_TSDB_ID  = '4328'
NBA_TSDB_ID   = '4387'

# ─── Model loading ────────────────────────────────────────────────────────────
_NBA_RES = _NBA_OU = _WNBA_RES = _WNBA_OU = None

def _load_models():
    global _NBA_RES, _NBA_OU, _WNBA_RES, _WNBA_OU
    try:
        _NBA_RES  = pickle.load(open(NBA_RESULT_PATH,  'rb'))
        _NBA_OU   = pickle.load(open(NBA_OU_PATH,      'rb'))
        _WNBA_RES = pickle.load(open(WNBA_RESULT_PATH, 'rb'))
        _WNBA_OU  = pickle.load(open(WNBA_OU_PATH,     'rb'))
    except Exception as e:
        print(f'[nba_predict] model load error: {e}')

_load_models()

# ─── WNBA team name mapping ───────────────────────────────────────────────────
# Maps TSDB / common full names -> abbreviations used in training data
WNBA_NAME_TO_ABBR = {
    'atlanta dream':        'ATL',
    'chicago sky':          'CHI',
    'connecticut sun':      'CON',
    'dallas wings':         'DAL',
    'golden state valkyries': 'GSV',
    'indiana fever':        'IND',
    'las vegas aces':       'LVA',
    'los angeles sparks':   'LAS',
    'minnesota lynx':       'MIN',
    'new york liberty':     'NYL',
    'phoenix mercury':      'PHO',
    'portland fire':        'POR',
    'sacramento monarchs':  'SAC',
    'san antonio stars':    'SAN',
    'seattle storm':        'SEA',
    'utah starzz':          'UTA',
    'washington mystics':   'WAS',
    'houston comets':       'HOU',
    'cleveland rockers':    'CLE',
    'charlotte sting':      'CHA',
    'detroit shock':        'DET',
    'new england fury':     'NEF',
    'orlando miracle':      'ORL',
    'miami sol':            'MIA',
    'portland trail blazers wnba': 'POR',
}

def _wnba_abbr(name):
    return WNBA_NAME_TO_ABBR.get(name.lower().strip(), name[:3].upper())

# ─── NBA team name → ESPN abbreviation ───────────────────────────────────────
_NBA_NAME_TO_ABBR = {
    'atlanta hawks':          'ATL',
    'boston celtics':         'BOS',
    'brooklyn nets':          'BKN',
    'charlotte hornets':      'CHA',
    'chicago bulls':          'CHI',
    'cleveland cavaliers':    'CLE',
    'dallas mavericks':       'DAL',
    'denver nuggets':         'DEN',
    'detroit pistons':        'DET',
    'golden state warriors':  'GSW',
    'houston rockets':        'HOU',
    'indiana pacers':         'IND',
    'los angeles clippers':   'LAC',
    'los angeles lakers':     'LAL',
    'memphis grizzlies':      'MEM',
    'miami heat':             'MIA',
    'milwaukee bucks':        'MIL',
    'minnesota timberwolves': 'MIN',
    'new orleans pelicans':   'NOP',
    'new york knicks':        'NYK',
    'oklahoma city thunder':  'OKC',
    'orlando magic':          'ORL',
    'philadelphia 76ers':     'PHI',
    'phoenix suns':           'PHX',
    'portland trail blazers': 'POR',
    'sacramento kings':       'SAC',
    'san antonio spurs':      'SAS',
    'toronto raptors':        'TOR',
    'utah jazz':              'UTA',
    'washington wizards':     'WAS',
}

def _nba_abbr(name):
    return _NBA_NAME_TO_ABBR.get(name.lower().strip(), name[:3].upper())


# ─── BallDontLie helpers ──────────────────────────────────────────────────────
_bdl_teams_cache = None

def _bdl_get(path, params=None, retries=2):
    for i in range(retries):
        try:
            r = requests.get(f'{BDL_BASE}/{path}', params=params, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                return {}
            time.sleep(1)
    return {}


def _bdl_wnba_get(path, params=None, retries=2):
    """GET from BallDontLie WNBA API (api.balldontlie.io/wnba/v1).
    Uses BDL_API_KEY env var if set."""
    api_key = os.environ.get('BDL_API_KEY', '')
    headers = {'Authorization': api_key} if api_key else {}
    for i in range(retries):
        try:
            r = requests.get(f'{BDL_WNBA_BASE}/{path}', params=params,
                             headers=headers, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                return {}
            time.sleep(1)
    return {}


def _get_bdl_teams():
    global _bdl_teams_cache
    if _bdl_teams_cache:
        return _bdl_teams_cache
    data = _bdl_get('teams', {'per_page': 100})
    teams = data.get('data', [])
    _bdl_teams_cache = {t['full_name']: t['id'] for t in teams}
    _bdl_teams_cache.update({t['name']: t['id'] for t in teams})   # "Hawks" style too
    return _bdl_teams_cache


def _get_bdl_today(date_str=None):
    """Return NBA games scheduled for date_str (YYYY-MM-DD), defaulting to today."""
    if not date_str:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    data = _bdl_get('games', {'start_date': date_str, 'end_date': date_str, 'per_page': 50})
    return data.get('data', [])


def _get_bdl_team_games(team_id, season=None):
    """Fetch up to 30 recent completed games for a team."""
    if season is None:
        season = datetime.utcnow().year - (1 if datetime.utcnow().month < 7 else 0)
    data = _bdl_get('games', {
        'team_ids[]': team_id, 'seasons[]': season,
        'per_page': 100, 'page': 1,
    })
    games = [g for g in data.get('data', [])
             if g.get('home_team_score') and g.get('visitor_team_score')]
    games.sort(key=lambda g: g['date'], reverse=True)
    return games[:30]


def _compute_nba_form(games, team_id):
    """Derive rolling-form feature dict from BallDontLie game list."""
    if not games:
        return _default_nba_form()
    wins, pts_f, pts_a = [], [], []
    last_date = None
    for g in games:
        is_home = g['home_team']['id'] == team_id
        gf = float(g['home_team_score']  if is_home else g['visitor_team_score'])
        ga = float(g['visitor_team_score'] if is_home else g['home_team_score'])
        pts_f.append(gf); pts_a.append(ga)
        wins.append(1.0 if gf > ga else 0.0)
        if last_date is None:
            last_date = pd.to_datetime(g['date'])

    n10 = min(10, len(games))
    w10  = wins[:n10];  pf10 = pts_f[:n10];  pa10 = pts_a[:n10]
    rest = max(1, (pd.Timestamp.now('UTC') - last_date).days) if last_date else 3
    return {
        'win_rate_l10':     float(np.mean(w10)),
        'avg_pts_l10':      float(np.mean(pf10)),
        'avg_conceded_l10': float(np.mean(pa10)),
        'pt_diff_l10':      float(np.mean(pf10)) - float(np.mean(pa10)),
        'ortg_l10':         float(np.mean(pf10)) * 1.04,
        'season_win_pct':   float(np.mean(wins)),
        'rest_days':        float(min(rest, 14)),
        'b2b':              1.0 if rest == 1 else 0.0,
    }


def _default_nba_form():
    return {'win_rate_l10': 0.5, 'avg_pts_l10': 112.0, 'avg_conceded_l10': 112.0,
            'pt_diff_l10': 0.0, 'ortg_l10': 116.0, 'season_win_pct': 0.5,
            'rest_days': 3.0, 'b2b': 0.0}


def _default_wnba_form():
    return {'win_rate_l10': 0.5, 'avg_pts_l10': 82.0, 'avg_conceded_l10': 82.0,
            'pt_diff_l10': 0.0, 'ortg_l10': 100.0, 'drtg_l10': 100.0,
            'season_win_pct': 0.5, 'rest_days': 3.0, 'b2b': 0.0}


def _nba_form_from_cache(team_name):
    """Fall back to CSV form cache (2022-23 season)."""
    try:
        cache = pd.read_csv(NBA_FORM_CACHE, parse_dates=['game_date'])
        tg = cache[cache['team'] == team_name].tail(10)
        if len(tg) < 3:
            tg = cache[cache['team'].str.contains(
                team_name.split()[-1], case=False, na=False)].tail(10)
        if len(tg) < 3:
            return _default_nba_form()
        wins = (tg['win'] > 0.5).astype(float)
        return {
            'win_rate_l10':     float(wins.mean()),
            'avg_pts_l10':      float(tg['gf'].mean()),
            'avg_conceded_l10': float(tg['ga'].mean()),
            'pt_diff_l10':      float(tg['pt_diff'].mean()),
            'ortg_l10':         float(tg['ortg'].mean()),
            'season_win_pct':   float(wins.mean()),
            'rest_days':        3.0,
            'b2b':              0.0,
        }
    except Exception:
        return _default_nba_form()


def _wnba_form_from_cache(abbr):
    """Get WNBA team form from historical cache."""
    try:
        cache = pd.read_csv(WNBA_FORM_CACHE, parse_dates=['gmDate'])
        tg = cache[cache['team'] == abbr].tail(10)
        if len(tg) < 3:
            return _default_wnba_form()
        wins = (tg['win'] > 0.5).astype(float)
        return {
            'win_rate_l10':     float(wins.mean()),
            'avg_pts_l10':      float(tg['gf'].mean()),
            'avg_conceded_l10': float(tg['ga'].mean()),
            'pt_diff_l10':      float(tg['pt_diff'].mean()),
            'ortg_l10':         float(tg['ortg'].mean()),
            'drtg_l10':         float(tg['drtg'].mean()),
            'season_win_pct':   float(wins.mean()),
            'rest_days':        3.0,
            'b2b':              0.0,
        }
    except Exception:
        return _default_wnba_form()


def _nba_h2h(home_name, away_name):
    """Home-team historical win rate vs away team from SQLite."""
    try:
        conn = sqlite3.connect(NBA_DB)
        df = pd.read_sql_query("""
            SELECT SUM(CASE WHEN wl_home='W' THEN 1.0 ELSE 0.0 END) AS hw,
                   COUNT(*) AS total
            FROM game
            WHERE team_name_home=? AND team_name_away=?
        """, conn, params=(home_name, away_name))
        conn.close()
        total = df['total'].iloc[0]
        return float(df['hw'].iloc[0]) / total if total > 0 else 0.5
    except Exception:
        return 0.5


# ─── Feature vector builders ──────────────────────────────────────────────────

def _build_nba_fv(hf, af, h2h):
    return [
        hf['win_rate_l10'], hf['avg_pts_l10'], hf['avg_conceded_l10'],
        hf['pt_diff_l10'],  hf['ortg_l10'],    hf['season_win_pct'],
        hf['rest_days'],    hf['b2b'],
        af['win_rate_l10'], af['avg_pts_l10'], af['avg_conceded_l10'],
        af['pt_diff_l10'],  af['ortg_l10'],    af['season_win_pct'],
        af['rest_days'],    af['b2b'],
        hf['win_rate_l10'] - af['win_rate_l10'],
        hf['pt_diff_l10']  - af['pt_diff_l10'],
        hf['rest_days']    - af['rest_days'],
        h2h,
    ]


def _build_wnba_fv(hf, af, h2h):
    return [
        hf['win_rate_l10'], hf['avg_pts_l10'], hf['avg_conceded_l10'],
        hf['pt_diff_l10'],  hf['ortg_l10'],    hf.get('drtg_l10', hf['ortg_l10']),
        hf['season_win_pct'], hf['rest_days'], hf['b2b'],
        af['win_rate_l10'], af['avg_pts_l10'], af['avg_conceded_l10'],
        af['pt_diff_l10'],  af['ortg_l10'],    af.get('drtg_l10', af['ortg_l10']),
        af['season_win_pct'], af['rest_days'], af['b2b'],
        hf['win_rate_l10'] - af['win_rate_l10'],
        hf['pt_diff_l10']  - af['pt_diff_l10'],
        hf['rest_days']    - af['rest_days'],
        h2h,
    ]


def _run_prediction(fv, res_data, ou_data, home_name, away_name, ou_line):
    res_proba = res_data['model'].predict_proba([fv])[0]
    ou_proba  = ou_data['model'].predict_proba([fv])[0]

    home_win_pct = float(res_proba[1]) * 100
    away_win_pct = float(res_proba[0]) * 100
    over_pct     = float(ou_proba[1]) * 100
    under_pct    = 100 - over_pct

    if home_win_pct >= away_win_pct:
        winner, win_pct = home_name, home_win_pct
    else:
        winner, win_pct = away_name, away_win_pct

    best_bets = [
        (win_pct,  winner,             'result'),
        (max(over_pct, under_pct),
         f"Over {ou_line}" if over_pct > 50 else f"Under {ou_line}", 'ou'),
    ]
    best_bets.sort(key=lambda x: x[0], reverse=True)
    best_bet_label, best_bet_type = best_bets[0][1], best_bets[0][2]

    return {
        'home_win_pct':   round(home_win_pct, 1),
        'away_win_pct':   round(away_win_pct, 1),
        'predicted_winner': winner,
        'win_probability':  round(win_pct, 1),
        'over_pct':    round(over_pct, 1),
        'under_pct':   round(under_pct, 1),
        'predicted_ou': f"Over {ou_line}" if over_pct > 50 else f"Under {ou_line}",
        'ou_probability': round(max(over_pct, under_pct), 1),
        'best_bet':      best_bet_label,
        'best_bet_type': best_bet_type,
    }


# ─── Public prediction API ────────────────────────────────────────────────────

def predict_nba(home_name, away_name, ou_line=None):
    """Return prediction dict for an NBA match.
    ou_line: real bookmaker O/U line for this game; falls back to NBA_OU_LINE."""
    if _NBA_RES is None:
        _load_models()
    try:
        bdl_teams = _get_bdl_teams()
        home_id   = bdl_teams.get(home_name)
        away_id   = bdl_teams.get(away_name)

        home_games = _get_bdl_team_games(home_id) if home_id else []
        hf = _compute_nba_form(home_games, home_id) if home_games else _nba_form_from_cache(home_name)

        away_games = _get_bdl_team_games(away_id) if away_id else []
        af = _compute_nba_form(away_games, away_id) if away_games else _nba_form_from_cache(away_name)

        h2h = _nba_h2h(home_name, away_name)
        fv  = _build_nba_fv(hf, af, h2h)
        effective_line = ou_line if ou_line is not None else NBA_OU_LINE
        return _run_prediction(fv, _NBA_RES, _NBA_OU, home_name, away_name, effective_line)
    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}


def predict_wnba(home_name, away_name, ou_line=None):
    """Return prediction dict for a WNBA match.
    ou_line: real bookmaker O/U line for this game; falls back to WNBA_OU_LINE."""
    if _WNBA_RES is None:
        _load_models()
    try:
        home_abbr = _wnba_abbr(home_name)
        away_abbr = _wnba_abbr(away_name)
        hf  = _wnba_form_from_cache(home_abbr)
        af  = _wnba_form_from_cache(away_abbr)
        h2h = 0.5  # not enough current data for WNBA H2H
        fv  = _build_wnba_fv(hf, af, h2h)
        effective_line = ou_line if ou_line is not None else WNBA_OU_LINE
        return _run_prediction(fv, _WNBA_RES, _WNBA_OU, home_name, away_name, effective_line)
    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}


# ─── Fixture fetching ─────────────────────────────────────────────────────────

def get_nba_fixtures(start_date=None, end_date=None):
    """Return NBA games from start_date through end_date using TheSportsDB (l=4387)."""
    from datetime import date as _date, timedelta as _td, datetime as _dtt
    if not start_date:
        start_date = _date.today().strftime('%Y-%m-%d')
    if not end_date:
        end_date = (_date.today() + _td(days=3)).strftime('%Y-%m-%d')

    start = _dtt.strptime(start_date, '%Y-%m-%d').date()
    end_  = _dtt.strptime(end_date,   '%Y-%m-%d').date()
    result = []

    d = start
    while d <= end_:
        date_str = d.strftime('%Y-%m-%d')
        try:
            r = requests.get(
                f'{TSDB_BASE}/eventsday.php',
                params={'d': date_str, 'l': NBA_TSDB_ID},
                timeout=10,
            )
            for e in (r.json().get('events') or []):
                hs   = e.get('intHomeScore')
                as_  = e.get('intAwayScore')
                home = e.get('strHomeTeam', '')
                away = e.get('strAwayTeam', '')
                ev   = (e.get('strEvent') or '').lower()
                rnd  = (e.get('strRound') or '').lower()
                if 'finals' in ev or 'finals' in rnd:
                    comp = 'NBA Finals'
                elif 'playoff' in ev or 'playoff' in rnd or 'semi' in rnd or 'conference' in rnd:
                    comp = 'NBA Playoffs'
                else:
                    comp = 'NBA'
                result.append({
                    'id':          e.get('idEvent'),
                    'home_team':   home,
                    'away_team':   away,
                    'home_abbr':   _nba_abbr(home),
                    'away_abbr':   _nba_abbr(away),
                    'home_score':  int(hs)  if hs  and str(hs).strip()  not in ('', 'None') else None,
                    'away_score':  int(as_) if as_ and str(as_).strip() not in ('', 'None') else None,
                    'status':      e.get('strStatus', ''),
                    'time':        e.get('strTime', ''),
                    'date':        date_str,
                    'competition': comp,
                    'postseason':  comp != 'NBA',
                })
        except Exception:
            pass
        d += _td(days=1)

    result.sort(key=lambda g: g['date'])
    return result


def get_wnba_fixtures(start_date=None, end_date=None):
    """Return WNBA games using ESPN WNBA scoreboard API (primary).
    Fallback: TheSportsDB l=4328 per-day loop."""
    from datetime import date as _date, timedelta as _td, datetime as _dtt
    if not start_date:
        start_date = _date.today().strftime('%Y-%m-%d')
    if not end_date:
        end_date = (_date.today() + _td(days=3)).strftime('%Y-%m-%d')

    start = _dtt.strptime(start_date, '%Y-%m-%d').date()
    end_  = _dtt.strptime(end_date,   '%Y-%m-%d').date()
    result = []

    # ── Primary: ESPN WNBA scoreboard (UTC times in ISO 8601) ─────────────────
    d = start
    while d <= end_:
        try:
            r = requests.get(
                'https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard',
                params={'dates': d.strftime('%Y%m%d')},
                timeout=10,
            )
            for ev in (r.json().get('events') or []):
                comp0       = ev.get('competitions', [{}])[0]
                competitors = comp0.get('competitors', [])
                home_c = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                away_c = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                if not home_c or not away_c:
                    continue
                ht = home_c.get('team', {})
                at = away_c.get('team', {})
                ev_date   = ev.get('date', '')
                game_date = d.strftime('%Y-%m-%d')           # use query date, not UTC ISO date
                time_utc  = ev_date[11:16] if len(ev_date) >= 16 else ''
                sname     = comp0.get('status', {}).get('type', {}).get('name', '')
                if 'FINAL' in sname:
                    status = 'Final'
                elif 'PROGRESS' in sname:
                    status = 'InProgress'
                else:
                    status = 'Scheduled'
                hs  = home_c.get('score')
                as_ = away_c.get('score')
                # Extract real bookmaker O/U line from inline DraftKings odds
                inline_odds = comp0.get('odds', [])
                ou_line = None
                if inline_odds:
                    try:
                        ou_val = inline_odds[0].get('overUnder')
                        if ou_val is not None:
                            ou_line = float(ou_val)
                    except (TypeError, ValueError):
                        pass
                result.append({
                    'id':         ev.get('id'),
                    'home_team':  ht.get('displayName', ''),
                    'away_team':  at.get('displayName', ''),
                    'home_abbr':  ht.get('abbreviation', ''),
                    'away_abbr':  at.get('abbreviation', ''),
                    'home_score': int(hs)  if hs  is not None and str(hs).strip() not in ('', 'None') else None,
                    'away_score': int(as_) if as_ is not None and str(as_).strip() not in ('', 'None') else None,
                    'status':     status,
                    'time':       time_utc,
                    'date':       game_date,
                    'ou_line':    ou_line,
                    'postseason': False,
                })
        except Exception:
            pass
        d += _td(days=1)

    if result:
        result.sort(key=lambda g: g['date'])
        return result

    # ── Fallback: TheSportsDB WNBA per-day (l=4328) ───────────────────────────
    d = start
    while d <= end_:
        date_str = d.strftime('%Y-%m-%d')
        try:
            r = requests.get(
                f'{TSDB_BASE}/eventsday.php',
                params={'d': date_str, 'l': WNBA_TSDB_ID},
                timeout=10,
            )
            for e in (r.json().get('events') or []):
                hs  = e.get('intHomeScore')
                as_ = e.get('intAwayScore')
                home = e.get('strHomeTeam', '')
                away = e.get('strAwayTeam', '')
                result.append({
                    'id':         e.get('idEvent'),
                    'home_team':  home,
                    'away_team':  away,
                    'home_abbr':  _wnba_abbr(home),
                    'away_abbr':  _wnba_abbr(away),
                    'home_score': int(hs)  if hs  and str(hs).strip()  not in ('', 'None') else None,
                    'away_score': int(as_) if as_ and str(as_).strip() not in ('', 'None') else None,
                    'status':     e.get('strStatus', ''),
                    'time':       e.get('strTime', ''),
                    'date':       date_str,
                    'postseason': False,
                })
        except Exception:
            pass
        d += _td(days=1)
    result.sort(key=lambda g: g['date'])
    return result


# ─── Shared site-query helper ────────────────────────────────────────────────

# Terminal scripts query this endpoint so they display the SAME values as the
# website — identical BDL snapshot, identical probabilities, no divergence.
SITE_URL = os.environ.get('AZPREDICTS_URL', 'https://azpredicts.com').rstrip('/')


def _query_site(home, away, date='', sport=''):
    """
    Fetch the stored prediction for a match from azpredicts.com.
    Returns the prediction_payload dict on success, or None if unavailable.
    """
    import urllib.request, urllib.parse, json, ssl
    params = urllib.parse.urlencode({k: v for k, v in {
        'home': home, 'away': away, 'date': date, 'sport': sport,
    }.items() if v})
    url = f"{SITE_URL}/api/stored-prediction?{params}"
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=8, context=ctx) as r:
            d = json.loads(r.read())
            if d.get('found') and d.get('prediction', {}).get('prediction_payload'):
                return d['prediction']['prediction_payload']
    except Exception:
        pass
    return None


def _print_basketball_pred(home, away, date, pred, source):
    """Pretty-print a basketball prediction dict to stdout."""
    sep = '═' * 64
    sport_tag = pred.get('sport', '').upper() or 'NBA'
    ou_line   = WNBA_OU_LINE if 'WNBA' in sport_tag else NBA_OU_LINE
    print(f"\n{sep}")
    print(f"  {away} @ {home}  ·  {date}  [{sport_tag}]")
    print(f"  Source: {source}")
    print(f"{'─' * 64}")
    print(f"  Home Win  : {pred.get('home_win_pct', '?')}%   ({home})")
    print(f"  Away Win  : {pred.get('away_win_pct', '?')}%   ({away})")
    print(f"{'─' * 64}")
    ou_label = pred.get('predicted_ou') or f"Over {ou_line}"
    over_pct  = pred.get('over_pct', '?')
    under_pct = pred.get('under_pct', '?')
    print(f"  Over  {ou_line} : {over_pct}%")
    print(f"  Under {ou_line} : {under_pct}%")
    print(f"{'─' * 64}")
    bb     = pred.get('best_bet', pred.get('predicted_winner', '?'))
    bb_pct = pred.get('win_probability') if pred.get('best_bet_type') == 'result' else pred.get('ou_probability')
    btype  = pred.get('best_bet_type', '?')
    print(f"  ★ BEST BET  : {bb}  ({bb_pct}%)  [{btype}]")
    print(f"{sep}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    _force_local = '--local' in sys.argv  # bypass site query for debugging

    def _show(g, sport):
        home  = g['home_team']
        away  = g['away_team']
        date  = g.get('date', '')
        print(f"  {away} @ {home}  [{g.get('status', '')}]")

        # Step 1: try site (authoritative — same data as website)
        if not _force_local:
            payload = _query_site(home, away, date=date, sport=sport)
            if payload:
                _print_basketball_pred(home, away, date, payload,
                                       source=SITE_URL)
                return

        # Step 2: local computation fallback (may differ from website if BDL data
        #         has changed since the website last refreshed — expected)
        print(f"  [local computation — may differ from website by up to 30 min]")
        pred = predict_nba(home, away) if sport == 'nba' else predict_wnba(home, away)
        if 'error' not in pred:
            pred['sport'] = sport.upper()
            _print_basketball_pred(home, away, date, pred,
                                   source='local BDL fetch (NOT from website cache)')
        else:
            print(f"  Prediction error: {pred['error']}")

    print("═" * 64)
    print("  NBA — upcoming games")
    print("═" * 64)
    games = get_nba_fixtures()
    if games:
        for g in games[:5]:
            _show(g, 'nba')
    else:
        print("  No NBA games in window. Test: Lakers vs Celtics")
        g = {'home_team': 'Boston Celtics', 'away_team': 'Oklahoma City Thunder',
             'date': '', 'status': 'Test'}
        _show(g, 'nba')

    print("═" * 64)
    print("  WNBA — upcoming games")
    print("═" * 64)
    wgames = get_wnba_fixtures()
    if wgames:
        for g in wgames[:5]:
            _show(g, 'wnba')
    else:
        print("  No WNBA games in window. Test: Seattle Storm vs Las Vegas Aces")
        g = {'home_team': 'Seattle Storm', 'away_team': 'Las Vegas Aces',
             'date': '', 'status': 'Test'}
        _show(g, 'wnba')

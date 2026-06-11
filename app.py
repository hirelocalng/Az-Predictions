"""
app.py  —  Flask API for the Football Prediction Website.

Endpoints:
  GET /api/club/tips          daily club-football prediction cards
  GET /api/worldcup/fixtures  upcoming WC 2026 fixtures with predictions
  GET /api/worldcup/countdown seconds to next WC 2026 match

Club tips: uses result_model / goals_model / corners_model (train.py).
WC tips  : predict_match() is defined directly in this file (copied from
           worldcup_predict.py) so the exact same code runs for both the
           terminal tool and the API — no imports, no divergence.

Run:
    python app.py
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from contextlib import contextmanager
import os, sys, pickle, warnings, time, logging, math, difflib, unicodedata, json, threading, secrets
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

_log = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
CORS(app)

# ── Load club models ──────────────────────────────────────────────────────────

def _load(p):
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
club_result  = _load(os.path.join(_BASE_DIR, 'result_model.pkl'))
club_goals   = _load(os.path.join(_BASE_DIR, 'goals_model.pkl'))
club_corners = _load(os.path.join(_BASE_DIR, 'corners_model.pkl'))

LEAGUE_MAP  = club_result['league_map']  if club_result  else {}
RES_ENCODER = club_result['result_encoder'] if club_result else None

# ── Prediction history (PostgreSQL primary, JSON fallback) ────────────────────

_HISTORY_PATH = os.path.join(_BASE_DIR, 'prediction_history.json')
_HISTORY_LOCK   = threading.Lock()
_DB_URL         = os.environ.get('DATABASE_URL', '')
_KORAPAY_SECRET = os.environ.get('KORAPAY_SECRET_KEY', '')
_KORAPAY_PUBLIC = os.environ.get('KORAPAY_PUBLIC_KEY', '')

logging.basicConfig(level=logging.INFO)
_log.info('KORAPAY_PUBLIC_KEY set: %s', bool(_KORAPAY_PUBLIC))
_log.info('KORAPAY_SECRET_KEY set: %s', bool(_KORAPAY_SECRET))


def _build_db_url():
    """Return a psycopg-ready URL: postgresql:// scheme + sslmode=require for Railway."""
    url = _DB_URL
    if not url:
        return url
    # Railway sometimes provides postgres:// — psycopg requires postgresql://
    if url.startswith('postgres://'):
        url = 'postgresql://' + url[len('postgres://'):]
    # Railway PostgreSQL requires SSL; append only if not already specified
    if 'sslmode' not in url:
        url += ('&' if '?' in url else '?') + 'sslmode=require'
    return url

_DB_CONN_URL = _build_db_url()


@contextmanager
def _db():
    """Yield (conn, cur); both None when DATABASE_URL is unset."""
    if not _DB_CONN_URL:
        yield None, None
        return
    conn = cur = None
    try:
        import psycopg  # psycopg3 — bundled binary, no system libpq needed
        conn = psycopg.connect(_DB_CONN_URL, autocommit=False)
        cur  = conn.cursor()
        yield conn, cur
        conn.commit()
    except Exception:
        try:
            if conn: conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            if cur:  cur.close()
        except Exception:
            pass
        try:
            if conn: conn.close()
        except Exception:
            pass


def _init_db():
    """Create all tables via CREATE TABLE IF NOT EXISTS — safe to run on every startup."""
    if not _DB_CONN_URL:
        print('DB init skipped — DATABASE_URL not set (using JSON fallback)', flush=True)
        return
    print(f'DB init: connecting to PostgreSQL…', flush=True)
    try:
        with _db() as (conn, cur):
            if conn is None:
                print('DB init: connection returned None unexpectedly', flush=True)
                return
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id                SERIAL PRIMARY KEY,
                    match_id          VARCHAR(255) UNIQUE NOT NULL,
                    home_team         VARCHAR(100),
                    away_team         VARCHAR(100),
                    match_date        VARCHAR(20),
                    match_time        VARCHAR(10),
                    competition       VARCHAR(200),
                    predicted_winner  VARCHAR(200),
                    predicted_goals   VARCHAR(20),
                    predicted_btts    VARCHAR(10),
                    predicted_corners VARCHAR(20),
                    actual_home_score INTEGER,
                    actual_away_score INTEGER,
                    result_status     VARCHAR(10) DEFAULT 'PENDING',
                    kickoff_utc       VARCHAR(50),
                    created_at        TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            SERIAL PRIMARY KEY,
                    name          VARCHAR(100) NOT NULL,
                    email         VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    is_premium    BOOLEAN DEFAULT FALSE,
                    premium_until TIMESTAMPTZ,
                    created_at    TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id         SERIAL PRIMARY KEY,
                    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token      VARCHAR(64) UNIQUE NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            for _col in [
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS match_status VARCHAR(20) DEFAULT 'SCHEDULED'",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS live_home_score INTEGER",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS live_away_score INTEGER",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS live_minute VARCHAR(10)",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_corners INTEGER",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS sport VARCHAR(20) DEFAULT 'football'",
            ]:
                cur.execute(_col)
        print('DB init: all tables ready (predictions, users, sessions) ✓', flush=True)
        _log.info('DB: all tables ready')
    except Exception as e:
        print(f'DB init FAILED: {e}', flush=True)
        _log.error('DB init: %s', e)


# ── JSON helpers (used when DATABASE_URL is unset) ────────────────────────────

def _read_raw_history():
    try:
        with open(_HISTORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'predictions': []}
    except Exception:
        return {'predictions': []}


def _write_raw_history(data):
    try:
        with open(_HISTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        _log.error('Failed to write history: %s', e)


def _match_key(home, away, date):
    return f"{date}/{home.lower().strip()}/{away.lower().strip()}"


def _save_prediction(tip):
    home       = (tip.get('home_team') or tip.get('home') or '').strip()
    away       = (tip.get('away_team') or tip.get('away') or '').strip()
    kickoff    = (tip.get('utc_kickoff') or '').strip()
    match_date = kickoff[:10] if kickoff else (tip.get('date') or '')
    if not home or not away or not match_date:
        return
    best_bet   = tip.get('best_bet') or {}
    over_goals = tip.get('over_goals', 0.5)
    btts       = tip.get('btts', 0.48)
    corners    = tip.get('over_corners', 0.52)
    mid        = _match_key(home, away, match_date)
    mtime      = tip.get('time', kickoff[11:16] if len(kickoff) > 15 else '')
    comp       = (tip.get('league') or tip.get('competition') or 'Football')
    pw         = best_bet.get('label', '')
    pg_str     = 'Over 2.5' if over_goals >= 0.5 else 'Under 2.5'
    btts_str   = 'Yes' if btts >= 0.5 else 'No'
    cor_str    = 'Over 9.5' if corners >= 0.5 else 'Under 9.5'

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute("""
                    INSERT INTO predictions
                        (match_id, home_team, away_team, match_date, match_time,
                         competition, predicted_winner, predicted_goals,
                         predicted_btts, predicted_corners, result_status, kickoff_utc)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'PENDING',%s)
                    ON CONFLICT (match_id) DO NOTHING
                """, (mid, home, away, match_date, mtime, comp, pw,
                      pg_str, btts_str, cor_str, kickoff))
                return
    except Exception as e:
        _log.warning('DB save failed, falling back to JSON: %s', e)

    # ── JSON fallback ─────────────────────────────────────────────────────────
    entry = {
        'match_id': mid, 'home_team': home, 'away_team': away,
        'match_date': match_date, 'match_time': mtime, 'competition': comp,
        'predicted_winner': pw, 'predicted_goals': pg_str,
        'predicted_btts': btts_str, 'predicted_corners': cor_str,
        'actual_home_score': None, 'actual_away_score': None,
        'result_status': 'PENDING', 'kickoff_utc': kickoff,
        'saved_at': datetime.now(timezone.utc).isoformat(),
    }
    with _HISTORY_LOCK:
        data = _read_raw_history()
        if mid not in {p['match_id'] for p in data['predictions']}:
            data['predictions'].append(entry)
            _write_raw_history(data)


def _save_basketball_prediction(game, pred, sport):
    """Save an NBA or WNBA prediction to history."""
    home  = (game.get('home_team') or '').strip()
    away  = (game.get('away_team') or '').strip()
    date  = (game.get('date') or '')
    if not home or not away or not date:
        return
    mid     = _match_key(home, away, date)
    comp    = game.get('competition', sport.upper())
    pw      = pred.get('best_bet', pred.get('predicted_winner', ''))
    pg      = pred.get('predicted_ou', '')
    time_   = (game.get('time') or '')
    kickoff = f"{date}T{time_}:00Z" if time_ else f"{date}T23:00:00Z"

    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute("""
                    INSERT INTO predictions
                        (match_id, home_team, away_team, match_date, match_time,
                         competition, predicted_winner, predicted_goals,
                         result_status, kickoff_utc, sport)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'PENDING',%s,%s)
                    ON CONFLICT (match_id) DO NOTHING
                """, (mid, home, away, date, time_, comp, pw, pg, kickoff, sport))
                return
    except Exception as e:
        _log.warning('Basketball DB save failed: %s', e)

    entry = {
        'match_id': mid, 'home_team': home, 'away_team': away,
        'match_date': date, 'match_time': time_, 'competition': comp,
        'predicted_winner': pw, 'predicted_goals': pg,
        'predicted_btts': None, 'predicted_corners': None,
        'actual_home_score': None, 'actual_away_score': None,
        'result_status': 'PENDING', 'kickoff_utc': kickoff,
        'sport': sport,
        'saved_at': datetime.now(timezone.utc).isoformat(),
    }
    with _HISTORY_LOCK:
        data = _read_raw_history()
        if mid not in {p['match_id'] for p in data['predictions']}:
            data['predictions'].append(entry)
            _write_raw_history(data)


def _get_history():
    """Return {predictions: [...]} from DB or JSON fallback."""
    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute("""
                    SELECT match_id, home_team, away_team, match_date, match_time,
                           competition, predicted_winner, predicted_goals,
                           predicted_btts, predicted_corners,
                           actual_home_score, actual_away_score, actual_corners,
                           result_status, kickoff_utc, created_at,
                           COALESCE(sport, 'football') AS sport
                    FROM predictions
                    ORDER BY match_date DESC, created_at DESC
                """)
                cols = [d[0] for d in cur.description]
                rows = []
                for r in cur.fetchall():
                    row = dict(zip(cols, r))
                    if row.get('created_at'):
                        row['saved_at'] = row['created_at'].isoformat()
                    row.pop('created_at', None)
                    rows.append(row)
                return {'predictions': rows}
    except Exception as e:
        _log.warning('DB get_history failed: %s', e)
    with _HISTORY_LOCK:
        return _read_raw_history()


def _completed_keys():
    """Return set of match_ids that are WON or LOST."""
    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute(
                    "SELECT match_id FROM predictions "
                    "WHERE result_status IN ('WON','LOST')"
                )
                return {r[0] for r in cur.fetchall()}
    except Exception as e:
        _log.warning('DB completed_keys failed: %s', e)
    data = _get_history()
    return {p['match_id'] for p in data.get('predictions', [])
            if p['result_status'] in ('WON', 'LOST')}


def _get_current_user():
    """Return user dict from Bearer token, or None if invalid/expired."""
    auth_hdr = request.headers.get('Authorization', '')
    if not auth_hdr.startswith('Bearer '):
        return None
    token = auth_hdr[7:].strip()
    if not token:
        return None
    try:
        with _db() as (conn, cur):
            if conn is None:
                return None
            cur.execute("""
                SELECT u.id, u.name, u.email, u.is_premium, u.premium_until
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = %s AND s.expires_at > NOW()
            """, (token,))
            row = cur.fetchone()
            if not row:
                return None
            now_utc = datetime.now(timezone.utc)
            is_prem  = bool(row[3]) and (row[4] is None or row[4] > now_utc)
            return {
                'id': row[0], 'name': row[1], 'email': row[2],
                'is_premium': is_prem,
                'premium_until': row[4].isoformat() if row[4] else None,
            }
    except Exception as e:
        _log.warning('_get_current_user: %s', e)
        return None


def _migrate_json_to_db():
    """
    On startup, push any predictions from prediction_history.json into PostgreSQL.
    Uses INSERT ... ON CONFLICT DO UPDATE so that if a JSON record is already WON/LOST
    it overwrites a stale PENDING row in the DB (handles the case where DB was broken
    when the result arrived and only JSON got updated).
    """
    if not _DB_CONN_URL:
        return
    try:
        with _HISTORY_LOCK:
            data = _read_raw_history()
        preds = data.get('predictions', [])
        if not preds:
            print('DB migration: prediction_history.json is empty — nothing to sync', flush=True)
            return
        synced = 0
        with _db() as (conn, cur):
            if conn is None:
                return
            for p in preds:
                mid = p.get('match_id')
                if not mid:
                    continue
                try:
                    cur.execute("""
                        INSERT INTO predictions
                            (match_id, home_team, away_team, match_date, match_time,
                             competition, predicted_winner, predicted_goals,
                             predicted_btts, predicted_corners,
                             actual_home_score, actual_away_score,
                             result_status, kickoff_utc)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (match_id) DO UPDATE
                            SET result_status     = EXCLUDED.result_status,
                                actual_home_score = EXCLUDED.actual_home_score,
                                actual_away_score = EXCLUDED.actual_away_score
                            WHERE predictions.result_status IN ('PENDING','LIVE')
                              AND EXCLUDED.result_status IN ('WON','LOST')
                    """, (
                        mid,
                        p.get('home_team'), p.get('away_team'),
                        p.get('match_date'), p.get('match_time'),
                        p.get('competition', 'Football'),
                        p.get('predicted_winner'), p.get('predicted_goals'),
                        p.get('predicted_btts'), p.get('predicted_corners'),
                        p.get('actual_home_score'), p.get('actual_away_score'),
                        p.get('result_status', 'PENDING'), p.get('kickoff_utc'),
                    ))
                    synced += 1
                except Exception as exc:
                    _log.warning('migrate: skip %s — %s', mid, exc)
        print(f'DB migration: {synced}/{len(preds)} JSON predictions synced to DB ✓', flush=True)
    except Exception as exc:
        print(f'DB migration error: {exc}', flush=True)


# ── International prediction ──────────────────────────────────────────────────
# Single source of truth: worldcup_predict.py is authoritative for both the
# terminal tool and the API.  Importing it here means they always agree.
sys.path.insert(0, _BASE_DIR)
from worldcup_predict import predict_match  # noqa: E402

_WC_DATA_PATH         = os.path.join(_BASE_DIR, 'data', 'results.csv')
_WC_RESULT_MODEL_PATH = os.path.join(_BASE_DIR, 'worldcup_result_model.pkl')
_WC_GOALS_MODEL_PATH  = os.path.join(_BASE_DIR, 'worldcup_goals_model.pkl')
_WC_FORM_WINDOW       = 10

_TOURNAMENT_IMPORTANCE = {
    'FIFA World Cup':                            1.00,
    'Confederations Cup':                        0.92,
    'UEFA Euro':                                 0.90,
    'Copa America':                              0.88,
    'African Cup of Nations':                    0.85,
    'FIFA World Cup qualification':              0.85,
    'Gold Cup':                                  0.80,
    'AFC Asian Cup':                             0.82,
    'CONCACAF Championship':                     0.78,
    'Olympic Games':                             0.75,
    'UEFA Euro qualification':                   0.75,
    'British Home Championship':                 0.72,
    'UEFA Nations League':                       0.72,
    'African Cup of Nations qualification':      0.70,
    'AFC Asian Cup qualification':               0.68,
    'CONCACAF Nations League':                   0.68,
    'Gold Cup qualification':                    0.65,
    'Copa America qualification':                0.65,
    'Oceania Nations Cup':                       0.65,
    'CONCACAF Championship qualification':       0.60,
    'CONCACAF Nations League qualification':     0.55,
    'Oceania Nations Cup qualification':         0.55,
    'FIFA Series':                               0.40,
    'Friendly':                                  0.30,
}

_TEAM_ALIASES = {
    'usa':                          'United States',
    'us':                           'United States',
    'america':                      'United States',
    'uk':                           'England',
    'great britain':                'England',
    'south korea':                  'South Korea',
    'korea':                        'South Korea',
    'korea republic':               'South Korea',
    'republic of korea':            'South Korea',
    'dpr korea':                    'North Korea',
    'north korea':                  'North Korea',
    'iran':                         'Iran',
    'russia':                       'Russia',
    'czechia':                      'Czech Republic',
    'czech':                        'Czech Republic',
    'türkiye':                      'Turkey',
    'turkiye':                      'Turkey',
    'ivory coast':                  'Ivory Coast',
    "cote d'ivoire":                'Ivory Coast',
    "côte d'ivoire":                'Ivory Coast',
    'bosnia':                       'Bosnia and Herzegovina',
    'bosnia & herzegovina':         'Bosnia and Herzegovina',
    'bosnia-herzegovina':           'Bosnia and Herzegovina',
    'dr congo':                     'DR Congo',
    'democratic republic of congo': 'DR Congo',
    'congo dr':                     'DR Congo',
    'drc':                          'DR Congo',
    'republic of ireland':          'Republic of Ireland',
    'northern ireland':             'Northern Ireland',
    'cape verde':                   'Cape Verde',
    'cabo verde':                   'Cape Verde',
    'curacao':                      'Curaçao',
    'trinidad & tobago':            'Trinidad and Tobago',
    'trinidad':                     'Trinidad and Tobago',
    'north macedonia':              'North Macedonia',
    'uae':                          'United Arab Emirates',
    'saudi arabia':                 'Saudi Arabia',
    'new zealand':                  'New Zealand',
}


def _wc_norm(text):
    return unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii')


def _wc_get_importance(tournament):
    t_norm = _wc_norm(tournament)
    for key, val in _TOURNAMENT_IMPORTANCE.items():
        if t_norm == _wc_norm(key):
            return val
    t_lower = t_norm.lower()
    if 'world cup' in t_lower:
        return 0.80 if 'qualif' not in t_lower else 0.75
    if 'qualif' in t_lower:
        return 0.60
    if 'friendly' in t_lower:
        return 0.30
    if any(x in t_lower for x in ['cup', 'championship', 'nations', 'league']):
        return 0.60
    return 0.45


def _wc_load_data():
    df = pd.read_csv(_WC_DATA_PATH, encoding='latin-1')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def _wc_load_models():
    with open(_WC_RESULT_MODEL_PATH, 'rb') as f:
        rd = pickle.load(f)
    with open(_WC_GOALS_MODEL_PATH, 'rb') as f:
        gd = pickle.load(f)
    return rd['model'], rd['features'], gd['model'], gd['features']


def _wc_resolve_team(name, all_teams):
    alias = _TEAM_ALIASES.get(name.lower().strip())
    if alias:
        return alias
    name_lower = name.lower().strip()
    for t in all_teams:
        if t.lower() == name_lower:
            return t
    matches = difflib.get_close_matches(name, all_teams, n=5, cutoff=0.5)
    if matches:
        raise ValueError(
            f"Team '{name}' not found. Did you mean: {', '.join(matches)}"
        )
    raise ValueError(f"Team '{name}' not found in dataset.")


def _wc_get_team_form(df, team, n=_WC_FORM_WINDOW):
    mask = (df['home_team'] == team) | (df['away_team'] == team)
    matches = df[mask].sort_values('date').tail(n)
    if len(matches) == 0:
        return {k: 0.0 for k in [
            'win_rate', 'draw_rate', 'loss_rate',
            'goals_scored', 'goals_conceded', 'form_pts', 'form_count', 'goal_diff'
        ]}
    wins = draws = losses = gf = ga = 0
    for _, row in matches.iterrows():
        if row['home_team'] == team:
            g, gc = row['home_score'], row['away_score']
        else:
            g, gc = row['away_score'], row['home_score']
        gf += g; ga += gc
        if g > gc:       wins += 1
        elif g == gc:    draws += 1
        else:            losses += 1
    n_m = len(matches)
    return {
        'win_rate':       wins / n_m,
        'draw_rate':      draws / n_m,
        'loss_rate':      losses / n_m,
        'goals_scored':   gf / n_m,
        'goals_conceded': ga / n_m,
        'form_pts':       (wins * 3 + draws) / n_m,
        'form_count':     float(n_m),
        'goal_diff':      (gf - ga) / n_m,
    }


def _wc_get_major_form(df, team, n=10, importance_threshold=0.70):
    mask = (
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['tournament'].apply(_wc_get_importance) >= importance_threshold)
    )
    matches = df[mask].sort_values('date').tail(n)
    if len(matches) == 0:
        return {'major_win_rate': 0.0, 'major_form_pts': 0.0, 'major_count': 0.0}
    wins = pts = 0
    for _, row in matches.iterrows():
        if row['home_team'] == team:
            gf, ga = row['home_score'], row['away_score']
        else:
            gf, ga = row['away_score'], row['home_score']
        if gf > ga:
            wins += 1; pts += 3
        elif gf == ga:
            pts += 1
    n_m = len(matches)
    return {
        'major_win_rate': wins / n_m,
        'major_form_pts': pts / n_m,
        'major_count':    float(n_m),
    }


def _wc_get_h2h(df, team_a, team_b, n=20):
    mask = (
        ((df['home_team'] == team_a) & (df['away_team'] == team_b)) |
        ((df['home_team'] == team_b) & (df['away_team'] == team_a))
    )
    h2h = df[mask].sort_values('date').tail(n)
    if len(h2h) == 0:
        return {'count': 0, 'home_win_rate': 1/3}
    wins_a = 0
    for _, row in h2h.iterrows():
        if row['home_team'] == team_a:
            if row['home_score'] > row['away_score']: wins_a += 1
        else:
            if row['away_score'] > row['home_score']: wins_a += 1
    return {
        'count':         float(len(h2h)),
        'home_win_rate': wins_a / len(h2h),
    }


def _wc_build_feature_vector(home_form, away_form, h2h, features,
                              is_neutral=True, tournament_importance=1.0):
    row = {
        'home_win_rate':       home_form['win_rate'],
        'home_draw_rate':      home_form['draw_rate'],
        'home_loss_rate':      home_form['loss_rate'],
        'home_goals_scored':   home_form['goals_scored'],
        'home_goals_conceded': home_form['goals_conceded'],
        'home_form_pts':       home_form['form_pts'],
        'home_form_count':     home_form['form_count'],
        'home_goal_diff':      home_form['goal_diff'],
        'away_win_rate':       away_form['win_rate'],
        'away_draw_rate':      away_form['draw_rate'],
        'away_loss_rate':      away_form['loss_rate'],
        'away_goals_scored':   away_form['goals_scored'],
        'away_goals_conceded': away_form['goals_conceded'],
        'away_form_pts':       away_form['form_pts'],
        'away_form_count':     away_form['form_count'],
        'away_goal_diff':      away_form['goal_diff'],
        'is_neutral':              int(is_neutral),
        'tournament_importance':   tournament_importance,
        'h2h_count':           h2h['count'],
        'h2h_home_win_rate':   h2h['home_win_rate'],
        'win_rate_diff':       home_form['win_rate']      - away_form['win_rate'],
        'goals_scored_diff':   home_form['goals_scored']  - away_form['goals_scored'],
        'goals_conceded_diff': home_form['goals_conceded']- away_form['goals_conceded'],
        'form_pts_diff':       home_form['form_pts']      - away_form['form_pts'],
        'home_major_win_rate':  home_form.get('major_win_rate', 0.0),
        'home_major_form_pts':  home_form.get('major_form_pts', 0.0),
        'home_major_count':     home_form.get('major_count', 0.0),
        'away_major_win_rate':  away_form.get('major_win_rate', 0.0),
        'away_major_form_pts':  away_form.get('major_form_pts', 0.0),
        'away_major_count':     away_form.get('major_count', 0.0),
    }
    return np.array([[row[f] for f in features]])


def _api_btts_prob(home_gs: float, away_gs: float) -> float:
    p_h = 1.0 - math.exp(-max(0.05, home_gs))
    p_a = 1.0 - math.exp(-max(0.05, away_gs))
    return round(p_h * p_a, 4)


def _api_corners_prob(home_gs: float, away_gs: float,
                      home_gc: float, away_gc: float) -> float:
    avg = 1.2
    exp_c = (9.2
             + (home_gs - avg) * 1.1 + (away_gs - avg) * 1.1
             + (home_gc - avg) * 0.35 + (away_gc - avg) * 0.35)
    z = (exp_c - 9.5) / (2.2 * math.sqrt(2))
    p = round(0.5 * (1.0 + math.erf(z)), 4)
    return max(0.20, min(0.82, p))




# ── Fetch fixtures helpers ────────────────────────────────────────────────────

try:
    from fetch_fixtures import get_daily_tips, get_club_tips, get_intl_tips
    print('fetch_fixtures imported OK', flush=True)
except Exception as _ff_err:
    print(f'WARNING: fetch_fixtures import failed: {_ff_err}', flush=True)
    def get_daily_tips(*a, **kw): return []
    def get_club_tips(*a, **kw):  return []
    def get_intl_tips(*a, **kw):  return []

# ── Club feature builder ──────────────────────────────────────────────────────
# BASE_FEATURES order must match train.py exactly.

def _club_feat(h, a, elo_diff, form5_h, form5_a, oh, od, oa, league_enc):
    """Returns (base_X [26 features], corner_X [28 features])."""
    h_pts      = 3 * h['win'] + h['draw']
    a_pts      = 3 * a['win'] + a['draw']
    elo_prob_h = 1 / (1 + 10 ** (-elo_diff / 400))
    ih, id_, ia = 1/oh, 1/od, 1/oa
    rs = ih + id_ + ia
    base = [
        h['gf'], h['ga'], h['gf'] - h['ga'],
        h.get('sot', 4.5),
        h['win'], h['draw'], h_pts, h.get('hwn', min(h['win'] + 0.12, 1.0)),
        a['gf'], a['ga'], a['gf'] - a['ga'],
        a.get('sot', 3.8),
        a['win'], a['draw'], a_pts, a.get('awn', max(a['win'] - 0.12, 0.0)),
        elo_diff, elo_prob_h,
        form5_h, form5_a, form5_h - form5_a,
        ih / rs, id_ / rs, ia / rs, rs - 1.0,
        float(league_enc),
    ]
    corner_ext = base + [h.get('corners', 5.2), a.get('corners', 4.8)]
    return np.array([base]), np.array([corner_ext])


def _club_predict(h, a, elo_diff, form5_h, form5_a, oh, od, oa, league_code):
    enc    = LEAGUE_MAP.get(league_code, 10)
    X, Xc  = _club_feat(h, a, elo_diff, form5_h, form5_a, oh, od, oa, enc)

    # Result  classes_=['A','D','H']  → idx [0,1,2]
    rp  = club_result['model'].predict_proba(X)[0]   if club_result  else [0.3, 0.25, 0.45]
    gp  = club_goals['model'].predict_proba(X)[0]    if club_goals   else [0.45, 0.55]
    cp  = club_corners['model'].predict_proba(Xc)[0] if club_corners else [0.50, 0.50]

    # result encoder order: A→0, D→1, H→2
    p_home  = float(rp[2])
    p_draw  = float(rp[1])
    p_away  = float(rp[0])
    p_over_goals   = float(gp[1])
    p_over_corners = float(cp[1])
    return p_home, p_draw, p_away, p_over_goals, p_over_corners


# ── International prediction ──────────────────────────────────────────────────

def _wc_predict(home, away, is_neutral=True):
    """
    Single international prediction function — calls predict_match() from
    worldcup_predict.py, which runs the exact same feature engineering, model
    inputs, and probability outputs as the terminal tool (worldcup_predict.py).
    Returns None if the team cannot be resolved; caller skips the fixture.
    """
    try:
        return predict_match(home, away, is_neutral=is_neutral)
    except Exception as exc:
        _log.warning('predict_match failed %s vs %s — %s', home, away, exc)
        return None


# ── TheSportsDB result lookup ─────────────────────────────────────────────────

_TSDB_URL = 'https://www.thesportsdb.com/api/v1/json/3/eventsday.php'


def _team_similar(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.72


_TSDB_NOT_DONE = frozenset({
    'scheduled', 'not started', 'in progress', 'halftime', 'half time',
    'live', 'postponed', 'cancelled', 'suspended', 'abandoned',
})


_TSDB_STATS_URL = 'https://www.thesportsdb.com/api/v1/json/3/lookupevenstats.php'


def _fetch_tsdb_corners(event_id):
    """Return total corners for a TheSportsDB event ID, or None."""
    import requests as _req
    try:
        r = _req.get(_TSDB_STATS_URL, params={'id': event_id}, timeout=8)
        if r.status_code != 200:
            return None
        stats = r.json().get('eventstats') or []
        for s in stats:
            if 'corner' in (s.get('strStat') or '').lower():
                try:
                    return int(s.get('intHome', 0)) + int(s.get('intAway', 0))
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return None


def _fetch_tsdb_result(home_team, away_team, match_date):
    """Return (home_score, away_score, corners) from TheSportsDB or None if not found."""
    import requests as _req
    try:
        r = _req.get(_TSDB_URL, params={'d': match_date, 's': 'Soccer'}, timeout=10)
        if r.status_code != 200:
            return None
        events = r.json().get('events') or []
        for ev in events:
            hs_raw = ev.get('intHomeScore')
            as_raw = ev.get('intAwayScore')
            if hs_raw is None or as_raw is None:
                continue
            try:
                hs, as_ = int(float(hs_raw)), int(float(as_raw))
            except (ValueError, TypeError):
                continue
            status = (ev.get('strStatus') or '').lower().strip()
            # Accept any event with real scores unless explicitly not-final.
            # TheSportsDB uses 'Match Finished', 'FT', 'AET', 'AP', or empty
            # for historical matches — never rely on specific status keywords.
            if status and status in _TSDB_NOT_DONE:
                continue
            if not status and 'live' in (ev.get('strProgress') or '').lower():
                continue
            eh = ev.get('strHomeTeam', '')
            ea = ev.get('strAwayTeam', '')
            matched = False
            swapped = False
            if _team_similar(eh, home_team) and _team_similar(ea, away_team):
                matched = True
            elif _team_similar(eh, away_team) and _team_similar(ea, home_team):
                matched = True
                swapped = True
            if matched:
                corners = _fetch_tsdb_corners(ev.get('idEvent'))
                if swapped:
                    return as_, hs, corners
                return hs, as_, corners
    except Exception as exc:
        _log.warning('TSDB lookup failed: %s', exc)
    return None


_FD_RESULT_URL = 'https://api.football-data.org/v4/matches'
_FD_RESULT_KEY = '118333be3eb84d0ca4e10740f6d62255'


def _fetch_fd_result(home_team, away_team, match_date):
    """Return (home_score, away_score, corners=None) from football-data.org or None."""
    import requests as _req
    try:
        r = _req.get(
            _FD_RESULT_URL,
            headers={'X-Auth-Token': _FD_RESULT_KEY},
            params={'dateFrom': match_date, 'dateTo': match_date},
            timeout=12,
        )
        if r.status_code != 200:
            return None
        for m in r.json().get('matches', []):
            if m.get('status') != 'FINISHED':
                continue
            ft = (m.get('score') or {}).get('fullTime') or {}
            hs, as_ = ft.get('home'), ft.get('away')
            if hs is None or as_ is None:
                continue
            hn = (m.get('homeTeam') or {}).get('name', '')
            an = (m.get('awayTeam') or {}).get('name', '')
            if _team_similar(hn, home_team) and _team_similar(an, away_team):
                return int(hs), int(as_), None
            if _team_similar(hn, away_team) and _team_similar(an, home_team):
                return int(as_), int(hs), None
    except Exception as exc:
        _log.warning('FD result lookup failed: %s', exc)
    return None


def _fetch_espn_result(home_team, away_team):
    """Return (home_score, away_score, corners) from ESPN if match is fully finished, else None."""
    espn = _fetch_espn_events()
    ev   = _espn_match(espn, home_team, away_team)
    if ev and ev.get('espn_status') in _ESPN_DONE:
        return ev['home_score'], ev['away_score'], ev.get('corners')
    return None


def _get_result(home_team, away_team, match_date):
    """Try every result source; return (home_score, away_score, corners) or None."""
    r = _fetch_tsdb_result(home_team, away_team, match_date)
    if r:
        return r
    r = _fetch_fd_result(home_team, away_team, match_date)
    if r:
        return r
    r = _fetch_espn_result(home_team, away_team)
    if r:
        return r
    r = _fetch_espn_result_dated(home_team, away_team, match_date)
    if r:
        return r
    return None


def _sub_results(pred):
    """
    Compute per-market WON/LOST for all four prediction markets.
    Returns a dict or None if actual scores are not yet available.
    """
    hs  = pred.get('actual_home_score')
    as_ = pred.get('actual_away_score')
    if hs is None or as_ is None:
        return None

    home_l  = (pred.get('home_team') or '').lower()
    away_l  = (pred.get('away_team') or '').lower()
    total   = hs + as_
    sub     = {}

    # Result
    pw = (pred.get('predicted_winner') or '').lower()
    if pw:
        if hs > as_:
            ok = home_l in pw
        elif as_ > hs:
            ok = away_l in pw
        else:
            ok = 'draw' in pw
        sub['result'] = 'WON' if ok else 'LOST'

    # Goals / O/U Over/Under
    pg = (pred.get('predicted_goals') or '').lower()
    if pg:
        predicted_over = 'over' in pg
        import re as _re
        _nums = _re.findall(r'\d+\.?\d*', pg)
        thresh = float(_nums[-1]) if _nums else 2.5
        actual_over    = total > thresh
        sub['goals'] = 'WON' if predicted_over == actual_over else 'LOST'

    # BTTS
    pbtts = (pred.get('predicted_btts') or '').lower()
    if pbtts in ('yes', 'no'):
        predicted_yes = pbtts == 'yes'
        actual_btts   = hs > 0 and as_ > 0
        sub['btts'] = 'WON' if predicted_yes == actual_btts else 'LOST'

    # Corners
    pc = (pred.get('predicted_corners') or '').lower()
    if pc:
        actual_c = pred.get('actual_corners')
        if actual_c is not None:
            predicted_over = 'over' in pc
            try:
                thresh = float(pc.split()[1])
            except (IndexError, ValueError):
                thresh = 9.5
            sub['corners'] = 'WON' if predicted_over == (actual_c > thresh) else 'LOST'
        else:
            sub['corners'] = None   # prediction exists but no data available

    return sub


def _resolve_result(pw, home, away, hs, as_):
    """Return 'WON' or 'LOST' for any bet type given actual score."""
    if not pw:
        return 'LOST'
    pw_l  = pw.lower()
    total = hs + as_

    # Goals Over/Under (e.g. "Over 2.5 Goals", "Under 2.5 Goals")
    if 'goal' in pw_l:
        over   = 'over' in pw_l
        thresh = 3.5 if '3.5' in pw_l else 2.5
        return 'WON' if (over and total > thresh) or (not over and total <= thresh) else 'LOST'

    # Corners — can't determine from score alone; mark LOST to avoid inflating win rate
    if 'corner' in pw_l:
        return 'LOST'

    # Basketball / generic Over/Under (e.g. "Over 220.5", "Under 170.5")
    if 'over' in pw_l or 'under' in pw_l:
        import re as _re
        over   = 'over' in pw_l
        _nums  = _re.findall(r'\d+\.?\d*', pw_l)
        thresh = float(_nums[-1]) if _nums else 220.5
        return 'WON' if (over and total > thresh) or (not over and total <= thresh) else 'LOST'

    # Match result
    home_l, away_l = home.lower(), away.lower()
    if hs > as_:
        return 'WON' if home_l in pw_l else 'LOST'
    elif as_ > hs:
        return 'WON' if away_l in pw_l else 'LOST'
    return 'WON' if 'draw' in pw_l else 'LOST'


def _get_nba_game_result(home, away, match_date):
    """Fetch NBA game result from BallDontLie → TheSportsDB → ESPN."""
    import requests as _req
    # BallDontLie (free, misses playoffs/Finals)
    try:
        r = _req.get(
            'https://www.balldontlie.io/api/v1/games',
            params={'start_date': match_date, 'end_date': match_date, 'per_page': 30},
            timeout=10,
        )
        for g in r.json().get('data', []):
            ht  = g['home_team']['full_name']
            at  = g['visitor_team']['full_name']
            hs  = g.get('home_team_score')
            as_ = g.get('visitor_team_score')
            if not hs or not as_:
                continue
            if _team_similar(ht, home) and _team_similar(at, away):
                return int(hs), int(as_)
    except Exception as e:
        _log.warning('NBA result BDL: %s', e)
    # TheSportsDB l=4387 (covers playoffs)
    try:
        r = _req.get(
            'https://www.thesportsdb.com/api/v1/json/3/eventsday.php',
            params={'d': match_date, 'l': '4387'},
            timeout=10,
        )
        for ev in (r.json().get('events') or []):
            hs_r = ev.get('intHomeScore')
            as_r = ev.get('intAwayScore')
            if hs_r is None or as_r is None:
                continue
            if str(hs_r).strip() in ('', 'None') or str(as_r).strip() in ('', 'None'):
                continue
            eh = ev.get('strHomeTeam', '')
            ea = ev.get('strAwayTeam', '')
            if _team_similar(eh, home) and _team_similar(ea, away):
                return int(float(hs_r)), int(float(as_r))
    except Exception as e:
        _log.warning('NBA result TSDB: %s', e)
    # ESPN NBA scoreboard (date-filtered)
    try:
        r = _req.get(
            'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
            params={'dates': str(match_date).replace('-', '')},
            timeout=10,
        )
        for ev in (r.json().get('events') or []):
            comp0 = (ev.get('competitions') or [{}])[0]
            sname = comp0.get('status', {}).get('type', {}).get('name', '')
            if 'FINAL' not in sname.upper():
                continue
            competitors = comp0.get('competitors') or []
            hc = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            ac = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            if not hc or not ac:
                continue
            hn  = hc.get('team', {}).get('displayName', '')
            an  = ac.get('team', {}).get('displayName', '')
            hs  = hc.get('score')
            as_ = ac.get('score')
            if hs is None or as_ is None:
                continue
            if _team_similar(hn, home) and _team_similar(an, away):
                return int(hs), int(as_)
    except Exception as e:
        _log.warning('NBA result ESPN: %s', e)
    return None


def _get_wnba_game_result(home, away, match_date):
    """Fetch WNBA game result from TheSportsDB → ESPN."""
    import requests as _req
    # TheSportsDB l=4328
    try:
        r = _req.get(
            'https://www.thesportsdb.com/api/v1/json/3/eventsday.php',
            params={'d': match_date, 'l': '4328'},
            timeout=10,
        )
        for ev in (r.json().get('events') or []):
            hs_r = ev.get('intHomeScore')
            as_r = ev.get('intAwayScore')
            if hs_r is None or as_r is None:
                continue
            if str(hs_r).strip() in ('', 'None') or str(as_r).strip() in ('', 'None'):
                continue
            eh = ev.get('strHomeTeam', '')
            ea = ev.get('strAwayTeam', '')
            if _team_similar(eh, home) and _team_similar(ea, away):
                return int(float(hs_r)), int(float(as_r))
    except Exception as e:
        _log.warning('WNBA result TSDB: %s', e)
    # ESPN WNBA scoreboard (date-filtered, primary fixture source)
    try:
        r = _req.get(
            'https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard',
            params={'dates': str(match_date).replace('-', '')},
            timeout=10,
        )
        for ev in (r.json().get('events') or []):
            comp0 = (ev.get('competitions') or [{}])[0]
            sname = comp0.get('status', {}).get('type', {}).get('name', '')
            if 'FINAL' not in sname.upper():
                continue
            competitors = comp0.get('competitors') or []
            hc = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            ac = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            if not hc or not ac:
                continue
            hn  = hc.get('team', {}).get('displayName', '')
            an  = ac.get('team', {}).get('displayName', '')
            hs  = hc.get('score')
            as_ = ac.get('score')
            if hs is None or as_ is None:
                continue
            if _team_similar(hn, home) and _team_similar(an, away):
                return int(hs), int(as_)
    except Exception as e:
        _log.warning('WNBA result ESPN: %s', e)
    return None


def _check_basketball_results():
    """Hourly job: fetch final scores for PENDING NBA/WNBA predictions."""
    _log.info('Basketball results check: scanning pending...')
    now = datetime.now(timezone.utc)
    resolved = 0
    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute("""
                    SELECT match_id, home_team, away_team, match_date,
                           predicted_winner, kickoff_utc, sport
                    FROM predictions
                    WHERE result_status IN ('PENDING', 'LIVE')
                      AND sport IN ('nba', 'wnba')
                """)
                for mid, home, away, match_date, pw, kickoff_str, sport in cur.fetchall():
                    try:
                        if kickoff_str:
                            kickoff = datetime.fromisoformat(
                                str(kickoff_str).replace('Z', '+00:00'))
                        elif match_date:
                            kickoff = datetime.fromisoformat(
                                f'{match_date}T23:00:00+00:00')
                        else:
                            continue
                    except Exception:
                        continue
                    if kickoff + timedelta(hours=3) > now:
                        continue
                    res = (_get_nba_game_result(home, away, str(match_date))
                           if sport == 'nba'
                           else _get_wnba_game_result(home, away, str(match_date)))
                    if res is None:
                        continue
                    hs, as_ = res
                    status = _resolve_result(pw or '', home or '', away or '', hs, as_)
                    cur.execute("""
                        UPDATE predictions
                        SET actual_home_score=%s, actual_away_score=%s,
                            result_status=%s, match_status='FINISHED'
                        WHERE match_id=%s
                    """, (hs, as_, status, mid))
                    _log.info('Basketball result: %s %d-%d %s -> %s',
                              home, hs, as_, away, status)
                    resolved += 1
                _log.info('Basketball results check done: %d resolved', resolved)
                if resolved > 0:
                    _BEST_BET_CACHE["data"] = None
                    _DAILY_TIPS_CACHE["ts"] = 0
    except Exception as e:
        _log.warning('Basketball results check failed: %s', e)


def _check_pending_results():
    """Hourly job: fetch final scores for every PENDING prediction past kickoff+2h."""
    _log.info('Results check: scanning PENDING predictions…')
    now = datetime.now(timezone.utc)
    resolved = 0

    # ── PostgreSQL path ────────────────────────────────────────────────────────
    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute("""
                    SELECT match_id, home_team, away_team, match_date,
                           predicted_winner, kickoff_utc
                    FROM predictions
                    WHERE result_status IN ('PENDING', 'LIVE')
                      AND (sport IS NULL OR sport NOT IN ('nba', 'wnba'))
                """)
                pending = cur.fetchall()
                for mid, home, away, match_date, pw, kickoff_str in pending:
                    try:
                        if kickoff_str:
                            kickoff = datetime.fromisoformat(
                                kickoff_str.replace('Z', '+00:00'))
                        elif match_date:
                            kickoff = datetime.fromisoformat(
                                f'{match_date}T23:00:00+00:00')
                        else:
                            continue
                    except Exception:
                        continue
                    if kickoff + timedelta(hours=2) > now:
                        continue
                    result = _get_result(home, away, match_date)
                    if result is None:
                        continue
                    hs, as_, corners = result
                    status  = _resolve_result(pw or '', home or '', away or '', hs, as_)
                    cur.execute("""
                        UPDATE predictions
                        SET actual_home_score=%s, actual_away_score=%s,
                            actual_corners=%s,
                            result_status=%s, match_status='FINISHED'
                        WHERE match_id=%s
                    """, (hs, as_, corners, status, mid))
                    _log.info('Result: %s %d-%d %s → %s (corners=%s)',
                              home, hs, as_, away, status, corners)
                    resolved += 1
                _log.info('Results check done: %d resolved', resolved)
                if resolved > 0:
                    _DAILY_TIPS_CACHE["ts"] = 0
                    _BEST_BET_CACHE["data"] = None
                return
    except Exception as e:
        _log.warning('DB check_pending failed, falling back to JSON: %s', e)

    # ── JSON fallback ──────────────────────────────────────────────────────────
    with _HISTORY_LOCK:
        data = _read_raw_history()
        changed = False
        for pred in data['predictions']:
            if pred['result_status'] not in ('PENDING', 'LIVE'):
                continue
            kickoff_str = pred.get('kickoff_utc', '')
            match_date  = pred.get('match_date', '')
            try:
                if kickoff_str:
                    kickoff = datetime.fromisoformat(kickoff_str.replace('Z', '+00:00'))
                elif match_date:
                    kickoff = datetime.fromisoformat(f'{match_date}T23:00:00+00:00')
                else:
                    continue
            except Exception:
                continue
            if kickoff + timedelta(hours=2) > now:
                continue
            result = _get_result(pred['home_team'], pred['away_team'], match_date)
            if result is None:
                continue
            hs, as_, corners = result
            pred['actual_home_score'] = hs
            pred['actual_away_score'] = as_
            if corners is not None:
                pred['actual_corners'] = corners
            pred['result_status'] = _resolve_result(
                pred.get('predicted_winner', ''),
                pred['home_team'], pred['away_team'], hs, as_)
            changed = True
            _log.info('JSON result: %s %d-%d %s → %s (corners=%s)',
                      pred['home_team'], hs, as_, pred['away_team'],
                      pred['result_status'], corners)
        if changed:
            _write_raw_history(data)

# ── ESPN live-score polling ───────────────────────────────────────────────────

_ESPN_LIVE_LEAGUES = [
    'fifa.world', 'fifa.friendly',
    'uefa.friendly', 'conmebol.friendly',
    'concacaf.friendly',     # CONCACAF international friendlies
    'afc.friendly',          # Asian international friendlies
    'caf.friendly',          # African international friendlies
    'afc.cupqualification',  # Asian World Cup Qualifying
    'caf.nations',           # Africa Cup of Nations
    'concacaf.nations.league',
]


def _fetch_espn_events():
    """Return dict of (home_lower, away_lower) → event data for live/final matches."""
    import requests as _req
    found = {}
    for slug in _ESPN_LIVE_LEAGUES:
        url = f'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard'
        try:
            r = _req.get(url, timeout=8)
            if r.status_code != 200:
                continue
            for ev in (r.json().get('events') or []):
                comp = (ev.get('competitions') or [{}])[0]
                sname = (comp.get('status') or {}).get('type', {}).get('name', '')
                if sname not in _ESPN_ACTIVE:
                    continue
                competitors = comp.get('competitors') or []
                hc = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                ac = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                if not hc or not ac:
                    continue
                hn  = (hc.get('team', {}).get('displayName') or '').strip()
                an  = (ac.get('team', {}).get('displayName') or '').strip()
                hs  = int(hc.get('score') or 0)
                as_ = int(ac.get('score') or 0)
                minute = (comp.get('status') or {}).get('displayClock', '')
                # Extract total corners from statistics if available
                corners = None
                for stat in (comp.get('statistics') or []):
                    if stat.get('name', '').lower() in ('cornerkicks', 'corners', 'corner kicks'):
                        try:
                            corners = int(stat.get('displayValue') or '0')
                        except (ValueError, TypeError):
                            pass
                        break
                found[(hn.lower(), an.lower())] = {
                    'home': hn, 'away': an,
                    'home_score': hs, 'away_score': as_,
                    'minute': minute, 'espn_status': sname,
                    'corners': corners,
                }
        except Exception as exc:
            _log.warning('ESPN poll %s: %s', slug, exc)
    return found


def _espn_match(espn, home_pred, away_pred):
    """Find ESPN event for predicted teams (fuzzy); swap scores if reversed."""
    for (eh, ea), ev in espn.items():
        if _team_similar(eh, home_pred) and _team_similar(ea, away_pred):
            return ev
        if _team_similar(eh, away_pred) and _team_similar(ea, home_pred):
            return {**ev, 'home_score': ev['away_score'], 'away_score': ev['home_score']}
    return None


_ESPN_ACTIVE = frozenset({
    'STATUS_IN_PROGRESS', 'STATUS_HALFTIME', 'STATUS_SECOND_HALF',
    'STATUS_FINAL', 'STATUS_FULL_TIME', 'STATUS_FULL_ET', 'STATUS_FULL_PEN',
})
_ESPN_DONE = frozenset({
    'STATUS_FINAL', 'STATUS_FULL_TIME', 'STATUS_FULL_ET', 'STATUS_FULL_PEN',
})


def _fetch_espn_result_dated(home_team, away_team, match_date):
    """Like _fetch_espn_result but queries ESPN with a specific date — catches past-day results."""
    import requests as _req
    date_compact = str(match_date).replace('-', '')
    for slug in _ESPN_LIVE_LEAGUES:
        url = f'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard'
        try:
            r = _req.get(url, params={'dates': date_compact}, timeout=8)
            if r.status_code != 200:
                continue
            for ev in (r.json().get('events') or []):
                comp = (ev.get('competitions') or [{}])[0]
                sname = comp.get('status', {}).get('type', {}).get('name', '')
                if sname not in _ESPN_DONE:
                    continue
                competitors = comp.get('competitors') or []
                hc = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                ac = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                if not hc or not ac:
                    continue
                hn  = (hc.get('team', {}).get('displayName') or '').strip()
                an  = (ac.get('team', {}).get('displayName') or '').strip()
                hs  = int(hc.get('score') or 0)
                as_ = int(ac.get('score') or 0)
                corners = None
                for stat in (comp.get('statistics') or []):
                    if stat.get('name', '').lower() in ('cornerkicks', 'corners', 'corner kicks'):
                        try:
                            corners = int(stat.get('displayValue') or '0')
                        except (ValueError, TypeError):
                            pass
                        break
                if _team_similar(hn, home_team) and _team_similar(an, away_team):
                    return hs, as_, corners
                if _team_similar(hn, away_team) and _team_similar(an, home_team):
                    return as_, hs, corners
        except Exception as exc:
            _log.warning('ESPN dated result %s: %s', slug, exc)
    return None


def _update_live_scores():
    """APScheduler job (every 2 min): update live scores then check final results."""
    _log.info('APScheduler: live poll + result check…')
    now  = datetime.now(timezone.utc)
    espn = _fetch_espn_events()

    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute("""
                    SELECT match_id, home_team, away_team, match_date,
                           predicted_winner, kickoff_utc, result_status, sport
                    FROM predictions
                    WHERE result_status IN ('PENDING', 'LIVE')
                """)
                rows = cur.fetchall()
                for mid, home, away, match_date, pw, kickoff_str, curr_status, sport in rows:
                    # ── Basketball: use dedicated result fetchers (not soccer ESPN) ──
                    if sport in ('nba', 'wnba'):
                        try:
                            kickoff = (datetime.fromisoformat(kickoff_str.replace('Z', '+00:00'))
                                       if kickoff_str
                                       else datetime.fromisoformat(f'{match_date}T23:00:00+00:00'))
                        except Exception:
                            continue
                        if kickoff + timedelta(hours=3) > now:
                            continue
                        res = (_get_nba_game_result(home, away, str(match_date))
                               if sport == 'nba'
                               else _get_wnba_game_result(home, away, str(match_date)))
                        if res:
                            hs, as_ = res
                            status = _resolve_result(pw or '', home or '', away or '', hs, as_)
                            cur.execute("""
                                UPDATE predictions
                                SET actual_home_score=%s, actual_away_score=%s,
                                    result_status=%s, match_status='FINISHED'
                                WHERE match_id=%s
                            """, (hs, as_, status, mid))
                            _log.info('Basketball result (2-min check): %s %d-%d %s -> %s',
                                      home, hs, as_, away, status)
                        continue  # skip soccer ESPN check for basketball

                    # ── Football: check ESPN soccer events ────────────────────────
                    ev = _espn_match(espn, home or '', away or '')
                    if ev:
                        hs, as_ = ev['home_score'], ev['away_score']
                        minute  = ev.get('minute', '')
                        if ev['espn_status'] in _ESPN_DONE:
                            status  = _resolve_result(pw or '', home or '', away or '', hs, as_)
                            corners = ev.get('corners')
                            cur.execute("""
                                UPDATE predictions
                                SET actual_home_score=%s, actual_away_score=%s,
                                    actual_corners=%s,
                                    result_status=%s, match_status='FINISHED',
                                    live_home_score=%s, live_away_score=%s
                                WHERE match_id=%s
                            """, (hs, as_, corners, status, hs, as_, mid))
                            _log.info('FINISHED %s %d-%d %s → %s (corners=%s)',
                                      home, hs, as_, away, status, corners)
                        else:
                            cur.execute("""
                                UPDATE predictions
                                SET result_status='LIVE', match_status='LIVE',
                                    live_home_score=%s, live_away_score=%s,
                                    live_minute=%s
                                WHERE match_id=%s
                            """, (hs, as_, minute, mid))
                            _log.info('LIVE %s %d-%d %s (%s)', home, hs, as_, away, minute)
                    else:
                        # ESPN not tracking it — try all result sources
                        try:
                            kickoff = (datetime.fromisoformat(kickoff_str.replace('Z', '+00:00'))
                                       if kickoff_str
                                       else datetime.fromisoformat(f'{match_date}T23:00:00+00:00'))
                        except Exception:
                            continue
                        if kickoff + timedelta(hours=2) > now:
                            continue
                        result = _get_result(home or '', away or '', match_date or '')
                        if result:
                            hs, as_, corners = result
                            status = _resolve_result(pw or '', home or '', away or '', hs, as_)
                            cur.execute("""
                                UPDATE predictions
                                SET actual_home_score=%s, actual_away_score=%s,
                                    actual_corners=%s,
                                    result_status=%s, match_status='FINISHED'
                                WHERE match_id=%s
                            """, (hs, as_, corners, status, mid))
                            _log.info('Result %s %d-%d %s → %s (corners=%s)',
                                      home, hs, as_, away, status, corners)
    except Exception as e:
        _log.warning('DB live update failed: %s', e)
        _check_pending_results()
    # Invalidate tips/best-bet caches so next request gets fresh picks
    _DAILY_TIPS_CACHE["ts"] = 0
    _BEST_BET_CACHE["data"] = None


WC_FIXTURES_RAW = [
    # All times UTC.  EDT = UTC-4 in June.  Midnight-crossover dates adjusted.
    # Sources: ESPN + NBC Sports official schedule (verified Jun 2026).

    # ── GROUP A  Mexico · South Africa · South Korea · Czechia ───────────────
    {"id":"wcA1","group":"A","home":"Mexico",       "away":"South Africa",
     "home_code":"mx","away_code":"za",
     "date":"2026-06-11","time":"19:00","venue":"Estadio Azteca, Mexico City"},
    {"id":"wcA2","group":"A","home":"South Korea",  "away":"Czechia",
     "home_code":"kr","away_code":"cz",
     "date":"2026-06-12","time":"02:00","venue":"Estadio Akron, Guadalajara"},
    {"id":"wcA3","group":"A","home":"Czechia",      "away":"South Africa",
     "home_code":"cz","away_code":"za",
     "date":"2026-06-18","time":"16:00","venue":"Mercedes-Benz Stadium, Atlanta"},
    {"id":"wcA4","group":"A","home":"Mexico",       "away":"South Korea",
     "home_code":"mx","away_code":"kr",
     "date":"2026-06-19","time":"01:00","venue":"Estadio Akron, Guadalajara"},
    {"id":"wcA5","group":"A","home":"Czechia",      "away":"Mexico",
     "home_code":"cz","away_code":"mx",
     "date":"2026-06-25","time":"01:00","venue":"Estadio Azteca, Mexico City"},
    {"id":"wcA6","group":"A","home":"South Africa", "away":"South Korea",
     "home_code":"za","away_code":"kr",
     "date":"2026-06-25","time":"01:00","venue":"Estadio BBVA, Monterrey"},

    # ── GROUP B  Canada · Bosnia and Herzegovina · Qatar · Switzerland ────────
    {"id":"wcB1","group":"B","home":"Canada",       "away":"Bosnia and Herzegovina",
     "home_code":"ca","away_code":"ba",
     "date":"2026-06-12","time":"19:00","venue":"BMO Field, Toronto"},
    {"id":"wcB2","group":"B","home":"Qatar",        "away":"Switzerland",
     "home_code":"qa","away_code":"ch",
     "date":"2026-06-13","time":"19:00","venue":"Levi's Stadium, San Francisco Bay Area"},
    {"id":"wcB3","group":"B","home":"Switzerland",  "away":"Bosnia and Herzegovina",
     "home_code":"ch","away_code":"ba",
     "date":"2026-06-18","time":"19:00","venue":"SoFi Stadium, Los Angeles"},
    {"id":"wcB4","group":"B","home":"Canada",       "away":"Qatar",
     "home_code":"ca","away_code":"qa",
     "date":"2026-06-18","time":"22:00","venue":"BC Place, Vancouver"},
    {"id":"wcB5","group":"B","home":"Switzerland",  "away":"Canada",
     "home_code":"ch","away_code":"ca",
     "date":"2026-06-24","time":"19:00","venue":"BC Place, Vancouver"},
    {"id":"wcB6","group":"B","home":"Bosnia and Herzegovina","away":"Qatar",
     "home_code":"ba","away_code":"qa",
     "date":"2026-06-24","time":"19:00","venue":"Lumen Field, Seattle"},

    # ── GROUP C  Brazil · Morocco · Haiti · Scotland ──────────────────────────
    {"id":"wcC1","group":"C","home":"Brazil",   "away":"Morocco",
     "home_code":"br","away_code":"ma",
     "date":"2026-06-13","time":"22:00","venue":"MetLife Stadium, East Rutherford NJ"},
    {"id":"wcC2","group":"C","home":"Haiti",    "away":"Scotland",
     "home_code":"ht","away_code":"gb-sct",
     "date":"2026-06-14","time":"01:00","venue":"Gillette Stadium, Boston"},
    {"id":"wcC3","group":"C","home":"Scotland", "away":"Morocco",
     "home_code":"gb-sct","away_code":"ma",
     "date":"2026-06-19","time":"22:00","venue":"Gillette Stadium, Boston"},
    {"id":"wcC4","group":"C","home":"Brazil",   "away":"Haiti",
     "home_code":"br","away_code":"ht",
     "date":"2026-06-20","time":"01:00","venue":"Lincoln Financial Field, Philadelphia"},
    {"id":"wcC5","group":"C","home":"Scotland", "away":"Brazil",
     "home_code":"gb-sct","away_code":"br",
     "date":"2026-06-24","time":"22:00","venue":"Hard Rock Stadium, Miami Gardens"},
    {"id":"wcC6","group":"C","home":"Morocco",  "away":"Haiti",
     "home_code":"ma","away_code":"ht",
     "date":"2026-06-24","time":"22:00","venue":"Mercedes-Benz Stadium, Atlanta"},

    # ── GROUP D  USA · Paraguay · Australia · Turkey ──────────────────────────
    {"id":"wcD1","group":"D","home":"USA",       "away":"Paraguay",
     "home_code":"us","away_code":"py",
     "date":"2026-06-13","time":"01:00","venue":"SoFi Stadium, Los Angeles"},
    {"id":"wcD2","group":"D","home":"Australia", "away":"Turkey",
     "home_code":"au","away_code":"tr",
     "date":"2026-06-13","time":"04:00","venue":"BC Place, Vancouver"},
    {"id":"wcD3","group":"D","home":"USA",       "away":"Australia",
     "home_code":"us","away_code":"au",
     "date":"2026-06-19","time":"19:00","venue":"Lumen Field, Seattle"},
    {"id":"wcD4","group":"D","home":"Turkey",    "away":"Paraguay",
     "home_code":"tr","away_code":"py",
     "date":"2026-06-19","time":"04:00","venue":"Levi's Stadium, San Francisco Bay Area"},
    {"id":"wcD5","group":"D","home":"Turkey",    "away":"USA",
     "home_code":"tr","away_code":"us",
     "date":"2026-06-26","time":"02:00","venue":"SoFi Stadium, Los Angeles"},
    {"id":"wcD6","group":"D","home":"Paraguay",  "away":"Australia",
     "home_code":"py","away_code":"au",
     "date":"2026-06-26","time":"02:00","venue":"Levi's Stadium, San Francisco Bay Area"},

    # ── GROUP E  Germany · Curacao · Ivory Coast · Ecuador ────────────────────
    {"id":"wcE1","group":"E","home":"Germany",     "away":"Curacao",
     "home_code":"de","away_code":"cw",
     "date":"2026-06-14","time":"17:00","venue":"NRG Stadium, Houston"},
    {"id":"wcE2","group":"E","home":"Ivory Coast", "away":"Ecuador",
     "home_code":"ci","away_code":"ec",
     "date":"2026-06-14","time":"23:00","venue":"Lincoln Financial Field, Philadelphia"},
    {"id":"wcE3","group":"E","home":"Germany",     "away":"Ivory Coast",
     "home_code":"de","away_code":"ci",
     "date":"2026-06-20","time":"20:00","venue":"BMO Field, Toronto"},
    {"id":"wcE4","group":"E","home":"Ecuador",     "away":"Curacao",
     "home_code":"ec","away_code":"cw",
     "date":"2026-06-21","time":"00:00","venue":"Arrowhead Stadium, Kansas City"},
    {"id":"wcE5","group":"E","home":"Ecuador",     "away":"Germany",
     "home_code":"ec","away_code":"de",
     "date":"2026-06-25","time":"20:00","venue":"MetLife Stadium, East Rutherford NJ"},
    {"id":"wcE6","group":"E","home":"Curacao",     "away":"Ivory Coast",
     "home_code":"cw","away_code":"ci",
     "date":"2026-06-25","time":"20:00","venue":"Lincoln Financial Field, Philadelphia"},

    # ── GROUP F  Netherlands · Japan · Sweden · Tunisia ───────────────────────
    {"id":"wcF1","group":"F","home":"Netherlands","away":"Japan",
     "home_code":"nl","away_code":"jp",
     "date":"2026-06-14","time":"20:00","venue":"AT&T Stadium, Dallas"},
    {"id":"wcF2","group":"F","home":"Sweden",     "away":"Tunisia",
     "home_code":"se","away_code":"tn",
     "date":"2026-06-15","time":"02:00","venue":"Estadio BBVA, Monterrey"},
    {"id":"wcF3","group":"F","home":"Netherlands","away":"Sweden",
     "home_code":"nl","away_code":"se",
     "date":"2026-06-20","time":"17:00","venue":"NRG Stadium, Houston"},
    {"id":"wcF4","group":"F","home":"Tunisia",    "away":"Japan",
     "home_code":"tn","away_code":"jp",
     "date":"2026-06-20","time":"04:00","venue":"Estadio BBVA, Monterrey"},
    {"id":"wcF5","group":"F","home":"Japan",      "away":"Sweden",
     "home_code":"jp","away_code":"se",
     "date":"2026-06-25","time":"23:00","venue":"AT&T Stadium, Dallas"},
    {"id":"wcF6","group":"F","home":"Tunisia",    "away":"Netherlands",
     "home_code":"tn","away_code":"nl",
     "date":"2026-06-25","time":"23:00","venue":"Arrowhead Stadium, Kansas City"},

    # ── GROUP G  Belgium · Egypt · Iran · New Zealand ─────────────────────────
    {"id":"wcG1","group":"G","home":"Belgium",     "away":"Egypt",
     "home_code":"be","away_code":"eg",
     "date":"2026-06-15","time":"19:00","venue":"Lumen Field, Seattle"},
    {"id":"wcG2","group":"G","home":"Iran",        "away":"New Zealand",
     "home_code":"ir","away_code":"nz",
     "date":"2026-06-16","time":"01:00","venue":"SoFi Stadium, Los Angeles"},
    {"id":"wcG3","group":"G","home":"Belgium",     "away":"Iran",
     "home_code":"be","away_code":"ir",
     "date":"2026-06-21","time":"19:00","venue":"SoFi Stadium, Los Angeles"},
    {"id":"wcG4","group":"G","home":"New Zealand", "away":"Egypt",
     "home_code":"nz","away_code":"eg",
     "date":"2026-06-22","time":"01:00","venue":"BC Place, Vancouver"},
    {"id":"wcG5","group":"G","home":"Egypt",       "away":"Iran",
     "home_code":"eg","away_code":"ir",
     "date":"2026-06-27","time":"03:00","venue":"Lumen Field, Seattle"},
    {"id":"wcG6","group":"G","home":"New Zealand", "away":"Belgium",
     "home_code":"nz","away_code":"be",
     "date":"2026-06-27","time":"03:00","venue":"BC Place, Vancouver"},

    # ── GROUP H  Spain · Cape Verde · Saudi Arabia · Uruguay ──────────────────
    {"id":"wcH1","group":"H","home":"Spain",        "away":"Cape Verde",
     "home_code":"es","away_code":"cv",
     "date":"2026-06-15","time":"16:00","venue":"Mercedes-Benz Stadium, Atlanta"},
    {"id":"wcH2","group":"H","home":"Saudi Arabia", "away":"Uruguay",
     "home_code":"sa","away_code":"uy",
     "date":"2026-06-15","time":"22:00","venue":"Hard Rock Stadium, Miami Gardens"},
    {"id":"wcH3","group":"H","home":"Spain",        "away":"Saudi Arabia",
     "home_code":"es","away_code":"sa",
     "date":"2026-06-21","time":"16:00","venue":"Mercedes-Benz Stadium, Atlanta"},
    {"id":"wcH4","group":"H","home":"Uruguay",      "away":"Cape Verde",
     "home_code":"uy","away_code":"cv",
     "date":"2026-06-21","time":"22:00","venue":"Hard Rock Stadium, Miami Gardens"},
    {"id":"wcH5","group":"H","home":"Cape Verde",   "away":"Saudi Arabia",
     "home_code":"cv","away_code":"sa",
     "date":"2026-06-27","time":"00:00","venue":"NRG Stadium, Houston"},
    {"id":"wcH6","group":"H","home":"Uruguay",      "away":"Spain",
     "home_code":"uy","away_code":"es",
     "date":"2026-06-27","time":"00:00","venue":"Estadio Akron, Guadalajara"},

    # ── GROUP I  France · Senegal · Iraq · Norway ─────────────────────────────
    {"id":"wcI1","group":"I","home":"France",  "away":"Senegal",
     "home_code":"fr","away_code":"sn",
     "date":"2026-06-16","time":"19:00","venue":"MetLife Stadium, East Rutherford NJ"},
    {"id":"wcI2","group":"I","home":"Iraq",    "away":"Norway",
     "home_code":"iq","away_code":"no",
     "date":"2026-06-16","time":"22:00","venue":"Gillette Stadium, Boston"},
    {"id":"wcI3","group":"I","home":"France",  "away":"Iraq",
     "home_code":"fr","away_code":"iq",
     "date":"2026-06-22","time":"21:00","venue":"Lincoln Financial Field, Philadelphia"},
    {"id":"wcI4","group":"I","home":"Norway",  "away":"Senegal",
     "home_code":"no","away_code":"sn",
     "date":"2026-06-23","time":"00:00","venue":"MetLife Stadium, East Rutherford NJ"},
    {"id":"wcI5","group":"I","home":"Norway",  "away":"France",
     "home_code":"no","away_code":"fr",
     "date":"2026-06-26","time":"19:00","venue":"Gillette Stadium, Boston"},
    {"id":"wcI6","group":"I","home":"Senegal", "away":"Iraq",
     "home_code":"sn","away_code":"iq",
     "date":"2026-06-26","time":"19:00","venue":"BMO Field, Toronto"},

    # ── GROUP J  Argentina · Algeria · Austria · Jordan ───────────────────────
    {"id":"wcJ1","group":"J","home":"Argentina","away":"Algeria",
     "home_code":"ar","away_code":"dz",
     "date":"2026-06-17","time":"01:00","venue":"Arrowhead Stadium, Kansas City"},
    {"id":"wcJ2","group":"J","home":"Austria",  "away":"Jordan",
     "home_code":"at","away_code":"jo",
     "date":"2026-06-16","time":"04:00","venue":"Levi's Stadium, San Francisco Bay Area"},
    {"id":"wcJ3","group":"J","home":"Argentina","away":"Austria",
     "home_code":"ar","away_code":"at",
     "date":"2026-06-22","time":"17:00","venue":"AT&T Stadium, Dallas"},
    {"id":"wcJ4","group":"J","home":"Jordan",   "away":"Algeria",
     "home_code":"jo","away_code":"dz",
     "date":"2026-06-23","time":"03:00","venue":"Levi's Stadium, San Francisco Bay Area"},
    {"id":"wcJ5","group":"J","home":"Algeria",  "away":"Austria",
     "home_code":"dz","away_code":"at",
     "date":"2026-06-28","time":"02:00","venue":"Arrowhead Stadium, Kansas City"},
    {"id":"wcJ6","group":"J","home":"Jordan",   "away":"Argentina",
     "home_code":"jo","away_code":"ar",
     "date":"2026-06-28","time":"02:00","venue":"AT&T Stadium, Dallas"},

    # ── GROUP K  Portugal · DR Congo · Uzbekistan · Colombia ──────────────────
    {"id":"wcK1","group":"K","home":"Portugal",   "away":"DR Congo",
     "home_code":"pt","away_code":"cd",
     "date":"2026-06-17","time":"17:00","venue":"NRG Stadium, Houston"},
    {"id":"wcK2","group":"K","home":"Uzbekistan", "away":"Colombia",
     "home_code":"uz","away_code":"co",
     "date":"2026-06-18","time":"02:00","venue":"Estadio Azteca, Mexico City"},
    {"id":"wcK3","group":"K","home":"Portugal",   "away":"Uzbekistan",
     "home_code":"pt","away_code":"uz",
     "date":"2026-06-23","time":"17:00","venue":"NRG Stadium, Houston"},
    {"id":"wcK4","group":"K","home":"Colombia",   "away":"DR Congo",
     "home_code":"co","away_code":"cd",
     "date":"2026-06-24","time":"02:00","venue":"Estadio Akron, Guadalajara"},
    {"id":"wcK5","group":"K","home":"Colombia",   "away":"Portugal",
     "home_code":"co","away_code":"pt",
     "date":"2026-06-27","time":"23:30","venue":"Hard Rock Stadium, Miami Gardens"},
    {"id":"wcK6","group":"K","home":"DR Congo",   "away":"Uzbekistan",
     "home_code":"cd","away_code":"uz",
     "date":"2026-06-27","time":"23:30","venue":"Mercedes-Benz Stadium, Atlanta"},

    # ── GROUP L  England · Croatia · Ghana · Panama ───────────────────────────
    {"id":"wcL1","group":"L","home":"England", "away":"Croatia",
     "home_code":"gb-eng","away_code":"hr",
     "date":"2026-06-17","time":"20:00","venue":"AT&T Stadium, Dallas"},
    {"id":"wcL2","group":"L","home":"Ghana",   "away":"Panama",
     "home_code":"gh","away_code":"pa",
     "date":"2026-06-17","time":"23:00","venue":"BMO Field, Toronto"},
    {"id":"wcL3","group":"L","home":"England", "away":"Ghana",
     "home_code":"gb-eng","away_code":"gh",
     "date":"2026-06-23","time":"20:00","venue":"Gillette Stadium, Boston"},
    {"id":"wcL4","group":"L","home":"Panama",  "away":"Croatia",
     "home_code":"pa","away_code":"hr",
     "date":"2026-06-23","time":"23:00","venue":"BMO Field, Toronto"},
    {"id":"wcL5","group":"L","home":"Panama",  "away":"England",
     "home_code":"pa","away_code":"gb-eng",
     "date":"2026-06-27","time":"21:00","venue":"MetLife Stadium, East Rutherford NJ"},
    {"id":"wcL6","group":"L","home":"Croatia", "away":"Ghana",
     "home_code":"hr","away_code":"gh",
     "date":"2026-06-27","time":"21:00","venue":"Lincoln Financial Field, Philadelphia"},
]

def _build_wc_fixtures():
    fixtures = []
    for raw in WC_FIXTURES_RAW:
        pred = _wc_predict(raw['home'], raw['away'])
        if pred is None:
            _log.error('No prediction for WC fixture %s vs %s — skipping',
                       raw['home'], raw['away'])
            continue

        ph, pd_, pa = pred['home_win'], pred['draw'], pred['away_win']
        pg  = pred['over_goals']
        best_prob = max(ph, pd_, pa)
        if best_prob == ph:
            bet = {'label': f"{raw['home']} to Win", 'confidence': ph}
        elif best_prob == pa:
            bet = {'label': f"{raw['away']} to Win", 'confidence': pa}
        else:
            bet = {'label': 'Draw', 'confidence': pd_}

        entry = {
            **raw,
            'utc_kickoff':  f"{raw['date']}T{raw['time']}:00Z",
            'result':       {'home': ph, 'draw': pd_, 'away': pa},
            'over_goals':   pg,
            'btts':         pred.get('btts',         0.48),
            'over_corners': pred.get('over_corners', 0.52),
            'best_bet':     bet,
            'competition':  'FIFA World Cup 2026',
        }
        _save_prediction(entry)
        fixtures.append(entry)
    return fixtures


try:
    _init_db()
except Exception as _e:
    print(f'WARNING: _init_db failed: {_e}', flush=True)

try:
    _migrate_json_to_db()
except Exception as _e:
    print(f'WARNING: _migrate_json_to_db failed: {_e}', flush=True)

print("Loading WC predictions…", flush=True)
try:
    _WC_FIXTURES = _build_wc_fixtures()
    print(f"  WC fixtures ready ({len(_WC_FIXTURES)} matches)", flush=True)
except Exception as _e:
    print(f'WARNING: _build_wc_fixtures failed: {_e}', flush=True)
    _WC_FIXTURES = []

# ── API routes ────────────────────────────────────────────────────────────────

_CACHE_TTL = 600  # 10 minutes

_CLUB_TIPS_CACHE: dict  = {"data": None, "ts": 0.0}


@app.route('/api/club/tips')
def club_tips():
    now_ts = time.time()
    if _CLUB_TIPS_CACHE["data"] is None or (now_ts - _CLUB_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_club_tips(_club_predict)
            for t in tips:
                _save_prediction(t)
            _CLUB_TIPS_CACHE["data"] = tips
            _CLUB_TIPS_CACHE["ts"]   = now_ts
        except Exception as exc:
            app.logger.error("club-tips fetch failed: %s", exc)
            if _CLUB_TIPS_CACHE["data"] is None:
                _CLUB_TIPS_CACHE["data"] = []
    done = _completed_keys()
    active = [t for t in (_CLUB_TIPS_CACHE["data"] or [])
              if _match_key(t.get('home_team',''), t.get('away_team',''),
                            (t.get('utc_kickoff','') or '')[:10]) not in done]
    return jsonify(active)


def _wc_round_cutoff():
    """Return ISO date string — only show WC fixtures up to this date (auto-advances by round)."""
    from datetime import date as _d
    today = datetime.now(timezone.utc).date()
    if today < _d(2026, 6, 15): return '2026-06-14'   # Round 1 first batch
    if today < _d(2026, 6, 18): return '2026-06-17'   # Full round 1
    if today < _d(2026, 6, 24): return '2026-06-23'   # Round 2
    return '2026-06-28'                                 # Round 3


@app.route('/api/worldcup/fixtures')
def wc_fixtures():
    now    = datetime.now(timezone.utc)
    done   = _completed_keys()
    cutoff = _wc_round_cutoff()
    upcoming = []
    for f in _WC_FIXTURES:
        if f['date'] > cutoff:
            continue
        mk = _match_key(f['home'], f['away'], f['date'])
        if mk in done:
            continue
        try:
            kickoff = datetime.fromisoformat(
                f['date'] + 'T' + f['time'] + ':00'
            ).replace(tzinfo=timezone.utc)
            if kickoff + timedelta(minutes=120) > now:
                upcoming.append(f)
        except Exception:
            upcoming.append(f)
    return jsonify(upcoming)


@app.route('/api/countdown')
def countdown():
    # Find the next WC match after now
    now = datetime.now(timezone.utc)
    next_match = None
    for f in _WC_FIXTURES:
        dt = datetime.fromisoformat(f['date'] + 'T' + f['time'] + ':00').replace(tzinfo=timezone.utc)
        if dt > now:
            next_match = {'fixture': f, 'utc_kickoff': dt.isoformat()}
            diff = dt - now
            next_match['seconds_remaining'] = int(diff.total_seconds())
            break
    if next_match is None:
        next_match = {'seconds_remaining': 0, 'fixture': _WC_FIXTURES[0]}
    return jsonify(next_match)


@app.route('/api/debug/predict')
def debug_predict():
    """Temporary debug endpoint — returns raw prediction for any team pair."""
    from flask import request as req
    home = req.args.get('home', '')
    away = req.args.get('away', '')
    if not home or not away:
        return jsonify({'error': 'Pass ?home=X&away=Y'}), 400
    try:
        r = predict_match(home, away)
        return jsonify({
            'home': home, 'away': away,
            'resolved_home': r['resolved_home'],
            'resolved_away': r['resolved_away'],
            'home_win':     r['home_win'],
            'draw':         r['draw'],
            'away_win':     r['away_win'],
            'over_goals':   r['over_goals'],
            'btts':         r['btts'],
            'over_corners': r['over_corners'],
            'display': {
                'home_win':   f"{r['home_win']*100:.1f}%",
                'draw':       f"{r['draw']*100:.1f}%",
                'away_win':   f"{r['away_win']*100:.1f}%",
                'over_goals': f"{r['over_goals']*100:.1f}%",
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ── Daily tips (live fixtures + predictions) ──────────────────────────────────

_DAILY_TIPS_CACHE: dict = {"data": None, "ts": 0.0}
_INTL_TIPS_CACHE:  dict = {"data": None, "ts": 0.0}
_BEST_BET_CACHE:   dict = {"data": None, "ts": 0.0}


@app.route("/api/intl/fixtures")
def intl_fixtures():
    now = time.time()
    if _INTL_TIPS_CACHE["data"] is None or (now - _INTL_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_intl_tips(_wc_predict)
            for t in tips:
                _save_prediction(t)
            _INTL_TIPS_CACHE["data"] = tips
            _INTL_TIPS_CACHE["ts"]   = now
        except Exception as exc:
            app.logger.error("intl-fixtures fetch failed: %s", exc)
            if _INTL_TIPS_CACHE["data"] is None:
                _INTL_TIPS_CACHE["data"] = []
    done = _completed_keys()
    active = [t for t in (_INTL_TIPS_CACHE["data"] or [])
              if _match_key(t.get('home_team',''), t.get('away_team',''),
                            (t.get('utc_kickoff','') or '')[:10]) not in done]
    return jsonify(active)


@app.route("/api/daily-tips")
def daily_tips():
    now = time.time()
    if _DAILY_TIPS_CACHE["data"] is None or (now - _DAILY_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_daily_tips(_club_predict, _wc_predict)
            for t in tips:
                _save_prediction(t)
            _DAILY_TIPS_CACHE["data"] = tips
            _DAILY_TIPS_CACHE["ts"]   = now
        except Exception as exc:
            app.logger.error("daily-tips fetch failed: %s", exc)
            return jsonify([])

    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    done = _completed_keys()
    active = [t for t in (_DAILY_TIPS_CACHE["data"] or [])
              if _match_key(t.get('home_team','') or t.get('home',''),
                            t.get('away_team','') or t.get('away',''),
                            (t.get('utc_kickoff','') or '')[:10]) not in done]

    # Merge today's World Cup fixtures (not covered by football-data.org free tier)
    seen_keys = {_match_key(
        t.get('home_team','') or t.get('home',''),
        t.get('away_team','') or t.get('away',''),
        (t.get('utc_kickoff','') or '')[:10]
    ) for t in active}
    for f in _WC_FIXTURES:
        if f.get('date','') != today_str:
            continue
        mk = _match_key(f['home'], f['away'], f['date'])
        if mk in done or mk in seen_keys:
            continue
        seen_keys.add(mk)
        active.append({
            'id':               mk,
            'league':           'FIFA World Cup 2026',
            'home_team':        f['home'],
            'away_team':        f['away'],
            'home_crest':       f"https://flagcdn.com/w40/{f.get('home_code','')}.png",
            'away_crest':       f"https://flagcdn.com/w40/{f.get('away_code','')}.png",
            'is_international': True,
            'time':             f.get('time',''),
            'utc_kickoff':      f.get('utc_kickoff',''),
            'result':           f['result'],
            'over_goals':       f.get('over_goals', 0.5),
            'btts':             f.get('btts', 0.48),
            'over_corners':     f.get('over_corners', 0.52),
            'best_bet':         f.get('best_bet', {}),
        })
    return jsonify(active)


@app.route('/api/best-bet')
def best_bet_of_day():
    """Return the single highest-confidence outcome and a 3-pick accumulator from all today's fixtures."""
    now_ts = time.time()
    if _BEST_BET_CACHE["data"] is not None and (now_ts - _BEST_BET_CACHE["ts"]) < _CACHE_TTL:
        return jsonify(_BEST_BET_CACHE["data"])

    # Warm club cache
    if _CLUB_TIPS_CACHE["data"] is None or (now_ts - _CLUB_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_club_tips(_club_predict)
            for t in tips: _save_prediction(t)
            _CLUB_TIPS_CACHE["data"] = tips
            _CLUB_TIPS_CACHE["ts"]   = now_ts
        except Exception as exc:
            _log.warning('best-bet: club warm: %s', exc)
            _CLUB_TIPS_CACHE.setdefault("data", [])

    # Warm international cache
    if _INTL_TIPS_CACHE["data"] is None or (now_ts - _INTL_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_intl_tips(_wc_predict)
            for t in tips: _save_prediction(t)
            _INTL_TIPS_CACHE["data"] = tips
            _INTL_TIPS_CACHE["ts"]   = now_ts
        except Exception as exc:
            _log.warning('best-bet: intl warm: %s', exc)
            _INTL_TIPS_CACHE.setdefault("data", [])

    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Aggregate all sources; include today's WC fixtures
    raw = list(_CLUB_TIPS_CACHE["data"] or []) + list(_INTL_TIPS_CACHE["data"] or [])
    for f in _WC_FIXTURES:
        if f.get('date', '') == today_str:
            raw.append(f)

    # Deduplicate by match_id
    seen_ids = set()
    unique_tips = []
    for t in raw:
        h   = t.get('home_team') or t.get('home', '')
        a   = t.get('away_team') or t.get('away', '')
        d   = (t.get('utc_kickoff') or '')[:10] or t.get('date', '')
        mid = _match_key(h, a, d)
        if mid not in seen_ids:
            seen_ids.add(mid)
            unique_tips.append(t)

    # Build all outcome candidates
    candidates = []
    for t in unique_tips:
        h      = t.get('home_team') or t.get('home', '')
        a      = t.get('away_team') or t.get('away', '')
        d      = (t.get('utc_kickoff') or '')[:10] or t.get('date', '')
        mid    = _match_key(h, a, d)
        lbl    = f"{h} vs {a}"
        league = t.get('league') or t.get('competition', '')
        res    = t.get('result', {})
        ph, pd_, pa = res.get('home', 0), res.get('draw', 0), res.get('away', 0)
        og  = t.get('over_goals',   0.5)
        bt  = t.get('btts',         0.5)
        oc  = t.get('over_corners', 0.5)
        koff = t.get('utc_kickoff', '')

        def _c(pick, prob, typ):
            return {'match_id': mid, 'match': lbl, 'league': league,
                    'pick': pick, 'prob': round(prob, 4), 'type': typ,
                    'utc_kickoff': koff}

        # Match result
        res_prob = max(ph, pd_, pa)
        if res_prob > 0:
            res_lbl = (f"{h} Win" if res_prob == ph else
                       f"{a} Win" if res_prob == pa else "Draw")
            candidates.append(_c(res_lbl, res_prob, 'result'))

        # Goals Over/Under 2.5
        candidates.append(_c('Over 2.5 Goals'  if og >= 0.5 else 'Under 2.5 Goals',
                              max(og, 1 - og), 'goals'))
        # BTTS
        candidates.append(_c('Both Teams to Score' if bt >= 0.5 else 'Both Teams Not to Score',
                              max(bt, 1 - bt), 'btts'))
        # Corners Over/Under 9.5
        candidates.append(_c('Corners Over 9.5' if oc >= 0.5 else 'Corners Under 9.5',
                              max(oc, 1 - oc), 'corners'))

    # ── Add NBA / WNBA candidates to the pool ─────────────────────────────────
    for sport_label, bball_games, default_ou in [
        ('NBA',  _NBA_CACHE.get('data')  or [], 220.5),
        ('WNBA', _WNBA_CACHE.get('data') or [], 170.5),
    ]:
        for g in bball_games:
            if g.get('date','') != today_str:
                continue
            if g.get('status','').lower() in ('final', 'ft', 'finished'):
                continue
            bh = g.get('home_team','')
            ba = g.get('away_team','')
            if not bh or not ba:
                continue
            bmid  = _match_key(bh, ba, g['date'])
            blbl  = f"{bh} vs {ba}"
            bcomp = g.get('competition', sport_label)
            btime = g.get('time','')
            bkoff = (f"{g['date']}T{btime}:00Z"
                     if btime and btime not in ('','00:00')
                     else f"{g['date']}T23:59:59Z")
            bres  = g.get('result', {})
            bph, bpa = bres.get('home', 0.5), bres.get('away', 0.5)
            if max(bph, bpa) > 0:
                rlbl = f"{bh} Win" if bph >= bpa else f"{ba} Win"
                candidates.append({'match_id': bmid, 'match': blbl, 'league': bcomp,
                                    'pick': rlbl, 'prob': round(max(bph,bpa),4),
                                    'type': 'result', 'utc_kickoff': bkoff})
            bou = g.get('over_total', 0.5)
            bline = g.get('ou_line', default_ou)
            bou_pick = f"Over {bline}" if bou >= 0.5 else f"Under {bline}"
            candidates.append({'match_id': bmid, 'match': blbl, 'league': bcomp,
                                'pick': bou_pick, 'prob': round(max(bou,1-bou),4),
                                'type': 'goals', 'utc_kickoff': bkoff})

    # Only include candidates whose game hasn't started yet
    now_utc = datetime.now(timezone.utc)

    def _kickoff_upcoming(c):
        koff = c.get('utc_kickoff', '')
        if not koff:
            return True
        try:
            ko = datetime.fromisoformat(koff.replace('Z', '+00:00'))
            return ko > now_utc
        except Exception:
            return True

    candidates = [c for c in candidates if _kickoff_upcoming(c)]

    if not candidates:
        payload = {'best_bet': None, 'accumulator': [], 'combined_prob': 0.0}
        _BEST_BET_CACHE.update(data=payload, ts=now_ts)
        return jsonify(payload)

    candidates.sort(key=lambda x: x['prob'], reverse=True)
    best = candidates[0]

    # Accumulator: top 3 picks from DIFFERENT matches
    acca, seen_matches = [], set()
    for c in candidates:
        if c['match_id'] not in seen_matches:
            acca.append(c)
            seen_matches.add(c['match_id'])
        if len(acca) == 3:
            break

    combined = 1.0
    for pick in acca:
        combined *= pick['prob']

    payload = {
        'best_bet':      best,
        'accumulator':   acca,
        'combined_prob': round(combined, 4),
        'updated':       datetime.now(timezone.utc).isoformat(),
    }
    _BEST_BET_CACHE.update(data=payload, ts=now_ts)
    return jsonify(payload)


# ── NBA / WNBA endpoints ──────────────────────────────────────────────────────

_NBA_CACHE:  dict = {"data": None, "ts": 0.0}
_WNBA_CACHE: dict = {"data": None, "ts": 0.0}

def _nba_logo(abbr, sport='nba'):
    return f"https://a.espncdn.com/i/teamlogos/{sport}/500/{abbr.lower()}.png"


@app.route('/api/nba/fixtures')
def nba_fixtures():
    now_ts = time.time()
    if _NBA_CACHE["data"] is None or (now_ts - _NBA_CACHE["ts"]) > _CACHE_TTL:
        try:
            from nba_predict import get_nba_fixtures as _get_games, predict_nba as _predict
            games   = _get_games()
            result  = []
            for g in games:
                pred = _predict(g['home_team'], g['away_team'])
                if pred.get('error'):
                    continue
                comp = g.get('competition') or ('NBA Playoffs' if g.get('postseason') else 'NBA')
                result.append({
                    **g,
                    'sport':            'nba',
                    'competition':      comp,
                    'ou_line':          220.5,
                    'result':           {'home': round(pred['home_win_pct'] / 100, 3),
                                         'away': round(pred['away_win_pct'] / 100, 3)},
                    'over_total':       round(pred['over_pct'] / 100, 3),
                    'predicted_winner': pred['predicted_winner'],
                    'win_probability':  pred['win_probability'],
                    'predicted_ou':     pred['predicted_ou'],
                    'ou_probability':   pred['ou_probability'],
                    'best_bet':         pred['best_bet'],
                    'best_bet_type':    pred['best_bet_type'],
                    'home_logo':        _nba_logo(g.get('home_abbr', ''), 'nba'),
                    'away_logo':        _nba_logo(g.get('away_abbr', ''), 'nba'),
                })
                _save_basketball_prediction(
                    {**g, 'competition': comp}, pred, 'nba'
                )
            _NBA_CACHE["data"] = result
            _NBA_CACHE["ts"]   = now_ts
        except Exception as exc:
            app.logger.error("NBA fixtures failed: %s", exc)
            if _NBA_CACHE["data"] is None:
                _NBA_CACHE["data"] = []
    return jsonify(_NBA_CACHE["data"] or [])


@app.route('/api/wnba/fixtures')
def wnba_fixtures():
    now_ts = time.time()
    if _WNBA_CACHE["data"] is None or (now_ts - _WNBA_CACHE["ts"]) > _CACHE_TTL:
        try:
            from nba_predict import get_wnba_fixtures as _get_games, predict_wnba as _predict
            games   = _get_games()
            result  = []
            for g in games:
                pred = _predict(g['home_team'], g['away_team'])
                if pred.get('error'):
                    continue
                result.append({
                    **g,
                    'sport':            'wnba',
                    'competition':      'WNBA',
                    'ou_line':          170.5,
                    'result':           {'home': round(pred['home_win_pct'] / 100, 3),
                                         'away': round(pred['away_win_pct'] / 100, 3)},
                    'over_total':       round(pred['over_pct'] / 100, 3),
                    'predicted_winner': pred['predicted_winner'],
                    'win_probability':  pred['win_probability'],
                    'predicted_ou':     pred['predicted_ou'],
                    'ou_probability':   pred['ou_probability'],
                    'best_bet':         pred['best_bet'],
                    'best_bet_type':    pred['best_bet_type'],
                    'home_logo':        _nba_logo(g.get('home_abbr', ''), 'wnba'),
                    'away_logo':        _nba_logo(g.get('away_abbr', ''), 'wnba'),
                })
                _save_basketball_prediction(
                    {**g, 'competition': 'WNBA'}, pred, 'wnba'
                )
            _WNBA_CACHE["data"] = result
            _WNBA_CACHE["ts"]   = now_ts
        except Exception as exc:
            app.logger.error("WNBA fixtures failed: %s", exc)
            if _WNBA_CACHE["data"] is None:
                _WNBA_CACHE["data"] = []
    return jsonify(_WNBA_CACHE["data"] or [])


@app.route('/api/results')
def results_history():
    """Return completed prediction history with stats."""
    now   = datetime.now(timezone.utc)
    data  = _get_history()
    preds = data.get('predictions', [])

    finished = []
    for p in preds:
        if p['result_status'] in ('WON', 'LOST'):
            finished.append(p)
        elif p['result_status'] in ('PENDING', 'LIVE'):
            kickoff_str = p.get('kickoff_utc')
            try:
                if kickoff_str:
                    ko = datetime.fromisoformat(str(kickoff_str).replace('Z', '+00:00'))
                elif p.get('match_date'):
                    ko = datetime.fromisoformat(f"{p['match_date']}T23:00:00+00:00")
                else:
                    continue
                if ko + timedelta(hours=2) < now:
                    finished.append(p)
            except Exception:
                pass

    finished = sorted(finished, key=lambda p: p.get('match_date', ''), reverse=True)

    # Attach per-market sub-results to each finished prediction
    for p in finished:
        p['sub_results'] = _sub_results(p)

    resolved   = [p for p in finished if p['result_status'] in ('WON', 'LOST')]
    won        = sum(1 for p in resolved if p['result_status'] == 'WON')
    lost       = len(resolved) - won
    win_rate   = round(won / len(resolved) * 100, 1) if resolved else 0.0

    streak = 0
    streak_type = None
    for p in resolved:
        if streak_type is None:
            streak_type = p['result_status']
            streak = 1
        elif p['result_status'] == streak_type:
            streak += 1
        else:
            break

    return jsonify({
        'stats': {
            'total':       len(resolved),
            'won':         won,
            'lost':        lost,
            'win_rate':    win_rate,
            'streak':      streak,
            'streak_type': streak_type,
        },
        'predictions': finished,
    })


@app.route('/api/live-scores')
def live_scores():
    """Return all currently LIVE predictions keyed by match_id."""
    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute("""
                    SELECT match_id, home_team, away_team,
                           live_home_score, live_away_score, live_minute
                    FROM predictions WHERE result_status = 'LIVE'
                """)
                cols = [d[0] for d in cur.description]
                live = {
                    row[0]: dict(zip(cols, row))
                    for row in cur.fetchall()
                }
                return jsonify({'live': live})
    except Exception as e:
        _log.warning('live-scores endpoint: %s', e)
    return jsonify({'live': {}})


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    db_ok  = False
    db_msg = 'DATABASE_URL not set'
    if _DB_CONN_URL:
        try:
            with _db() as (conn, cur):
                if conn:
                    cur.execute('SELECT 1')
                    db_ok  = True
                    db_msg = 'connected'
        except Exception as e:
            db_msg = str(e)
    return jsonify({
        'status':       'ok',
        'db':           db_ok,
        'db_msg':       db_msg,
        'db_url_set':   bool(_DB_CONN_URL),
        'korapay_set':  bool(_KORAPAY_PUBLIC),
    })


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    data     = request.get_json() or {}
    name     = (data.get('name') or '').strip()
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password', '')
    if not name or not email or not password:
        return jsonify({'error': 'Name, email and password are required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if not _DB_CONN_URL:
        return jsonify({'error': 'Server database is not configured — contact support'}), 503
    pw_hash = generate_password_hash(password)
    token   = None
    user_id = None
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'Could not connect to database — try again in a moment'}), 503
            cur.execute('SELECT id FROM users WHERE email=%s', (email,))
            if cur.fetchone():
                return jsonify({'error': 'An account with that email already exists'}), 409
            cur.execute(
                'INSERT INTO users (name,email,password_hash) VALUES (%s,%s,%s) RETURNING id',
                (name, email, pw_hash)
            )
            row = cur.fetchone()
            user_id = row[0]
            token   = secrets.token_hex(32)
            expires = datetime.now(timezone.utc) + timedelta(days=30)
            cur.execute(
                'INSERT INTO sessions (user_id,token,expires_at) VALUES (%s,%s,%s)',
                (user_id, token, expires)
            )
        return jsonify({'token': token,
                        'user':  {'id': user_id, 'name': name,
                                  'email': email, 'is_premium': False}}), 201
    except Exception as e:
        msg = str(e) or repr(e) or 'Unknown server error'
        print(f'auth_register error: {type(e).__name__}: {msg}', flush=True)
        return jsonify({'error': f'Registration error: {msg}'}), 500


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    data     = request.get_json() or {}
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password', '')
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    if not _DB_CONN_URL:
        return jsonify({'error': 'Server database is not configured — contact support'}), 503
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'Could not connect to database — try again in a moment'}), 503
            cur.execute(
                'SELECT id,name,email,password_hash,is_premium FROM users WHERE email=%s',
                (email,)
            )
            row = cur.fetchone()
            if not row or not check_password_hash(row[3], password):
                return jsonify({'error': 'Invalid email or password'}), 401
            uid, uname, uemail, _, is_prem = row
            token   = secrets.token_hex(32)
            expires = datetime.now(timezone.utc) + timedelta(days=30)
            cur.execute(
                'INSERT INTO sessions (user_id,token,expires_at) VALUES (%s,%s,%s)',
                (uid, token, expires)
            )
        return jsonify({'token': token,
                        'user':  {'id': uid, 'name': uname,
                                  'email': uemail, 'is_premium': bool(is_prem)}})
    except Exception as e:
        msg = str(e) or repr(e) or 'Unknown server error'
        print(f'auth_login error: {type(e).__name__}: {msg}', flush=True)
        return jsonify({'error': f'Login error: {msg}'}), 500


@app.route('/api/auth/me')
def auth_me():
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({'user': user})


# ── Payment endpoints (Korapay) ───────────────────────────────────────────────

_KORA_INIT_URL   = 'https://api.korapay.com/merchant/api/v1/charges/initialize'
_KORA_VERIFY_URL = 'https://api.korapay.com/merchant/api/v1/charges/{reference}'

# Amount thresholds in Naira
_PLAN_AMOUNTS = {
    'monthly': 5_000,
    '3month':  15_000,
}


@app.route('/api/payment/initialize', methods=['POST'])
def payment_initialize():
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    if not _KORAPAY_SECRET:
        return jsonify({'error': 'Payment not configured'}), 503
    body   = request.get_json(silent=True) or {}
    plan   = body.get('plan', 'monthly')
    amount = _PLAN_AMOUNTS.get(plan, _PLAN_AMOUNTS['monthly'])
    ref    = f"azpred_{user['id']}_{secrets.token_hex(8)}"
    # Build redirect URL back to /subscribe so the callback is caught by React
    base        = request.host_url.rstrip('/')
    redirect_url = f"{base}/subscribe"
    import requests as _req
    try:
        r = _req.post(
            _KORA_INIT_URL,
            json={
                'amount':       amount,
                'currency':     'NGN',
                'reference':    ref,
                'redirect_url': redirect_url,
                'customer':     {'email': user['email']},
                'channels':     ['card', 'bank_transfer'],
                'metadata':     {'plan': plan, 'user_id': user['id']},
            },
            headers={
                'Authorization': f'Bearer {_KORAPAY_SECRET}',
                'Content-Type':  'application/json',
            },
            timeout=15,
        )
        if r.status_code not in (200, 201):
            return jsonify({'error': 'Korapay initialization failed', 'detail': r.text}), 502
        data = r.json().get('data', {})
        checkout_url = data.get('checkout_url') or data.get('authorization_url')
        if not checkout_url:
            return jsonify({'error': 'No checkout URL returned by Korapay'}), 502
        return jsonify({'checkout_url': checkout_url, 'reference': ref})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/payment/verify')
def payment_verify():
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    reference = request.args.get('reference', '')
    if not reference:
        return jsonify({'error': 'Reference required'}), 400
    if not _KORAPAY_SECRET:
        return jsonify({'error': 'Payment not configured'}), 503
    import requests as _req
    try:
        r = _req.get(
            _KORA_VERIFY_URL.format(reference=reference),
            headers={'Authorization': f'Bearer {_KORAPAY_SECRET}'},
            timeout=10,
        )
        if r.status_code != 200:
            return jsonify({'error': 'Korapay verification failed', 'detail': r.text}), 502
        data = r.json().get('data', {})
        if data.get('status') != 'success':
            return jsonify({'error': 'Payment not successful', 'status': data.get('status')}), 400
        amount  = data.get('amount', 0)           # Naira
        days    = 90 if amount >= 14_000 else 30
        now_u   = datetime.now(timezone.utc)
        expires = now_u + timedelta(days=days)
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503
            cur.execute(
                'UPDATE users SET is_premium=TRUE,premium_until=%s WHERE id=%s',
                (expires, user['id'])
            )
        updated = _get_current_user()
        return jsonify({'success': True, 'user': updated or user})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config')
def app_config():
    return jsonify({'korapay_public_key': _KORAPAY_PUBLIC})


# ── Serve React build ─────────────────────────────────────────────────────────

DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist')
_INDEX_HTML = os.path.join(DIST, 'index.html')


def _serve_index():
    """Send index.html, with a helpful 503 if the dist hasn't been built yet."""
    if not os.path.isfile(_INDEX_HTML):
        return ('Frontend not built. Run: cd frontend && npm run build', 503)
    return send_from_directory(DIST, 'index.html')


@app.route('/')
def index():
    return _serve_index()


# Explicit SPA routes so hard-refresh / Korapay redirect never hits a 404
@app.route('/subscribe')
@app.route('/login')
@app.route('/register')
def spa_pages():
    return _serve_index()


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(os.path.join(DIST, 'assets'), filename)


@app.route('/<path:path>')
def serve(path):
    file_path = os.path.join(DIST, path)
    if os.path.isfile(file_path):
        return send_from_directory(DIST, path)
    return _serve_index()


def _refresh_caches():
    """Reset all prediction caches and rebuild WC fixtures every 6h."""
    global _WC_FIXTURES
    _INTL_TIPS_CACHE["ts"]  = 0.0
    _DAILY_TIPS_CACHE["ts"] = 0.0
    _CLUB_TIPS_CACHE["ts"]  = 0.0
    _BEST_BET_CACHE["ts"]   = 0.0
    _log.info('Caches cleared — fresh fetch on next request')
    try:
        _WC_FIXTURES = _build_wc_fixtures()
        _log.info('WC fixtures refreshed (%d matches)', len(_WC_FIXTURES))
    except Exception as _e:
        _log.warning('WC fixtures refresh failed: %s', _e)


# ── APScheduler — live update every 2 min + cache refresh every 6 h ──────────
try:
    from apscheduler.schedulers.background import BackgroundScheduler as _BgSched
    _scheduler = _BgSched(daemon=True)
    _scheduler.add_job(_update_live_scores, 'interval', minutes=2,
                       id='live_checker', misfire_grace_time=60)
    _scheduler.add_job(_check_pending_results, 'interval', minutes=30,
                       id='results_checker', misfire_grace_time=120)
    _scheduler.add_job(_check_basketball_results, 'interval', minutes=30,
                       id='basketball_checker', misfire_grace_time=120)
    _scheduler.add_job(_refresh_caches, 'interval', hours=6,
                       id='cache_refresh', misfire_grace_time=300)
    _scheduler.start()
    import atexit as _atexit
    _atexit.register(lambda: _scheduler.shutdown(wait=False))
    _log.info('APScheduler started — live/2 min, results/30 min, cache/6 h')
    try:
        _check_pending_results()
    except Exception as _cpr_err:
        _log.warning('Startup results check failed: %s', _cpr_err)
    try:
        _check_basketball_results()
    except Exception as _cbr_err:
        _log.warning('Startup basketball check failed: %s', _cbr_err)
except Exception as _sch_err:
    _log.warning('APScheduler not available: %s', _sch_err)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

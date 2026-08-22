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

from flask import Flask, jsonify, send_from_directory, request, g, session
from flask_cors import CORS
from contextlib import contextmanager
import os, sys, pickle, warnings, time, logging, math, difflib, unicodedata, json, threading, secrets, functools, re, hmac, hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

_log = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

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
_KORAPAY_SECRET    = os.environ.get('KORAPAY_SECRET_KEY', '')
_KORAPAY_PUBLIC    = os.environ.get('KORAPAY_PUBLIC_KEY', '')
_ONESIGNAL_APP_ID  = os.environ.get('ONESIGNAL_APP_ID', '15961974-7c73-430f-a7f4-23a69f27f7b8')
_ONESIGNAL_REST_KEY = os.environ.get('ONESIGNAL_REST_API_KEY', '')
_APP_URL           = os.environ.get('APP_URL', 'https://azpredicts.com')

logging.basicConfig(level=logging.INFO)
_log.info('KORAPAY_PUBLIC_KEY set: %s',   bool(_KORAPAY_PUBLIC))
_log.info('KORAPAY_SECRET_KEY set: %s',   bool(_KORAPAY_SECRET))
_log.info('ONESIGNAL configured: %s',     bool(_ONESIGNAL_APP_ID and _ONESIGNAL_REST_KEY))


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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    id            SERIAL PRIMARY KEY,
                    reference     VARCHAR(100) UNIQUE NOT NULL,
                    user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    plan          VARCHAR(20),
                    amount        NUMERIC(10,2),
                    currency      VARCHAR(10) DEFAULT 'NGN',
                    status        VARCHAR(20) NOT NULL DEFAULT 'initialized',
                    days_granted  INTEGER,
                    premium_until TIMESTAMPTZ,
                    source        VARCHAR(20),
                    raw_payload   JSONB,
                    created_at    TIMESTAMPTZ DEFAULT NOW(),
                    processed_at  TIMESTAMPTZ
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_payments_user ON payments(user_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_payments_status ON payments(status)"
            )
            for _col in [
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS match_status VARCHAR(20) DEFAULT 'SCHEDULED'",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS live_home_score INTEGER",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS live_away_score INTEGER",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS live_minute VARCHAR(10)",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS live_updated_at TIMESTAMPTZ",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_corners INTEGER",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS sport VARCHAR(20) DEFAULT 'football'",
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS prediction_payload JSONB",
            ]:
                cur.execute(_col)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analytics_logs (
                    id               SERIAL PRIMARY KEY,
                    timestamp        TIMESTAMPTZ DEFAULT NOW(),
                    endpoint         VARCHAR(300),
                    method           VARCHAR(10),
                    ip_address       VARCHAR(45),
                    country          VARCHAR(100),
                    user_id          INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    user_agent       TEXT,
                    device_type      VARCHAR(10),
                    response_time_ms INTEGER,
                    status_code      INTEGER
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_analytics_ts ON analytics_logs(timestamp)"
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS odds_tips (
                    id              SERIAL PRIMARY KEY,
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    date            DATE NOT NULL,
                    time            VARCHAR(10),
                    league          VARCHAR(200),
                    team_home       VARCHAR(100) NOT NULL,
                    team_away       VARCHAR(100) NOT NULL,
                    score           VARCHAR(20),
                    tip_description VARCHAR(300),
                    odd_value       NUMERIC(6,2),
                    tier            VARCHAR(10) NOT NULL DEFAULT 'free'
                                        CHECK (tier IN ('free','vip')),
                    status          VARCHAR(10) NOT NULL DEFAULT 'pending'
                                        CHECK (status IN ('pending','won','lost'))
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_odds_tips_date ON odds_tips(date, tier)"
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS saved_predictions (
                    id            SERIAL PRIMARY KEY,
                    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
                    status        VARCHAR(20) DEFAULT 'pending'
                                      CHECK (status IN ('pending','won','lost')),
                    match_result  TEXT,
                    saved_at      TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, prediction_id)
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_saved_predictions_user_id ON saved_predictions(user_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_saved_predictions_status ON saved_predictions(status)"
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_notifications (
                    id                SERIAL PRIMARY KEY,
                    title             VARCHAR(255) NOT NULL,
                    message           TEXT NOT NULL,
                    recipient_type    VARCHAR(20) NOT NULL,
                    recipient_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    sent_count        INTEGER DEFAULT 0,
                    sent_at           TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_admin_notifs_sent ON admin_notifications(sent_at DESC)"
            )
            # ── Notification tables ───────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS notification_tokens (
                    id                  SERIAL PRIMARY KEY,
                    user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    onesignal_player_id VARCHAR(128) NOT NULL,
                    created_at          TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, onesignal_player_id)
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_notif_tokens_user ON notification_tokens(user_id)"
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS won_predictions_notified (
                    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
                    notified_at   TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS premium_upsell_notifications (
                    id      SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    sent_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_premium_upsell_user ON premium_upsell_notifications(user_id, sent_at)"
            )
            # Anonymous push subscribers (no account required — claimed on login)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS anonymous_notification_tokens (
                    id         SERIAL PRIMARY KEY,
                    player_id  VARCHAR(128) UNIQUE NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # ── Extend saved_predictions with notification flags ───────────────
            for _col in [
                "ALTER TABLE saved_predictions ADD COLUMN IF NOT EXISTS match_start_notification_sent BOOLEAN DEFAULT FALSE",
                "ALTER TABLE saved_predictions ADD COLUMN IF NOT EXISTS result_notification_sent BOOLEAN DEFAULT FALSE",
                "ALTER TABLE saved_predictions ADD COLUMN IF NOT EXISTS result_notification_sent_at TIMESTAMPTZ",
            ]:
                cur.execute(_col)
        print('DB init: all tables ready ✓', flush=True)
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
    # Build a serialisable payload of all probabilities so the terminal can
    # read back the exact same values the website computed.
    import json as _json
    _payload = _json.dumps({
        'home_win':      round(float(tip.get('home_win', 0) or 0), 4),
        'draw':          round(float(tip.get('draw', 0) or 0), 4),
        'away_win':      round(float(tip.get('away_win', 0) or 0), 4),
        'over_goals':    round(float(tip.get('over_goals', 0.5) or 0.5), 4),
        'btts':          round(float(tip.get('btts', 0.48) or 0.48), 4),
        'over_corners':  round(float(tip.get('over_corners', 0.52) or 0.52), 4),
        'best_bet':      (tip.get('best_bet') or {}).get('label', pw),
        'best_bet_conf': round((tip.get('best_bet') or {}).get('confidence', 0) or 0, 4),
        'resolved_home': tip.get('resolved_home', home),
        'resolved_away': tip.get('resolved_away', away),
        'sport':         'football',
    })
    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute("""
                    INSERT INTO predictions
                        (match_id, home_team, away_team, match_date, match_time,
                         competition, predicted_winner, predicted_goals,
                         predicted_btts, predicted_corners, result_status,
                         kickoff_utc, prediction_payload)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'PENDING',%s,%s::jsonb)
                    ON CONFLICT (match_id) DO UPDATE
                        SET predicted_winner   = EXCLUDED.predicted_winner,
                            predicted_goals    = EXCLUDED.predicted_goals,
                            predicted_btts     = EXCLUDED.predicted_btts,
                            predicted_corners  = EXCLUDED.predicted_corners,
                            kickoff_utc        = EXCLUDED.kickoff_utc,
                            prediction_payload = EXCLUDED.prediction_payload
                        WHERE predictions.result_status = 'PENDING'
                """, (mid, home, away, match_date, mtime, comp, pw,
                      pg_str, btts_str, cor_str, kickoff, _payload))
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


def _compute_best_bet(tip):
    """Return best_bet dict whose label+confidence is the highest-probability market."""
    result = tip.get('result') or {}
    ph  = float(result.get('home', 0) or 0)
    pd_ = float(result.get('draw', 0) or 0)
    pa  = float(result.get('away', 0) or 0)
    pg  = float(tip.get('over_goals', 0.5) or 0.5)
    bt  = float(tip.get('btts', 0.48) or 0.48)
    co  = float(tip.get('over_corners', 0.52) or 0.52)
    home = tip.get('home_team') or tip.get('home') or ''
    away = tip.get('away_team') or tip.get('away') or ''

    candidates = []
    if ph:  candidates.append((ph,  f"{home} to Win"))
    if pd_: candidates.append((pd_, 'Draw'))
    if pa:  candidates.append((pa,  f"{away} to Win"))
    goals_pct = pg if pg >= 0.5 else (1.0 - pg)
    candidates.append((goals_pct, 'Over 2.5 Goals' if pg >= 0.5 else 'Under 2.5 Goals'))
    btts_pct = bt if bt >= 0.5 else (1.0 - bt)
    candidates.append((btts_pct, 'BTTS Yes' if bt >= 0.5 else 'BTTS No'))
    cor_pct = co if co >= 0.5 else (1.0 - co)
    candidates.append((cor_pct, 'Over 9.5 Corners' if co >= 0.5 else 'Under 9.5 Corners'))

    if not candidates:
        return tip.get('best_bet') or {}
    best_prob, best_label = max(candidates, key=lambda x: x[0])
    return {'label': best_label, 'confidence': round(best_prob, 4)}


def _save_basketball_prediction(game, pred, sport):
    """Save an NBA or WNBA prediction to history."""
    import json as _json
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

    # Store complete prediction so terminal can read back exact same values.
    _payload = _json.dumps({
        'home_win_pct':   pred.get('home_win_pct'),
        'away_win_pct':   pred.get('away_win_pct'),
        'predicted_winner': pred.get('predicted_winner'),
        'win_probability':  pred.get('win_probability'),
        'over_pct':         pred.get('over_pct'),
        'under_pct':        pred.get('under_pct'),
        'predicted_ou':     pred.get('predicted_ou'),
        'ou_probability':   pred.get('ou_probability'),
        'best_bet':         pred.get('best_bet'),
        'best_bet_type':    pred.get('best_bet_type'),
        'sport':            sport,
    })

    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute("""
                    INSERT INTO predictions
                        (match_id, home_team, away_team, match_date, match_time,
                         competition, predicted_winner, predicted_goals,
                         result_status, kickoff_utc, sport, prediction_payload)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'PENDING',%s,%s,%s::jsonb)
                    ON CONFLICT (match_id) DO UPDATE
                        SET predicted_winner   = EXCLUDED.predicted_winner,
                            predicted_goals    = EXCLUDED.predicted_goals,
                            kickoff_utc        = EXCLUDED.kickoff_utc,
                            prediction_payload = EXCLUDED.prediction_payload
                        WHERE predictions.result_status = 'PENDING'
                """, (mid, home, away, date, time_, comp, pw, pg, kickoff, sport, _payload))
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
                           COALESCE(sport, 'football') AS sport,
                           live_home_score, live_away_score, live_minute,
                           COALESCE(match_status, 'SCHEDULED') AS match_status
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


def _sync_saved_statuses(cur):
    """Propagate WON/LOST from predictions → saved_predictions (called after result updates)."""
    try:
        cur.execute("""
            UPDATE saved_predictions sp
               SET status       = CASE WHEN p.result_status = 'WON' THEN 'won' ELSE 'lost' END,
                   match_result = CONCAT(p.actual_home_score, '-', p.actual_away_score)
              FROM predictions p
             WHERE sp.prediction_id = p.id
               AND p.result_status IN ('WON', 'LOST')
               AND sp.status = 'pending'
        """)
    except Exception as e:
        _log.warning('_sync_saved_statuses: %s', e)


# ── OneSignal push notifications ──────────────────────────────────────────────

def _onesignal_send(payload: dict):
    """POST a notification to OneSignal's REST API.
    Returns (True, response_body) on success, (False, error_str) on failure."""
    if not _ONESIGNAL_REST_KEY:
        return False, 'ONESIGNAL_REST_API_KEY not set'
    try:
        import requests as _rq
        r = _rq.post(
            'https://onesignal.com/api/v1/notifications',
            headers={
                'Authorization': f'Basic {_ONESIGNAL_REST_KEY}',
                'Content-Type': 'application/json',
            },
            json={'app_id': _ONESIGNAL_APP_ID, **payload},
            timeout=10,
        )
        body = r.text[:500]
        if not r.ok:
            _log.warning('OneSignal %s: %s', r.status_code, body)
            return False, f'OneSignal {r.status_code}: {body}'
        _log.info('OneSignal sent OK: %s', body[:200])
        return True, body
    except Exception as e:
        _log.warning('OneSignal send error: %s', e)
        return False, str(e)


def _notify_kickoff():
    """Every 10 min: alert users whose saved bets kick off in ~15 minutes."""
    if not _ONESIGNAL_APP_ID:
        return
    try:
        with _db() as (conn, cur):
            if conn is None:
                return
            cur.execute("""
                SELECT sp.id, p.home_team, p.away_team, p.predicted_winner,
                       array_agg(nt.onesignal_player_id) AS player_ids
                  FROM saved_predictions sp
                  JOIN predictions p ON sp.prediction_id = p.id
                  JOIN notification_tokens nt ON sp.user_id = nt.user_id
                 WHERE sp.match_start_notification_sent = FALSE
                   AND sp.status = 'pending'
                   AND p.kickoff_utc IS NOT NULL
                   AND p.kickoff_utc::timestamptz
                       BETWEEN NOW() + INTERVAL '10 min'
                           AND NOW() + INTERVAL '20 min'
                 GROUP BY sp.id, p.home_team, p.away_team, p.predicted_winner
            """)
            rows = cur.fetchall()
            sent_ids = []
            for sp_id, home, away, pred_winner, player_ids in rows:
                bet = pred_winner or 'your saved bet'
                ok, _ = _onesignal_send({
                    'include_player_ids': player_ids,
                    'headings': {'en': '⚽ Kick-off in 15 minutes!'},
                    'contents': {'en': f'{home} vs {away} — {bet} starts soon'},
                    'url':      f'{_APP_URL}/saved',
                })
                if ok:
                    sent_ids.append(sp_id)
            if sent_ids:
                cur.execute(
                    'UPDATE saved_predictions SET match_start_notification_sent=TRUE WHERE id=ANY(%s)',
                    (sent_ids,)
                )
                _log.info('Kickoff alerts sent: %d', len(sent_ids))
    except Exception as e:
        _log.warning('_notify_kickoff: %s', e)


def _notify_results():
    """Every 5 min: tell users when their saved bets resolve (won or lost)."""
    if not _ONESIGNAL_APP_ID:
        return
    try:
        with _db() as (conn, cur):
            if conn is None:
                return
            cur.execute("""
                SELECT sp.id, sp.status,
                       p.home_team, p.away_team, p.predicted_winner,
                       p.actual_home_score, p.actual_away_score,
                       array_agg(nt.onesignal_player_id) AS player_ids
                  FROM saved_predictions sp
                  JOIN predictions p ON sp.prediction_id = p.id
                  JOIN notification_tokens nt ON sp.user_id = nt.user_id
                 WHERE sp.result_notification_sent = FALSE
                   AND sp.status IN ('won', 'lost')
                 GROUP BY sp.id, sp.status, p.home_team, p.away_team,
                          p.predicted_winner, p.actual_home_score, p.actual_away_score
            """)
            rows = cur.fetchall()
            sent_ids = []
            now = datetime.now(timezone.utc)
            for sp_id, status, home, away, pred_winner, hs, as_, player_ids in rows:
                icon    = '✅' if status == 'won' else '❌'
                outcome = 'WON' if status == 'won' else 'Lost'
                score   = f' ({hs}–{as_})' if hs is not None and as_ is not None else ''
                bet     = pred_winner or f'{home} vs {away}'
                ok, _ = _onesignal_send({
                    'include_player_ids': player_ids,
                    'headings': {'en': f'{icon} Prediction result'},
                    'contents': {'en': f'{home} vs {away}{score} — {bet} {outcome}'},
                    'url':      f'{_APP_URL}/saved',
                })
                if ok:
                    sent_ids.append(sp_id)
            if sent_ids:
                cur.execute(
                    'UPDATE saved_predictions SET result_notification_sent=TRUE, result_notification_sent_at=%s WHERE id=ANY(%s)',
                    (now, sent_ids)
                )
                _log.info('Result notifications sent: %d', len(sent_ids))
    except Exception as e:
        _log.warning('_notify_results: %s', e)


def _notify_won_predictions():
    """Every 10 min: broadcast newly-resolved winning predictions to all subscribers."""
    if not _ONESIGNAL_APP_ID:
        return
    try:
        with _db() as (conn, cur):
            if conn is None:
                return
            cur.execute("""
                SELECT p.id, p.home_team, p.away_team, p.predicted_winner,
                       COALESCE((p.prediction_payload->>'best_bet_conf')::float, 0) AS confidence,
                       COALESCE(p.sport, 'football') AS sport
                  FROM predictions p
                 WHERE p.result_status = 'WON'
                   AND NOT EXISTS (
                       SELECT 1 FROM won_predictions_notified wn
                        WHERE wn.prediction_id = p.id
                   )
                 ORDER BY p.created_at DESC
                 LIMIT 20
            """)
            preds = cur.fetchall()
            if not preds:
                return
            notified_ids = []
            now = datetime.now(timezone.utc)
            for pred_id, home, away, pred_winner, confidence, sport in preds:
                icon     = '🏀' if sport in ('nba', 'wnba') else '⚽'
                conf_str = f' ({confidence*100:.0f}% confidence)' if confidence > 0 else ''
                bet      = pred_winner or 'Prediction'
                # Broadcast to all OneSignal subscribers — includes anonymous
                # visitors who allowed push but never created an account.
                ok, _ = _onesignal_send({
                    'included_segments': ['All'],
                    'headings': {'en': f'✅ {icon} {home} vs {away}'},
                    'contents': {'en': f'{bet} WON{conf_str}'},
                    'url':      _APP_URL,
                })
                if ok:
                    notified_ids.append(pred_id)
            if notified_ids:
                cur.executemany(
                    'INSERT INTO won_predictions_notified (prediction_id, notified_at) VALUES (%s, %s) ON CONFLICT DO NOTHING',
                    [(pid, now) for pid in notified_ids]
                )
                _log.info('Won-prediction broadcasts sent: %d', len(notified_ids))
    except Exception as e:
        _log.warning('_notify_won_predictions: %s', e)


def _send_premium_upsell():
    """Daily at 9 AM UTC: push an upsell notification to free-tier users only."""
    if not _ONESIGNAL_APP_ID:
        return
    try:
        with _db() as (conn, cur):
            if conn is None:
                return
            cur.execute("""
                SELECT u.id, array_agg(DISTINCT nt.onesignal_player_id) AS player_ids
                  FROM users u
                  JOIN notification_tokens nt ON u.id = nt.user_id
                 WHERE (u.is_premium = FALSE
                        OR u.premium_until IS NULL
                        OR u.premium_until < NOW())
                   AND NOT EXISTS (
                       SELECT 1 FROM premium_upsell_notifications pun
                        WHERE pun.user_id = u.id
                          AND DATE(pun.sent_at AT TIME ZONE 'UTC') = CURRENT_DATE
                   )
                 GROUP BY u.id
            """)
            rows = cur.fetchall()
            user_ids = [r[0] for r in rows]
            known_ids = [pid for _, pids in rows for pid in pids]

            # Anonymous subscribers have no account → effectively free
            cur.execute('SELECT player_id FROM anonymous_notification_tokens')
            anon_ids = [r[0] for r in cur.fetchall()]

            all_player_ids = list({*known_ids, *anon_ids})
            if not all_player_ids:
                _log.info('Premium upsell: no eligible devices today')
                return
            # OneSignal limit: 2000 player IDs per call
            for i in range(0, len(all_player_ids), 2000):
                _onesignal_send({
                    'include_player_ids': all_player_ids[i:i+2000],
                    'headings': {'en': '👑 Unlock VIP Predictions'},
                    'contents': {'en': 'Get Daily Accumulator, VIP Tips & match alerts — Upgrade now'},
                    'url':      f'{_APP_URL}/subscribe',
                })
            now = datetime.now(timezone.utc)
            cur.executemany(
                'INSERT INTO premium_upsell_notifications (user_id, sent_at) VALUES (%s, %s)',
                [(uid, now) for uid in user_ids]
            )
            _log.info('Premium upsell sent to %d free users', len(user_ids))
    except Exception as e:
        _log.warning('_send_premium_upsell: %s', e)


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


# ── Analytics middleware ───────────────────────────────────────────────────────

_ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '')
_IP_GEO_CACHE: dict = {}
_IP_GEO_LOCK = threading.Lock()


def _get_country(ip: str) -> str:
    if not ip or ip in ('127.0.0.1', '::1'):
        return 'Local'
    now = time.time()
    with _IP_GEO_LOCK:
        entry = _IP_GEO_CACHE.get(ip)
        if entry and now - entry[1] < 3600:
            return entry[0]
    try:
        import requests as _rq
        r = _rq.get(f'http://ip-api.com/json/{ip}?fields=country', timeout=2)
        country = r.json().get('country', 'Unknown') if r.status_code == 200 else 'Unknown'
    except Exception:
        country = 'Unknown'
    with _IP_GEO_LOCK:
        _IP_GEO_CACHE[ip] = (country, now)
        if len(_IP_GEO_CACHE) > 4000:
            oldest = sorted(_IP_GEO_CACHE, key=lambda k: _IP_GEO_CACHE[k][1])[:1000]
            for k in oldest:
                del _IP_GEO_CACHE[k]
    return country


def _detect_device(ua: str) -> str:
    if re.search(r'mobile|android|iphone|ipad|ipod|tablet', (ua or '').lower()):
        return 'mobile'
    return 'desktop'


def _do_log_analytics(endpoint, method, ip, ua, user_id, status_code, elapsed_ms):
    if not _DB_CONN_URL:
        return
    try:
        country     = _get_country(ip)
        device_type = _detect_device(ua)
        with _db() as (conn, cur):
            if conn is None:
                return
            cur.execute(
                """INSERT INTO analytics_logs
                   (endpoint, method, ip_address, country, user_id,
                    user_agent, device_type, response_time_ms, status_code)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (endpoint, method, ip, country, user_id,
                 ua[:500], device_type, elapsed_ms, status_code),
            )
    except Exception as e:
        _log.debug('analytics insert: %s', e)


@app.before_request
def _analytics_before():
    g._req_start = time.time()
    g._analytics_user_id = None
    if request.headers.get('Authorization', '').startswith('Bearer '):
        try:
            u = _get_current_user()
            if u:
                g._analytics_user_id = u['id']
        except Exception:
            pass


@app.after_request
def _analytics_after(response):
    try:
        path = request.path
        if path.startswith('/assets/') or path.startswith('/static/'):
            return response
        elapsed_ms = round((time.time() - getattr(g, '_req_start', time.time())) * 1000)
        ip  = (request.headers.get('X-Forwarded-For', '') or request.remote_addr or '').split(',')[0].strip()
        ua  = request.headers.get('User-Agent', '')
        uid = getattr(g, '_analytics_user_id', None)
        threading.Thread(
            target=_do_log_analytics,
            args=(path, request.method, ip, ua, uid, response.status_code, elapsed_ms),
            daemon=True,
        ).start()
    except Exception as e:
        _log.debug('after_request analytics: %s', e)
    return response


def _require_admin(f):
    @functools.wraps(f)
    def _wrapper(*args, **kwargs):
        if not _ADMIN_PASSWORD:
            return jsonify({'error': 'Admin not configured — set ADMIN_PASSWORD env var'}), 503
        pw = (request.headers.get('X-Admin-Password', '')
              or request.args.get('admin_pw', '')
              or (request.get_json(silent=True) or {}).get('password', ''))
        if pw != _ADMIN_PASSWORD:
            return jsonify({'error': 'Unauthorized'}), 403
        session['az_admin'] = True   # stamp session so /admin/analytics can check it
        return f(*args, **kwargs)
    return _wrapper


def _require_admin_session(f):
    """Protects routes that browsers navigate to directly — checks the session
    cookie set by any successful _require_admin call, not a URL parameter."""
    @functools.wraps(f)
    def _wrapper(*args, **kwargs):
        if not session.get('az_admin'):
            return jsonify({'error': 'Unauthorized — open the admin panel first'}), 403
        return f(*args, **kwargs)
    return _wrapper


@app.route('/admin/analytics')
@_require_admin_session
def admin_analytics():
    if not _DB_CONN_URL:
        return jsonify({'error': 'Database not configured'}), 503
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB connection failed'}), 503

            now_u  = datetime.now(timezone.utc)
            t_day  = now_u.replace(hour=0, minute=0, second=0, microsecond=0)
            t_week = t_day - timedelta(days=t_day.weekday())
            t_mon  = t_day.replace(day=1)

            cur.execute("SELECT COUNT(*) FROM analytics_logs WHERE timestamp >= %s", (t_day,))
            v_today = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM analytics_logs WHERE timestamp >= %s", (t_week,))
            v_week = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM analytics_logs WHERE timestamp >= %s", (t_mon,))
            v_month = cur.fetchone()[0]

            cur.execute("""
                SELECT endpoint, COUNT(*) v FROM analytics_logs
                WHERE timestamp >= %s
                GROUP BY endpoint ORDER BY v DESC LIMIT 10
            """, (t_mon,))
            top_pages = [{'endpoint': r[0], 'visits': r[1]} for r in cur.fetchall()]

            cur.execute("""
                SELECT
                  COUNT(DISTINCT CASE WHEN u.is_premium IS TRUE  THEN al.user_id END) premium,
                  COUNT(DISTINCT CASE WHEN u.is_premium IS FALSE THEN al.user_id END) free_u,
                  COUNT(CASE WHEN al.user_id IS NULL THEN 1 END)                     anon
                FROM analytics_logs al
                LEFT JOIN users u ON al.user_id = u.id
                WHERE al.timestamp >= %s
            """, (t_mon,))
            row = cur.fetchone()
            user_breakdown = {'premium': row[0], 'free': row[1], 'anonymous': row[2]}

            cur.execute("""
                SELECT country, COUNT(*) v FROM analytics_logs
                WHERE timestamp >= %s AND country NOT IN ('Unknown','Local')
                GROUP BY country ORDER BY v DESC LIMIT 15
            """, (t_mon,))
            countries = [{'country': r[0], 'visits': r[1]} for r in cur.fetchall()]

            cur.execute("""
                SELECT EXTRACT(HOUR FROM timestamp AT TIME ZONE 'UTC')::int h, COUNT(*) v
                FROM analytics_logs WHERE timestamp >= %s
                GROUP BY h ORDER BY h
            """, (t_mon,))
            peak_hours = [{'hour': r[0], 'visits': r[1]} for r in cur.fetchall()]

        return jsonify({
            'visits':         {'today': v_today, 'week': v_week, 'month': v_month},
            'top_pages':      top_pages,
            'user_breakdown': user_breakdown,
            'countries':      countries,
            'peak_hours':     peak_hours,
        })
    except Exception as e:
        _log.exception('admin_analytics: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/admin/subscribers')
@_require_admin
def admin_subscribers():
    """Subscriber analytics: counts, growth, recent list."""
    if not _DB_CONN_URL:
        return jsonify({'error': 'Database not configured'}), 503
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB connection failed'}), 503

            now_u  = datetime.now(timezone.utc)
            t_week = now_u - timedelta(days=7)
            t_mon  = now_u.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            # ── Totals ────────────────────────────────────────────────────────
            cur.execute("SELECT COUNT(*) FROM users")
            total = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM users WHERE is_premium IS TRUE")
            premium = cur.fetchone()[0]

            free = total - premium
            conv = round(premium / total * 100, 1) if total else 0.0

            cur.execute("SELECT COUNT(*) FROM users WHERE created_at >= %s", (t_week,))
            new_week = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM users WHERE created_at >= %s", (t_mon,))
            new_month = cur.fetchone()[0]

            # ── Recent subscribers (last 30) ──────────────────────────────────
            cur.execute("""
                SELECT u.id, u.name, u.email, u.is_premium, u.created_at,
                       MAX(s.created_at) AS last_login
                FROM users u
                LEFT JOIN sessions s ON s.user_id = u.id
                GROUP BY u.id, u.name, u.email, u.is_premium, u.created_at
                ORDER BY u.created_at DESC
                LIMIT 30
            """)
            recent = []
            for r in cur.fetchall():
                recent.append({
                    'id':         r[0],
                    'name':       r[1],
                    'email':      r[2],
                    'is_premium': bool(r[3]),
                    'joined':     r[4].isoformat() if r[4] else None,
                    'last_login': r[5].isoformat() if r[5] else None,
                })

            # ── Daily growth — last 60 days ───────────────────────────────────
            t_60 = now_u - timedelta(days=60)
            cur.execute("""
                SELECT DATE(created_at AT TIME ZONE 'UTC') AS d, COUNT(*) AS n
                FROM users
                WHERE created_at >= %s
                GROUP BY d
                ORDER BY d
            """, (t_60,))
            daily_rows = cur.fetchall()

            # Build cumulative series
            cur.execute("SELECT COUNT(*) FROM users WHERE created_at < %s", (t_60,))
            base = cur.fetchone()[0]
            growth = []
            running = base
            for row in daily_rows:
                running += row[1]
                growth.append({
                    'date':       str(row[0]),
                    'new_users':  row[1],
                    'cumulative': running,
                })

        return jsonify({
            'totals': {
                'total':           total,
                'free':            free,
                'premium':         premium,
                'conversion_rate': conv,
            },
            'new_this_week':  new_week,
            'new_this_month': new_month,
            'recent':         recent,
            'growth':         growth,
        })
    except Exception as e:
        _log.exception('admin_subscribers: %s', e)
        return jsonify({'error': str(e)}), 500


# ── Odds Tips — helpers ───────────────────────────────────────────────────────

def _tip_to_dict(row):
    return {
        'id':              row[0],
        'created_at':      row[1].isoformat() if row[1] else None,
        'date':            str(row[2]) if row[2] else None,
        'time':            row[3],
        'league':          row[4],
        'team_home':       row[5],
        'team_away':       row[6],
        'score':           row[7],
        'tip_description': row[8],
        'odd_value':       float(row[9]) if row[9] is not None else None,
        'tier':            row[10],
        'status':          row[11],
    }

_TIP_COLS = """id, created_at, date, time, league, team_home, team_away,
               score, tip_description, odd_value, tier, status"""


# ── Odds Tips — admin CRUD ────────────────────────────────────────────────────

@app.route('/admin/odds-tips', methods=['POST'])
@_require_admin
def admin_create_tip():
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503
    b = request.get_json(silent=True) or {}
    if not b.get('team_home') or not b.get('team_away') or not b.get('date'):
        return jsonify({'error': 'date, team_home, team_away are required'}), 400
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503
            cur.execute("""
                INSERT INTO odds_tips
                  (date, time, league, team_home, team_away, score,
                   tip_description, odd_value, tier, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (
                b.get('date'), b.get('time'), b.get('league'),
                b['team_home'], b['team_away'], b.get('score'),
                b.get('tip_description'),
                float(b['odd_value']) if b.get('odd_value') else None,
                b.get('tier', 'free'), b.get('status', 'pending'),
            ))
            tip_id = cur.fetchone()[0]
        return jsonify({'id': tip_id}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/odds-tips/<int:tip_id>', methods=['PUT'])
@_require_admin
def admin_update_tip(tip_id):
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503
    b = request.get_json(silent=True) or {}
    allowed = ('date','time','league','team_home','team_away','score',
               'tip_description','odd_value','tier','status')
    sets, vals = [], []
    for field in allowed:
        if field in b:
            sets.append(f'{field} = %s')
            v = b[field]
            if field == 'odd_value' and v is not None:
                v = float(v)
            vals.append(v)
    if not sets:
        return jsonify({'error': 'Nothing to update'}), 400
    vals.append(tip_id)
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503
            cur.execute(f"UPDATE odds_tips SET {', '.join(sets)} WHERE id = %s", vals)
            if cur.rowcount == 0:
                return jsonify({'error': 'Tip not found'}), 404
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/odds-tips', methods=['GET'])
@_require_admin
def admin_list_tips():
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503
            cur.execute(f"""
                SELECT {_TIP_COLS} FROM odds_tips
                ORDER BY date DESC, time DESC NULLS LAST LIMIT 200
            """)
            tips = [_tip_to_dict(r) for r in cur.fetchall()]
        return jsonify({'tips': tips})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/odds-tips/<int:tip_id>', methods=['DELETE'])
@_require_admin
def admin_delete_tip(tip_id):
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503
            cur.execute('DELETE FROM odds_tips WHERE id = %s', (tip_id,))
            if cur.rowcount == 0:
                return jsonify({'error': 'Tip not found'}), 404
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Odds Tips — public ────────────────────────────────────────────────────────

@app.route('/api/odds-tips/free')
def odds_tips_free():
    if not _DB_CONN_URL:
        return jsonify({'tips': []})
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'tips': []})
            cur.execute(f"""
                SELECT {_TIP_COLS} FROM odds_tips
                WHERE tier = 'free' AND date = %s
                ORDER BY time ASC NULLS LAST, id ASC
            """, (today,))
            tips = [_tip_to_dict(r) for r in cur.fetchall()]
        return jsonify({'tips': tips, 'date': today})
    except Exception as e:
        _log.warning('odds_tips_free: %s', e)
        return jsonify({'tips': [], 'error': str(e)})


@app.route('/api/odds-tips/vip')
def odds_tips_vip():
    user  = _get_current_user()
    today = datetime.now(timezone.utc).date().isoformat()
    locked = not (user and user['is_premium'])
    if locked:
        return jsonify({'tips': [], 'locked': True,
                        'message': 'VIP tips require a premium subscription'})
    if not _DB_CONN_URL:
        return jsonify({'tips': [], 'locked': False})
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'tips': [], 'locked': False})
            cur.execute(f"""
                SELECT {_TIP_COLS} FROM odds_tips
                WHERE tier = 'vip' AND date = %s
                ORDER BY time ASC NULLS LAST, id ASC
            """, (today,))
            tips = [_tip_to_dict(r) for r in cur.fetchall()]
        return jsonify({'tips': tips, 'locked': False, 'date': today})
    except Exception as e:
        _log.warning('odds_tips_vip: %s', e)
        return jsonify({'tips': [], 'locked': False, 'error': str(e)})


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
    from fetch_fixtures import (
        get_daily_tips, get_club_tips, get_intl_tips,
        get_pl_tips, get_pl_next_match,
        get_laliga_tips, get_laliga_next_match,
    )
    print('fetch_fixtures imported OK', flush=True)
except Exception as _ff_err:
    print(f'WARNING: fetch_fixtures import failed: {_ff_err}', flush=True)
    def get_daily_tips(*a, **kw): return []
    def get_club_tips(*a, **kw):  return []
    def get_intl_tips(*a, **kw):  return []
    def get_pl_tips(*a, **kw):    return []
    def get_pl_next_match(*a, **kw): return None
    def get_laliga_tips(*a, **kw): return []
    def get_laliga_next_match(*a, **kw): return None

# ── Club feature builder ──────────────────────────────────────────────────────
# BASE_FEATURES order must match train.py exactly.
#
# No odds features (imp_h/imp_d/imp_a/book_margin) — there is no live
# pre-match odds feed for upcoming fixtures, only historical closing odds
# in Matches.csv. Feeding the model a constant fake odds triple for every
# live fixture flattened every prediction toward a near-uniform split
# regardless of the real matchup (2026-08-21 incident). Models were
# retrained on the odds-free feature set (see train.py BASE_FEATURES /
# retrain_no_odds.py) so training matches what serving can actually provide.

def _club_feat(h, a, elo_diff, form5_h, form5_a, league_enc):
    """Returns (base_X [22 features], corner_X [24 features])."""
    h_pts      = 3 * h['win'] + h['draw']
    a_pts      = 3 * a['win'] + a['draw']
    elo_prob_h = 1 / (1 + 10 ** (-elo_diff / 400))
    base = [
        h['gf'], h['ga'], h['gf'] - h['ga'],
        h.get('sot', 4.5),
        h['win'], h['draw'], h_pts, h.get('hwn', min(h['win'] + 0.12, 1.0)),
        a['gf'], a['ga'], a['gf'] - a['ga'],
        a.get('sot', 3.8),
        a['win'], a['draw'], a_pts, a.get('awn', max(a['win'] - 0.12, 0.0)),
        elo_diff, elo_prob_h,
        form5_h, form5_a, form5_h - form5_a,
        float(league_enc),
    ]
    corner_ext = base + [h.get('corners', 5.2), a.get('corners', 4.8)]
    return np.array([base]), np.array([corner_ext])


def _club_predict(h, a, elo_diff, form5_h, form5_a, league_code):
    enc    = LEAGUE_MAP.get(league_code, 10)
    X, Xc  = _club_feat(h, a, elo_diff, form5_h, form5_a, enc)

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
    def _norm(s):
        s = s.lower().strip()
        return _TEAM_ALIASES.get(s, s).lower()
    a, b = _norm(a), _norm(b)
    if a == b or a in b or b in a:
        return True
    # Word-order-independent match — providers disagree on order for multi-word
    # names (ESPN calls it "Congo DR", everywhere else uses "DR Congo").
    if set(a.split()) == set(b.split()):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.72


_TSDB_NOT_DONE = frozenset({
    'scheduled', 'not started', 'ns', 'tbd', 'in progress', 'halftime', 'half time',
    'live', 'postponed', 'cancelled', 'suspended', 'abandoned',
    # TheSportsDB also reports live matches with short clock-style codes —
    # confirmed live: a finished 6-0 match still read strStatus "2H" hours
    # after full time, and without these the in-progress score gets accepted
    # as final and frozen (the periodic checks only ever re-scan PENDING/LIVE rows).
    '1h', '2h', 'ht', 'et', 'p', 'pen',
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

    # Result — only for genuine match-result bets (team name or "draw").
    # When predicted_winner holds an O/U, BTTS, or corners label (because that
    # market had the highest confidence), skip this chip: it would always show
    # LOST because the team name is not present in "Over 170.5" etc., and the
    # correct market chip (goals/btts/corners) already covers the evaluation.
    pw = (pred.get('predicted_winner') or '').lower()
    _non_result_bet = any(x in pw for x in ('over', 'under', 'btts', 'corner'))
    if pw and not _non_result_bet:
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


def _resolve_result(pw, home, away, hs, as_, actual_corners=None):
    """Return 'WON' or 'LOST' for any bet type given actual score (and optionally corners)."""
    import re as _re
    if not pw:
        return 'LOST'
    pw_l  = pw.lower()
    total = hs + as_

    # Goals Over/Under (e.g. "Over 2.5 Goals", "Under 2.5 Goals")
    if 'goal' in pw_l:
        over   = 'over' in pw_l
        thresh = 3.5 if '3.5' in pw_l else 2.5
        return 'WON' if (over and total > thresh) or (not over and total <= thresh) else 'LOST'

    # Corners — evaluate if we have actual_corners, otherwise LOST (data unavailable)
    if 'corner' in pw_l:
        if actual_corners is not None:
            over   = 'over' in pw_l
            _nums  = _re.findall(r'\d+\.?\d*', pw_l)
            thresh = float(_nums[-1]) if _nums else 9.5
            return 'WON' if (over and actual_corners > thresh) or (not over and actual_corners <= thresh) else 'LOST'
        return 'LOST'

    # BTTS (Both Teams To Score) — "BTTS Yes" or "BTTS No"
    if 'btts' in pw_l:
        predicted_yes = 'yes' in pw_l
        actual_btts = hs > 0 and as_ > 0
        return 'WON' if predicted_yes == actual_btts else 'LOST'

    # Basketball / generic Over/Under (e.g. "Over 220.5", "Under 170.5")
    if 'over' in pw_l or 'under' in pw_l:
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


def _bball_search_dates(match_date, max_back=3, max_fwd=1):
    """
    Date candidates to query, nearest first. TheSportsDB sometimes corrects a
    game's scheduled date after we've already stored the original (e.g. a
    fixture stored as 2026-06-17 turned out to be played 2026-06-14) — without
    this, the exact-date lookup below would 404/empty forever and the
    prediction would stay PENDING until the multi-day stale fallback kicks in.
    """
    from datetime import date as _date, timedelta as _td
    try:
        d = _date.fromisoformat(str(match_date))
    except (ValueError, TypeError):
        return [str(match_date)]
    out = [str(match_date)]
    for delta in range(1, max_back + 1):
        out.append((d - _td(days=delta)).isoformat())
    for delta in range(1, max_fwd + 1):
        out.append((d + _td(days=delta)).isoformat())
    return out


def _tsdb_day_result(date_str, league_id, home, away):
    """Look up one day of TheSportsDB eventsday.php for a home/away match (either order)."""
    import requests as _req
    try:
        r = _req.get(
            'https://www.thesportsdb.com/api/v1/json/3/eventsday.php',
            params={'d': date_str, 'l': league_id},
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
            if _team_similar(eh, away) and _team_similar(ea, home):
                return int(float(as_r)), int(float(hs_r))
    except Exception as e:
        _log.warning('Basketball TSDB %s (%s): %s', league_id, date_str, e)
    return None


def _espn_bball_day_result(sport_slug, date_str, home, away):
    """Look up one day of an ESPN basketball scoreboard for a home/away match (either order)."""
    import requests as _req
    try:
        r = _req.get(
            f'https://site.api.espn.com/apis/site/v2/sports/basketball/{sport_slug}/scoreboard',
            params={'dates': date_str.replace('-', '')},
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
            if _team_similar(hn, away) and _team_similar(an, home):
                return int(as_), int(hs)
    except Exception as e:
        _log.warning('Basketball ESPN %s (%s): %s', sport_slug, date_str, e)
    return None


def _get_nba_game_result(home, away, match_date):
    """Fetch NBA game result from TheSportsDB → ESPN, scanning nearby dates."""
    for d in _bball_search_dates(match_date):
        res = _tsdb_day_result(d, '4387', home, away)  # covers playoffs
        if res:
            return res
        res = _espn_bball_day_result('nba', d, home, away)
        if res:
            return res
    return None


def _get_wnba_game_result(home, away, match_date):
    """Fetch WNBA game result from TheSportsDB → ESPN, scanning nearby dates."""
    for d in _bball_search_dates(match_date):
        res = _tsdb_day_result(d, '4328', home, away)
        if res:
            return res
        res = _espn_bball_day_result('wnba', d, home, away)
        if res:
            return res
    return None


def _check_basketball_results():
    """30-min job: fetch final scores for PENDING NBA/WNBA predictions."""
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
                    if kickoff + timedelta(hours=2) > now:
                        continue
                    # Games stuck PENDING for >3 days: mark LOST (result unrecoverable)
                    if now - kickoff > timedelta(days=3):
                        cur.execute("""
                            UPDATE predictions SET result_status='LOST', match_status='FINISHED'
                            WHERE match_id=%s
                        """, (mid,))
                        _log.info('Basketball stale PENDING → LOST: %s vs %s (%s)', home, away, match_date)
                        resolved += 1
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
                    _sync_saved_statuses(cur)
                    _BEST_BET_CACHE["data"] = None
                    _DAILY_TIPS_CACHE["ts"] = 0
        return resolved
    except Exception as e:
        _log.warning('Basketball results check failed: %s', e)
        return resolved


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
                    # Games stuck PENDING for >5 days: mark LOST (result unrecoverable)
                    if now - kickoff > timedelta(days=5):
                        cur.execute("""
                            UPDATE predictions SET result_status='LOST', match_status='FINISHED'
                            WHERE match_id=%s
                        """, (mid,))
                        _log.info('Stale PENDING → LOST: %s vs %s (%s)', home, away, match_date)
                        resolved += 1
                        continue
                    result = _get_result(home, away, match_date)
                    if result is None:
                        continue
                    hs, as_, corners = result
                    status  = _resolve_result(pw or '', home or '', away or '', hs, as_,
                                              actual_corners=corners)
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
                    _sync_saved_statuses(cur)
                    _DAILY_TIPS_CACHE["ts"] = 0
                    _BEST_BET_CACHE["data"] = None
                return resolved
    except Exception as e:
        _log.warning('DB check_pending failed, falling back to JSON: %s', e)

    # ── JSON fallback ──────────────────────────────────────────────────────────
    with _HISTORY_LOCK:
        data = _read_raw_history()
        changed = False
        json_resolved = 0
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
                pred['home_team'], pred['away_team'], hs, as_,
                actual_corners=corners)
            changed = True
            json_resolved += 1
            _log.info('JSON result: %s %d-%d %s → %s (corners=%s)',
                      pred['home_team'], hs, as_, pred['away_team'],
                      pred['result_status'], corners)
        if changed:
            _write_raw_history(data)
        return json_resolved


def _daily_results_cleanup():
    """Daily 2 AM UTC sweep — safety net behind the 30-min checkers above,
    in case a game's result lands late (provider delay, rate limit, etc.)."""
    _log.info('Daily results cleanup: running football + basketball checks…')
    football   = _check_pending_results() or 0
    basketball = _check_basketball_results() or 0
    bracket    = _wc_sync_bracket() or 0
    _log.info('Daily results cleanup done: %d football, %d basketball resolved, %d bracket slot(s) resolved',
               football, basketball, bracket)
    return {'football_resolved': football, 'basketball_resolved': basketball, 'bracket_resolved': bracket}

# ── ESPN live-score polling ───────────────────────────────────────────────────

_ESPN_LIVE_LEAGUES = [
    # FIFA World Cup 2026 (ongoing tournament)
    'fifa.world',
    # WC qualifying (all confederations)
    'fifa.worldq.conmebol', 'fifa.worldq.concacaf',
    'fifa.worldq.afc',      'fifa.worldq.caf',
    'fifa.worldq.uefa',     'fifa.worldq.ofc',
    # Internationals / friendlies
    'fifa.friendly',
    'uefa.friendly', 'conmebol.friendly',
    'concacaf.friendly', 'afc.friendly', 'caf.friendly',
    # Other international competitions
    'afc.cupqualification',
    'caf.nations',
    'concacaf.nations.league',
    'conmebol.america',      # Copa América
    'uefa.euro',             # UEFA European Championship
    # Domestic club leagues — matches _COMP_TO_LEAGUE in fetch_fixtures.py.
    # Added 2026-08-22: club prediction cards already carried a LIVE badge
    # (driven by football-data.org's own per-match `status`), but no score
    # ever appeared because this list only covered international football —
    # the shared live-score pipeline (_update_live_scores, /api/live-scores,
    # ResultsSection, _resolve_result) never actually ran for club matches.
    'eng.1',            # Premier League
    'eng.2',            # EFL Championship
    'esp.1',            # La Liga
    'ger.1',            # Bundesliga
    'ita.1',            # Serie A
    'fra.1',            # Ligue 1
    'por.1',            # Primeira Liga
    'ned.1',            # Eredivisie
    'bra.1',            # Brasileirão
    'uefa.champions', 'uefa.europa', 'uefa.europa.conf',
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
                corners = _espn_extract_corners(comp, hc, ac)
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
    'STATUS_IN_PROGRESS', 'STATUS_FIRST_HALF', 'STATUS_HALFTIME', 'STATUS_SECOND_HALF',
    'STATUS_EXTRA_TIME', 'STATUS_PENALTIES',
    'STATUS_FINAL', 'STATUS_FULL_TIME', 'STATUS_FULL_ET', 'STATUS_FULL_PEN',
})
_ESPN_DONE = frozenset({
    'STATUS_FINAL', 'STATUS_FULL_TIME', 'STATUS_FULL_ET', 'STATUS_FULL_PEN',
})


_ESPN_CORNER_KEYS = frozenset({'cornerkicks', 'corners', 'corner kicks', 'corner', 'woncorners'})


def _espn_extract_corners(comp, hc, ac):
    """
    ESPN stores corner counts in competitors[i].statistics (per-team) for
    dated/historical scoreboard calls — NOT in competitions[0].statistics.
    Sum both sides; fall back to the competition-level statistics block.
    """
    total = None
    for side in (hc, ac):
        for stat in (side.get('statistics') or []):
            name = (stat.get('name') or stat.get('abbreviation') or '').lower().strip()
            if name in _ESPN_CORNER_KEYS:
                try:
                    total = (total or 0) + int(
                        stat.get('displayValue') or stat.get('value') or '0'
                    )
                except (ValueError, TypeError):
                    pass
    if total is None:
        for stat in (comp.get('statistics') or []):
            if (stat.get('name') or '').lower().strip() in _ESPN_CORNER_KEYS:
                try:
                    total = int(stat.get('displayValue') or '0')
                except (ValueError, TypeError):
                    pass
                break
    return total


def _espn_search_date(home_team, away_team, date_compact):
    """Search one date across all ESPN soccer slugs; return (hs, as_, corners) or None."""
    import requests as _req
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
                corners = _espn_extract_corners(comp, hc, ac)
                if _team_similar(hn, home_team) and _team_similar(an, away_team):
                    return hs, as_, corners
                if _team_similar(hn, away_team) and _team_similar(an, home_team):
                    return as_, hs, corners
        except Exception as exc:
            _log.warning('ESPN dated result %s: %s', slug, exc)
    return None


def _fetch_espn_result_dated(home_team, away_team, match_date):
    """
    Query ESPN scoreboard for home_team vs away_team on match_date.
    Also retries the previous day to handle US evening games where the
    local date (used by ESPN) is one day behind the UTC kickoff date we store.
    """
    from datetime import date as _date, timedelta as _td
    try:
        d = _date.fromisoformat(str(match_date))
    except (ValueError, TypeError):
        d = None

    dates_to_try = [str(match_date).replace('-', '')]
    if d:
        prev = (d - _td(days=1)).strftime('%Y%m%d')
        if prev != dates_to_try[0]:
            dates_to_try.append(prev)

    for dc in dates_to_try:
        result = _espn_search_date(home_team, away_team, dc)
        if result is not None:
            return result
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
                        if kickoff + timedelta(hours=2) > now:
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
                            corners = ev.get('corners')
                            status  = _resolve_result(pw or '', home or '', away or '', hs, as_,
                                                      actual_corners=corners)
                            cur.execute("""
                                UPDATE predictions
                                SET actual_home_score=%s, actual_away_score=%s,
                                    actual_corners=%s,
                                    result_status=%s, match_status='FINISHED',
                                    live_home_score=%s, live_away_score=%s,
                                    live_updated_at=NOW()
                                WHERE match_id=%s
                            """, (hs, as_, corners, status, hs, as_, mid))
                            _log.info('FINISHED %s %d-%d %s → %s (corners=%s)',
                                      home, hs, as_, away, status, corners)
                        else:
                            cur.execute("""
                                UPDATE predictions
                                SET result_status='LIVE', match_status='LIVE',
                                    live_home_score=%s, live_away_score=%s,
                                    live_minute=%s,
                                    live_updated_at=NOW()
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
                            status = _resolve_result(pw or '', home or '', away or '', hs, as_,
                                                     actual_corners=corners)
                            cur.execute("""
                                UPDATE predictions
                                SET actual_home_score=%s, actual_away_score=%s,
                                    actual_corners=%s,
                                    result_status=%s, match_status='FINISHED'
                                WHERE match_id=%s
                            """, (hs, as_, corners, status, mid))
                            _log.info('Result %s %d-%d %s → %s (corners=%s)',
                                      home, hs, as_, away, status, corners)
                # This job resolves most PENDING/LIVE rows straight to WON/LOST
                # before the 30-min checkers ever see them (they only sync
                # saved_predictions when *they* find something to resolve), so
                # saved_predictions would otherwise stay 'pending' forever.
                _sync_saved_statuses(cur)
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

    # ── ROUND OF 32  (June 28 – July 4 UTC) ──────────────────────────────────
    {"id":"wcR32_73","group":"R32","home":"South Africa","away":"Canada",
     "home_code":"za","away_code":"ca",
     "date":"2026-06-28","time":"19:00","venue":"SoFi Stadium, Inglewood"},
    {"id":"wcR32_76","group":"R32","home":"Brazil","away":"Japan",
     "home_code":"br","away_code":"jp",
     "date":"2026-06-29","time":"17:00","venue":"NRG Stadium, Houston"},
    {"id":"wcR32_74","group":"R32","home":"Germany","away":"Paraguay",
     "home_code":"de","away_code":"py",
     "date":"2026-06-29","time":"20:30","venue":"Gillette Stadium, Foxborough"},
    {"id":"wcR32_75","group":"R32","home":"Netherlands","away":"Morocco",
     "home_code":"nl","away_code":"ma",
     "date":"2026-06-30","time":"01:00","venue":"Estadio BBVA, Guadalupe"},
    {"id":"wcR32_78","group":"R32","home":"Ivory Coast","away":"Norway",
     "home_code":"ci","away_code":"no",
     "date":"2026-06-30","time":"17:00","venue":"AT&T Stadium, Arlington"},
    {"id":"wcR32_77","group":"R32","home":"France","away":"Sweden",
     "home_code":"fr","away_code":"se",
     "date":"2026-06-30","time":"21:00","venue":"MetLife Stadium, East Rutherford"},
    {"id":"wcR32_79","group":"R32","home":"Mexico","away":"Ecuador",
     "home_code":"mx","away_code":"ec",
     "date":"2026-07-01","time":"01:00","venue":"Estadio Azteca, Mexico City"},
    {"id":"wcR32_80","group":"R32","home":"England","away":"DR Congo",
     "home_code":"gb-eng","away_code":"cd",
     "date":"2026-07-01","time":"16:00","venue":"Mercedes-Benz Stadium, Atlanta"},
    {"id":"wcR32_82","group":"R32","home":"Belgium","away":"Senegal",
     "home_code":"be","away_code":"sn",
     "date":"2026-07-01","time":"20:00","venue":"Lumen Field, Seattle"},
    {"id":"wcR32_81","group":"R32","home":"United States","away":"Bosnia and Herzegovina",
     "home_code":"us","away_code":"ba",
     "date":"2026-07-02","time":"00:00","venue":"Levi's Stadium, Santa Clara"},
    {"id":"wcR32_84","group":"R32","home":"Spain","away":"Austria",
     "home_code":"es","away_code":"at",
     "date":"2026-07-02","time":"19:00","venue":"SoFi Stadium, Inglewood"},
    {"id":"wcR32_83","group":"R32","home":"Portugal","away":"Croatia",
     "home_code":"pt","away_code":"hr",
     "date":"2026-07-02","time":"23:00","venue":"BMO Field, Toronto"},
    {"id":"wcR32_85","group":"R32","home":"Switzerland","away":"Algeria",
     "home_code":"ch","away_code":"dz",
     "date":"2026-07-03","time":"03:00","venue":"BC Place, Vancouver"},
    {"id":"wcR32_88","group":"R32","home":"Australia","away":"Egypt",
     "home_code":"au","away_code":"eg",
     "date":"2026-07-03","time":"18:00","venue":"AT&T Stadium, Arlington"},
    {"id":"wcR32_86","group":"R32","home":"Argentina","away":"Cape Verde",
     "home_code":"ar","away_code":"cv",
     "date":"2026-07-03","time":"22:00","venue":"Hard Rock Stadium, Miami Gardens"},
    {"id":"wcR32_87","group":"R32","home":"Colombia","away":"Ghana",
     "home_code":"co","away_code":"gh",
     "date":"2026-07-04","time":"01:30","venue":"Arrowhead Stadium, Kansas City"},

    # ── ROUND OF 16  (July 4 – 7 UTC) ────────────────────────────────────────
    {"id":"wcR16_90","group":"R16","home":"Canada","away":"TBD (NED/MAR)",
     "home_code":"ca","away_code":"tbd",
     "date":"2026-07-04","time":"17:00","venue":"NRG Stadium, Houston"},
    {"id":"wcR16_89","group":"R16","home":"TBD (GER/PAR)","away":"TBD (FRA/SWE)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-04","time":"21:00","venue":"Lincoln Financial Field, Philadelphia"},
    {"id":"wcR16_91","group":"R16","home":"TBD (BRA/JPN)","away":"TBD (CIV/NOR)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-05","time":"20:00","venue":"MetLife Stadium, East Rutherford"},
    {"id":"wcR16_92","group":"R16","home":"TBD (MEX/ECU)","away":"TBD (ENG/CGO)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-06","time":"00:00","venue":"Estadio Azteca, Mexico City"},
    {"id":"wcR16_93","group":"R16","home":"TBD (POR/CRO)","away":"TBD (ESP/AUT)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-06","time":"19:00","venue":"AT&T Stadium, Arlington"},
    {"id":"wcR16_94","group":"R16","home":"TBD (USA/BIH)","away":"TBD (BEL/SEN)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-07","time":"00:00","venue":"Lumen Field, Seattle"},
    {"id":"wcR16_95","group":"R16","home":"TBD (ARG/CPV)","away":"TBD (AUS/EGY)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-07","time":"16:00","venue":"Mercedes-Benz Stadium, Atlanta"},
    {"id":"wcR16_96","group":"R16","home":"TBD (SUI/ALG)","away":"TBD (COL/GHA)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-07","time":"20:00","venue":"BC Place, Vancouver"},

    # ── QUARTER-FINALS  (July 9 – 12 UTC) ────────────────────────────────────
    {"id":"wcQF_97","group":"QF","home":"TBD (QF1)","away":"TBD (QF2)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-09","time":"20:00","venue":"Gillette Stadium, Foxborough"},
    {"id":"wcQF_98","group":"QF","home":"TBD (QF3)","away":"TBD (QF4)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-10","time":"19:00","venue":"SoFi Stadium, Inglewood"},
    {"id":"wcQF_99","group":"QF","home":"TBD (QF5)","away":"TBD (QF6)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-11","time":"21:00","venue":"Hard Rock Stadium, Miami Gardens"},
    {"id":"wcQF_100","group":"QF","home":"TBD (QF7)","away":"TBD (QF8)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-12","time":"01:00","venue":"Arrowhead Stadium, Kansas City"},

    # ── SEMI-FINALS  (July 14 – 15) ──────────────────────────────────────────
    {"id":"wcSF_101","group":"SF","home":"TBD (SF1)","away":"TBD (SF2)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-14","time":"19:00","venue":"AT&T Stadium, Arlington"},
    {"id":"wcSF_102","group":"SF","home":"TBD (SF3)","away":"TBD (SF4)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-15","time":"19:00","venue":"Mercedes-Benz Stadium, Atlanta"},

    # ── THIRD PLACE  (July 18) ────────────────────────────────────────────────
    {"id":"wcTP_103","group":"3rd","home":"TBD (3rd1)","away":"TBD (3rd2)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-18","time":"21:00","venue":"Hard Rock Stadium, Miami Gardens"},

    # ── FINAL  (July 19) ──────────────────────────────────────────────────────
    {"id":"wcFinal","group":"Final","home":"TBD (FIN1)","away":"TBD (FIN2)",
     "home_code":"tbd","away_code":"tbd",
     "date":"2026-07-19","time":"19:00","venue":"MetLife Stadium, East Rutherford"},
]

def _build_wc_fixtures():
    fixtures = []
    for raw in WC_FIXTURES_RAW:
        home, away = raw['home'], raw['away']

        # Knockout placeholder — teams not yet determined
        if 'TBD' in home or 'TBD' in away:
            entry = {
                **raw,
                'utc_kickoff':  f"{raw['date']}T{raw['time']}:00Z",
                'result':       {'home': 0.33, 'draw': 0.34, 'away': 0.33},
                'over_goals':   0.45,
                'btts':         0.48,
                'over_corners': 0.52,
                'competition':  'FIFA World Cup 2026',
                'best_bet':     {'label': 'TBD', 'confidence': 0.0},
            }
            fixtures.append(entry)
            continue

        pred = _wc_predict(home, away)
        if pred is None:
            _log.error('No prediction for WC fixture %s vs %s — skipping',
                       home, away)
            continue

        ph, pd_, pa = pred['home_win'], pred['draw'], pred['away_win']
        pg  = pred['over_goals']

        entry = {
            **raw,
            'utc_kickoff':  f"{raw['date']}T{raw['time']}:00Z",
            'result':       {'home': ph, 'draw': pd_, 'away': pa},
            'over_goals':   pg,
            'btts':         pred.get('btts',         0.48),
            'over_corners': pred.get('over_corners', 0.52),
            'competition':  'FIFA World Cup 2026',
        }
        entry['best_bet'] = _compute_best_bet(entry)
        _save_prediction(entry)
        fixtures.append(entry)
    return fixtures


def _wc_is_resolved(fixture: dict) -> bool:
    """True once both bracket slots hold real, confirmed team names (no TBD placeholder)."""
    return 'TBD' not in fixture.get('home', '') and 'TBD' not in fixture.get('away', '')


# ── WC bracket auto-progression ───────────────────────────────────────────────
#
# ESPN's fifa.world scoreboard resolves a knockout fixture's real teams (e.g.
# "Brazil" instead of "TBD (BRA/JPN)") as soon as both feeder matches finish —
# often days before that fixture kicks off. Confirmed 2026-07-05: the R16 slot
# at MetLife Stadium (our wcR16_91, date 2026-07-05) already returned
# "Norway at Brazil" from ESPN hours ahead of kickoff, while still-undecided
# slots returned generic names like "Quarterfinal 1 Winner" / "Semifinal 2
# Loser" — so `_wc_espn_event_resolved` treats any "Winner"/"Loser"/"TBD" name
# as not yet known.
#
# Fixtures are matched to ESPN events by exact UTC kickoff instant (not venue
# name) — each knockout slot's kickoff time is pinned by the tournament
# schedule regardless of who ends up playing there, and venue names drift
# (e.g. ESPN lists Mexico City's stadium as "Estadio Banorte", we recorded it
# as "Estadio Azteca"). ESPN also buckets late-UTC kickoffs (evening US local
# time) under the previous calendar day, so both the fixture's date and the
# day before are queried — the same quirk `_fetch_espn_result_dated` already
# works around for score lookups.

def _wc_fetch_espn_day(date_str: str) -> list:
    """Return [{'kickoff': datetime, 'home', 'away'}] for fifa.world events
    on date_str (YYYY-MM-DD) and the UTC day before."""
    import requests as _req
    from datetime import date as _date_cls
    url = 'https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard'
    d = _date_cls.fromisoformat(date_str)
    date_codes = [date_str.replace('-', ''), (d - timedelta(days=1)).strftime('%Y%m%d')]
    out = []
    for dc in date_codes:
        try:
            r = _req.get(url, params={'dates': dc}, timeout=10)
            r.raise_for_status()
        except Exception:
            continue
        for ev in (r.json().get('events') or []):
            comp = (ev.get('competitions') or [{}])[0]
            competitors = comp.get('competitors') or []
            home_c = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away_c = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            if not home_c or not away_c:
                continue
            try:
                kickoff = datetime.fromisoformat(ev.get('date', '').replace('Z', '+00:00'))
            except Exception:
                continue
            out.append({
                'kickoff': kickoff,
                'home':    (home_c.get('team') or {}).get('displayName', ''),
                'away':    (away_c.get('team') or {}).get('displayName', ''),
            })
    return out


def _wc_espn_event_resolved(name: str) -> bool:
    """False for ESPN's own placeholders ('Quarterfinal 1 Winner', 'Semifinal 2 Loser', 'TBD', etc.)."""
    return bool(name) and not any(kw in name for kw in ('Winner', 'Loser', 'TBD'))


def _wc_sync_bracket():
    """
    Scheduler job: replace TBD knockout placeholders with the real qualified
    teams once ESPN's bracket confirms them, then rebuild WC fixtures so the
    trained WC model scores the newly-known matchup immediately (same model /
    accuracy as every other WC prediction — no generic placeholder stats).
    """
    global _WC_FIXTURES
    tbd = [f for f in WC_FIXTURES_RAW if not _wc_is_resolved(f)]
    if not tbd:
        return 0

    flag_map = {}
    for r in WC_FIXTURES_RAW:
        if r.get('home_code') and r['home_code'] != 'tbd':
            flag_map[r['home']] = r['home_code']
        if r.get('away_code') and r['away_code'] != 'tbd':
            flag_map[r['away']] = r['away_code']

    events_by_date = {}
    resolved_count = 0
    for raw in tbd:
        d = raw['date']
        if d not in events_by_date:
            try:
                events_by_date[d] = _wc_fetch_espn_day(d)
            except Exception as exc:
                _log.warning('WC bracket sync: ESPN fetch failed for %s: %s', d, exc)
                events_by_date[d] = []
        events = events_by_date[d]

        try:
            expected_kickoff = datetime.fromisoformat(f"{raw['date']}T{raw['time']}:00+00:00")
        except Exception:
            continue
        match = next(
            (e for e in events if abs((e['kickoff'] - expected_kickoff).total_seconds()) <= 300),
            None,
        )
        if match is None:
            continue

        new_home, new_away = match['home'], match['away']
        if not (_wc_espn_event_resolved(new_home) and _wc_espn_event_resolved(new_away)):
            continue
        if new_home == raw['home'] and new_away == raw['away']:
            continue

        _log.info('WC bracket resolved: %s → %s vs %s (was %s vs %s)',
                   raw['id'], new_home, new_away, raw['home'], raw['away'])
        raw['home'] = new_home
        raw['away'] = new_away
        raw['home_code'] = flag_map.get(new_home, 'tbd')
        raw['away_code'] = flag_map.get(new_away, 'tbd')
        resolved_count += 1

    if resolved_count:
        _WC_FIXTURES = _build_wc_fixtures()
        _DAILY_TIPS_CACHE['ts']  = 0
        _BEST_BET_CACHE['data'] = None
        _INTL_TIPS_CACHE['ts']  = 0
        _log.info('WC bracket sync: %d fixture(s) resolved, fixtures rebuilt', resolved_count)
    return resolved_count


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
_BBALL_CACHE_TTL = 60  # NBA/WNBA fixtures need to surface live scores promptly

_CLUB_TIPS_CACHE: dict  = {"data": None, "ts": 0.0}


@app.route('/api/club/tips')
def club_tips():
    now_ts = time.time()
    if _CLUB_TIPS_CACHE["data"] is None or (now_ts - _CLUB_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_club_tips(_club_predict)
            for t in tips:
                t['best_bet'] = _compute_best_bet(t)
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
    if today < _d(2026, 6, 29): return '2026-06-28'   # Round 3
    if today < _d(2026, 7,  5): return '2026-07-04'   # Round of 32
    if today < _d(2026, 7,  9): return '2026-07-07'   # Round of 16
    if today < _d(2026, 7, 14): return '2026-07-12'   # Quarter-finals
    if today < _d(2026, 7, 16): return '2026-07-15'   # Semi-finals
    return '2026-07-19'                                 # 3rd place + Final


@app.route('/api/worldcup/fixtures')
def wc_fixtures():
    now    = datetime.now(timezone.utc)
    done   = _completed_keys()
    cutoff = _wc_round_cutoff()
    upcoming = []
    for f in _WC_FIXTURES:
        if f['date'] > cutoff:
            continue
        if not _wc_is_resolved(f):
            continue  # hide until real teams are confirmed — no fabricated placeholder stats
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


_PL_TIPS_CACHE: dict = {"data": None, "ts": 0.0}


@app.route('/api/pl/fixtures')
def pl_fixtures():
    now_ts = time.time()
    if _PL_TIPS_CACHE["data"] is None or (now_ts - _PL_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_pl_tips(_club_predict)
            for t in tips:
                t['best_bet'] = _compute_best_bet(t)
                _save_prediction(t)
            _PL_TIPS_CACHE["data"] = tips
            _PL_TIPS_CACHE["ts"]   = now_ts
        except Exception as exc:
            app.logger.error("pl-fixtures fetch failed: %s", exc)
            if _PL_TIPS_CACHE["data"] is None:
                _PL_TIPS_CACHE["data"] = []
    done = _completed_keys()
    active = [t for t in (_PL_TIPS_CACHE["data"] or [])
              if _match_key(t.get('home_team',''), t.get('away_team',''),
                            (t.get('utc_kickoff','') or '')[:10]) not in done]
    return jsonify(active)


# Fallback used only if football-data.org is unreachable — the real 2026/27
# season opener (confirmed fixture, in case the live API is down).
_PL_COUNTDOWN_FALLBACK = {
    'home': 'Arsenal', 'away': 'Coventry City',
    'utc_kickoff': '2026-08-21T15:00:00+00:00',
    'venue': 'Emirates Stadium',
    'home_crest': '', 'away_crest': '',
}
_PL_COUNTDOWN_CACHE: dict = {"data": None, "ts": 0.0}
_PL_COUNTDOWN_TTL = 3600  # 1 hour — the fixture list rarely changes


@app.route('/api/pl/countdown')
def pl_countdown():
    now_ts = time.time()
    if _PL_COUNTDOWN_CACHE["data"] is None or (now_ts - _PL_COUNTDOWN_CACHE["ts"]) > _PL_COUNTDOWN_TTL:
        fixture = get_pl_next_match() or _PL_COUNTDOWN_FALLBACK
        _PL_COUNTDOWN_CACHE["data"] = fixture
        _PL_COUNTDOWN_CACHE["ts"]   = now_ts
    fixture = _PL_COUNTDOWN_CACHE["data"]
    now = datetime.now(timezone.utc)
    try:
        kickoff = datetime.fromisoformat(fixture['utc_kickoff'].replace('Z', '+00:00'))
    except Exception:
        kickoff = now
    seconds_remaining = max(0, int((kickoff - now).total_seconds()))
    return jsonify({'fixture': fixture, 'seconds_remaining': seconds_remaining})


_LALIGA_TIPS_CACHE: dict = {"data": None, "ts": 0.0}


@app.route('/api/laliga/fixtures')
def laliga_fixtures():
    now_ts = time.time()
    if _LALIGA_TIPS_CACHE["data"] is None or (now_ts - _LALIGA_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_laliga_tips(_club_predict)
            for t in tips:
                t['best_bet'] = _compute_best_bet(t)
                _save_prediction(t)
            _LALIGA_TIPS_CACHE["data"] = tips
            _LALIGA_TIPS_CACHE["ts"]   = now_ts
        except Exception as exc:
            app.logger.error("laliga-fixtures fetch failed: %s", exc)
            if _LALIGA_TIPS_CACHE["data"] is None:
                _LALIGA_TIPS_CACHE["data"] = []
    done = _completed_keys()
    active = [t for t in (_LALIGA_TIPS_CACHE["data"] or [])
              if _match_key(t.get('home_team',''), t.get('away_team',''),
                            (t.get('utc_kickoff','') or '')[:10]) not in done]
    return jsonify(active)


_LALIGA_COUNTDOWN_CACHE: dict = {"data": None, "ts": 0.0}
_LALIGA_COUNTDOWN_TTL = 3600  # 1 hour — the fixture list rarely changes


@app.route('/api/laliga/countdown')
def laliga_countdown():
    now_ts = time.time()
    if _LALIGA_COUNTDOWN_CACHE["data"] is None or (now_ts - _LALIGA_COUNTDOWN_CACHE["ts"]) > _LALIGA_COUNTDOWN_TTL:
        # No hardcoded fallback fixture here (unlike PL) — we don't have a
        # confirmed La Liga opener date, so an API miss just means "unknown"
        # rather than a fabricated match.
        _LALIGA_COUNTDOWN_CACHE["data"] = get_laliga_next_match()
        _LALIGA_COUNTDOWN_CACHE["ts"]   = now_ts
    fixture = _LALIGA_COUNTDOWN_CACHE["data"]
    if not fixture:
        return jsonify({'fixture': None, 'seconds_remaining': 0})
    now = datetime.now(timezone.utc)
    try:
        kickoff = datetime.fromisoformat(fixture['utc_kickoff'].replace('Z', '+00:00'))
    except Exception:
        kickoff = now
    seconds_remaining = max(0, int((kickoff - now).total_seconds()))
    return jsonify({'fixture': fixture, 'seconds_remaining': seconds_remaining})


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
                t['best_bet'] = _compute_best_bet(t)
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
                t['best_bet'] = _compute_best_bet(t)
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
        if not _wc_is_resolved(f):
            continue  # hide until real teams are confirmed — no fabricated placeholder stats
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
            'best_bet':         f.get('best_bet') or _compute_best_bet(f),
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
            for t in tips:
                t['best_bet'] = _compute_best_bet(t)
                _save_prediction(t)
            _CLUB_TIPS_CACHE["data"] = tips
            _CLUB_TIPS_CACHE["ts"]   = now_ts
        except Exception as exc:
            _log.warning('best-bet: club warm: %s', exc)
            _CLUB_TIPS_CACHE.setdefault("data", [])

    # Warm international cache
    if _INTL_TIPS_CACHE["data"] is None or (now_ts - _INTL_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_intl_tips(_wc_predict)
            for t in tips:
                t['best_bet'] = _compute_best_bet(t)
                _save_prediction(t)
            _INTL_TIPS_CACHE["data"] = tips
            _INTL_TIPS_CACHE["ts"]   = now_ts
        except Exception as exc:
            _log.warning('best-bet: intl warm: %s', exc)
            _INTL_TIPS_CACHE.setdefault("data", [])

    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Aggregate all sources; include today's WC fixtures
    raw = list(_CLUB_TIPS_CACHE["data"] or []) + list(_INTL_TIPS_CACHE["data"] or [])
    for f in _WC_FIXTURES:
        if f.get('date', '') == today_str and _wc_is_resolved(f):
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
        ('WNBA', _WNBA_CACHE.get('data') or [], 155.5),
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
    if _NBA_CACHE["data"] is None or (now_ts - _NBA_CACHE["ts"]) > _BBALL_CACHE_TTL:
        try:
            from nba_predict import get_nba_fixtures as _get_games, predict_nba as _predict
            from datetime import date as _date, timedelta as _td
            _end = (_date.today() + _td(days=1)).strftime('%Y-%m-%d')
            games   = _get_games(end_date=_end)
            result  = []
            for g in games:
                pred = _predict(g['home_team'], g['away_team'], ou_line=g.get('ou_line'))
                if pred.get('error'):
                    continue
                comp = g.get('competition') or ('NBA Playoffs' if g.get('postseason') else 'NBA')
                result.append({
                    **g,
                    'sport':            'nba',
                    'competition':      comp,
                    'ou_line':          g.get('ou_line') or 220.5,
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
            # Mark first 2 games per date as free_tier
            _dc: dict = {}
            for _g in result:
                _d = _g.get('date', '')
                _dc[_d] = _dc.get(_d, 0) + 1
                _g['free_tier'] = _dc[_d] <= 2
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
    if _WNBA_CACHE["data"] is None or (now_ts - _WNBA_CACHE["ts"]) > _BBALL_CACHE_TTL:
        try:
            from nba_predict import get_wnba_fixtures as _get_games, predict_wnba as _predict
            from datetime import date as _date, timedelta as _td
            _end = (_date.today() + _td(days=1)).strftime('%Y-%m-%d')
            games   = _get_games(end_date=_end)
            result  = []
            for g in games:
                pred = _predict(g['home_team'], g['away_team'], ou_line=g.get('ou_line'))
                if pred.get('error'):
                    continue
                result.append({
                    **g,
                    'sport':            'wnba',
                    'competition':      'WNBA',
                    'ou_line':          g.get('ou_line') or 163.5,
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
            # Mark first 2 games per date as free_tier
            _dc: dict = {}
            for _g in result:
                _d = _g.get('date', '')
                _dc[_d] = _dc.get(_d, 0) + 1
                _g['free_tier'] = _dc[_d] <= 2
            _WNBA_CACHE["data"] = result
            _WNBA_CACHE["ts"]   = now_ts
        except Exception as exc:
            app.logger.error("WNBA fixtures failed: %s", exc)
            if _WNBA_CACHE["data"] is None:
                _WNBA_CACHE["data"] = []
    return jsonify(_WNBA_CACHE["data"] or [])


@app.route('/admin/backfill-corners', methods=['GET', 'POST'])
@_require_admin
def admin_backfill_corners():
    """
    GET  → count finished football games with actual_corners IS NULL.
    POST → attempt to re-fetch corners for every such game, returns per-match report.

    Tries TSDB first (has a per-event stats endpoint that returns corners),
    then ESPN dated scoreboard.  Basketball games are skipped (no corners).
    """
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503

    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503

            cur.execute("""
                SELECT match_id, home_team, away_team, match_date
                FROM predictions
                WHERE result_status IN ('WON', 'LOST')
                  AND actual_corners IS NULL
                  AND COALESCE(sport, 'football') NOT IN ('nba', 'wnba')
                ORDER BY match_date DESC
            """)
            rows = cur.fetchall()
            pending = [{'match_id': r[0], 'home_team': r[1],
                        'away_team': r[2], 'match_date': str(r[3])} for r in rows]

            if request.method == 'GET':
                return jsonify({
                    'action': 'preview — POST to run backfill',
                    'missing_count': len(pending),
                    'games_missing_corners': len(pending),
                    'sample': pending[:10],
                })

            # POST — attempt to fill each one
            rows_out = []
            filled_n = failed_n = 0
            for g in pending:
                home  = g['home_team']
                away  = g['away_team']
                date  = g['match_date']
                mid   = g['match_id']
                corners = None

                # Source 1: TSDB (returns corners alongside score)
                try:
                    _r = _fetch_tsdb_result(home, away, date)
                    if _r and _r[2] is not None:
                        corners = _r[2]
                except Exception:
                    pass

                # Source 2: ESPN dated scoreboard
                if corners is None:
                    try:
                        _r = _fetch_espn_result_dated(home, away, date)
                        if _r and _r[2] is not None:
                            corners = _r[2]
                    except Exception:
                        pass

                if corners is not None:
                    # Write corners
                    cur.execute(
                        "UPDATE predictions SET actual_corners=%s WHERE match_id=%s",
                        (corners, mid),
                    )
                    # Re-evaluate result_status when the best_bet is a corners market
                    cur.execute(
                        "SELECT predicted_winner, home_team, away_team, "
                        "actual_home_score, actual_away_score "
                        "FROM predictions WHERE match_id=%s",
                        (mid,)
                    )
                    _row = cur.fetchone()
                    if _row:
                        _pw, _ht, _at, _hs, _as = _row
                        if _pw and 'corner' in _pw.lower() and _hs is not None and _as is not None:
                            _new_status = _resolve_result(
                                _pw, _ht or '', _at or '', _hs, _as,
                                actual_corners=corners,
                            )
                            cur.execute(
                                "UPDATE predictions SET result_status=%s WHERE match_id=%s",
                                (_new_status, mid),
                            )
                    filled_n += 1
                    rows_out.append({'home_team': home, 'away_team': away,
                                     'match_date': date, 'corners': corners,
                                     'status': 'filled'})
                else:
                    failed_n += 1
                    rows_out.append({'home_team': home, 'away_team': away,
                                     'match_date': date, 'corners': None,
                                     'status': 'failed'})

            return jsonify({
                'action': 'backfill complete',
                'total_processed': len(pending),
                'filled':  filled_n,
                'failed':  failed_n,
                'skipped': 0,
                'rows':    rows_out,
            })

    except Exception as e:
        _log.exception('admin_backfill_corners: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/admin/deduplicate', methods=['GET', 'POST'])
@_require_admin
def admin_deduplicate():
    """
    GET  → find duplicate prediction rows (same home+away, date ±2 days).
    POST → delete the less-complete duplicate from each pair; keep the row
           with actual scores, or if both/neither have scores, the newer one.
    """
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503

            cur.execute("""
                SELECT p1.match_id  AS id1,
                       p2.match_id  AS id2,
                       p1.home_team, p1.away_team,
                       p1.match_date AS date1, p2.match_date AS date2,
                       p1.result_status AS status1, p2.result_status AS status2,
                       p1.actual_home_score AS hs1, p1.actual_away_score AS as1,
                       p2.actual_home_score AS hs2, p2.actual_away_score AS as2,
                       p1.actual_corners AS c1, p2.actual_corners AS c2,
                       p1.created_at AS at1, p2.created_at AS at2,
                       p1.predicted_winner AS pw1, p2.predicted_winner AS pw2
                FROM predictions p1
                JOIN predictions p2
                  ON LOWER(p1.home_team) = LOWER(p2.home_team)
                 AND LOWER(p1.away_team) = LOWER(p2.away_team)
                 AND p1.match_id < p2.match_id
                 AND ABS(p1.match_date::date - p2.match_date::date) <= 2
                ORDER BY p1.home_team, p1.away_team, p1.match_date
            """)
            cols = [d[0] for d in cur.description]
            pairs = [dict(zip(cols, r)) for r in cur.fetchall()]

            if request.method == 'GET':
                # Serialize dates/timestamps
                for p in pairs:
                    for k in ('date1','date2','at1','at2'):
                        if p.get(k) is not None:
                            p[k] = str(p[k])
                return jsonify({'duplicate_pairs': len(pairs), 'pairs': pairs})

            # POST — delete the less-complete record from each pair
            deleted = []
            for p in pairs:
                # Prefer the row that has actual scores; if tie prefer the one with corners;
                # if still tied keep the newer one (at2 >= at1 since id1 < id2 by creation order)
                p1_score = p['hs1'] is not None
                p2_score = p['hs2'] is not None
                p1_corners = p['c1'] is not None
                p2_corners = p['c2'] is not None

                if p1_score and not p2_score:
                    delete_id = p['id2']
                elif p2_score and not p1_score:
                    delete_id = p['id1']
                elif p1_corners and not p2_corners:
                    delete_id = p['id2']
                elif p2_corners and not p1_corners:
                    delete_id = p['id1']
                else:
                    # Both equally complete — delete the older one
                    delete_id = p['id1']

                cur.execute("DELETE FROM predictions WHERE match_id=%s", (delete_id,))
                deleted.append({
                    'deleted': delete_id,
                    'kept': p['id2'] if delete_id == p['id1'] else p['id1'],
                    'match': f"{p['home_team']} vs {p['away_team']}",
                })

            return jsonify({
                'action': 'deduplication complete',
                'pairs_found': len(pairs),
                'deleted': len(deleted),
                'records': deleted,
            })
    except Exception as e:
        _log.exception('admin_deduplicate: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/admin/audit-history')
@_require_admin
def admin_audit_history():
    """
    Diagnostic: show all prediction records, status breakdown, past-date PENDING,
    and per-outcome bet counts so the user can spot rogue or missing records.
    """
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503

            # Status breakdown
            cur.execute("""
                SELECT result_status, COUNT(*) FROM predictions
                GROUP BY result_status ORDER BY result_status
            """)
            status_counts = {r[0]: r[1] for r in cur.fetchall()}

            # All WON/LOST records (ordered newest first)
            cur.execute("""
                SELECT match_id, home_team, away_team, match_date,
                       competition, predicted_winner, actual_home_score,
                       actual_away_score, actual_corners, result_status,
                       COALESCE(sport,'football') AS sport, created_at
                FROM predictions
                WHERE result_status IN ('WON','LOST')
                ORDER BY match_date DESC, created_at DESC
            """)
            cols = [d[0] for d in cur.description]
            won_lost = [dict(zip(cols, r)) for r in cur.fetchall()]
            for row in won_lost:
                row['created_at'] = str(row.get('created_at',''))
                row['match_date']  = str(row.get('match_date',''))

            # Past-date PENDING records (finished but not resolved — likely wrongly PENDING)
            cur.execute("""
                SELECT match_id, home_team, away_team, match_date,
                       competition, predicted_winner, result_status,
                       COALESCE(sport,'football') AS sport, created_at
                FROM predictions
                WHERE result_status IN ('PENDING','LIVE')
                  AND match_date::date < CURRENT_DATE - INTERVAL '1 day'
                ORDER BY match_date DESC
            """)
            cols2 = [d[0] for d in cur.description]
            stale_pending = [dict(zip(cols2, r)) for r in cur.fetchall()]
            for row in stale_pending:
                row['created_at'] = str(row.get('created_at',''))
                row['match_date']  = str(row.get('match_date',''))

            # Duplicates still remaining
            cur.execute("""
                SELECT p1.match_id AS id1, p2.match_id AS id2,
                       p1.home_team, p1.away_team,
                       p1.match_date AS date1, p2.match_date AS date2,
                       p1.result_status AS status1, p2.result_status AS status2
                FROM predictions p1
                JOIN predictions p2
                  ON LOWER(p1.home_team) = LOWER(p2.home_team)
                 AND LOWER(p1.away_team) = LOWER(p2.away_team)
                 AND p1.match_id < p2.match_id
                 AND ABS(p1.match_date::date - p2.match_date::date) <= 2
                ORDER BY p1.home_team
            """)
            cols3 = [d[0] for d in cur.description]
            remaining_dups = [dict(zip(cols3, r)) for r in cur.fetchall()]
            for row in remaining_dups:
                row['date1'] = str(row.get('date1',''))
                row['date2'] = str(row.get('date2',''))

            # Total bets (per-outcome count, same logic as /api/results)
            total_outcomes = won_outcomes = 0
            for rec in won_lost:
                pw = rec.get('predicted_winner','')
                hs = rec.get('actual_home_score')
                as_ = rec.get('actual_away_score')
                ac  = rec.get('actual_corners')
                sport = rec.get('sport','football')
                if hs is None:
                    continue
                # Count outcomes that would appear in sub_results
                sub = _sub_results(rec)
                for v in sub.values():
                    if v in ('WON','LOST'):
                        total_outcomes += 1
                        if v == 'WON':
                            won_outcomes += 1

            return jsonify({
                'status_counts':    status_counts,
                'total_won_lost':   len(won_lost),
                'stale_pending':    len(stale_pending),
                'remaining_dups':   len(remaining_dups),
                'bet_outcomes_total': total_outcomes,
                'bet_outcomes_won':   won_outcomes,
                'win_rate': round(won_outcomes / total_outcomes * 100, 1) if total_outcomes else 0.0,
                'won_lost_records':   won_lost,
                'stale_pending_records': stale_pending,
                'remaining_dup_pairs':   remaining_dups,
            })
    except Exception as e:
        _log.exception('admin_audit_history: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/admin/re-resolve-stale', methods=['POST'])
@_require_admin
def admin_re_resolve_stale():
    """
    Re-fetch results for all past-date PENDING football records using ESPN -1 day retry.
    Updates actual_home_score, actual_away_score, actual_corners, result_status.
    """
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503

            cur.execute("""
                SELECT match_id, home_team, away_team, match_date, predicted_winner,
                       COALESCE(sport,'football') AS sport
                FROM predictions
                WHERE result_status IN ('PENDING','LIVE')
                  AND match_date::date < CURRENT_DATE - INTERVAL '1 day'
                ORDER BY match_date DESC
            """)
            rows = cur.fetchall()

            resolved = []
            failed   = []
            for mid, home, away, match_date, pw, sport in rows:
                if sport in ('nba','wnba'):
                    failed.append({'match_id': mid, 'teams': f'{home} vs {away}',
                                   'reason': 'basketball — use basketball checker'})
                    continue
                # Try ESPN (with -1 day retry built in)
                r = _fetch_espn_result_dated(home, away, str(match_date))
                if r is None:
                    # Also try TSDB
                    r = _fetch_tsdb_result(home, away, str(match_date))
                if r is None:
                    failed.append({'match_id': mid, 'teams': f'{home} vs {away}',
                                   'date': str(match_date), 'reason': 'not found'})
                    continue
                hs, as_, corners = r
                status = _resolve_result(pw or '', home, away, hs, as_,
                                         actual_corners=corners)
                cur.execute("""
                    UPDATE predictions
                    SET actual_home_score=%s, actual_away_score=%s,
                        actual_corners=%s, result_status=%s, match_status='FINISHED'
                    WHERE match_id=%s
                """, (hs, as_, corners, status, mid))
                resolved.append({'match_id': mid, 'teams': f'{home} vs {away}',
                                 'date': str(match_date),
                                 'score': f'{hs}–{as_}', 'corners': corners,
                                 'status': status})

            return jsonify({
                'resolved': len(resolved),
                'failed':   len(failed),
                'resolved_records': resolved,
                'failed_records':   failed,
            })
    except Exception as e:
        _log.exception('admin_re_resolve_stale: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/sync-scores', methods=['GET', 'POST'])
@_require_admin
def admin_sync_scores():
    """
    Manual score sync — same checkers the scheduler runs (30-min interval
    + the 2 AM UTC daily cleanup), triggered on demand for an immediate update.

    GET  → preview: counts of PENDING/LIVE predictions awaiting a result.
    POST → run the sync now (football + basketball) and report what changed.
    """
    if request.method == 'GET':
        if not _DB_CONN_URL:
            return jsonify({'error': 'DB not configured'}), 503
        try:
            with _db() as (conn, cur):
                if conn is None:
                    return jsonify({'error': 'DB unavailable'}), 503
                cur.execute("""
                    SELECT match_id, home_team, away_team, match_date,
                           result_status, COALESCE(sport, 'football') AS sport
                    FROM predictions
                    WHERE result_status IN ('PENDING', 'LIVE')
                    ORDER BY match_date DESC
                """)
                rows = cur.fetchall()
                pending = [{'match_id': r[0], 'teams': f'{r[1]} vs {r[2]}',
                            'date': str(r[3]), 'status': r[4], 'sport': r[5]}
                           for r in rows]
                return jsonify({
                    'action': 'preview — POST to run sync now',
                    'pending_count': len(pending),
                    'pending': pending[:25],
                })
        except Exception as e:
            _log.exception('admin_sync_scores GET: %s', e)
            return jsonify({'error': str(e)}), 500

    # POST — run it now
    try:
        before = None
        if _DB_CONN_URL:
            try:
                with _db() as (conn, cur):
                    if conn is not None:
                        cur.execute(
                            "SELECT COUNT(*) FROM predictions WHERE result_status IN ('PENDING','LIVE')")
                        before = cur.fetchone()[0]
            except Exception:
                before = None

        summary = _daily_results_cleanup()

        after = None
        if _DB_CONN_URL:
            try:
                with _db() as (conn, cur):
                    if conn is not None:
                        cur.execute(
                            "SELECT COUNT(*) FROM predictions WHERE result_status IN ('PENDING','LIVE')")
                        after = cur.fetchone()[0]
            except Exception:
                after = None

        return jsonify({
            'football_resolved':   summary['football_resolved'],
            'basketball_resolved': summary['basketball_resolved'],
            'bracket_resolved':    summary['bracket_resolved'],
            'total_resolved':      summary['football_resolved'] + summary['basketball_resolved'],
            'pending_before': before,
            'pending_after':  after,
        })
    except Exception as e:
        _log.exception('admin_sync_scores POST: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/admin/fix-history-records', methods=['GET', 'POST'])
@_require_admin
def admin_fix_history_records():
    """
    GET  → diagnostic dump: Mystics fix state + all WC/Paraguay records
    POST → apply corrections for Mystics + Paraguay (using broad search)
    Remove after confirmed.
    """
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503

    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503

            # ── 1. Washington Mystics ──────────────────────────────────────────
            cur.execute("""
                SELECT match_id, home_team, away_team, match_date, competition,
                       predicted_winner, actual_home_score, actual_away_score, result_status
                FROM predictions
                WHERE (LOWER(home_team) LIKE %s AND LOWER(away_team) LIKE %s)
                   OR (LOWER(home_team) LIKE %s AND LOWER(away_team) LIKE %s)
            """, ('%mystics%', '%tempo%', '%tempo%', '%mystics%'))
            mystics_rows = cur.fetchall()

            # ── 2. Paraguay — broad search (no date/competition filter) ────────
            cur.execute("""
                SELECT match_id, home_team, away_team, match_date, competition,
                       predicted_winner, actual_home_score, actual_away_score, result_status
                FROM predictions
                WHERE LOWER(home_team) LIKE %s OR LOWER(away_team) LIKE %s
                ORDER BY match_date DESC
            """, ('%paraguay%', '%paraguay%'))
            paraguay_rows = cur.fetchall()

            # ── 3. All records from 10–15 Jun 2026 (show raw names/competition) ─
            cur.execute("""
                SELECT match_id, home_team, away_team, match_date, competition,
                       predicted_winner, actual_home_score, actual_away_score, result_status
                FROM predictions
                WHERE match_date BETWEEN '2026-06-10' AND '2026-06-15'
                ORDER BY match_date, home_team
            """)
            jun_rows = cur.fetchall()

            cols = ['match_id', 'home_team', 'away_team', 'match_date', 'competition',
                    'predicted_winner', 'actual_home_score', 'actual_away_score', 'result_status']

            def rows_to_list(rows):
                return [dict(zip(cols, r)) for r in rows]

            mystics_list  = rows_to_list(mystics_rows)
            paraguay_list = rows_to_list(paraguay_rows)
            jun_list      = rows_to_list(jun_rows)

            # GSV @ Seattle Storm — broad search, any date
            cur.execute("""
                SELECT match_id, home_team, away_team, match_date, competition,
                       predicted_winner, actual_home_score, actual_away_score, result_status
                FROM predictions
                WHERE (LOWER(home_team) LIKE %s AND LOWER(away_team) LIKE %s)
                   OR (LOWER(home_team) LIKE %s AND LOWER(away_team) LIKE %s)
            """, ('%valkyries%', '%storm%', '%storm%', '%valkyries%'))
            gsv_rows = cur.fetchall()
            gsv_list = rows_to_list(gsv_rows)

            if request.method == 'GET':
                return jsonify({
                    'action': 'diagnostic — POST to apply fixes',
                    'mystics_records':   mystics_list,
                    'paraguay_records':  paraguay_list,
                    'gsv_records':       gsv_list,
                    'jun10_15_records':  jun_list,
                })

            # ── POST: apply fixes ─────────────────────────────────────────────
            applied = []

            def _apply_fix(records, new_pw, label):
                """Update predicted_winner; re-derive result_status from actual scores."""
                for r in records:
                    hs  = r['actual_home_score']
                    as_ = r['actual_away_score']
                    if hs is not None and as_ is not None:
                        new_rs = _resolve_result(new_pw, r['home_team'], r['away_team'], hs, as_)
                    else:
                        new_rs = r['result_status']  # can't re-evaluate without scores
                    cur.execute("""
                        UPDATE predictions
                        SET predicted_winner = %s, result_status = %s, match_status = 'FINISHED'
                        WHERE match_id = %s
                    """, (new_pw, new_rs, r['match_id']))
                    applied.append({
                        'fixed': label, 'match_id': r['match_id'],
                        'before': {'predicted_winner': r['predicted_winner'],
                                   'result_status':    r['result_status']},
                        'after':  {'predicted_winner': new_pw, 'result_status': new_rs},
                        'score':  f"{hs}-{as_}" if hs is not None else 'unknown',
                    })

            # Mystics — O/U Over 170.5 was the correct best bet
            _apply_fix(mystics_list, 'Over 170.5', 'mystics')

            # Paraguay — Goals Over 2.5 was the correct best bet
            if paraguay_list:
                _apply_fix(paraguay_list, 'Over 2.5 Goals', 'paraguay')
            else:
                applied.append({'fixed': 'paraguay', 'found': False,
                                'note': 'No Paraguay record — check diagnostic GET'})

            # GSV @ Seattle Storm — Under 170.5 matches terminal output
            if gsv_list:
                _apply_fix(gsv_list, 'Under 170.5', 'gsv_storm')
            else:
                applied.append({'fixed': 'gsv_storm', 'found': False,
                                'note': 'No GSV/Storm record found'})

            _BEST_BET_CACHE["data"] = None
            _DAILY_TIPS_CACHE["ts"] = 0

            return jsonify({'action': 'fixes applied', 'applied': applied})

    except Exception as e:
        _log.exception('admin_fix_history_records: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/admin/fix-wc-scores', methods=['GET', 'POST'])
@_require_admin
def admin_fix_wc_scores():
    """
    One-off correction for two WC 2026 records whose score got frozen
    mid-match: TheSportsDB's strStatus used short live-clock codes ('2H') that
    _TSDB_NOT_DONE didn't recognize as not-finished, so a snapshot score got
    saved as final and the row left WON/LOST (never re-scanned, since the
    periodic checkers only touch PENDING/LIVE rows). Now fixed in
    _TSDB_NOT_DONE; this corrects the two rows already frozen wrong.

    Verified correct values against ESPN's STATUS_FULL_TIME boxscore (score +
    wonCorners) on 2026-06-19:
      - Canada 6-0 Qatar (corners 19+1=20, unchanged — only home/away score was wrong)
      - Uzbekistan 1-3 Colombia (corners 3+4=7, unchanged — only away score was wrong)

    GET  → diagnostic: current stored values for both match_ids.
    POST → apply the corrected scores; result_status is re-derived via
           _resolve_result, not hardcoded. Remove after confirmed.
    """
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503

    # (match_id, correct_home_score, correct_away_score, correct_corners)
    fixes = [
        ('2026-06-18/canada/qatar',        6, 0, 20),
        ('2026-06-18/uzbekistan/colombia', 1, 3, 7),
    ]

    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503

            cur.execute("""
                SELECT match_id, home_team, away_team, predicted_winner,
                       actual_home_score, actual_away_score, actual_corners, result_status
                FROM predictions
                WHERE match_id = ANY(%s)
            """, ([f[0] for f in fixes],))
            cols = ['match_id', 'home_team', 'away_team', 'predicted_winner',
                    'actual_home_score', 'actual_away_score', 'actual_corners', 'result_status']
            before = {row[0]: dict(zip(cols, row)) for row in cur.fetchall()}

            if request.method == 'GET':
                return jsonify({'action': 'diagnostic — POST to apply', 'before': before})

            applied = []
            for match_id, hs, as_, corners in fixes:
                row = before.get(match_id)
                if not row:
                    applied.append({'match_id': match_id, 'found': False})
                    continue
                new_status = _resolve_result(
                    row['predicted_winner'] or '', row['home_team'] or '', row['away_team'] or '',
                    hs, as_, actual_corners=corners)
                cur.execute("""
                    UPDATE predictions
                    SET actual_home_score=%s, actual_away_score=%s, actual_corners=%s,
                        result_status=%s, match_status='FINISHED'
                    WHERE match_id=%s
                """, (hs, as_, corners, new_status, match_id))
                applied.append({
                    'match_id': match_id,
                    'before': row,
                    'after': {'actual_home_score': hs, 'actual_away_score': as_,
                              'actual_corners': corners, 'result_status': new_status},
                })

            _sync_saved_statuses(cur)
            _BEST_BET_CACHE["data"] = None
            _DAILY_TIPS_CACHE["ts"] = 0
            return jsonify({'action': 'fixes applied', 'applied': applied})

    except Exception as e:
        _log.exception('admin_fix_wc_scores: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/admin/rederive-statuses', methods=['GET', 'POST'])
@_require_admin
def admin_rederive_statuses():
    """
    GET  → dry-run: re-derives result_status for every WON/LOST record using
           corrected _resolve_result logic (BTTS branch + corners). Lists mismatches.
    POST → applies all corrections.
    """
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503

    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503

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

            mismatches = []
            checked = 0
            for mid, home, away, match_date, pw, hs, as_, corners, curr_status in rows:
                checked += 1
                computed = _resolve_result(pw or '', home or '', away or '', hs, as_,
                                           actual_corners=corners)
                if computed != curr_status:
                    mismatches.append({
                        'match_id':   mid,
                        'teams':      f'{home} vs {away}',
                        'match_date': str(match_date),
                        'best_bet':   pw,
                        'score':      f'{hs}–{as_}',
                        'corners':    corners,
                        'was':        curr_status,
                        'should_be':  computed,
                    })

            if request.method == 'GET':
                return jsonify({
                    'checked':    checked,
                    'mismatches': len(mismatches),
                    'records':    mismatches,
                })

            # POST: apply fixes
            fixed = 0
            for m in mismatches:
                cur.execute("""
                    UPDATE predictions SET result_status = %s
                    WHERE match_id = %s
                """, (m['should_be'], m['match_id']))
                fixed += 1

            _BEST_BET_CACHE["data"] = None
            _DAILY_TIPS_CACHE["ts"] = 0

            return jsonify({
                'action':  'fixes applied',
                'checked': checked,
                'fixed':   fixed,
                'records': mismatches,
            })

    except Exception as e:
        _log.exception('admin_rederive_statuses: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/admin/recalc-pending', methods=['GET', 'POST'])
@_require_admin
def admin_recalc_pending():
    """
    GET  → dry-run: re-runs every PENDING/LIVE prediction through the shared
           prediction function and returns a before/after comparison table.
    POST → same but writes the new values to the DB.

    Football : predict_match(home, away) + _compute_best_bet()
    NBA/WNBA : predict_nba/predict_wnba(home, away) from nba_predict.py

    NOTE: win_probability, ou_probability, best_bet_type are computed live and
    are not stored as DB columns — they will automatically reflect the updated
    predicted_winner when the API serves predictions.
    """
    if not _DB_CONN_URL:
        return jsonify({'error': 'DB not configured'}), 503

    try:
        from nba_predict import predict_nba as _pred_nba, predict_wnba as _pred_wnba
    except ImportError as e:
        return jsonify({'error': f'Cannot import nba_predict: {e}'}), 500

    apply = (request.method == 'POST')
    rows  = []

    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503

            cur.execute("""
                SELECT match_id, home_team, away_team, match_date, competition,
                       predicted_winner, predicted_goals, predicted_btts,
                       predicted_corners, result_status,
                       COALESCE(sport, 'football') AS sport
                FROM predictions
                WHERE result_status IN ('PENDING', 'LIVE')
                ORDER BY match_date, home_team
            """)
            col_names = [d[0] for d in cur.description]
            pending = [dict(zip(col_names, r)) for r in cur.fetchall()]

            for rec in pending:
                home  = rec['home_team'] or ''
                away  = rec['away_team'] or ''
                sport = rec['sport']
                mid   = rec['match_id']

                before = {
                    'predicted_winner':  rec['predicted_winner'],
                    'predicted_goals':   rec['predicted_goals'],
                    'predicted_btts':    rec['predicted_btts'],
                    'predicted_corners': rec['predicted_corners'],
                }
                after  = dict(before)
                err    = None

                try:
                    if sport == 'nba':
                        pred     = _pred_nba(home, away)
                        after['predicted_winner'] = pred.get('best_bet', '')
                        after['predicted_goals']  = pred.get('predicted_ou', '')
                        after['predicted_btts']   = None
                        after['predicted_corners'] = None

                    elif sport == 'wnba':
                        pred     = _pred_wnba(home, away)
                        after['predicted_winner'] = pred.get('best_bet', '')
                        after['predicted_goals']  = pred.get('predicted_ou', '')
                        after['predicted_btts']   = None
                        after['predicted_corners'] = None

                    else:  # football
                        tip = _wc_predict(home, away, is_neutral=True)
                        if tip is None:
                            raise ValueError(f'predict_match returned None for {home} vs {away}')
                        bb  = _compute_best_bet(tip)
                        pg  = float(tip.get('over_goals', 0.5) or 0.5)
                        bt  = float(tip.get('btts',       0.48) or 0.48)
                        co  = float(tip.get('over_corners', 0.52) or 0.52)
                        after['predicted_winner']  = bb.get('label', '')
                        after['predicted_goals']   = 'Over 2.5'  if pg >= 0.5 else 'Under 2.5'
                        after['predicted_btts']    = 'Yes'       if bt >= 0.5 else 'No'
                        after['predicted_corners'] = 'Over 9.5'  if co >= 0.5 else 'Under 9.5'

                except Exception as exc:
                    err = str(exc)

                changed = (err is None) and (after != before)

                if apply and changed:
                    cur.execute("""
                        UPDATE predictions
                        SET predicted_winner  = %s,
                            predicted_goals   = %s,
                            predicted_btts    = %s,
                            predicted_corners = %s
                        WHERE match_id = %s
                          AND result_status IN ('PENDING', 'LIVE')
                    """, (after['predicted_winner'], after['predicted_goals'],
                          after['predicted_btts'],   after['predicted_corners'],
                          mid))

                rows.append({
                    'match_id':   mid,
                    'home_team':  home,
                    'away_team':  away,
                    'match_date': rec['match_date'],
                    'sport':      sport,
                    'status':     rec['result_status'],
                    'before':     before,
                    'after':      after,
                    'changed':    changed,
                    'error':      err,
                })

            if apply:
                _BEST_BET_CACHE["data"]   = None
                _DAILY_TIPS_CACHE["ts"]   = 0
                _NBA_CACHE["data"]        = None
                _WNBA_CACHE["data"]       = None

            updated = sum(1 for r in rows if r['changed'])
            errors  = sum(1 for r in rows if r['error'])
            return jsonify({
                'action':        'applied' if apply else 'preview',
                'total_pending': len(pending),
                'updated':       updated,
                'errors':        errors,
                'rows':          rows,
            })

    except Exception as e:
        _log.exception('admin_recalc_pending: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/stored-prediction')
def stored_prediction():
    """
    Return the stored prediction (including full probability payload) for a match.

    Terminal scripts call this so they display the exact same values as the website.

    Query params:
      home   – home team name (required)
      away   – away team name (required)
      date   – YYYY-MM-DD (optional; searched by team names if omitted)
      sport  – 'football' | 'nba' | 'wnba' (optional filter)

    On match: returns {'found': True, 'prediction': { ... }}
    No match: returns {'found': False}
    """
    home  = (request.args.get('home', '') or '').strip()
    away  = (request.args.get('away', '') or '').strip()
    date  = (request.args.get('date', '') or '').strip()
    sport = (request.args.get('sport', '') or '').strip().lower()

    if not home or not away:
        return jsonify({'error': 'home and away are required'}), 400

    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'DB unavailable'}), 503

            cols_sql = """
                match_id, home_team, away_team, match_date, match_time,
                competition, predicted_winner, predicted_goals,
                predicted_btts, predicted_corners,
                result_status, kickoff_utc,
                COALESCE(sport, 'football') AS sport,
                prediction_payload
            """

            if date:
                mid = _match_key(home, away, date)
                cur.execute(f"SELECT {cols_sql} FROM predictions WHERE match_id = %s", (mid,))
                row = cur.fetchone()
            else:
                # Search by team names; prefer PENDING, then most recent
                sport_filter = "AND LOWER(sport) = %s" if sport else ""
                params = [f'%{home.lower()}%', f'%{away.lower()}%']
                if sport:
                    params.append(sport)
                cur.execute(f"""
                    SELECT {cols_sql} FROM predictions
                    WHERE (LOWER(home_team) LIKE %s AND LOWER(away_team) LIKE %s)
                      {sport_filter}
                    ORDER BY
                        CASE result_status WHEN 'PENDING' THEN 0 WHEN 'LIVE' THEN 1 ELSE 2 END,
                        match_date DESC
                    LIMIT 1
                """, params)
                row = cur.fetchone()

            if not row:
                return jsonify({'found': False, 'home': home, 'away': away, 'date': date})

            col_names = ['match_id', 'home_team', 'away_team', 'match_date', 'match_time',
                         'competition', 'predicted_winner', 'predicted_goals',
                         'predicted_btts', 'predicted_corners',
                         'result_status', 'kickoff_utc', 'sport', 'prediction_payload']
            d = dict(zip(col_names, row))
            # prediction_payload comes back as dict from psycopg3 JSONB; serialise for older clients
            if isinstance(d.get('prediction_payload'), dict):
                pass  # already a dict — leave it
            return jsonify({'found': True, 'prediction': d})

    except Exception as e:
        _log.exception('stored_prediction: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/results')
def results_history():
    """Return completed prediction history with stats."""
    now  = datetime.now(timezone.utc)
    user = _get_current_user()
    is_premium = bool(user and user.get('is_premium'))

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
                # Show recently-played pending games (being resolved); hide stale ones
                stale_days = 3 if p.get('sport') in ('nba', 'wnba') else 5
                if ko + timedelta(hours=2) < now < ko + timedelta(days=stale_days):
                    finished.append(p)
            except Exception:
                pass

    finished = sorted(finished, key=lambda p: p.get('match_date', ''), reverse=True)

    # Free-tier filter: basketball limited to first 2 per (date, sport) per day
    if not is_premium:
        _bball_seen: dict = {}
        free_list = []
        # Sort within each day by match_id for a deterministic "first 2" selection
        for p in sorted(finished, key=lambda x: (
                x.get('match_date', ''), x.get('sport', ''), x.get('match_id', ''))):
            sport = p.get('sport', 'football')
            if sport in ('nba', 'wnba'):
                key = (p.get('match_date', ''), sport)
                n = _bball_seen.get(key, 0)
                if n >= 2:
                    continue   # premium-only — hide entirely from free users
                _bball_seen[key] = n + 1
            free_list.append(p)
        # Restore date-DESC order after filtering
        finished = sorted(free_list, key=lambda p: p.get('match_date', ''), reverse=True)

    # Attach per-market sub-results to each finished prediction
    for p in finished:
        p['sub_results'] = _sub_results(p)

    # Filter list: PENDING/LIVE always shown; resolved only if at least one market is WON
    def _has_any_won(p):
        if p['result_status'] in ('PENDING', 'LIVE'):
            return True
        sub = p.get('sub_results') or {}
        return any(v == 'WON' for v in sub.values())

    display = [p for p in finished if _has_any_won(p)]

    # Stats from visible (display) resolved games only, counted per individual outcome
    display_resolved = [p for p in display if p['result_status'] in ('WON', 'LOST')]
    total_outcomes = 0
    won_outcomes   = 0
    for p in display_resolved:
        for v in (p.get('sub_results') or {}).values():
            if v in ('WON', 'LOST'):   # skip None / no-data
                total_outcomes += 1
                if v == 'WON':
                    won_outcomes += 1

    win_rate = round(won_outcomes / total_outcomes * 100, 1) if total_outcomes else 0.0

    streak = 0
    streak_type = None
    for p in display_resolved:
        if streak_type is None:
            streak_type = p['result_status']
            streak = 1
        elif p['result_status'] == streak_type:
            streak += 1
        else:
            break

    return jsonify({
        'stats': {
            'total':       total_outcomes,
            'won':         won_outcomes,
            'win_rate':    win_rate,
            'streak':      streak,
            'streak_type': streak_type,
        },
        'predictions': display,
    })


@app.route('/api/live-scores')
def live_scores():
    """
    Return currently LIVE predictions keyed by match_id.

    Excludes rows whose live_updated_at is more than 10 minutes old (5x the
    2-min poll cadence) -- if the ESPN poll has stopped confirming a match
    is still live (outage, team-name mismatch, etc.), stop serving its last
    known score rather than freezing it in place. The frontend already
    falls back to its normal (non-live) card appearance when a match_id is
    absent from this response.
    """
    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute("""
                    SELECT match_id, home_team, away_team,
                           live_home_score, live_away_score, live_minute
                    FROM predictions
                    WHERE result_status = 'LIVE'
                      AND live_updated_at > NOW() - INTERVAL '10 minutes'
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
    import os as _os
    def _pkl_info(path, obj):
        try:
            sz = round(_os.path.getsize(path) / 1024)
            mtime = datetime.fromtimestamp(_os.path.getmtime(path), tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
            return {'loaded': obj is not None, 'size_kb': sz, 'mtime': mtime}
        except Exception:
            return {'loaded': obj is not None}
    return jsonify({
        'status':       'ok',
        'db':           db_ok,
        'db_msg':       db_msg,
        'db_url_set':   bool(_DB_CONN_URL),
        'korapay_set':  bool(_KORAPAY_PUBLIC),
        'models': {
            'club_result':  _pkl_info(os.path.join(_BASE_DIR, 'result_model.pkl'),       club_result),
            'club_goals':   _pkl_info(os.path.join(_BASE_DIR, 'goals_model.pkl'),        club_goals),
            'club_corners': _pkl_info(os.path.join(_BASE_DIR, 'corners_model.pkl'),      club_corners),
            'nba_result':   _pkl_info(os.path.join(_BASE_DIR, 'nba_result_model.pkl'),   True),
            'nba_ou':       _pkl_info(os.path.join(_BASE_DIR, 'nba_ou_model.pkl'),       True),
            'wnba_result':  _pkl_info(os.path.join(_BASE_DIR, 'wnba_result_model.pkl'),  True),
            'wnba_ou':      _pkl_info(os.path.join(_BASE_DIR, 'wnba_ou_model.pkl'),      True),
        },
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


# ── Saved predictions endpoints ───────────────────────────────────────────────

@app.route('/api/predictions/<path:match_id>/save', methods=['POST'])
def save_prediction(match_id):
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'Database unavailable'}), 503
            cur.execute('SELECT id FROM predictions WHERE match_id=%s', (match_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({'error': 'Prediction not found'}), 404
            pred_id = row[0]
            cur.execute(
                'SELECT id FROM saved_predictions WHERE user_id=%s AND prediction_id=%s',
                (user['id'], pred_id)
            )
            existing = cur.fetchone()
            if existing:
                cur.execute('DELETE FROM saved_predictions WHERE id=%s', (existing[0],))
                return jsonify({'saved': False})
            else:
                cur.execute(
                    'INSERT INTO saved_predictions (user_id, prediction_id) VALUES (%s, %s)',
                    (user['id'], pred_id)
                )
                return jsonify({'saved': True}), 201
    except Exception as e:
        _log.error('save_prediction: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions/<path:match_id>/is-saved')
def is_prediction_saved(match_id):
    user = _get_current_user()
    if not user:
        return jsonify({'saved': False})
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'saved': False})
            cur.execute("""
                SELECT 1 FROM saved_predictions sp
                JOIN predictions p ON sp.prediction_id = p.id
                WHERE sp.user_id=%s AND p.match_id=%s
            """, (user['id'], match_id))
            return jsonify({'saved': cur.fetchone() is not None})
    except Exception as e:
        _log.warning('is_prediction_saved: %s', e)
        return jsonify({'saved': False})


@app.route('/api/user/saved-predictions')
def user_saved_predictions():
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'Database unavailable'}), 503
            cur.execute("""
                SELECT sp.id, sp.status, sp.match_result, sp.saved_at,
                       p.match_id, p.home_team, p.away_team, p.match_date,
                       p.match_time, p.competition, p.predicted_winner,
                       p.predicted_goals, p.predicted_btts, p.result_status,
                       p.kickoff_utc, p.prediction_payload
                  FROM saved_predictions sp
                  JOIN predictions p ON sp.prediction_id = p.id
                 WHERE sp.user_id = %s
                 ORDER BY sp.saved_at DESC
            """, (user['id'],))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            # Coerce non-serialisable types
            for r in rows:
                if r.get('saved_at'):
                    r['saved_at'] = r['saved_at'].isoformat()
            total   = len(rows)
            won     = sum(1 for r in rows if r['status'] == 'won')
            lost    = sum(1 for r in rows if r['status'] == 'lost')
            pending = sum(1 for r in rows if r['status'] == 'pending')
            win_rate = round(won / (won + lost) * 100, 1) if (won + lost) > 0 else 0
            return jsonify({
                'predictions': rows,
                'stats': {
                    'total': total, 'won': won, 'lost': lost,
                    'pending': pending, 'win_rate': win_rate,
                },
            })
    except Exception as e:
        _log.error('user_saved_predictions: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/user/saved-predictions/<int:saved_id>', methods=['PATCH'])
def update_saved_prediction(saved_id):
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    data   = request.get_json(silent=True) or {}
    status = data.get('status', '')
    if status not in ('pending', 'won', 'lost'):
        return jsonify({'error': 'status must be pending, won, or lost'}), 400
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'Database unavailable'}), 503
            cur.execute(
                'UPDATE saved_predictions SET status=%s WHERE id=%s AND user_id=%s RETURNING id',
                (status, saved_id, user['id'])
            )
            if not cur.fetchone():
                return jsonify({'error': 'Not found'}), 404
            return jsonify({'success': True})
    except Exception as e:
        _log.error('update_saved_prediction: %s', e)
        return jsonify({'error': str(e)}), 500


# ── Public config (safe to expose) ───────────────────────────────────────────

@app.route('/api/config')
def get_config():
    return jsonify({'onesignal_app_id': _ONESIGNAL_APP_ID})


# ── Notification token registration ──────────────────────────────────────────

@app.route('/api/user/notification-token', methods=['POST'])
def save_notification_token():
    user = _get_current_user()
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    data      = request.get_json(silent=True) or {}
    player_id = (data.get('player_id') or '').strip()
    if not player_id:
        return jsonify({'error': 'player_id is required'}), 400
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'Database unavailable'}), 503
            cur.execute("""
                INSERT INTO notification_tokens (user_id, onesignal_player_id)
                VALUES (%s, %s)
                ON CONFLICT (user_id, onesignal_player_id) DO NOTHING
            """, (user['id'], player_id))
            # Claim the token from the anonymous table now that we know the owner
            cur.execute(
                'DELETE FROM anonymous_notification_tokens WHERE player_id = %s',
                (player_id,)
            )
            return jsonify({'success': True})
    except Exception as e:
        _log.error('save_notification_token: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/push/register', methods=['POST'])
def register_anonymous_push_token():
    """Store a OneSignal player ID for a visitor who hasn't logged in yet.
    No auth required — anyone who allowed push can register here.
    On login, /api/user/notification-token claims the ID and moves it."""
    data      = request.get_json(silent=True) or {}
    player_id = (data.get('player_id') or '').strip()
    if not player_id:
        return jsonify({'error': 'player_id is required'}), 400
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'Database unavailable'}), 503
            # Skip if already claimed by a logged-in user
            cur.execute(
                'SELECT 1 FROM notification_tokens WHERE onesignal_player_id = %s LIMIT 1',
                (player_id,)
            )
            if cur.fetchone():
                return jsonify({'success': True})
            cur.execute("""
                INSERT INTO anonymous_notification_tokens (player_id)
                VALUES (%s)
                ON CONFLICT (player_id) DO NOTHING
            """, (player_id,))
        return jsonify({'success': True})
    except Exception as e:
        _log.error('register_anonymous_push_token: %s', e)
        return jsonify({'error': str(e)}), 500


# ── Admin push notification endpoints ────────────────────────────────────────

_VALID_RECIPIENT_TYPES = ('all', 'free', 'premium', 'specific')


@app.route('/api/admin/send-notification', methods=['POST'])
@_require_admin
def admin_send_notification():
    data           = request.get_json(silent=True) or {}
    title          = (data.get('title') or '').strip()
    message        = (data.get('message') or '').strip()
    recipient_type = (data.get('recipientType') or 'all').strip()
    user_id        = data.get('userId')

    if not title or not message:
        return jsonify({'error': 'title and message are required'}), 400
    if recipient_type not in _VALID_RECIPIENT_TYPES:
        return jsonify({'error': f'recipientType must be one of {_VALID_RECIPIENT_TYPES}'}), 400
    if recipient_type == 'specific' and not user_id:
        return jsonify({'error': 'userId is required for specific recipient'}), 400
    if not _ONESIGNAL_APP_ID:
        return jsonify({'error': 'OneSignal not configured — set ONESIGNAL_APP_ID env var'}), 503

    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'error': 'Database unavailable'}), 503

            if recipient_type == 'all':
                # Use OneSignal's own segment — reaches all subscribers
                # (logged-in users AND anonymous visitors who allowed push).
                ok, body = _onesignal_send({
                    'included_segments': ['All'],
                    'headings': {'en': title},
                    'contents': {'en': message},
                    'url':      _APP_URL,
                })
                if not ok:
                    return jsonify({'error': f'OneSignal API error: {body}'}), 500
                try:
                    sent_count = json.loads(body).get('recipients', 0)
                except Exception:
                    sent_count = 0
                cur.execute("""
                    INSERT INTO admin_notifications
                        (title, message, recipient_type, recipient_user_id, sent_count)
                    VALUES (%s, %s, %s, %s, %s)
                """, (title, message, recipient_type, None, sent_count))
                return jsonify({'success': True, 'sent_count': sent_count,
                                'message': f'Sent to {sent_count} device(s)'})

            if recipient_type == 'free':
                cur.execute("""
                    SELECT DISTINCT nt.onesignal_player_id
                      FROM notification_tokens nt
                      JOIN users u ON nt.user_id = u.id
                     WHERE u.is_premium = FALSE
                        OR u.premium_until IS NULL
                        OR u.premium_until < NOW()
                """)
                known_ids = [r[0] for r in cur.fetchall()]
                # Anonymous subscribers have no account → treated as free
                cur.execute('SELECT player_id FROM anonymous_notification_tokens')
                anon_ids = [r[0] for r in cur.fetchall()]
                player_ids = list({*known_ids, *anon_ids})
            elif recipient_type == 'premium':
                cur.execute("""
                    SELECT DISTINCT nt.onesignal_player_id
                      FROM notification_tokens nt
                      JOIN users u ON nt.user_id = u.id
                     WHERE u.is_premium = TRUE
                       AND u.premium_until IS NOT NULL
                       AND u.premium_until > NOW()
                """)
                player_ids = [r[0] for r in cur.fetchall()]
            else:  # specific
                cur.execute(
                    'SELECT onesignal_player_id FROM notification_tokens WHERE user_id = %s',
                    (int(user_id),)
                )
                player_ids = [r[0] for r in cur.fetchall()]

            if not player_ids:
                return jsonify({'success': True, 'sent_count': 0,
                                'message': 'No subscribed devices found for this audience'}), 200

            # OneSignal limit: 2000 player IDs per call — batch larger audiences
            for i in range(0, len(player_ids), 2000):
                ok, os_detail = _onesignal_send({
                    'include_player_ids': player_ids[i:i + 2000],
                    'headings': {'en': title},
                    'contents': {'en': message},
                    'url':      _APP_URL,
                })
                if not ok:
                    return jsonify({'error': f'OneSignal API error: {os_detail}'}), 500
            sent_count = len(player_ids)

            cur.execute("""
                INSERT INTO admin_notifications
                    (title, message, recipient_type, recipient_user_id, sent_count)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
            """, (title, message, recipient_type,
                  int(user_id) if recipient_type == 'specific' and user_id else None,
                  sent_count))
            notif_id = cur.fetchone()[0]

        return jsonify({'success': True, 'sent_count': sent_count,
                        'notification_id': notif_id, 'message': f'Sent to {sent_count} device(s)'})
    except Exception as e:
        _log.error('admin_send_notification: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/notification-history')
@_require_admin
def admin_notification_history():
    try:
        with _db() as (conn, cur):
            if conn is None:
                return jsonify({'notifications': []})
            cur.execute("""
                SELECT n.id, n.title, n.message, n.recipient_type,
                       n.sent_count, n.sent_at,
                       u.email AS recipient_user_email
                  FROM admin_notifications n
                  LEFT JOIN users u ON n.recipient_user_id = u.id
                 ORDER BY n.sent_at DESC
                 LIMIT 100
            """)
            cols = [d[0] for d in cur.description]
            rows = []
            for r in cur.fetchall():
                row = dict(zip(cols, r))
                if row.get('sent_at'):
                    row['sent_at'] = row['sent_at'].isoformat()
                rows.append(row)
            return jsonify({'notifications': rows})
    except Exception as e:
        _log.error('admin_notification_history: %s', e)
        return jsonify({'error': str(e)}), 500


# ── Payment endpoints (Korapay) ───────────────────────────────────────────────

_KORA_INIT_URL   = 'https://api.korapay.com/merchant/api/v1/charges/initialize'
_KORA_VERIFY_URL = 'https://api.korapay.com/merchant/api/v1/charges/{reference}'

# Amount thresholds in Naira
_PLAN_AMOUNTS = {
    '1week':   1_500,
    'monthly': 5_000,
    '3month':  15_000,
}
_PLAN_DAYS = {'1week': 7, 'monthly': 30, '3month': 90}


def _plan_days(plan, amount):
    """Resolve a plan id (preferred) or a raw Naira amount (fallback) to a day count."""
    if plan in _PLAN_DAYS:
        return _PLAN_DAYS[plan]
    if amount >= 14_000:
        return 90
    if amount and amount <= 2_000:
        return 7
    return 30


def _user_row_to_dict(row):
    if row is None:
        return None
    return {
        'id': row[0], 'name': row[1], 'email': row[2],
        'is_premium': bool(row[3]),
        'premium_until': row[4].isoformat() if row[4] else None,
    }


def _verify_korapay_signature(raw_body: bytes, signature: str) -> bool:
    """
    Korapay signs webhooks as HMAC-SHA256(secretKey, JSON-encoded `data` field
    only — not the whole envelope), sent in the x-korapay-signature header.
    Uses the SECRET key (never the public key) and a constant-time compare.
    """
    if not signature or not _KORAPAY_SECRET:
        return False
    try:
        payload = json.loads(raw_body)
    except Exception:
        return False
    data = payload.get('data')
    if not isinstance(data, dict):
        return False
    encoded = json.dumps(data, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    expected = hmac.new(_KORAPAY_SECRET.encode('utf-8'), encoded, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(expected, signature)
    except Exception:
        return False


def _process_successful_charge(reference, plan, amount, raw_payload, source):
    """
    Idempotently grant premium for a successful Korapay charge.

    Safe to call more than once for the same reference — the webhook can be
    retried by Korapay, and the client's /verify call can race the webhook.
    Only the first caller that finds the payments row not already 'success'
    performs the grant; every later call is a no-op that returns the same
    (already-current) user. Extends from the later of "now" or the user's
    existing premium_until, so a renewal never loses paid-for time.

    Returns (ok: bool, user: dict|None, reason: str).
    """
    with _db() as (conn, cur):
        if conn is None:
            return False, None, 'db_unavailable'

        cur.execute(
            "SELECT user_id, status, plan FROM payments WHERE reference=%s FOR UPDATE",
            (reference,)
        )
        row = cur.fetchone()
        if row is None:
            _log.warning(
                'Charge rejected: unknown reference=%s amount=%r type=%s plan=%s source=%s',
                reference, amount, type(amount).__name__, plan, source)
            return False, None, 'unknown_reference'

        user_id, status, stored_plan = row
        if status == 'success':
            _log.info('Idempotent replay ignored: reference=%s already processed (source=%s)',
                      reference, source)
            cur.execute(
                "SELECT id,name,email,is_premium,premium_until FROM users WHERE id=%s",
                (user_id,)
            )
            return True, _user_row_to_dict(cur.fetchone()), 'already_processed'

        plan = (plan or stored_plan or '').lower()
        days = _plan_days(plan, amount)

        expected = _PLAN_AMOUNTS.get(plan)
        if expected is not None and amount and abs(amount - expected) > 1:
            _log.warning('Amount mismatch: reference=%s plan=%s expected=%s got=%s (%s)',
                         reference, plan, expected, amount, type(amount).__name__)

        cur.execute("SELECT premium_until FROM users WHERE id=%s FOR UPDATE", (user_id,))
        urow = cur.fetchone()
        if urow is None:
            _log.warning('Charge rejected: reference=%s -> user_id=%s no longer exists',
                         reference, user_id)
            return False, None, 'user_not_found'

        now_u         = datetime.now(timezone.utc)
        current_until = urow[0]
        base    = current_until if (current_until and current_until > now_u) else now_u
        expires = base + timedelta(days=days)

        cur.execute(
            'UPDATE users SET is_premium=TRUE, premium_until=%s WHERE id=%s',
            (expires, user_id)
        )
        cur.execute(
            "UPDATE payments SET status='success', plan=%s, amount=%s, days_granted=%s, "
            "premium_until=%s, source=%s, raw_payload=%s::jsonb, processed_at=NOW() "
            "WHERE reference=%s",
            (plan, amount, days, expires, source, json.dumps(raw_payload or {}), reference)
        )
        _log.info('Premium granted: reference=%s user_id=%s plan=%s amount=%s days=%d '
                  'expires=%s source=%s', reference, user_id, plan, amount, days,
                  expires.date(), source)

        cur.execute(
            "SELECT id,name,email,is_premium,premium_until FROM users WHERE id=%s",
            (user_id,)
        )
        return True, _user_row_to_dict(cur.fetchone()), 'granted'


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

    # Record the attempt up front so a webhook/verify call for this reference
    # always has a row to match against (the idempotency guard in
    # _process_successful_charge rejects references it's never seen).
    try:
        with _db() as (conn, cur):
            if conn is not None:
                cur.execute(
                    "INSERT INTO payments (reference, user_id, plan, amount, currency, status) "
                    "VALUES (%s,%s,%s,%s,'NGN','initialized') ON CONFLICT (reference) DO NOTHING",
                    (ref, user['id'], plan, amount)
                )
    except Exception as e:
        _log.warning('payment_initialize: could not record payments row ref=%s: %s', ref, e)

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
            _log.warning('payment_initialize: Korapay init HTTP %s ref=%s body=%s',
                         r.status_code, ref, r.text[:300])
            return jsonify({'error': 'Korapay initialization failed', 'detail': r.text}), 502
        data = r.json().get('data', {})
        checkout_url = data.get('checkout_url') or data.get('authorization_url')
        if not checkout_url:
            _log.warning('payment_initialize: no checkout_url ref=%s data=%r', ref, data)
            return jsonify({'error': 'No checkout URL returned by Korapay'}), 502
        return jsonify({'checkout_url': checkout_url, 'reference': ref})
    except Exception as e:
        _log.warning('payment_initialize error: ref=%s %s', ref, e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/payment/webhook', methods=['POST'])
def payment_webhook():
    """Server-to-server Korapay notification — the source of truth for premium
    grants. Configure this URL (POST, `<APP_URL>/api/payment/webhook`) in the
    Korapay dashboard. Idempotent and independent of whether the customer's
    browser ever makes it back to /subscribe."""
    raw       = request.get_data()
    sig       = request.headers.get('x-korapay-signature', '')
    payload   = request.get_json(silent=True) or {}
    data      = payload.get('data') or {}
    reference = data.get('reference', '')

    if not _verify_korapay_signature(raw, sig):
        _log.warning('Webhook rejected: bad/missing signature reference=%s',
                     reference or '(missing)')
        return jsonify({'error': 'invalid signature'}), 401

    event = payload.get('event', '')
    if event != 'charge.success':
        _log.info('Webhook ignored: event=%r reference=%s', event, reference or '(missing)')
        return jsonify({'status': 'ignored'}), 200

    if not reference:
        _log.warning('Webhook rejected: charge.success with no reference, payload=%r', payload)
        return jsonify({'error': 'missing reference'}), 400

    if data.get('status') != 'success':
        _log.info('Webhook ignored: reference=%s data.status=%r (not success)',
                  reference, data.get('status'))
        return jsonify({'status': 'ignored'}), 200

    try:
        amount = float(data.get('amount') or 0)
    except (TypeError, ValueError):
        _log.warning('Webhook: reference=%s non-numeric amount=%r (type=%s)',
                     reference, data.get('amount'), type(data.get('amount')).__name__)
        amount = 0.0

    meta = data.get('metadata') or {}
    plan = str(meta.get('plan', '')).lower()

    ok, _user, reason = _process_successful_charge(
        reference=reference, plan=plan, amount=amount, raw_payload=payload, source='webhook')
    if not ok:
        _log.warning('Webhook processing failed: reference=%s reason=%s amount=%s plan=%s',
                     reference, reason, amount, plan)
    # 200 regardless of business-logic outcome so Korapay doesn't retry-storm
    # a reference we can never resolve (signature failures above still 4xx).
    return jsonify({'status': 'ok' if ok else reason}), 200


@app.route('/api/payment/verify')
def payment_verify():
    """Client-driven fast path: called when the browser lands back on
    /subscribe. Shares _process_successful_charge with the webhook, so
    whichever of the two arrives first performs the grant and the other
    is a no-op — this is a UX accelerant, not the source of truth."""
    user = _get_current_user()
    if not user:
        _log.warning('payment_verify rejected: unauthorized request')
        return jsonify({'error': 'Unauthorized'}), 401
    reference = request.args.get('reference', '')
    if not reference:
        _log.warning('payment_verify rejected: missing reference, user_id=%s', user['id'])
        return jsonify({'error': 'Reference required'}), 400
    if not _KORAPAY_SECRET:
        _log.warning('payment_verify rejected: Korapay not configured, reference=%s', reference)
        return jsonify({'error': 'Payment not configured'}), 503
    import requests as _req
    try:
        r = _req.get(
            _KORA_VERIFY_URL.format(reference=reference),
            headers={'Authorization': f'Bearer {_KORAPAY_SECRET}'},
            timeout=10,
        )
        if r.status_code != 200:
            _log.warning('payment_verify: Korapay verify HTTP %s reference=%s body=%s',
                         r.status_code, reference, r.text[:300])
            return jsonify({'error': 'Korapay verification failed', 'detail': r.text}), 502
        data = r.json().get('data', {})
        _log.info('Korapay verify raw: reference=%s status=%s amount=%r type=%s metadata=%r',
                  reference, data.get('status'), data.get('amount'),
                  type(data.get('amount')).__name__, data.get('metadata'))
        if data.get('status') != 'success':
            _log.warning('payment_verify: reference=%s not successful, status=%s',
                         reference, data.get('status'))
            return jsonify({'error': 'Payment not successful', 'status': data.get('status')}), 400
        try:
            amount = float(data.get('amount') or 0)
        except (ValueError, TypeError):
            _log.warning('payment_verify: reference=%s non-numeric amount=%r',
                         reference, data.get('amount'))
            amount = 0.0
        meta = data.get('metadata') or {}
        plan = str(meta.get('plan', '')).lower()

        ok, updated_user, reason = _process_successful_charge(
            reference=reference, plan=plan, amount=amount, raw_payload=data, source='verify')
        if not ok:
            _log.warning('payment_verify: grant failed reference=%s reason=%s', reference, reason)
            return jsonify({'error': f'Could not confirm payment ({reason})'}), 409
        return jsonify({'success': True, 'user': updated_user or user})
    except Exception as e:
        _log.warning('payment_verify error: reference=%s %s', reference, e)
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
@app.route('/saved')
@app.route('/admin')
@app.route('/admin/subscribers')
@app.route('/admin/push-notifications')
def spa_pages():
    return _serve_index()


@app.route('/default-icon')
def default_icon():
    """OneSignal v16 fetches /default-icon as the notification icon fallback."""
    return send_from_directory(DIST, 'icon-192.png')


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


try:
    _wc_sync_bracket()
except Exception as _e:
    _log.warning('WC bracket sync (startup) failed: %s', _e)

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
    _scheduler.add_job(_wc_sync_bracket, 'interval', minutes=30,
                       id='wc_bracket_sync', misfire_grace_time=120)
    _scheduler.add_job(_refresh_caches, 'interval', hours=6,
                       id='cache_refresh', misfire_grace_time=300)
    _scheduler.add_job(_notify_kickoff,         'interval', minutes=10,
                       id='notify_kickoff',      misfire_grace_time=60)
    _scheduler.add_job(_notify_results,         'interval', minutes=5,
                       id='notify_results',      misfire_grace_time=60)
    _scheduler.add_job(_notify_won_predictions, 'interval', minutes=10,
                       id='notify_won',          misfire_grace_time=60)
    _scheduler.add_job(_send_premium_upsell,    'cron',     hour=9, minute=0,
                       id='premium_upsell',      misfire_grace_time=3600)
    _scheduler.add_job(_daily_results_cleanup,  'cron',     hour=2, minute=0,
                       timezone='UTC', id='daily_results_cleanup',
                       misfire_grace_time=3600)
    _scheduler.start()
    import atexit as _atexit
    _atexit.register(lambda: _scheduler.shutdown(wait=False))
    _log.info('APScheduler started — live/2 min, results/30 min, daily cleanup/2am UTC, cache/6 h')
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

"""
app.py  —  Flask API for the Football Prediction Website.

Endpoints:
  GET /api/club/tips          daily club-football prediction cards
  GET /api/worldcup/fixtures  upcoming WC 2026 fixtures with predictions
  GET /api/worldcup/countdown seconds to next WC 2026 match

Club tips: uses result_model / goals_model / corners_model (train.py).
WC tips  : uses worldcup_result_model / worldcup_goals_model via the
           helper functions in worldcup_predict.py (no double-loading).

Run:
    python app.py
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

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

club_result  = _load('result_model.pkl')
club_goals   = _load('goals_model.pkl')
club_corners = _load('corners_model.pkl')

LEAGUE_MAP  = club_result['league_map']  if club_result  else {}
RES_ENCODER = club_result['result_encoder'] if club_result else None

# ── Load WC helpers & models ──────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from worldcup_predict import (
    load_data, get_team_form, get_major_form,
    get_h2h, build_feature_vector, get_importance, resolve_team,
)
from fetch_fixtures import get_daily_tips, get_club_tips

_df_intl     = None
wc_result    = _load('worldcup_result_model.pkl')
wc_goals     = _load('worldcup_goals_model.pkl')

def _intl():
    global _df_intl
    if _df_intl is None:
        _df_intl = load_data()
    return _df_intl

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


def _wc_predict(home, away, is_neutral=True):
    df = _intl()
    all_teams = sorted(set(df['home_team'].tolist() + df['away_team'].tolist()))
    ta = resolve_team(home,  all_teams)
    tb = resolve_team(away, all_teams)
    imp = get_importance('FIFA World Cup')

    hf = get_team_form(df, ta)
    af = get_team_form(df, tb)
    h2h = get_h2h(df, ta, tb)
    hf.update(get_major_form(df, ta))
    af.update(get_major_form(df, tb))

    X = build_feature_vector(hf, af, h2h,
                             wc_result['features'],
                             is_neutral=is_neutral,
                             tournament_importance=imp)

    rp = wc_result['model'].predict_proba(X)[0]   # [away,draw,home]
    gp = wc_goals['model'].predict_proba(X)[0]    # [under,over]

    return {
        'home_win':    round(float(rp[2]), 4),
        'draw':        round(float(rp[1]), 4),
        'away_win':    round(float(rp[0]), 4),
        'over_goals':  round(float(gp[1]), 4),
        'resolved_home': ta,
        'resolved_away': tb,
    }

WC_FIXTURES_RAW = [
    {"id":"wc1","group":"A","home":"Mexico","away":"Uruguay",
     "home_code":"mx","away_code":"uy",
     "date":"2026-06-11","time":"20:00","venue":"SoFi Stadium, Los Angeles"},
    {"id":"wc2","group":"B","home":"USA","away":"Panama",
     "home_code":"us","away_code":"pa",
     "date":"2026-06-11","time":"23:00","venue":"MetLife Stadium, New York"},
    {"id":"wc3","group":"C","home":"Brazil","away":"Morocco",
     "home_code":"br","away_code":"ma",
     "date":"2026-06-12","time":"17:00","venue":"Hard Rock Stadium, Miami"},
    {"id":"wc4","group":"D","home":"Argentina","away":"Poland",
     "home_code":"ar","away_code":"pl",
     "date":"2026-06-12","time":"20:00","venue":"AT&T Stadium, Dallas"},
    {"id":"wc5","group":"E","home":"France","away":"Japan",
     "home_code":"fr","away_code":"jp",
     "date":"2026-06-13","time":"17:00","venue":"Rose Bowl, Pasadena"},
    {"id":"wc6","group":"F","home":"Spain","away":"Nigeria",
     "home_code":"es","away_code":"ng",
     "date":"2026-06-13","time":"20:00","venue":"Lincoln Financial, Philadelphia"},
    {"id":"wc7","group":"G","home":"England","away":"Iran",
     "home_code":"gb-eng","away_code":"ir",
     "date":"2026-06-14","time":"14:00","venue":"Gillette Stadium, Boston"},
    {"id":"wc8","group":"H","home":"Germany","away":"Colombia",
     "home_code":"de","away_code":"co",
     "date":"2026-06-14","time":"17:00","venue":"Soldier Field, Chicago"},
    {"id":"wc9","group":"I","home":"Portugal","away":"Serbia",
     "home_code":"pt","away_code":"rs",
     "date":"2026-06-14","time":"20:00","venue":"Levi's Stadium, San Francisco"},
    {"id":"wc10","group":"J","home":"Netherlands","away":"Senegal",
     "home_code":"nl","away_code":"sn",
     "date":"2026-06-15","time":"17:00","venue":"Lumen Field, Seattle"},
]

def _build_wc_fixtures():
    fixtures = []
    for raw in WC_FIXTURES_RAW:
        try:
            pred = _wc_predict(raw['home'], raw['away'])
        except Exception as e:
            pred = {'home_win': 0.40, 'draw': 0.28, 'away_win': 0.32,
                    'over_goals': 0.52,
                    'resolved_home': raw['home'], 'resolved_away': raw['away']}

        ph, pd_, pa = pred['home_win'], pred['draw'], pred['away_win']
        pg = pred['over_goals']
        best_prob = max(ph, pd_, pa)
        if best_prob == ph:
            bet = {'label': f"{raw['home']} to Win", 'confidence': ph}
        elif best_prob == pa:
            bet = {'label': f"{raw['away']} to Win", 'confidence': pa}
        else:
            bet = {'label': 'Draw', 'confidence': pd_}

        fixtures.append({
            **raw,
            'result': {'home': ph, 'draw': pd_, 'away': pa},
            'over_goals': pg,
            'best_bet': bet,
        })
    return fixtures


print("Loading predictions…", flush=True)
_WC_FIXTURES = _build_wc_fixtures()
print(f"  WC fixtures ready ({len(_WC_FIXTURES)} matches)", flush=True)

# ── API routes ────────────────────────────────────────────────────────────────

_CACHE_TTL = 600  # 10 minutes

_CLUB_TIPS_CACHE: dict  = {"data": None, "ts": 0.0}


@app.route('/api/club/tips')
def club_tips():
    now_ts = time.time()
    if _CLUB_TIPS_CACHE["data"] is None or (now_ts - _CLUB_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_club_tips(_club_predict)
            _CLUB_TIPS_CACHE["data"] = tips
            _CLUB_TIPS_CACHE["ts"]   = now_ts
        except Exception as exc:
            app.logger.error("club-tips fetch failed: %s", exc)
            if _CLUB_TIPS_CACHE["data"] is None:
                _CLUB_TIPS_CACHE["data"] = []
    return jsonify(_CLUB_TIPS_CACHE["data"])


@app.route('/api/worldcup/fixtures')
def wc_fixtures():
    now = datetime.now(timezone.utc)
    upcoming = []
    for f in _WC_FIXTURES:
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


# ── Daily tips (live fixtures + predictions) ──────────────────────────────────

_DAILY_TIPS_CACHE: dict = {"data": None, "ts": 0.0}


@app.route("/api/daily-tips")
def daily_tips():
    now = time.time()
    if _DAILY_TIPS_CACHE["data"] is None or (now - _DAILY_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_daily_tips(_club_predict, _wc_predict)
            _DAILY_TIPS_CACHE["data"] = tips
            _DAILY_TIPS_CACHE["ts"]   = now
        except Exception as exc:
            app.logger.error("daily-tips fetch failed: %s", exc)
            return jsonify([])
    return jsonify(_DAILY_TIPS_CACHE["data"])


# ── Serve React build ─────────────────────────────────────────────────────────

DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist')


@app.route('/')
def index():
    return send_from_directory(DIST, 'index.html')


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(os.path.join(DIST, 'assets'), filename)


@app.route('/<path:path>')
def serve(path):
    file_path = os.path.join(DIST, path)
    if os.path.isfile(file_path):
        return send_from_directory(DIST, path)
    return send_from_directory(DIST, 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

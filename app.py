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
import os, sys, pickle, warnings, time, logging
import numpy as np
from datetime import datetime, timezone, timedelta

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

club_result  = _load('result_model.pkl')
club_goals   = _load('goals_model.pkl')
club_corners = _load('corners_model.pkl')

LEAGUE_MAP  = club_result['league_map']  if club_result  else {}
RES_ENCODER = club_result['result_encoder'] if club_result else None

# ── Load WC helpers & models ──────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from worldcup_predict import predict_match
from fetch_fixtures import get_daily_tips, get_club_tips, get_intl_tips

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

        fixtures.append({
            **raw,
            'utc_kickoff':  f"{raw['date']}T{raw['time']}:00Z",
            'result':       {'home': ph, 'draw': pd_, 'away': pa},
            'over_goals':   pg,
            'btts':         pred.get('btts',         0.48),
            'over_corners': pred.get('over_corners', 0.52),
            'best_bet':     bet,
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
_INTL_TIPS_CACHE:  dict = {"data": None, "ts": 0.0}


@app.route("/api/intl/fixtures")
def intl_fixtures():
    now = time.time()
    if _INTL_TIPS_CACHE["data"] is None or (now - _INTL_TIPS_CACHE["ts"]) > _CACHE_TTL:
        try:
            tips = get_intl_tips(_wc_predict)
            _INTL_TIPS_CACHE["data"] = tips
            _INTL_TIPS_CACHE["ts"]   = now
        except Exception as exc:
            app.logger.error("intl-fixtures fetch failed: %s", exc)
            if _INTL_TIPS_CACHE["data"] is None:
                _INTL_TIPS_CACHE["data"] = []
    return jsonify(_INTL_TIPS_CACHE["data"])


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

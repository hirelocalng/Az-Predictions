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
import os, sys, pickle, warnings, time, math, logging
import numpy as np
import pandas as pd
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
from worldcup_predict import (
    load_data, get_team_form, get_major_form,
    get_h2h, build_feature_vector, get_importance, resolve_team,
)
from fetch_fixtures import get_daily_tips, get_club_tips, get_intl_tips

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


# ── FIFA World Rankings (approx. June 2025, WC 2026 participants) ─────────────

_FIFA_PTS = {
    # Top-tier qualifiers
    'Argentina':     1897, 'France':        1870, 'Spain':         1852,
    'England':       1818, 'Brazil':        1793, 'Portugal':      1772,
    'Netherlands':   1749, 'Belgium':       1737, 'Germany':       1728,
    'Colombia':      1709, 'Uruguay':       1704, 'Croatia':       1683,
    'Morocco':       1672, 'United States': 1658, 'USA':           1658,
    'Japan':         1651, 'Mexico':        1641, 'Senegal':       1632,
    'Switzerland':   1620, 'Ecuador':       1618,
    'Iran':          1610, 'Australia':     1601,
    'South Korea':   1591,
    'Tunisia':       1554, 'Paraguay':      1539, 'Qatar':         1526,
    'Canada':        1524, 'Saudi Arabia':  1522,
    'Panama':        1506, 'Iraq':          1502,
    'Algeria':       1499, 'Uzbekistan':    1498,
    'Ghana':         1490, 'New Zealand':   1452,
    # WC 2026 qualified teams added for prediction fallback
    'Sweden':        1578, 'Austria':       1572, 'Scotland':      1555,
    'Norway':        1548, 'Turkey':        1544, 'Czech Republic': 1518,
    'South Africa':  1505, 'Cape Verde':    1497, 'Egypt':         1492,
    'Ivory Coast':   1568, 'Bosnia and Herzegovina': 1485,
    'DR Congo':      1442, 'Haiti':         1335, 'Jordan':        1392,
    'Curacao':       1340,
}

# H2H win-rate for the listed-first team (approximated from historical meetings)
_H2H_WIN_RATE = {
    ('Argentina', 'Brazil'):    0.45, ('Brazil', 'Argentina'):    0.40,
    ('Germany', 'France'):      0.45, ('France', 'Germany'):      0.50,
    ('Spain', 'Portugal'):      0.50, ('Portugal', 'Spain'):      0.40,
    ('England', 'Germany'):     0.40, ('Germany', 'England'):     0.50,
    ('Netherlands', 'Germany'): 0.42, ('Germany', 'Netherlands'): 0.48,
    ('Brazil', 'France'):       0.38, ('France', 'Brazil'):       0.50,
    ('Argentina', 'France'):    0.40, ('France', 'Argentina'):    0.50,
    ('Spain', 'Germany'):       0.50, ('Germany', 'Spain'):       0.40,
    ('Brazil', 'England'):      0.40, ('England', 'Brazil'):      0.42,
}


def _form_from_ranking(team: str) -> dict:
    """Derive team form features from FIFA ranking points."""
    pts = _FIFA_PTS.get(team, 1400)
    n = max(0.0, min(1.0, (pts - 1200) / 700.0))   # 0 = weakest, 1 = best

    win_rate       = round(0.22 + n * 0.48, 4)      # 0.22–0.70
    draw_rate      = round(0.28 - n * 0.08, 4)      # 0.28–0.20
    loss_rate      = round(max(0.02, 1.0 - win_rate - draw_rate), 4)
    goals_scored   = round(0.85 + n * 1.25, 4)      # 0.85–2.10
    goals_conceded = round(1.65 - n * 0.90, 4)      # 1.65–0.75
    form_pts       = round(win_rate * 3 + draw_rate, 4)
    goal_diff      = round(goals_scored - goals_conceded, 4)
    major_win_rate = round(win_rate * 0.88, 4)
    major_form_pts = round(major_win_rate * 3 + draw_rate * 0.85, 4)

    return {
        'win_rate':        win_rate,   'draw_rate':       draw_rate,
        'loss_rate':       loss_rate,  'goals_scored':    goals_scored,
        'goals_conceded':  goals_conceded, 'form_pts':    form_pts,
        'form_count':      10.0,       'goal_diff':       goal_diff,
        'major_win_rate':  major_win_rate, 'major_form_pts': major_form_pts,
        'major_count':     6.0,
    }


def _btts_prob(home_gs: float, away_gs: float) -> float:
    """P(both teams score ≥1 goal) using Poisson approximation."""
    p_h = 1.0 - math.exp(-max(0.05, home_gs))
    p_a = 1.0 - math.exp(-max(0.05, away_gs))
    return round(p_h * p_a, 4)


def _corners_prob(home_gs: float, away_gs: float,
                  home_gc: float, away_gc: float) -> float:
    """Estimate P(total corners > 9.5) from team attacking/defensive style."""
    avg = 1.2
    exp = (9.2
           + (home_gs - avg) * 1.1
           + (away_gs - avg) * 1.1
           + (home_gc - avg) * 0.35
           + (away_gc - avg) * 0.35)
    z = (exp - 9.5) / (2.2 * math.sqrt(2))
    p = round(0.5 * (1.0 + math.erf(z)), 4)
    return max(0.20, min(0.82, p))


def _pure_ranking_predict(home: str, away: str) -> dict:
    """ELO-style last-resort — no ML models or data files required."""
    hp       = _FIFA_PTS.get(home, 1400)
    ap       = _FIFA_PTS.get(away, 1400)
    diff     = hp - ap
    p_home_r = 1.0 / (1.0 + 10.0 ** (-diff / 600.0))
    abs_diff = abs(diff)
    p_draw   = round(max(0.12, min(0.30, 0.27 - abs_diff * 0.00014)), 4)
    scale    = 1.0 - p_draw
    p_home   = round(p_home_r * scale, 4)
    p_away   = round((1.0 - p_home_r) * scale, 4)
    avg_pts  = (hp + ap) / 2.0
    p_over   = round(max(0.35, min(0.72,
                    0.48 + (avg_pts - 1550) / 2000.0 - abs_diff / 4000.0)), 4)
    n_h = max(0.0, min(1.0, (hp - 1200) / 700.0))
    n_a = max(0.0, min(1.0, (ap - 1200) / 700.0))
    h_gs = 0.85 + n_h * 1.25;  a_gs = 0.85 + n_a * 1.25
    h_gc = 1.65 - n_h * 0.90;  a_gc = 1.65 - n_a * 0.90
    return {
        'home_win':     p_home,   'draw':         p_draw,
        'away_win':     p_away,   'over_goals':   p_over,
        'btts':         _btts_prob(h_gs, a_gs),
        'over_corners': _corners_prob(h_gs, a_gs, h_gc, a_gc),
        'resolved_home': home,    'resolved_away': away,
    }


def _wc_predict(home, away, is_neutral=True):
    """
    Predict international match — same logic as worldcup_predict.py terminal tool.
    Tier 1: exact historical-data path (identical to terminal output).
    Tier 2: worldcup models + FIFA ranking-derived features (Railway fallback).
    Tier 3: pure ELO formula (last resort, no files needed).
    Failures are logged so the Railway log reveals any data/model issues.
    """
    # ── Tier 1: exact same path as worldcup_predict.py predict() ─────────────
    try:
        df        = _intl()          # loads data/results.csv
        all_teams = sorted(set(df['home_team'].tolist() + df['away_team'].tolist()))
        ta        = resolve_team(home, all_teams)
        tb        = resolve_team(away, all_teams)
        imp       = get_importance('FIFA World Cup')
        hf        = get_team_form(df, ta)
        af        = get_team_form(df, tb)
        h2h       = get_h2h(df, ta, tb)
        hf.update(get_major_form(df, ta))
        af.update(get_major_form(df, tb))
        X         = build_feature_vector(hf, af, h2h, wc_result['features'],
                                         is_neutral=is_neutral,
                                         tournament_importance=imp)
        rp        = wc_result['model'].predict_proba(X)[0]   # [away, draw, home]
        gp        = wc_goals['model'].predict_proba(X)[0]    # [under, over]
        return {
            'home_win':      round(float(rp[2]), 4),
            'draw':          round(float(rp[1]), 4),
            'away_win':      round(float(rp[0]), 4),
            'over_goals':    round(float(gp[1]), 4),
            'btts':          _btts_prob(hf['goals_scored'], af['goals_scored']),
            'over_corners':  _corners_prob(hf['goals_scored'], af['goals_scored'],
                                           hf['goals_conceded'], af['goals_conceded']),
            'resolved_home': ta,
            'resolved_away': tb,
        }
    except Exception as exc:
        _log.warning('wc_predict Tier1 failed %s vs %s — %s', home, away, exc)

    # ── Tier 2: worldcup models + FIFA ranking features ───────────────────────
    if wc_result and wc_goals:
        try:
            hf       = _form_from_ranking(home)
            af       = _form_from_ranking(away)
            h2h_rate = _H2H_WIN_RATE.get((home, away),
                       _H2H_WIN_RATE.get((away, home), 1 / 3))
            h2h      = {'count': 5.0, 'home_win_rate': h2h_rate}
            imp      = get_importance('FIFA World Cup')
            X        = build_feature_vector(hf, af, h2h, wc_result['features'],
                                            is_neutral=is_neutral,
                                            tournament_importance=imp)
            rp       = wc_result['model'].predict_proba(X)[0]
            gp       = wc_goals['model'].predict_proba(X)[0]
            return {
                'home_win':      round(float(rp[2]), 4),
                'draw':          round(float(rp[1]), 4),
                'away_win':      round(float(rp[0]), 4),
                'over_goals':    round(float(gp[1]), 4),
                'btts':          _btts_prob(hf['goals_scored'], af['goals_scored']),
                'over_corners':  _corners_prob(hf['goals_scored'], af['goals_scored'],
                                               hf['goals_conceded'], af['goals_conceded']),
                'resolved_home': home,
                'resolved_away': away,
            }
        except Exception as exc:
            _log.warning('wc_predict Tier2 failed %s vs %s — %s', home, away, exc)

    # ── Tier 3: pure ELO formula ──────────────────────────────────────────────
    return _pure_ranking_predict(home, away)

WC_FIXTURES_RAW = [
    # Group A — opener: Mexico vs South Africa, Estadio Azteca
    {"id":"wc1","group":"A","home":"Mexico","away":"South Africa",
     "home_code":"mx","away_code":"za",
     "date":"2026-06-11","time":"15:00","venue":"Estadio Azteca, Mexico City"},
    # Group B — opener: Canada vs Bosnia and Herzegovina, BMO Field
    {"id":"wc2","group":"B","home":"Canada","away":"Bosnia and Herzegovina",
     "home_code":"ca","away_code":"ba",
     "date":"2026-06-12","time":"15:00","venue":"BMO Field, Toronto"},
    # Group C — opener: Brazil vs Morocco, MetLife Stadium
    {"id":"wc3","group":"C","home":"Brazil","away":"Morocco",
     "home_code":"br","away_code":"ma",
     "date":"2026-06-13","time":"18:00","venue":"MetLife Stadium, East Rutherford NJ"},
    # Group D — opener: USA vs Paraguay, SoFi Stadium
    {"id":"wc4","group":"D","home":"USA","away":"Paraguay",
     "home_code":"us","away_code":"py",
     "date":"2026-06-12","time":"21:00","venue":"SoFi Stadium, Los Angeles"},
    # Group E — opener: Germany vs Curacao, NRG Stadium
    {"id":"wc5","group":"E","home":"Germany","away":"Curacao",
     "home_code":"de","away_code":"cw",
     "date":"2026-06-14","time":"13:00","venue":"NRG Stadium, Houston"},
    # Group F — opener: Netherlands vs Japan, AT&T Stadium
    {"id":"wc6","group":"F","home":"Netherlands","away":"Japan",
     "home_code":"nl","away_code":"jp",
     "date":"2026-06-14","time":"16:00","venue":"AT&T Stadium, Dallas"},
    # Group G — opener: Belgium vs Egypt, Lumen Field
    {"id":"wc7","group":"G","home":"Belgium","away":"Egypt",
     "home_code":"be","away_code":"eg",
     "date":"2026-06-15","time":"15:00","venue":"Lumen Field, Seattle"},
    # Group H — opener: Spain vs Cape Verde, Mercedes-Benz Stadium
    {"id":"wc8","group":"H","home":"Spain","away":"Cape Verde",
     "home_code":"es","away_code":"cv",
     "date":"2026-06-15","time":"12:00","venue":"Mercedes-Benz Stadium, Atlanta"},
    # Group I — opener: France vs Senegal, MetLife Stadium
    {"id":"wc9","group":"I","home":"France","away":"Senegal",
     "home_code":"fr","away_code":"sn",
     "date":"2026-06-16","time":"15:00","venue":"MetLife Stadium, East Rutherford NJ"},
    # Group J — opener: Argentina vs Algeria, Arrowhead Stadium
    {"id":"wc10","group":"J","home":"Argentina","away":"Algeria",
     "home_code":"ar","away_code":"dz",
     "date":"2026-06-16","time":"21:00","venue":"Arrowhead Stadium, Kansas City"},
    # Group K — opener: Portugal vs DR Congo, NRG Stadium
    {"id":"wc11","group":"K","home":"Portugal","away":"DR Congo",
     "home_code":"pt","away_code":"cd",
     "date":"2026-06-17","time":"13:00","venue":"NRG Stadium, Houston"},
    # Group L — opener: England vs Croatia, AT&T Stadium
    {"id":"wc12","group":"L","home":"England","away":"Croatia",
     "home_code":"gb-eng","away_code":"hr",
     "date":"2026-06-17","time":"16:00","venue":"AT&T Stadium, Dallas"},
]

def _build_wc_fixtures():
    fixtures = []
    for raw in WC_FIXTURES_RAW:
        try:
            pred = _wc_predict(raw['home'], raw['away'])
        except Exception as exc:
            _log.error('_build_wc_fixtures failed for %s vs %s: %s',
                       raw['home'], raw['away'], exc)
            pred = _pure_ranking_predict(raw['home'], raw['away'])

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
            'result':       {'home': ph, 'draw': pd_, 'away': pa},
            'over_goals':   pg,
            'btts':         pred.get('btts',         _btts_prob(1.2, 1.2)),
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

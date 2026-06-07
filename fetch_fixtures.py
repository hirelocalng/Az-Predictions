"""
fetch_fixtures.py — Fetch today's fixtures from football-data.org,
score each through XGBoost models, and return predictions.

Endpoints used:
  GET /v4/matches?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD
"""

import difflib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── football-data.org API ──────────────────────────────────────────────────────

_FD_URL     = "https://api.football-data.org/v4/matches"
_FD_HEADERS = {"X-Auth-Token": "118333be3eb84d0ca4e10740f6d62255"}

_MIN_CONF = 0.55
_MAX_TIPS = 5

# Statuses from football-data.org that mean the match is over / won't be played
_FINISHED_STATUSES = frozenset({
    "FINISHED", "POSTPONED", "SUSPENDED", "CANCELLED", "AWARDED",
})

# Statuses that mean the match is currently in progress
_LIVE_STATUSES = frozenset({"IN_PLAY", "PAUSED"})

# competition.code values for national-team tournaments → route to WC models
_INTL_COMP_CODES = frozenset({
    "WC", "EC", "AMC", "NL", "WCQ", "ECQ",
    "AFCON", "CAC", "GCUP", "AFC", "CAN",
})

# competition.code → Matches.csv Division code
_COMP_TO_LEAGUE = {
    "PL":  "E0",   # Premier League
    "ELC": "E0",   # Championship
    "PD":  "SP1",  # La Liga
    "BL1": "D1",   # Bundesliga
    "SA":  "I1",   # Serie A
    "FL1": "F1",   # Ligue 1
    "PPL": "P1",   # Primeira Liga
    "DED": "N1",   # Eredivisie
    "BSA": "BRA",  # Brasileirão
    "CL":  "E0",   # Champions League (use EPL history as proxy)
    "EL":  "E0",   # Europa League
    "ECL": "E0",   # Conference League
}

# competition name keywords that indicate international / national-team matches
_INTL_KEYWORDS = frozenset({
    "world cup", "euro", "copa america", "nations league", "gold cup",
    "africa cup", "asian cup", "friendly", "international", "warm-up",
    "concacaf", "conmebol", "afcon",
})

# ── Extraction helpers (football-data.org response shape) ─────────────────────

def _extract_teams(match: dict):
    home = match.get("homeTeam", {}).get("name", "").strip()
    away = match.get("awayTeam", {}).get("name", "").strip()
    return home, away


def _extract_league(match: dict) -> str:
    return match.get("competition", {}).get("name", "").lower()


def _extract_comp_code(match: dict) -> str:
    return match.get("competition", {}).get("code", "")


def _extract_logo(match: dict, side: str) -> str:
    key = "homeTeam" if side == "home" else "awayTeam"
    return match.get(key, {}).get("crest", "")


def _get_match_status(match: dict) -> str:
    return match.get("status", "").upper()


def _is_finished(match: dict) -> bool:
    return _get_match_status(match) in _FINISHED_STATUSES


def _extract_time(match: dict) -> str:
    utc = match.get("utcDate", "")
    if utc:
        try:
            dt = datetime.fromisoformat(utc.replace("Z", "+00:00"))
            return dt.strftime("%H:%M")
        except Exception:
            pass
    return "?"

# ── Fetch today's fixtures ────────────────────────────────────────────────────

def _fetch_today() -> list:
    """Return all of today's non-finished matches from football-data.org."""
    today = datetime.now().strftime("%Y-%m-%d")
    r = requests.get(
        _FD_URL,
        headers=_FD_HEADERS,
        params={"dateFrom": today, "dateTo": today},
        timeout=15,
    )
    r.raise_for_status()
    matches = r.json().get("matches", [])
    return [m for m in matches if not _is_finished(m)]

# ── Historical club stats ─────────────────────────────────────────────────────

_hist_cache: dict = {}

_DEFAULT_STATS = {
    "gf": 1.2, "ga": 1.2, "sot": 4.5, "corners": 5.0,
    "win": 0.33, "draw": 0.27, "hwn": 0.40, "awn": 0.25,
}


def _load_history(league_code: str) -> pd.DataFrame:
    if league_code in _hist_cache:
        return _hist_cache[league_code]

    frames = []
    try:
        df = pd.read_csv(
            "data/Matches.csv",
            usecols=[
                "Division", "MatchDate", "HomeTeam", "AwayTeam",
                "FTHome", "FTAway", "HomeTarget", "AwayTarget",
                "HomeCorners", "AwayCorners",
            ],
        )
        sub = (
            df[df["Division"] == league_code]
            .rename(columns={
                "MatchDate": "date", "HomeTeam": "home", "AwayTeam": "away",
                "FTHome": "hg",    "FTAway": "ag",
                "HomeTarget": "h_sot", "AwayTarget": "a_sot",
                "HomeCorners": "h_cor", "AwayCorners": "a_cor",
            })[["date", "home", "away", "hg", "ag", "h_sot", "a_sot", "h_cor", "a_cor"]]
        )
        frames.append(sub)
    except Exception:
        pass

    for csv_path, code in [("data/brazil_2025.csv", "BRA"), ("data/argentina_2025.csv", "ARG")]:
        if league_code == code:
            try:
                fb = pd.read_csv(csv_path).rename(columns={
                    "home_team": "home", "away_team": "away",
                    "home_goals": "hg", "away_goals": "ag",
                    "home_sot": "h_sot", "away_sot": "a_sot",
                    "home_corners": "h_cor", "away_corners": "a_cor",
                })[["date", "home", "away", "hg", "ag", "h_sot", "a_sot", "h_cor", "a_cor"]]
                frames.append(fb)
            except Exception:
                pass

    if not frames:
        _hist_cache[league_code] = pd.DataFrame()
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = (
        combined.dropna(subset=["date", "hg", "ag"])
        .sort_values("date").reset_index(drop=True)
    )
    _hist_cache[league_code] = combined
    return combined


def _resolve(name: str, teams: list) -> str | None:
    nl = name.lower().strip()
    for t in teams:
        if t.lower().strip() == nl:
            return t
    ms = difflib.get_close_matches(name, teams, n=1, cutoff=0.4)
    return ms[0] if ms else None


def _rolling(df: pd.DataFrame, team: str, n: int = 5) -> dict:
    mask   = (df["home"] == team) | (df["away"] == team)
    recent = df[mask].sort_values("date").tail(n)
    if recent.empty:
        return dict(_DEFAULT_STATS)

    gf = ga = sot_s = sot_n = cor_s = cor_n = 0.0
    wins = draws = pts = hwin = hgames = awin = agames = 0

    for _, row in recent.iterrows():
        ih = (row["home"] == team)
        g  = float(row["hg"] if ih else row["ag"])
        gc = float(row["ag"] if ih else row["hg"])
        gf += g; ga += gc

        s = row["h_sot" if ih else "a_sot"]
        c = row["h_cor" if ih else "a_cor"]
        if pd.notna(s) and float(s) > 0: sot_s += float(s); sot_n += 1
        if pd.notna(c) and float(c) > 0: cor_s += float(c); cor_n += 1

        if g > gc:    wins += 1; pts += 3
        elif g == gc: draws += 1; pts += 1
        if ih:   hgames += 1; hwin += int(g > gc)
        else:    agames += 1; awin += int(g > gc)

    n_m = len(recent)
    return {
        "gf":      gf / n_m,
        "ga":      ga / n_m,
        "sot":     sot_s / sot_n if sot_n else 4.5,
        "corners": cor_s / cor_n if cor_n else 5.0,
        "win":     wins  / n_m,
        "draw":    draws / n_m,
        "hwn":     hwin  / hgames if hgames else min(wins / n_m + 0.08, 1.0),
        "awn":     awin  / agames if agames else max(wins / n_m - 0.08, 0.0),
        "_pts":    pts,
    }

# ── Best bet selector ─────────────────────────────────────────────────────────

def _best_bet(home_name, away_name, ph, pd_, pa, pg, pc=None):
    options = []

    result_conf = max(ph, pd_, pa)
    if result_conf == ph:
        options.append((f"{home_name} to Win", result_conf))
    elif result_conf == pa:
        options.append((f"{away_name} to Win", result_conf))
    else:
        options.append(("Draw", result_conf))

    goals_conf = max(pg, 1.0 - pg)
    options.append(("Over 2.5 Goals" if pg >= 0.5 else "Under 2.5 Goals", goals_conf))

    if pc is not None:
        corners_conf = max(pc, 1.0 - pc)
        options.append(("Corners Over 9.5" if pc >= 0.5 else "Corners Under 9.5", corners_conf))

    label, best_conf = max(options, key=lambda x: x[1])
    return label, round(best_conf, 4), round(max(x[1] for x in options), 4)

# ── Shared feature builder for club matches ───────────────────────────────────

def _score_club_match(match, club_predict_fn):
    """
    Given a football-data.org match dict and a predict function, return a
    scored result dict or None if it should be skipped.
    """
    home_name, away_name = _extract_teams(match)
    if not home_name or not away_name:
        return None

    comp_code  = _extract_comp_code(match)
    league_code = _COMP_TO_LEAGUE.get(comp_code, "E0")
    comp_name  = match.get("competition", {}).get("name", "")
    status     = _get_match_status(match)
    time_str   = _extract_time(match)
    home_crest = _extract_logo(match, "home")
    away_crest = _extract_logo(match, "away")

    df_hist = _load_history(league_code)
    if df_hist.empty:
        df_hist = _load_history("E0")

    if not df_hist.empty:
        teams = sorted(set(df_hist["home"].tolist() + df_hist["away"].tolist()))
        h_res = _resolve(home_name, teams)
        a_res = _resolve(away_name, teams)
        hs  = _rolling(df_hist, h_res) if h_res else dict(_DEFAULT_STATS)
        as_ = _rolling(df_hist, a_res) if a_res else dict(_DEFAULT_STATS)
    else:
        hs  = dict(_DEFAULT_STATS)
        as_ = dict(_DEFAULT_STATS)

    form5_h = hs.pop("_pts", round((hs["win"] * 3 + hs["draw"]) * 5))
    form5_a = as_.pop("_pts", round((as_["win"] * 3 + as_["draw"]) * 5))

    try:
        ph, pd_, pa, pg, pc = club_predict_fn(
            hs, as_, 0, form5_h, form5_a, 2.60, 3.10, 2.80, league_code
        )
    except Exception:
        return None

    best_prob = max(ph, pd_, pa)
    if best_prob == ph:
        bet = {"label": f"{home_name} Win", "odds": 2.60, "confidence": round(ph, 4)}
    elif best_prob == pa:
        bet = {"label": f"{away_name} Win", "odds": 2.80, "confidence": round(pa, 4)}
    else:
        bet = {"label": "Draw", "odds": 3.10, "confidence": round(pd_, 4)}

    return {
        "league":       comp_name,
        "home_team":    home_name,
        "away_team":    away_name,
        "home_color":   "#444",
        "away_color":   "#666",
        "home_crest":   home_crest,
        "away_crest":   away_crest,
        "time":         time_str,
        "date_label":   "Live" if status in _LIVE_STATUSES else "Today",
        "status":       status,
        "result":       {"home": round(ph, 4), "draw": round(pd_, 4), "away": round(pa, 4)},
        "over_goals":   round(pg, 4),
        "over_corners": round(pc, 4),
        "best_bet":     bet,
        "_conf":        max(ph, pd_, pa, max(pg, 1 - pg)),
    }

# ── Club tips (domestic leagues only) ────────────────────────────────────────

def get_club_tips(club_predict_fn) -> list:
    """
    Fetch today's matches from football-data.org, filter to domestic club
    competitions, run predictions, return all scored matches (no confidence
    floor — callers decide whether to show 'no matches' message).
    """
    matches = _fetch_today()
    tips = []

    for idx, match in enumerate(matches):
        comp_code  = _extract_comp_code(match)
        league_raw = _extract_league(match)

        # Skip national-team competitions
        if comp_code in _INTL_COMP_CODES:
            continue
        if any(kw in league_raw for kw in _INTL_KEYWORDS):
            continue

        scored = _score_club_match(match, club_predict_fn)
        if scored is None:
            continue

        scored["id"] = idx + 1
        scored.pop("_conf", None)
        tips.append(scored)

    return tips

# ── Daily tips (all competitions, top-5 by confidence ≥ 55%) ─────────────────

def get_daily_tips(club_predict_fn, wc_predict_fn) -> list:
    """
    Fetch today's matches from football-data.org, score every match through
    the appropriate model, filter to ≥55% confidence, return the top 5.
    """
    matches = _fetch_today()
    results = []

    for idx, match in enumerate(matches):
        home_name, away_name = _extract_teams(match)
        if not home_name or not away_name:
            continue

        comp_code  = _extract_comp_code(match)
        league_raw = _extract_league(match)
        status     = _get_match_status(match)
        time_str   = _extract_time(match)
        home_logo  = _extract_logo(match, "home")
        away_logo  = _extract_logo(match, "away")
        comp_label = match.get("competition", {}).get("name", "") or league_raw.title()
        date_label = "Live" if status in _LIVE_STATUSES else "Today"

        is_intl = (comp_code in _INTL_COMP_CODES
                   or any(kw in league_raw for kw in _INTL_KEYWORDS))

        # ── International / national-team match ───────────────────────────
        if is_intl:
            pred = None
            try:
                pred = wc_predict_fn(home_name, away_name, is_neutral=True)
            except Exception:
                pass

            if pred:
                ph, pd_, pa = pred["home_win"], pred["draw"], pred["away_win"]
                pg           = pred["over_goals"]
                display_home = pred["resolved_home"]
                display_away = pred["resolved_away"]
                bet_lbl, bet_conf, overall = _best_bet(
                    display_home, display_away, ph, pd_, pa, pg
                )
            else:
                # Fall back to club models with default stats
                hs  = dict(_DEFAULT_STATS)
                as_ = dict(_DEFAULT_STATS)
                form5_h = round((hs["win"] * 3 + hs["draw"]) * 5)
                form5_a = round((as_["win"] * 3 + as_["draw"]) * 5)
                try:
                    ph, pd_, pa, pg, pc = club_predict_fn(
                        hs, as_, 0, form5_h, form5_a, 2.60, 3.10, 2.80, "E0"
                    )
                except Exception:
                    continue
                display_home, display_away = home_name, away_name
                bet_lbl, bet_conf, overall = _best_bet(
                    home_name, away_name, ph, pd_, pa, pg, pc
                )

            if overall < _MIN_CONF:
                continue

            results.append({
                "id":               f"dt{idx+1}",
                "league":           comp_label,
                "home_team":        display_home,
                "away_team":        display_away,
                "home_crest":       home_logo,
                "away_crest":       away_logo,
                "is_international": True,
                "time":             time_str,
                "date_label":       date_label,
                "result":           {"home": round(ph, 4), "draw": round(pd_, 4), "away": round(pa, 4)},
                "over_goals":       round(pg, 4),
                "best_bet":         {"label": bet_lbl, "confidence": bet_conf},
                "_conf":            overall,
            })

        # ── Club match ────────────────────────────────────────────────────
        else:
            scored = _score_club_match(match, club_predict_fn)
            if scored is None:
                continue

            overall = scored.pop("_conf")
            if overall < _MIN_CONF:
                continue

            results.append({
                "id":               f"dt{idx+1}",
                "league":           comp_label,
                "home_team":        scored["home_team"],
                "away_team":        scored["away_team"],
                "home_crest":       home_logo,
                "away_crest":       away_logo,
                "is_international": False,
                "time":             time_str,
                "date_label":       date_label,
                "result":           scored["result"],
                "over_goals":       scored["over_goals"],
                "over_corners":     scored["over_corners"],
                "best_bet":         scored["best_bet"],
                "_conf":            overall,
            })

    results.sort(key=lambda x: x["_conf"], reverse=True)
    out = results[:_MAX_TIPS]
    for p in out:
        p.pop("_conf", None)
    return out

"""
fetch_fixtures.py — Fetch today's fixtures via football-data.org, score with
XGBoost models, and return predictions with ≥60% confidence on any market.

Competitions polled (per-competition codes + general /v4/matches endpoint for
international friendlies and WC warm-up matches):
  WC, CL, PL, PD (La Liga), BL1, SA, FL1, BSA, AR
"""

import difflib
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

_API_KEY  = "118333be3eb84d0ca4e10740f6d62255"
_BASE_URL = "https://api.football-data.org/v4"
_HEADERS  = {"X-Auth-Token": _API_KEY}
_STATUSES = {"TIMED", "SCHEDULED", "IN_PLAY"}
_MIN_CONF = 0.55   # minimum confidence on ANY market to include
_MAX_TIPS = 10     # safety cap on results returned

# ── Competition metadata ──────────────────────────────────────────────────────

# Per-competition codes fetched as fallback / supplement
_COMP_CODES = ["WC", "CL", "PL", "PD", "BL1", "SA", "FL1", "BSA", "AR"]

# football-data.org competition code → Matches.csv Division code (club model)
_COMP_TO_LEAGUE = {
    "CL":  "E0",   # Champions League   (EPL calibration)
    "EL":  "E0",   # Europa League
    "PL":  "E0",   # English Premier League
    "ELC": "E0",   # Championship
    "PD":  "SP1",  # Spanish La Liga
    "SD":  "SP2",  # Spanish Segunda
    "BL1": "D1",   # German Bundesliga
    "BL2": "D2",   # 2. Bundesliga
    "SA":  "I1",   # Italian Serie A
    "SB":  "I2",   # Serie B
    "FL1": "F1",   # French Ligue 1
    "FL2": "F2",   # Ligue 2
    "DED": "N1",   # Eredivisie
    "PPL": "P1",   # Portuguese Primeira Liga
    "BSA": "BRA",  # Brazil Série A
    "CLI": "BRA",  # Copa Libertadores (South-American proxy)
    "AR":  "ARG",  # Argentina Primera División
    "APD": "ARG",  # Argentina (alternate code)
}

# Competition codes / API types that indicate national-team (international) matches
_INTL_COMP_CODES = {"WC", "EC", "CA", "AFCON", "AFC", "GC", "CONC", "INT"}
_INTL_KEYWORDS   = {
    "world cup", "euro", "copa america", "nations", "gold cup",
    "africa", "asian cup", "friendly", "international", "warm-up",
    "concacaf", "conmebol",
}

def _is_intl(match: dict) -> bool:
    comp  = match.get("competition", {})
    code  = comp.get("code", "")
    ctype = comp.get("type", "")
    name  = comp.get("name", "").lower()
    return (
        code  in _INTL_COMP_CODES
        or ctype == "INTERNATIONAL"
        or any(kw in name for kw in _INTL_KEYWORDS)
    )


# ── Fixture fetch ─────────────────────────────────────────────────────────────

def _fetch_all_today() -> list:
    """
    1. Try the general /v4/matches endpoint (catches friendlies & all accessible comps).
    2. Supplement with explicit per-competition calls (deduped by match ID).
    """
    from datetime import date
    today = date.today().isoformat()
    seen: dict = {}

    # General endpoint — returns everything the key can access today
    try:
        r = requests.get(
            f"{_BASE_URL}/matches",
            headers=_HEADERS,
            params={"dateFrom": today, "dateTo": today},
            timeout=20,
        )
        if r.status_code == 200:
            for m in r.json().get("matches", []):
                if m.get("status") in _STATUSES:
                    seen[m["id"]] = m
    except Exception:
        pass

    # Per-competition fallback (deduped)
    for code in _COMP_CODES:
        try:
            r = requests.get(
                f"{_BASE_URL}/competitions/{code}/matches",
                headers=_HEADERS,
                params={"dateFrom": today, "dateTo": today},
                timeout=15,
            )
            if r.status_code == 200:
                for m in r.json().get("matches", []):
                    if m.get("status") in _STATUSES and m["id"] not in seen:
                        seen[m["id"]] = m
        except Exception:
            pass

    return list(seen.values())


# ── Club historical data ──────────────────────────────────────────────────────

_hist_cache: dict = {}


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
                "FTHome": "hg", "FTAway": "ag",
                "HomeTarget": "h_sot", "AwayTarget": "a_sot",
                "HomeCorners": "h_cor", "AwayCorners": "a_cor",
            })[["date", "home", "away", "hg", "ag", "h_sot", "a_sot", "h_cor", "a_cor"]]
        )
        frames.append(sub)
    except Exception:
        pass

    # FBref 2025 extract (Brazil / Argentina only)
    fbref = {"BRA": "data/brazil_2025.csv", "ARG": "data/argentina_2025.csv"}.get(league_code)
    if fbref:
        try:
            fb = pd.read_csv(fbref).rename(columns={
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
        .sort_values("date")
        .reset_index(drop=True)
    )
    _hist_cache[league_code] = combined
    return combined


_DEFAULT_STATS = {
    "gf": 1.2, "ga": 1.2, "sot": 4.5, "corners": 5.0,
    "win": 0.33, "draw": 0.27, "hwn": 0.40, "awn": 0.25,
}


def _resolve(name: str, teams: list):
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

    gf = ga = 0.0
    sot_s = sot_n = cor_s = cor_n = 0.0
    wins = draws = pts = 0
    hwin = hgames = awin = agames = 0

    for _, row in recent.iterrows():
        ih = row["home"] == team
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


# ── Kick-off helpers ──────────────────────────────────────────────────────────

def _kickoff_str(utc_date: str) -> str:
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except Exception:
        return "?"


def _best_bet(home_team, away_team, ph, pd_, pa, pg, pc=None):
    """
    Find the highest-confidence tip across result, goals, and corners markets.
    Returns (best_market_label, confidence, all_market_conf).
    """
    options = []

    # Result market
    result_conf = max(ph, pd_, pa)
    if result_conf == ph:
        options.append((f"{home_team} to Win", result_conf))
    elif result_conf == pa:
        options.append((f"{away_team} to Win", result_conf))
    else:
        options.append(("Draw", result_conf))

    # Goals market
    goals_conf = max(pg, 1.0 - pg)
    goals_lbl  = "Over 2.5 Goals" if pg >= 0.5 else "Under 2.5 Goals"
    options.append((goals_lbl, goals_conf))

    # Corners market (club only)
    corners_conf = 0.0
    if pc is not None:
        corners_conf = max(pc, 1.0 - pc)
        corners_lbl  = "Corners Over 9.5" if pc >= 0.5 else "Corners Under 9.5"
        options.append((corners_lbl, corners_conf))

    label, best_conf = max(options, key=lambda x: x[1])
    overall_best = max(result_conf, goals_conf, corners_conf)
    return label, round(best_conf, 4), round(overall_best, 4)


# ── Main entry point ──────────────────────────────────────────────────────────

def get_daily_tips(club_predict_fn, wc_predict_fn) -> list:
    """
    Fetch today's fixtures, score each, filter to ≥60% on any market,
    sort highest first, return up to _MAX_TIPS results.
    """
    raw_matches = _fetch_all_today()
    results     = []
    tip_id      = 1

    for match in raw_matches:
        home_api  = match["homeTeam"]["name"]
        away_api  = match["awayTeam"]["name"]
        utc_date  = match.get("utcDate", "")
        time_str  = _kickoff_str(utc_date)
        comp      = match.get("competition", {})
        comp_code = comp.get("code", "")
        comp_name = comp.get("name", comp_code)

        home_crest = match["homeTeam"].get("crest") or ""
        away_crest = match["awayTeam"].get("crest") or ""
        is_intl    = _is_intl(match)

        # ── International (national-team) match ──────────────────────────────
        if is_intl:
            try:
                pred = wc_predict_fn(home_api, away_api, is_neutral=True)
            except Exception:
                continue

            ph, pd_, pa = pred["home_win"], pred["draw"], pred["away_win"]
            pg           = pred["over_goals"]
            bet_lbl, bet_conf, overall = _best_bet(
                pred["resolved_home"], pred["resolved_away"], ph, pd_, pa, pg
            )
            if overall < _MIN_CONF:
                continue

            results.append({
                "id":             f"dt{tip_id}",
                "source":         "INTL",
                "league":         comp_name,
                "home_team":      pred["resolved_home"],
                "away_team":      pred["resolved_away"],
                "home_crest":     home_crest,
                "away_crest":     away_crest,
                "is_international": True,
                "time":           time_str,
                "utc_date":       utc_date,
                "date_label":     "Today",
                "result":         {"home": round(ph, 4), "draw": round(pd_, 4), "away": round(pa, 4)},
                "over_goals":     round(pg, 4),
                "best_bet":       {"label": bet_lbl, "confidence": bet_conf},
                "_conf":          overall,
            })

        # ── Club match ────────────────────────────────────────────────────────
        else:
            league_code = _COMP_TO_LEAGUE.get(comp_code)
            if not league_code:
                continue

            df_hist = _load_history(league_code)
            if df_hist.empty:
                continue

            teams = sorted(set(df_hist["home"].tolist() + df_hist["away"].tolist()))
            h_res = _resolve(home_api, teams)
            a_res = _resolve(away_api, teams)

            hs  = _rolling(df_hist, h_res) if h_res else dict(_DEFAULT_STATS)
            as_ = _rolling(df_hist, a_res) if a_res else dict(_DEFAULT_STATS)

            form5_h = hs.pop("_pts", round((hs["win"] * 3 + hs["draw"]) * 5))
            form5_a = as_.pop("_pts", round((as_["win"] * 3 + as_["draw"]) * 5))

            try:
                ph, pd_, pa, pg, pc = club_predict_fn(
                    hs, as_, 0, form5_h, form5_a, 2.60, 3.10, 2.80, league_code
                )
            except Exception:
                continue

            bet_lbl, bet_conf, overall = _best_bet(
                home_api, away_api, ph, pd_, pa, pg, pc
            )
            if overall < _MIN_CONF:
                continue

            results.append({
                "id":             f"dt{tip_id}",
                "source":         comp_code,
                "league":         comp_name,
                "home_team":      home_api,
                "away_team":      away_api,
                "home_crest":     home_crest,
                "away_crest":     away_crest,
                "is_international": False,
                "time":           time_str,
                "utc_date":       utc_date,
                "date_label":     "Today",
                "result":         {"home": round(ph, 4), "draw": round(pd_, 4), "away": round(pa, 4)},
                "over_goals":     round(pg, 4),
                "over_corners":   round(pc, 4),
                "best_bet":       {"label": bet_lbl, "confidence": bet_conf},
                "_conf":          overall,
            })

        tip_id += 1

    results.sort(key=lambda x: x["_conf"], reverse=True)
    out = results[:_MAX_TIPS]
    for p in out:
        p.pop("_conf", None)
    return out

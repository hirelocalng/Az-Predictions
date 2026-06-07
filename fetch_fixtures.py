"""
fetch_fixtures.py — Fetch live matches from Free API Live Football Data (RapidAPI),
score each with XGBoost models, and return top-5 predictions with ≥55% confidence.

Endpoint: GET /football-current-live
Response:  { "response": { "live": [ <match>, ... ] } }
"""

import difflib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── RapidAPI ──────────────────────────────────────────────────────────────────

_LIVE_URL     = "https://free-api-live-football-data.p.rapidapi.com/football-current-live"
_FIXTURES_URL = "https://free-api-live-football-data.p.rapidapi.com/football-get-all-fixtures-by-date"
_HEADERS  = {
    "x-rapidapi-key":  "c9b2df19ddmsh44246571fb2e3c6p1f4bfjsn0f74392f56f5",
    "x-rapidapi-host": "free-api-live-football-data.p.rapidapi.com",
}

_MIN_CONF = 0.55   # minimum confidence on any market
_MAX_TIPS = 5      # return only the top 5

_FINISHED_STATUSES = frozenset({
    "ft", "aet", "pen", "abd", "fin", "final", "finished",
    "full_time", "full-time", "ended", "completed", "closed",
    "match_finished", "after extra time", "after penalties",
})

_LIVE_STATUSES = frozenset({
    "1h", "ht", "2h", "et", "bt", "p", "live", "in_play",
    "in play", "playing", "ongoing", "inprogress",
})

# ── League name → Matches.csv Division code ────────────────────────────────────

_LEAGUE_TO_CODE = {
    "premier league":         "E0",
    "championship":           "E0",
    "la liga":                "SP1",
    "bundesliga":             "D1",
    "serie a":                "I1",
    "ligue 1":                "F1",
    "eredivisie":             "N1",
    "primeira liga":          "P1",
    "brasileirao":            "BRA",
    "serie a brazil":         "BRA",
    "argentina primera":      "ARG",
    "superliga argentina":    "ARG",
    "champions league":       "E0",
    "europa league":          "E0",
}

_INTL_KEYWORDS = {
    "world cup", "euro", "copa america", "nations league", "gold cup",
    "africa cup", "asian cup", "friendly", "international", "warm-up",
    "concacaf", "conmebol", "afcon",
}

# ── Team name extraction helpers ──────────────────────────────────────────────

def _team_name(obj) -> str:
    """Extract a team name from whatever shape the match dict uses."""
    if isinstance(obj, dict):
        for key in ("name", "team_name", "teamName"):
            if obj.get(key):
                return str(obj[key]).strip()
    return str(obj).strip() if obj else ""


def _extract_teams(match: dict):
    """Return (home_name, away_name) from a match dict."""
    # Shape 1: homeTeam / awayTeam objects
    if "homeTeam" in match or "awayTeam" in match:
        return _team_name(match.get("homeTeam", {})), _team_name(match.get("awayTeam", {}))
    # Shape 2: home / away objects or strings
    if "home" in match or "away" in match:
        return _team_name(match.get("home", "")), _team_name(match.get("away", ""))
    # Shape 3: flat string fields
    for h_key in ("homeName", "home_team", "home_name", "match_hometeam_name"):
        if match.get(h_key):
            for a_key in ("awayName", "away_team", "away_name", "match_awayteam_name"):
                if match.get(a_key):
                    return str(match[h_key]).strip(), str(match[a_key]).strip()
    return "", ""


def _extract_league(match: dict) -> str:
    """Return the competition/league name as a lowercase string."""
    for key in ("competition", "league", "tournament"):
        val = match.get(key)
        if isinstance(val, dict):
            name = val.get("name") or val.get("league_name") or ""
            return str(name).lower().strip()
        if isinstance(val, str):
            return val.lower().strip()
    for key in ("league_name", "competition_name", "tournament_name"):
        if match.get(key):
            return str(match[key]).lower().strip()
    return ""


def _extract_logo(match: dict, side: str) -> str:
    """Return crest/logo URL for 'home' or 'away' side."""
    team_key = "homeTeam" if side == "home" else "awayTeam"
    alt_key  = "home"     if side == "home" else "away"
    obj = match.get(team_key) or match.get(alt_key) or {}
    if isinstance(obj, dict):
        return obj.get("logo") or obj.get("crest") or obj.get("image") or ""
    return ""


def _get_match_status(match: dict) -> str:
    """Extract and normalise match status string."""
    for key in ("status", "state", "matchStatus", "match_status", "statusShort", "match_state"):
        val = match.get(key)
        if isinstance(val, dict):
            val = val.get("short") or val.get("long") or val.get("name") or ""
        if val:
            return str(val).lower().strip()
    return ""


def _is_finished(match: dict) -> bool:
    """Return True if this match has already been completed."""
    return _get_match_status(match) in _FINISHED_STATUSES


def _extract_time(match: dict) -> str:
    """Return a HH:MM kick-off string."""
    for key in ("time", "startTime", "utcDate", "date", "kickoff", "match_time"):
        val = match.get(key, "")
        if val:
            val = str(val)
            for fmt in ("%H:%M", "%H:%M:%S"):
                try:
                    return datetime.strptime(val, fmt).strftime("%H:%M")
                except ValueError:
                    pass
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                return dt.strftime("%H:%M")
            except Exception:
                pass
            if len(val) >= 5 and ":" in val:
                return val[:5]
    return "?"

# ── Fetch live matches ────────────────────────────────────────────────────────

def _fetch_live() -> list:
    """Fetch today's fixtures (scheduled + live), excluding finished matches."""
    today = datetime.now().strftime("%Y-%m-%d")
    matches: list = []
    seen_ids: set = set()

    def _add(items):
        for m in items:
            mid = (m.get("id") or m.get("match_id") or m.get("fixture_id")
                   or m.get("matchId") or m.get("fixtureId"))
            if mid and mid in seen_ids:
                continue
            if mid:
                seen_ids.add(mid)
            if not _is_finished(m):
                matches.append(m)

    # Try date-based endpoint (all of today's fixtures)
    try:
        r = requests.get(_FIXTURES_URL, headers=_HEADERS,
                         params={"date": today}, timeout=20)
        r.raise_for_status()
        data = r.json()
        resp = data.get("response", data)
        added = False
        if isinstance(resp, dict):
            for key in ("fixtures", "matches", "schedule", "live", "games", "data"):
                items = resp.get(key, [])
                if isinstance(items, list) and items:
                    _add(items)
                    added = True
                    break
        if not added and isinstance(resp, list):
            _add(resp)
    except Exception:
        pass

    # Also fetch currently live matches (complements the schedule)
    try:
        r = requests.get(_LIVE_URL, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        live = data.get("response", {}).get("live", [])
        if isinstance(live, list):
            _add(live)
    except Exception as exc:
        if not matches:
            raise RuntimeError(f"Failed to fetch fixtures: {exc}") from exc

    return matches

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

    corners_conf = 0.0
    if pc is not None:
        corners_conf = max(pc, 1.0 - pc)
        options.append(("Corners Over 9.5" if pc >= 0.5 else "Corners Under 9.5", corners_conf))

    label, best_conf = max(options, key=lambda x: x[1])
    return label, round(best_conf, 4), round(max(result_conf, goals_conf, corners_conf), 4)

# ── Main entry point ──────────────────────────────────────────────────────────

def get_daily_tips(club_predict_fn, wc_predict_fn) -> list:
    """
    Fetch /football-current-live, score each match through the ML models,
    filter to ≥55% confidence, return the top 5 sorted by confidence.
    """
    live_matches = _fetch_live()
    results = []

    for idx, match in enumerate(live_matches):
        if _is_finished(match):
            continue

        home_name, away_name = _extract_teams(match)
        if not home_name or not away_name:
            continue

        status      = _get_match_status(match)
        date_label  = "Live" if status in _LIVE_STATUSES else "Today"
        league_raw  = _extract_league(match)
        time_str    = _extract_time(match)
        home_logo   = _extract_logo(match, "home")
        away_logo   = _extract_logo(match, "away")
        comp_label  = match.get("competition", {}).get("name") \
                      or match.get("league", {}).get("name") \
                      or league_raw.title() or "Today"

        # ── International match ───────────────────────────────────────────
        is_intl = any(kw in league_raw for kw in _INTL_KEYWORDS)
        if is_intl:
            try:
                pred = wc_predict_fn(home_name, away_name, is_neutral=True)
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
                "id":               f"dt{idx+1}",
                "league":           comp_label,
                "home_team":        pred["resolved_home"],
                "away_team":        pred["resolved_away"],
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
            league_code = next(
                (code for kw, code in _LEAGUE_TO_CODE.items() if kw in league_raw),
                None
            )
            if not league_code:
                league_code = "E0"   # generic fallback so we still score the match

            df_hist = _load_history(league_code)
            if df_hist.empty:
                # Try generic EPL history as fallback
                df_hist = _load_history("E0")
            if df_hist.empty:
                continue

            teams  = sorted(set(df_hist["home"].tolist() + df_hist["away"].tolist()))
            h_res  = _resolve(home_name, teams)
            a_res  = _resolve(away_name, teams)

            hs   = _rolling(df_hist, h_res)  if h_res else dict(_DEFAULT_STATS)
            as_  = _rolling(df_hist, a_res)  if a_res else dict(_DEFAULT_STATS)

            form5_h = hs.pop("_pts", round((hs["win"] * 3 + hs["draw"]) * 5))
            form5_a = as_.pop("_pts", round((as_["win"] * 3 + as_["draw"]) * 5))

            try:
                ph, pd_, pa, pg, pc = club_predict_fn(
                    hs, as_, 0, form5_h, form5_a, 2.60, 3.10, 2.80, league_code
                )
            except Exception:
                continue

            bet_lbl, bet_conf, overall = _best_bet(
                home_name, away_name, ph, pd_, pa, pg, pc
            )
            if overall < _MIN_CONF:
                continue

            results.append({
                "id":               f"dt{idx+1}",
                "league":           comp_label,
                "home_team":        home_name,
                "away_team":        away_name,
                "home_crest":       home_logo,
                "away_crest":       away_logo,
                "is_international": False,
                "time":             time_str,
                "date_label":       date_label,
                "result":           {"home": round(ph, 4), "draw": round(pd_, 4), "away": round(pa, 4)},
                "over_goals":       round(pg, 4),
                "over_corners":     round(pc, 4),
                "best_bet":         {"label": bet_lbl, "confidence": bet_conf},
                "_conf":            overall,
            })

    results.sort(key=lambda x: x["_conf"], reverse=True)
    out = results[:_MAX_TIPS]
    for p in out:
        p.pop("_conf", None)
    return out

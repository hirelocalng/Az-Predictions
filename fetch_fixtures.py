"""
fetch_fixtures.py — Fetch today's fixtures from football-data.org,
score each through XGBoost models, and return predictions.

Endpoints used:
  GET /v4/matches?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD
"""

import difflib
import math
import unicodedata
import warnings
from datetime import datetime, timezone

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── football-data.org API ──────────────────────────────────────────────────────

_FD_URL     = "https://api.football-data.org/v4/matches"
_FD_HEADERS = {"X-Auth-Token": "118333be3eb84d0ca4e10740f6d62255"}

# ESPN unofficial scoreboard API — fallback for international friendlies not
# covered by the football-data.org free tier. No API key required.
_ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
)
# ESPN league slugs to query for international matches
_ESPN_INTL_LEAGUES = ["fifa.friendly", "uefa.friendly"]

_ESPN_STATUS_MAP = {
    "STATUS_SCHEDULED":   "TIMED",
    "STATUS_IN_PROGRESS": "IN_PLAY",
    "STATUS_HALFTIME":    "PAUSED",
    "STATUS_FINAL":       "FINISHED",
}

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


def _extract_utc_iso(match: dict) -> str:
    """Return the raw UTC ISO-8601 string from a football-data.org match."""
    return match.get("utcDate", "")

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


def _fetch_window(days: int) -> list:
    """
    Non-finished football-data.org matches across ALL tracked competitions,
    from today through the next `days` days (football-data.org caps any
    single date range at 10 days). Used by get_daily_tips() to fall through
    to the nearest upcoming matchday when today itself has nothing on.
    """
    date_from = datetime.now().strftime("%Y-%m-%d")
    date_to   = (datetime.now() + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    r = requests.get(
        _FD_URL, headers=_FD_HEADERS,
        params={"dateFrom": date_from, "dateTo": date_to},
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

# A team with genuinely zero historical rows anywhere (never observed in any
# league we track) falls back to _DEFAULT_STATS above -- a plausible
# "average team" prior. But a team that HAS history, just none of it recent
# (promoted after a long absence -- e.g. Coventry's last E0 match was 2001),
# should not have that stale history quietly presented as "current form".
# Promoted/newly-returned teams are also NOT average: they historically
# underperform the division. This prior is the average of every top-flight
# team's first 5 E0 games immediately after any promotion/return with a gap
# of >400 days since their previous E0 appearance (25 such events, 2000-2025,
# computed from data/Matches.csv during the 2026-27 pre-season data audit).
PROMOTED_TEAM_PRIOR = {
    "gf": 1.01, "ga": 1.65, "sot": 4.20, "corners": 4.45,
    "win": 0.22, "draw": 0.27, "hwn": 0.30, "awn": 0.13,
}
_STALE_HISTORY_DAYS = 400  # older than this -> treat as no current-era history

# Pre-computed form cache (from train.py _build_form_cache).
# Used as fallback when data/Matches.csv is not present on Railway.
_CLUB_FORM_CACHE: dict = {}
try:
    import os as _os
    _fc_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             "data", "club_form_cache.csv")
    if _os.path.exists(_fc_path):
        import pandas as _pd_fc
        _fc = _pd_fc.read_csv(_fc_path)
        _CLUB_FORM_CACHE = _fc.set_index("team")[
            ["gf", "ga", "sot", "corners", "win", "draw", "hwn", "awn"]
        ].to_dict("index")
        del _pd_fc, _fc, _fc_path, _os
except Exception:
    pass

# Pre-computed per-club ELO cache (most recent rating per club, from
# data/EloRatings.csv via the one-off build in this repo's session notes).
# EloRatings.csv itself isn't deployed (gitignored, large), so this small
# derived cache is what actually ships to Railway.
_TEAM_ELO_CACHE: dict = {}
try:
    import os as _os
    _elo_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "data", "team_elo_cache.csv")
    if _os.path.exists(_elo_path):
        import pandas as _pd_elo
        _ec = _pd_elo.read_csv(_elo_path)
        _TEAM_ELO_CACHE = dict(zip(_ec["club"], _ec["elo"]))
        del _pd_elo, _ec, _elo_path, _os
except Exception:
    pass


def _team_elo(name: str):
    """Resolve a team name against the ELO cache; None if no reasonable match."""
    if name in _TEAM_ELO_CACHE:
        return _TEAM_ELO_CACHE[name]
    match = _resolve(name, list(_TEAM_ELO_CACHE.keys()))
    return _TEAM_ELO_CACHE.get(match) if match else None


_ELO_CACHE_MEAN = (sum(_TEAM_ELO_CACHE.values()) / len(_TEAM_ELO_CACHE)
                    if _TEAM_ELO_CACHE else 1500.0)


def _elo_diff_safe(home_name: str, away_name: str) -> float:
    """
    home_elo - away_elo, but never silently collapses to 0 (exact-parity)
    just because one side is missing from the cache -- that used to happen
    for any team _team_elo() couldn't resolve, telling the model "these two
    teams are exactly equally rated" with no signal anything was missing.
    A missing side is defaulted to the cache-wide average Elo instead, and
    logged so it's visible rather than silent.
    """
    home_elo = _team_elo(home_name)
    away_elo = _team_elo(away_name)
    if home_elo is None or away_elo is None:
        missing = [n for n, e in ((home_name, home_elo), (away_name, away_elo)) if e is None]
        print(f"[fetch_fixtures] WARNING: no Elo cache match for {missing} -- "
              f"defaulting to cache average ({_ELO_CACHE_MEAN:.0f}) instead of forcing diff=0")
        home_elo = _ELO_CACHE_MEAN if home_elo is None else home_elo
        away_elo = _ELO_CACHE_MEAN if away_elo is None else away_elo
    return home_elo - away_elo


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

    for csv_path, code in [
        ("data/brazil_2025.csv", "BRA"),
        ("data/brazil_2026_supplement.csv", "BRA"),
        ("data/argentina_2025.csv", "ARG"),
    ]:
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


# Known official-name spelling variants that beat the substring/fuzzy logic
# below on their own — e.g. "RCD Espanyol de Barcelona" contains the literal
# substring "Barcelona" (the city qualifier), which would otherwise outrank
# "Espanol" (Matches.csv's spelling, missing the Catalan "y") and wrongly
# resolve Espanyol fixtures to FC Barcelona.
_FD_TEAM_ALIAS = {
    "rcd espanyol de barcelona": "Espanol",
    "espanyol": "Espanol",
    "club atletico de madrid": "Ath Madrid",   # "Atletico" doesn't abbreviate to "Ath" via substring/fuzzy
    "atletico madrid": "Ath Madrid",
    "athletic club": "Ath Bilbao",             # no shared token with "Ath Bilbao" at all
    "athletic bilbao": "Ath Bilbao",
    "wolverhampton wanderers fc": "Wolves",    # no shared token with "Wolves" at all
    "wolverhampton wanderers": "Wolves",
}


def _normalize_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return n.lower().strip()


def _resolve(name: str, teams: list) -> str | None:
    """
    Resolve a live fixture's team name (often a verbose official name, e.g.
    "Rayo Vallecano de Madrid" or "Real Betis Balompié") against a list of
    canonical short names (e.g. "Vallecano", "Betis") used in the historical
    data. Priority order:
      1. exact match (accent/case-insensitive)
      2. known alias override (spelling variants that'd otherwise mismatch)
      3. longest canonical name that appears as a substring of the input —
         substring containment is far more reliable than raw character-diff
         similarity for these "<Club> de <City>" official names, where a
         short/generic candidate (e.g. "Real Madrid") can otherwise win on
         a shared city name alone (e.g. "Rayo Vallecano de Madrid")
      4. fuzzy fallback (tightened cutoff — was silently matching unrelated
         teams at 0.4)
    """
    nl = _normalize_name(name)

    for t in teams:
        if _normalize_name(t) == nl:
            return t

    alias = _FD_TEAM_ALIAS.get(nl)
    if alias and alias in teams:
        return alias

    contained = [t for t in teams if _normalize_name(t) and _normalize_name(t) in nl]
    if contained:
        return max(contained, key=lambda t: len(_normalize_name(t)))

    ms = difflib.get_close_matches(name, teams, n=1, cutoff=0.55)
    return ms[0] if ms else None


def _rolling(df: pd.DataFrame, team: str, n: int = 5) -> dict:
    mask   = (df["home"] == team) | (df["away"] == team)
    recent = df[mask].sort_values("date").tail(n)
    if recent.empty:
        if team in _CLUB_FORM_CACHE:
            return dict(_CLUB_FORM_CACHE[team])
        return dict(_DEFAULT_STATS)

    # Team has history, but is it CURRENT? A promoted team returning after a
    # long absence (e.g. Coventry's most recent E0 row is from 2001) would
    # otherwise have that decades-stale form silently presented as "recent
    # form" with no signal to the model that anything is off. Fall back to
    # the promoted-team prior instead of quietly serving ancient results.
    days_since_last = (pd.Timestamp(datetime.now(timezone.utc).date()) - recent["date"].max()).days
    if days_since_last > _STALE_HISTORY_DAYS:
        return dict(PROMOTED_TEAM_PRIOR)

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

# ── BTTS helper ───────────────────────────────────────────────────────────────

def _btts_prob(home_gs: float, away_gs: float) -> float:
    """P(both teams score ≥1 goal) using Poisson approximation."""
    p_h = 1.0 - math.exp(-max(0.05, home_gs))
    p_a = 1.0 - math.exp(-max(0.05, away_gs))
    return round(p_h * p_a, 4)


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
    utc_iso    = _extract_utc_iso(match)
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

    elo_diff = _elo_diff_safe(home_name, away_name)

    # No live pre-match odds feed exists for upcoming fixtures (only
    # historical closing odds in Matches.csv) — this used to be faked with
    # a hardcoded 2.60/3.10/2.80 triple, which flattened every prediction
    # toward a near-uniform split regardless of the real matchup (2026-08-21
    # incident). Models are trained odds-free now (train.py BASE_FEATURES);
    # no odds are fetched, computed, or displayed anywhere in this path.
    try:
        ph, pd_, pa, pg, pc = club_predict_fn(
            hs, as_, elo_diff, form5_h, form5_a, league_code
        )
    except Exception:
        return None

    btts = _btts_prob(hs.get("gf", 1.2), as_.get("gf", 1.2))

    best_prob = max(ph, pd_, pa)
    if best_prob == ph:
        bet = {"label": f"{home_name} Win", "confidence": round(ph, 4)}
    elif best_prob == pa:
        bet = {"label": f"{away_name} Win", "confidence": round(pa, 4)}
    else:
        bet = {"label": "Draw", "confidence": round(pd_, 4)}

    return {
        "league":       comp_name,
        "home_team":    home_name,
        "away_team":    away_name,
        "home_color":   "#444",
        "away_color":   "#666",
        "home_crest":   home_crest,
        "away_crest":   away_crest,
        "time":         time_str,
        "utc_kickoff":  utc_iso,
        "date_label":   "Live" if status in _LIVE_STATUSES else "Today",
        "status":       status,
        "result":       {"home": round(ph, 4), "draw": round(pd_, 4), "away": round(pa, 4)},
        "over_goals":   round(pg, 4),
        "btts":         btts,
        "over_corners": round(pc, 4),
        "best_bet":     bet,
        "_conf":        max(ph, pd_, pa, max(pg, 1 - pg)),
    }

# ── Club tips (domestic leagues only) ────────────────────────────────────────

def get_club_tips(club_predict_fn) -> list:
    """
    Fetch the next 9 days of matches from football-data.org (not just
    today — a single quiet day left this returning nothing for entire
    leagues, e.g. Brasileirão, whose next fixture might be 2-3 days out),
    filter to domestic club competitions, run predictions, return all
    scored matches (no confidence floor — callers decide whether to show
    'no matches' message; each card carries its own utc_kickoff so the
    frontend can show/sort by date).
    """
    matches = _fetch_window(9)
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

# ── Single-competition dedicated homepage sections (Premier League, La Liga) ──

_SINGLE_COMP_WINDOW_DAYS = 10   # look this many days ahead for upcoming fixtures


def _fetch_comp_window(comp_code: str, days: int) -> list:
    """
    Non-finished football-data.org matches for one competition code over the
    next `days`. Uses the /v4/competitions/{code}/matches endpoint rather
    than /v4/matches?competitions= — the latter rejects any date range over
    10 days ("Specified period must not exceed 10 days"), which silently
    broke the wide (~120 day) lookahead used by _get_comp_next_match during
    the off-season / early in a season.
    """
    date_from = datetime.now().strftime("%Y-%m-%d")
    date_to   = (datetime.now() + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    r = requests.get(
        f"https://api.football-data.org/v4/competitions/{comp_code}/matches",
        headers=_FD_HEADERS,
        params={"dateFrom": date_from, "dateTo": date_to},
        timeout=15,
    )
    r.raise_for_status()
    matches = r.json().get("matches", [])
    return [m for m in matches if not _is_finished(m)]


def _get_comp_tips(comp_code: str, id_prefix: str, club_predict_fn) -> list:
    """
    Fetch upcoming fixtures (next _SINGLE_COMP_WINDOW_DAYS days) for one
    football-data.org competition and score each through the general club
    model (league code resolved via _COMP_TO_LEAGUE from the match itself).
    No confidence floor; caller decides what to do with an empty list
    (e.g. off-season).
    """
    try:
        matches = _fetch_comp_window(comp_code, _SINGLE_COMP_WINDOW_DAYS)
    except Exception:
        matches = []

    tips = []
    for idx, match in enumerate(matches):
        scored = _score_club_match(match, club_predict_fn)
        if scored is None:
            continue
        scored["id"] = f"{id_prefix}{idx + 1}"
        scored.pop("_conf", None)
        tips.append(scored)

    return tips


def _get_comp_next_match(comp_code: str):
    """
    Return the next scheduled fixture for one competition (widest lookahead,
    up to ~120 days so it works during the summer off-season) as a dict with
    keys home, away, home_crest, away_crest, utc_kickoff, venue — or None if
    the API is unreachable/returns nothing.
    """
    try:
        matches = _fetch_comp_window(comp_code, 120)
    except Exception:
        return None
    if not matches:
        return None

    def _kickoff(m):
        try:
            return datetime.fromisoformat(m.get("utcDate", "").replace("Z", "+00:00"))
        except Exception:
            return datetime(9999, 1, 1, tzinfo=timezone.utc)

    matches.sort(key=_kickoff)
    m = matches[0]
    home, away = _extract_teams(m)
    if not home or not away:
        return None
    return {
        "home":        home,
        "away":        away,
        "home_crest":  _extract_logo(m, "home"),
        "away_crest":  _extract_logo(m, "away"),
        "utc_kickoff": _extract_utc_iso(m),
        "venue":       (m.get("venue") or "").strip(),
    }


def get_pl_tips(club_predict_fn) -> list:
    return _get_comp_tips("PL", "pl", club_predict_fn)


def get_pl_next_match():
    return _get_comp_next_match("PL")


def get_laliga_tips(club_predict_fn) -> list:
    return _get_comp_tips("PD", "ll", club_predict_fn)


def get_laliga_next_match():
    return _get_comp_next_match("PD")

# ── TheSportsDB helpers ───────────────────────────────────────────────────────

def _norm_pair(a: str, b: str) -> frozenset:
    """Canonical key for deduplication regardless of home/away order."""
    return frozenset({a.lower().strip(), b.lower().strip()})


def _fetch_intl_from_espn(today: str) -> list:
    """
    Fetch today's international fixtures from ESPN's free scoreboard API.
    Queries multiple league slugs (fifa.friendly, uefa.friendly) and merges.
    Returns a list of dicts with keys: home, away, league, time,
    home_crest, away_crest, status.
    """
    date_str = today.replace("-", "")   # YYYYMMDD
    result   = []
    seen     = set()

    for slug in _ESPN_INTL_LEAGUES:
        try:
            url = _ESPN_SCOREBOARD.format(league=slug)
            r   = requests.get(url, params={"dates": date_str}, timeout=10)
            if r.status_code != 200:
                continue
            events = r.json().get("events") or []
            for ev in events:
                competitors = (ev.get("competitions") or [{}])[0].get("competitors", [])
                home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home_c or not away_c:
                    continue
                home = (home_c.get("team", {}).get("displayName") or "").strip()
                away = (away_c.get("team", {}).get("displayName") or "").strip()
                if not home or not away:
                    continue
                pair = _norm_pair(home, away)
                if pair in seen:
                    continue
                seen.add(pair)

                # Parse UTC kickoff time
                raw_date = ev.get("date", "")
                try:
                    time_str = (
                        datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                        .strftime("%H:%M")
                    )
                except Exception:
                    time_str = "?"

                esp_status = (ev.get("status") or {}).get("type", {}).get("name", "")
                status = _ESPN_STATUS_MAP.get(esp_status, "TIMED")
                if status == "FINISHED":
                    continue  # don't surface completed matches

                league_name = ev.get("name", "").split(" at ")[0] if " at " in ev.get("name","") else "International Friendly"
                # Use competition display name if available
                comp_name = (ev.get("competitions") or [{}])[0].get("type", {}).get("text", "") or "International Friendly"

                result.append({
                    "home":         home,
                    "away":         away,
                    "league":       comp_name,
                    "time":         time_str,
                    "utc_kickoff":  raw_date if raw_date else "",
                    "home_crest":   home_c.get("team", {}).get("logo", ""),
                    "away_crest":   away_c.get("team", {}).get("logo", ""),
                    "status":       status,
                })
        except Exception:
            continue

    return result


# ── All today's matches via worldcup model (no filter, no confidence floor) ───

def get_intl_tips(wc_predict_fn) -> list:
    """
    Fetch ALL of today's international matches and return predictions.

    Primary source  : football-data.org — covers WCQ, Nations League, EURO, etc.
    Secondary source: TheSportsDB free API — covers international friendlies that
                      the football-data.org free tier omits.
    Matches are deduplicated by team pair so no fixture appears twice.
    """
    today  = datetime.now().strftime("%Y-%m-%d")
    tips   = []
    seen   = set()   # frozenset of normalised {home, away} pairs

    # ── Primary: football-data.org ────────────────────────────────────────────
    try:
        fd_matches = _fetch_today()
    except Exception:
        fd_matches = []

    for idx, match in enumerate(fd_matches):
        home_name, away_name = _extract_teams(match)
        if not home_name or not away_name:
            continue

        comp_name  = match.get("competition", {}).get("name", "") or "International"
        status     = _get_match_status(match)
        time_str   = _extract_time(match)
        utc_iso    = _extract_utc_iso(match)
        home_crest = _extract_logo(match, "home")
        away_crest = _extract_logo(match, "away")

        try:
            pred = wc_predict_fn(home_name, away_name, is_neutral=True)
        except Exception:
            continue
        if not pred:
            continue

        ph, pd_, pa = pred["home_win"], pred["draw"], pred["away_win"]
        pg           = pred["over_goals"]
        display_home = pred.get("resolved_home", home_name)
        display_away = pred.get("resolved_away", away_name)

        best_prob = max(ph, pd_, pa)
        if best_prob == ph:
            bet = {"label": f"{display_home} to Win", "confidence": ph}
        elif best_prob == pa:
            bet = {"label": f"{display_away} to Win", "confidence": pa}
        else:
            bet = {"label": "Draw", "confidence": pd_}

        seen.add(_norm_pair(home_name, away_name))
        tips.append({
            "id":               f"intl{idx + 1}",
            "league":           comp_name,
            "home_team":        display_home,
            "away_team":        display_away,
            "home_crest":       home_crest,
            "away_crest":       away_crest,
            "is_international": True,
            "time":             time_str,
            "utc_kickoff":      utc_iso,
            "status":           status,
            "result":           {"home": ph, "draw": pd_, "away": pa},
            "over_goals":       pg,
            "btts":             pred.get("btts",         _btts_prob(1.2, 1.2)),
            "over_corners":     pred.get("over_corners", 0.52),
            "best_bet":         bet,
        })

    # ── Secondary: ESPN (international friendlies & more) ────────────────────
    espn_matches = _fetch_intl_from_espn(today)
    base_idx     = len(tips)

    for jdx, ev in enumerate(espn_matches):
        pair = _norm_pair(ev["home"], ev["away"])
        if pair in seen:
            continue  # already covered by football-data.org

        try:
            pred = wc_predict_fn(ev["home"], ev["away"], is_neutral=True)
        except Exception:
            continue
        if not pred:
            continue

        ph, pd_, pa = pred["home_win"], pred["draw"], pred["away_win"]
        pg           = pred["over_goals"]
        display_home = pred.get("resolved_home", ev["home"])
        display_away = pred.get("resolved_away", ev["away"])
        status       = ev.get("status", "TIMED")
        date_label   = "Live" if status in _LIVE_STATUSES else "Today"

        best_prob = max(ph, pd_, pa)
        if best_prob == ph:
            bet = {"label": f"{display_home} to Win", "confidence": ph}
        elif best_prob == pa:
            bet = {"label": f"{display_away} to Win", "confidence": pa}
        else:
            bet = {"label": "Draw", "confidence": pd_}

        seen.add(pair)
        tips.append({
            "id":               f"intl{base_idx + jdx + 1}",
            "league":           ev["league"],
            "home_team":        display_home,
            "away_team":        display_away,
            "home_crest":       ev["home_crest"],
            "away_crest":       ev["away_crest"],
            "is_international": True,
            "time":             ev["time"],
            "utc_kickoff":      ev.get("utc_kickoff", ""),
            "status":           status,
            "result":           {"home": ph, "draw": pd_, "away": pa},
            "over_goals":       pg,
            "btts":             pred.get("btts",         _btts_prob(1.2, 1.2)),
            "over_corners":     pred.get("over_corners", 0.52),
            "best_bet":         bet,
        })

    # Drop any tip whose kickoff date is before today (stale data guard)
    tips = [t for t in tips
            if not t.get("utc_kickoff") or t["utc_kickoff"][:10] >= today]
    return tips


# ── Daily tips (all competitions + ESPN friendlies, top-5 by confidence) ──────

def get_daily_tips(club_predict_fn, wc_predict_fn) -> list:
    """
    Score matches across every tracked competition and return the top 5 by
    confidence — today's fixtures if there are any scoreable ones, otherwise
    the nearest upcoming matchday within the next 9 days (a quiet weekday
    with no fixtures anywhere shouldn't just return an empty list).

    Sources (merged, deduplicated):
      1. football-data.org  — club leagues, WCQ, Nations League, etc.
      2. ESPN free API      — international friendlies missing from fd.org
    """
    today   = datetime.now().strftime("%Y-%m-%d")
    results = []
    seen    = set()   # _norm_pair keys to avoid duplicates across sources

    # ── Source 1: football-data.org (9-day window, not just today) ───────────
    try:
        fd_matches = _fetch_window(9)
    except Exception:
        fd_matches = []

    for idx, match in enumerate(fd_matches):
        home_name, away_name = _extract_teams(match)
        if not home_name or not away_name:
            continue

        comp_code  = _extract_comp_code(match)
        league_raw = _extract_league(match)
        status     = _get_match_status(match)
        time_str   = _extract_time(match)
        utc_iso    = _extract_utc_iso(match)
        home_logo  = _extract_logo(match, "home")
        away_logo  = _extract_logo(match, "away")
        comp_label = match.get("competition", {}).get("name", "") or league_raw.title()

        is_intl = (comp_code in _INTL_COMP_CODES
                   or any(kw in league_raw for kw in _INTL_KEYWORDS))

        # ── International / national-team match ───────────────────────────
        if is_intl:
            try:
                pred = wc_predict_fn(home_name, away_name, is_neutral=True)
            except Exception:
                pred = None
            if not pred:
                continue  # skip — never use club model for international matches

            ph, pd_, pa = pred["home_win"], pred["draw"], pred["away_win"]
            pg           = pred["over_goals"]
            display_home = pred.get("resolved_home", home_name)
            display_away = pred.get("resolved_away", away_name)
            bet_lbl, bet_conf, overall = _best_bet(
                display_home, display_away, ph, pd_, pa, pg
            )

            if overall < _MIN_CONF:
                continue

            seen.add(_norm_pair(home_name, away_name))
            results.append({
                "id":               f"dt{idx+1}",
                "league":           comp_label,
                "home_team":        display_home,
                "away_team":        display_away,
                "home_crest":       home_logo,
                "away_crest":       away_logo,
                "is_international": True,
                "time":             time_str,
                "utc_kickoff":      utc_iso,
                "result":           {"home": ph, "draw": pd_, "away": pa},
                "over_goals":       pg,
                "btts":             (pred or {}).get("btts",         _btts_prob(1.2, 1.2)),
                "over_corners":     (pred or {}).get("over_corners", 0.52),
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

            seen.add(_norm_pair(home_name, away_name))
            results.append({
                "id":               f"dt{idx+1}",
                "league":           comp_label,
                "home_team":        scored["home_team"],
                "away_team":        scored["away_team"],
                "home_crest":       home_logo,
                "away_crest":       away_logo,
                "is_international": False,
                "time":             time_str,
                "utc_kickoff":      utc_iso,
                "result":           scored["result"],
                "over_goals":       scored["over_goals"],
                "btts":             scored.get("btts", _btts_prob(1.2, 1.2)),
                "over_corners":     scored["over_corners"],
                "best_bet":         scored["best_bet"],
                "_conf":            overall,
            })

    # ── Source 2: ESPN international friendlies ───────────────────────────────
    espn_matches = _fetch_intl_from_espn(today)
    base_idx     = len(results)

    for jdx, ev in enumerate(espn_matches):
        pair = _norm_pair(ev["home"], ev["away"])
        if pair in seen:
            continue

        try:
            pred = wc_predict_fn(ev["home"], ev["away"], is_neutral=True)
        except Exception:
            continue
        if not pred:
            continue

        ph, pd_, pa = pred["home_win"], pred["draw"], pred["away_win"]
        pg           = pred["over_goals"]
        display_home = pred.get("resolved_home", ev["home"])
        display_away = pred.get("resolved_away", ev["away"])
        status       = ev.get("status", "TIMED")
        date_label   = "Live" if status in _LIVE_STATUSES else "Today"
        bet_lbl, bet_conf, overall = _best_bet(display_home, display_away, ph, pd_, pa, pg)

        if overall < _MIN_CONF:
            continue

        seen.add(pair)
        results.append({
            "id":               f"dt{base_idx + jdx + 1}",
            "league":           ev["league"],
            "home_team":        display_home,
            "away_team":        display_away,
            "home_crest":       ev["home_crest"],
            "away_crest":       ev["away_crest"],
            "is_international": True,
            "time":             ev["time"],
            "utc_kickoff":      ev.get("utc_kickoff", ""),
            "result":           {"home": ph, "draw": pd_, "away": pa},
            "over_goals":       pg,
            "btts":             pred.get("btts",         _btts_prob(1.2, 1.2)),
            "over_corners":     pred.get("over_corners", 0.52),
            "best_bet":         {"label": bet_lbl, "confidence": bet_conf},
            "_conf":            overall,
        })

    # Drop any result whose kickoff date is before today (stale data guard)
    results = [r for r in results
               if not r.get("utc_kickoff") or r["utc_kickoff"][:10] >= today]

    # If today has no scoreable fixtures, fall through to the nearest
    # upcoming matchday (the earliest date present among the results) rather
    # than returning an empty list — same information, just not today.
    dated = [r for r in results if r.get("utc_kickoff")]
    if dated:
        todays = [r for r in dated if r["utc_kickoff"][:10] == today]
        if todays:
            results = todays + [r for r in results if not r.get("utc_kickoff")]
        else:
            nearest_date = min(r["utc_kickoff"][:10] for r in dated)
            results = [r for r in dated if r["utc_kickoff"][:10] == nearest_date]

    results.sort(key=lambda x: x["_conf"], reverse=True)
    out = results[:_MAX_TIPS]
    for p in out:
        p.pop("_conf", None)
    return out

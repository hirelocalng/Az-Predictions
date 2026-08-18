"""
fetch_wnba_supplement.py — Backfill WNBA game results from 2025-09-15 (the day
after data/wnba_player_and_team_stats_2003-2025 stops, 2025-09-14) through
today, via ESPN's public scoreboard + boxscore-summary endpoints (the same
integration nba_predict.py already uses for live WNBA fixtures).

Output: data/wnba_2026_supplement.csv, with the same columns load_wnba_v2()
in nba_train.py reads from wnba_player_and_team_stats_2003-2025 (one row per
team per game — team_abbreviation/opponent_team_abbreviation, team_score/
opponent_team_score, team_winner, home_away, season, season_type, plus the
four box-score fields used to compute ORTG/DRTG: field_goals_attempted,
offensive_rebounds, turnovers, free_throws_attempted).

Usage:
    python fetch_wnba_supplement.py
"""

import csv
import time
from datetime import datetime, timedelta

import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
SUMMARY_URL    = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary"

START_DATE = datetime(2025, 9, 15)
END_DATE   = datetime.utcnow()
OUT_PATH   = "data/wnba_2026_supplement.csv"

COLUMNS = [
    "game_id", "game_date", "season", "season_type",
    "team_abbreviation", "opponent_team_abbreviation",
    "team_score", "opponent_team_score", "team_winner", "home_away",
    "field_goals_attempted", "offensive_rebounds", "turnovers", "free_throws_attempted",
]


def _stat_val(stats, name):
    for s in stats:
        if s.get("name") == name:
            return s.get("displayValue", "")
    return ""


def _split_second(v):
    """'27-75' -> 75 (the 'attempted' half of made-attempted stats)."""
    try:
        return int(str(v).split("-")[1])
    except Exception:
        return None


def fetch_day(date_str):
    """Return list of (event_id, season, home_abbr, away_abbr, home_score, away_score) for FINAL games."""
    try:
        r = requests.get(SCOREBOARD_URL, params={"dates": date_str}, timeout=15)
        r.raise_for_status()
    except Exception:
        return []
    events = r.json().get("events", [])
    out = []
    for ev in events:
        status = (ev.get("status") or {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue
        comp = (ev.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        season = (ev.get("season") or {}).get("year")
        season_type = (ev.get("season") or {}).get("type")  # 2 = regular, 3 = post
        out.append({
            "event_id": ev["id"],
            "season": season,
            "season_type": season_type,
            "home_abbr": home.get("team", {}).get("abbreviation", ""),
            "away_abbr": away.get("team", {}).get("abbreviation", ""),
            "home_score": home.get("score", ""),
            "away_score": away.get("score", ""),
        })
    return out


def fetch_boxscore(event_id):
    """Return {abbr: {'fga':.., 'oreb':.., 'tov':.., 'fta':..}} for both teams, or {} on failure."""
    try:
        r = requests.get(SUMMARY_URL, params={"event": event_id}, timeout=15)
        r.raise_for_status()
    except Exception:
        return {}
    teams = (r.json().get("boxscore") or {}).get("teams", [])
    result = {}
    for t in teams:
        abbr = t.get("team", {}).get("abbreviation", "")
        stats = t.get("statistics", [])
        fga = _split_second(_stat_val(stats, "fieldGoalsMade-fieldGoalsAttempted"))
        fta = _split_second(_stat_val(stats, "freeThrowsMade-freeThrowsAttempted"))
        oreb = _stat_val(stats, "offensiveRebounds")
        tov  = _stat_val(stats, "turnovers")
        if not abbr:
            continue
        result[abbr] = {
            "fga": fga,
            "oreb": int(oreb) if str(oreb).isdigit() else None,
            "tov":  int(tov) if str(tov).isdigit() else None,
            "fta": fta,
        }
    return result


def main():
    print(f"Fetching WNBA games {START_DATE.date()} -> {END_DATE.date()} via ESPN...")
    rows = []
    n_days = (END_DATE - START_DATE).days + 1
    n_games = 0

    d = START_DATE
    while d <= END_DATE:
        date_str = d.strftime("%Y%m%d")
        games = fetch_day(date_str)
        for g in games:
            box = fetch_boxscore(g["event_id"])
            n_games += 1
            game_date = d.strftime("%Y-%m-%d")
            for side, abbr, opp_abbr, score, opp_score in [
                ("home", g["home_abbr"], g["away_abbr"], g["home_score"], g["away_score"]),
                ("away", g["away_abbr"], g["home_abbr"], g["away_score"], g["home_score"]),
            ]:
                b = box.get(abbr, {})
                try:
                    win = float(score) > float(opp_score)
                except Exception:
                    win = None
                rows.append({
                    "game_id": g["event_id"],
                    "game_date": game_date,
                    "season": g["season"],
                    "season_type": g["season_type"],
                    "team_abbreviation": abbr,
                    "opponent_team_abbreviation": opp_abbr,
                    "team_score": score,
                    "opponent_team_score": opp_score,
                    "team_winner": win,
                    "home_away": side,
                    "field_goals_attempted": b.get("fga"),
                    "offensive_rebounds": b.get("oreb"),
                    "turnovers": b.get("tov"),
                    "free_throws_attempted": b.get("fta"),
                })
            time.sleep(0.15)  # be polite to ESPN's public endpoint
        d += timedelta(days=1)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"Scanned {n_days} days, found {n_games} finished games -> {len(rows)} team-game rows")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

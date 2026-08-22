
"""
Football match prediction script (v4).
Loads the three trained models and predicts:
  1. Match result   (Home Win / Draw / Away Win)  with percentages
  2. Goals          Over / Under 2.5
  3. Corners        Over / Under 9.5

Feature order mirrors BASE_FEATURES in train.py exactly.

Usage:
    python predict.py
"""

import math
import pickle
import sys
import numpy as np

MODEL_FILES = {

    "result":  "result_model.pkl",
    "goals":   "goals_model.pkl",
    "corners": "corners_model.pkl",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models() -> dict:
    models = {}
    for key, path in MODEL_FILES.items():
        try:
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        except FileNotFoundError:
            print(f"\n  ERROR: '{path}' not found. Run train.py first.")
            sys.exit(1)
    return models


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def ask_float(prompt: str, default: float, lo: float = None, hi: float = None) -> float:
    raw = input(f"    {prompt} [default {default}]: ").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        if lo is not None and v < lo:
            print(f"    Value clipped to {lo}")
            return lo
        if hi is not None and v > hi:
            print(f"    Value clipped to {hi}")
            return hi
        return v
    except ValueError:
        print(f"    Invalid — using {default}")
        return default


def ask_team_stats(team_name: str, is_home: bool) -> dict:
    role = "home" if is_home else "away"
    print(f"\n  {team_name}  ({role})  —  enter last-{5}-game rolling averages")
    print("  (press Enter to accept the shown default)")
    gf   = ask_float("Goals scored per game      ", 1.50, 0)
    ga   = ask_float("Goals conceded per game    ", 1.20, 0)
    sot  = ask_float("Shots on target per game   ", 4.00, 0)
    win  = ask_float("Win rate  (0.0–1.0)        ", 0.40, 0, 1)
    draw = ask_float("Draw rate (0.0–1.0)        ", 0.25, 0, 1)
    if win + draw > 1.0:
        draw = max(0.0, 1.0 - win)
        print(f"    Draw rate adjusted to {draw:.2f} (win+draw cannot exceed 1.0)")

    if is_home:
        home_away_win = ask_float(
            "Home win rate (0.0–1.0)    ", round(min(1.0, win + 0.10), 2), 0, 1)
        return {"gf": gf, "ga": ga, "sot": sot, "win": win, "draw": draw,
                "home_away_win": home_away_win,
                "corners": ask_float("Corners per game           ", 5.00, 0)}
    else:
        home_away_win = ask_float(
            "Away win rate (0.0–1.0)    ", round(max(0.0, win - 0.10), 2), 0, 1)
        return {"gf": gf, "ga": ga, "sot": sot, "win": win, "draw": draw,
                "home_away_win": home_away_win,
                "corners": ask_float("Corners per game           ", 4.50, 0)}


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

def _elo_win_prob(elo_diff: float) -> float:
    """ELO-implied home win probability."""
    if math.isnan(elo_diff):
        return math.nan
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def build_features(home: dict, away: dict,
                   elo_diff: float,
                   form5_h: float, form5_a: float,
                   league_enc: int) -> tuple:
    """
    Build (base_feat, corner_feat) matching BASE_FEATURES / CORNER_FEATURES
    defined in train.py.

    No odds features — there is no live pre-match odds feed for upcoming
    fixtures (only historical closing odds in Matches.csv), so imp_h/imp_d/
    imp_a/book_margin were dropped from training (2026-08-21 fix: faking
    them with a constant odds triple was flattening every live prediction
    toward a near-uniform split). See train.py BASE_FEATURES.

    BASE_FEATURES order (22 features):
      h_r_gf, h_r_ga, h_r_gd,
      h_r_sot,
      h_r_win, h_r_draw, h_r_pts, h_r_home_win,
      a_r_gf, a_r_ga, a_r_gd,
      a_r_sot,
      a_r_win, a_r_draw, a_r_pts, a_r_away_win,
      elo_diff, elo_prob_h,
      form5_h, form5_a, form5_diff,
      league_enc

    CORNER_FEATURES = BASE_FEATURES + [h_r_corners, a_r_corners]
    """
    h_pts = 3.0 * home["win"] + 1.0 * home["draw"]
    a_pts = 3.0 * away["win"] + 1.0 * away["draw"]

    elo_prob_h = _elo_win_prob(elo_diff)

    base = [
        home["gf"],                  # h_r_gf
        home["ga"],                  # h_r_ga
        home["gf"] - home["ga"],     # h_r_gd
        home["sot"],                 # h_r_sot
        home["win"],                 # h_r_win
        home["draw"],                # h_r_draw
        h_pts,                       # h_r_pts
        home["home_away_win"],       # h_r_home_win
        away["gf"],                  # a_r_gf
        away["ga"],                  # a_r_ga
        away["gf"] - away["ga"],     # a_r_gd
        away["sot"],                 # a_r_sot
        away["win"],                 # a_r_win
        away["draw"],                # a_r_draw
        a_pts,                       # a_r_pts
        away["home_away_win"],       # a_r_away_win
        elo_diff,                    # elo_diff
        elo_prob_h,                  # elo_prob_h
        form5_h,                     # form5_h
        form5_a,                     # form5_a
        form5_h - form5_a,           # form5_diff
        float(league_enc),           # league_enc
    ]

    corner_ext = base + [home["corners"], away["corners"]]
    return np.array([base], dtype=float), np.array([corner_ext], dtype=float)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bar(p: float, width: int = 24) -> str:
    filled = max(0, min(width, round(p * width)))
    return "█" * filled + "░" * (width - filled)


def _over_prob(model, feat_vec: np.ndarray) -> float:
    proba   = model.predict_proba(feat_vec)[0]
    classes = list(model.classes_)
    return float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])


def print_predictions(home_team: str, away_team: str,
                      models: dict,
                      base_feat: np.ndarray, corner_feat: np.ndarray) -> None:

    W = 56
    divider = "─" * W

    r_data  = models["result"]
    r_enc   = r_data["result_encoder"]
    r_proba = r_data["model"].predict_proba(base_feat)[0]
    r_pairs = sorted(zip(r_enc.classes_, r_proba), key=lambda x: -x[1])

    label_map = {
        "H": f"  {home_team} Win",
        "D": "  Draw",
        "A": f"  {away_team} Win",
    }

    over_goals   = _over_prob(models["goals"]["model"],   base_feat)
    over_corners = _over_prob(models["corners"]["model"], corner_feat)
    under_goals  = 1.0 - over_goals
    under_corners = 1.0 - over_corners

    print("\n" + "═" * W)
    print(f"  PREDICTION  ·  {home_team}  vs  {away_team}")
    print("═" * W)

    print(f"\n  ┌ [1] MATCH RESULT")
    for cls, prob in r_pairs:
        label = label_map.get(cls, cls)
        print(f"  │  {label:<24} {_bar(prob)}  {prob * 100:5.1f}%")
    print(f"  └{divider[1:]}")

    print(f"\n  ┌ [2] TOTAL GOALS  (line: 2.5)")
    print(f"  │  {'  Over  2.5':<24} {_bar(over_goals)}  {over_goals * 100:5.1f}%")
    print(f"  │  {'  Under 2.5':<24} {_bar(under_goals)}  {under_goals * 100:5.1f}%")
    print(f"  └{divider[1:]}")

    print(f"\n  ┌ [3] TOTAL CORNERS  (line: 9.5)")
    print(f"  │  {'  Over  9.5':<24} {_bar(over_corners)}  {over_corners * 100:5.1f}%")
    print(f"  │  {'  Under 9.5':<24} {_bar(under_corners)}  {under_corners * 100:5.1f}%")
    print(f"  └{divider[1:]}")

    top_cls,  top_prob  = r_pairs[0]
    top_label = label_map.get(top_cls, top_cls).strip()
    goals_call   = "Over  2.5" if over_goals   >= 0.5 else "Under 2.5"
    corners_call = "Over  9.5" if over_corners >= 0.5 else "Under 9.5"

    print(f"\n  RECOMMENDED SELECTIONS")
    print(f"  {'Result':<14}  {top_label:<22}  ({top_prob * 100:.0f}% confidence)")
    print(f"  {'Goals':<14}  {goals_call:<22}  ({max(over_goals, under_goals) * 100:.0f}% confidence)")
    print(f"  {'Corners':<14}  {corners_call:<22}  ({max(over_corners, under_corners) * 100:.0f}% confidence)")
    print("\n" + "═" * W + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║     Football Match Prediction System  v2    ║")
    print("╚══════════════════════════════════════════════╝")

    print("\nLoading models … ", end="", flush=True)
    models = load_models()
    print("OK\n")

    home_team = input("Home team name : ").strip() or "Home Team"
    away_team = input("Away team name : ").strip() or "Away Team"

    league_map = models["result"].get("league_map", {"E0": 0})
    print(f"\nLeague options (sample): {list(league_map.keys())[:10]}")
    league_raw = input("League code (e.g. E0, SP1, D1, BRA) [default E0]: ").strip()
    league_enc = league_map.get(league_raw, league_map.get("E0", 0))

    home_stats = ask_team_stats(home_team, is_home=True)
    away_stats = ask_team_stats(away_team, is_home=False)

    print("\n  Optional pre-match context  (press Enter to skip / use default)")
    elo_diff = ask_float("ELO diff  HomeElo − AwayElo   ", 0.0)
    form5_h  = ask_float("Home team Form5 points (0–15) ", round(home_stats["win"] * 15), 0, 15)
    form5_a  = ask_float("Away team Form5 points (0–15) ", round(away_stats["win"] * 15), 0, 15)

    base_feat, corner_feat = build_features(
        home_stats, away_stats,
        elo_diff, form5_h, form5_a,
        league_enc,
    )

    print_predictions(home_team, away_team, models, base_feat, corner_feat)


if __name__ == "__main__":
    main()

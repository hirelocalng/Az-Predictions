"""
Football match prediction training script (v5).

Data sources:
  - Matches.csv          : 230 k rows, 38 leagues, 2000-2025 (primary)
  - data/brazil_2025.csv : FBref Série A 2025 (new, no overlap with Matches.csv)
  - data/argentina_2025.csv : FBref Liga Profesional 2025 (new)
  Note: brazil_2024.csv is skipped — fully covered by Matches.csv BRA division.

Key features:
  - Normalised implied probabilities from betting odds (imp_h/d/a)
  - ELO-implied win probability (elo_prob_h)
  - Rolling per-team stats: gf, ga, gd, sot, win, draw, pts, corners,
    home-specific win rate, away-specific win rate

Trains three XGBoost classifiers:
  1. Match result    (H / D / A)
  2. Goals O/U 2.5
  3. Corners O/U 9.5

Usage:
    python train.py
    (run extract_fbref.py first to generate the FBref CSVs)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR  = "data"
N_ROLLING = 5    # rolling-window size (games)
TUNE_ITER = 40   # RandomizedSearchCV iterations

# ---------------------------------------------------------------------------
# Feature lists  (order is significant — predict.py must match exactly)
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    # Rolling team stats (last N games, shift-1 — no leakage)
    "h_r_gf",  "h_r_ga",  "h_r_gd",          # home: goals for/against/diff
    "h_r_sot",                                  # home: shots on target
    "h_r_win", "h_r_draw", "h_r_pts",          # home: win/draw rate, pts/game
    "h_r_home_win",                             # home: win rate in home games only
    "a_r_gf",  "a_r_ga",  "a_r_gd",           # away: goals for/against/diff
    "a_r_sot",
    "a_r_win", "a_r_draw", "a_r_pts",
    "a_r_away_win",                             # away: win rate in away games only
    # ELO features (NaN where ELO unavailable — XGBoost handles natively)
    "elo_diff",                                 # HomeElo − AwayElo (raw rating gap)
    "elo_prob_h",                               # ELO implied P(home win) via logistic
    # Form (pre-computed in Matches.csv)
    "form5_h", "form5_a", "form5_diff",
    # Odds features — normalized implied probabilities (NaN-safe)
    "imp_h",  "imp_d",  "imp_a",               # P(H), P(D), P(A) after removing margin
    "book_margin",                              # bookmaker overround (signals uncertainty)
    "league_enc",
]
CORNER_FEATURES = BASE_FEATURES + ["h_r_corners", "a_r_corners"]

# Only these must be non-null for a row to be used in training
# (first game per team has NaN rolling stats from shift(1))
CORE_FEATURES  = ["h_r_gf", "h_r_ga", "a_r_gf", "a_r_ga"]
# For result/goals models, also require odds (sharpest signal)
RESULT_REQUIRE = CORE_FEATURES + ["imp_h", "imp_d", "imp_a"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _elo_win_prob(elo_diff: pd.Series) -> pd.Series:
    """Standard ELO win probability: P = 1 / (1 + 10^(−diff/400))."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_matches(data_dir: str) -> pd.DataFrame:
    """
    Load Matches.csv — 230 k rows, 38 leagues, 2000-2025.

    Pre-computed columns used directly (all pre-match, no leakage):
      HomeElo, AwayElo  →  elo_diff, elo_prob_h
      Form5Home, Form5Away, their diff
      OddHome, OddDraw, OddAway  →  imp_h, imp_d, imp_a, book_margin

    Post-game columns (HomeTarget, AwayTarget, HomeCorners, AwayCorners,
    FTHome, FTAway) are only used for rolling-feature computation via shift(1).
    """
    path = os.path.join(data_dir, "Matches.csv")
    raw  = pd.read_csv(path, low_memory=False)
    raw  = raw[raw["FTResult"].isin(["H", "D", "A"])].copy()

    hg    = pd.to_numeric(raw["FTHome"],  errors="coerce")
    ag    = pd.to_numeric(raw["FTAway"],  errors="coerce")
    h_elo = _col(raw, "HomeElo")
    a_elo = _col(raw, "AwayElo")
    f5h   = _col(raw, "Form5Home")
    f5a   = _col(raw, "Form5Away")

    # Betting odds → implied probabilities (guard against 0 or negative odds → inf)
    oh = _col(raw, "OddHome").where(lambda x: x > 0)   # non-positive → NaN
    od = _col(raw, "OddDraw").where(lambda x: x > 0)
    oa = _col(raw, "OddAway").where(lambda x: x > 0)
    raw_h = 1.0 / oh
    raw_d = 1.0 / od
    raw_a = 1.0 / oa
    raw_sum = raw_h + raw_d + raw_a          # includes bookmaker margin

    elo_diff = h_elo - a_elo

    df = pd.DataFrame({
        "home_team":    raw["HomeTeam"].astype(str).str.strip(),
        "away_team":    raw["AwayTeam"].astype(str).str.strip(),
        "home_goals":   hg,
        "away_goals":   ag,
        "result":       raw["FTResult"].str.strip(),
        "home_sot":     _col(raw, "HomeTarget"),
        "away_sot":     _col(raw, "AwayTarget"),
        "home_corners": _col(raw, "HomeCorners"),
        "away_corners": _col(raw, "AwayCorners"),
        "date":         pd.to_datetime(raw["MatchDate"], errors="coerce"),
        "league":       raw["Division"].astype(str),
        # ELO
        "elo_diff":     elo_diff,
        "elo_prob_h":   _elo_win_prob(elo_diff),
        # Form
        "form5_h":      f5h,
        "form5_a":      f5a,
        "form5_diff":   f5h - f5a,
        # Normalised implied probabilities
        "imp_h":        raw_h / raw_sum,
        "imp_d":        raw_d / raw_sum,
        "imp_a":        raw_a / raw_sum,
        "book_margin":  raw_sum - 1.0,
    })

    df.dropna(subset=["home_goals", "away_goals", "date"], inplace=True)
    n_odds = df["imp_h"].notna().sum()
    n_elo  = df["elo_diff"].notna().sum()
    print(f"  Loaded {len(df):,} matches  |  {df['league'].nunique()} leagues  |  "
          f"odds: {n_odds:,}  elo: {n_elo:,}")
    return df


def load_fbref_supplements(data_dir: str) -> pd.DataFrame:
    """
    Load FBref-extracted CSVs for Brazil 2025 and Argentina 2025.
    Brazil 2024 is skipped — it overlaps with Matches.csv (BRA goes to 2024-12-07).
    Matches from 2025-01-01 onward only, so no duplication.

    These CSVs have no odds / ELO / form — those columns are NaN and XGBoost
    handles them as missing values. The extra data still trains rolling stats.
    """
    files = [
        ("brazil_2025.csv",    "BRA_fbref"),
        ("argentina_2025.csv", "ARG_fbref"),
    ]

    dfs = []
    for fname, league in files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  Skipping {fname} (not found — run extract_fbref.py first)")
            continue
        raw = pd.read_csv(path, parse_dates=["date"])
        # Keep only 2025+ to avoid any overlap with Matches.csv
        raw = raw[raw["date"] >= "2025-01-01"].copy()
        raw["league"] = league
        # Map to the same column schema as load_matches()
        df = pd.DataFrame({
            "home_team":    raw["home_team"].astype(str).str.strip(),
            "away_team":    raw["away_team"].astype(str).str.strip(),
            "home_goals":   raw["home_goals"],
            "away_goals":   raw["away_goals"],
            "result":       raw["result"],
            "home_sot":     raw.get("home_sot", pd.Series(np.nan, index=raw.index)),
            "away_sot":     raw.get("away_sot", pd.Series(np.nan, index=raw.index)),
            "home_corners": raw.get("home_corners", pd.Series(np.nan, index=raw.index)),
            "away_corners": raw.get("away_corners", pd.Series(np.nan, index=raw.index)),
            "date":         raw["date"],
            "league":       raw["league"],
            # Pre-computed features unavailable — NaN; XGBoost handles natively
            "elo_diff":     np.nan,
            "elo_prob_h":   np.nan,
            "form5_h":      np.nan,
            "form5_a":      np.nan,
            "form5_diff":   np.nan,
            "imp_h":        np.nan,
            "imp_d":        np.nan,
            "imp_a":        np.nan,
            "book_margin":  np.nan,
        })
        df.dropna(subset=["home_goals", "away_goals"], inplace=True)
        dfs.append(df)
        print(f"  FBref {fname:<22} {len(df):>4} matches  "
              f"({df['date'].min().date()} – {df['date'].max().date()})")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_rolling_stats(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Per-team rolling averages over the last n games (shift(1) — no leakage):
      gf, ga, gd, sot, corners, win_rate, draw_rate, pts_per_game,
      home_win_rate  (home team, home games only),
      away_win_rate  (away team, away games only).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    home_v = pd.DataFrame({
        "date":      df["date"],
        "team":      df["home_team"],
        "gf":        df["home_goals"],
        "ga":        df["away_goals"],
        "sot":       df["home_sot"],
        "corners":   df["home_corners"],
        "win":       df["result"].map({"H": 1.0, "D": 0.0, "A": 0.0}),
        "draw":      df["result"].map({"H": 0.0, "D": 1.0, "A": 0.0}),
        "pts":       df["result"].map({"H": 3.0, "D": 1.0, "A": 0.0}),
        "match_idx": df.index,
        "side":      "home",
    })
    away_v = pd.DataFrame({
        "date":      df["date"],
        "team":      df["away_team"],
        "gf":        df["away_goals"],
        "ga":        df["home_goals"],
        "sot":       df["away_sot"],
        "corners":   df["away_corners"],
        "win":       df["result"].map({"H": 0.0, "D": 0.0, "A": 1.0}),
        "draw":      df["result"].map({"H": 0.0, "D": 1.0, "A": 0.0}),
        "pts":       df["result"].map({"H": 0.0, "D": 1.0, "A": 3.0}),
        "match_idx": df.index,
        "side":      "away",
    })

    all_g = (pd.concat([home_v, away_v], ignore_index=True)
               .sort_values(["team", "date"])
               .reset_index(drop=True))

    for col in ("gf", "ga", "sot", "win", "draw", "pts", "corners"):
        all_g[f"r_{col}"] = (
            all_g.groupby("team", sort=False)[col]
            .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        )
    all_g["r_gd"] = all_g["r_gf"] - all_g["r_ga"]

    # Home-only win rate for home team
    home_only = all_g[all_g["side"] == "home"].sort_values(["team", "date"])
    home_win_r = (
        home_only.groupby("team", sort=False)["win"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )
    all_g["r_home_win"] = np.nan
    all_g.loc[home_only.index, "r_home_win"] = home_win_r

    # Away-only win rate for away team
    away_only = all_g[all_g["side"] == "away"].sort_values(["team", "date"])
    away_win_r = (
        away_only.groupby("team", sort=False)["win"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )
    all_g["r_away_win"] = np.nan
    all_g.loc[away_only.index, "r_away_win"] = away_win_r

    roll_base = ["match_idx", "r_gf", "r_ga", "r_gd", "r_sot",
                 "r_win", "r_draw", "r_pts", "r_corners"]

    h_roll = (all_g[all_g["side"] == "home"][roll_base + ["r_home_win"]]
              .set_index("match_idx"))
    a_roll = (all_g[all_g["side"] == "away"][roll_base + ["r_away_win"]]
              .set_index("match_idx"))

    df = df.join(h_roll.add_prefix("h_"))
    df = df.join(a_roll.add_prefix("a_"))
    return df


# ---------------------------------------------------------------------------
# Model training with hyperparameter tuning
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray, label: str,
                multiclass: bool = False) -> tuple:
    """
    RandomizedSearchCV on a 40k-row sample → retrain best params on full set.
    """
    # Replace any stray inf (e.g. from 1/0 edge cases in odds) with NaN
    X = np.where(np.isinf(X), np.nan, X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_dist = {
        "n_estimators":     [300, 500, 700, 1000, 1500],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.02, 0.05, 0.08, 0.10],
        "subsample":        [0.60, 0.70, 0.80, 0.90],
        "colsample_bytree": [0.50, 0.60, 0.70, 0.80],
        "min_child_weight": [1, 3, 5, 10],
        "gamma":            [0, 0.05, 0.10, 0.20, 0.30],
        "reg_alpha":        [0, 0.01, 0.05, 0.10, 0.50],
        "reg_lambda":       [0.5, 1.0, 1.5, 2.0, 3.0],
    }

    eval_metric = "mlogloss" if multiclass else "logloss"
    base_clf = XGBClassifier(
        eval_metric=eval_metric,
        random_state=42,
        verbosity=0,
        tree_method="hist",
    )

    sample_size = min(len(X_tr), 40_000)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_tr), sample_size, replace=False)

    print(f"  Tuning on {sample_size:,} samples  ({TUNE_ITER} iters × 3-fold CV) …")
    search = RandomizedSearchCV(
        base_clf, param_dist,
        n_iter=TUNE_ITER, cv=3,
        scoring="accuracy", random_state=42,
        n_jobs=-1, verbose=0,
    )
    search.fit(X_tr[idx], y_tr[idx])
    best_p = search.best_params_
    print(f"  CV best accuracy: {search.best_score_ * 100:.1f}%  "
          f"(depth={best_p.get('max_depth')}, lr={best_p.get('learning_rate')}, "
          f"n={best_p.get('n_estimators')})")

    # Retrain on full training data with best params
    final_clf = XGBClassifier(
        **best_p,
        eval_metric=eval_metric,
        random_state=42,
        verbosity=0,
        tree_method="hist",
    )
    final_clf.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, final_clf.predict(X_te))
    print(f"  {label:<28}  test accuracy: {acc * 100:.1f}%   ({len(X):,} samples)")
    return final_clf, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 62)
    print("  FOOTBALL PREDICTION MODEL TRAINING  (v5)")
    print("=" * 62)

    # --- Load ---
    print("\n[1] Loading datasets")
    df_main = load_matches(DATA_DIR)

    print("\n  FBref supplements (2025 seasons, no overlap with Matches.csv):")
    df_fbref = load_fbref_supplements(DATA_DIR)

    if len(df_fbref) > 0:
        df = pd.concat([df_main, df_fbref], ignore_index=True)
    else:
        df = df_main

    print(f"\n  Combined total: {len(df):,} matches")

    # --- Rolling features ---
    print(f"\n[2] Engineering rolling features  (window = {N_ROLLING} games)")
    df = compute_rolling_stats(df, N_ROLLING)

    le_league = LabelEncoder().fit(df["league"])
    df["league_enc"] = le_league.transform(df["league"])
    league_map = dict(zip(
        le_league.classes_.tolist(),
        le_league.transform(le_league.classes_).tolist(),
    ))
    print(f"  League encoding: {len(league_map)} leagues")

    # --- Model 1: Match result ---
    print("\n[3] Match Result model  (H / D / A)")
    le_result = LabelEncoder()
    # Require core rolling stats + odds (ensures we use the sharp odds signal)
    df_r = df[df["result"].isin(["H", "D", "A"])].dropna(subset=RESULT_REQUIRE)
    X_r  = df_r[BASE_FEATURES].values.astype(float)
    y_r  = le_result.fit_transform(df_r["result"])
    dist = dict(zip(le_result.classes_, np.bincount(y_r)))
    print(f"  Samples: {len(X_r):,}  |  class dist: {dist}")

    # Naive odds baseline on this exact subset
    naive_pred = np.where(
        (df_r["imp_h"] >= df_r["imp_d"]) & (df_r["imp_h"] >= df_r["imp_a"]), "H",
        np.where(df_r["imp_d"] >= df_r["imp_a"], "D", "A")
    )
    naive_acc = (naive_pred == df_r["result"].values).mean()
    print(f"  Naive implied-prob baseline: {naive_acc * 100:.1f}%  (model must beat this)")

    model_result, acc_r = train_model(X_r, y_r, "Match result (H/D/A)", multiclass=True)

    with open("result_model.pkl", "wb") as f:
        pickle.dump({
            "model":          model_result,
            "result_encoder": le_result,
            "league_encoder": le_league,
            "league_map":     league_map,
            "features":       BASE_FEATURES,
        }, f)
    print("  Saved: result_model.pkl")

    # --- Model 2: Goals O/U 2.5 ---
    print("\n[4] Goals Over/Under 2.5 model")
    df_g = df.copy()
    df_g["over_2_5"] = ((df_g["home_goals"] + df_g["away_goals"]) > 2.5).astype(int)
    df_g = df_g.dropna(subset=RESULT_REQUIRE + ["over_2_5"])
    X_g  = df_g[BASE_FEATURES].values.astype(float)
    y_g  = df_g["over_2_5"].values

    model_goals, acc_g = train_model(X_g, y_g, "Goals O/U 2.5")

    with open("goals_model.pkl", "wb") as f:
        pickle.dump({
            "model":          model_goals,
            "league_encoder": le_league,
            "league_map":     league_map,
            "features":       BASE_FEATURES,
        }, f)
    print("  Saved: goals_model.pkl")

    # --- Model 3: Corners O/U 9.5 ---
    print("\n[5] Corners Over/Under 9.5 model")
    df_c = df.copy()
    df_c["total_corners"] = df_c["home_corners"] + df_c["away_corners"]
    df_c["over_9_5"]      = (df_c["total_corners"] > 9.5).astype(int)
    df_c = df_c.dropna(subset=CORE_FEATURES + ["h_r_corners", "a_r_corners", "over_9_5"])
    print(f"  Rows with corners data: {len(df_c):,}")
    X_c  = df_c[CORNER_FEATURES].values.astype(float)
    y_c  = df_c["over_9_5"].values

    model_corners, acc_c = train_model(X_c, y_c, "Corners O/U 9.5")

    with open("corners_model.pkl", "wb") as f:
        pickle.dump({
            "model":          model_corners,
            "league_encoder": le_league,
            "league_map":     league_map,
            "features":       CORNER_FEATURES,
        }, f)
    print("  Saved: corners_model.pkl")

    # --- Summary ---
    print("\n" + "=" * 62)
    print("  TRAINING COMPLETE — ACCURACY SUMMARY")
    print("=" * 62)
    print(f"  Match result  (H/D/A)    :  {acc_r * 100:.1f}%")
    print(f"  Goals Over/Under 2.5     :  {acc_g * 100:.1f}%")
    print(f"  Corners Over/Under 9.5   :  {acc_c * 100:.1f}%")
    print("=" * 62)
    print("\n  Models saved:")
    print("    result_model.pkl  |  goals_model.pkl  |  corners_model.pkl")


if __name__ == "__main__":
    main()

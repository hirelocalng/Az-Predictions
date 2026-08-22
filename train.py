"""
Football match prediction training script (v5).

Data sources:
  - Matches.csv          : 244 k rows, 38 leagues, 2000-2026 (primary)
  - data/brazil_2025.csv : FBref Serie A 2025 (new, no overlap with Matches.csv)
  - data/argentina_2025.csv : FBref Liga Profesional 2025 (new)
  Note: brazil_2024.csv is skipped -- fully covered by Matches.csv BRA division.

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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import epl_data

warnings.filterwarnings("ignore")

DATA_DIR  = "data"
N_ROLLING = 5    # rolling-window size (games)
TUNE_ITER = 40   # RandomizedSearchCV iterations

# ---------------------------------------------------------------------------
# Feature lists  (order is significant — predict.py must match exactly)
# ---------------------------------------------------------------------------


ODDS_FEATURES = ["imp_h", "imp_d", "imp_a", "book_margin"]

# No live pre-match odds feed exists for upcoming fixtures (only historical
# closing odds in Matches.csv) — serving code was faking imp_h/imp_d/imp_a/
# book_margin with a hardcoded constant triple for every fixture, which
# flattened every live prediction toward a near-uniform split (see
# 2026-08-21 incident: azpredicts.com showing 34/34/31 for Arsenal-Coventry).
# BASE_FEATURES is odds-free so training always matches what serving can
# actually provide at prediction time.
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
    "league_enc",
]
CORNER_FEATURES = BASE_FEATURES + ["h_r_corners", "a_r_corners"]

# Only these must be non-null for a row to be used in training
# (first game per team has NaN rolling stats from shift(1))
CORE_FEATURES  = ["h_r_gf", "h_r_ga", "a_r_gf", "a_r_ga"]
# No odds requirement — odds are never available at serve time, so rows
# without historical odds are now usable for training too.
RESULT_REQUIRE = CORE_FEATURES


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


LALIGA_TEAM_ALIAS = {"Villareal": "Villarreal"}  # spelling variant in older files


def load_laliga_early(data_dir: str, cutoff) -> pd.DataFrame:
    """
    data/LaLiga/*.csv — football-data.co.uk La Liga (SP1) season files,
    user-provided, 1993-94 through 2025-26 (2001-02 missing). Only seasons
    strictly before `cutoff` are used (Matches.csv's own SP1 coverage starts
    2000-09-09), so there's no overlap/dedup needed with the main dataset.

    Results only, no corners/odds/elo/form for this era — NaN, same as the
    other supplement sources below; XGBoost handles missing values natively.
    """
    import glob
    laliga_dir = os.path.join(data_dir, "LaLiga")
    if not os.path.isdir(laliga_dir):
        return pd.DataFrame()

    dfs = []
    for path in sorted(glob.glob(os.path.join(laliga_dir, "*.csv"))):
        raw = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip", engine="python")
        raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce")
        raw = raw[raw["Date"] < cutoff]
        if raw.empty:
            continue
        raw["HomeTeam"] = raw["HomeTeam"].map(lambda t: LALIGA_TEAM_ALIAS.get(t, t))
        raw["AwayTeam"] = raw["AwayTeam"].map(lambda t: LALIGA_TEAM_ALIAS.get(t, t))
        df = pd.DataFrame({
            "home_team":    raw["HomeTeam"].astype(str).str.strip(),
            "away_team":    raw["AwayTeam"].astype(str).str.strip(),
            "home_goals":   pd.to_numeric(raw["FTHG"], errors="coerce"),
            "away_goals":   pd.to_numeric(raw["FTAG"], errors="coerce"),
            "result":       raw["FTR"].astype(str).str.strip(),
            "home_sot":     np.nan, "away_sot": np.nan,
            "home_corners": np.nan, "away_corners": np.nan,
            "date":         raw["Date"],
            "league":       "SP1",
            "elo_diff": np.nan, "elo_prob_h": np.nan,
            "form5_h": np.nan, "form5_a": np.nan, "form5_diff": np.nan,
            "imp_h": np.nan, "imp_d": np.nan, "imp_a": np.nan, "book_margin": np.nan,
        })
        df = df[df["result"].isin(["H", "D", "A"])]
        df.dropna(subset=["home_goals", "away_goals", "date"], inplace=True)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    print(f"  La Liga early era (pre-{cutoff.date()}): {len(out):,} matches  "
          f"({out['date'].min().date()} - {out['date'].max().date()})  [results only]")
    return out


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
# EPL-only (E0) dataset -- 2026-27 pre-season retrain
# ---------------------------------------------------------------------------

RECENCY_HALF_LIFE_YEARS = 8.0  # age-decay sample weight: a game this old counts half


def build_epl_frame(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    E0-only training frame. Elo and Form5 come exclusively from epl_data.py's
    match-by-match computation -- Matches.csv's own HomeElo/AwayElo/Form3*/
    Form5* columns are never read for this frame. One source per feature, no
    mixing with the 37-league pipeline in load_matches()/compute_rolling_stats
    above (that pipeline is untouched and still serves the deployed
    multi-league model independently of this function).
    """
    epl = epl_data.build_epl_dataset(data_dir)

    elo_diff = epl["elo_home_pre"] - epl["elo_away_pre"]
    oh = _col(epl, "OddHome").where(lambda x: x > 0)
    od = _col(epl, "OddDraw").where(lambda x: x > 0)
    oa = _col(epl, "OddAway").where(lambda x: x > 0)
    raw_h, raw_d, raw_a = 1.0 / oh, 1.0 / od, 1.0 / oa
    raw_sum = raw_h + raw_d + raw_a

    df = pd.DataFrame({
        "home_team":    epl["HomeTeam"],
        "away_team":    epl["AwayTeam"],
        "home_goals":   pd.to_numeric(epl["FTHome"], errors="coerce"),
        "away_goals":   pd.to_numeric(epl["FTAway"], errors="coerce"),
        "result":       epl["FTResult"],
        "home_sot":     _col(epl, "HomeTarget"),
        "away_sot":     _col(epl, "AwayTarget"),
        "home_corners": _col(epl, "HomeCorners"),
        "away_corners": _col(epl, "AwayCorners"),
        "date":         epl["MatchDate"],
        "league":       "E0",
        "elo_diff":     elo_diff,
        "elo_prob_h":   _elo_win_prob(elo_diff),
        "form5_h":      epl["form_n_home"],
        "form5_a":      epl["form_n_away"],
        "form5_diff":   epl["form_n_home"] - epl["form_n_away"],
        "imp_h":        raw_h / raw_sum,
        "imp_d":        raw_d / raw_sum,
        "imp_a":        raw_a / raw_sum,
        "book_margin":  raw_sum - 1.0,
    })
    df.dropna(subset=["home_goals", "away_goals", "date"], inplace=True)
    df.sort_values("date", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df


def add_recency_weight(df: pd.DataFrame, as_of, half_life_years: float = RECENCY_HALF_LIFE_YEARS) -> pd.Series:
    """Exponential age-decay sample weight: weight = 0.5 ** (years_before_as_of / half_life)."""
    as_of = pd.Timestamp(as_of)
    years_ago = (as_of - df["date"]).dt.days / 365.25
    return 0.5 ** (years_ago.clip(lower=0) / half_life_years)


# ---------------------------------------------------------------------------
# Model training with hyperparameter tuning
# ---------------------------------------------------------------------------

def _tune_and_fit(X_tr: np.ndarray, y_tr: np.ndarray,
                  X_te: np.ndarray, y_te: np.ndarray,
                  label: str, multiclass: bool = False):
    """
    RandomizedSearchCV on a 40k-row subsample of X_tr, then refit on full X_tr.
    Returns (fitted_model, test_accuracy).
    """
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    X_te = np.where(np.isinf(X_te), np.nan, X_te)

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
        eval_metric=eval_metric, random_state=42, verbosity=0, tree_method="hist",
    )

    sample_size = min(len(X_tr), 40_000)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_tr), sample_size, replace=False)

    print(f"  Tuning {label}: {sample_size:,} samples ({TUNE_ITER} iters x 3-fold CV)")
    search = RandomizedSearchCV(
        base_clf, param_dist,
        n_iter=TUNE_ITER, cv=3, scoring="accuracy",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_tr[idx], y_tr[idx])
    best_p = search.best_params_
    print(f"  CV best: {search.best_score_ * 100:.1f}%  "
          f"(depth={best_p.get('max_depth')}, lr={best_p.get('learning_rate')}, "
          f"n={best_p.get('n_estimators')})")

    clf = XGBClassifier(
        **best_p, eval_metric=eval_metric,
        random_state=42, verbosity=0, tree_method="hist",
    )
    clf.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  {label:<28}  test accuracy: {acc * 100:.1f}%  ({len(X_tr)+len(X_te):,} samples)")
    return clf, acc


def _tune_and_fit_weighted(X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
                            X_te: np.ndarray, y_te: np.ndarray,
                            label: str, multiclass: bool = False):
    """
    Identical to _tune_and_fit -- same param_dist, TUNE_ITER, cv, scoring,
    random_state -- except sample_weight is passed through to both the
    RandomizedSearchCV fit and the final refit. Used only for the recency-
    weighting A/B comparison; the hyperparameter search itself is unchanged.
    """
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    X_te = np.where(np.isinf(X_te), np.nan, X_te)

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
        eval_metric=eval_metric, random_state=42, verbosity=0, tree_method="hist",
    )

    sample_size = min(len(X_tr), 40_000)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_tr), sample_size, replace=False)

    print(f"  Tuning {label} [weighted]: {sample_size:,} samples ({TUNE_ITER} iters x 3-fold CV)")
    search = RandomizedSearchCV(
        base_clf, param_dist,
        n_iter=TUNE_ITER, cv=3, scoring="accuracy",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_tr[idx], y_tr[idx], sample_weight=w_tr[idx])
    best_p = search.best_params_
    print(f"  CV best: {search.best_score_ * 100:.1f}%  "
          f"(depth={best_p.get('max_depth')}, lr={best_p.get('learning_rate')}, "
          f"n={best_p.get('n_estimators')})")

    clf = XGBClassifier(
        **best_p, eval_metric=eval_metric,
        random_state=42, verbosity=0, tree_method="hist",
    )
    clf.fit(X_tr, y_tr, sample_weight=w_tr)

    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  {label:<28}  test accuracy: {acc * 100:.1f}%  ({len(X_tr)+len(X_te):,} samples)")
    return clf, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 62)
    print("  FOOTBALL PREDICTION MODEL TRAINING  (v6)")
    print("=" * 62)

    # --- Load ---
    print("\n[1] Loading datasets")
    df_main = load_matches(DATA_DIR)

    print("\n  FBref supplements (2025 seasons, no overlap with Matches.csv):")
    df_fbref = load_fbref_supplements(DATA_DIR)

    sp1_min = df_main.loc[df_main["league"] == "SP1", "date"].min()
    print("\n  La Liga early era supplement:")
    df_laliga = load_laliga_early(DATA_DIR, sp1_min) if pd.notna(sp1_min) else pd.DataFrame()

    extra = [d for d in (df_fbref, df_laliga) if len(d) > 0]
    df = pd.concat([df_main] + extra, ignore_index=True) if extra else df_main

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

    # --- Chronological 80/20 split ---
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    split_date = df.iloc[split_idx]["date"]
    print(f"\n[3] Chronological split: train = pre {split_date.date()}, "
          f"test = {split_date.date()} to {df['date'].max().date()}")
    tr_mask = df["date"] < split_date
    te_mask = ~tr_mask
    print(f"  Train: {tr_mask.sum():,}  Test: {te_mask.sum():,}")

    # --- Model 1: Match result ---
    print("\n[4] Match Result model  (H / D / A)")
    le_result = LabelEncoder()
    df_r  = df[df["result"].isin(["H", "D", "A"])].dropna(subset=RESULT_REQUIRE)
    tr_r  = df_r[df_r["date"] < split_date]
    te_r  = df_r[df_r["date"] >= split_date]
    le_result.fit(df_r["result"])
    X_tr_r = tr_r[BASE_FEATURES].values.astype(float)
    y_tr_r = le_result.transform(tr_r["result"])
    X_te_r = te_r[BASE_FEATURES].values.astype(float)
    y_te_r = le_result.transform(te_r["result"])
    print(f"  Train: {len(X_tr_r):,}  Test: {len(X_te_r):,}")

    # Naive odds baseline on test set — informational only (odds are no longer
    # a model feature, but historical odds still exist on rows that have them,
    # so this stays useful as a sanity floor). Restricted to the subset of the
    # test set that actually has odds; RESULT_REQUIRE no longer guarantees it.
    te_r_odds = te_r.dropna(subset=ODDS_FEATURES)
    if len(te_r_odds):
        naive_pred = np.where(
            (te_r_odds["imp_h"] >= te_r_odds["imp_d"]) & (te_r_odds["imp_h"] >= te_r_odds["imp_a"]), "H",
            np.where(te_r_odds["imp_d"] >= te_r_odds["imp_a"], "D", "A")
        )
        naive_acc = (naive_pred == te_r_odds["result"].values).mean()
        print(f"  (naive baseline computed on {len(te_r_odds):,}/{len(te_r):,} test rows that have odds)")
    else:
        naive_acc = 0.0
    print(f"  Naive implied-prob baseline (test): {naive_acc * 100:.1f}%")

    # Backtest: old model on test set (informational only -- old models used random
    # splits so some test-period data leaked into their training set; we use
    # naive_acc as the deploy bar, not old model accuracy)
    try:
        with open("result_model.pkl", "rb") as f:
            old_res = pickle.load(f)
        old_acc_r_info = accuracy_score(y_te_r, old_res["model"].predict(X_te_r))
        print(f"  Old model accuracy (test, FYI): {old_acc_r_info * 100:.1f}%  "
              f"[note: old model trained with random splits -- inflated on this test set]")
    except Exception as e:
        print(f"  Old result model not loaded: {e}")

    model_result, new_acc_r = _tune_and_fit(
        X_tr_r, y_tr_r, X_te_r, y_te_r, "Match result (H/D/A)", multiclass=True)

    # --- Model 2: Goals O/U 2.5 ---
    print("\n[5] Goals Over/Under 2.5 model")
    df_g = df.copy()
    df_g["over_2_5"] = ((df_g["home_goals"] + df_g["away_goals"]) > 2.5).astype(int)
    df_g = df_g.dropna(subset=RESULT_REQUIRE + ["over_2_5"])
    tr_g = df_g[df_g["date"] < split_date]
    te_g = df_g[df_g["date"] >= split_date]
    X_tr_g = tr_g[BASE_FEATURES].values.astype(float)
    y_tr_g = tr_g["over_2_5"].values
    X_te_g = te_g[BASE_FEATURES].values.astype(float)
    y_te_g = te_g["over_2_5"].values
    print(f"  Train: {len(X_tr_g):,}  Test: {len(X_te_g):,}")

    # Naive goals baseline: majority class (over 2.5 is ~56% of games)
    naive_acc_g = max(y_te_g.mean(), 1 - y_te_g.mean())
    print(f"  Naive majority-class baseline (test): {naive_acc_g * 100:.1f}%")
    try:
        with open("goals_model.pkl", "rb") as f:
            old_gls = pickle.load(f)
        old_acc_g_info = accuracy_score(y_te_g, old_gls["model"].predict(X_te_g))
        print(f"  Old model accuracy (test, FYI): {old_acc_g_info * 100:.1f}%")
    except Exception as e:
        print(f"  Old goals model not loaded: {e}")

    model_goals, new_acc_g = _tune_and_fit(
        X_tr_g, y_tr_g, X_te_g, y_te_g, "Goals O/U 2.5")

    # --- Model 3: Corners O/U 9.5 ---
    print("\n[6] Corners Over/Under 9.5 model")
    df_c = df.copy()
    df_c["total_corners"] = df_c["home_corners"] + df_c["away_corners"]
    df_c["over_9_5"]      = (df_c["total_corners"] > 9.5).astype(int)
    df_c = df_c.dropna(subset=CORE_FEATURES + ["h_r_corners", "a_r_corners", "over_9_5"])
    tr_c = df_c[df_c["date"] < split_date]
    te_c = df_c[df_c["date"] >= split_date]
    print(f"  Train: {len(tr_c):,}  Test: {len(te_c):,}")
    X_tr_c = tr_c[CORNER_FEATURES].values.astype(float)
    y_tr_c = tr_c["over_9_5"].values
    X_te_c = te_c[CORNER_FEATURES].values.astype(float)
    y_te_c = te_c["over_9_5"].values

    naive_acc_c = max(y_te_c.mean(), 1 - y_te_c.mean())
    print(f"  Naive majority-class baseline (test): {naive_acc_c * 100:.1f}%")
    try:
        with open("corners_model.pkl", "rb") as f:
            old_cor = pickle.load(f)
        old_acc_c_info = accuracy_score(y_te_c, old_cor["model"].predict(X_te_c))
        print(f"  Old model accuracy (test, FYI): {old_acc_c_info * 100:.1f}%")
    except Exception as e:
        print(f"  Old corners model not loaded: {e}")

    model_corners, new_acc_c = _tune_and_fit(
        X_tr_c, y_tr_c, X_te_c, y_te_c, "Corners O/U 9.5")

    # --- Backtest summary ---
    print("\n" + "=" * 62)
    print("  BACKTEST SUMMARY  (chronological test set 2022-2026)")
    print("=" * 62)
    print(f"  Result  (H/D/A)  naive={naive_acc * 100:.1f}%   new={new_acc_r * 100:.1f}%"
          f"  vs-naive={new_acc_r - naive_acc:+.3f}")
    print(f"  Goals   O/U 2.5  naive={naive_acc_g * 100:.1f}%   new={new_acc_g * 100:.1f}%"
          f"  vs-naive={new_acc_g - naive_acc_g:+.3f}")
    print(f"  Corners O/U 9.5  naive={naive_acc_c * 100:.1f}%   new={new_acc_c * 100:.1f}%"
          f"  vs-naive={new_acc_c - naive_acc_c:+.3f}")

    # --- League-specific breakdown (SP1 / La Liga) ---
    sp1_te_r = te_r[te_r["league"] == "SP1"]
    sp1_te_g = te_g[te_g["league"] == "SP1"]
    sp1_te_c = te_c[te_c["league"] == "SP1"]
    print("\n" + "-" * 62)
    print("  LA LIGA (SP1) — HELD-OUT TEST SUBSET (same split as above)")
    print("-" * 62)
    if len(sp1_te_r):
        sp1_acc_r = accuracy_score(
            le_result.transform(sp1_te_r["result"]),
            model_result.predict(sp1_te_r[BASE_FEATURES].values.astype(float)))
        sp1_naive_r = (np.where(
            (sp1_te_r["imp_h"] >= sp1_te_r["imp_d"]) & (sp1_te_r["imp_h"] >= sp1_te_r["imp_a"]), "H",
            np.where(sp1_te_r["imp_d"] >= sp1_te_r["imp_a"], "D", "A")
        ) == sp1_te_r["result"].values).mean()
        print(f"  Result  (H/D/A)  n={len(sp1_te_r)}  naive={sp1_naive_r*100:.1f}%  new={sp1_acc_r*100:.1f}%")
    else:
        print("  Result: no SP1 rows in held-out test set")
    if len(sp1_te_g):
        sp1_acc_g = accuracy_score(
            sp1_te_g["over_2_5"].values,
            model_goals.predict(sp1_te_g[BASE_FEATURES].values.astype(float)))
        sp1_naive_g = max(sp1_te_g["over_2_5"].mean(), 1 - sp1_te_g["over_2_5"].mean())
        print(f"  Goals   O/U 2.5  n={len(sp1_te_g)}  naive={sp1_naive_g*100:.1f}%  new={sp1_acc_g*100:.1f}%")
    else:
        print("  Goals: no SP1 rows in held-out test set")
    if len(sp1_te_c):
        sp1_acc_c = accuracy_score(
            sp1_te_c["over_9_5"].values,
            model_corners.predict(sp1_te_c[CORNER_FEATURES].values.astype(float)))
        sp1_naive_c = max(sp1_te_c["over_9_5"].mean(), 1 - sp1_te_c["over_9_5"].mean())
        print(f"  Corners O/U 9.5  n={len(sp1_te_c)}  naive={sp1_naive_c*100:.1f}%  new={sp1_acc_c*100:.1f}%")
    else:
        print("  Corners: no SP1 rows in held-out test set")
    print("-" * 62)

    # --- Deploy gate: new model must beat naive baseline on result + goals ---
    # (Old model scores are informational only -- they were trained with random
    #  splits so they had data leakage into this chronological test period.)
    deploy = (new_acc_r >= naive_acc) and (new_acc_g >= naive_acc_g)

    if deploy:
        print("\n  [DEPLOY] New models beat naive baseline -- deploying")
        with open("result_model.pkl", "wb") as f:
            pickle.dump({
                "model":          model_result,
                "result_encoder": le_result,
                "league_encoder": le_league,
                "league_map":     league_map,
                "features":       BASE_FEATURES,
            }, f)
        with open("goals_model.pkl", "wb") as f:
            pickle.dump({
                "model":          model_goals,
                "league_encoder": le_league,
                "league_map":     league_map,
                "features":       BASE_FEATURES,
            }, f)
        with open("corners_model.pkl", "wb") as f:
            pickle.dump({
                "model":          model_corners,
                "league_encoder": le_league,
                "league_map":     league_map,
                "features":       CORNER_FEATURES,
            }, f)
        print("  Saved: result_model.pkl  goals_model.pkl  corners_model.pkl")
        _build_form_cache(df)
    else:
        reasons = []
        if new_acc_r <= naive_acc:
            reasons.append(f"result {new_acc_r*100:.1f}% <= naive {naive_acc*100:.1f}%")
        if new_acc_g <= naive_acc_g:
            reasons.append(f"goals {new_acc_g*100:.1f}% <= naive {naive_acc_g*100:.1f}%")
        print(f"\n  [SKIP] NOT deploying: {'; '.join(reasons)}")

    print("=" * 62)


def _build_form_cache(df: pd.DataFrame, n: int = 5, lookback_days: int = 365):
    """
    Compute per-team rolling-average stats (last n games) for all active teams
    and save to data/club_form_cache.csv.  Deployed to Railway so the API can
    give real team stats even when data/Matches.csv is absent.
    """
    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    active_teams = set(
        df.loc[df["date"] >= cutoff, "home_team"].tolist() +
        df.loc[df["date"] >= cutoff, "away_team"].tolist()
    )

    rows = []
    for team in sorted(active_teams):
        mask   = (df["home_team"] == team) | (df["away_team"] == team)
        recent = df[mask].sort_values("date").tail(n)
        if recent.empty:
            continue

        gf = ga = sot_s = sot_n = cor_s = cor_n = 0.0
        wins = draws = pts = hwin = hg = awin = ag_ = 0

        for _, row in recent.iterrows():
            ih = (row["home_team"] == team)
            g  = float(row["home_goals"] if ih else row["away_goals"])
            gc = float(row["away_goals"] if ih else row["home_goals"])
            gf += g; ga += gc

            s = row["home_sot"] if ih else row["away_sot"]
            c = row["home_corners"] if ih else row["away_corners"]
            if pd.notna(s) and float(s) > 0: sot_s += float(s); sot_n += 1
            if pd.notna(c) and float(c) > 0: cor_s += float(c); cor_n += 1

            if g > gc:    wins += 1; pts += 3
            elif g == gc: draws += 1; pts += 1
            if ih:   hg  += 1; hwin += int(g > gc)
            else:    ag_ += 1; awin += int(g > gc)

        nm = len(recent)
        # Derive league from most recent game
        last = recent.iloc[-1]
        league = last["league"] if "league" in recent.columns else ""

        rows.append({
            "team":      team,
            "league":    league,
            "gf":        round(gf / nm, 3),
            "ga":        round(ga / nm, 3),
            "sot":       round(sot_s / sot_n if sot_n else 4.5, 3),
            "corners":   round(cor_s / cor_n if cor_n else 5.0, 3),
            "win":       round(wins  / nm, 3),
            "draw":      round(draws / nm, 3),
            "hwn":       round(hwin  / hg  if hg  else min(wins / nm + 0.08, 1.0), 3),
            "awn":       round(awin  / ag_ if ag_ else max(wins / nm - 0.08, 0.0), 3),
            "last_date": recent["date"].max().strftime("%Y-%m-%d"),
            "n_games":   nm,
        })

    cache = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, "club_form_cache.csv")
    cache.to_csv(out, index=False)
    print(f"  Form cache: {len(cache)} teams -> {out}")


# ---------------------------------------------------------------------------
# EPL-only (E0) retrain -- 2026-27 pre-season audit, Phase 3
#
# Does NOT write result_model.pkl / goals_model.pkl / corners_model.pkl or
# any other file. Returns trained models + all metrics for Phase 4 review;
# saving to the deployed filenames only happens after explicit Phase 5
# approval, elsewhere.
# ---------------------------------------------------------------------------

HOLDOUT_START = pd.Timestamp("2025-07-01")   # 2025-26 season boundary
RECENCY_CUTOFF = pd.Timestamp("2010-07-01")  # "2010-11 onward" variant


def train_epl_only():
    from sklearn.metrics import log_loss, precision_recall_fscore_support

    print("\n" + "=" * 62)
    print("  EPL-ONLY (E0) RETRAIN -- 2026-27 pre-season")
    print("=" * 62)

    df = build_epl_frame(DATA_DIR)
    assert (df["league"] == "E0").all(), "non-E0 rows leaked into the EPL-only frame"
    print(f"\n[1] Training frame: E0 ONLY, {len(df):,} rows, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")

    print(f"\n[2] Rolling features (window={N_ROLLING}), single-league so "
          f"league_enc is constant by design")
    df = compute_rolling_stats(df, N_ROLLING)
    le_league = LabelEncoder().fit(df["league"])
    df["league_enc"] = le_league.transform(df["league"])
    league_map = {"E0": 0}

    df = df.sort_values("date").reset_index(drop=True)
    holdout_mask = df["date"] >= HOLDOUT_START
    print(f"\n[3] Chronological split: train < {HOLDOUT_START.date()}, "
          f"holdout = 2025-26 season ({holdout_mask.sum()} raw rows before feature-dropna)")

    print("\n[4] Hyperparameter search (UNCHANGED from train.py's existing "
          "_tune_and_fit): XGBClassifier, RandomizedSearchCV, "
          f"{TUNE_ITER} iterations, cv=3, scoring=accuracy, random_state=42.")
    print("    Search space:", {
        "n_estimators": [300, 500, 700, 1000, 1500], "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.10], "subsample": [0.60, 0.70, 0.80, 0.90],
        "colsample_bytree": [0.50, 0.60, 0.70, 0.80], "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.05, 0.10, 0.20, 0.30], "reg_alpha": [0, 0.01, 0.05, 0.10, 0.50],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
    })

    targets = {
        "result": dict(require=RESULT_REQUIRE, feat=BASE_FEATURES, multiclass=True),
        "goals":  dict(require=RESULT_REQUIRE, feat=BASE_FEATURES, multiclass=False),
        "corners": dict(require=CORE_FEATURES + ["h_r_corners", "a_r_corners"],
                         feat=CORNER_FEATURES, multiclass=False),
    }

    df["over_2_5"] = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)
    df["over_9_5"] = ((df["home_corners"] + df["away_corners"]) > 9.5).astype(int)
    y_col = {"result": "result", "goals": "over_2_5", "corners": "over_9_5"}

    le_result = LabelEncoder().fit(["A", "D", "H"])  # fixed order, independent of split

    results = {}
    for name, cfg in targets.items():
        print(f"\n{'=' * 62}\n  MODEL: {name}\n{'=' * 62}")
        d = df.dropna(subset=cfg["require"] + [y_col[name]]).copy()
        tr_full = d[d["date"] < HOLDOUT_START]
        te      = d[d["date"] >= HOLDOUT_START]
        tr_recent = tr_full[tr_full["date"] >= RECENCY_CUTOFF]

        X_te = te[cfg["feat"]].values.astype(float)
        if name == "result":
            y_tr_full = le_result.transform(tr_full["result"])
            y_tr_recent = le_result.transform(tr_recent["result"])
            y_te = le_result.transform(te["result"])
        else:
            y_tr_full = tr_full[y_col[name]].values
            y_tr_recent = tr_recent[y_col[name]].values
            y_te = te[y_col[name]].values

        X_tr_full = tr_full[cfg["feat"]].values.astype(float)
        X_tr_recent = tr_recent[cfg["feat"]].values.astype(float)
        w_tr_full = add_recency_weight(tr_full, HOLDOUT_START).values

        print(f"  Full-history (weighted): train={len(X_tr_full):,}  "
              f"2010-11+ only: train={len(X_tr_recent):,}  holdout={len(X_te):,}")

        print("\n  -- Variant A: full history, age-decay sample weights "
              f"(half-life={RECENCY_HALF_LIFE_YEARS}y) --")
        model_a, _ = _tune_and_fit_weighted(
            X_tr_full, y_tr_full, w_tr_full, X_te, y_te,
            f"{name} [A:full+decay]", multiclass=cfg["multiclass"])
        proba_a = model_a.predict_proba(X_te)
        ll_a = log_loss(y_te, proba_a, labels=list(range(proba_a.shape[1])))

        print("\n  -- Variant B: 2010-11 onward only, unweighted --")
        model_b, _ = _tune_and_fit(
            X_tr_recent, y_tr_recent, X_te, y_te,
            f"{name} [B:2010+]", multiclass=cfg["multiclass"])
        proba_b = model_b.predict_proba(X_te)
        ll_b = log_loss(y_te, proba_b, labels=list(range(proba_b.shape[1])))

        print(f"\n  Holdout log loss:  A(full+decay)={ll_a:.4f}   B(2010+)={ll_b:.4f}")
        if ll_a <= ll_b:
            winner, model, proba = "A:full+decay", model_a, proba_a
        else:
            winner, model, proba = "B:2010+", model_b, proba_b
        print(f"  WINNER: {winner}")

        pred = np.argmax(proba, axis=1) if cfg["multiclass"] else model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        print(f"  Final ({winner}) holdout accuracy: {acc * 100:.2f}%")

        if name == "result":
            prec, rec, f1, support = precision_recall_fscore_support(
                y_te, pred, labels=[0, 1, 2], zero_division=0)
            print("\n  Per-class precision/recall (1X2), classes A/D/H:")
            for cls, p, r, f, s in zip(["A", "D", "H"], prec, rec, f1, support):
                print(f"    {cls}:  precision={p:.3f}  recall={r:.3f}  f1={f:.3f}  n={s}")

        results[name] = {
            "model": model, "winner": winner, "features": cfg["feat"],
            "X_te": X_te, "y_te": y_te, "proba": proba, "pred": pred, "acc": acc,
            "te_df": te, "ll_a": ll_a, "ll_b": ll_b,
            "result_encoder": le_result if name == "result" else None,
            "league_encoder": le_league, "league_map": league_map,
        }

    return results, df


if __name__ == "__main__":
    main()

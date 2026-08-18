"""
Premier League prediction training script (v1).

Data sources (merged/deduped):
  - data/Matches.csv (Division == 'E0')   : primary — 2000-08-19 to 2026-05-24,
    9,790 EPL matches with pre-computed ELO, Form5, betting odds, corners, SOT.
    This is the SAME underlying feed as data/epl_final.csv and covers a superset
    of its date range (epl_final.csv stops 2025-05-05; Matches.csv E0 continues
    through the 2025/26 season). epl_final.csv is used only as a cross-check
    (goal/corner counts on shared dates) — it adds no new rows once Matches.csv
    is loaded.
  - data/premier-league-matches.csv : supplement — 1992-08-15 to 2000-05-14
    (results only, no corners/odds/elo). Extends rolling-form history earlier
    than Matches.csv's E0 coverage. Team names normalized via TEAM_ALIAS.
  - data/EPLStandings.csv : final league position only, ends at the 2020 season
    and isn't recorded per-matchday, so it can't be used as a leak-free
    pre-match feature (and is stale for the 2021-2026 seasons we most need).
    Skipped as a feature; not needed for dedup since Matches.csv/epl_final.csv
    already key on date+teams.
  - data/EloRatings.csv : cross-checked against Matches.csv's HomeElo/AwayElo
    (same club-elo.com-style source) for sanity, but not merged — Matches.csv
    already supplies leakage-safe pre-match ELO for ~96% of rows.

Feature set mirrors train.py's BASE_FEATURES (the model with the strongest
backtested accuracy site-wide, driven mainly by the odds/ELO features) plus:
  - h2h_home_win_rate  : head-to-head history, mirrored from nba_train.py's
    _h2h_rate() helper (WNBA/NBA models were the only ones using H2H).
  - h_r10_pts / a_r10_pts : a longer 10-game form window alongside the
    existing 5-game rolling window, per the "last 5-10 games" requirement.

Usage:
    python train_epl.py
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

warnings.filterwarnings("ignore")

DATA_DIR   = "data"
N_ROLLING  = 5     # short rolling-window size (games) — matches train.py
N_ROLLING2 = 10    # longer rolling-window size (games)
TUNE_ITER  = 40    # RandomizedSearchCV iterations

TEST_SEASON_START = "2025-08-01"   # held-out backtest: 2025/26 season
TEST_SEASON_END   = "2026-07-01"

TEAM_ALIAS = {
    "Barnsley": "Barnsley",
    "Bradford City": "Bradford",
    "Charlton Ath": "Charlton",
    "Coventry City": "Coventry",
    "Derby County": "Derby",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Manchester City": "Man City",
    "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle",
    "Norwich City": "Norwich",
    "Nott'ham Forest": "Nott'm Forest",
    "Oldham Athletic": "Oldham Athletic",
    "Sheffield Utd": "Sheffield United",
    "Sheffield Weds": "Sheffield Weds",
    "Swindon Town": "Swindon Town",
    "Wimbledon": "Wimbledon",
}

# ---------------------------------------------------------------------------
# Feature lists (order is significant — predict_epl.py must match exactly)
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    "h_r_gf", "h_r_ga", "h_r_gd",
    "h_r_sot",
    "h_r_win", "h_r_draw", "h_r_pts",
    "h_r_home_win",
    "h_r10_pts",
    "a_r_gf", "a_r_ga", "a_r_gd",
    "a_r_sot",
    "a_r_win", "a_r_draw", "a_r_pts",
    "a_r_away_win",
    "a_r10_pts",
    "elo_diff", "elo_prob_h",
    "form5_h", "form5_a", "form5_diff",
    "imp_h", "imp_d", "imp_a", "book_margin",
    "h2h_home_win_rate",
]
CORNER_FEATURES = BASE_FEATURES + ["h_r_corners", "a_r_corners"]

CORE_FEATURES  = ["h_r_gf", "h_r_ga", "a_r_gf", "a_r_ga"]
RESULT_REQUIRE = CORE_FEATURES + ["imp_h", "imp_d", "imp_a"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _elo_win_prob(elo_diff: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def _h2h_rate(df, home_col, away_col, home_win_col, date_col):
    """Cumulative home-team win rate in all prior meetings between this pair."""
    tmp = df[[date_col, home_col, away_col, home_win_col]].copy()
    tmp["_pair"]   = tmp.apply(
        lambda r: tuple(sorted([r[home_col], r[away_col]])), axis=1
    )
    tmp["_h_is_a"] = tmp[home_col] == tmp["_pair"].apply(lambda p: p[0])
    tmp["_a_win"]  = np.where(tmp["_h_is_a"], tmp[home_win_col], 1 - tmp[home_win_col])
    tmp = tmp.sort_values(["_pair", date_col]).reset_index(drop=True)
    g = tmp.groupby("_pair")
    tmp["_cnt"]     = g.cumcount()
    tmp["_cum_a_w"] = g["_a_win"].transform(lambda x: x.shift(1).fillna(0).cumsum())
    tmp["_h2h_a"]   = np.where(
        tmp["_cnt"] > 0, tmp["_cum_a_w"] / tmp["_cnt"].clip(lower=1), 0.5
    )
    tmp["_h2h_home"] = np.where(tmp["_h_is_a"], tmp["_h2h_a"], 1 - tmp["_h2h_a"])
    tmp.index = df.index
    return tmp["_h2h_home"].values


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_primary(data_dir: str) -> pd.DataFrame:
    """Matches.csv, Division == 'E0' (Premier League)."""
    path = os.path.join(data_dir, "Matches.csv")
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["Division"] == "E0"].copy()
    raw = raw[raw["FTResult"].isin(["H", "D", "A"])].copy()

    hg, ag = pd.to_numeric(raw["FTHome"], errors="coerce"), pd.to_numeric(raw["FTAway"], errors="coerce")
    h_elo, a_elo = _col(raw, "HomeElo"), _col(raw, "AwayElo")
    f5h, f5a = _col(raw, "Form5Home"), _col(raw, "Form5Away")

    oh = _col(raw, "OddHome").where(lambda x: x > 0)
    od = _col(raw, "OddDraw").where(lambda x: x > 0)
    oa = _col(raw, "OddAway").where(lambda x: x > 0)
    raw_h, raw_d, raw_a = 1.0 / oh, 1.0 / od, 1.0 / oa
    raw_sum = raw_h + raw_d + raw_a
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
        "elo_diff":     elo_diff,
        "elo_prob_h":   _elo_win_prob(elo_diff),
        "form5_h":      f5h,
        "form5_a":      f5a,
        "form5_diff":   f5h - f5a,
        "imp_h":        raw_h / raw_sum,
        "imp_d":        raw_d / raw_sum,
        "imp_a":        raw_a / raw_sum,
        "book_margin":  raw_sum - 1.0,
        "source":       "matches_e0",
    })
    df.dropna(subset=["home_goals", "away_goals", "date"], inplace=True)
    print(f"  Matches.csv (E0):        {len(df):,} matches  "
          f"({df['date'].min().date()} - {df['date'].max().date()})  "
          f"odds: {df['imp_h'].notna().sum():,}  elo: {df['elo_diff'].notna().sum():,}")
    return df


def load_early_supplement(data_dir: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """premier-league-matches.csv, rows strictly before `cutoff` (no overlap with Matches.csv)."""
    path = os.path.join(data_dir, "premier-league-matches.csv")
    raw = pd.read_csv(path, parse_dates=["Date"])
    raw = raw[raw["Date"] < cutoff].copy()
    raw["Home"] = raw["Home"].map(lambda t: TEAM_ALIAS.get(t, t))
    raw["Away"] = raw["Away"].map(lambda t: TEAM_ALIAS.get(t, t))

    df = pd.DataFrame({
        "home_team":    raw["Home"].astype(str).str.strip(),
        "away_team":    raw["Away"].astype(str).str.strip(),
        "home_goals":   pd.to_numeric(raw["HomeGoals"], errors="coerce"),
        "away_goals":   pd.to_numeric(raw["AwayGoals"], errors="coerce"),
        "result":       raw["FTR"].astype(str).str.strip(),
        "home_sot":     np.nan,
        "away_sot":     np.nan,
        "home_corners": np.nan,
        "away_corners": np.nan,
        "date":         raw["Date"],
        "elo_diff":     np.nan,
        "elo_prob_h":   np.nan,
        "form5_h":      np.nan,
        "form5_a":      np.nan,
        "form5_diff":   np.nan,
        "imp_h":        np.nan,
        "imp_d":        np.nan,
        "imp_a":        np.nan,
        "book_margin":  np.nan,
        "source":       "plm_early",
    })
    df = df[df["result"].isin(["H", "D", "A"])]
    df.dropna(subset=["home_goals", "away_goals", "date"], inplace=True)
    print(f"  premier-league-matches (pre-{cutoff.date()}): {len(df):,} matches  "
          f"({df['date'].min().date()} - {df['date'].max().date()})  [results only, no corners/odds/elo]")
    return df


def cross_check_epl_final(data_dir: str, primary: pd.DataFrame) -> None:
    """Sanity-check Matches.csv(E0) against epl_final.csv on shared dates+teams."""
    path = os.path.join(data_dir, "epl_final.csv")
    if not os.path.exists(path):
        print("  epl_final.csv not found — skipping cross-check")
        return
    ef = pd.read_csv(path, parse_dates=["MatchDate"])
    merged = ef.merge(
        primary, left_on=["MatchDate", "HomeTeam", "AwayTeam"],
        right_on=["date", "home_team", "away_team"], how="inner",
    )
    goal_mismatch = (
        (merged["FullTimeHomeGoals"] != merged["home_goals"]) |
        (merged["FullTimeAwayGoals"] != merged["away_goals"])
    ).sum()
    corner_mismatch = (
        (merged["HomeCorners_x"] != merged["home_corners"]) |
        (merged["AwayCorners_x"] != merged["away_corners"])
    ).sum() if "HomeCorners_x" in merged.columns else "n/a"
    print(f"  Cross-check vs epl_final.csv: {len(merged):,}/{len(ef):,} rows matched by "
          f"date+teams; goal mismatches: {goal_mismatch}; corner mismatches: {corner_mismatch}")
    unmatched = len(ef) - len(merged)
    if unmatched:
        print(f"  ({unmatched} epl_final rows have no exact date+team match in Matches.csv — "
              f"likely postponed/rescheduled fixture dates; not added, since Matches.csv is "
              f"already the richer/more current source)")


def cross_check_elo_ratings(data_dir: str, primary: pd.DataFrame) -> None:
    """Sanity-check Matches.csv's HomeElo against EloRatings.csv on a sample."""
    path = os.path.join(data_dir, "EloRatings.csv")
    if not os.path.exists(path):
        print("  EloRatings.csv not found — skipping cross-check")
        return
    elo = pd.read_csv(path)
    elo.columns = [c.strip() for c in elo.columns]
    elo["date"] = pd.to_datetime(elo["date"])
    sample = primary.dropna(subset=["elo_diff"]).sample(min(500, len(primary)), random_state=42)
    diffs = []
    for _, row in sample.iterrows():
        snap = elo[(elo["club"] == row["home_team"]) & (elo["date"] <= row["date"])]
        if len(snap):
            diffs.append(abs(snap.sort_values("date").iloc[-1]["elo"] - (row["elo_diff"] + 1500)))
    if diffs:
        print(f"  Cross-check vs EloRatings.csv: {len(diffs)}/{len(sample)} sampled home teams "
              f"found; median |diff| vs Matches.csv-implied rating: {np.median(diffs):.0f} pts "
              f"(same underlying rating system — Matches.csv ELO used directly, not remerged)")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    home_v = pd.DataFrame({
        "date": df["date"], "team": df["home_team"],
        "gf": df["home_goals"], "ga": df["away_goals"],
        "sot": df["home_sot"], "corners": df["home_corners"],
        "win":  df["result"].map({"H": 1.0, "D": 0.0, "A": 0.0}),
        "draw": df["result"].map({"H": 0.0, "D": 1.0, "A": 0.0}),
        "pts":  df["result"].map({"H": 3.0, "D": 1.0, "A": 0.0}),
        "match_idx": df.index, "side": "home",
    })
    away_v = pd.DataFrame({
        "date": df["date"], "team": df["away_team"],
        "gf": df["away_goals"], "ga": df["home_goals"],
        "sot": df["away_sot"], "corners": df["away_corners"],
        "win":  df["result"].map({"H": 0.0, "D": 0.0, "A": 1.0}),
        "draw": df["result"].map({"H": 0.0, "D": 1.0, "A": 0.0}),
        "pts":  df["result"].map({"H": 0.0, "D": 1.0, "A": 3.0}),
        "match_idx": df.index, "side": "away",
    })

    all_g = (pd.concat([home_v, away_v], ignore_index=True)
               .sort_values(["team", "date"]).reset_index(drop=True))

    for col in ("gf", "ga", "sot", "win", "draw", "pts", "corners"):
        all_g[f"r_{col}"] = (
            all_g.groupby("team", sort=False)[col]
            .transform(lambda x: x.shift(1).rolling(N_ROLLING, min_periods=1).mean())
        )
    all_g["r_gd"] = all_g["r_gf"] - all_g["r_ga"]

    all_g["r10_pts"] = (
        all_g.groupby("team", sort=False)["pts"]
        .transform(lambda x: x.shift(1).rolling(N_ROLLING2, min_periods=1).mean())
    )

    home_only = all_g[all_g["side"] == "home"].sort_values(["team", "date"])
    home_win_r = (home_only.groupby("team", sort=False)["win"]
                  .transform(lambda x: x.shift(1).rolling(N_ROLLING, min_periods=1).mean()))
    all_g["r_home_win"] = np.nan
    all_g.loc[home_only.index, "r_home_win"] = home_win_r

    away_only = all_g[all_g["side"] == "away"].sort_values(["team", "date"])
    away_win_r = (away_only.groupby("team", sort=False)["win"]
                  .transform(lambda x: x.shift(1).rolling(N_ROLLING, min_periods=1).mean()))
    all_g["r_away_win"] = np.nan
    all_g.loc[away_only.index, "r_away_win"] = away_win_r

    roll_base = ["match_idx", "r_gf", "r_ga", "r_gd", "r_sot", "r_win", "r_draw", "r_pts",
                 "r_corners", "r10_pts"]

    h_roll = all_g[all_g["side"] == "home"][roll_base + ["r_home_win"]].set_index("match_idx")
    a_roll = all_g[all_g["side"] == "away"][roll_base + ["r_away_win"]].set_index("match_idx")

    df = df.join(h_roll.add_prefix("h_"))
    df = df.join(a_roll.add_prefix("a_"))

    df["home_win_flag"] = (df["result"] == "H").astype(float)
    df["h2h_home_win_rate"] = _h2h_rate(df, "home_team", "away_team", "home_win_flag", "date")
    df.drop(columns=["home_win_flag"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _tune_and_fit(X_tr, y_tr, X_te, y_te, label, multiclass=False):
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    X_te = np.where(np.isinf(X_te), np.nan, X_te)

    param_dist = {
        "n_estimators":     [200, 300, 500, 700, 1000],
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
    base_clf = XGBClassifier(eval_metric=eval_metric, random_state=42, verbosity=0, tree_method="hist")

    sample_size = min(len(X_tr), 40_000)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_tr), sample_size, replace=False)

    print(f"  Tuning {label}: {sample_size:,} samples ({TUNE_ITER} iters x 3-fold CV)")
    search = RandomizedSearchCV(base_clf, param_dist, n_iter=TUNE_ITER, cv=3,
                                 scoring="accuracy", random_state=42, n_jobs=-1, verbose=0)
    search.fit(X_tr[idx], y_tr[idx])
    best_p = search.best_params_
    print(f"  CV best: {search.best_score_ * 100:.1f}%  "
          f"(depth={best_p.get('max_depth')}, lr={best_p.get('learning_rate')}, n={best_p.get('n_estimators')})")

    clf = XGBClassifier(**best_p, eval_metric=eval_metric, random_state=42, verbosity=0, tree_method="hist")
    clf.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  {label:<28}  test accuracy: {acc * 100:.1f}%  ({len(X_tr)+len(X_te):,} samples)")
    return clf, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 62)
    print("  PREMIER LEAGUE PREDICTION MODEL TRAINING  (v1)")
    print("=" * 62)

    print("\n[1] Loading & merging datasets")
    primary = load_primary(DATA_DIR)
    cross_check_epl_final(DATA_DIR, primary)
    cross_check_elo_ratings(DATA_DIR, primary)
    early = load_early_supplement(DATA_DIR, primary["date"].min())

    df = pd.concat([early, primary], ignore_index=True)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    print(f"  Combined total: {len(df):,} matches  "
          f"({df['date'].min().date()} - {df['date'].max().date()})")

    print(f"\n[2] Engineering rolling features  (windows = {N_ROLLING}/{N_ROLLING2} games, + H2H)")
    df = compute_rolling_stats(df)

    split_date = pd.Timestamp(TEST_SEASON_START)
    test_end   = pd.Timestamp(TEST_SEASON_END)
    print(f"\n[3] Held-out backtest: last full season "
          f"({split_date.date()} - {(test_end - pd.Timedelta(days=1)).date()})")
    tr_mask = df["date"] < split_date
    te_mask = (df["date"] >= split_date) & (df["date"] < test_end)
    print(f"  Train: {tr_mask.sum():,}  Test (2025/26 season): {te_mask.sum():,}")

    # --- Model 1: Match result ---
    print("\n[4] Match Result model  (H / D / A)")
    le_result = LabelEncoder()
    df_r = df[df["result"].isin(["H", "D", "A"])].dropna(subset=RESULT_REQUIRE)
    tr_r = df_r[df_r["date"] < split_date]
    te_r = df_r[(df_r["date"] >= split_date) & (df_r["date"] < test_end)]
    le_result.fit(df_r["result"])
    X_tr_r = tr_r[BASE_FEATURES].values.astype(float)
    y_tr_r = le_result.transform(tr_r["result"])
    X_te_r = te_r[BASE_FEATURES].values.astype(float)
    y_te_r = le_result.transform(te_r["result"])
    print(f"  Train: {len(X_tr_r):,}  Test: {len(X_te_r):,}")

    naive_pred = np.where(
        (te_r["imp_h"] >= te_r["imp_d"]) & (te_r["imp_h"] >= te_r["imp_a"]), "H",
        np.where(te_r["imp_d"] >= te_r["imp_a"], "D", "A")
    )
    naive_acc = (naive_pred == te_r["result"].values).mean()
    print(f"  Naive implied-prob baseline (test): {naive_acc * 100:.1f}%")

    model_result, new_acc_r = _tune_and_fit(
        X_tr_r, y_tr_r, X_te_r, y_te_r, "Match result (H/D/A)", multiclass=True)

    # --- Model 2: Goals O/U 2.5 ---
    print("\n[5] Goals Over/Under 2.5 model")
    df_g = df.copy()
    df_g["over_2_5"] = ((df_g["home_goals"] + df_g["away_goals"]) > 2.5).astype(int)
    df_g = df_g.dropna(subset=RESULT_REQUIRE + ["over_2_5"])
    tr_g = df_g[df_g["date"] < split_date]
    te_g = df_g[(df_g["date"] >= split_date) & (df_g["date"] < test_end)]
    X_tr_g = tr_g[BASE_FEATURES].values.astype(float)
    y_tr_g = tr_g["over_2_5"].values
    X_te_g = te_g[BASE_FEATURES].values.astype(float)
    y_te_g = te_g["over_2_5"].values
    print(f"  Train: {len(X_tr_g):,}  Test: {len(X_te_g):,}")

    naive_acc_g = max(y_te_g.mean(), 1 - y_te_g.mean())
    print(f"  Naive majority-class baseline (test): {naive_acc_g * 100:.1f}%")

    model_goals, new_acc_g = _tune_and_fit(X_tr_g, y_tr_g, X_te_g, y_te_g, "Goals O/U 2.5")

    # --- Model 3: Corners O/U 9.5 ---
    print("\n[6] Corners Over/Under 9.5 model")
    df_c = df.copy()
    df_c["total_corners"] = df_c["home_corners"] + df_c["away_corners"]
    df_c["over_9_5"] = (df_c["total_corners"] > 9.5).astype(int)
    df_c = df_c.dropna(subset=CORE_FEATURES + ["h_r_corners", "a_r_corners", "over_9_5"])
    tr_c = df_c[df_c["date"] < split_date]
    te_c = df_c[(df_c["date"] >= split_date) & (df_c["date"] < test_end)]
    print(f"  Train: {len(tr_c):,}  Test: {len(te_c):,}")
    X_tr_c = tr_c[CORNER_FEATURES].values.astype(float)
    y_tr_c = tr_c["over_9_5"].values
    X_te_c = te_c[CORNER_FEATURES].values.astype(float)
    y_te_c = te_c["over_9_5"].values

    naive_acc_c = max(y_te_c.mean(), 1 - y_te_c.mean())
    print(f"  Naive majority-class baseline (test): {naive_acc_c * 100:.1f}%")

    model_corners, new_acc_c = _tune_and_fit(X_tr_c, y_tr_c, X_te_c, y_te_c, "Corners O/U 9.5")

    # --- Backtest summary ---
    print("\n" + "=" * 62)
    print("  BACKTEST SUMMARY  (held-out: 2025/26 Premier League season)")
    print("=" * 62)
    print(f"  Result  (H/D/A)  naive={naive_acc * 100:.1f}%   new={new_acc_r * 100:.1f}%"
          f"  vs-naive={new_acc_r - naive_acc:+.3f}")
    print(f"  Goals   O/U 2.5  naive={naive_acc_g * 100:.1f}%   new={new_acc_g * 100:.1f}%"
          f"  vs-naive={new_acc_g - naive_acc_g:+.3f}")
    print(f"  Corners O/U 9.5  naive={naive_acc_c * 100:.1f}%   new={new_acc_c * 100:.1f}%"
          f"  vs-naive={new_acc_c - naive_acc_c:+.3f}")
    site_avg = 0.638
    print(f"\n  Site average (informational target): {site_avg*100:.1f}%")
    print(f"  Result vs site avg: {new_acc_r*100:.1f}%  ({'ABOVE' if new_acc_r>=site_avg else 'BELOW'})")

    deploy = (new_acc_r >= naive_acc) and (new_acc_g >= naive_acc_g)

    if deploy:
        print("\n  [DEPLOY] New models beat naive baseline -- saving pkl files")
        with open("epl_result_model.pkl", "wb") as f:
            pickle.dump({"model": model_result, "result_encoder": le_result,
                         "features": BASE_FEATURES, "test_accuracy": new_acc_r}, f)
        with open("epl_goals_model.pkl", "wb") as f:
            pickle.dump({"model": model_goals, "features": BASE_FEATURES,
                         "test_accuracy": new_acc_g}, f)
        with open("epl_corners_model.pkl", "wb") as f:
            pickle.dump({"model": model_corners, "features": CORNER_FEATURES,
                         "test_accuracy": new_acc_c}, f)
        print("  Saved: epl_result_model.pkl  epl_goals_model.pkl  epl_corners_model.pkl")
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
    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    active_teams = set(
        df.loc[df["date"] >= cutoff, "home_team"].tolist() +
        df.loc[df["date"] >= cutoff, "away_team"].tolist()
    )
    rows = []
    for team in sorted(active_teams):
        mask = (df["home_team"] == team) | (df["away_team"] == team)
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
        rows.append({
            "team": team, "league": "Premier League",
            "gf": round(gf / nm, 3), "ga": round(ga / nm, 3),
            "sot": round(sot_s / sot_n if sot_n else 4.5, 3),
            "corners": round(cor_s / cor_n if cor_n else 5.0, 3),
            "win": round(wins / nm, 3), "draw": round(draws / nm, 3),
            "hwn": round(hwin / hg if hg else min(wins / nm + 0.08, 1.0), 3),
            "awn": round(awin / ag_ if ag_ else max(wins / nm - 0.08, 0.0), 3),
            "last_date": recent["date"].max().strftime("%Y-%m-%d"), "n_games": nm,
        })
    cache = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, "epl_team_form_cache.csv")
    cache.to_csv(out, index=False)
    print(f"  Form cache: {len(cache)} teams -> {out}")


if __name__ == "__main__":
    main()

"""
compute_bestbet_baselines.py — empirical per-market confidence baselines
for Best Bet normalisation.

_compute_best_bet (app.py) used to pick whichever market stated the
highest RAW probability across Result (3-way) / Goals / BTTS / Corners
(all binary). That's not a fair comparison: Result's theoretical floor is
1/3 vs 1/2 for the binary markets, and empirically Corners rarely drops
far below ~52% while Result/Goals routinely sit near a genuine coin-flip
on close matchups -- so Corners won Best Bet 44% of the time on live
traffic, Result 32%, Goals 24%, BTTS 0%, independent of which market
actually carried the most genuine edge on a given fixture (2026-08-22
audit).

Fix: instead of raw probability, use a z-score of each market's stated
confidence relative to that market's OWN typical confidence distribution
-- computed once here across the full chronological test split (the same
last-20%-by-date split train.py already uses), so the numbers are stable
and not skewed by any single week's fixture list. Pick the market whose
current confidence is the most unusual relative to its own baseline, not
whichever raw number happens to be biggest.

Writes bestbet_baselines.json: {"result": [mean, std], "goals": [...],
"btts": [...], "corners": [...]}, each computed as max-class probability
(three-way max for result, max(p, 1-p) for the binary markets).
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import train as T


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


print("Loading data...")
df_main = T.load_matches(T.DATA_DIR)
df_fbref = T.load_fbref_supplements(T.DATA_DIR)
sp1_min = df_main.loc[df_main["league"] == "SP1", "date"].min()
df_laliga = T.load_laliga_early(T.DATA_DIR, sp1_min) if pd.notna(sp1_min) else pd.DataFrame()
extra = [d for d in (df_fbref, df_laliga) if len(d) > 0]
df = pd.concat([df_main] + extra, ignore_index=True) if extra else df_main
df = T.compute_rolling_stats(df, T.N_ROLLING)

le_league = LabelEncoder().fit(df["league"])
df["league_enc"] = le_league.transform(df["league"])

df["over_2_5"] = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)
df["over_9_5"] = ((df["home_corners"] + df["away_corners"]) > 9.5).astype(int)
df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

# Same chronological 80/20 split train.py's main() uses.
df = df.sort_values("date").reset_index(drop=True)
split_idx = int(len(df) * 0.80)
split_date = df.iloc[split_idx]["date"]
test = df[df["date"] >= split_date]
print(f"Test split: {len(test):,} rows from {split_date.date()} onward")

club_result = _load("result_model.pkl")
club_goals = _load("goals_model.pkl")
club_corners = _load("corners_model.pkl")
club_btts = _load("btts_model.pkl")

baselines = {}

d = test.dropna(subset=T.RESULT_REQUIRE)
X = d[T.BASE_FEATURES].values.astype(float)
proba = club_result["model"].predict_proba(X)
conf = proba.max(axis=1)
baselines["result"] = [float(conf.mean()), float(conf.std())]
print(f"result:  n={len(conf):6,}  mean={conf.mean():.4f}  std={conf.std():.4f}")

for name, model, feat, require in [
    ("goals", club_goals, T.BASE_FEATURES, T.RESULT_REQUIRE),
    ("btts", club_btts, T.BASE_FEATURES, T.RESULT_REQUIRE),
    ("corners", club_corners, T.CORNER_FEATURES, T.CORE_FEATURES + ["h_r_corners", "a_r_corners"]),
]:
    d = test.dropna(subset=require)
    X = d[feat].values.astype(float)
    proba = model["model"].predict_proba(X)
    p1 = proba[:, list(model["model"].classes_).index(1)]
    conf = np.maximum(p1, 1 - p1)
    baselines[name] = [float(conf.mean()), float(conf.std())]
    print(f"{name:8s} n={len(conf):6,}  mean={conf.mean():.4f}  std={conf.std():.4f}")

with open("bestbet_baselines.json", "w") as f:
    json.dump(baselines, f, indent=2)
print("\nSaved bestbet_baselines.json")
print(json.dumps(baselines, indent=2))

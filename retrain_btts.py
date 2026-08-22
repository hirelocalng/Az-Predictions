"""
retrain_btts.py — train a real BTTS (both teams to score) classifier.

Root cause audit (2026-08-22): BTTS had no trained model at all. It was a
Poisson-independence formula (fetch_fixtures._btts_prob) fed ONLY each
team's rolling goals-scored average -- no elo, no form, nothing else. On
production this formula received the same constant DEFAULT_STATS gf (1.2)
for literally every fixture in every league, because data/Matches.csv
(and the BRA/ARG supplement CSVs) are gitignored and never reach Railway,
so _load_history() always returned empty and _rolling()'s _CLUB_FORM_CACHE
fallback was unreachable -- confirmed via a 20-fixture audit: BTTS
std=0.00 (hard constant 48.8%) on production-simulated features vs
std=10.16 (range 24.9-74.2%) with real per-team data. That data-starvation
bug is fixed separately in fetch_fixtures.py (_team_stats helper). This
script addresses the other half: there was never a trained BTTS model to
feed in the first place.

Trains BTTS on the same odds-free BASE_FEATURES used for Result (train.py),
full multi-league frame, chronological split (train < 2025-07-01), holdout
= 2025-26 EPL season -- identical methodology to retrain_no_odds.py.

Target: FullTimeHomeGoals > 0 AND FullTimeAwayGoals > 0 (both teams scored).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder

import train as T

HOLDOUT_START = pd.Timestamp("2025-07-01")

print("=" * 70)
print("LOADING DATA")
print("=" * 70)
df_main = T.load_matches(T.DATA_DIR)
df_fbref = T.load_fbref_supplements(T.DATA_DIR)
sp1_min = df_main.loc[df_main["league"] == "SP1", "date"].min()
df_laliga = T.load_laliga_early(T.DATA_DIR, sp1_min) if pd.notna(sp1_min) else pd.DataFrame()
extra = [d for d in (df_fbref, df_laliga) if len(d) > 0]
df = pd.concat([df_main] + extra, ignore_index=True) if extra else df_main
print(f"Combined multi-league total: {len(df):,}")

df = T.compute_rolling_stats(df, T.N_ROLLING)
le_league = LabelEncoder().fit(df["league"])
df["league_enc"] = le_league.transform(df["league"])
league_map = dict(zip(le_league.classes_.tolist(), le_league.transform(le_league.classes_).tolist()))

df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

print("\n" + "=" * 70)
print("TRAIN: btts")
print("=" * 70)
d = df.dropna(subset=T.RESULT_REQUIRE + ["btts"]).copy()
tr = d[d["date"] < HOLDOUT_START]
te = d[(d["date"] >= HOLDOUT_START) & (d["league"] == "E0")]
print(f"Train (all leagues, < 2025-07-01): {len(tr):,}   Holdout (E0 2025-26 only): {len(te):,}")
print(f"Base rate: train {tr['btts'].mean()*100:.1f}%  holdout {te['btts'].mean()*100:.1f}%")

X_tr = tr[T.BASE_FEATURES].values.astype(float)
y_tr = tr["btts"].values
X_te = te[T.BASE_FEATURES].values.astype(float)
y_te = te["btts"].values

model, _ = T._tune_and_fit(X_tr, y_tr, X_te, y_te, "btts", multiclass=False)

# ── Evaluate new model on holdout ──────────────────────────────────────────
proba_new = model.predict_proba(X_te)
classes = list(model.classes_)
p1_new = proba_new[:, classes.index(1)] if 1 in classes else proba_new[:, -1]
pred_new = (p1_new >= 0.5).astype(int)
acc_new = accuracy_score(y_te, pred_new)
ll_new = log_loss(y_te, p1_new, labels=[0, 1])
brier_new = brier_score_loss(y_te, p1_new)

# ── Evaluate OLD formula (fed REAL, properly-resolved rolling gf -- i.e.
#    what BTTS will look like now that the feature-starvation bug is
#    fixed, not the broken constant-1.2 version) ───────────────────────────
import math
def btts_formula(hg, ag):
    p_h = 1.0 - math.exp(-max(0.05, hg))
    p_a = 1.0 - math.exp(-max(0.05, ag))
    return p_h * p_a

p1_old = te.apply(lambda r: btts_formula(r["h_r_gf"], r["a_r_gf"]), axis=1).values
pred_old = (p1_old >= 0.5).astype(int)
acc_old = accuracy_score(y_te, pred_old)
ll_old = log_loss(y_te, p1_old, labels=[0, 1])
brier_old = brier_score_loss(y_te, p1_old)

print("\n" + "=" * 70)
print("VERIFY: BTTS -- old Poisson-gf formula (real features) vs new trained model")
print("=" * 70)
print(f"{'':28s} {'acc':>8s} {'logloss':>10s} {'brier':>9s}")
print(f"{'(a) OLD formula':28s} {acc_old*100:7.2f}% {ll_old:10.4f} {brier_old:9.4f}")
print(f"{'(b) NEW trained model':28s} {acc_new*100:7.2f}% {ll_new:10.4f} {brier_new:9.4f}")
deploy_ok = (ll_new < ll_old) and (brier_new < brier_old)
print(f"\nDEPLOY DECISION: {'PROCEED -- new BTTS model beats the old formula' if deploy_ok else 'DO NOT DEPLOY'}")

print("\n" + "=" * 70)
print("CALIBRATION TABLE -- new BTTS model, 2025-26 EPL holdout")
print("=" * 70)
buckets = [(0.0,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.01)]
print(f"{'bucket':>14s}  {'n':>5s}  {'avg stated p(yes)':>18s}  {'actual BTTS rate':>18s}")
for lo, hi in buckets:
    mask = (p1_new >= lo) & (p1_new < hi)
    n = int(mask.sum())
    if n == 0:
        print(f"  [{lo:.2f},{hi:.2f})  {n:>5d}  {'--':>18s}  {'--':>18s}")
        continue
    print(f"  [{lo:.2f},{hi:.2f})  {n:>5d}  {p1_new[mask].mean()*100:17.1f}%  {y_te[mask].mean()*100:17.1f}%")

if deploy_ok:
    with open("btts_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "league_encoder": le_league,
            "league_map": league_map,
            "features": T.BASE_FEATURES,
            "odds_free": True,
        }, f)
    print("\nSaved btts_model.pkl")

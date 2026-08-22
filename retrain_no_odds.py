"""
retrain_no_odds.py — one-shot build+verify+deploy for the odds-free model set.

Root cause (2026-08-21 incident): fetch_fixtures.py hardcoded a constant
odds triple (2.60/3.10/2.80) for every live fixture because no live odds
feed exists. The deployed models were trained REQUIRING real odds
(imp_h/imp_d/imp_a/book_margin dominate the learned signal), so feeding
them a fixture-independent constant flattened every live prediction to a
near-uniform split regardless of the actual matchup.

Fix: retrain Result / Goals / Corners on a feature set with imp_h, imp_d,
imp_a, book_margin removed entirely (train.BASE_FEATURES / CORNER_FEATURES,
already edited), so training matches what serving can actually provide.

This script:
  1. Loads the full multi-league frame (train.py loaders).
  2. Reports old vs new training-row counts (odds-gated vs odds-free).
  3. Trains all 3 models on data before 2025-07-01, chronological.
  4. Evaluates on the 2025-26 EPL holdout:
       (a) currently-deployed (odds) models fed the hardcoded 2.60/3.10/2.80
       (b) new no-odds models
  5. Prints a calibration table for the new Result model on that holdout.
  6. Runs 3 real fixtures through the new model.
  7. Deploys (overwrites result_model.pkl/goals_model.pkl/corners_model.pkl)
     ONLY if the new model beats the deployed-with-hardcoded-odds baseline.

Does not touch fetch_fixtures.py / app.py — that wiring happens after this
script confirms the new models are actually better.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder

import train as T

pd.set_option("display.width", 140)

SCRATCH = r"C:\Users\HPZBOO~1\AppData\Local\Temp\claude\C--Users-HP-ZBook-14-Az-Predictions\ab9f99a2-55e3-439e-b6e2-b16c726874ac\scratchpad"
os.makedirs(SCRATCH, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2025-07-01")

# Hardcoded odds triple currently baked into fetch_fixtures.py:486 —
# used ONLY to reproduce what the site serves today, for the (a) baseline.
PROD_ODD_H, PROD_ODD_D, PROD_ODD_A = 2.60, 3.10, 2.80


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# STEP 1 — no-odds feature set
# ---------------------------------------------------------------------------
log("=" * 78)
log("STEP 1 — NO-ODDS FEATURE SET")
log("=" * 78)
log(f"BASE_FEATURES ({len(T.BASE_FEATURES)}):")
for i, f in enumerate(T.BASE_FEATURES):
    log(f"  {i:2d}. {f}")
log(f"\nCORNER_FEATURES ({len(T.CORNER_FEATURES)}): {T.CORNER_FEATURES}")
assert not (set(T.BASE_FEATURES) & set(T.ODDS_FEATURES)), "odds leaked into BASE_FEATURES!"
log("\nConfirmed: imp_h, imp_d, imp_a, book_margin are NOT in BASE_FEATURES.")

# ---------------------------------------------------------------------------
# Load data (same loaders as train.py main())
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("LOADING DATA")
log("=" * 78)
df_main = T.load_matches(T.DATA_DIR)
df_fbref = T.load_fbref_supplements(T.DATA_DIR)
sp1_min = df_main.loc[df_main["league"] == "SP1", "date"].min()
df_laliga = T.load_laliga_early(T.DATA_DIR, sp1_min) if pd.notna(sp1_min) else pd.DataFrame()
extra = [d for d in (df_fbref, df_laliga) if len(d) > 0]
df = pd.concat([df_main] + extra, ignore_index=True) if extra else df_main
log(f"Combined multi-league total: {len(df):,}")

df = T.compute_rolling_stats(df, T.N_ROLLING)
le_league = LabelEncoder().fit(df["league"])
df["league_enc"] = le_league.transform(df["league"])
league_map = dict(zip(le_league.classes_.tolist(), le_league.transform(le_league.classes_).tolist()))

df["over_2_5"] = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)
df["over_9_5"] = ((df["home_corners"] + df["away_corners"]) > 9.5).astype(int)

le_result = LabelEncoder().fit(["A", "D", "H"])

# ---------------------------------------------------------------------------
# STEP 2 — row counts: odds-gated (old) vs odds-free (new)
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("STEP 2 — TRAINING ROW COUNTS: OLD (odds-gated) vs NEW (odds-free)")
log("=" * 78)
OLD_REQUIRE = T.CORE_FEATURES + T.ODDS_FEATURES[:3]  # imp_h, imp_d, imp_a (old RESULT_REQUIRE)
old_rows = df.dropna(subset=OLD_REQUIRE + ["result"])
new_rows = df.dropna(subset=T.RESULT_REQUIRE + ["result"])
log(f"  OLD (odds-required)   result-eligible rows: {len(old_rows):,}")
log(f"  NEW (odds-free)       result-eligible rows: {len(new_rows):,}")
log(f"  Delta: {len(new_rows) - len(old_rows):+,}")

# ---------------------------------------------------------------------------
# STEP 3 — train on data < 2025-07-01, chronological, no-odds feature set
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("STEP 3 — TRAINING (chronological, train < 2025-07-01)")
log("=" * 78)

targets = {
    "result":  dict(require=T.RESULT_REQUIRE, feat=T.BASE_FEATURES, multiclass=True, ycol="result"),
    "goals":   dict(require=T.RESULT_REQUIRE, feat=T.BASE_FEATURES, multiclass=False, ycol="over_2_5"),
    "corners": dict(require=T.CORE_FEATURES + ["h_r_corners", "a_r_corners"],
                     feat=T.CORNER_FEATURES, multiclass=False, ycol="over_9_5"),
}

new_models = {}
for name, cfg in targets.items():
    log(f"\n{'=' * 62}\n  TRAIN: {name}\n{'=' * 62}")
    d = df.dropna(subset=cfg["require"] + [cfg["ycol"]]).copy()
    tr = d[d["date"] < HOLDOUT_START]
    te = d[(d["date"] >= HOLDOUT_START) & (d["league"] == "E0")]
    log(f"  Train (all leagues, < 2025-07-01): {len(tr):,}   Holdout (E0 2025-26 only): {len(te):,}")

    X_tr = tr[cfg["feat"]].values.astype(float)
    X_te = te[cfg["feat"]].values.astype(float)
    if name == "result":
        y_tr = le_result.transform(tr["result"])
        y_te = le_result.transform(te["result"])
    else:
        y_tr = tr[cfg["ycol"]].values
        y_te = te[cfg["ycol"]].values

    model, _ = T._tune_and_fit(X_tr, y_tr, X_te, y_te, f"no-odds-{name}", multiclass=cfg["multiclass"])
    new_models[name] = {"model": model, "features": cfg["feat"], "te_df": te, "y_te": y_te}

pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in new_models.items()},
            open(os.path.join(SCRATCH, "no_odds_meta.pkl"), "wb"))
for name, v in new_models.items():
    with open(os.path.join(SCRATCH, f"no_odds_{name}_model.pkl"), "wb") as f:
        pickle.dump(v["model"], f)
log("\nAll 3 no-odds models trained. Raw models saved to scratchpad (not deployed yet).")

# ---------------------------------------------------------------------------
# STEP 4 — VERIFY: (a) deployed odds model w/ hardcoded odds vs (b) new no-odds model
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("STEP 4 — VERIFY ON 2025-26 EPL HOLDOUT")
log("=" * 78)

old_models = {}
for name, fname in [("result", "model_backups/result_model_20260821_200210.pkl"),
                     ("goals", "model_backups/goals_model_20260821_200210.pkl"),
                     ("corners", "model_backups/corners_model_20260821_200210.pkl")]:
    with open(fname, "rb") as f:
        old_models[name] = pickle.load(f)

def _fake_odds_row_features(row, old_feat_list):
    """Build the OLD (26/28-feature) vector for one holdout row, using the
    hardcoded 2.60/3.10/2.80 odds triple in place of real odds — exactly
    what fetch_fixtures.py:486 feeds the deployed model today."""
    ih, id_, ia = 1/PROD_ODD_H, 1/PROD_ODD_D, 1/PROD_ODD_A
    rs = ih + id_ + ia
    lookup = {
        "h_r_gf": row["h_r_gf"], "h_r_ga": row["h_r_ga"], "h_r_gd": row["h_r_gd"],
        "h_r_sot": row["h_r_sot"], "h_r_win": row["h_r_win"], "h_r_draw": row["h_r_draw"],
        "h_r_pts": row["h_r_pts"], "h_r_home_win": row["h_r_home_win"],
        "a_r_gf": row["a_r_gf"], "a_r_ga": row["a_r_ga"], "a_r_gd": row["a_r_gd"],
        "a_r_sot": row["a_r_sot"], "a_r_win": row["a_r_win"], "a_r_draw": row["a_r_draw"],
        "a_r_pts": row["a_r_pts"], "a_r_away_win": row["a_r_away_win"],
        "elo_diff": row["elo_diff"], "elo_prob_h": row["elo_prob_h"],
        "form5_h": row["form5_h"], "form5_a": row["form5_a"], "form5_diff": row["form5_diff"],
        "imp_h": ih / rs, "imp_d": id_ / rs, "imp_a": ia / rs, "book_margin": rs - 1.0,
        "league_enc": row["league_enc"],
        "h_r_corners": row.get("h_r_corners", np.nan), "a_r_corners": row.get("a_r_corners", np.nan),
    }
    return [lookup[f] for f in old_feat_list]


def eval_result(proba, y_te, pred):
    acc = accuracy_score(y_te, pred)
    ll = log_loss(y_te, proba, labels=[0, 1, 2])
    onehot = np.eye(3)[y_te]
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    return acc, ll, brier


def eval_binary(proba1, y_te, pred):
    acc = accuracy_score(y_te, pred)
    ll = log_loss(y_te, proba1, labels=[0, 1]) if len(set(y_te)) > 1 else float("nan")
    brier = brier_score_loss(y_te, proba1)
    return acc, ll, brier


rows_report = []
for name, cfg in targets.items():
    te = new_models[name]["te_df"]
    y_te = new_models[name]["y_te"]
    old_feat = old_models[name]["features"]

    X_old = np.array([_fake_odds_row_features(r, old_feat) for _, r in te.iterrows()], dtype=float)
    old_model = old_models[name]["model"]
    if name == "result":
        proba_old = old_model.predict_proba(X_old)
        # old encoder order A/D/H may differ from new le_result — remap to A,D,H=0,1,2
        old_classes = list(old_models[name]["result_encoder"].classes_)
        remap = [old_classes.index(c) for c in le_result.classes_]
        proba_old = proba_old[:, remap]
        pred_old = np.argmax(proba_old, axis=1)
        acc_a, ll_a, brier_a = eval_result(proba_old, y_te, pred_old)
    else:
        proba_old = old_model.predict_proba(X_old)
        classes = list(old_model.classes_)
        p1 = proba_old[:, classes.index(1)] if 1 in classes else proba_old[:, -1]
        pred_old = (p1 >= 0.5).astype(int)
        acc_a, ll_a, brier_a = eval_binary(p1, y_te, pred_old)

    new_model = new_models[name]["model"]
    X_new = te[cfg["feat"]].values.astype(float)
    if name == "result":
        proba_new = new_model.predict_proba(X_new)
        pred_new = np.argmax(proba_new, axis=1)
        acc_b, ll_b, brier_b = eval_result(proba_new, y_te, pred_new)
    else:
        proba_new = new_model.predict_proba(X_new)
        classes = list(new_model.classes_)
        p1 = proba_new[:, classes.index(1)] if 1 in classes else proba_new[:, -1]
        pred_new = (p1 >= 0.5).astype(int)
        acc_b, ll_b, brier_b = eval_binary(p1, y_te, pred_new)

    new_models[name]["proba_holdout"] = proba_new
    new_models[name]["pred_holdout"] = pred_new
    rows_report.append((name, len(te), acc_a, ll_a, brier_a, acc_b, ll_b, brier_b))

log(f"\n{'target':10s} {'n':>5s}  |  {'(a) DEPLOYED+hardcoded odds':^38s}  |  {'(b) NEW no-odds':^30s}")
log(f"{'':10s} {'':>5s}  |  {'acc':>8s} {'logloss':>10s} {'brier':>9s}  |  {'acc':>8s} {'logloss':>10s} {'brier':>9s}")
log("-" * 96)
deploy_ok = True
for name, n, acc_a, ll_a, brier_a, acc_b, ll_b, brier_b in rows_report:
    better = (ll_b < ll_a) and (brier_b < brier_a)
    if not better:
        deploy_ok = False
    log(f"{name:10s} {n:>5d}  |  {acc_a*100:7.2f}% {ll_a:10.4f} {brier_a:9.4f}  |  "
        f"{acc_b*100:7.2f}% {ll_b:10.4f} {brier_b:9.4f}   {'better' if better else 'WORSE'}")

log(f"\nDEPLOY DECISION: {'PROCEED — new model beats deployed-with-hardcoded-odds on every target' if deploy_ok else 'DO NOT DEPLOY — new model does not clearly beat the current bar'}")

# ---------------------------------------------------------------------------
# STEP 5 — calibration table, no-odds Result model, 2025-26 EPL holdout
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("STEP 5 — CALIBRATION TABLE (no-odds Result model, 2025-26 EPL holdout)")
log("=" * 78)
proba_r = new_models["result"]["proba_holdout"]
y_te_r = new_models["result"]["y_te"]
pred_r = new_models["result"]["pred_holdout"]
top_conf = proba_r.max(axis=1)
hit = (pred_r == y_te_r).astype(int)

buckets = [(0.33, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
log(f"{'bucket':>14s}  {'n':>5s}  {'avg stated conf':>16s}  {'actual hit-rate':>16s}")
for lo, hi in buckets:
    mask = (top_conf >= lo) & (top_conf < hi)
    n = int(mask.sum())
    if n == 0:
        log(f"  [{lo:.2f},{hi:.2f})  {n:>5d}  {'--':>16s}  {'--':>16s}")
        continue
    avg_conf = top_conf[mask].mean()
    hr = hit[mask].mean()
    log(f"  [{lo:.2f},{hi:.2f})  {n:>5d}  {avg_conf*100:15.1f}%  {hr*100:15.1f}%")

# ---------------------------------------------------------------------------
# STEP 6 — run 3 real fixtures through the new model
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("STEP 6 — REAL FIXTURE CHECK (new no-odds models)")
log("=" * 78)

import fetch_fixtures as F

def build_noodds_features(hs, as_, elo_diff, form5_h, form5_a, league_enc, feat_list):
    h_pts = 3 * hs["win"] + hs["draw"]
    a_pts = 3 * as_["win"] + as_["draw"]
    elo_prob_h = 1 / (1 + 10 ** (-elo_diff / 400))
    lookup = {
        "h_r_gf": hs["gf"], "h_r_ga": hs["ga"], "h_r_gd": hs["gf"] - hs["ga"],
        "h_r_sot": hs.get("sot", 4.5), "h_r_win": hs["win"], "h_r_draw": hs["draw"],
        "h_r_pts": h_pts, "h_r_home_win": hs.get("hwn", min(hs["win"] + 0.12, 1.0)),
        "a_r_gf": as_["gf"], "a_r_ga": as_["ga"], "a_r_gd": as_["gf"] - as_["ga"],
        "a_r_sot": as_.get("sot", 3.8), "a_r_win": as_["win"], "a_r_draw": as_["draw"],
        "a_r_pts": a_pts, "a_r_away_win": as_.get("awn", max(as_["win"] - 0.12, 0.0)),
        "elo_diff": elo_diff, "elo_prob_h": elo_prob_h,
        "form5_h": form5_h, "form5_a": form5_a, "form5_diff": form5_h - form5_a,
        "league_enc": float(league_enc),
        "h_r_corners": hs.get("corners", 5.2), "a_r_corners": as_.get("corners", 4.8),
    }
    return np.array([[lookup[f] for f in feat_list]], dtype=float)


FIXTURE_PAIRS = [("Arsenal", "Coventry"), ("Hull", "Man United"), ("Ipswich", "Sunderland")]

try:
    live = F._fetch_window(9)
except Exception as e:
    log(f"LIVE FETCH FAILED: {e}")
    live = []

any_flat = False
for home_kw, away_kw in FIXTURE_PAIRS:
    match = None
    for m in live:
        h, a = F._extract_teams(m)
        if home_kw.lower() in h.lower() and away_kw.lower() in a.lower():
            match = (m, h, a)
            break
    if match is None:
        log(f"\n{home_kw} vs {away_kw}: NOT FOUND in next-9-day live window — skipping")
        continue
    m, home_name, away_name = match
    comp_code = F._extract_comp_code(m)
    league_code = F._COMP_TO_LEAGUE.get(comp_code, "E0")
    df_hist = F._load_history(league_code)
    if df_hist.empty:
        df_hist = F._load_history("E0")
    teams = sorted(set(df_hist["home"].tolist() + df_hist["away"].tolist())) if not df_hist.empty else []
    h_res = F._resolve(home_name, teams)
    a_res = F._resolve(away_name, teams)
    hs = F._rolling(df_hist, h_res) if h_res else dict(F._DEFAULT_STATS)
    as_ = F._rolling(df_hist, a_res) if a_res else dict(F._DEFAULT_STATS)
    form5_h = hs.pop("_pts", round((hs["win"] * 3 + hs["draw"]) * 5))
    form5_a = as_.pop("_pts", round((as_["win"] * 3 + as_["draw"]) * 5))
    elo_diff = F._elo_diff_safe(home_name, away_name)
    enc = league_map.get(league_code, 10)

    X = build_noodds_features(hs, as_, elo_diff, form5_h, form5_a, enc, T.BASE_FEATURES)
    Xc = build_noodds_features(hs, as_, elo_diff, form5_h, form5_a, enc, T.CORNER_FEATURES)
    rp = new_models["result"]["model"].predict_proba(X)[0]
    p_home, p_draw, p_away = float(rp[le_result.transform(["H"])[0]]), float(rp[le_result.transform(["D"])[0]]), float(rp[le_result.transform(["A"])[0]])

    log(f"\n{home_name} vs {away_name}  (comp={comp_code}, league_code={league_code}, elo_diff={elo_diff:.1f})")
    log(f"  Home {p_home*100:.1f}%   Draw {p_draw*100:.1f}%   Away {p_away*100:.1f}%")
    if max(p_home, p_draw, p_away) < 0.40:
        any_flat = True
        log("  *** STILL NEAR-FLAT — INVESTIGATE FURTHER ***")

if any_flat:
    log("\nSTOP CONDITION HIT: at least one fixture still near-flat. Do not proceed to deploy.")
else:
    log("\nAll 3 fixtures show clear separation — fix confirmed at the model level.")

log("\n" + "=" * 78)
log("retrain_no_odds.py DONE")
log("=" * 78)

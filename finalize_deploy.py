"""
finalize_deploy.py — bundle the already-trained no-odds models (from
retrain_no_odds.py, saved raw in scratch) into the same pickle dict schema
app.py / predict.py expect, and deploy them as result_model.pkl /
goals_model.pkl / corners_model.pkl.

Does NOT retrain anything — league_map/league_encoder/result_encoder are
deterministic given the same data, so they're rebuilt cheaply (no XGBoost
fitting) and combined with the trained model objects already on disk.
"""
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import train as T

SCRATCH = r"C:\Users\HPZBOO~1\AppData\Local\Temp\claude\C--Users-HP-ZBook-14-Az-Predictions\ab9f99a2-55e3-439e-b6e2-b16c726874ac\scratchpad"

print("Reloading data to rebuild league_map/encoders (deterministic, no retraining)...")
df_main = T.load_matches(T.DATA_DIR)
df_fbref = T.load_fbref_supplements(T.DATA_DIR)
sp1_min = df_main.loc[df_main["league"] == "SP1", "date"].min()
df_laliga = T.load_laliga_early(T.DATA_DIR, sp1_min) if pd.notna(sp1_min) else pd.DataFrame()
extra = [d for d in (df_fbref, df_laliga) if len(d) > 0]
df = pd.concat([df_main] + extra, ignore_index=True) if extra else df_main
df = T.compute_rolling_stats(df, T.N_ROLLING)

le_league = LabelEncoder().fit(df["league"])
league_map = dict(zip(le_league.classes_.tolist(), le_league.transform(le_league.classes_).tolist()))
le_result = LabelEncoder().fit(["A", "D", "H"])

print(f"league_map: {len(league_map)} leagues")
print(f"result classes: {list(le_result.classes_)}")

with open(f"{SCRATCH}/no_odds_result_model.pkl", "rb") as f:
    model_result = pickle.load(f)
with open(f"{SCRATCH}/no_odds_goals_model.pkl", "rb") as f:
    model_goals = pickle.load(f)
with open(f"{SCRATCH}/no_odds_corners_model.pkl", "rb") as f:
    model_corners = pickle.load(f)

with open("result_model.pkl", "wb") as f:
    pickle.dump({
        "model": model_result,
        "result_encoder": le_result,
        "league_encoder": le_league,
        "league_map": league_map,
        "features": T.BASE_FEATURES,
        "odds_free": True,
    }, f)

with open("goals_model.pkl", "wb") as f:
    pickle.dump({
        "model": model_goals,
        "league_encoder": le_league,
        "league_map": league_map,
        "features": T.BASE_FEATURES,
        "odds_free": True,
    }, f)

with open("corners_model.pkl", "wb") as f:
    pickle.dump({
        "model": model_corners,
        "league_encoder": le_league,
        "league_map": league_map,
        "features": T.CORNER_FEATURES,
        "odds_free": True,
    }, f)

print("\nDeployed: result_model.pkl  goals_model.pkl  corners_model.pkl")
print(f"result_model features ({len(T.BASE_FEATURES)}): {T.BASE_FEATURES}")
print(f"corners_model features ({len(T.CORNER_FEATURES)}): {T.CORNER_FEATURES}")

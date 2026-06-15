"""
nba_train.py — Train NBA and WNBA XGBoost prediction models.

NBA  : game table from data/nba.sqlite  (~65 k games 2000-2023)
WNBA : data/wnba_player_and_team_stats_2003-2025  (player-level ESPN data;
       aggregated to game level here, covering 2003-2025 regular season + playoffs)
       Falls back to data/WNBA_BoxScore-1997-2020/... if v2 path missing.

Outputs
-------
nba_result_model.pkl   nba_ou_model.pkl
wnba_result_model.pkl  wnba_ou_model.pkl
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# ─── Paths & constants ────────────────────────────────────────────────────────
NBA_DB           = 'data/nba.sqlite'
WNBA_BS_PATH     = 'data/WNBA_BoxScore-1997-2020/Data/1997-2020_officialBoxScore.csv'
WNBA_V2_PATH     = 'data/wnba_player_and_team_stats_2003-2025'
NBA_RESULT_PATH  = 'nba_result_model.pkl'
NBA_OU_PATH      = 'nba_ou_model.pkl'
WNBA_RESULT_PATH = 'wnba_result_model.pkl'
WNBA_OU_PATH     = 'wnba_ou_model.pkl'

NBA_OU_LINE   = 220.5
WNBA_OU_LINE  = 155.5  # updated: median total is ~156; 170.5 was only 23% over
FORM_N        = 10

# Normalise ESPN abbreviations in v2 data → canonical abbrs used in nba_predict.py
_ABBR_NORM = {
    'CONN': 'CON',  # Connecticut Sun (ESPN used CONN in 2021-2024)
    'LA':   'LAS',  # Los Angeles Sparks
    'LV':   'LVA',  # Las Vegas Aces
    'NY':   'NYL',  # New York Liberty
    'PHX':  'PHO',  # Phoenix Mercury
    'WSH':  'WAS',  # Washington Mystics
    'GS':   'GSV',  # Golden State Valkyries (2025 expansion)
}
# All-Star / exhibition pseudo-team codes to exclude from training
_EXHIBITION = {'CLA', 'COL', 'USA', 'WNBASTARS', 'STE', 'WIL', 'ALL'}
MIN_YEAR_NBA  = 2000
TEST_SPLIT    = 0.20

NBA_FEATURES = [
    'home_win_rate_l10', 'home_avg_pts_l10', 'home_avg_conceded_l10',
    'home_pt_diff_l10',  'home_ortg_l10',    'home_season_win_pct',
    'home_rest_days',    'home_b2b',
    'away_win_rate_l10', 'away_avg_pts_l10', 'away_avg_conceded_l10',
    'away_pt_diff_l10',  'away_ortg_l10',    'away_season_win_pct',
    'away_rest_days',    'away_b2b',
    'win_rate_diff', 'pt_diff_diff', 'rest_diff', 'h2h_home_win_rate',
]

WNBA_FEATURES = [
    'home_win_rate_l10', 'home_avg_pts_l10', 'home_avg_conceded_l10',
    'home_pt_diff_l10',  'home_ortg_l10',    'home_drtg_l10',
    'home_season_win_pct', 'home_rest_days', 'home_b2b',
    'away_win_rate_l10', 'away_avg_pts_l10', 'away_avg_conceded_l10',
    'away_pt_diff_l10',  'away_ortg_l10',    'away_drtg_l10',
    'away_season_win_pct', 'away_rest_days', 'away_b2b',
    'win_rate_diff', 'pt_diff_diff', 'rest_diff', 'h2h_home_win_rate',
]

PARAM_GRID = {
    'n_estimators':     [500],
    'max_depth':        [4, 5, 6],
    'learning_rate':    [0.05, 0.08],
    'subsample':        [0.80],
    'colsample_bytree': [0.80],
    'min_child_weight': [3, 5],
    'gamma':            [0.1],
}


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _rolling(group_col, val_col, df, n=FORM_N):
    """shift(1) then rolling mean — no data leakage."""
    return df.groupby(group_col)[val_col].transform(
        lambda x: x.shift(1).rolling(n, min_periods=3).mean()
    )


def _h2h_rate(df, home_col, away_col, home_win_col, date_col):
    """
    Cumulative home-team win rate in all prior meetings between this pair.
    home_win_col must be a column name (string) present in df.
    Returns a numpy array aligned to df's original index order.
    """
    tmp = df[[date_col, home_col, away_col, home_win_col]].copy()
    tmp['_pair']   = tmp.apply(
        lambda r: tuple(sorted([r[home_col], r[away_col]])), axis=1
    )
    tmp['_h_is_a'] = tmp[home_col] == tmp['_pair'].apply(lambda p: p[0])
    tmp['_a_win']  = np.where(tmp['_h_is_a'], tmp[home_win_col], 1 - tmp[home_win_col])
    tmp = tmp.sort_values(['_pair', date_col]).reset_index(drop=True)
    g = tmp.groupby('_pair')
    tmp['_cnt']     = g.cumcount()
    tmp['_cum_a_w'] = g['_a_win'].transform(lambda x: x.shift(1).fillna(0).cumsum())
    tmp['_h2h_a']   = np.where(
        tmp['_cnt'] > 0,
        tmp['_cum_a_w'] / tmp['_cnt'].clip(lower=1),
        0.5
    )
    tmp['_h2h_home'] = np.where(tmp['_h_is_a'], tmp['_h2h_a'], 1 - tmp['_h2h_a'])
    # Re-align to original df order
    tmp.index = df.index
    return tmp['_h2h_home'].values


def _train_models(X_tr, y_tr, X_te, y_te, label, weights=None):
    print(f"\n  GridSearchCV - {label} ...")
    gs = GridSearchCV(
        XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                      random_state=42, n_jobs=-1),
        PARAM_GRID, cv=3, scoring='accuracy', n_jobs=1, verbose=0,
    )
    gs.fit(X_tr, y_tr, sample_weight=weights)
    best = gs.best_estimator_
    acc  = accuracy_score(y_te, best.predict(X_te))
    print(f"  Best: {gs.best_params_}")
    print(f"  Accuracy: {acc:.1%}")
    print(classification_report(y_te, best.predict(X_te), zero_division=0))
    return best


# ─── NBA ──────────────────────────────────────────────────────────────────────

NBA_SUPPLEMENT_PATH = 'data/nba_supplement_2023_2026.csv'

def load_nba():
    import os
    conn = sqlite3.connect(NBA_DB)
    df = pd.read_sql_query("""
        SELECT game_id, game_date, season_id,
               team_name_home, team_name_away,
               pts_home, pts_away, wl_home, wl_away,
               fga_home, oreb_home, tov_home, fta_home,
               fga_away, oreb_away, tov_away, fta_away
        FROM game
        WHERE season_type IN ('Regular Season','Playoffs')
          AND pts_home IS NOT NULL
          AND pts_away IS NOT NULL
          AND wl_home IS NOT NULL
    """, conn, parse_dates=['game_date'])
    conn.close()
    df = df[df['game_date'].dt.year >= MIN_YEAR_NBA].sort_values('game_date').reset_index(drop=True)
    print(f"  SQLite: {len(df):,} NBA games  ({df['game_date'].min().date()} to {df['game_date'].max().date()})")

    # Append supplement (2023-24, 2024-25, 2025-26) if available
    if os.path.exists(NBA_SUPPLEMENT_PATH):
        sup = pd.read_csv(NBA_SUPPLEMENT_PATH, parse_dates=['game_date'])
        sup = sup[sup['pts_home'].notna() & sup['pts_away'].notna() & sup['wl_home'].notna()]
        # Drop any dates already in SQLite to avoid duplicates
        cutoff = df['game_date'].max()
        sup = sup[sup['game_date'] > cutoff].copy()
        if len(sup) > 0:
            # Align columns
            sup['season_id'] = sup['season_id'].astype(str)
            df['season_id']  = df['season_id'].astype(str)
            df = pd.concat([df, sup[df.columns]], ignore_index=True)
            df = df.sort_values('game_date').reset_index(drop=True)
            print(f"  +supplement: {len(sup):,} games  ({sup['game_date'].min().date()} to {sup['game_date'].max().date()})")
        else:
            print(f"  Supplement found but all dates already in SQLite.")
    else:
        print(f"  NOTE: {NBA_SUPPLEMENT_PATH} not found -- run fetch_nba_supplement.py to add 2023-2026 data")

    print(f"  Combined: {len(df):,} NBA games  ({df['game_date'].min().date()} to {df['game_date'].max().date()})")
    return df


def _approx_ortg(pts, fga, oreb, tov, fta):
    """Approximate offensive rating (pts per 100 possessions)."""
    fga  = fga.fillna(70);  oreb = oreb.fillna(10)
    tov  = tov.fillna(14);  fta  = fta.fillna(20)
    poss = (fga - oreb + tov + 0.44 * fta).clip(lower=1)
    return (pts / poss * 100).clip(80, 140)


def build_nba_features(df):
    df = df.copy()
    df['ortg_h'] = _approx_ortg(df['pts_home'], df['fga_home'], df['oreb_home'], df['tov_home'], df['fta_home'])
    df['ortg_a'] = _approx_ortg(df['pts_away'], df['fga_away'], df['oreb_away'], df['tov_away'], df['fta_away'])

    # Long-form: one row per team per game
    home = df[['game_id','game_date','season_id',
               'team_name_home','team_name_away',
               'pts_home','pts_away','wl_home','ortg_h']].rename(columns={
        'team_name_home':'team','team_name_away':'opp',
        'pts_home':'gf','pts_away':'ga','wl_home':'wl','ortg_h':'ortg'})
    home['is_home'] = 1

    away = df[['game_id','game_date','season_id',
               'team_name_away','team_name_home',
               'pts_away','pts_home','wl_away','ortg_a']].rename(columns={
        'team_name_away':'team','team_name_home':'opp',
        'pts_away':'gf','pts_home':'ga','wl_away':'wl','ortg_a':'ortg'})
    away['is_home'] = 0

    tg = pd.concat([home, away], ignore_index=True)
    tg['win']     = (tg['wl'] == 'W').astype(float)
    tg['pt_diff'] = tg['gf'] - tg['ga']
    tg = tg.sort_values(['team','game_date','game_id']).reset_index(drop=True)

    # Rolling form (shift-1 = no leakage)
    for col, feat in [('win','win_rate'), ('gf','avg_pts'), ('ga','avg_conceded'),
                      ('pt_diff','pt_diff'), ('ortg','ortg')]:
        tg[f'{feat}_l10'] = _rolling('team', col, tg)

    # Rest / back-to-back
    tg['prev_date'] = tg.groupby('team')['game_date'].transform(lambda x: x.shift(1))
    tg['rest_days'] = (tg['game_date'] - tg['prev_date']).dt.days.clip(1, 14).fillna(5)
    tg['b2b']       = (tg['rest_days'] == 1).astype(float)

    # Season win %
    tg = tg.sort_values(['team','season_id','game_date']).reset_index(drop=True)
    sg = tg.groupby(['team','season_id'])
    tg['s_wins']  = sg['win'].transform(lambda x: x.shift(1).cumsum().fillna(0))
    tg['s_games'] = sg['win'].transform(lambda x: x.shift(1).expanding().count().fillna(0))
    tg['season_win_pct'] = (tg['s_wins'] / tg['s_games'].clip(lower=1)).fillna(0.5)

    roll_cols = ['win_rate_l10','avg_pts_l10','avg_conceded_l10',
                 'pt_diff_l10','ortg_l10','season_win_pct','rest_days','b2b']

    home_f = (tg[tg['is_home']==1].set_index('game_id')[roll_cols]
              .rename(columns=lambda c: f'home_{c}'))
    away_f = (tg[tg['is_home']==0].set_index('game_id')[roll_cols]
              .rename(columns=lambda c: f'away_{c}'))

    feat = df.set_index('game_id').join(home_f).join(away_f).reset_index()
    return feat


def train_nba():
    print("\n" + "="*60)
    print("NBA MODEL TRAINING  (SQLite + 2023-2026 supplement)")
    print("="*60)
    print("Loading data ...")
    df = load_nba()

    print("Building features ...")
    feat = build_nba_features(df)

    feat['_hw'] = (feat['wl_home'] == 'W').astype(float)
    feat['h2h_home_win_rate'] = _h2h_rate(feat, 'team_name_home', 'team_name_away',
                                           '_hw', 'game_date')

    feat['target_result'] = (feat['wl_home'] == 'W').astype(int)
    feat['target_ou']     = ((feat['pts_home'] + feat['pts_away']) > NBA_OU_LINE).astype(int)
    feat['win_rate_diff'] = feat['home_win_rate_l10'] - feat['away_win_rate_l10']
    feat['pt_diff_diff']  = feat['home_pt_diff_l10']  - feat['away_pt_diff_l10']
    feat['rest_diff']     = feat['home_rest_days']     - feat['away_rest_days']

    feat = feat.dropna(subset=NBA_FEATURES).sort_values('game_date').reset_index(drop=True)
    print(f"  Feature rows after dropna: {len(feat):,}")

    split = int(len(feat) * (1 - TEST_SPLIT))
    tr, te = feat.iloc[:split], feat.iloc[split:]
    print(f"  Train: {len(tr):,}  Test: {len(te):,}  "
          f"({te['game_date'].min().date()} to {te['game_date'].max().date()})")

    X_tr, X_te = tr[NBA_FEATURES].values, te[NBA_FEATURES].values
    w = 0.5 + 0.5 * np.arange(len(tr)) / len(tr)

    # --- Backtest: evaluate OLD model on same test period -------------------
    print("\n--- BACKTEST: old model vs new model on held-out test set ---")
    old_res_acc = old_ou_acc = None
    try:
        with open(NBA_RESULT_PATH, 'rb') as f:
            old_res_data = pickle.load(f)
        with open(NBA_OU_PATH, 'rb') as f:
            old_ou_data  = pickle.load(f)
        old_res_pred = old_res_data['model'].predict(X_te)
        old_ou_pred  = old_ou_data['model'].predict(X_te)
        old_res_acc  = accuracy_score(te['target_result'].values, old_res_pred)
        old_ou_acc   = accuracy_score(te['target_ou'].values,     old_ou_pred)
        print(f"  Old model - result: {old_res_acc:.1%}   O/U: {old_ou_acc:.1%}"
              f"  (trained to 2023, tested {te['game_date'].min().date()} to {te['game_date'].max().date()})")
    except Exception as e:
        print(f"  Could not load old models for backtest: {e}")

    # --- Train new models ---------------------------------------------------
    print()
    res_model = _train_models(X_tr, tr['target_result'].values,
                               X_te, te['target_result'].values, 'NBA result', w)
    ou_model  = _train_models(X_tr, tr['target_ou'].values,
                               X_te, te['target_ou'].values, 'NBA O/U', w)

    new_res_acc = accuracy_score(te['target_result'].values, res_model.predict(X_te))
    new_ou_acc  = accuracy_score(te['target_ou'].values,     ou_model.predict(X_te))

    # --- Decision: deploy only if new >= old --------------------------------
    print("\n--- DEPLOYMENT DECISION ---")
    print(f"  Result model - old: {f'{old_res_acc:.1%}' if old_res_acc is not None else 'N/A'}"
          f"   new: {new_res_acc:.1%}")
    print(f"  O/U model    - old: {f'{old_ou_acc:.1%}' if old_ou_acc is not None else 'N/A'}"
          f"   new: {new_ou_acc:.1%}")

    deploy_res = (old_res_acc is None) or (new_res_acc >= old_res_acc)
    deploy_ou  = (old_ou_acc  is None) or (new_ou_acc  >= old_ou_acc)

    if deploy_res and deploy_ou:
        print("  [DEPLOY] New models equal-or-better on both metrics -- DEPLOYING")
        with open(NBA_RESULT_PATH, 'wb') as f:
            pickle.dump({'model': res_model, 'features': NBA_FEATURES, 'ou_line': None}, f)
        with open(NBA_OU_PATH, 'wb') as f:
            pickle.dump({'model': ou_model, 'features': NBA_FEATURES, 'ou_line': NBA_OU_LINE}, f)
        print(f"  Saved: {NBA_RESULT_PATH}, {NBA_OU_PATH}")
        _save_nba_team_form(df)
    else:
        reasons = []
        if not deploy_res:
            reasons.append(f"result regressed ({old_res_acc:.1%} -> {new_res_acc:.1%})")
        if not deploy_ou:
            reasons.append(f"O/U regressed ({old_ou_acc:.1%} -> {new_ou_acc:.1%})")
        print(f"  [SKIP] NOT deploying -- {'; '.join(reasons)}")
        print("    Old model files left unchanged.")


def _save_nba_team_form(df):
    """Build and save last-N-games rolling stats for every team (for prediction time)."""
    df = df.copy()
    df['ortg_h'] = _approx_ortg(df['pts_home'], df['fga_home'], df['oreb_home'], df['tov_home'], df['fta_home'])
    df['ortg_a'] = _approx_ortg(df['pts_away'], df['fga_away'], df['oreb_away'], df['tov_away'], df['fta_away'])

    home = df[['game_date','season_id','team_name_home','pts_home','pts_away','wl_home','ortg_h']].rename(
        columns={'team_name_home':'team','pts_home':'gf','pts_away':'ga','wl_home':'wl','ortg_h':'ortg'})
    home['is_home'] = 1
    away = df[['game_date','season_id','team_name_away','pts_away','pts_home','wl_away','ortg_a']].rename(
        columns={'team_name_away':'team','pts_away':'gf','pts_home':'ga','wl_away':'wl','ortg_a':'ortg'})
    away['is_home'] = 0

    tg = pd.concat([home, away], ignore_index=True)
    tg['win']     = (tg['wl'] == 'W').astype(float)
    tg['pt_diff'] = tg['gf'] - tg['ga']
    tg = tg.sort_values(['team','game_date']).reset_index(drop=True)

    # Take last FORM_N games per team (already in history for fallback)
    last_n = tg.groupby('team').tail(FORM_N * 2).copy()
    last_n.to_csv('data/nba_team_form_cache.csv', index=False)
    print("  Saved team form cache -> data/nba_team_form_cache.csv")


# ─── WNBA ─────────────────────────────────────────────────────────────────────

def load_wnba():
    bs = pd.read_csv(WNBA_BS_PATH, encoding='latin-1')
    bs['gmDate'] = pd.to_datetime(bs['gmDate'])
    bs = bs.dropna(subset=['teamPTS','opptPTS','teamOrtg','teamDrtg']).copy()
    bs = bs.sort_values('gmDate').reset_index(drop=True)
    print(f"  Loaded {len(bs):,} WNBA rows  ({bs['gmDate'].min().date()} to {bs['gmDate'].max().date()})")
    return bs


def load_wnba_v2():
    """
    Build WNBA game-level data from the ESPN player-level dataset (2003-2025).
    Aggregates ~23 player rows per team-game into one row per game (away perspective),
    matching the schema expected by build_wnba_features() and build_wnba_long_form().

    Abbreviation normalization aligns ESPN's abbreviated codes to the canonical set
    used in nba_predict.WNBA_NAME_TO_ABBR so the rebuilt form cache is look-up compatible.
    """
    import os
    if not os.path.exists(WNBA_V2_PATH):
        print(f"  WARNING: {WNBA_V2_PATH} not found -- falling back to legacy boxscore")
        return load_wnba()

    df = pd.read_csv(WNBA_V2_PATH, low_memory=False)
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    df = df.dropna(subset=['game_date', 'team_score', 'opponent_team_score'])

    # Keep regular season (2) and playoffs (3) only; drop All-Star / exhibition rows
    df = df[df['season_type'].isin([2, 3])].copy()
    df = df[~df['team_abbreviation'].isin(_EXHIBITION) &
            ~df['opponent_team_abbreviation'].isin(_EXHIBITION)].copy()

    # Standardise abbreviations → canonical form used in nba_predict.py
    df['team_abbreviation']          = df['team_abbreviation'].map(
        lambda a: _ABBR_NORM.get(a, a))
    df['opponent_team_abbreviation'] = df['opponent_team_abbreviation'].map(
        lambda a: _ABBR_NORM.get(a, a))

    # ── Aggregate player rows → one row per (game_id, team) ──────────────────
    tg = df.groupby(['game_id', 'team_abbreviation'], sort=False).agg(
        game_date  = ('game_date',                  'first'),
        season     = ('season',                     'first'),
        home_away  = ('home_away',                  'first'),
        team_score = ('team_score',                 'first'),
        opp_score  = ('opponent_team_score',        'first'),
        team_winner= ('team_winner',                'first'),
        opp_abbr   = ('opponent_team_abbreviation', 'first'),
        fga        = ('field_goals_attempted',      'sum'),
        oreb       = ('offensive_rebounds',         'sum'),
        tov        = ('turnovers',                  'sum'),
        fta        = ('free_throws_attempted',      'sum'),
    ).reset_index()

    # ── Offensive rating (pts per 100 estimated possessions) ─────────────────
    poss = (tg['fga'] - tg['oreb'] + tg['tov'] + 0.44 * tg['fta']).clip(lower=1)
    tg['ortg'] = (tg['team_score'] / poss * 100).clip(60, 160)

    # Defensive rating = opponent's offensive rating for the same game
    ortg_lookup = tg.set_index(['game_id', 'team_abbreviation'])['ortg'].to_dict()
    tg['drtg'] = [ortg_lookup.get((gid, opp), np.nan)
                  for gid, opp in zip(tg['game_id'], tg['opp_abbr'])]

    # ── Rest days and cumulative season W/L (shift-1, no leakage) ────────────
    tg = tg.sort_values(['team_abbreviation', 'game_date']).reset_index(drop=True)
    tg['prev_date'] = tg.groupby('team_abbreviation')['game_date'].transform(
        lambda x: x.shift(1))
    tg['dayOff'] = (tg['game_date'] - tg['prev_date']).dt.days.clip(1, 10).fillna(5)

    tg['win'] = tg['team_winner'].astype(float)
    sg = tg.groupby(['team_abbreviation', 'season'])
    tg['s_wins']   = sg['win'].transform(lambda x: x.shift(1).cumsum().fillna(0))
    tg['s_losses'] = sg['win'].transform(lambda x: (1.0 - x).shift(1).cumsum().fillna(0))

    # ── Split into away (main rows) and home (oppt* columns) ─────────────────
    away = tg[tg['home_away'] == 'away'].copy()
    home = tg[tg['home_away'] == 'home'].copy()

    away = away.rename(columns={
        'team_abbreviation': 'teamAbbr',
        'opp_abbr':          'opptAbbr',
        'team_score':        'teamPTS',
        'opp_score':         'opptPTS',
        'ortg':              'teamOrtg',
        'drtg':              'teamDrtg',
        'dayOff':            'teamDayOff',
        's_wins':            'teamWins',
        's_losses':          'teamLosses',
        'game_date':         'gmDate',
    })
    away['teamRslt'] = np.where(away['team_winner'], 'Win', 'Loss')

    home_stats = home.set_index('game_id')[
        ['ortg', 'drtg', 'dayOff', 's_wins', 's_losses']
    ].rename(columns={
        'ortg':     'opptOrtg',
        'drtg':     'opptDrtg',
        'dayOff':   'opptDayOff',
        's_wins':   'opptWins',
        's_losses': 'opptLosses',
    })

    away = away.set_index('game_id').join(home_stats, how='left').reset_index()
    away.rename(columns={'index': 'game_id'}, inplace=True, errors='ignore')

    keep = ['game_id', 'gmDate', 'season', 'teamAbbr', 'opptAbbr',
            'teamPTS', 'opptPTS', 'teamRslt', 'teamOrtg', 'teamDrtg',
            'teamDayOff', 'teamWins', 'teamLosses',
            'opptOrtg', 'opptDrtg', 'opptDayOff', 'opptWins', 'opptLosses']
    away = away[[c for c in keep if c in away.columns]]
    away = away.dropna(subset=['teamPTS', 'opptPTS']).sort_values('gmDate').reset_index(drop=True)

    print(f"  Loaded {len(away):,} WNBA games (v2, 2003-2025)  "
          f"({away['gmDate'].min().date()} – {away['gmDate'].max().date()})")
    return away


def build_wnba_features(bs):
    """
    All rows in this dataset are from the AWAY team's perspective.
    Construct long-form: team = teamAbbr (away) or opptAbbr (home).
    """
    bs = bs.copy()

    # Canonical game id
    bs['game_id'] = bs.apply(
        lambda r: f"{r['gmDate'].date()}_{min(r['teamAbbr'],r['opptAbbr'])}_{max(r['teamAbbr'],r['opptAbbr'])}",
        axis=1
    )

    # Away team perspective (original row)
    away_rows = bs[['game_id','gmDate','season','teamAbbr','opptAbbr',
                    'teamPTS','opptPTS','teamRslt',
                    'teamOrtg','teamDrtg','teamDayOff',
                    'teamWins','teamLosses']].copy()
    away_rows = away_rows.rename(columns={
        'teamAbbr':'team','opptAbbr':'opp',
        'teamPTS':'gf','opptPTS':'ga','teamRslt':'rslt',
        'teamOrtg':'ortg','teamDrtg':'drtg','teamDayOff':'dayoff',
        'teamWins':'s_wins','teamLosses':'s_losses',
    })
    away_rows['is_home'] = 0
    away_rows['win'] = (away_rows['rslt'] == 'Win').astype(float)

    # Home team perspective (from oppt* columns)
    home_rows = bs[['game_id','gmDate','season','opptAbbr','teamAbbr',
                    'opptPTS','teamPTS','teamRslt',
                    'opptOrtg','opptDrtg','opptDayOff',
                    'opptWins','opptLosses']].copy()
    home_rows = home_rows.rename(columns={
        'opptAbbr':'team','teamAbbr':'opp',
        'opptPTS':'gf','teamPTS':'ga','teamRslt':'rslt',
        'opptOrtg':'ortg','opptDrtg':'drtg','opptDayOff':'dayoff',
        'opptWins':'s_wins','opptLosses':'s_losses',
    })
    home_rows['is_home'] = 1
    home_rows['win'] = (home_rows['rslt'] != 'Win').astype(float)  # home won if away didn't

    tg = pd.concat([away_rows, home_rows], ignore_index=True)
    tg['pt_diff'] = tg['gf'] - tg['ga']
    tg = tg.sort_values(['team','gmDate']).reset_index(drop=True)

    # Rolling form (shift-1)
    for col, feat in [('win','win_rate'), ('gf','avg_pts'), ('ga','avg_conceded'),
                      ('pt_diff','pt_diff'), ('ortg','ortg'), ('drtg','drtg')]:
        tg[f'{feat}_l10'] = _rolling('team', col, tg)

    # Rest & B2B (also keep pre-computed dayoff as fallback)
    tg['prev_date'] = tg.groupby('team')['gmDate'].transform(lambda x: x.shift(1))
    tg['rest_days'] = (tg['gmDate'] - tg['prev_date']).dt.days.clip(1, 10).fillna(tg['dayoff'].clip(1,10))
    tg['b2b']       = (tg['rest_days'] <= 1).astype(float)

    # Season win %: use pre-computed wins/losses (already pre-game values)
    total = (tg['s_wins'] + tg['s_losses']).clip(lower=1)
    tg['season_win_pct'] = (tg['s_wins'] / total).fillna(0.5)

    roll_cols = ['win_rate_l10','avg_pts_l10','avg_conceded_l10','pt_diff_l10',
                 'ortg_l10','drtg_l10','season_win_pct','rest_days','b2b']

    # Build game-level: home team row joined with away team rolling features
    home_f = (tg[tg['is_home']==1].set_index('game_id')[roll_cols]
              .rename(columns=lambda c: f'home_{c}'))
    away_f = (tg[tg['is_home']==0].set_index('game_id')[roll_cols]
              .rename(columns=lambda c: f'away_{c}'))

    # Game-level base: take home perspective rows
    base = (tg[tg['is_home']==1]
            .set_index('game_id')[['gmDate','team','opp','win','gf','ga']]
            .rename(columns={'team':'home_team','opp':'away_team',
                             'win':'home_win','gf':'home_pts','ga':'away_pts'}))

    feat = base.join(home_f, how='left').join(away_f, how='left').reset_index()

    # H2H
    feat['h2h_home_win_rate'] = _h2h_rate(feat, 'home_team', 'away_team', 'home_win', 'gmDate')
    return feat


def train_wnba():
    print("\n" + "="*60)
    print("WNBA MODEL TRAINING  (v2 data: 2003-2025)")
    print("="*60)

    print("Loading data ...")
    bs = load_wnba_v2()

    print("Building features ...")
    feat = build_wnba_features(bs)

    feat['target_result'] = feat['home_win'].astype(int)
    feat['target_ou']     = ((feat['home_pts'] + feat['away_pts']) > WNBA_OU_LINE).astype(int)
    feat['win_rate_diff'] = feat['home_win_rate_l10'] - feat['away_win_rate_l10']
    feat['pt_diff_diff']  = feat['home_pt_diff_l10']  - feat['away_pt_diff_l10']
    feat['rest_diff']     = feat['home_rest_days']     - feat['away_rest_days']

    feat = feat.dropna(subset=WNBA_FEATURES).sort_values('gmDate').reset_index(drop=True)
    print(f"  Feature rows after dropna: {len(feat):,}")

    split = int(len(feat) * (1 - TEST_SPLIT))
    tr, te = feat.iloc[:split], feat.iloc[split:]
    print(f"  Train: {len(tr):,}  Test: {len(te):,}  "
          f"({te['gmDate'].min().date()} – {te['gmDate'].max().date()})")

    X_tr, X_te = tr[WNBA_FEATURES].values, te[WNBA_FEATURES].values
    w = 0.5 + 0.5 * np.arange(len(tr)) / len(tr)

    # --- Backtest: evaluate OLD model on same test period -------------------
    print("\n--- BACKTEST: old model vs new model on held-out test set ---")
    old_res_acc = old_ou_acc = None
    try:
        with open(WNBA_RESULT_PATH, 'rb') as f:
            old_res_data = pickle.load(f)
        with open(WNBA_OU_PATH, 'rb') as f:
            old_ou_data  = pickle.load(f)
        old_res_pred = old_res_data['model'].predict(X_te)
        old_ou_pred  = old_ou_data['model'].predict(X_te)
        old_res_acc  = accuracy_score(te['target_result'].values, old_res_pred)
        old_ou_acc   = accuracy_score(te['target_ou'].values,     old_ou_pred)
        print(f"  Old model  - result: {old_res_acc:.1%}   O/U: {old_ou_acc:.1%}"
              f"  (trained on 1997-2020 data, tested on {te['gmDate'].min().date()} to {te['gmDate'].max().date()})")
    except Exception as e:
        print(f"  Could not load old models for backtest: {e}")

    # --- Train new models ---------------------------------------------------
    print()
    res_model = _train_models(X_tr, tr['target_result'].values,
                               X_te, te['target_result'].values, 'WNBA result', w)
    ou_model  = _train_models(X_tr, tr['target_ou'].values,
                               X_te, te['target_ou'].values, 'WNBA O/U', w)

    new_res_acc = accuracy_score(te['target_result'].values, res_model.predict(X_te))
    new_ou_acc  = accuracy_score(te['target_ou'].values,     ou_model.predict(X_te))

    # --- Decision: deploy only if new >= old --------------------------------
    print("\n--- DEPLOYMENT DECISION ---")
    print(f"  Result model  - old: {f'{old_res_acc:.1%}' if old_res_acc is not None else 'N/A'}"
          f"   new: {new_res_acc:.1%}")
    print(f"  O/U model     - old: {f'{old_ou_acc:.1%}' if old_ou_acc is not None else 'N/A'}"
          f"   new: {new_ou_acc:.1%}")

    deploy_res = (old_res_acc is None) or (new_res_acc >= old_res_acc)
    deploy_ou  = (old_ou_acc  is None) or (new_ou_acc  >= old_ou_acc)

    if deploy_res and deploy_ou:
        print("  [DEPLOY] New models equal-or-better on both metrics -- DEPLOYING")
        with open(WNBA_RESULT_PATH, 'wb') as f:
            pickle.dump({'model': res_model, 'features': WNBA_FEATURES, 'ou_line': None}, f)
        with open(WNBA_OU_PATH, 'wb') as f:
            pickle.dump({'model': ou_model, 'features': WNBA_FEATURES, 'ou_line': WNBA_OU_LINE}, f)
        print(f"  Saved: {WNBA_RESULT_PATH}, {WNBA_OU_PATH}")

        # Rebuild team form cache from new data
        tg_all = build_wnba_long_form(bs)
        tg_all.groupby('team').tail(FORM_N * 2).to_csv('data/wnba_team_form_cache.csv', index=False)
        print("  Saved team form cache -> data/wnba_team_form_cache.csv")
    else:
        reasons = []
        if not deploy_res:
            reasons.append(f"result model regressed ({old_res_acc:.1%} -> {new_res_acc:.1%})")
        if not deploy_ou:
            reasons.append(f"O/U model regressed ({old_ou_acc:.1%} -> {new_ou_acc:.1%})")
        print(f"  [SKIP] NOT deploying -- {'; '.join(reasons)}")
        print("    Old model files left unchanged.")


def build_wnba_long_form(bs):
    """Return long-form team-game DataFrame (used for prediction-time form lookup)."""
    bs = bs.copy()
    away_rows = bs[['gmDate','season','teamAbbr','opptAbbr',
                    'teamPTS','opptPTS','teamRslt',
                    'teamOrtg','teamDrtg','teamDayOff',
                    'teamWins','teamLosses']].rename(columns={
        'teamAbbr':'team','opptAbbr':'opp',
        'teamPTS':'gf','opptPTS':'ga','teamRslt':'rslt',
        'teamOrtg':'ortg','teamDrtg':'drtg','teamDayOff':'dayoff',
        'teamWins':'s_wins','teamLosses':'s_losses',
    })
    away_rows['is_home'] = 0
    away_rows['win'] = (away_rows['rslt'] == 'Win').astype(float)

    home_rows = bs[['gmDate','season','opptAbbr','teamAbbr',
                    'opptPTS','teamPTS','teamRslt',
                    'opptOrtg','opptDrtg','opptDayOff',
                    'opptWins','opptLosses']].rename(columns={
        'opptAbbr':'team','teamAbbr':'opp',
        'opptPTS':'gf','teamPTS':'ga','teamRslt':'rslt',
        'opptOrtg':'ortg','opptDrtg':'drtg','opptDayOff':'dayoff',
        'opptWins':'s_wins','opptLosses':'s_losses',
    })
    home_rows['is_home'] = 1
    home_rows['win'] = (home_rows['rslt'] != 'Win').astype(float)

    tg = pd.concat([away_rows, home_rows], ignore_index=True)
    tg['pt_diff'] = tg['gf'] - tg['ga']
    return tg.sort_values(['team','gmDate']).reset_index(drop=True)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    train_nba()
    train_wnba()
    print("\nAll models saved.")


if __name__ == '__main__':
    main()

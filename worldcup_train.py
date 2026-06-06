"""
worldcup_train.py — Train XGBoost models on international football results.
Models:
  worldcup_result_model.pkl  — Match result: 2=Home Win, 1=Draw, 0=Away Win
  worldcup_goals_model.pkl   — Goals: 1=Over 2.5, 0=Under/Equal 2.5
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import unicodedata
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_PATH          = 'data/results.csv'
RESULT_MODEL_PATH  = 'worldcup_result_model.pkl'
GOALS_MODEL_PATH   = 'worldcup_goals_model.pkl'
FORM_WINDOW        = 10   # rolling match window for form stats
TEST_SPLIT         = 0.20  # last 20% of data (chronological)

TOURNAMENT_IMPORTANCE = {
    'FIFA World Cup':                            1.00,
    'Confederations Cup':                        0.92,
    'UEFA Euro':                                 0.90,
    'Copa America':                              0.88,  # normalised
    'African Cup of Nations':                    0.85,
    'FIFA World Cup qualification':              0.85,
    'Gold Cup':                                  0.80,
    'AFC Asian Cup':                             0.82,
    'CONCACAF Championship':                     0.78,
    'Olympic Games':                             0.75,
    'UEFA Euro qualification':                   0.75,
    'British Home Championship':                 0.72,
    'UEFA Nations League':                       0.72,
    'African Cup of Nations qualification':      0.70,
    'AFC Asian Cup qualification':               0.68,
    'CONCACAF Nations League':                   0.68,
    'Gold Cup qualification':                    0.65,
    'Copa America qualification':                0.65,
    'Oceania Nations Cup':                       0.65,
    'CONCACAF Championship qualification':       0.60,
    'CONCACAF Nations League qualification':     0.55,
    'Oceania Nations Cup qualification':         0.55,
    'FIFA Series':                               0.40,
    'Friendly':                                  0.30,
}

# Features that must match exactly between train and predict
FEATURES = [
    'home_win_rate', 'home_draw_rate', 'home_loss_rate',
    'home_goals_scored', 'home_goals_conceded', 'home_form_pts',
    'home_form_count', 'home_goal_diff',
    'away_win_rate', 'away_draw_rate', 'away_loss_rate',
    'away_goals_scored', 'away_goals_conceded', 'away_form_pts',
    'away_form_count', 'away_goal_diff',
    'is_neutral', 'tournament_importance',
    'h2h_count', 'h2h_home_win_rate',
    'win_rate_diff', 'goals_scored_diff', 'goals_conceded_diff', 'form_pts_diff',
    # Major-tournament specific form (importance >= 0.70)
    'home_major_win_rate', 'home_major_form_pts', 'home_major_count',
    'away_major_win_rate', 'away_major_form_pts', 'away_major_count',
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _norm(text):
    """Normalise unicode so é/ó etc. match ASCII keys."""
    return unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii')


def get_importance(tournament):
    t_norm = _norm(tournament)
    for key, val in TOURNAMENT_IMPORTANCE.items():
        if t_norm == _norm(key):
            return val
    t_lower = t_norm.lower()
    if 'world cup' in t_lower:
        return 0.80 if 'qualif' not in t_lower else 0.75
    if 'qualif' in t_lower:
        return 0.60
    if 'friendly' in t_lower:
        return 0.30
    if any(x in t_lower for x in ['cup', 'championship', 'nations', 'league']):
        return 0.60
    return 0.45


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH, encoding='latin-1')

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['tournament_importance'] = df['tournament'].apply(get_importance)
    df['is_neutral'] = df['neutral'].astype(int)

    # Targets (only for rows with known scores)
    df['result'] = np.where(
        df['home_score'] > df['away_score'], 2,
        np.where(df['home_score'] == df['away_score'], 1, 0)
    )
    df['over_2_5'] = ((df['home_score'] + df['away_score']) > 2.5).astype(int)

    return df


# ─── Team Form (vectorised) ───────────────────────────────────────────────────

def build_team_form(df, n=FORM_WINDOW):
    """
    Long-form: one row per team per match.
    Computes shift(1).rolling(n) stats so the current match is never in its own window.
    Returns DataFrame indexed by match_idx with prefixed columns.
    """
    home = df[['date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    home['match_idx'] = df.index
    home['side'] = 'home'
    home.rename(columns={'home_team': 'team', 'home_score': 'gf', 'away_score': 'ga'}, inplace=True)

    away = df[['date', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
    away['match_idx'] = df.index
    away['side'] = 'away'
    away.rename(columns={'away_team': 'team', 'away_score': 'gf', 'home_score': 'ga'}, inplace=True)

    long = pd.concat([home, away], ignore_index=True)
    long['win']  = (long['gf'] > long['ga']).astype(int)
    long['draw'] = (long['gf'] == long['ga']).astype(int)
    long['loss'] = (long['gf'] < long['ga']).astype(int)
    long['pts']  = long['win'] * 3 + long['draw']
    long = long.sort_values(['team', 'date']).reset_index(drop=True)

    g = long.groupby('team')
    for col in ['win', 'draw', 'loss', 'pts', 'gf', 'ga']:
        long[f'roll_{col}'] = g[col].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean()
        )
    long['roll_count'] = g['gf'].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).count()
    )
    long['roll_gd'] = long['roll_gf'] - long['roll_ga']

    # Split back by side and index on match_idx
    home_out = long[long['side'] == 'home'].set_index('match_idx')
    away_out  = long[long['side'] == 'away'].set_index('match_idx')

    roll_cols = ['roll_win', 'roll_draw', 'roll_loss', 'roll_pts',
                 'roll_gf', 'roll_ga', 'roll_count', 'roll_gd']

    home_feats = home_out[roll_cols].rename(columns={
        'roll_win': 'home_win_rate', 'roll_draw': 'home_draw_rate',
        'roll_loss': 'home_loss_rate', 'roll_pts': 'home_form_pts',
        'roll_gf': 'home_goals_scored', 'roll_ga': 'home_goals_conceded',
        'roll_count': 'home_form_count', 'roll_gd': 'home_goal_diff',
    })
    away_feats = away_out[roll_cols].rename(columns={
        'roll_win': 'away_win_rate', 'roll_draw': 'away_draw_rate',
        'roll_loss': 'away_loss_rate', 'roll_pts': 'away_form_pts',
        'roll_gf': 'away_goals_scored', 'roll_ga': 'away_goals_conceded',
        'roll_count': 'away_form_count', 'roll_gd': 'away_goal_diff',
    })
    return home_feats, away_feats


# ─── Major-tournament Form ────────────────────────────────────────────────────

def build_major_form(df, n=10, importance_threshold=0.70):
    """
    Same rolling approach as build_team_form but restricted to high-importance
    tournaments (World Cup, Euros, Copa America, etc.).
    """
    major = df[df['tournament_importance'] >= importance_threshold].copy()
    major_idx = major.index

    home = major[['date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    home['match_idx'] = major_idx
    home['side'] = 'home'
    home.rename(columns={'home_team': 'team', 'home_score': 'gf', 'away_score': 'ga'}, inplace=True)

    away = major[['date', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
    away['match_idx'] = major_idx
    away['side'] = 'away'
    away.rename(columns={'away_team': 'team', 'away_score': 'gf', 'home_score': 'ga'}, inplace=True)

    long = pd.concat([home, away], ignore_index=True)
    long['win'] = (long['gf'] > long['ga']).astype(int)
    long['pts'] = long['win'] * 3 + (long['gf'] == long['ga']).astype(int)
    long = long.sort_values(['team', 'date']).reset_index(drop=True)

    g = long.groupby('team')
    long['roll_win'] = g['win'].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    long['roll_pts'] = g['pts'].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    long['roll_count'] = g['win'].transform(lambda x: x.shift(1).rolling(n, min_periods=1).count())

    home_out = long[long['side'] == 'home'].set_index('match_idx')
    away_out  = long[long['side'] == 'away'].set_index('match_idx')

    home_mf = home_out[['roll_win', 'roll_pts', 'roll_count']].rename(columns={
        'roll_win': 'home_major_win_rate',
        'roll_pts': 'home_major_form_pts',
        'roll_count': 'home_major_count',
    })
    away_mf = away_out[['roll_win', 'roll_pts', 'roll_count']].rename(columns={
        'roll_win': 'away_major_win_rate',
        'roll_pts': 'away_major_form_pts',
        'roll_count': 'away_major_count',
    })
    return home_mf, away_mf


# ─── H2H Features (vectorised cumsum) ────────────────────────────────────────

def build_h2h_features(df):
    """
    Canonical pair (alphabetical). Cumulative H2H win-rate for the home team
    using only matches that precede the current one.
    Returns a DataFrame aligned to df.index.
    """
    tmp = df[['date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    tmp['match_idx'] = df.index

    tmp['pair_a'] = tmp[['home_team', 'away_team']].min(axis=1)
    tmp['pair_b'] = tmp[['home_team', 'away_team']].max(axis=1)
    tmp['a_is_home'] = tmp['pair_a'] == tmp['home_team']

    gf_a = np.where(tmp['a_is_home'], tmp['home_score'], tmp['away_score'])
    gf_b = np.where(tmp['a_is_home'], tmp['away_score'], tmp['home_score'])

    tmp['a_win']  = (gf_a > gf_b).astype(float)
    tmp['b_win']  = (gf_b > gf_a).astype(float)
    tmp['ab_draw'] = (gf_a == gf_b).astype(float)

    tmp = tmp.sort_values(['pair_a', 'pair_b', 'date']).reset_index(drop=True)
    g = tmp.groupby(['pair_a', 'pair_b'])

    tmp['h2h_count'] = g.cumcount()  # number of prior meetings (0-indexed)
    tmp['cum_a_wins'] = g['a_win'].transform(lambda x: x.shift(1).fillna(0).cumsum())
    tmp['cum_b_wins'] = g['b_win'].transform(lambda x: x.shift(1).fillna(0).cumsum())

    # Win count for whichever team is home in this specific match
    home_wins = np.where(tmp['a_is_home'], tmp['cum_a_wins'], tmp['cum_b_wins'])
    tmp['h2h_home_win_rate'] = np.where(
        tmp['h2h_count'] > 0,
        home_wins / tmp['h2h_count'].clip(lower=1),
        1/3  # no history → neutral prior
    )

    return tmp.set_index('match_idx')[['h2h_count', 'h2h_home_win_rate']]


# ─── Sample Weights ───────────────────────────────────────────────────────────

def compute_weights(df):
    """Tournament importance × recency decay (linear from 0.5 → 1.0)."""
    year = df['date'].dt.year
    y_min, y_max = year.min(), year.max()
    recency = 0.5 + 0.5 * (year - y_min) / max(y_max - y_min, 1)
    return (df['tournament_importance'] * recency).values


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    df_all = load_data()
    print(f"  Total rows: {len(df_all):,}  |  score available: {df_all['home_score'].notna().sum():,}")

    print("Computing team form features ...")
    home_feats, away_feats = build_team_form(df_all)

    print("Computing major-tournament form features ...")
    home_mf, away_mf = build_major_form(df_all)

    print("Computing H2H features ...")
    h2h_feats = build_h2h_features(df_all)

    # Assemble feature matrix
    feat = (df_all[['tournament_importance', 'is_neutral', 'result', 'over_2_5',
                     'date', 'home_score', 'away_score']]
            .join(home_feats, how='left')
            .join(away_feats,  how='left')
            .join(home_mf,     how='left')
            .join(away_mf,     how='left')
            .join(h2h_feats,   how='left'))

    # Differential features
    feat['win_rate_diff']        = feat['home_win_rate']      - feat['away_win_rate']
    feat['goals_scored_diff']    = feat['home_goals_scored']  - feat['away_goals_scored']
    feat['goals_conceded_diff']  = feat['home_goals_conceded']- feat['away_goals_conceded']
    feat['form_pts_diff']        = feat['home_form_pts']      - feat['away_form_pts']

    # Keep only rows with known scores
    feat = feat.dropna(subset=['home_score', 'away_score'])

    # Fill remaining NaNs (teams with no prior matches -> 0)
    feat[FEATURES] = feat[FEATURES].fillna(0)

    print(f"  Training rows after filtering: {len(feat):,}")

    # Chronological split
    split_idx = int(len(feat) * (1 - TEST_SPLIT))
    train = feat.iloc[:split_idx]
    test  = feat.iloc[split_idx:]
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}")
    print(f"  Test period: {test['date'].min().date()} to {test['date'].max().date()}")

    X_train, y_res_train  = train[FEATURES].values, train['result'].values
    X_test,  y_res_test   = test[FEATURES].values,  test['result'].values
    y_goals_train = train['over_2_5'].values
    y_goals_test  = test['over_2_5'].values

    weights_train = compute_weights(train)

    # ── Result model ────────────────────────────────────────────────────────
    print("\nTraining result model (Win / Draw / Loss) ...")
    result_model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=3,
        gamma=0.1,
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    result_model.fit(X_train, y_res_train, sample_weight=weights_train)

    res_pred = result_model.predict(X_test)
    res_acc  = accuracy_score(y_res_test, res_pred)
    print(f"  Result accuracy: {res_acc:.1%}")
    print(classification_report(
        y_res_test, res_pred,
        target_names=['Away Win', 'Draw', 'Home Win'],
        zero_division=0,
    ))

    # ── Goals model ─────────────────────────────────────────────────────────
    print("Training goals model (Over / Under 2.5) ...")
    goals_model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=3,
        gamma=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    goals_model.fit(X_train, y_goals_train, sample_weight=weights_train)

    goals_pred = goals_model.predict(X_test)
    goals_acc  = accuracy_score(y_goals_test, goals_pred)
    print(f"  Goals O/U accuracy: {goals_acc:.1%}")
    print(classification_report(
        y_goals_test, goals_pred,
        target_names=['Under 2.5', 'Over 2.5'],
        zero_division=0,
    ))

    # ── Save models ─────────────────────────────────────────────────────────
    with open(RESULT_MODEL_PATH, 'wb') as f:
        pickle.dump({'model': result_model, 'features': FEATURES}, f)
    print(f"\nSaved: {RESULT_MODEL_PATH}")

    with open(GOALS_MODEL_PATH, 'wb') as f:
        pickle.dump({'model': goals_model, 'features': FEATURES}, f)
    print(f"Saved: {GOALS_MODEL_PATH}")

    # WC-only subset accuracy
    wc_test = test[test['tournament_importance'] >= 0.85]
    if len(wc_test) > 0:
        wc_acc = accuracy_score(wc_test['result'].values,
                                result_model.predict(wc_test[FEATURES].values))
        print(f"\nWorld Cup / major tournament accuracy on test set: {wc_acc:.1%}  (n={len(wc_test)})")


if __name__ == '__main__':
    main()

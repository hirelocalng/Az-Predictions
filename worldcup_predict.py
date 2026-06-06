"""
worldcup_predict.py -- Predict international match outcomes using trained XGBoost models.

Usage:
    python worldcup_predict.py "Brazil" "Argentina"
    python worldcup_predict.py  (interactive prompt)
"""

import sys
import difflib
import pickle
import warnings
import unicodedata

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# --- Config ------------------------------------------------------------------

DATA_PATH          = 'data/results.csv'
RESULT_MODEL_PATH  = 'worldcup_result_model.pkl'
GOALS_MODEL_PATH   = 'worldcup_goals_model.pkl'
FORM_WINDOW        = 10

TOURNAMENT_IMPORTANCE = {
    'FIFA World Cup':                            1.00,
    'Confederations Cup':                        0.92,
    'UEFA Euro':                                 0.90,
    'Copa America':                              0.88,
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

# Common name aliases -> canonical name in dataset
TEAM_ALIASES = {
    'usa':              'United States',
    'us':               'United States',
    'america':          'United States',
    'uk':               'England',
    'great britain':    'England',
    'south korea':      'South Korea',
    'korea':            'South Korea',
    'dpr korea':        'North Korea',
    'north korea':      'North Korea',
    'ivory coast':      "Ivory Coast",
    "cote d'ivoire":    "Ivory Coast",
    'iran':             'Iran',
    'russia':           'Russia',
    'czechia':          'Czech Republic',
    'czech':            'Czech Republic',
    'türkiye':          'Turkey',
    'turkiye':          'Turkey',
}


# --- Helpers -----------------------------------------------------------------

def _norm(text):
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


# --- Data & Model Loading -----------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def load_models():
    with open(RESULT_MODEL_PATH, 'rb') as f:
        rd = pickle.load(f)
    with open(GOALS_MODEL_PATH, 'rb') as f:
        gd = pickle.load(f)
    return rd['model'], rd['features'], gd['model'], gd['features']


# --- Team Resolution ----------------------------------------------------------

def resolve_team(name, all_teams):
    """Return canonical team name or raise with suggestions."""
    # Alias lookup (lowercase)
    alias = TEAM_ALIASES.get(name.lower().strip())
    if alias:
        return alias

    # Exact match (case-insensitive)
    name_lower = name.lower().strip()
    for t in all_teams:
        if t.lower() == name_lower:
            return t

    # Fuzzy match
    matches = difflib.get_close_matches(name, all_teams, n=5, cutoff=0.5)
    if matches:
        raise ValueError(
            f"Team '{name}' not found.\n"
            f"Did you mean: {', '.join(matches)}\n"
            f"Use: python worldcup_predict.py \"<exact name>\""
        )
    raise ValueError(
        f"Team '{name}' not found in dataset.\n"
        f"Run python worldcup_predict.py --list to see all teams."
    )


# --- Feature Computation ------------------------------------------------------

def get_team_form(df, team, n=FORM_WINDOW):
    """Last-n-match stats for a team, from all matches in df."""
    mask = (df['home_team'] == team) | (df['away_team'] == team)
    matches = df[mask].sort_values('date').tail(n)

    if len(matches) == 0:
        return {k: 0.0 for k in [
            'win_rate', 'draw_rate', 'loss_rate',
            'goals_scored', 'goals_conceded', 'form_pts', 'form_count', 'goal_diff'
        ]}

    wins = draws = losses = gf = ga = 0
    for _, row in matches.iterrows():
        if row['home_team'] == team:
            g, gc = row['home_score'], row['away_score']
        else:
            g, gc = row['away_score'], row['home_score']
        gf += g
        ga += gc
        if g > gc:
            wins += 1
        elif g == gc:
            draws += 1
        else:
            losses += 1

    n_m = len(matches)
    return {
        'win_rate':        wins / n_m,
        'draw_rate':       draws / n_m,
        'loss_rate':       losses / n_m,
        'goals_scored':    gf / n_m,
        'goals_conceded':  ga / n_m,
        'form_pts':        (wins * 3 + draws) / n_m,
        'form_count':      float(n_m),
        'goal_diff':       (gf - ga) / n_m,
    }


def get_major_form(df, team, n=10, importance_threshold=0.70):
    """Form stats restricted to high-importance matches only."""
    mask = (
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['tournament'].apply(get_importance) >= importance_threshold)
    )
    matches = df[mask].sort_values('date').tail(n)

    if len(matches) == 0:
        return {'major_win_rate': 0.0, 'major_form_pts': 0.0, 'major_count': 0.0}

    wins = pts = 0
    for _, row in matches.iterrows():
        if row['home_team'] == team:
            gf, ga = row['home_score'], row['away_score']
        else:
            gf, ga = row['away_score'], row['home_score']
        if gf > ga:
            wins += 1
            pts += 3
        elif gf == ga:
            pts += 1

    n_m = len(matches)
    return {
        'major_win_rate': wins / n_m,
        'major_form_pts': pts / n_m,
        'major_count':    float(n_m),
    }


def get_h2h(df, team_a, team_b, n=20):
    """
    H2H record from team_a's perspective.
    team_a is the 'home' team in our prediction frame.
    """
    mask = (
        ((df['home_team'] == team_a) & (df['away_team'] == team_b)) |
        ((df['home_team'] == team_b) & (df['away_team'] == team_a))
    )
    h2h = df[mask].sort_values('date').tail(n)

    if len(h2h) == 0:
        return {'count': 0, 'home_win_rate': 1/3}

    wins_a = 0
    for _, row in h2h.iterrows():
        if row['home_team'] == team_a:
            if row['home_score'] > row['away_score']:
                wins_a += 1
        else:
            if row['away_score'] > row['home_score']:
                wins_a += 1

    return {
        'count':         float(len(h2h)),
        'home_win_rate': wins_a / len(h2h),
    }


def build_feature_vector(home_form, away_form, h2h, features,
                         is_neutral=True, tournament_importance=1.0):
    row = {
        'home_win_rate':       home_form['win_rate'],
        'home_draw_rate':      home_form['draw_rate'],
        'home_loss_rate':      home_form['loss_rate'],
        'home_goals_scored':   home_form['goals_scored'],
        'home_goals_conceded': home_form['goals_conceded'],
        'home_form_pts':       home_form['form_pts'],
        'home_form_count':     home_form['form_count'],
        'home_goal_diff':      home_form['goal_diff'],

        'away_win_rate':       away_form['win_rate'],
        'away_draw_rate':      away_form['draw_rate'],
        'away_loss_rate':      away_form['loss_rate'],
        'away_goals_scored':   away_form['goals_scored'],
        'away_goals_conceded': away_form['goals_conceded'],
        'away_form_pts':       away_form['form_pts'],
        'away_form_count':     away_form['form_count'],
        'away_goal_diff':      away_form['goal_diff'],

        'is_neutral':              int(is_neutral),
        'tournament_importance':   tournament_importance,

        'h2h_count':           h2h['count'],
        'h2h_home_win_rate':   h2h['home_win_rate'],

        'win_rate_diff':       home_form['win_rate']      - away_form['win_rate'],
        'goals_scored_diff':   home_form['goals_scored']  - away_form['goals_scored'],
        'goals_conceded_diff': home_form['goals_conceded']- away_form['goals_conceded'],
        'form_pts_diff':       home_form['form_pts']      - away_form['form_pts'],
        # Major-tournament form (filled by caller)
        'home_major_win_rate':  home_form.get('major_win_rate', 0.0),
        'home_major_form_pts':  home_form.get('major_form_pts', 0.0),
        'home_major_count':     home_form.get('major_count', 0.0),
        'away_major_win_rate':  away_form.get('major_win_rate', 0.0),
        'away_major_form_pts':  away_form.get('major_form_pts', 0.0),
        'away_major_count':     away_form.get('major_count', 0.0),
    }
    return np.array([[row[f] for f in features]])


# --- Display -----------------------------------------------------------------

def confidence_label(prob):
    if prob >= 0.70:
        return 'Very likely'
    if prob >= 0.55:
        return 'Likely'
    if prob >= 0.45:
        return 'Slight edge'
    return 'Uncertain'


def print_recent_form(df, team, n=10):
    mask = (df['home_team'] == team) | (df['away_team'] == team)
    matches = df[mask].sort_values('date').tail(n)
    form_str = ''
    for _, r in matches.iterrows():
        if r['home_team'] == team:
            gf, ga = r['home_score'], r['away_score']
        else:
            gf, ga = r['away_score'], r['home_score']
        form_str += 'W' if gf > ga else ('D' if gf == ga else 'L')
    return form_str  # oldest -> newest, left to right


def print_h2h_summary(df, team_a, team_b, n=10):
    mask = (
        ((df['home_team'] == team_a) & (df['away_team'] == team_b)) |
        ((df['home_team'] == team_b) & (df['away_team'] == team_a))
    )
    h2h = df[mask].sort_values('date').tail(n)
    lines = []
    for _, r in h2h.iterrows():
        hs, as_ = r['home_score'], r['away_score']
        if hs == as_:
            winner = 'Draw'
        elif r['home_team'] == team_a:
            winner = team_a if hs > as_ else team_b
        else:
            winner = team_b if hs > as_ else team_a
        lines.append(
            f"  {str(r['date'].date())}  "
            f"{r['home_team']} {int(r['home_score'])}-{int(r['away_score'])} {r['away_team']}  "
            f"({winner})"
        )
    return lines


# --- Main ---------------------------------------------------------------------

def predict(team_a_raw, team_b_raw, is_neutral=True, tournament='FIFA World Cup'):
    # Load
    result_model, res_features, goals_model, goals_features = load_models()
    df = load_data()
    all_teams = sorted(set(df['home_team'].tolist() + df['away_team'].tolist()))

    # Handle --list flag
    if team_a_raw == '--list':
        print('\n'.join(all_teams))
        return

    team_a = resolve_team(team_a_raw, all_teams)
    team_b = resolve_team(team_b_raw, all_teams)

    importance = get_importance(tournament)

    # Compute features
    home_form = get_team_form(df, team_a)
    away_form = get_team_form(df, team_b)
    h2h       = get_h2h(df, team_a, team_b)

    # Major-tournament form (merged into form dicts for build_feature_vector)
    home_mf = get_major_form(df, team_a)
    away_mf = get_major_form(df, team_b)
    home_form.update(home_mf)
    away_form.update(away_mf)

    X = build_feature_vector(home_form, away_form, h2h, res_features,
                             is_neutral=is_neutral,
                             tournament_importance=importance)

    # Predictions
    res_proba   = result_model.predict_proba(X)[0]   # [away_win, draw, home_win]
    goals_proba = goals_model.predict_proba(X)[0]    # [under, over]

    p_home_win  = res_proba[2]
    p_draw      = res_proba[1]
    p_away_win  = res_proba[0]
    p_over      = goals_proba[1]
    p_under     = goals_proba[0]

    # Expected goals (rough estimate from averages)
    exp_home_goals = (home_form['goals_scored'] + away_form['goals_conceded']) / 2
    exp_away_goals = (away_form['goals_scored'] + home_form['goals_conceded']) / 2
    exp_total      = exp_home_goals + exp_away_goals

    # -- Output ---------------------------------------------------------------
    sep = '=' * 58
    print(f'\n{sep}')
    print(f'  MATCH PREDICTION')
    print(f'  {team_a}  vs  {team_b}')
    print(f'  Tournament : {tournament}')
    print(f'  Venue      : {"Neutral" if is_neutral else f"{team_a} home"}')
    print(sep)

    # Recent form
    form_a = print_recent_form(df, team_a)
    form_b = print_recent_form(df, team_b)
    print(f'\n  Recent form (last {FORM_WINDOW}, oldest->newest)')
    print(f'  {team_a:<30} {form_a}')
    print(f'  {team_b:<30} {form_b}')

    # H2H
    h2h_lines = print_h2h_summary(df, team_a, team_b, n=5)
    if h2h_lines:
        print(f'\n  Last {min(5, len(h2h_lines))} H2H meetings:')
        for ln in h2h_lines:
            print(ln)
    else:
        print('\n  No previous H2H meetings found.')

    # Result prediction
    print(f'\n  -- RESULT PREDICTION --')
    winner = (team_a if p_home_win > p_away_win and p_home_win > p_draw
              else (team_b if p_away_win > p_home_win and p_away_win > p_draw
                    else 'Draw'))
    top_prob = max(p_home_win, p_draw, p_away_win)

    print(f'  {team_a} Win  : {p_home_win:6.1%}   {confidence_label(p_home_win) if winner==team_a else ""}')
    print(f'  Draw         : {p_draw:6.1%}   {confidence_label(p_draw) if winner=="Draw" else ""}')
    print(f'  {team_b} Win  : {p_away_win:6.1%}   {confidence_label(p_away_win) if winner==team_b else ""}')
    print(f'\n  Most likely: {winner}  ({top_prob:.1%} confidence)')

    # Goals prediction
    print(f'\n  -- GOALS PREDICTION --')
    goals_call = 'Over 2.5' if p_over >= 0.50 else 'Under 2.5'
    print(f'  Over 2.5     : {p_over:6.1%}   {confidence_label(p_over) if goals_call=="Over 2.5" else ""}')
    print(f'  Under 2.5    : {p_under:6.1%}   {confidence_label(p_under) if goals_call=="Under 2.5" else ""}')
    print(f'  Expected goals: {team_a} {exp_home_goals:.1f} vs {exp_away_goals:.1f} {team_b}  '
          f'(total {exp_total:.1f})')
    print(f'  Most likely: {goals_call}  ({max(p_over, p_under):.1%} confidence)')

    print(f'\n{sep}\n')

    # Betting summary
    print('  MARKET SUMMARY')
    print(f'  Match Result  -> {winner} ({top_prob:.1%})')
    print(f'  Goals O/U 2.5 -> {goals_call} ({max(p_over, p_under):.1%})')
    both_teams = (home_form['goals_scored'] > 0.8 and away_form['goals_scored'] > 0.8)
    btts = 'Yes' if both_teams else 'No'
    print(f'  Both Teams Score (estimate) -> {btts}')
    print(f'{sep}\n')


def main():
    if '--list' in sys.argv:
        predict('--list', '')
        return

    if len(sys.argv) >= 3:
        team_a = sys.argv[1]
        team_b = sys.argv[2]
        neutral = '--home' not in sys.argv
        tournament = 'FIFA World Cup'
        for arg in sys.argv:
            if arg.startswith('--tournament='):
                tournament = arg.split('=', 1)[1]
        predict(team_a, team_b, is_neutral=neutral, tournament=tournament)
    else:
        print('World Cup Match Predictor')
        print('-' * 30)
        team_a = input('Enter Team 1 (home / listed first): ').strip()
        team_b = input('Enter Team 2 (away / listed second): ').strip()
        neutral_in = input('Neutral venue? [Y/n]: ').strip().lower()
        is_neutral = neutral_in != 'n'
        predict(team_a, team_b, is_neutral=is_neutral)


if __name__ == '__main__':
    main()

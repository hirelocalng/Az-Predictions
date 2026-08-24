"""
compute_bball_bestbet_baselines.py — empirical NBA/WNBA Best Bet baselines.

Companion to compute_bestbet_baselines.py (football). Same problem, same
fix: nba_predict.py's best-bet selection compared raw win_pct vs raw
ou_pct with no normalisation, and O/U's mean confidence runs several
points higher than Result's for both sports, so it won every live
matchup regardless of which market actually carried more genuine edge
(2026-08-24 audit: 182/182 WNBA permutations picked O/U before this fix).

Computes mean/std of each market's max-class confidence by running the
actual trained result/O-U models over every current team pairing (uses
cached form data, no live API calls needed -- works even in the NBA
off-season). Prints the values to paste into nba_predict.py's
_BESTBET_BBALL_BASELINES; not wired to read a JSON file at runtime since
the team rosters here are hardcoded and reviewed by hand each time this
is re-run (unlike football's, which reads bestbet_baselines.json).
"""

import itertools
import statistics

import nba_predict as N

WNBA_TEAMS = [
    'Golden State Valkyries', 'Minnesota Lynx', 'Los Angeles Sparks', 'Atlanta Dream',
    'Dallas Wings', 'Seattle Storm', 'Chicago Sky', 'Indiana Fever', 'Portland Fire',
    'Washington Mystics', 'Toronto Tempo', 'Las Vegas Aces', 'Connecticut Sun', 'Phoenix Mercury',
]

NBA_TEAMS = [
    'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
    'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets',
    'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers',
    'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
    'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks',
    'Oklahoma City Thunder', 'Orlando Magic', 'Phoenix Suns', 'Portland Trail Blazers',
    'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz', 'Washington Wizards',
]


def _baseline(pairs, predict_fn):
    win_pcts, ou_pcts = [], []
    for home, away in pairs:
        try:
            pred = predict_fn(home, away)
        except Exception:
            continue
        if not pred or 'home_win_pct' not in pred:
            continue
        win_pcts.append(max(pred['home_win_pct'], pred['away_win_pct']))
        ou_pcts.append(max(pred['over_pct'], pred['under_pct']))
    return win_pcts, ou_pcts


def main():
    print("WNBA (all team permutations):")
    win, ou = _baseline(itertools.permutations(WNBA_TEAMS, 2), N.predict_wnba)
    print(f"  n={len(win)}  result: mean={statistics.mean(win):.2f} std={statistics.pstdev(win):.2f}")
    print(f"         ou:     mean={statistics.mean(ou):.2f} std={statistics.pstdev(ou):.2f}")

    print("\nNBA (every 3rd team as home x all away, for speed):")
    pairs = [(h, a) for h in NBA_TEAMS[::3] for a in NBA_TEAMS if a != h]
    win, ou = _baseline(pairs, N.predict_nba)
    print(f"  n={len(win)}  result: mean={statistics.mean(win):.2f} std={statistics.pstdev(win):.2f}")
    print(f"         ou:     mean={statistics.mean(ou):.2f} std={statistics.pstdev(ou):.2f}")


if __name__ == "__main__":
    main()

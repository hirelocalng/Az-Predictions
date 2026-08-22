"""
epl_data.py — clean, leakage-safe EPL match-level dataset builder.

Built for the 2026-27 pre-season retrain (train.py v6+). Supersedes reading
HomeElo/AwayElo/Form3Home/Form3Away/Form5Home/Form5Away directly from
Matches.csv: those columns are 100% missing for the entire 2025-26 season and
mixing precomputed-vintage columns with self-computed ones across seasons
would introduce a distribution shift the model could learn as signal. Instead
elo_diff/elo_prob_h/form5_h/form5_a/form5_diff are derived here, match by
match, from full E0 history in one consistent pass, 2000-01 onward.

Data source: data/Matches.csv filtered to Division=='E0' only -- confirmed in
the Phase 1 inventory as the sole clean, deduplicated, current (through
2025-26) EPL match source. The 21 legacy per-season CSVs at data/ root are
NOT used: 2000-2001.csv is mislabeled (actually the 2001-02 season),
2001-2002.csv and 2002-2003.csv are byte-identical duplicates of the real
2002-03 season, 2003-2004.csv is malformed CSV, and 2020-2021.csv is
truncated at 184/380 games.
"""
import os

import numpy as np
import pandas as pd

# Canonical team-name merge. Confirmed via a cross-file census (root season
# CSVs, epl_final.csv, premier-league-matches.csv, Matches.csv E0,
# EloRatings.csv) that "Nott'm Forest" / "Nottm Forest" is the ONLY genuine
# same-club split reaching Matches.csv's E0 subset -- every other apparent
# variant (Man United/Manchester Utd, Leeds/Leeds United, etc.) lives only in
# the legacy files that aren't used for training. former_names.csv was
# checked and is national-team/country rename data (Dahomey->Benin etc.),
# not applicable to club names.
TEAM_NAME_MAP = {
    "Nottm Forest": "Nott'm Forest",
}

ELO_K = 20.0
ELO_HOME_ADV = 100.0
FORM_WINDOW = 5
STALE_ABSENCE_DAYS = 400   # gap since a team's last E0 match -> treat as a promotion/return
ELO_DECAY_CAP_YEARS = 10.0  # years absent at which decay toward league-avg-promoted is complete


def _goal_diff_multiplier(gd: int) -> float:
    """eloratings.net-style margin-of-victory multiplier."""
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def load_epl_matches(data_dir: str = "data") -> pd.DataFrame:
    """Matches.csv, Division=='E0' only, canonical team names, chronological.
    Does NOT include HomeElo/AwayElo/Form3*/Form5* -- see compute_elo/compute_form."""
    path = os.path.join(data_dir, "Matches.csv")
    raw = pd.read_csv(path, low_memory=False)
    epl = raw[raw["Division"] == "E0"].copy()
    epl["MatchDate"] = pd.to_datetime(epl["MatchDate"], errors="coerce")
    epl["HomeTeam"] = epl["HomeTeam"].astype(str).str.strip().map(lambda t: TEAM_NAME_MAP.get(t, t))
    epl["AwayTeam"] = epl["AwayTeam"].astype(str).str.strip().map(lambda t: TEAM_NAME_MAP.get(t, t))
    epl = epl[epl["FTResult"].isin(["H", "D", "A"])].copy()
    epl["FTHome"] = pd.to_numeric(epl["FTHome"], errors="coerce")
    epl["FTAway"] = pd.to_numeric(epl["FTAway"], errors="coerce")
    epl.dropna(subset=["MatchDate", "FTHome", "FTAway"], inplace=True)
    epl.sort_values("MatchDate", inplace=True, kind="mergesort")  # stable: preserves same-day order
    epl.reset_index(drop=True, inplace=True)
    return epl


def _seed_ratings(epl_df: pd.DataFrame, elo_eng: pd.DataFrame) -> dict:
    first_seen = {}
    for r in epl_df.itertuples():
        for team in (r.HomeTeam, r.AwayTeam):
            if team not in first_seen or r.MatchDate < first_seen[team]:
                first_seen[team] = r.MatchDate

    ratings = {}
    missing_seed = []
    for team, fdate in first_seen.items():
        sub = elo_eng[(elo_eng["club"] == team) & (elo_eng["date"] <= fdate)]
        if sub.empty:
            ratings[team] = 1500.0
            missing_seed.append(team)
        else:
            ratings[team] = float(sub.sort_values("date").iloc[-1]["elo"])
    if missing_seed:
        print(f"  [compute_elo] WARNING: no EloRatings.csv seed for {missing_seed} -- defaulted to 1500.0")
    return ratings


def _run_elo_pass(epl_df: pd.DataFrame, ratings: dict, reentry_override=None):
    """
    One full chronological Elo pass. `reentry_override(team, frozen_elo,
    years_absent) -> new_elo` is called whenever a team returns after more
    than STALE_ABSENCE_DAYS since its last E0 appearance, letting the caller
    replace the otherwise-frozen rating (or return frozen_elo unchanged: the
    default). Returns (elo_home_pre, elo_away_pre, reentry_events) where
    reentry_events is [(row_idx, team, side, frozen_elo, years_absent), ...].
    """
    ratings = dict(ratings)
    last_seen = {}
    n = len(epl_df)
    elo_home_pre = np.empty(n)
    elo_away_pre = np.empty(n)
    events = []

    for i, r in enumerate(epl_df.itertuples()):
        h, a = r.HomeTeam, r.AwayTeam
        for team, side in ((h, "home"), (a, "away")):
            prev = last_seen.get(team)
            if prev is not None:
                gap_days = (r.MatchDate - prev).days
                if gap_days > STALE_ABSENCE_DAYS:
                    years_absent = gap_days / 365.25
                    frozen = ratings[team]
                    events.append((i, team, side, frozen, years_absent))
                    if reentry_override is not None:
                        ratings[team] = reentry_override(team, frozen, years_absent)

        rh, ra = ratings[h], ratings[a]
        elo_home_pre[i] = rh
        elo_away_pre[i] = ra

        exp_home = 1.0 / (1.0 + 10.0 ** (-((rh + ELO_HOME_ADV) - ra) / 400.0))
        hg, ag = r.FTHome, r.FTAway
        actual_home = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        gd = abs(int(hg) - int(ag))
        k_eff = ELO_K * _goal_diff_multiplier(gd)
        delta = k_eff * (actual_home - exp_home)

        ratings[h] = rh + delta
        ratings[a] = ra - delta
        last_seen[h] = r.MatchDate
        last_seen[a] = r.MatchDate

    return elo_home_pre, elo_away_pre, events


def compute_elo(epl_df: pd.DataFrame, data_dir: str = "data") -> pd.DataFrame:
    """
    Match-by-match Elo. Each team is seeded EXACTLY ONCE, from EloRatings.csv,
    at the nearest available date at-or-before that team's own first E0
    appearance (not a single global date -- most clubs weren't top-flight in
    2000).

    On RETURN after an absence of more than STALE_ABSENCE_DAYS (a promotion
    or a comeback after relegation), the otherwise-frozen rating is decayed
    toward the league-average promoted-side Elo, proportional to years
    absent (fully replaced by ELO_DECAY_CAP_YEARS). This beat both plain
    "frozen" and "always reset to league average" on log loss over every
    promoted team's first 10 top-flight matches since 2000 (0.5858 vs 0.5894
    / 0.5862 -- decision recorded during the 2026-27 pre-season retrain).
    The league average is computed from this dataset's own re-entry events
    in a first frozen-policy pass, not hardcoded.

    Adds elo_home_pre / elo_away_pre: the PRE-match rating for each side --
    i.e. computed from all matches strictly before this row (leak-safe).
    """
    elo_raw = pd.read_csv(os.path.join(data_dir, "EloRatings.csv"))
    elo_raw["date"] = pd.to_datetime(elo_raw["date"], errors="coerce")
    elo_eng = elo_raw[elo_raw["country"] == "ENG"].copy()
    elo_eng["club"] = elo_eng["club"].astype(str).str.strip().map(lambda t: TEAM_NAME_MAP.get(t, t))

    ratings = _seed_ratings(epl_df, elo_eng)

    # Pass 1: frozen policy, purely to learn the league-average promoted-side
    # Elo from this dataset's own history.
    _, _, events = _run_elo_pass(epl_df, ratings)
    league_avg_promoted_elo = float(np.mean([frozen for _, _, _, frozen, _ in events]))

    # Pass 2: real pass, decaying toward that average on every re-entry.
    def _decay(team, frozen, years_absent):
        w = min(1.0, years_absent / ELO_DECAY_CAP_YEARS)
        return frozen + w * (league_avg_promoted_elo - frozen)

    elo_home_pre, elo_away_pre, _ = _run_elo_pass(epl_df, ratings, reentry_override=_decay)

    out = epl_df.copy()
    out["elo_home_pre"] = elo_home_pre
    out["elo_away_pre"] = elo_away_pre
    return out


def compute_form(epl_df: pd.DataFrame, n: int = FORM_WINDOW) -> pd.DataFrame:
    """
    Points earned in each team's last n games (any venue), shift(1) applied
    before the rolling window so the current match is excluded -- leak-safe.
    Summed (0..3n), matching the scale/semantics of Matches.csv's own
    Form5Home/Form5Away (points, not a per-game rate) for a fair correlation
    check.
    """
    df = epl_df.sort_values("MatchDate", kind="mergesort").reset_index(drop=True)
    home_v = pd.DataFrame({
        "date": df["MatchDate"], "team": df["HomeTeam"],
        "pts": df["FTResult"].map({"H": 3.0, "D": 1.0, "A": 0.0}),
        "match_idx": df.index, "side": "home",
    })
    away_v = pd.DataFrame({
        "date": df["MatchDate"], "team": df["AwayTeam"],
        "pts": df["FTResult"].map({"H": 0.0, "D": 1.0, "A": 3.0}),
        "match_idx": df.index, "side": "away",
    })
    all_g = (pd.concat([home_v, away_v], ignore_index=True)
               .sort_values(["team", "date"], kind="mergesort"))
    all_g["form_n"] = (
        all_g.groupby("team", sort=False)["pts"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum())
    )
    h_form = all_g[all_g["side"] == "home"].set_index("match_idx")["form_n"]
    a_form = all_g[all_g["side"] == "away"].set_index("match_idx")["form_n"]

    out = epl_df.copy()
    out["form_n_home"] = h_form.reindex(out.index)
    out["form_n_away"] = a_form.reindex(out.index)
    return out


def build_epl_dataset(data_dir: str = "data") -> pd.DataFrame:
    df = load_epl_matches(data_dir)
    df = compute_elo(df, data_dir)
    df = compute_form(df, FORM_WINDOW)
    return df


if __name__ == "__main__":
    d = build_epl_dataset()
    print(f"Built {len(d):,} E0 matches, {d['MatchDate'].min().date()} .. {d['MatchDate'].max().date()}")
    print(d[["MatchDate", "HomeTeam", "AwayTeam", "elo_home_pre", "elo_away_pre",
              "form_n_home", "form_n_away"]].tail(10).to_string())

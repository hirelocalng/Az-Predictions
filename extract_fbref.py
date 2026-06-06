"""
extract_fbref.py
Reads the saved FBref HTML pages from data/ and extracts match results.

Output CSVs (schema matches the common loader used by train.py):
  data/brazil_2024.csv
  data/brazil_2025.csv
  data/argentina_2025.csv

Columns extracted:
  date, home_team, away_team, home_goals, away_goals, result,
  attendance, venue, referee, league, season

NOTE: xG and shots are rendered by JavaScript on FBref pages.
They are NOT present in a browser-saved HTML file and cannot be
extracted here. The CSVs will contain NaN for those columns so
that train.py can still benefit from the date/team/score data.

Usage:
    python extract_fbref.py
"""

import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


DATA_DIR = "data"

PAGES = [
    {
        "html":   "2024 Série A Scores & Fixtures _ FBref.com.html",
        "out":    "brazil_2024.csv",
        "league": "BRA_fbref",
        "season": 2024,
    },
    {
        "html":   "2025 Série A Scores & Fixtures _ FBref.com.html",
        "out":    "brazil_2025.csv",
        "league": "BRA_fbref",
        "season": 2025,
    },
    {
        "html":   "2025 Liga Profesional Argentina Scores & Fixtures _ FBref.com.html",
        "out":    "argentina_2025.csv",
        "league": "ARG_fbref",
        "season": 2025,
    },
]


def _parse_score(score_text: str):
    """
    FBref score strings look like '2–1' (en dash) or '0–0'.
    Extract the two goal counts regardless of the separator character.
    Returns (home_goals, away_goals) as ints, or (None, None) if unparseable.
    """
    nums = re.findall(r"\d+", score_text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return None, None


def _clean_str(s: str) -> str:
    """Collapse whitespace and strip; handles bytes replaced with �."""
    return re.sub(r"\s+", " ", s).strip()


def extract_page(config: dict) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, config["html"])
    with open(path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")

    # FBref fixtures table always has id="sched_<year>_<league_id>_<n>"
    table = soup.find("table", id=lambda x: x and x.startswith("sched_"))
    if table is None:
        raise ValueError(f"No sched_* table found in {path}")

    tbody = table.find("tbody")
    if tbody is None:
        raise ValueError(f"Table has no <tbody> in {path}")

    records = []
    for row in tbody.find_all("tr"):
        cells = {
            c.get("data-stat", ""): _clean_str(c.get_text())
            for c in row.find_all(["th", "td"])
        }

        date_val = cells.get("date", "")
        home     = cells.get("home_team", "")
        score    = cells.get("score", "")
        away     = cells.get("away_team", "")

        # Skip header repeat rows, spacer rows, and future fixtures
        if not date_val or not home or not away:
            continue
        if not score or re.search(r"[A-Za-z]", score):
            # Unplayed matches show empty or a link text like "Match Report"
            continue

        hg, ag = _parse_score(score)
        if hg is None:
            continue

        result = "H" if hg > ag else ("A" if ag > hg else "D")

        att_raw = cells.get("attendance", "").replace(",", "").replace(".", "")
        attendance = int(att_raw) if att_raw.isdigit() else np.nan

        records.append({
            "date":       date_val,
            "home_team":  home,
            "away_team":  away,
            "home_goals": hg,
            "away_goals": ag,
            "result":     result,
            "home_sot":   np.nan,   # not in saved HTML (JS-rendered)
            "away_sot":   np.nan,
            "home_xg":    np.nan,
            "away_xg":    np.nan,
            "home_corners": np.nan,
            "away_corners": np.nan,
            "attendance": attendance,
            "venue":      cells.get("venue", ""),
            "referee":    cells.get("referee", ""),
            "league":     config["league"],
            "season":     config["season"],
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = (df.dropna(subset=["date", "home_team", "away_team"])
            .sort_values("date")
            .reset_index(drop=True))
    return df


def main():
    print("\n" + "=" * 56)
    print("  FBref HTML Extractor")
    print("=" * 56)
    print("\n  NOTE: xG / shots are JavaScript-rendered on FBref and")
    print("  are NOT present in browser-saved HTML pages. Those")
    print("  columns are written as NaN (XGBoost handles them fine).\n")

    for config in PAGES:
        out_path = os.path.join(DATA_DIR, config["out"])
        try:
            df = extract_page(config)
            df.to_csv(out_path, index=False)
            date_min = df["date"].min().date()
            date_max = df["date"].max().date()
            h = (df["result"] == "H").sum()
            d = (df["result"] == "D").sum()
            a = (df["result"] == "A").sum()
            print(f"  {config['out']:<22} {len(df):>4} matches  "
                  f"{date_min} – {date_max}  "
                  f"H={h} D={d} A={a}")
            print(df[["date", "home_team", "home_goals", "away_goals", "away_team"]]
                  .head(3).to_string(index=False))
            print()
        except Exception as exc:
            print(f"  ERROR  {config['html']}: {exc}")

    print("=" * 56)
    print("  Done. CSVs written to data/")


if __name__ == "__main__":
    main()

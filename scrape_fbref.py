"""
Scrape Brazil Serie A and Argentina Primera Division match data from FBref.
Downloads Scores & Fixtures pages (includes xG, shots, match stats) for seasons 2018-2025.
Saves each season as data/brazil_YEAR.csv or data/argentina_YEAR.csv.

HOW IT WORKS
------------
FBref uses Cloudflare bot protection that blocks all headless browsers.
This script opens a VISIBLE Chrome window once, waits for you to pass the
challenge (usually auto-resolves in 5-10s; if a CAPTCHA appears, solve it),
then harvests the session cookies and reuses them for every subsequent page
silently in the background.

Run:  python scrape_fbref.py
Deps: pip install playwright beautifulsoup4 pandas
      python -m playwright install chromium
"""

import json
import os
import time
import re

import pandas as pd
from bs4 import BeautifulSoup, Comment
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

LEAGUES = [
    {"name": "brazil",    "comp_id": 24, "url_slug": "Serie-A-Scores-and-Fixtures"},
    {"name": "argentina", "comp_id": 21, "url_slug": "Primera-Division-Scores-and-Fixtures"},
]

SEASONS   = range(2018, 2026)
DATA_DIR  = "data"
COOKIE_FILE = "fbref_cookies.json"
DELAY_SECONDS = 6


def build_url(comp_id: int, year: int, url_slug: str) -> str:
    return (
        f"https://fbref.com/en/comps/{comp_id}/{year}/schedule/"
        f"{year}-{url_slug}"
    )


def page_is_ready(page) -> bool:
    """Return True once the Cloudflare challenge has cleared."""
    return "just a moment" not in page.title().lower()


def bootstrap_cookies(pw) -> list[dict]:
    """
    Open a visible browser window, navigate to FBref, wait for the
    Cloudflare challenge to clear (auto or manual), and return cookies.
    """
    print("\n[BOOTSTRAP] Opening visible Chrome window to solve Cloudflare challenge...")
    print("  -> A browser window will appear. If a CAPTCHA shows, solve it.")
    print("  -> The window closes automatically once the page loads.\n")

    browser = pw.chromium.launch(
        headless=False,
        args=["--window-size=1280,800", "--disable-blink-features=AutomationControlled"],
    )
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        viewport={"width": 1280, "height": 800},
    )

    # Patch navigator.webdriver before any navigation
    context.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    page = context.new_page()
    page.goto("https://fbref.com/", wait_until="domcontentloaded", timeout=30000)

    print("  Waiting up to 120s for challenge to clear (solve any CAPTCHA in the window)...")
    deadline = time.time() + 120
    cleared = False
    while time.time() < deadline:
        time.sleep(2)
        elapsed = int(time.time() - (deadline - 120))
        print(f"  ...{elapsed}s", end="\r", flush=True)
        # Check for cf_clearance cookie which Cloudflare sets after passing
        cookie_names = {c["name"] for c in context.cookies()}
        if "cf_clearance" in cookie_names or page_is_ready(page):
            cleared = True
            print(f"\n  Challenge cleared!")
            break

    if not cleared:
        print("\n  Challenge did not clear in 120s.")
        print("  If you see a CAPTCHA in the browser, please solve it now.")
        print("  Waiting an extra 60s...")
        time.sleep(60)
        cookie_names = {c["name"] for c in context.cookies()}
        if not ("cf_clearance" in cookie_names or page_is_ready(page)):
            print("  Still blocked. Close the browser window and re-run the script.")
            browser.close()
            raise SystemExit(1)

    cookies = context.cookies()
    browser.close()
    print(f"  Captured {len(cookies)} cookies.")
    return cookies


def expand_comments(soup: BeautifulSoup) -> BeautifulSoup:
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        inner = BeautifulSoup(comment, "html.parser")
        if inner.find(True):
            comment.replace_with(inner)
    return BeautifulSoup(str(soup), "html.parser")


def find_fixtures_table(html: str, comp_id: int, year: int) -> pd.DataFrame | None:
    soup = expand_comments(BeautifulSoup(html, "html.parser"))

    table_id = f"sched_{year}_{comp_id}_1"
    table = soup.find("table", {"id": table_id})
    if table is None:
        table = soup.find("table", id=lambda x: x and x.startswith("sched"))

    if table is None:
        print(f"  Could not find fixtures table (tried id='{table_id}')")
        # Debug: list all table ids on the page
        all_ids = [t.get("id", "") for t in soup.find_all("table") if t.get("id")]
        if all_ids:
            print(f"  Tables found on page: {all_ids[:8]}")
        return None

    df = pd.read_html(str(table))[0]
    df.dropna(how="all", inplace=True)
    if "Wk" in df.columns:
        df = df[df["Wk"] != "Wk"]
    df.reset_index(drop=True, inplace=True)
    return df


def scrape_all() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    total = len(LEAGUES) * len(list(SEASONS))
    done  = 0

    with sync_playwright() as pw:
        # ── Step 1: bootstrap cookies with a visible window ──────────────
        if os.path.exists(COOKIE_FILE):
            print(f"Reusing saved cookies from {COOKIE_FILE}")
            with open(COOKIE_FILE) as f:
                cookies = json.load(f)
        else:
            cookies = bootstrap_cookies(pw)
            with open(COOKIE_FILE, "w") as f:
                json.dump(cookies, f)
            print(f"  Cookies saved to {COOKIE_FILE} for future runs.\n")

        # ── Step 2: headless browser with injected cookies ────────────────
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        context.add_cookies(cookies)
        page = context.new_page()

        for league in LEAGUES:
            print(f"\n=== {league['name'].upper()} ===")
            for year in SEASONS:
                done += 1
                out_path = os.path.join(DATA_DIR, f"{league['name']}_{year}.csv")
                print(f"[{done}/{total}] {league['name']} {year}", end="", flush=True)

                if os.path.exists(out_path):
                    print(" — already exists, skipping")
                    continue

                url = build_url(league["comp_id"], year, league["url_slug"])
                print(f"\n  Fetching: {url}")

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                except PWTimeout:
                    print("  Timeout, skipping.")
                    continue

                # If challenge reappears wait a bit and retry
                if not page_is_ready(page):
                    print("  Challenge page detected, waiting 10s...")
                    time.sleep(10)

                if not page_is_ready(page):
                    print("  Still blocked — delete fbref_cookies.json and re-run to re-bootstrap.")
                    continue

                html = page.content()
                df = find_fixtures_table(html, league["comp_id"], year)

                if df is None or df.empty:
                    print(f"  No data parsed.")
                else:
                    df.to_csv(out_path, index=False, encoding="utf-8-sig")
                    print(f"  Saved {len(df)} rows -> {out_path}")

                if done < total:
                    time.sleep(DELAY_SECONDS)

        browser.close()

    print("\nDone.")


if __name__ == "__main__":
    scrape_all()

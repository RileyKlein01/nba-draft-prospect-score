"""
All-Star / All-NBA / All-Defensive model: Step 1 (player list + outcomes)

Builds one row per drafted college player from the 2008 to 2019 drafts, with
three outcome columns (All-Star, All-NBA, All-Defensive) and a
years-to-first-selection calculation for each.

Source: Basketball Reference (draft pages, All-Star rosters, award pages).
Sports Reference limits scrapers to about 20 requests per minute, so the
script waits between requests. Total runtime is roughly 2 to 3 minutes.

Setup:
    pip install requests beautifulsoup4 lxml pandas

Run:
    python allstar_step1.py

Outputs:
    players_outcomes.csv   one row per player, ready to join college stats onto
    years_to_first.csv     one row per player per award, for the timing check
"""

import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DRAFT_YEARS = range(2008, 2020)          # 2008 through 2019 drafts
LAST_SEASON = 2026                       # 2025-26 is the latest complete season
FIRST_SEASON = min(DRAFT_YEARS) + 1      # first season any sample player could play
BASE = "https://www.basketball-reference.com"
SLEEP_SECONDS = 3.5                      # stays under the 20 requests/min limit
HEADERS = {"User-Agent": "Mozilla/5.0 (student research project)"}

# Draft entries that are not college players. The draft page leaves the
# college cell blank for internationals and high schoolers; this catches the
# rest (G League Ignite, Overtime Elite, etc.).
NON_COLLEGE_PATTERNS = re.compile(r"ignite|overtime elite|g league", re.I)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_soup(url):
    """Fetch a page politely and return a BeautifulSoup that also includes
    tables Basketball Reference hides inside HTML comments."""
    time.sleep(SLEEP_SECONDS)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            c.replace_with(BeautifulSoup(c, "lxml"))
    return soup


def player_id_from_href(href):
    """'/players/w/willizi01.html' -> 'willizi01'"""
    m = re.search(r"/players/\w/([\w.]+)\.html", href or "")
    return m.group(1) if m else None


def season_end_year(text):
    """'2023-24' -> 2024"""
    m = re.match(r"(\d{4})-(\d{2})", text.strip())
    if not m:
        return None
    start = int(m.group(1))
    return start + 1


def cell(row, stat):
    tag = row.find(["td", "th"], {"data-stat": stat})
    return tag.get_text(strip=True) if tag else ""


# ---------------------------------------------------------------------------
# 1. Draft pages -> college player list
# ---------------------------------------------------------------------------
def scrape_drafts():
    rows = []
    for year in DRAFT_YEARS:
        print(f"Draft {year}")
        soup = get_soup(f"{BASE}/draft/NBA_{year}.html")
        table = soup.find("table", id="stats")
        for tr in table.find("tbody").find_all("tr"):
            if "thead" in (tr.get("class") or []):
                continue
            player_td = tr.find("td", {"data-stat": "player"})
            if player_td is None:
                continue
            link = player_td.find("a")
            rows.append({
                "player_id": player_id_from_href(link["href"]) if link else None,
                "player": player_td.get_text(strip=True),
                "draft_year": year,
                "pick": pd.to_numeric(cell(tr, "pick_overall"), errors="coerce"),
                "team": cell(tr, "team_id"),
                "college": cell(tr, "college_name"),
            })
    df = pd.DataFrame(rows)

    n_all = len(df)
    df = df[df["college"].str.len() > 0]
    df = df[~df["college"].str.contains(NON_COLLEGE_PATTERNS)]
    print(f"\nDraft picks: {n_all}, college players kept: {len(df)}")

    # Picks who never played an NBA game have no player link. Keep them:
    # they are real outcomes (0 on every award), they just need an ID.
    missing = df["player_id"].isna()
    df.loc[missing, "player_id"] = (
        "nolink_" + df.loc[missing, "draft_year"].astype(str)
        + "_" + df.loc[missing, "pick"].astype("Int64").astype(str)
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Award selections -> (player_id, season) pairs
# ---------------------------------------------------------------------------
def scrape_all_star():
    """One page per All-Star game. Collects every player listed on a roster
    table. CHECK: confirm injured selectees appear on the roster tables for a
    couple of years (e.g. 2024, 2025). If they are listed only in a footnote,
    add them by hand, because selection is what we are measuring."""
    picks = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        print(f"All-Star {season}")
        try:
            soup = get_soup(f"{BASE}/allstar/NBA_{season}.html")
        except requests.HTTPError as e:
            print(f"  skipped ({e})")
            continue
        ids = set()
        for td in soup.find_all(["td", "th"], {"data-stat": "player"}):
            a = td.find("a")
            pid = player_id_from_href(a["href"]) if a else None
            if pid:
                ids.add(pid)
        picks += [{"player_id": p, "season": season} for p in ids]
    return pd.DataFrame(picks)


def scrape_award_page(path, table_id, label):
    """All-NBA and All-Defensive pages: one row per season per team (1st, 2nd,
    3rd), each listing the selected players."""
    print(label)
    soup = get_soup(f"{BASE}{path}")
    table = soup.find("table", id=table_id)
    picks = []
    for tr in table.find("tbody").find_all("tr"):
        if cell(tr, "lg_id") not in ("NBA", ""):
            continue  # drop ABA / BAA rows
        season = season_end_year(cell(tr, "season"))
        if season is None or season < FIRST_SEASON:
            continue
        team = cell(tr, "all_team")
        for a in tr.find_all("a", href=re.compile(r"^/players/")):
            picks.append({
                "player_id": player_id_from_href(a["href"]),
                "season": season,
                "team": team,
            })
    return pd.DataFrame(picks)


# ---------------------------------------------------------------------------
# 3. Combine
# ---------------------------------------------------------------------------
def summarize_award(players, picks, name):
    """Adds <name> (0/1), <name>_count, <name>_first_season,
    <name>_first_team_count (All-NBA / All-Def only)."""
    if picks.empty:
        players[name] = 0
        return players
    g = picks.groupby("player_id")
    summary = pd.DataFrame({
        f"{name}_count": g["season"].nunique(),
        f"{name}_first_season": g["season"].min(),
    })
    if "team" in picks.columns:
        first_team = picks[picks["team"].str.startswith("1")]
        summary[f"{name}_first_team_count"] = (
            first_team.groupby("player_id")["season"].nunique()
        )
    players = players.merge(summary, how="left", left_on="player_id",
                            right_index=True)
    count_cols = [c for c in players.columns
                  if c.startswith(name) and c.endswith("count")]
    players[count_cols] = players[count_cols].fillna(0).astype(int)
    players[name] = (players[f"{name}_count"] > 0).astype(int)
    return players


def main():
    players = scrape_drafts()

    all_star = scrape_all_star()
    all_nba = scrape_award_page("/awards/all_league.html",
                                "awards_all_league", "All-NBA")
    all_def = scrape_award_page("/awards/all_defense.html",
                                "awards_all_defense", "All-Defensive")

    for picks, name in [(all_star, "all_star"),
                        (all_nba, "all_nba"),
                        (all_def, "all_def")]:
        players = summarize_award(players, picks, name)

    # Seasons each player has had to earn a selection (right-censoring check)
    players["seasons_available"] = LAST_SEASON - players["draft_year"]

    players.to_csv("players_outcomes.csv", index=False)

    # Years from draft to first selection, for the timing check
    timing = []
    for name in ["all_star", "all_nba", "all_def"]:
        sub = players[players[name] == 1]
        timing.append(pd.DataFrame({
            "player_id": sub["player_id"],
            "player": sub["player"],
            "draft_year": sub["draft_year"],
            "award": name,
            "years_to_first": sub[f"{name}_first_season"] - sub["draft_year"],
        }))
    timing = pd.concat(timing, ignore_index=True)
    timing.to_csv("years_to_first.csv", index=False)

    # ---- Quick report ----
    print("\n=== Sample ===")
    print(f"College players: {len(players)}")
    for name in ["all_star", "all_nba", "all_def"]:
        n = players[name].sum()
        print(f"{name:9s} positives: {n:3d} ({n / len(players):.1%})")

    print("\n=== Years from draft to first selection ===")
    for name, sub in timing.groupby("award"):
        y = sub["years_to_first"]
        print(f"{name:9s} median {y.median():.0f} | "
              f"within 5: {(y <= 5).mean():.0%} | "
              f"within 7: {(y <= 7).mean():.0%} | "
              f"within 8: {(y <= 8).mean():.0%}")

    print("\nPositives by draft class:")
    print(players.groupby("draft_year")[["all_star", "all_nba", "all_def"]]
          .sum().to_string())


if __name__ == "__main__":
    main()

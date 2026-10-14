"""
All-Star / All-NBA / All-Defensive model: Step 2 (rebuild outcomes from player pages)

Why: the All-Star game pages are box scores, so they miss injured selectees and
(in 2025 and 2026) whole teams that weren't in the championship game. Each
player's own Basketball Reference page lists their season awards instead, so
this script rebuilds all three outcomes from there.

It also saves the link to each player's college stats page
(Sports Reference CBB), which Step 3 uses to pull college stats.

Input:  players_outcomes.csv (from Step 1; only the player list is reused)
Output: players_outcomes_v2.csv, years_to_first_v2.csv

Run:    python allstar_step2_player_pages.py

Runtime: about 35 minutes (583 pages at 3.5 seconds each, to respect the
site's rate limit). Pages are cached in a "cache" folder, so if it stops
partway, just run it again and it picks up where it left off.
"""

import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

BASE = "https://www.basketball-reference.com"
LAST_SEASON = 2026
SLEEP_SECONDS = 3.5
HEADERS = {"User-Agent": "Mozilla/5.0 (student research project)"}
CACHE_DIR = "cache"


# ---------------------------------------------------------------------------
# Fetching with cache
# ---------------------------------------------------------------------------
def fetch_player_html(player_id):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{player_id}.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    url = f"{BASE}/players/{player_id[0]}/{player_id}.html"
    for attempt in range(3):
        time.sleep(SLEEP_SECONDS)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 429:          # rate limited: back off and retry
            print("   rate limited, waiting 90 seconds")
            time.sleep(90)
            continue
        resp.raise_for_status()
        resp.encoding = "utf-8"
        with open(path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        return resp.text
    raise RuntimeError(f"Could not fetch {url}")


def make_soup(html):
    soup = BeautifulSoup(html, "lxml")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            c.replace_with(BeautifulSoup(c, "lxml"))
    return soup


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def season_end_year(text):
    m = re.search(r"(\d{4})-(\d{2})", text or "")
    return int(m.group(1)) + 1 if m else None


def parse_season_awards(soup):
    """Returns {season_end_year: set of award tokens} from the regular season
    stats table's Awards column, e.g. 'AS,NBA2,DEF1' -> {'AS','NBA2','DEF1'}."""
    tables = [t for t in soup.find_all("table")
              if t.get("id") and "playoff" not in t.get("id")
              and t.find(["td", "th"], {"data-stat": "awards"})]
    # Prefer the per game table, fall back to any regular season table
    tables.sort(key=lambda t: 0 if "per_game" in t["id"] else 1)
    if not tables:
        return None

    awards = {}
    for tr in tables[0].find_all("tr"):
        first = tr.find("th")
        season = season_end_year(first.get_text(strip=True) if first else "")
        cell = tr.find("td", {"data-stat": "awards"})
        if season is None or cell is None:
            continue
        tokens = {t.strip().upper() for t in cell.get_text(",").split(",") if t.strip()}
        awards.setdefault(season, set()).update(tokens)
    return awards


def parse_bling_counts(soup):
    """Career totals from the badges at the top of the page, used as a check.
    e.g. '11x All Star' -> 11, 'All-Defensive' (no number) -> 1."""
    counts = {"all_star": 0, "all_nba": 0, "all_def": 0}
    bling = soup.find(id="bling")
    if bling is None:
        return counts
    for li in bling.find_all("li"):
        text = li.get_text(" ", strip=True)
        m = re.match(r"(\d+)x\s+(.*)", text)
        n, label = (int(m.group(1)), m.group(2)) if m else (1, text)
        label = label.lower()
        if label.startswith("all star") or label.startswith("all-star"):
            counts["all_star"] = n
        elif label.startswith("all-nba"):
            counts["all_nba"] = n
        elif label.startswith("all-defensive"):
            counts["all_def"] = n
    return counts


def parse_cbb_link(soup):
    a = soup.find("a", href=re.compile(r"sports-reference\.com/cbb/players/"))
    return a["href"] if a else None


AWARD_RULES = {
    "all_star": lambda tok: tok == "AS",
    "all_nba": lambda tok: re.fullmatch(r"NBA[123]", tok) is not None,
    "all_def": lambda tok: re.fullmatch(r"DEF[12]", tok) is not None,
}


def summarize_player(player_id):
    soup = make_soup(fetch_player_html(player_id))
    season_awards = parse_season_awards(soup)
    bling = parse_bling_counts(soup)
    row = {"player_id": player_id, "cbb_url": parse_cbb_link(soup),
           "awards_column_found": season_awards is not None}

    for name, rule in AWARD_RULES.items():
        if season_awards is None:
            seasons = []
        else:
            seasons = sorted(s for s, toks in season_awards.items()
                             if any(rule(t) for t in toks) and s <= LAST_SEASON)
        row[f"{name}_count"] = len(seasons)
        row[f"{name}_first_season"] = seasons[0] if seasons else None
        row[f"{name}_bling_count"] = bling[name]
        if name != "all_star" and season_awards is not None:
            first_tok = "NBA1" if name == "all_nba" else "DEF1"
            row[f"{name}_first_team_count"] = sum(
                first_tok in toks for s, toks in season_awards.items())
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base_cols = ["player_id", "player", "draft_year", "pick", "team", "college"]
    players = pd.read_csv("players_outcomes.csv", encoding="utf-8")[base_cols]

    rows = []
    to_fetch = players[~players["player_id"].str.startswith("nolink_")]
    for i, pid in enumerate(to_fetch["player_id"], start=1):
        cached = os.path.exists(os.path.join(CACHE_DIR, f"{pid}.html"))
        print(f"[{i}/{len(to_fetch)}] {pid}{' (cached)' if cached else ''}")
        try:
            rows.append(summarize_player(pid))
        except Exception as e:
            print(f"   FAILED: {e}")
    awards = pd.DataFrame(rows)

    out = players.merge(awards, on="player_id", how="left")
    for name in AWARD_RULES:
        cols = [c for c in out.columns if c.startswith(name) and c.endswith("count")]
        out[cols] = out[cols].fillna(0).astype(int)
        out[name] = (out[f"{name}_count"] > 0).astype(int)
    out["seasons_available"] = LAST_SEASON - out["draft_year"]

    # utf-8-sig so Excel shows names like Vucevic and Jokic correctly
    out.to_csv("players_outcomes_v2.csv", index=False, encoding="utf-8-sig")

    timing = []
    for name in AWARD_RULES:
        sub = out[out[name] == 1]
        timing.append(pd.DataFrame({
            "player_id": sub["player_id"], "player": sub["player"],
            "draft_year": sub["draft_year"], "award": name,
            "years_to_first": sub[f"{name}_first_season"] - sub["draft_year"],
        }))
    timing = pd.concat(timing, ignore_index=True)
    timing.to_csv("years_to_first_v2.csv", index=False, encoding="utf-8-sig")

    # ---- Report ----
    print("\n=== Sample ===")
    print(f"College players: {len(out)}")
    for name in AWARD_RULES:
        n = out[name].sum()
        print(f"{name:9s} positives: {n:3d} ({n / len(out):.1%})")

    print("\n=== Years from draft to first selection ===")
    for name, sub in timing.groupby("award"):
        y = sub["years_to_first"]
        print(f"{name:9s} median {y.median():.0f} | within 5: {(y <= 5).mean():.0%}"
              f" | within 7: {(y <= 7).mean():.0%} | within 8: {(y <= 8).mean():.0%}")

    print("\n=== Checks ===")
    missing_col = (~out["awards_column_found"].fillna(False).astype(bool)).sum()
    print(f"Player pages without an Awards column: {missing_col}")
    print(f"Players with no college stats link: {out['cbb_url'].isna().sum()}")
    for name in AWARD_RULES:
        bad = out[out[f"{name}_count"] != out[f"{name}_bling_count"]]
        print(f"{name}: {len(bad)} players where season awards and career badge disagree")
        if len(bad):
            print(bad[["player", f"{name}_count", f"{name}_bling_count"]]
                  .head(10).to_string(index=False))


if __name__ == "__main__":
    main()

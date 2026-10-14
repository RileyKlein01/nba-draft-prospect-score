"""
All-Star / All-NBA / All-Defensive model: Step 3 (every college stat available)

Visits each player's Sports Reference college basketball page and collects
every stats table on it (per game, totals, per 40 minutes, per 100
possessions, advanced, and anything else listed), every season, plus bio
details: position, height, weight, birth date, hometown and RSCI recruiting
rank when listed.

Input:  players_outcomes_v2.csv (from Step 2)
Output:
    college_seasons_long.csv   one row per player per college season, all stats
    college_features.csv       one row per player: outcomes + bio + final season
                               stats (final_*) + career stats (career_*) +
                               freshman-to-final improvement (delta_*)
    college_coverage.csv       how many players have each stat, so you can see
                               which stats are missing for older draft classes

Run:    python allstar_step3_college_stats.py
Runtime: about 35 minutes. Pages are cached in "cache_cbb", so if it stops,
run it again and it resumes.
"""

import os
import re
import time
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

SLEEP_SECONDS = 3.5
HEADERS = {"User-Agent": "Mozilla/5.0 (student research project)"}
CACHE_DIR = "cache_cbb"

# Columns that describe a row rather than measure anything
ID_STATS = {"year_id", "season", "team_name_abbr", "school_name", "team_id",
            "conf_abbr", "class", "pos", "awards", "ranker"}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch(url, key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    for _ in range(3):
        time.sleep(SLEEP_SECONDS)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
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
# Parsing stats tables
# ---------------------------------------------------------------------------
def season_label(text):
    m = re.search(r"(\d{4})-(\d{2})", text or "")
    return int(m.group(1)) + 1 if m else None


def table_prefix(table_id):
    """'players_per_poss' -> 'per_poss', 'players_advanced' -> 'advanced'."""
    return re.sub(r"^players_", "", table_id)


def row_values(tr, prefix):
    vals = {}
    for td in tr.find_all(["td", "th"]):
        stat = td.get("data-stat")
        if not stat:
            continue
        text = td.get_text(strip=True)
        if stat in ("team_name_abbr", "school_name", "team_id"):
            vals["school"] = text
        elif stat == "conf_abbr":
            vals["conf"] = text
        elif stat == "class":
            vals["class"] = text
        elif stat == "pos":
            vals["pos_season"] = text
        elif stat not in ID_STATS:
            vals[f"{prefix}__{stat}"] = text
    return vals


def parse_tables(soup):
    """Returns (seasons, career): seasons is {(season, school): {stat: value}},
    career is {stat: value} from each table's Career row."""
    seasons, career = {}, {}
    for table in soup.find_all("table"):
        tid = table.get("id")
        if not tid or "game_log" in tid or tid.startswith("sr_"):
            continue
        prefix = table_prefix(tid)
        for tr in table.find_all("tr"):
            th = tr.find("th")
            label = th.get_text(strip=True) if th else ""
            if label == "Career" and not any(k.startswith(prefix + "__") for k in career):
                career.update({k: v for k, v in row_values(tr, prefix).items()
                               if "__" in k})
                continue
            season = season_label(label)
            if season is None:
                continue
            vals = row_values(tr, prefix)
            key = (season, vals.get("school", ""))
            seasons.setdefault(key, {}).update(vals)
    return seasons, career


# ---------------------------------------------------------------------------
# Parsing bio
# ---------------------------------------------------------------------------
def parse_bio(soup):
    bio = {}
    meta = soup.find(id="meta")
    text = meta.get_text(" ", strip=True) if meta else ""

    m = re.search(r"Position:\s*([A-Za-z ]+?)\s*(?:▪|\d|Shoots|Height|$)", text)
    bio["position"] = m.group(1).strip() if m else None

    m = re.search(r"(\d)-(\d{1,2})\s*,?\s*(\d{2,3})\s*lb", text)
    if m:
        bio["height_in"] = int(m.group(1)) * 12 + int(m.group(2))
        bio["weight_lb"] = int(m.group(3))
    else:
        m = re.search(r"(\d)-(\d{1,2})", text)
        bio["height_in"] = int(m.group(1)) * 12 + int(m.group(2)) if m else None
        bio["weight_lb"] = None

    birth = soup.find(attrs={"data-birth": True})
    bio["birth_date"] = birth["data-birth"] if birth else None

    m = re.search(r"RSCI Top 100:\s*(\d+)", text)
    bio["rsci_rank"] = int(m.group(1)) if m else None

    m = re.search(r"Hometown:\s*([^▪]+?)(?:\s{2,}|High School|RSCI|$)", text)
    bio["hometown"] = m.group(1).strip() if m else None
    return bio


# ---------------------------------------------------------------------------
# Build per player
# ---------------------------------------------------------------------------
def to_number(series):
    cleaned = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    num = pd.to_numeric(cleaned, errors="coerce")
    # keep numeric if most non-empty values converted
    nonempty = series.astype(str).str.strip().ne("") & series.notna()
    if nonempty.sum() and num[nonempty].notna().mean() >= 0.8:
        return num
    return series


def age_on(birth, year, month, day):
    if not birth:
        return None
    b = date.fromisoformat(birth)
    return round((date(year, month, day) - b).days / 365.25, 2)


def process_player(row):
    html = fetch(row["cbb_url"], row["player_id"])
    soup = make_soup(html)
    seasons, career = parse_tables(soup)
    bio = parse_bio(soup)

    season_rows = []
    for (season, school), vals in sorted(seasons.items()):
        r = {"player_id": row["player_id"], "season": season, "school": school}
        r.update(vals)
        season_rows.append(r)
    return bio, season_rows, career


def main():
    players = pd.read_csv("players_outcomes_v2.csv", encoding="utf-8-sig")
    players = players[players["cbb_url"].notna()].reset_index(drop=True)
    print(f"Players with a college stats link: {len(players)}")

    bios, all_seasons, careers = [], [], []
    for i, row in players.iterrows():
        cached = os.path.exists(os.path.join(CACHE_DIR, f"{row['player_id']}.html"))
        print(f"[{i + 1}/{len(players)}] {row['player']}{' (cached)' if cached else ''}")
        try:
            bio, seasons, career = process_player(row)
        except Exception as e:
            print(f"   FAILED: {e}")
            continue
        bios.append({"player_id": row["player_id"], **bio})
        all_seasons += seasons
        careers.append({"player_id": row["player_id"],
                        **{f"career_{k}": v for k, v in career.items()}})

    # ---- Long table: every season ----
    long = pd.DataFrame(all_seasons)
    long = long[long["season"].notna()]
    for c in long.columns:
        if "__" in c:
            long[c] = to_number(long[c])
    long = long.merge(players[["player_id", "draft_year"]], on="player_id")
    long = long[long["season"] <= long["draft_year"]]   # drop anything after the draft
    long = long.sort_values(["player_id", "season"])
    long.to_csv("college_seasons_long.csv", index=False, encoding="utf-8-sig")

    # ---- One row per player ----
    stat_cols = [c for c in long.columns if "__" in c]
    num_cols = [c for c in stat_cols if pd.api.types.is_numeric_dtype(long[c])]

    final = long.groupby("player_id").tail(1).set_index("player_id")
    first = long.groupby("player_id").head(1).set_index("player_id")

    final_feats = final[["school", "conf", "class", "season"] + num_cols].add_prefix("final_")
    delta = (final[num_cols] - first[num_cols]).add_prefix("delta_")
    n_seasons = long.groupby("player_id")["season"].nunique().rename("n_college_seasons")
    n_schools = long.groupby("player_id")["school"].nunique().rename("n_schools")

    career = pd.DataFrame(careers).set_index("player_id")
    for c in career.columns:
        career[c] = to_number(career[c])

    bio = pd.DataFrame(bios).set_index("player_id")

    out = (players.set_index("player_id")
           .join(bio).join(n_seasons).join(n_schools)
           .join(final_feats).join(career).join(delta))

    # Ages: at the draft (late June) and during the final college season (Feb 1)
    out["age_at_draft"] = [age_on(b, int(y), 6, 25) if isinstance(b, str) else None
                           for b, y in zip(out["birth_date"], out["draft_year"])]
    out["age_final_season"] = [age_on(b, int(s), 2, 1)
                               if isinstance(b, str) and pd.notna(s) else None
                               for b, s in zip(out["birth_date"], out["final_season"])]
    # One-season players have no improvement to measure
    delta_cols = [c for c in out.columns if c.startswith("delta_")]
    out.loc[out["n_college_seasons"] <= 1, delta_cols] = None

    out.reset_index().to_csv("college_features.csv", index=False, encoding="utf-8-sig")

    # ---- Coverage report ----
    feature_cols = [c for c in out.columns if c.startswith(("final_", "career_"))]
    cov = pd.DataFrame({
        "stat": feature_cols,
        "share_of_players_with_value": [out[c].notna().mean().round(3) for c in feature_cols],
    })
    by_class = out.groupby("draft_year")[feature_cols].apply(lambda d: d.notna().mean())
    cov["share_2008_2010_classes"] = cov["stat"].map(
        by_class.loc[by_class.index <= 2010].mean().round(3))
    cov["share_2011_2019_classes"] = cov["stat"].map(
        by_class.loc[by_class.index >= 2011].mean().round(3))
    cov.sort_values("share_of_players_with_value").to_csv(
        "college_coverage.csv", index=False, encoding="utf-8-sig")

    # ---- Report ----
    print("\n=== Done ===")
    print(f"Players: {len(out)}  |  college seasons: {len(long)}")
    print(f"Stat columns per season: {len(stat_cols)} ({len(num_cols)} numeric)")
    print(f"Columns in college_features.csv: {out.shape[1] + 1}")
    print(f"Players with no seasons parsed: {out['n_college_seasons'].isna().sum()}")
    print(f"Have birth date: {out['birth_date'].notna().mean():.0%} | "
          f"height: {out['height_in'].notna().mean():.0%} | "
          f"RSCI rank listed: {out['rsci_rank'].notna().mean():.0%}")
    print("\nTables found (stat count per table):")
    print(pd.Series([c.split("__")[0] for c in stat_cols]).value_counts().to_string())
    print("\nStats with the least coverage:")
    print(cov.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

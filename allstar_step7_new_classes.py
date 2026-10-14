"""
Step 7: data for the fresh test (2020 to 2023 draft classes)

Reuses the Step 1 and Step 2 code, so keep this file in the same folder as
allstar_step1.py and allstar_step2_player_pages.py.

It does three things:
  1. Downloads Barttorvik season files for 2019-20 through 2022-23
  2. Pulls the 2020 to 2023 draft pages from Basketball Reference (college players)
  3. Visits each player's Basketball Reference page for All-Star, All-NBA and
     All-Defensive selections through 2025-26 (pages are cached in "cache")

Run:    python allstar_step7_new_classes.py
Runtime: about 12 minutes. If it stops, run it again; it picks up where it left off.
Then attach: new_classes_outcomes.csv and the four new files in the "barttorvik" folder
(barttorvik_2020.csv through barttorvik_2023.csv).
"""

import os
import time

import pandas as pd
import requests

import allstar_step1 as s1
import allstar_step2_player_pages as s2

NEW_DRAFTS = range(2020, 2024)
BT_YEARS = range(2020, 2024)
BT_URL = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"


def download_barttorvik():
    os.makedirs("barttorvik", exist_ok=True)
    for year in BT_YEARS:
        path = os.path.join("barttorvik", f"barttorvik_{year}.csv")
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            print(f"Barttorvik {year}: already downloaded")
            continue
        resp = requests.get(BT_URL.format(year=year), headers=s1.HEADERS, timeout=60)
        with open(path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        html = resp.text.lstrip().lower().startswith(("<!doctype", "<html"))
        print(f"Barttorvik {year}: {resp.text.count(chr(10))} rows"
              f"{'  LOOKS LIKE A WEB PAGE, NOT DATA' if html else ''}")
        time.sleep(2)


def main():
    download_barttorvik()

    s1.DRAFT_YEARS = NEW_DRAFTS
    players = s1.scrape_drafts()

    rows = []
    linked = players[~players["player_id"].str.startswith("nolink_")]
    for i, pid in enumerate(linked["player_id"], start=1):
        print(f"[{i}/{len(linked)}] {pid}")
        try:
            rows.append(s2.summarize_player(pid))
        except Exception as e:
            print(f"   FAILED: {e}")
    awards = pd.DataFrame(rows)

    out = players.merge(awards, on="player_id", how="left")
    for name in ["all_star", "all_nba", "all_def"]:
        cols = [c for c in out.columns if c.startswith(name) and c.endswith("count")]
        out[cols] = out[cols].fillna(0).astype(int)
        out[name] = (out[f"{name}_count"] > 0).astype(int)
    out["seasons_available"] = s2.LAST_SEASON - out["draft_year"]
    out.to_csv("new_classes_outcomes.csv", index=False, encoding="utf-8-sig")

    print(f"\nCollege players drafted 2020-2023: {len(out)}")
    for name in ["all_star", "all_nba", "all_def"]:
        print(f"{name}: {out[name].sum()}")
    print("\nAll-Stars found:")
    print(out.loc[out["all_star"] == 1, ["player", "draft_year", "pick", "college",
                                         "all_star_count"]].to_string(index=False))
    print("\nSaved new_classes_outcomes.csv")


if __name__ == "__main__":
    main()

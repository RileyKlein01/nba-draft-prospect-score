"""
Step 8: NBA Draft Combine measurements and athletic testing, 2000 to 2023

Downloads two tables from NBA.com for every combine:
  - Measurements: height, weight, wingspan, standing reach, hand size, body fat
  - Drills: standing and max vertical, lane agility, shuttle, 3/4 court sprint, bench press

Setup:  python -m pip install nba_api
Run:    python allstar_step8_combine.py
Runtime: about 2 minutes (48 requests).
Output: combine_anthro.csv, combine_drills.csv  (attach both)

NBA.com sometimes answers slowly or not at all. The script waits and retries;
if a season still fails, it says so and keeps going. Running it again only
re-downloads the seasons that failed.
"""

import os
import time

import pandas as pd
import requests

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2000, 2024)]   # 2000-01 ... 2023-24
ENDPOINTS = {
    "anthro": "https://stats.nba.com/stats/draftcombineplayeranthro",
    "drills": "https://stats.nba.com/stats/draftcombinedrillresults",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
CACHE = "cache_combine"


def download(kind, season):
    """Uses nba_api when installed (it keeps NBA.com's required headers current),
    otherwise a plain request."""
    try:
        from nba_api.stats.endpoints import (draftcombinedrillresults,
                                             draftcombineplayeranthro)
        ep = (draftcombineplayeranthro.DraftCombinePlayerAnthro if kind == "anthro"
              else draftcombinedrillresults.DraftCombineDrillResults)
        return ep(season_year=season, timeout=45).get_data_frames()[0]
    except ImportError:
        params = {"LeagueID": "00", "SeasonYear": season}
        resp = requests.get(ENDPOINTS[kind], params=params, headers=HEADERS, timeout=45)
        resp.raise_for_status()
        rs = resp.json()["resultSets"][0]
        return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


def fetch(kind, season, attempts=3):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{kind}_{season}.json")
    if os.path.exists(path):
        return pd.read_json(path)
    for attempt in range(attempts):
        try:
            df = download(kind, season)
            df["SEASON_LABEL"] = season
            df.to_json(path)
            return df
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"   {kind} {season}: {type(e).__name__}, retrying in {wait}s")
            time.sleep(wait)
    print(f"   {kind} {season}: FAILED, skipping (run again later)")
    return None


def connection_check():
    print("Checking the connection to NBA.com with the 2019 combine...")
    df = fetch("anthro", "2019-20", attempts=2)
    if df is None:
        print("\nNBA.com is not answering. Try: python -m pip install nba_api")
        print("then run this script again. If it still fails, NBA.com may be")
        print("blocking requests right now; wait an hour or try another network.")
        raise SystemExit(1)
    print(f"Connected: {len(df)} players in the 2019 combine.\n")


def main():
    connection_check()
    for kind in ENDPOINTS:
        frames = []
        for season in SEASONS:
            df = fetch(kind, season)
            if df is not None:
                print(f"{kind} {season}: {len(df)} players")
                frames.append(df)
            time.sleep(1.5)
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(f"combine_{kind}.csv", index=False, encoding="utf-8-sig")
        print(f"Saved combine_{kind}.csv ({len(out)} rows)\n")

    a = pd.read_csv("combine_anthro.csv")
    print("Wingspan recorded:", a["WINGSPAN"].notna().sum(), "players")
    check = a[a["PLAYER_NAME"].isin(["Zion Williamson", "Anthony Davis", "Draymond Green"])]
    print(check[["PLAYER_NAME", "SEASON_LABEL", "HEIGHT_WO_SHOES", "WINGSPAN"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
Step 6: download Barttorvik season player stats, 2007-08 through 2018-19

Saves one raw CSV per season into a "barttorvik" folder. Each file has every
Division I player that season: shooting by location (rim, mid-range, dunks),
2 and 3 point splits, usage, rates, Barttorvik's BPM and defensive metrics.

Run:    python allstar_step6_barttorvik_download.py
Runtime: under a minute (12 files).
Then attach the files from the "barttorvik" folder.
"""

import os
import time

import requests

OUT = "barttorvik"
YEARS = range(2008, 2020)        # season end years: 2007-08 through 2018-19
HEADERS = {"User-Agent": "Mozilla/5.0 (student research project)"}
URL = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"


def main():
    os.makedirs(OUT, exist_ok=True)
    for year in YEARS:
        path = os.path.join(OUT, f"barttorvik_{year}.csv")
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            print(f"{year}: already downloaded")
            continue
        resp = requests.get(URL.format(year=year), headers=HEADERS, timeout=60)
        text = resp.text
        looks_like_html = text.lstrip().lower().startswith(("<!doctype", "<html"))
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        lines = text.count("\n")
        status = "LOOKS LIKE A WEB PAGE, NOT DATA" if looks_like_html else "ok"
        print(f"{year}: {resp.status_code}, {lines} rows, {len(text) // 1024} KB, {status}")
        time.sleep(2)

    print(f"\nFiles saved in the '{OUT}' folder.")
    print("If any say 'looks like a web page', paste this output back instead of the files.")


if __name__ == "__main__":
    main()

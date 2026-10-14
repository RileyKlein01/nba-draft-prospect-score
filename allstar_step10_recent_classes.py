"""
Step 10: Barttorvik seasons for the 2024, 2025 and 2026 drafts

Downloads the 2023-24, 2024-25 and 2025-26 college seasons into the
"barttorvik" folder, alongside the earlier years.

Run:    python allstar_step10_recent_classes.py
Runtime: under a minute.
Then attach barttorvik_2024.csv, barttorvik_2025.csv and barttorvik_2026.csv
from the "barttorvik" folder.
"""

import os
import time

import requests

YEARS = [2024, 2025, 2026]
URL = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"
HEADERS = {"User-Agent": "Mozilla/5.0 (student research project)"}


def main():
    os.makedirs("barttorvik", exist_ok=True)
    for year in YEARS:
        path = os.path.join("barttorvik", f"barttorvik_{year}.csv")
        resp = requests.get(URL.format(year=year), headers=HEADERS, timeout=60)
        with open(path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        html = resp.text.lstrip().lower().startswith(("<!doctype", "<html"))
        print(f"{year}: {resp.text.count(chr(10))} rows"
              f"{'  LOOKS LIKE A WEB PAGE, NOT DATA' if html else ''}")
        time.sleep(2)
    print("\nDone. Attach the three files from the 'barttorvik' folder.")


if __name__ == "__main__":
    main()

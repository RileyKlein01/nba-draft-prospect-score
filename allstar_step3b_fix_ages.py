"""
Step 3b: add birth dates and ages to college_features.csv

The college pages didn't include birth dates, but the Basketball Reference
player pages saved in the "cache" folder during Step 2 do. This reads those
saved pages (no downloading), fills in birth dates, recomputes the age
columns, and reprints the coverage report sorted correctly.

Run:    python allstar_step3b_fix_ages.py
Takes a few seconds.
"""

import os
import re
from datetime import date, datetime

import pandas as pd
from bs4 import BeautifulSoup

CACHE_DIR = "cache"   # Basketball Reference pages from Step 2


def birth_from_page(path):
    with open(path, encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find(attrs={"data-birth": True})
    if tag:
        return tag["data-birth"]
    # Fallback: "Born: March 14, 1988"
    text = soup.get_text(" ", strip=True)
    m = re.search(r"Born:\s*([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    if m:
        return datetime.strptime(m.group(1), "%B %d, %Y").date().isoformat()
    return None


def age_on(birth, year, month, day):
    if not isinstance(birth, str) or pd.isna(year):
        return None
    b = date.fromisoformat(birth)
    return round((date(int(year), month, day) - b).days / 365.25, 2)


def main():
    df = pd.read_csv("college_features.csv", encoding="utf-8-sig", low_memory=False)

    births = {}
    for pid in df["player_id"]:
        path = os.path.join(CACHE_DIR, f"{pid}.html")
        births[pid] = birth_from_page(path) if os.path.exists(path) else None

    df["birth_date"] = df["player_id"].map(births)
    df["age_at_draft"] = [age_on(b, y, 6, 25)
                          for b, y in zip(df["birth_date"], df["draft_year"])]
    df["age_final_season"] = [age_on(b, s, 2, 1)
                              for b, s in zip(df["birth_date"], df["final_season"])]

    df.to_csv("college_features.csv", index=False, encoding="utf-8-sig")

    print(f"Birth dates found: {df['birth_date'].notna().sum()} of {len(df)}")
    missing = df[df["birth_date"].isna()]
    if len(missing):
        print("Missing (mostly players who never reached the NBA):")
        print(missing[["player", "draft_year", "pick"]].to_string(index=False))

    print("\nAge at draft:")
    print(df["age_at_draft"].describe().round(2).to_string())
    print("\nAverage age at draft by outcome:")
    for name in ["all_star", "all_nba", "all_def"]:
        print(f"  {name:9s} yes {df.loc[df[name] == 1, 'age_at_draft'].mean():.2f}"
              f" | no {df.loc[df[name] == 0, 'age_at_draft'].mean():.2f}")

    # ---- Coverage report, sorted this time ----
    feature_cols = [c for c in df.columns if c.startswith(("final_", "career_"))]
    by_class = df.groupby("draft_year")[feature_cols].apply(lambda d: d.notna().mean())
    cov = pd.DataFrame({
        "stat": feature_cols,
        "share_all": [round(df[c].notna().mean(), 3) for c in feature_cols],
        "share_2008_2010": by_class.loc[by_class.index <= 2010].mean().round(3).values,
        "share_2011_2019": by_class.loc[by_class.index >= 2011].mean().round(3).values,
    }).sort_values("share_all")
    cov.to_csv("college_coverage.csv", index=False, encoding="utf-8-sig")

    print("\nStats with the least coverage:")
    print(cov.head(25).to_string(index=False))
    print(f"\nStats with 95%+ coverage: {(cov['share_all'] >= 0.95).sum()} of {len(cov)}")


if __name__ == "__main__":
    main()

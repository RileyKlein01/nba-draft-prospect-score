"""
Step 9: the 2002 to 2007 draft classes (era check)

Reuses the earlier steps, so keep this file in the project folder next to:
  allstar_step1.py, allstar_step2_player_pages.py,
  allstar_step3_college_stats.py, allstar_step3b_fix_ages.py

For every college player drafted 2002 to 2007 it collects:
  - All-Star, All-NBA and All-Defensive selections (Basketball Reference player pages)
  - Birth date and age at draft
  - Every college stats table, every season (Sports Reference college pages)

Run:    python allstar_step9_older_classes.py
Runtime: about 35 to 40 minutes. Pages are cached, so if it stops, run it again.
Output: older_classes_features.csv and older_classes_seasons.csv (attach both)
"""

import os

import pandas as pd

import allstar_step1 as s1
import allstar_step2_player_pages as s2
import allstar_step3_college_stats as s3
import allstar_step3b_fix_ages as s3b

OLD_DRAFTS = range(2002, 2008)


def main():
    # 1. Draft lists (college players only)
    s1.DRAFT_YEARS = OLD_DRAFTS
    players = s1.scrape_drafts()

    # 2. Awards and college links from Basketball Reference player pages
    rows = []
    linked = players[~players["player_id"].str.startswith("nolink_")]
    for i, pid in enumerate(linked["player_id"], start=1):
        print(f"[NBA page {i}/{len(linked)}] {pid}")
        try:
            rows.append(s2.summarize_player(pid))
        except Exception as e:
            print(f"   FAILED: {e}")
    out = players.merge(pd.DataFrame(rows), on="player_id", how="left")
    for name in ["all_star", "all_nba", "all_def"]:
        cols = [c for c in out.columns if c.startswith(name) and c.endswith("count")]
        out[cols] = out[cols].fillna(0).astype(int)
        out[name] = (out[f"{name}_count"] > 0).astype(int)

    # 3. Birth dates from the cached Basketball Reference pages
    births = {}
    for pid in out["player_id"]:
        path = os.path.join(s2.CACHE_DIR, f"{pid}.html")
        births[pid] = s3b.birth_from_page(path) if os.path.exists(path) else None
    out["birth_date"] = out["player_id"].map(births)
    out["age_at_draft"] = [s3b.age_on(b, y, 6, 25)
                           for b, y in zip(out["birth_date"], out["draft_year"])]

    # 4. College stats from Sports Reference
    have_link = out[out["cbb_url"].notna()]
    bios, seasons = [], []
    for i, (_, row) in enumerate(have_link.iterrows(), start=1):
        print(f"[college page {i}/{len(have_link)}] {row['player']}")
        try:
            bio, season_rows, _ = s3.process_player(row)
        except Exception as e:
            print(f"   FAILED: {e}")
            continue
        bio.pop("birth_date", None)   # birth dates come from the NBA pages instead
        bios.append({"player_id": row["player_id"], **bio})
        seasons += season_rows

    long = pd.DataFrame(seasons)
    for c in long.columns:
        if "__" in c:
            long[c] = s3.to_number(long[c])
    long = long.merge(out[["player_id", "draft_year"]], on="player_id")
    long = long[long["season"] <= long["draft_year"]].sort_values(["player_id", "season"])
    long.to_csv("older_classes_seasons.csv", index=False, encoding="utf-8-sig")

    num = [c for c in long.columns if "__" in c and pd.api.types.is_numeric_dtype(long[c])]
    final = long.groupby("player_id").tail(1).set_index("player_id")
    id_cols = [c for c in ["school", "conf", "class", "season"] if c in final.columns]
    final_feats = final[id_cols + num].add_prefix("final_")
    n_seasons = long.groupby("player_id")["season"].nunique().rename("n_college_seasons")

    # Career per 40 from season totals, where the totals table exists
    tot = [c for c in long.columns if c.startswith("totals__")]
    career = pd.DataFrame(index=final.index)
    if "totals__mp" in tot:
        sums = long.groupby("player_id")[tot].sum(min_count=1)
        for c in tot:
            stat = c.split("__")[1]
            if stat in ("pts", "trb", "ast", "stl", "blk", "tov", "ft", "fta", "fg", "fga",
                        "fg3", "fg3a", "orb", "drb", "pf"):
                career[f"career_per40__{stat}"] = sums[c] / sums["totals__mp"] * 40

    feats = (out.set_index("player_id")
             .join(pd.DataFrame(bios).set_index("player_id") if bios else None)
             .join(n_seasons).join(final_feats).join(career))
    feats.reset_index().to_csv("older_classes_features.csv", index=False, encoding="utf-8-sig")

    print(f"\nCollege players drafted 2002-2007: {len(out)}")
    print(f"With college stats: {feats['n_college_seasons'].notna().sum()}")
    print(f"With birth dates: {out['birth_date'].notna().sum()}")
    for name in ["all_star", "all_nba", "all_def"]:
        print(f"{name}: {out[name].sum()}")
    print("\nAll-Stars found:")
    print(out.loc[out["all_star"] == 1, ["player", "draft_year", "pick", "college"]]
          .to_string(index=False))
    print("\nSaved older_classes_features.csv and older_classes_seasons.csv")


if __name__ == "__main__":
    main()

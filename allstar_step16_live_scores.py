"""
Step 16: live prospect scores for the 2024, 2025 and 2026 draft classes

Scores every college player Barttorvik lists as drafted in 2024-2026, using
only college stats and age. Percentiles are set on the 2008-2019 classes.
Draft picks and birth dates come from Barttorvik's data.

Inputs:  model_data_2008_2019.csv (Step 11), barttorvik/ folder through 2026 (Step 10)
Output:  results/prospect_scores_2024_2026.csv
Run:     python allstar_step16_live_scores.py   (a few seconds)
"""

import warnings

import pandas as pd

from allstar_common import (RESULTS, bt_features, ensure_results, load_barttorvik,
                            percentile_of, prospect_score)

warnings.filterwarnings("ignore")
TIERS = [(90, "90th and up (36% became All-Stars, 2008-2019)"),
         (75, "75th to 90th (17%)"), (50, "50th to 75th (10%)"), (0, "below 50th (2%)")]


def tier(p):
    return next(label for cut, label in TIERS if p >= cut)


def main():
    ensure_results()
    bt = load_barttorvik()
    last_year = bt.groupby("pid")["year"].max()
    drafted = bt[bt["nba_pick"].notna()].sort_values("year").groupby("pid").tail(1)
    drafted = drafted[["pid", "name", "team", "nba_pick"]].copy()
    drafted["draft_year"] = drafted["pid"].map(last_year)   # final college season = draft year
    drafted = drafted[drafted["draft_year"].between(2024, 2026)]
    drafted["player_id"] = "bt" + drafted["pid"].astype(int).astype(str)

    d = drafted.merge(bt_features(drafted, bt), on="player_id")
    d["score"] = prospect_score(d["bt_bpm"], d["age_at_draft_bt"])

    train = pd.read_csv("model_data_2008_2019.csv", encoding="utf-8-sig", low_memory=False)
    d["percentile"] = percentile_of(d["score"], train["prospect_score"].values)
    d["dbpm_percentile"] = percentile_of(d["bt_dbpm"], train["bt_dbpm"].values)
    d["tier"] = d["percentile"].map(tier)
    d["under_300_college_minutes"] = d["college_minutes"] < 300

    out = d.rename(columns={"name": "player", "team": "college", "nba_pick": "pick",
                            "age_at_draft_bt": "age_at_draft", "bt_bpm": "college_bpm",
                            "bt_dbpm": "college_dbpm", "stl_per40": "steals_per40"})
    out = out[["draft_year", "player", "college", "pick", "age_at_draft", "college_bpm", "score",
               "percentile", "tier", "college_dbpm", "dbpm_percentile", "steals_per40",
               "college_minutes", "under_300_college_minutes"]].round(2)
    out = out.sort_values(["draft_year", "score"], ascending=[False, False])
    out.to_csv(f"{RESULTS}/prospect_scores_2024_2026.csv", index=False, encoding="utf-8-sig")

    for year in [2026, 2025, 2024]:
        x = out[out["draft_year"] == year]
        print(f"\n=== {year} draft: {len(x)} college players ===")
        print(x.head(10)[["player", "college", "pick", "age_at_draft", "college_bpm",
                          "percentile"]].to_string(index=False))
    print(f"\nSaved {RESULTS}/prospect_scores_2024_2026.csv")


if __name__ == "__main__":
    main()

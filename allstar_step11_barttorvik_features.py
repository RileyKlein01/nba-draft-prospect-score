"""
Step 11: match players to Barttorvik and build the model inputs

Inputs:  college_features.csv (Steps 3 and 3b), new_classes_outcomes.csv (Step 7),
         barttorvik/barttorvik_2008.csv ... barttorvik_2023.csv (Steps 6 and 7)
Outputs: model_data_2008_2019.csv   training players with Barttorvik inputs added
         fresh_2020_2023.csv        test players with Barttorvik inputs

Run:     python allstar_step11_barttorvik_features.py   (a few seconds)
"""

import pandas as pd

from allstar_common import (bt_features, fix_text, load_barttorvik,
                            match_to_barttorvik, prospect_score)


def build(players, bt, label):
    ids = match_to_barttorvik(players, bt)
    unmatched = players[~players["player_id"].isin(ids["player_id"])]
    print(f"{label}: matched {len(ids)} of {len(players)} players to Barttorvik")
    if len(unmatched):
        print("   not matched:", ", ".join(unmatched["player"].map(fix_text).head(10)))
    with_pid = players.merge(ids, on="player_id")
    feats = bt_features(with_pid, bt)
    out = with_pid.merge(feats, on="player_id")
    out["prospect_score"] = prospect_score(out["bt_bpm"], out["age_at_draft_bt"])
    return out


def main():
    bt = load_barttorvik(years=range(2008, 2024))

    train = pd.read_csv("college_features.csv", encoding="utf-8-sig", low_memory=False)
    train = train[train["n_college_seasons"].notna()].copy()
    train["player"] = train["player"].map(fix_text)
    build(train, bt, "2008-2019").to_csv("model_data_2008_2019.csv", index=False,
                                         encoding="utf-8-sig")

    fresh = pd.read_csv("new_classes_outcomes.csv", encoding="utf-8-sig")
    fresh["player"] = fresh["player"].map(fix_text)
    build(fresh, bt, "2020-2023").to_csv("fresh_2020_2023.csv", index=False,
                                         encoding="utf-8-sig")
    print("Saved model_data_2008_2019.csv and fresh_2020_2023.csv")


if __name__ == "__main__":
    main()

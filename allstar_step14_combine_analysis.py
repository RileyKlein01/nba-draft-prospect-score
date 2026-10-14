"""
Step 14: do NBA Draft Combine measurements add anything?

Tests wingspan, standing reach, hand size, weight, body fat, vertical leap,
lane agility, sprint and bench press against All-Star, All-NBA, All-Defensive
and RAPM, and checks whether steal rate still predicts elite defense after
accounting for length.

Inputs:  model_data_with_rapm.csv (Step 13), combine_anthro.csv and
         combine_drills.csv (Step 8)
Output:  results/combine_report.txt
Run:     python allstar_step14_combine_analysis.py   (a few seconds)
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from allstar_common import ALIASES, RESULTS, ensure_results, fix_text, name_key

warnings.filterwarnings("ignore")
lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def load_combine():
    a = pd.read_csv("combine_anthro.csv", encoding="utf-8-sig")
    d = pd.read_csv("combine_drills.csv", encoding="utf-8-sig")
    drills = ["STANDING_VERTICAL_LEAP", "MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME",
              "THREE_QUARTER_SPRINT", "BENCH_PRESS"]
    c = a.merge(d[["PLAYER_ID", "PLAYER_NAME", "SEASON_LABEL"] + drills],
                on=["PLAYER_ID", "PLAYER_NAME", "SEASON_LABEL"], how="left")
    c = c.rename(columns={
        "HEIGHT_WO_SHOES": "c_height", "WEIGHT": "c_weight", "WINGSPAN": "c_wingspan",
        "STANDING_REACH": "c_reach", "BODY_FAT_PCT": "c_bodyfat", "HAND_LENGTH": "c_hand_len",
        "HAND_WIDTH": "c_hand_w", "STANDING_VERTICAL_LEAP": "c_stand_vert",
        "MAX_VERTICAL_LEAP": "c_max_vert", "LANE_AGILITY_TIME": "c_lane",
        "THREE_QUARTER_SPRINT": "c_sprint", "BENCH_PRESS": "c_bench"})
    c["c_weight"] = pd.to_numeric(c["c_weight"], errors="coerce")
    c["c_plus_wing"] = c["c_wingspan"] - c["c_height"]
    c["combine_year"] = c["SEASON_LABEL"].str[:4].astype(int)   # "2009-10" = 2009 combine
    c["key"] = c["PLAYER_NAME"].map(name_key)
    keep = [x for x in c.columns if x.startswith("c_")] + ["key", "combine_year"]
    return c[keep].sort_values("c_wingspan").drop_duplicates(["key", "combine_year"])


def main():
    ensure_results()
    d = pd.read_csv("model_data_with_rapm.csv", encoding="utf-8-sig", low_memory=False)
    c = load_combine()
    d["key"] = d["player"].map(fix_text).map(name_key).replace(ALIASES)
    m = d[["player_id", "key", "draft_year"]].merge(c, on="key")
    m = m[m["combine_year"].between(m["draft_year"] - 1, m["draft_year"])] \
        .sort_values("combine_year", ascending=False).drop_duplicates("player_id")
    d = d.merge(m.drop(columns=["key", "draft_year", "combine_year"]), on="player_id", how="left")
    meas = d[d["c_wingspan"].notna()]
    say(f"Measured at the combine: {len(meas)} of {len(d)} players")
    for o in ["all_star", "all_nba", "all_def"]:
        say(f"   {o}: {int(meas[o].sum())} of {int(d[o].sum())} measured")
    say("All-Stars who skipped measurements: " +
        ", ".join(d[(d['all_star'] == 1) & d['c_wingspan'].isna()]['player'].map(fix_text)))

    stats = ["c_height", "c_weight", "c_wingspan", "c_reach", "c_plus_wing", "c_bodyfat",
             "c_hand_len", "c_hand_w", "c_stand_vert", "c_max_vert", "c_lane", "c_sprint", "c_bench"]
    rows = []
    for s in stats:
        ok = meas[s].notna()
        r = {"stat": s, "n": int(ok.sum())}
        for o in ["all_star", "all_nba", "all_def"]:
            r[f"{o}_auc"] = round(roc_auc_score(meas.loc[ok, o], meas.loc[ok, s]), 3)
        k = ok & meas["rapm"].notna()
        r["rapm_spearman"] = round(spearmanr(meas.loc[k, s], meas.loc[k, "rapm"])[0], 3)
        rows.append(r)
    say("\nEach measurement on its own (AUC 0.5 = no signal):")
    say(pd.DataFrame(rows).to_string(index=False))

    def logit(y, cols):
        X = meas[cols]
        X = (X - X.mean()) / X.std()
        ok = X.notna().all(axis=1)
        r = sm.Logit(meas.loc[ok, y], sm.add_constant(X[ok])).fit(disp=0)
        ci = r.conf_int()
        return pd.DataFrame({"odds_ratio": np.exp(r.params), "ci_low": np.exp(ci[0]),
                             "ci_high": np.exp(ci[1]), "p": r.pvalues}).drop("const").round(3), ok.sum()

    t, n = logit("all_def", ["age_at_draft_bt", "stl_per40", "bt_dbpm", "c_plus_wing", "c_height"])
    say(f"\nAll-Defensive: steal rate and defensive BPM vs length (n={n}):")
    say(t.to_string())

    say("\nDoes any measurement add to the prospect score for All-Stars?")
    for extra in ["c_plus_wing", "c_max_vert", "c_lane", "c_hand_len", "c_weight", "c_bodyfat"]:
        t, n = logit("all_star", ["prospect_score", extra])
        say(f"   {extra:12s} n={n:3d} odds ratio {t.loc[extra, 'odds_ratio']:.2f} p={t.loc[extra, 'p']:.3f}")

    with open(f"{RESULTS}/combine_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved {RESULTS}/combine_report.txt")


if __name__ == "__main__":
    main()

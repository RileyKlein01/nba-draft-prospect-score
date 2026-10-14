"""
Step 15: era check (2002-2007) and refit on 2002-2019

College BPM does not exist before 2008, so this builds a stand-in from basic
per-40 box score stats, fit on 2008-2019:

  Basic Prospect Score = estimated BPM + 3 x (20 - age at draft)

What it does:
  1. Fits the box-score estimate of Barttorvik BPM on 2008-2019
  2. Scores the 2002-2007 classes with those weights, untouched (out of era)
  3. Refits the age weight on all 2002-2019 players
  4. Retests the basic score on the 2020-2023 classes (per-40 stats from Barttorvik)

Inputs:  model_data_2008_2019.csv, fresh_2020_2023.csv (Step 11),
         older_classes_features.csv (Step 9), barttorvik/ folder
Output:  results/era_check_report.txt
Run:     python allstar_step15_era_check.py   (under a minute)
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

from allstar_common import RESULTS, ensure_results, fix_text, load_barttorvik, percentile_of

warnings.filterwarnings("ignore")
BOX = ["pts", "trb", "ast", "stl", "blk", "tov", "pf", "fga", "fta", "fg3a"]
lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def per40_from_sports_reference(d):
    X = pd.DataFrame({s: d[f"final_per_min__{s}_per_min"].values for s in BOX}, index=d.index)
    X["ts"] = X["pts"] / (2 * (X["fga"] + 0.44 * X["fta"]))
    return X


def per40_from_barttorvik(players, bt, tov_fit):
    s = players[["player_id", "pid", "draft_year"]].merge(bt, on="pid")
    s = s[s["year"] <= s["draft_year"]].sort_values("year").groupby("player_id").tail(1)
    s = s.set_index("player_id").reindex(players["player_id"])
    f = 40 / (s["mpg"] * s["gp"])
    to = s["to_pct"] / 100
    X = pd.DataFrame({"pts": s["pts"] / s["mpg"] * 40, "trb": s["treb"] / s["mpg"] * 40,
                      "ast": s["ast"] / s["mpg"] * 40, "stl": s["stl"] / s["mpg"] * 40,
                      "blk": s["blk"] / s["mpg"] * 40, "fga": (s["p2a"] + s["p3a"]) * f,
                      "fta": s["fta"] * f, "fg3a": s["p3a"] * f, "pf": s["pfr"]})
    raw_tov = to * (X["fga"] + 0.44 * X["fta"]) / (1 - to)
    X["tov"] = tov_fit(raw_tov)
    X = X[BOX]
    X["ts"] = X["pts"] / (2 * (X["fga"] + 0.44 * X["fta"]))
    return X.reset_index(drop=True)


def main():
    ensure_results()
    new = pd.read_csv("model_data_2008_2019.csv", encoding="utf-8-sig", low_memory=False).copy()
    old = pd.read_csv("older_classes_features.csv", encoding="utf-8-sig", low_memory=False)
    old = old[old["n_college_seasons"].notna()].copy()
    old["player"] = old["player"].map(fix_text)
    fresh = pd.read_csv("fresh_2020_2023.csv", encoding="utf-8-sig", low_memory=False).copy()
    bt = load_barttorvik(years=range(2008, 2024))

    # 1. Box-score estimate of Barttorvik BPM
    Xn, Xo = per40_from_sports_reference(new), per40_from_sports_reference(old)
    ok = Xn.notna().all(axis=1) & new["bt_bpm"].notna()
    est = LinearRegression().fit(Xn[ok], new.loc[ok, "bt_bpm"])
    cv = cross_val_predict(LinearRegression(), Xn[ok], new.loc[ok, "bt_bpm"],
                           cv=KFold(5, shuffle=True, random_state=0))
    say("=== Box-score estimate of Barttorvik BPM (fit on 2008-2019) ===")
    say(f"Cross validated correlation with real BPM: {np.corrcoef(cv, new.loc[ok, 'bt_bpm'])[0, 1]:.3f}")
    say("Weights: " + ", ".join(f"{c} {w:+.2f}" for c, w in zip(Xn.columns, est.coef_)))

    new["est_bpm"] = est.predict(Xn.fillna(Xn.median()))
    old["est_bpm"] = est.predict(Xo.fillna(Xo.median()))
    for d in (new, old):
        d["basic_score"] = d["est_bpm"] + 3 * (20 - d["age_at_draft"])

    # 2. Out-of-era test
    say("\n=== 2002-2007 classes scored with 2008-2019 weights (out of era) ===")
    say(f"Players {len(old)} | All-Stars {int(old['all_star'].sum())} | "
        f"All-NBA {int(old['all_nba'].sum())} | All-Defensive {int(old['all_def'].sum())}")
    for o in ["all_star", "all_nba", "all_def"]:
        say(f"{o:9s} AUC 2002-2007 {roc_auc_score(old[o], old['basic_score']):.3f} | "
            f"2008-2019 {roc_auc_score(new[o], new['basic_score']):.3f}")
    old["stl40"] = old["final_per_min__stl_per_min"]
    old["blk40"] = old["career_per40__blk"]
    X = old[["age_at_draft", "stl40", "blk40"]]
    X = (X - X.mean()) / X.std()
    r = sm.Logit(old["all_def"], sm.add_constant(X)).fit(disp=0)
    say("All-Defensive 2002-2007, odds ratio per SD: " +
        ", ".join(f"{k} {np.exp(v):.2f} (p={r.pvalues[k]:.3f})" for k, v in r.params.drop("const").items()))
    X = old[["age_at_draft", "est_bpm", "career_per40__ft"]]
    X = (X - X.mean()) / X.std()
    r = sm.Logit(old["all_star"], sm.add_constant(X)).fit(disp=0)
    say("All-Star 2002-2007, odds ratio per SD: " +
        ", ".join(f"{k} {np.exp(v):.2f} (p={r.pvalues[k]:.3f})" for k, v in r.params.drop("const").items()))

    # 3. Refit on 2002-2019
    cols = ["player", "draft_year", "age_at_draft", "est_bpm", "all_star", "all_nba", "all_def"]
    comb = pd.concat([new[cols], old[cols]], ignore_index=True)
    say(f"\n=== Refit on 2002-2019: {len(comb)} players, {int(comb['all_star'].sum())} All-Stars ===")
    m = sm.Logit(comb["all_star"], sm.add_constant(comb[["est_bpm", "age_at_draft"]])).fit(disp=0)
    say(f"One year younger = {-m.params['age_at_draft'] / m.params['est_bpm']:.2f} BPM points")
    comb["basic_score"] = comb["est_bpm"] + 3 * (20 - comb["age_at_draft"])
    comb["pct"] = comb["basic_score"].rank(pct=True) * 100
    tiers = pd.cut(comb["pct"], [0, 50, 75, 90, 100.1], right=False,
                   labels=["below 50th", "50th to 75th", "75th to 90th", "90th and up"])
    say(comb.groupby(tiers, observed=True)["all_star"].agg(["size", "sum", "mean"]).round(3).to_string())
    say("Score cutoffs at the 50th, 75th and 90th percentiles: " +
        ", ".join(f"{np.percentile(comb['basic_score'], q):.1f}" for q in [50, 75, 90]))

    # 4. Fresh test with per-40 stats rebuilt from Barttorvik
    bt_new = per40_from_barttorvik(new, bt, lambda x: x)
    sr_tov = new["final_per_min__tov_per_min"].values
    k = bt_new["tov"].notna() & ~np.isnan(sr_tov)
    cal = LinearRegression().fit(bt_new.loc[k, ["tov"]], sr_tov[k])
    tov_fit = lambda x: cal.predict(x.fillna(x.median()).to_frame("tov"))
    Xf = per40_from_barttorvik(fresh, bt, tov_fit)
    fresh["basic_score"] = est.predict(Xf.fillna(Xf.median())) + 3 * (20 - fresh["age_at_draft_bt"])
    say("\n=== Basic score on the 2020-2023 classes ===")
    for o in ["all_star", "all_nba", "all_def"]:
        say(f"{o:9s} AUC {roc_auc_score(fresh[o], fresh['basic_score']):.3f}")
    fresh["pct"] = percentile_of(fresh["basic_score"], comb["basic_score"].values)
    say(fresh[fresh["all_star"] == 1].sort_values("pct", ascending=False)
        [["player", "pct"]].round(1).to_string(index=False))

    with open(f"{RESULTS}/era_check_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved {RESULTS}/era_check_report.txt")


if __name__ == "__main__":
    main()

"""
Step 12: the Age-Adjusted BPM prospect score and the fresh test

  Prospect score = Barttorvik BPM + 3 x (20 - age at draft)

What it does:
  1. Fits how many BPM points one year of age is worth (sets the weight of 3)
  2. Compares the one-number score with small multi-input models (cross validated)
  3. Builds the percentile table and All-Star rates by tier (2008-2019)
  4. Fresh test: trains on 2008-2019, scores the 2020-2023 classes blind

Inputs:  model_data_2008_2019.csv, fresh_2020_2023.csv (Step 11)
Outputs: results/prospect_score_report.txt, results/percentile_table.csv,
         results/fresh_test_scores.csv
Run:     python allstar_step12_prospect_score.py   (about a minute)
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from allstar_common import RESULTS, ensure_results, percentile_of

warnings.filterwarnings("ignore")
lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def cv_auc(df, cols, y):
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=7)
    aucs = []
    for tr, te in folds.split(df, df[y]):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=5000))
        m.fit(df.iloc[tr][cols], df.iloc[tr][y])
        aucs.append(roc_auc_score(df.iloc[te][y], m.predict_proba(df.iloc[te][cols])[:, 1]))
    return np.mean(aucs)


def tier(p):
    return ("90th and up" if p >= 90 else "75th to 90th" if p >= 75
            else "50th to 75th" if p >= 50 else "below 50th")


def main():
    ensure_results()
    tr = pd.read_csv("model_data_2008_2019.csv", encoding="utf-8-sig", low_memory=False)
    te = pd.read_csv("fresh_2020_2023.csv", encoding="utf-8-sig", low_memory=False)
    tr, te = tr.copy(), te.copy()
    age, bpm = "age_at_draft_bt", "bt_bpm"

    say("=== 1. How much is one year of age worth? ===")
    for o in ["all_star", "all_nba"]:
        m = sm.Logit(tr[o], sm.add_constant(tr[[bpm, age]])).fit(disp=0)
        say(f"{o}: one year younger = {-m.params[age] / m.params[bpm]:.2f} BPM points")

    say("\n=== 2. Age at draft and All-Star rate, 2008-2019 ===")
    bands = pd.cut(tr[age], [0, 20, 21, 22, 40], right=False,
                   labels=["Under 20", "20 to 21", "21 to 22", "22 or older"])
    say(tr.groupby(bands, observed=True)["all_star"].agg(["size", "sum", "mean"]).round(3).to_string())

    say("\n=== 3. Cross validated AUC, 2008-2019 ===")
    models = {
        "all_star": {"age only": [age], "age + BPM + FT/40": [age, bpm, "ft_per40_career"]},
        "all_nba": {"age only": [age], "age + BPM + FT/40": [age, bpm, "ft_per40_career"]},
        "all_def": {"age only": [age], "age + steals/40 + DBPM": [age, "stl_per40", "bt_dbpm"]},
    }
    for o, specs in models.items():
        for name, cols in specs.items():
            say(f"{o:9s} {name:24s} {cv_auc(tr, cols, o):.3f}")
        say(f"{o:9s} {'prospect score (one number)':24s} {roc_auc_score(tr[o], tr['prospect_score']):.3f}")

    say("\n=== 4. Percentile table (2008-2019 college draftees) ===")
    qs = [99, 95, 90, 80, 70, 60, 50, 30, 10]
    table = pd.DataFrame({"percentile": qs,
                          "score_at_least": [round(np.percentile(tr["prospect_score"], q), 1) for q in qs]})
    table.to_csv(f"{RESULTS}/percentile_table.csv", index=False)
    say(table.to_string(index=False))
    tr["percentile"] = tr["prospect_score"].rank(pct=True) * 100
    tr["tier"] = tr["percentile"].map(tier)
    say(tr.groupby("tier")[["all_star", "all_nba"]].agg(["size", "sum", "mean"]).round(3).to_string())

    say("\n=== 5. Fresh test: trained on 2008-2019, scored on 2020-2023 ===")
    te["percentile"] = percentile_of(te["prospect_score"], tr["prospect_score"].values)
    te["tier"] = te["percentile"].map(tier)
    for o in ["all_star", "all_nba", "all_def"]:
        auc_one = roc_auc_score(te[o], te["prospect_score"])
        cols = models[o][list(models[o])[1]]
        full = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=5000)).fit(tr[cols], tr[o])
        auc_full = roc_auc_score(te[o], full.predict_proba(te[cols])[:, 1])
        say(f"{o:9s} positives {int(te[o].sum()):2d} | prospect score AUC {auc_one:.3f} "
            f"| multi-input model AUC {auc_full:.3f}")
    say(te.groupby("tier")["all_star"].agg(["size", "sum"]).to_string())
    say("\n2020-2023 All-Stars by percentile:")
    say(te[te["all_star"] == 1].sort_values("percentile", ascending=False)
        [["player", "draft_year", "pick", "percentile"]].round(1).to_string(index=False))
    te.to_csv(f"{RESULTS}/fresh_test_scores.csv", index=False, encoding="utf-8-sig")

    with open(f"{RESULTS}/prospect_score_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved {RESULTS}/prospect_score_report.txt")


if __name__ == "__main__":
    main()

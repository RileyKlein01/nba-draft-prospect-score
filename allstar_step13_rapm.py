"""
Step 13: what predicts NBA impact (RAPM) from college stats

RAPM = points per 100 possessions a team gains with a player on the floor,
from the public gabriel1200/site_Data repository (2009-10 to 2022-23).
Each player's outcome = average RAPM over NBA seasons 2 to 5.

What it does:
  1. Downloads rapm.csv (once) and matches players by name
  2. Screens every college stat against overall, offensive and defensive RAPM,
     with a permutation test that corrects for testing them all
  3. Regressions: offensive BPM vs offensive RAPM, defensive BPM vs defensive RAPM
  4. Average RAPM by college BPM fifth

Inputs:  model_data_2008_2019.csv (Step 11)
Outputs: rapm_season.csv (download), results/rapm_report.txt, results/rapm_screen_*.csv
Run:     python allstar_step13_rapm.py   (about a minute)
"""

import os
import warnings

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

from allstar_common import RESULTS, ensure_results, fix_text, name_key

warnings.filterwarnings("ignore")
RAPM_URL = "https://raw.githubusercontent.com/gabriel1200/site_Data/master/rapm.csv"
lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def load_rapm():
    if not os.path.exists("rapm_season.csv"):
        print("Downloading RAPM data...")
        resp = requests.get(RAPM_URL, timeout=60)
        resp.raise_for_status()
        with open("rapm_season.csv", "w", encoding="utf-8") as f:
            f.write(resp.text)
    r = pd.read_csv("rapm_season.csv")
    r["end_year"] = r["season"].str[:4].astype(int) + 1
    r["key"] = r["playerName"].map(name_key)
    return r


def attach_rapm(d, r):
    d = d.copy()
    d["key"] = d["player"].map(fix_text).map(name_key)
    first = r.groupby(["playerId", "key"])["end_year"].min().reset_index()
    m = d[["player_id", "key", "draft_year"]].merge(first, on="key")
    m = m[(m["end_year"] >= m["draft_year"] + 1) & (m["end_year"] <= m["draft_year"] + 4)]
    m = m.sort_values("end_year").drop_duplicates("player_id")
    d = d.merge(m[["player_id", "playerId"]], on="player_id", how="left")
    rr = r.merge(d[["playerId", "draft_year"]].dropna(), on="playerId")
    w = rr[(rr["end_year"] >= rr["draft_year"] + 2) & (rr["end_year"] <= rr["draft_year"] + 5)]
    agg = w.groupby("playerId").agg(rapm=("RAPM", "mean"), rapm_off=("RAPM__Off", "mean"),
                                    rapm_def=("RAPM__Def", "mean"))
    return d.merge(agg, left_on="playerId", right_index=True, how="left")


def candidate_stats(d):
    skip = {"pick", "draft_year", "all_star", "all_nba", "all_def", "rapm", "rapm_off",
            "rapm_def", "playerId", "pid", "rsci_rank", "bt_year", "seasons_available",
            "prospect_score", "college_minutes", "weight_lb"}
    cols = [c for c in d.columns
            if c not in skip and pd.api.types.is_numeric_dtype(d[c])
            and not c.endswith(("_count", "_season")) and "_conf__" not in c
            and not c.startswith("delta_") and d[c].notna().mean() > 0.7 and d[c].nunique() > 5]
    seen, keep = set(), []
    for c in cols:
        k = tuple(d[c].round(5).fillna(-999))
        if k not in seen:
            seen.add(k)
            keep.append(c)
    return keep


def screen(d, y, cols, n_perm=1000, seed=3):
    """Spearman correlation of every stat with y, plus a max-statistic permutation
    p-value that accounts for testing every stat."""
    rng = np.random.default_rng(seed)
    ok = d[y].notna()
    ranks = d.loc[ok, cols].rank().apply(lambda s: s.fillna(s.median()))
    z = (ranks - ranks.mean()) / ranks.std()
    yr = d.loc[ok, y].rank().values

    def cors(v):
        vz = (v - v.mean()) / v.std()
        return z.mul(vz, axis=0).mean().values * len(v) / (len(v) - 1)

    real = pd.Series(cors(yr), index=cols)
    null = np.array([np.abs(cors(rng.permutation(yr))).max() for _ in range(n_perm)])
    out = pd.DataFrame({"spearman": real.round(3),
                        "p_corrected": [(np.sum(null >= abs(v)) + 1) / (n_perm + 1) for v in real]})
    return out.reindex(out["spearman"].abs().sort_values(ascending=False).index)


def ols(d, y, cols):
    X = d[cols]
    X = (X - X.mean()) / X.std()
    ok = X.notna().all(axis=1) & d[y].notna()
    m = sm.OLS(d.loc[ok, y], sm.add_constant(X[ok])).fit()
    ci = m.conf_int()
    t = pd.DataFrame({"effect_per_sd": m.params, "ci_low": ci[0], "ci_high": ci[1],
                      "p": m.pvalues}).drop("const").round(3)
    return t, ok.sum(), round(m.rsquared, 3)


def main():
    ensure_results()
    d = pd.read_csv("model_data_2008_2019.csv", encoding="utf-8-sig", low_memory=False)
    d = attach_rapm(d, load_rapm())
    say(f"Players with NBA RAPM in seasons 2 to 5: {d['rapm'].notna().sum()} of {len(d)}")
    say(f"All-Star RAPM median: {d.loc[d['all_star'] == 1, 'rapm'].median():.2f}")

    cols = candidate_stats(d)
    say(f"College stats screened: {len(cols)}")
    for y in ["rapm", "rapm_off", "rapm_def"]:
        s = screen(d, y, cols)
        s.to_csv(f"{RESULTS}/rapm_screen_{y}.csv")
        say(f"\nTop college stats for {y}:")
        say(s.head(10).to_string())

    for y, c in [("rapm", ["bt_obpm", "bt_dbpm", "age_at_draft_bt"]),
                 ("rapm_off", ["bt_obpm", "bt_dbpm", "age_at_draft_bt"]),
                 ("rapm_def", ["bt_obpm", "bt_dbpm", "age_at_draft_bt", "stl_per40"])]:
        t, n, r2 = ols(d, y, c)
        say(f"\n{y}: n={n}, R2={r2}")
        say(t.to_string())

    fifths = pd.qcut(d["bt_bpm"], 5, labels=["bottom 20%", "20-40%", "40-60%", "60-80%", "top 20%"])
    say("\nAverage RAPM by college BPM fifth:")
    say(d.groupby(fifths, observed=True).agg(
        players=("rapm", "count"), avg_rapm=("rapm", "mean"),
        share_plus1=("rapm", lambda s: (s > 1).mean())).round(2).to_string())

    d.to_csv("model_data_with_rapm.csv", index=False, encoding="utf-8-sig")
    with open(f"{RESULTS}/rapm_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved {RESULTS}/rapm_report.txt")


if __name__ == "__main__":
    main()

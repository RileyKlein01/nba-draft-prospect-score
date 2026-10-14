"""
All-Star / All-NBA / All-Defensive model: Step 4 (modeling)

For each outcome (All-Star, All-NBA, All-Defensive) this fits:

  Sample A  2008-2019 draft classes, stats every class has
  Sample B  2011-2019 draft classes, adds advanced stats (BPM, per 100 poss)

  each WITHOUT draft pick  ("what in college predicts NBA stardom?")
  and WITH draft pick      ("what predicts stardom that scouts missed?")

Method: LASSO logistic regression (it shrinks useless stats to exactly zero,
which is what you want with hundreds of stats and ~60 All-Stars), evaluated
with repeated stratified cross validation so every AUC is out of sample.
A stat only counts as a real predictor if LASSO keeps choosing it (80%+ of folds) across
the cross validation folds (selection frequency).

Input:  college_features.csv
Run:    python -m pip install scikit-learn
        python allstar_step4_model.py
Output (folder "results"):
  summary.csv              AUC for every model, plus pick-only and age-only baselines
  selected_<model>.csv     stats LASSO keeps, how often, and odds ratios
  univariate_<outcome>.csv every stat on its own: AUC and All-Star vs rest means
  temporal_check.csv       train on 2008-2016, test on 2017-2019
Runtime: a few minutes.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUT = "results"
OUTCOMES = ["all_star", "all_nba", "all_def"]
POWER_CONFS = {"ACC", "Big Ten", "Big 12", "SEC", "Pac-12", "Pac-10", "Big East"}
SEED = 42


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_base(df):
    """Bio and context features every player has."""
    X = pd.DataFrame(index=df.index)
    X["age_at_draft"] = df["age_at_draft"]
    X["height_in"] = df["height_in"]
    X["weight_lb"] = df["weight_lb"]
    X["n_college_seasons"] = df["n_college_seasons"]
    # Recruiting: unranked players get 0, #1 recruit gets 100
    X["rsci_score"] = (101 - df["rsci_rank"]).fillna(0)
    X["rsci_ranked"] = df["rsci_rank"].notna().astype(int)
    X["power_conf"] = df["final_conf"].isin(POWER_CONFS).astype(int)
    return X


def stat_columns(df, tables, prefixes=("final_", "career_", "delta_")):
    cols = []
    for c in df.columns:
        if not c.startswith(prefixes) or "__" not in c:
            continue
        table = c.split("__")[0].split("_", 1)[1]          # e.g. per_min
        stat = c.split("__")[1]
        if table.endswith("_conf"):
            continue                                       # conference-only splits: redundant
        if table in tables:
            cols.append(c)
        elif table == "per_game" and (stat.endswith("pct") or stat in
                                      ("games", "games_started", "mp_per_g")):
            cols.append(c)                                 # shooting % and playing time
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def feature_matrix(df, sample, with_pick):
    X = build_base(df)
    if sample == "A":
        cols = stat_columns(df, {"per_min"})
        # keep only stats nearly every player in 2008-2019 has
        cols = [c for c in cols if df[c].notna().mean() >= 0.95
                or (c.startswith("delta_") and df.loc[df["n_college_seasons"] > 1, c].notna().mean() >= 0.95)]
    else:
        cols = stat_columns(df, {"per_min", "per_poss", "advanced"})
        cols = [c for c in cols if df[c].notna().mean() >= 0.90
                or (c.startswith("delta_") and df.loc[df["n_college_seasons"] > 1, c].notna().mean() >= 0.90)]
    X = X.join(df[cols])
    if with_pick:
        X["log_pick"] = np.log(df["pick"])
    X = X.loc[:, X.notna().mean() > 0.5]
    X = X.loc[:, X.std(skipna=True) > 0]
    return X


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
def lasso_pipeline():
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegressionCV(Cs=np.logspace(-3, 1, 25), cv=inner, penalty="l1",
                             solver="liblinear", scoring="roc_auc",
                             class_weight="balanced", max_iter=5000),
    )


def simple_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=5000))


def cv_evaluate(X, y, make_model, track_selection=False):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    aucs, aps, picks = [], [], pd.Series(0.0, index=X.columns)
    for tr, te in outer.split(X, y):
        model = make_model().fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        aps.append(average_precision_score(y.iloc[te], p))
        if track_selection:
            coef = model[-1].coef_.ravel()
            picks += (np.abs(coef) > 1e-8).astype(float)
    n = outer.get_n_splits()
    return np.mean(aucs), np.std(aucs), np.mean(aps), (picks / n if track_selection else None)


def full_fit_odds(X, y, keep):
    """Refit a plain logistic on the stable stats to get readable odds ratios
    (per 1 standard deviation)."""
    if not keep:
        return pd.DataFrame()
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    m.fit(X[keep], y)
    coef = m[-1].coef_.ravel()
    return pd.DataFrame({"stat": keep, "coef_per_sd": coef.round(3),
                         "odds_ratio_per_sd": np.exp(coef).round(2)})


def univariate(X, y):
    rows = []
    for c in X.columns:
        x = X[c]
        ok = x.notna()
        if ok.sum() < 50 or y[ok].nunique() < 2:
            continue
        auc = roc_auc_score(y[ok], x[ok])
        rows.append({"stat": c, "auc": round(auc, 3),
                     "strength": round(abs(auc - 0.5) * 2, 3),
                     "direction": "higher is better" if auc >= 0.5 else "lower is better",
                     "mean_yes": round(x[ok & (y == 1)].mean(), 3),
                     "mean_no": round(x[ok & (y == 0)].mean(), 3),
                     "coverage": round(ok.mean(), 3)})
    return pd.DataFrame(rows).sort_values("strength", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv("college_features.csv", encoding="utf-8-sig", low_memory=False)
    df = df[df["n_college_seasons"].notna()].reset_index(drop=True)
    print(f"Players: {len(df)}\n")

    summary, temporal = [], []
    for outcome in OUTCOMES:
        for sample in ["A", "B"]:
            d = df if sample == "A" else df[df["draft_year"] >= 2011].reset_index(drop=True)
            y = d[outcome].astype(int)

            # Baselines
            pick_only = pd.DataFrame({"log_pick": np.log(d["pick"])})
            age_only = d[["age_at_draft"]]
            b_pick = cv_evaluate(pick_only, y, simple_pipeline)
            b_age = cv_evaluate(age_only, y, simple_pipeline)

            for with_pick in [False, True]:
                X = feature_matrix(d, sample, with_pick)
                name = f"{outcome}_{sample}_{'with' if with_pick else 'no'}_pick"
                print(f"{name}: {X.shape[1]} features, {y.sum()} positives of {len(y)}")
                auc, sd, ap, sel = cv_evaluate(X, y, lasso_pipeline, track_selection=True)

                stable = sel[sel >= 0.8].sort_values(ascending=False)
                odds = full_fit_odds(X, y, list(stable.index))
                table = pd.DataFrame({"stat": sel.index, "selection_freq": sel.values.round(2)})
                table = table[table["selection_freq"] > 0].merge(odds, on="stat", how="left")
                table.sort_values("selection_freq", ascending=False).to_csv(
                    f"{OUT}/selected_{name}.csv", index=False)

                summary.append({
                    "outcome": outcome, "sample": sample,
                    "classes": "2008-2019" if sample == "A" else "2011-2019",
                    "draft_pick": "with" if with_pick else "without",
                    "players": len(y), "positives": int(y.sum()),
                    "n_features": X.shape[1],
                    "cv_auc": round(auc, 3), "cv_auc_sd": round(sd, 3),
                    "cv_avg_precision": round(ap, 3),
                    "base_rate": round(y.mean(), 3),
                    "pick_only_auc": round(b_pick[0], 3),
                    "age_only_auc": round(b_age[0], 3),
                    "n_stable_stats": len(stable),
                    "stable_stats": ", ".join(stable.index[:8]),
                })

                # Temporal check: does a model trained on older classes work on newer ones?
                train, test = d["draft_year"] <= 2016, d["draft_year"] >= 2017
                if y[test].nunique() == 2:
                    m = lasso_pipeline().fit(X[train], y[train])
                    p = m.predict_proba(X[test])[:, 1]
                    temporal.append({"model": name,
                                     "test_positives": int(y[test].sum()),
                                     "test_auc": round(roc_auc_score(y[test], p), 3)})

        uni_X = feature_matrix(df, "A", True).join(
            feature_matrix(df[df["draft_year"] >= 2011], "B", False), rsuffix="_B")
        uni_X = uni_X.loc[:, ~uni_X.columns.str.endswith("_B")]
        univariate(uni_X, df[outcome].astype(int)).to_csv(
            f"{OUT}/univariate_{outcome}.csv", index=False)

    s = pd.DataFrame(summary)
    s.to_csv(f"{OUT}/summary.csv", index=False)
    pd.DataFrame(temporal).to_csv(f"{OUT}/temporal_check.csv", index=False)

    print("\n=== Cross validated results ===")
    print(s[["outcome", "sample", "draft_pick", "positives", "cv_auc", "cv_auc_sd",
             "pick_only_auc", "age_only_auc", "n_stable_stats"]].to_string(index=False))
    print("\n=== Stable predictors (chosen in 80%+ of folds) ===")
    for _, r in s.iterrows():
        print(f"{r['outcome']} | {r['sample']} | {r['draft_pick']} pick: {r['stable_stats'] or '(none)'}")
    print("\n=== Temporal check (train 2008-2016, test 2017-2019) ===")
    print(pd.DataFrame(temporal).to_string(index=False))
    print(f"\nAll files saved in the '{OUT}' folder.")


if __name__ == "__main__":
    main()

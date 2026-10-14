"""
All-Star / All-NBA / All-Defensive model: Step 5 (pressure test the defense finding)

Claim being tested: college steal rate (and block rate / DBPM) predicts
All-Defensive selection, even after accounting for draft pick.

Four checks:
  1. Duplicates   drop stats that appear twice under different table names
  2. Position     does steal rate still matter after controlling for position
                  and height, and does it work WITHIN guards and within bigs?
  3. Chance       permutation tests: shuffle the All-Defensive labels and see
                  how often noise produces a stat as strong as steal rate
  4. Confidence   bootstrap the AUC gap between college stats and draft pick

Input:  college_features.csv
Run:    python -m pip install statsmodels
        python allstar_step5_robustness.py
Output: results/robustness_report.txt (same text as printed) and supporting CSVs
Runtime: roughly 10 to 20 minutes, mostly check 3b. Progress is printed.
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUT = "results"
OUTCOME = "all_def"
SEED = 42
N_PERM_UNIVARIATE = 2000
N_PERM_LASSO = 100
N_BOOT = 2000
rng = np.random.default_rng(SEED)

STL = "final_per_min__stl_per_min"
BLK = "career_per_min__blk_per_min"
DBPM = "final_advanced__dbpm"
STL_PCT = "final_advanced__stl_pct"

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def load():
    df = pd.read_csv("college_features.csv", encoding="utf-8-sig", low_memory=False)
    df = df[df["n_college_seasons"].notna()].reset_index(drop=True)
    pos = df["position"].fillna("").str.lower()
    df["is_guard"] = pos.str.contains("guard") | pos.str.match(r"^g\b")
    df["is_center"] = pos.str.contains("center") | pos.str.match(r"^c\b")
    df["is_forward"] = pos.str.contains("forward") | pos.str.match(r"^f\b")
    df["pos_known"] = df["is_guard"] | df["is_center"] | df["is_forward"]
    # Simple groups for within-position tests
    df["pos_group"] = np.where(df["is_center"] | (df["is_forward"] & ~df["is_guard"]), "big",
                       np.where(df["is_guard"], "guard", "unknown"))
    df["log_pick"] = np.log(df["pick"])
    for c in ["is_guard", "is_center", "is_forward", "pos_known"]:
        df[c] = df[c].astype(int)
    return df


def drop_duplicate_columns(X):
    """Same values under two names (e.g. ft_pct in per game and per 40 tables)."""
    seen, keep, dropped = {}, [], []
    for c in X.columns:
        key = tuple(X[c].round(6).fillna(-9999).values)
        if key in seen:
            dropped.append((c, seen[key]))
        else:
            seen[key] = c
            keep.append(c)
    return X[keep], dropped


def candidate_stats(df, advanced):
    tables = {"per_min", "per_poss", "advanced"} if advanced else {"per_min"}
    min_cov = 0.90 if advanced else 0.95
    cols = []
    for c in df.columns:
        if not c.startswith(("final_", "career_", "delta_")) or "__" not in c:
            continue
        table = c.split("__")[0].split("_", 1)[1]
        stat = c.split("__")[1]
        if table.endswith("_conf"):
            continue
        ok_table = table in tables or (table == "per_game" and (
            stat.endswith("pct") or stat in ("games", "games_started", "mp_per_g")))
        if not ok_table or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cov = (df.loc[df["n_college_seasons"] > 1, c].notna().mean()
               if c.startswith("delta_") else df[c].notna().mean())
        if cov >= min_cov:
            cols.append(c)
    base = ["age_at_draft", "height_in", "weight_lb", "n_college_seasons"]
    X = df[base + cols].copy()
    X["rsci_score"] = (101 - df["rsci_rank"]).fillna(0)
    return X


# ---------------------------------------------------------------------------
# 1. Duplicates
# ---------------------------------------------------------------------------
def check_duplicates(df):
    say("=" * 70)
    say("CHECK 1: DUPLICATE COLUMNS")
    say("=" * 70)
    for label, adv in [("2008-2019 stats", False), ("2011-2019 stats", True)]:
        X = candidate_stats(df if not adv else df[df["draft_year"] >= 2011], adv)
        X2, dropped = drop_duplicate_columns(X)
        say(f"{label}: {X.shape[1]} columns -> {X2.shape[1]} after removing {len(dropped)} duplicates")
        for c, orig in dropped[:6]:
            say(f"   {c}  (same as {orig})")
        if len(dropped) > 6:
            say(f"   ... and {len(dropped) - 6} more")
    say()


# ---------------------------------------------------------------------------
# 2. Position
# ---------------------------------------------------------------------------
def logit_table(d, y, cols, title):
    data = d[cols].copy()
    for c in cols:
        if data[c].nunique() > 2:                       # standardize continuous
            data[c] = (data[c] - data[c].mean()) / data[c].std()
    data = sm.add_constant(data)
    ok = data.notna().all(axis=1)
    model = sm.Logit(y[ok], data[ok]).fit(disp=0)
    ci = model.conf_int()
    out = pd.DataFrame({
        "odds_ratio": np.exp(model.params).round(2),
        "ci_low": np.exp(ci[0]).round(2),
        "ci_high": np.exp(ci[1]).round(2),
        "p_value": model.pvalues.round(4),
    }).drop(index="const")
    say(f"\n{title}  (n={ok.sum()}, positives={int(y[ok].sum())})")
    say("Continuous stats are per 1 standard deviation; position flags are yes/no.")
    say(out.to_string())
    return out


def check_position(df):
    say("=" * 70)
    say("CHECK 2: POSITION")
    say("=" * 70)
    say(f"Position known for {df['pos_known'].mean():.0%} of players")
    say(df.groupby("pos_group")[[OUTCOME]].agg(["count", "sum"]).to_string())
    say("\nAverage final-season steals per 40 by position (All-Defensive vs not):")
    say(df.groupby(["pos_group", OUTCOME])[STL].mean().unstack().round(2).to_string())

    y = df[OUTCOME].astype(int)
    controls = ["log_pick", "age_at_draft", "height_in", "is_guard", "is_center"]
    logit_table(df, y, controls, "2a. Controls only")
    t = logit_table(df, y, controls + [STL, BLK],
                    "2b. Controls + steal rate + block rate (2008-2019)")
    t.to_csv(f"{OUT}/robust_logit_full_sample.csv")

    d11 = df[df["draft_year"] >= 2011].reset_index(drop=True)
    y11 = d11[OUTCOME].astype(int)
    logit_table(d11, y11, controls + [STL, BLK, DBPM],
                "2c. Controls + steals + blocks + DBPM (2011-2019)")

    # Within-position AUC: does steal rate separate All-Defensive players
    # from other players at the SAME position?
    say("\n2d. Steal rate AUC within each position group (0.5 = no signal)")
    rows = []
    for g in ["guard", "big"]:
        sub = df[df["pos_group"] == g]
        if sub[OUTCOME].nunique() < 2:
            continue
        for stat in [STL, BLK, "log_pick"]:
            ok = sub[stat].notna()
            auc = roc_auc_score(sub.loc[ok, OUTCOME], sub.loc[ok, stat])
            if stat == "log_pick":
                auc = 1 - auc                            # lower pick number is better
            rows.append({"group": g, "stat": stat, "positives": int(sub[OUTCOME].sum()),
                         "players": len(sub), "auc": round(auc, 3)})
    say(pd.DataFrame(rows).to_string(index=False))

    # Position-relative steal rate
    df["stl_z_within_pos"] = df.groupby("pos_group")[STL].transform(
        lambda s: (s - s.mean()) / s.std())
    logit_table(df, y, ["log_pick", "age_at_draft", "height_in", "stl_z_within_pos"],
                "2e. Steal rate measured relative to position average")
    say()


# ---------------------------------------------------------------------------
# 3. Chance
# ---------------------------------------------------------------------------
def auc_matrix(X, y):
    """AUC of every column at once, via ranks (columns already imputed)."""
    ranks = X.rank(axis=0).values
    n1 = y.sum()
    n0 = len(y) - n1
    pos_rank_sum = ranks[y.values == 1].sum(axis=0)
    return (pos_rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def check_chance_univariate(df):
    say("=" * 70)
    say("CHECK 3a: COULD A STAT THIS STRONG APPEAR BY CHANCE? (one stat at a time)")
    say("=" * 70)
    say(f"Shuffles the All-Defensive labels {N_PERM_UNIVARIATE} times. Each time, finds the")
    say("single strongest stat among ALL stats tested. If the real steal rate beats")
    say("nearly every one of those 'best by luck' stats, it is not a fluke of testing")
    say("hundreds of stats.\n")
    rows = []
    for label, d, adv, target in [("2008-2019", df, False, STL),
                                  ("2011-2019", df[df["draft_year"] >= 2011].reset_index(drop=True),
                                   True, DBPM)]:
        X, _ = drop_duplicate_columns(candidate_stats(d, adv))
        X = X.fillna(X.median())
        y = d[OUTCOME].astype(int).reset_index(drop=True)
        real = pd.Series(np.abs(auc_matrix(X, y) - 0.5) * 2, index=X.columns)
        null_max = np.empty(N_PERM_UNIVARIATE)
        for i in range(N_PERM_UNIVARIATE):
            yp = pd.Series(rng.permutation(y.values))
            null_max[i] = np.max(np.abs(auc_matrix(X, yp) - 0.5) * 2)
        for stat in [target, STL] if target != STL else [STL]:
            if stat not in real:
                continue
            p = (np.sum(null_max >= real[stat]) + 1) / (N_PERM_UNIVARIATE + 1)
            rows.append({"sample": label, "stat": stat, "stats_tested": X.shape[1],
                         "real_auc": round(0.5 + real[stat] / 2, 3),
                         "best_chance_auc_95th_pct": round(0.5 + np.percentile(null_max, 95) / 2, 3),
                         "p_value_corrected": round(p, 4)})
    out = pd.DataFrame(rows)
    say(out.to_string(index=False))
    say("p_value_corrected already accounts for testing every stat, not just this one.")
    out.to_csv(f"{OUT}/robust_permutation_univariate.csv", index=False)
    say()


def lasso_fold_selection(X, y, seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    counts = np.zeros(X.shape[1])
    for tr, _ in folds.split(X, y):
        m = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegressionCV(Cs=np.logspace(-3, 1, 15), cv=3, penalty="l1",
                                 solver="liblinear", scoring="roc_auc",
                                 class_weight="balanced", max_iter=3000))
        m.fit(X.iloc[tr], y.iloc[tr])
        counts += (np.abs(m[-1].coef_.ravel()) > 1e-8)
    return counts / 5


def check_chance_lasso(df):
    say("=" * 70)
    say("CHECK 3b: DOES LASSO 'FIND' STEAL-STRENGTH PREDICTORS IN PURE NOISE?")
    say("=" * 70)
    X, _ = drop_duplicate_columns(candidate_stats(df, False))
    X["log_pick"] = df["log_pick"]
    y = df[OUTCOME].astype(int)

    real = pd.Series(lasso_fold_selection(X, y, SEED), index=X.columns)
    say(f"Real data, selection in 5 folds: steal rate {real.get(STL, 0):.0%}, "
        f"block rate {real.get(BLK, 0):.0%}, draft pick {real['log_pick']:.0%}")

    hits = 0
    for i in range(N_PERM_LASSO):
        yp = pd.Series(rng.permutation(y.values), index=y.index)
        sel = lasso_fold_selection(X, yp, SEED + i + 1)
        if sel.max() >= 1.0:
            hits += 1
        if (i + 1) % 10 == 0:
            print(f"   permutation {i + 1}/{N_PERM_LASSO}")
    say(f"Shuffled labels: some stat was chosen in 5 of 5 folds in {hits} of "
        f"{N_PERM_LASSO} shuffles ({hits / N_PERM_LASSO:.0%}).")
    say("If that share is small, a stat being chosen every time (like steal rate)")
    say("is unlikely to be LASSO latching onto noise.")
    say()


# ---------------------------------------------------------------------------
# 4. Bootstrap AUC gap
# ---------------------------------------------------------------------------
def oof_predictions(X, y, make_model, repeats=5):
    preds = np.zeros(len(y))
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    for tr, te in rkf.split(X, y):
        m = make_model().fit(X.iloc[tr], y.iloc[tr])
        preds[te] += m.predict_proba(X.iloc[te])[:, 1] / repeats
    return preds


def simple():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=5000))


def check_bootstrap(df):
    say("=" * 70)
    say("CHECK 4: HOW SURE ARE WE COLLEGE STATS BEAT DRAFT PICK FOR DEFENSE?")
    say("=" * 70)
    rows = []
    specs = [
        ("2008-2019", df, ["age_at_draft", "height_in", "is_guard", "is_center", STL, BLK]),
        ("2011-2019", df[df["draft_year"] >= 2011].reset_index(drop=True),
         ["age_at_draft", "height_in", "is_guard", "is_center", STL, BLK, DBPM]),
    ]
    for label, d, college_cols in specs:
        y = d[OUTCOME].astype(int).reset_index(drop=True)
        d = d.reset_index(drop=True)
        p_pick = oof_predictions(d[["log_pick"]], y, simple)
        p_col = oof_predictions(d[college_cols], y, simple)
        p_both = oof_predictions(d[college_cols + ["log_pick"]], y, simple)

        gaps, gaps_add = [], []
        n = len(y)
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            if y.iloc[idx].nunique() < 2:
                continue
            a_pick = roc_auc_score(y.iloc[idx], p_pick[idx])
            gaps.append(roc_auc_score(y.iloc[idx], p_col[idx]) - a_pick)
            gaps_add.append(roc_auc_score(y.iloc[idx], p_both[idx]) - a_pick)
        gaps, gaps_add = np.array(gaps), np.array(gaps_add)
        rows.append({
            "sample": label, "positives": int(y.sum()),
            "auc_pick_only": round(roc_auc_score(y, p_pick), 3),
            "auc_college_only": round(roc_auc_score(y, p_col), 3),
            "auc_college_plus_pick": round(roc_auc_score(y, p_both), 3),
            "college_minus_pick_95ci": f"{np.percentile(gaps, 2.5):+.3f} to {np.percentile(gaps, 97.5):+.3f}",
            "share_college_better": f"{(gaps > 0).mean():.0%}",
            "adding_college_to_pick_95ci": f"{np.percentile(gaps_add, 2.5):+.3f} to {np.percentile(gaps_add, 97.5):+.3f}",
            "share_adding_helps": f"{(gaps_add > 0).mean():.0%}",
        })
    out = pd.DataFrame(rows)
    say(out.T.to_string(header=False))
    say("\nCollege model = age, height, position, steal rate, block rate (+ DBPM for 2011-2019).")
    say("All AUCs are out of sample. If a 95% interval stays above zero, the gain is solid.")
    out.to_csv(f"{OUT}/robust_bootstrap.csv", index=False)
    say()


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    say(f"Players: {len(df)} | All-Defensive: {int(df[OUTCOME].sum())}\n")
    check_duplicates(df)
    check_position(df)
    check_chance_univariate(df)
    check_bootstrap(df)
    check_chance_lasso(df)     # slowest, so last
    with open(f"{OUT}/robustness_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    say(f"Report saved to {OUT}/robustness_report.txt")


if __name__ == "__main__":
    main()

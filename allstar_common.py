"""
Shared helpers for steps 11 to 16: loading Barttorvik files, matching players
across sources by name, and building the Barttorvik-based model inputs.
"""

import glob
import os
import re
import unicodedata

import numpy as np
import pandas as pd

BT_DIR = "barttorvik"
RESULTS = "results"

# Barttorvik's getadvstats CSV has no header row; these are its 67 columns.
BT_COLUMNS = [
    "name", "team", "conf", "gp", "min_pct", "ortg", "usg", "efg", "ts", "orb_pct",
    "drb_pct", "ast_pct", "to_pct", "ftm", "fta", "ft_pct", "p2m", "p2a", "p2_pct",
    "p3m", "p3a", "p3_pct", "blk_pct", "stl_pct", "ftr", "cls", "ht", "num", "porpag",
    "adjoe", "pfr", "year", "pid", "hometown", "rec_rank", "ast_tov", "rim_m", "rim_a",
    "mid_m", "mid_a", "rim_pct", "mid_pct", "dunk_m", "dunk_a", "dunk_pct", "nba_pick",
    "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm", "dbpm", "gbpm", "mpg", "ogbpm",
    "dgbpm", "oreb", "dreb", "treb", "ast", "stl", "blk", "pts", "role", "c65", "birth",
]
TEXT_COLUMNS = {"name", "team", "conf", "cls", "ht", "hometown", "role", "birth"}

# Names that differ between Basketball Reference and Barttorvik
ALIASES = {
    "mauriceharkless": "moeharkless", "mobamba": "mohamedbamba",
    "nicclaxton": "nicolasclaxton", "kzokpala": "kezieokpala",
    "bamadebayo": "edriceadebayo", "jeffreytaylor": "jefftaylor",
}


def fix_text(s):
    """Repairs names that were saved as UTF-8 but read as Latin-1 (e.g. VuÄeviÄ)."""
    try:
        return s.encode("latin1").decode("utf8")
    except Exception:
        return s


def _plain(s):
    s = unicodedata.normalize("NFKD", fix_text(str(s))).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)


def name_key(s):
    return re.sub("[^a-z]", "", _plain(s))


def last_name(s):
    parts = re.sub("[^a-z ]", "", _plain(s)).split()
    return parts[-1] if parts else ""


def load_barttorvik(years=None):
    frames = []
    for path in sorted(glob.glob(os.path.join(BT_DIR, "barttorvik_*.csv"))):
        year = int(re.search(r"(\d{4})", os.path.basename(path)).group(1))
        if years is not None and year not in years:
            continue
        df = pd.read_csv(path, header=None, low_memory=False)
        df.columns = BT_COLUMNS[: df.shape[1]]
        frames.append(df)
    bt = pd.concat(frames, ignore_index=True)
    for c in bt.columns:
        if c not in TEXT_COLUMNS:
            bt[c] = pd.to_numeric(bt[c], errors="coerce")
    bt["key"] = bt["name"].map(name_key)
    bt["last"] = bt["name"].map(last_name)
    return bt


def match_to_barttorvik(players, bt):
    """players needs player_id, player, draft_year, pick (and optionally a college
    column). Returns player_id -> Barttorvik pid, matching on:
      1. draft year + pick + (full or last name)
      2. exact name, latest season at or before the draft
      3. last name + same school, final two seasons
    """
    p = players.copy()
    p["key"] = p["player"].map(fix_text).map(name_key).replace(ALIASES)
    p["last"] = p["player"].map(fix_text).map(last_name)

    drafted = bt[bt["nba_pick"].notna()][["pid", "key", "last", "year", "nba_pick"]] \
        .drop_duplicates(["pid", "year"])
    m = p[["player_id", "key", "last", "draft_year", "pick"]].merge(
        drafted, left_on=["draft_year", "pick"], right_on=["year", "nba_pick"], suffixes=("", "_b"))
    m = m[(m["key"] == m["key_b"]) | (m["last"] == m["last_b"])]
    found = m[["player_id", "pid"]].drop_duplicates("player_id")

    rest = p[~p["player_id"].isin(found["player_id"])]
    c = rest[["player_id", "key", "draft_year"]].merge(bt[["pid", "key", "year"]], on="key")
    c = c[c["year"] <= c["draft_year"]].sort_values("year").groupby("player_id").tail(1)
    found = pd.concat([found, c[["player_id", "pid"]]]).drop_duplicates("player_id")

    school_col = next((s for s in ["final_school", "college"] if s in p.columns), None)
    if school_col:
        rest = p[~p["player_id"].isin(found["player_id"])]
        c = rest[["player_id", "last", "draft_year", school_col]].merge(
            bt[["pid", "last", "year", "team"]], on="last")
        c = c[(c["year"] <= c["draft_year"]) & (c["year"] >= c["draft_year"] - 1) &
              (c["team"].str.lower().str[:5] == c[school_col].astype(str).str.lower().str[:5])]
        n = c.groupby("player_id")["pid"].nunique()
        c = c[c["player_id"].isin(n[n == 1].index)].drop_duplicates("player_id")
        found = pd.concat([found, c[["player_id", "pid"]]]).drop_duplicates("player_id")
    return found


def bt_features(players, bt):
    """players needs player_id, pid, draft_year. Builds the model inputs from
    each player's final Barttorvik season at or before the draft."""
    s = players[["player_id", "pid", "draft_year"]].merge(bt, on="pid")
    s = s[s["year"] <= s["draft_year"]].copy()
    s["mins"] = s["mpg"] * s["gp"]
    career = s.groupby("player_id").agg(ftm=("ftm", "sum"), mins=("mins", "sum"))
    final = s.sort_values("year").groupby("player_id").tail(1).set_index("player_id")

    out = pd.DataFrame(index=final.index)
    out["bt_year"] = final["year"]
    out["bt_team"] = final["team"]
    out["bt_role"] = final["role"]
    out["bt_bpm"] = final["bpm"]
    out["bt_obpm"] = final["obpm"]
    out["bt_dbpm"] = final["dbpm"]
    out["stl_per40"] = final["stl"] / final["mpg"] * 40
    out["blk_per40_final"] = final["blk"] / final["mpg"] * 40
    out["ft_per40_career"] = career["ftm"] / career["mins"] * 40
    out["college_minutes"] = career["mins"]
    birth = pd.to_datetime(final["birth"], errors="coerce")
    draft_day = pd.to_datetime(final["draft_year"].astype(int).astype(str) + "-06-25")
    out["age_at_draft_bt"] = ((draft_day - birth).dt.days / 365.25).round(2)
    return out.reset_index()


def prospect_score(bpm, age, weight=3.0):
    """Age-Adjusted BPM: college BPM plus `weight` points per year younger than 20."""
    return bpm + weight * (20 - age)


def percentile_of(values, reference):
    ref = np.sort(np.asarray(reference)[~np.isnan(reference)])
    return np.searchsorted(ref, np.asarray(values), side="right") / len(ref) * 100


def ensure_results():
    os.makedirs(RESULTS, exist_ok=True)

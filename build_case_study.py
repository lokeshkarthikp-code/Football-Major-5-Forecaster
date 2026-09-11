"""
Compute every number the case study page shows, once, and cache it.

    python build_case_study.py

Writes data/case_study.json. The Streamlit page reads that file and does
no computation of its own, so the page loads instantly and the numbers
are fixed rather than drifting between visits.

Re-run this after a data refresh if you want the case study updated.
Everything here is derived from caches that already exist:
  data/matches.parquet                (results + closing odds)
  data/backtest_all_leagues.parquet   (walk-forward predictions)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from plfc.ingest import load_matches
from plfc.leagues import display_name
from plfc.legfilter import shortlisted_legs, singles_return, sweep_thresholds
from plfc.multileague import load_backtest, summarise
from plfc.parlay import price_buckets, trio_counter, pooled_summary
from plfc.shortlist import tier_table

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "case_study.json"

ELITE_BAR = 0.60


def jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    print("Loading caches...")
    matches = load_matches()
    res = load_backtest()
    out: dict = {"built_at": datetime.now(timezone.utc).isoformat()}

    out["scope"] = {
        "matches": int(len(matches)),
        "leagues": int(matches["league"].nunique()),
        "seasons": int(matches["season"].nunique()),
        "backtest_predictions": int(len(res)),
    }

    # -- 1. Where we started: model vs market, per league -------------------
    print("1/5  model vs market by league")
    s = summarise(res)
    wide = (s[s["league"] != "ALL (reference only)"]
            .pivot(index="league", columns="model", values="brier")
            .reset_index())
    wide.columns.name = None
    out["brier_by_league"] = jsonable(wide.round(4))

    share = (s[["league", "share_of_market_edge"]].drop_duplicates()
             .dropna())
    out["edge_share"] = jsonable(share.round(4))

    # -- 2. The shortlist itself -------------------------------------------
    print("2/5  elite tier composition")
    tt = tier_table(matches)
    out["tier_table"] = jsonable(tt)

    # -- 3. Trio parlays ----------------------------------------------------
    print("3/5  three-leg parlays (slow)")
    trios = trio_counter(matches, k=3, min_rate=ELITE_BAR, min_weeks=8)
    out["trio_pooled"] = jsonable(pooled_summary(trios).round(4))
    out["trio_best"] = jsonable(trios.head(5).round(3))
    out["trio_worst"] = jsonable(trios.tail(5).round(3))

    # -- 4. The central chart: won vs needed to win, by price ---------------
    print("4/5  price buckets")
    pb = price_buckets(matches, k=3, min_rate=ELITE_BAR, min_weeks=30)
    out["price_buckets"] = jsonable(pb.round(4))

    # -- 5. The sharpest result: disagreement is anti-predictive ------------
    print("5/5  edge filter sweep")
    legs = shortlisted_legs(res, matches, min_rate=0.65)
    out["baseline_legs"] = {k: jsonable(v) for k, v in singles_return(legs).items()}
    out["sweep_prob"] = jsonable(sweep_thresholds(legs, by="model_p").round(4))
    out["sweep_ev"] = jsonable(sweep_thresholds(legs, by="ev").round(4))

    # -- 6. Correct score vs a constant guess ------------------------------
    if {"likely_score", "actual_home_goals"}.issubset(res.columns):
        print("6/6  correct score")
        d = res.dropna(subset=["actual_home_goals", "actual_away_goals"]).copy()
        d["actual_score"] = (d["actual_home_goals"].astype(int).astype(str)
                             + "-" + d["actual_away_goals"].astype(int).astype(str))
        hit = float((d["likely_score"].astype(str) == d["actual_score"]).mean())
        always_11 = float((d["actual_score"] == "1-1").mean())
        by_lg = (d.assign(h=d["likely_score"].astype(str) == d["actual_score"])
                 .groupby("league")["h"].agg(["size", "mean"])
                 .reset_index()
                 .rename(columns={"size": "n", "mean": "hit_rate"}))
        by_lg["league"] = by_lg["league"].map(display_name)
        out["correct_score"] = {
            "model_hit_rate": hit,
            "always_1_1": always_11,
            "n": int(len(d)),
            "breakeven_odds": 1 / hit,
            "share_predicting_1_1": float(
                (d["likely_score"].astype(str) == "1-1").mean()),
            "by_league": jsonable(by_lg.round(4)),
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
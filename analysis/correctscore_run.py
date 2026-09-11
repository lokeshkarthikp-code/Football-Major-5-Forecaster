"""
Does the model's most likely scoreline actually land 12.4% of the time?

    python correctscore_run.py

Reads data/backtest_all_leagues.parquet -- 12,000+ walk-forward
predictions the model had not seen when it made them. `likely_score` is
the single most likely cell of its scoreline grid.

WHY THIS MATTERS MORE THAN IT LOOKS
-----------------------------------
Every correct-score betting calculation runs off one number: how often the
modal scoreline is right. At 12.4% you break even around odds of 8.06. At
10% you need 10.0. The gap between those two is the difference between a
marginal bet and a steady loss, and nothing in this project has ever
measured it -- every Brier score reported so far is on match result only.

THE BASELINE THAT MATTERS
-------------------------
A hit rate means nothing on its own. Always guessing 1-1 lands maybe 9-11%
of the time by itself, because football is low-scoring and 1-1 is common.
If the model matches that, its scoreline grid is adding nothing over a
constant guess, and the "2 exact scores from 10" story is really a story
about how often 1-1 happens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plfc.multileague import load_backtest

pd.set_option("display.width", 200)

res = load_backtest()

need = {"likely_score", "actual_home_goals", "actual_away_goals"}
missing = need - set(res.columns)
if missing:
    raise SystemExit(
        f"Backtest cache is missing {sorted(missing)}.\n"
        "Re-run `python -m plfc.multileague` with the current backtest.py."
    )

d = res.dropna(subset=["actual_home_goals", "actual_away_goals"]).copy()
d["actual_score"] = (d["actual_home_goals"].astype(int).astype(str) + "-"
                     + d["actual_away_goals"].astype(int).astype(str))
d["hit"] = d["likely_score"].astype(str) == d["actual_score"]

n, hits = len(d), int(d["hit"].sum())
rate = hits / n
se = float(np.sqrt(rate * (1 - rate) / n))

print(f"Walk-forward predictions: {n:,}")
print(f"Exact scoreline correct:  {hits:,}  ({rate:.2%})")
print(f"95% interval:             {rate - 1.96*se:.2%} to {rate + 1.96*se:.2%}")
print(f"Breakeven odds needed:    {1/rate:.2f}")

# --- baselines: is the grid beating a constant guess? ----------------------
print("\n=== Baselines ===")
for guess in ("1-1", "1-0", "2-1"):
    r = float((d["actual_score"] == guess).mean())
    print(f"  always guess {guess}: {r:.2%}   (breakeven odds {1/r:.2f})" if r
          else f"  always guess {guess}: never")

best_constant = d["actual_score"].value_counts(normalize=True).head(1)
print(f"  best possible constant guess: {best_constant.index[0]} "
      f"at {best_constant.iloc[0]:.2%}")
print(f"\n  model beats best constant by: "
      f"{(rate - best_constant.iloc[0]) * 100:+.2f} percentage points")

# --- what the model picks, and how each pick does --------------------------
print("\n=== By predicted scoreline ===")
by_pred = (d.groupby("likely_score")
           .agg(times_predicted=("hit", "size"), correct=("hit", "sum"))
           .assign(hit_rate=lambda x: x["correct"] / x["times_predicted"],
                   breakeven_odds=lambda x: 1 / x["hit_rate"].replace(0, np.nan))
           .sort_values("times_predicted", ascending=False))
by_pred["share_of_bets"] = by_pred["times_predicted"] / n
print(by_pred.head(8).round(4).to_string())

print("\n=== By league ===")
if "league" in d.columns:
    by_lg = (d.groupby("league")
             .agg(n=("hit", "size"), correct=("hit", "sum"))
             .assign(hit_rate=lambda x: x["correct"] / x["n"],
                     breakeven_odds=lambda x: 1 / x["hit_rate"]))
    print(by_lg.round(4).to_string())

# --- what it means for the weekly plan ------------------------------------
print("\n=== 10 bets x $10 x 5 leagues = $500/week ===")
staked = 500.0
print(f"{'odds':>6} {'wins/wk':>8} {'return':>9} {'profit/wk':>10} {'ROI':>8} {'38 weeks':>11}")
for o in (6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0):
    wins = 50 * rate
    ret = wins * 10 * o
    prof = ret - staked
    print(f"{o:>6.2f} {wins:>8.2f} {ret:>9.0f} {prof:>+10.0f} "
          f"{prof/staked:>+8.1%} {prof*38:>+11,.0f}")

print(f"\nYou need odds above {1/rate:.2f} on every bet just to break even.")
print("Common scorelines (1-0, 1-1, 2-1) typically price 6.00-9.00, and are")
print("the most efficiently priced part of the correct-score market.")
"""
Two routes to a result prediction. Which one is better?

    python route_compare.py

THE QUESTION
------------
The model produces two opinions about every match and nobody has checked
which to trust:

  argmax route
      pick whichever of home_win / draw / away_win is largest.

  scoreline route
      take the single most likely SCORELINE and read the result off it.
      1-1 implies a draw, 2-0 implies a home win.

These disagree often, and they disagree in a structured way. Draw
probability rarely exceeds both win probabilities, so the argmax route
almost never predicts a draw -- while the scoreline route predicts 1-1
constantly and therefore predicts draws constantly. Around 25% of
football matches are drawn, so a route that never says "draw" gives up
that entire category.

WHY A TEN-MATCH SAMPLE CANNOT SETTLE IT
---------------------------------------
A week with five draws will make the scoreline route look brilliant. A
week with none will make it look broken. Both are noise. This runs the
comparison over every walk-forward prediction instead.

WHAT TO LOOK AT
---------------
Accuracy is the weakest of the measures here and is reported because it
is the one people ask for. The route that wins on accuracy may still be
the worse forecaster. Draw recall is the diagnostic that explains the
difference; the confusion counts show where each route actually spends
its predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plfc.multileague import load_backtest
from plfc.leagues import display_name

pd.set_option("display.width", 220)

OUTCOMES = ["home_win", "draw", "away_win"]


def result_from_score(s) -> str | None:
    """Read a result off a 'H-A' scoreline string."""
    try:
        h, a = str(s).split("-")
        h, a = int(h), int(a)
    except (ValueError, AttributeError):
        return None
    return "home_win" if h > a else "away_win" if h < a else "draw"


res = load_backtest()
need = {"likely_score", "actual"}
if not need.issubset(res.columns):
    raise SystemExit(
        f"Backtest cache missing {sorted(need - set(res.columns))}. "
        "Re-run `python -m plfc.multileague`."
    )

d = res.dropna(subset=["actual"]).copy()
d["argmax_pick"] = d[OUTCOMES].idxmax(axis=1)
d["score_pick"] = d["likely_score"].map(result_from_score)
d = d.dropna(subset=["score_pick"])

d["argmax_hit"] = d["argmax_pick"] == d["actual"]
d["score_hit"] = d["score_pick"] == d["actual"]

n = len(d)
print(f"Predictions compared: {n:,}\n")

# --- headline -------------------------------------------------------------
a_acc, s_acc = d["argmax_hit"].mean(), d["score_hit"].mean()

# Paired standard error: both routes are scored on the SAME matches, so
# the per-match differences are paired and their spread is the right
# yardstick -- much tighter than treating the two accuracies as
# independent samples.
diff = d["score_hit"].astype(float) - d["argmax_hit"].astype(float)
se = float(diff.std(ddof=1) / np.sqrt(n))

print("=== Accuracy ===")
print(f"  argmax route     {a_acc:.4f}")
print(f"  scoreline route  {s_acc:.4f}")
print(f"  difference       {s_acc - a_acc:+.4f}   (paired SE {se:.4f})")
if abs(s_acc - a_acc) > 2 * se:
    better = "scoreline" if s_acc > a_acc else "argmax"
    print(f"  -> {better} route is genuinely better (beyond two SE)")
else:
    print("  -> inside two standard errors: no real difference")

# --- what each route actually predicts ------------------------------------
print("\n=== What each route predicts ===")
mix = pd.DataFrame({
    "argmax": d["argmax_pick"].value_counts(normalize=True),
    "scoreline": d["score_pick"].value_counts(normalize=True),
    "actually happened": d["actual"].value_counts(normalize=True),
}).reindex(OUTCOMES).fillna(0.0)
print(mix.round(4).to_string())

# --- per outcome: does each route ever catch it? --------------------------
print("\n=== Recall: of the matches that ended this way, how many were called? ===")
rows = []
for oc in OUTCOMES:
    sub = d[d["actual"] == oc]
    rows.append({
        "outcome": oc,
        "n": len(sub),
        "argmax_recall": float((sub["argmax_pick"] == oc).mean()),
        "scoreline_recall": float((sub["score_pick"] == oc).mean()),
    })
print(pd.DataFrame(rows).round(4).to_string(index=False))

# --- precision: when a route says X, how often is it right? ---------------
print("\n=== Precision: when a route says this, how often is it right? ===")
rows = []
for oc in OUTCOMES:
    am = d[d["argmax_pick"] == oc]
    sc = d[d["score_pick"] == oc]
    rows.append({
        "says": oc,
        "argmax_n": len(am),
        "argmax_precision": float((am["actual"] == oc).mean()) if len(am) else np.nan,
        "scoreline_n": len(sc),
        "scoreline_precision": float((sc["actual"] == oc).mean()) if len(sc) else np.nan,
    })
print(pd.DataFrame(rows).round(4).to_string(index=False))

# --- where they disagree --------------------------------------------------
dis = d[d["argmax_pick"] != d["score_pick"]]
print(f"\n=== The two routes disagree on {len(dis):,} matches "
      f"({len(dis)/n:.1%}) ===")
if len(dis):
    print(f"  argmax right, scoreline wrong:  {int((dis['argmax_hit'] & ~dis['score_hit']).sum()):,}")
    print(f"  scoreline right, argmax wrong:  {int((dis['score_hit'] & ~dis['argmax_hit']).sum()):,}")
    print(f"  both wrong:                     {int((~dis['argmax_hit'] & ~dis['score_hit']).sum()):,}")

# --- per league -----------------------------------------------------------
if "league" in d.columns:
    print("\n=== By league ===")
    lg = (d.groupby("league")
          .agg(n=("argmax_hit", "size"),
               argmax=("argmax_hit", "mean"),
               scoreline=("score_hit", "mean"))
          .reset_index())
    lg["difference"] = lg["scoreline"] - lg["argmax"]
    lg["league"] = lg["league"].map(display_name)
    print(lg.round(4).to_string(index=False))

print("\nNote: accuracy is the weakest measure of a probabilistic forecast.")
print("The argmax route is the one backed by the model's Brier and log loss;")
print("the scoreline route collapses the grid to a single cell and throws")
print("the uncertainty away. Read the recall table before drawing a lesson.")
"""
Can a hybrid rule beat both routes?

    python route_hybrid.py

The precision table from route_compare.py says two things at once:

  * when the scoreline route calls a HOME or AWAY win it is much more
    precise than argmax (0.640 and 0.589 against 0.539 and 0.500)
  * when it calls a draw it is right 27.7% of the time, barely above the
    25.3% base rate, and 60% of its calls are draws

Meanwhile argmax has effectively abandoned draws: 67 calls out of 12,413,
catching 0.6% of the 3,137 draws that happened. Not because the model is
blind to them -- it assigns draws 25-29% routinely -- but because a 29%
draw never outranks a 40% home win.

So both routes fail, in opposite directions. The rules below try to take
the useful half of each.

THE TRAP IN THIS KIND OF TEST
-----------------------------
Trying six rules on one dataset and reporting the best is how you find
noise. With 12,413 matches the differences here are large enough that
this matters less than usual, but the honest procedure is still to pick
the rule on early data and score it on later data it has never seen. That
is what the split at the bottom does, and the OUT-OF-SAMPLE column is the
only one worth quoting.

AND A LIMIT WORTH STATING
-------------------------
Every number here is accuracy. Accuracy ignores how confident a forecast
was, so a rule can win on it while being a worse probabilistic forecast.
None of these rules change the model's probabilities -- they only change
which single label gets displayed. Brier and log loss are unaffected by
anything in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plfc.leagues import display_name
from plfc.multileague import load_backtest

pd.set_option("display.width", 220)
OUTCOMES = ["home_win", "draw", "away_win"]


def result_from_score(s):
    try:
        h, a = str(s).split("-")
        h, a = int(h), int(a)
    except (ValueError, AttributeError):
        return None
    return "home_win" if h > a else "away_win" if h < a else "draw"


# ---------------------------------------------------------------------------
# Decision rules. Each maps a row to a predicted outcome.
# ---------------------------------------------------------------------------

def rule_argmax(d):
    return d["argmax_pick"]


def rule_scoreline(d):
    return d["score_pick"]


def rule_hybrid(d):
    """Trust the scoreline when it names a winner; fall back otherwise."""
    return np.where(d["score_pick"] == "draw", d["argmax_pick"], d["score_pick"])


def make_draw_threshold(t):
    """Call a draw whenever the model gives it at least `t`, else argmax.

    Attacks the blindness directly: argmax never surfaces a draw because
    a draw probability almost never leads the vector. Lowering the bar
    for that one outcome is the minimal fix.
    """
    def rule(d):
        return np.where(d["draw"] >= t, "draw", d["argmax_pick"])
    rule.__name__ = f"draw if p>={t:.2f}"
    return rule


def make_hybrid_draw_threshold(t):
    """Scoreline names a winner -> take it. Otherwise draw only if p>=t."""
    def rule(d):
        winner = d["score_pick"] != "draw"
        return np.where(winner, d["score_pick"],
                        np.where(d["draw"] >= t, "draw", d["argmax_pick"]))
    rule.__name__ = f"hybrid + draw if p>={t:.2f}"
    return rule


def score(d, pred) -> dict:
    pred = pd.Series(pred, index=d.index)
    hit = pred == d["actual"]
    out = {"accuracy": float(hit.mean()), "n": len(d)}
    for oc in OUTCOMES:
        said = pred == oc
        out[f"calls_{oc}"] = int(said.sum())
        out[f"prec_{oc}"] = (float((d.loc[said, "actual"] == oc).mean())
                             if said.any() else np.nan)
    return out


# ---------------------------------------------------------------------------

res = load_backtest()
d = res.dropna(subset=["actual"]).copy()
d["argmax_pick"] = d[OUTCOMES].idxmax(axis=1)
d["score_pick"] = d["likely_score"].map(result_from_score)
d = d.dropna(subset=["score_pick"]).reset_index(drop=True)
d["date"] = pd.to_datetime(d["date"])
d = d.sort_values("date").reset_index(drop=True)

rules = {
    "argmax": rule_argmax,
    "scoreline": rule_scoreline,
    "hybrid (scoreline, argmax on draw)": rule_hybrid,
}
for t in (0.24, 0.26, 0.28, 0.30):
    rules[f"draw if p>={t:.2f}"] = make_draw_threshold(t)
for t in (0.26, 0.28, 0.30):
    rules[f"hybrid + draw if p>={t:.2f}"] = make_hybrid_draw_threshold(t)

print(f"Predictions: {len(d):,}   "
      f"actual draws: {(d['actual'] == 'draw').mean():.1%}\n")

print("=== All rules, full sample (IN-SAMPLE — see the split below) ===")
rows = []
for name, fn in rules.items():
    s = score(d, fn(d))
    rows.append({
        "rule": name, "accuracy": s["accuracy"],
        "draw_calls": s["calls_draw"], "draw_precision": s["prec_draw"],
        "home_precision": s["prec_home_win"],
        "away_precision": s["prec_away_win"],
    })
full = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
print(full.round(4).to_string(index=False))

# --- the honest version ---------------------------------------------------
cut = d["date"].quantile(0.5)
early, late = d[d["date"] < cut], d[d["date"] >= cut]
print(f"\n=== Time split: pick on early, score on late ===")
print(f"early {len(early):,} matches (to {cut.date()}), "
      f"late {len(late):,} matches")

rows = []
for name, fn in rules.items():
    rows.append({
        "rule": name,
        "in_sample_early": score(early, fn(early))["accuracy"],
        "out_of_sample_late": score(late, fn(late))["accuracy"],
    })
split = pd.DataFrame(rows)
split["drop"] = split["out_of_sample_late"] - split["in_sample_early"]
split = split.sort_values("in_sample_early", ascending=False)
print(split.round(4).to_string(index=False))

best_early = split.iloc[0]
base_late = split.loc[split["rule"] == "argmax", "out_of_sample_late"].iloc[0]
gain = best_early["out_of_sample_late"] - base_late

hit_best = pd.Series(rules[best_early["rule"]](late), index=late.index) == late["actual"]
hit_base = late["argmax_pick"] == late["actual"]
diff = hit_best.astype(float) - hit_base.astype(float)
se = float(diff.std(ddof=1) / np.sqrt(len(late)))

print(f"\nBest rule on early data: {best_early['rule']!r}")
print(f"  out-of-sample accuracy   {best_early['out_of_sample_late']:.4f}")
print(f"  argmax on the same data  {base_late:.4f}")
print(f"  difference               {gain:+.4f}  (paired SE {se:.4f})")
print(f"  -> {'real' if abs(gain) > 2*se else 'inside two SE, not distinguishable'}")

# --- per league on the winner --------------------------------------------
if "league" in d.columns:
    fn = rules[best_early["rule"]]
    late = late.copy()
    late["best_hit"] = pd.Series(fn(late), index=late.index) == late["actual"]
    late["argmax_hit"] = late["argmax_pick"] == late["actual"]
    lg = (late.groupby("league")
          .agg(n=("best_hit", "size"), argmax=("argmax_hit", "mean"),
               best=("best_hit", "mean")).reset_index())
    lg["difference"] = lg["best"] - lg["argmax"]
    lg["league"] = lg["league"].map(display_name)
    print(f"\n=== Out-of-sample by league: {best_early['rule']} vs argmax ===")
    print(lg.round(4).to_string(index=False))
"""
Walk-forward evaluation.

WHY NOT K-FOLD
--------------
Shuffled cross-validation on time series leaks the future into the past --
the model would train on May to predict September. Every reported number
would be optimistic and meaningless. We instead walk forward: fit on
everything strictly before date D, predict matches on date D, advance.
That is exactly the information a real forecaster has.

WHAT WE MEASURE
---------------
Accuracy is the weakest metric here and we report it mainly because people
ask. Football outcomes are near the noise ceiling; bookmaker closing odds
land around 53-55% on three-way results, and they see far more than we do.
A model claiming 70% has a bug.

The metrics that matter:

Brier score (multiclass)
    Mean squared error of the probability vector. Rewards being right AND
    appropriately uncertain. Lower is better.

Log loss
    Punishes confident errors harshly. Sensitive to miscalibration.

Calibration curve
    Of the matches where we said 30%, did roughly 30% happen? A model can
    be accurate and badly calibrated, and calibration is what you actually
    need if the output informs a decision.

The benchmark is the closing line, not zero. "Within X of the market" is
an honest, defensible claim; "68% accurate" usually is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import DixonColes, odds_to_probs

OUTCOMES = ["home_win", "draw", "away_win"]


def actual_outcome(hg: float, ag: float) -> str:
    if hg > ag:
        return "home_win"
    if hg < ag:
        return "away_win"
    return "draw"


def brier(probs: pd.DataFrame, actual: pd.Series) -> float:
    """Multiclass Brier score. Range 0 (perfect) to 2 (maximally wrong)."""
    onehot = pd.get_dummies(actual).reindex(columns=OUTCOMES, fill_value=0)
    return float(((probs[OUTCOMES].to_numpy() - onehot.to_numpy()) ** 2).sum(axis=1).mean())


def log_loss(probs: pd.DataFrame, actual: pd.Series, eps: float = 1e-15) -> float:
    onehot = pd.get_dummies(actual).reindex(columns=OUTCOMES, fill_value=0)
    p = np.clip(probs[OUTCOMES].to_numpy(), eps, 1)
    return float(-(onehot.to_numpy() * np.log(p)).sum(axis=1).mean())


def accuracy(probs: pd.DataFrame, actual: pd.Series) -> float:
    pred = probs[OUTCOMES].idxmax(axis=1)
    return float((pred.to_numpy() == actual.to_numpy()).mean())


def calibration_table(probs: pd.DataFrame, actual: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency, pooled across all three outcomes."""
    rows = []
    for oc in OUTCOMES:
        rows.append(pd.DataFrame({
            "predicted": probs[oc].to_numpy(),
            "occurred": (actual.to_numpy() == oc).astype(float),
        }))
    long = pd.concat(rows, ignore_index=True)
    long["bin"] = pd.cut(long["predicted"], np.linspace(0, 1, bins + 1),
                         include_lowest=True)
    out = long.groupby("bin", observed=True).agg(
        mean_predicted=("predicted", "mean"),
        observed_rate=("occurred", "mean"),
        n=("occurred", "size"),
    ).reset_index()
    return out.dropna()


def walk_forward(matches: pd.DataFrame, *, xi: float = 0.0018,
                 shrinkage: float = 8.0, min_train_matches: int = 760,
                 step_days: int = 7) -> pd.DataFrame:
    """Refit periodically, predict forward. Returns one row per test match.

    step_days=7 refits weekly, mirroring how the deployed app behaves. Refit
    cadence is a real cost/accuracy tradeoff and worth stating explicitly.
    """
    df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) <= min_train_matches:
        raise ValueError(f"Need > {min_train_matches} matches; got {len(df)}.")

    start = df.loc[min_train_matches, "date"]
    end = df["date"].max()

    preds: list[dict] = []
    cursor = start
    while cursor <= end:
        nxt = cursor + pd.Timedelta(days=step_days)
        window = df[(df["date"] >= cursor) & (df["date"] < nxt)]
        if window.empty:
            cursor = nxt
            continue

        try:
            model = DixonColes(xi=xi, shrinkage=shrinkage).fit(df, as_of=cursor)
        except ValueError:
            cursor = nxt
            continue

        for _, r in window.iterrows():
            p = model.predict(r["home_team"], r["away_team"])
            preds.append({
                "date": r["date"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "home_win": p["home_win"],
                "draw": p["draw"],
                "away_win": p["away_win"],
                "actual": actual_outcome(r["home_goals"], r["away_goals"]),
                "odds_home": r.get("odds_home", np.nan),
                "odds_draw": r.get("odds_draw", np.nan),
                "odds_away": r.get("odds_away", np.nan),
            })
        cursor = nxt

    return pd.DataFrame(preds)


def evaluate(results: pd.DataFrame) -> pd.DataFrame:
    """Compare the model against baselines and the closing line."""
    actual = results["actual"]
    rows = []

    rows.append({
        "model": "Dixon-Coles",
        "brier": brier(results, actual),
        "log_loss": log_loss(results, actual),
        "accuracy": accuracy(results, actual),
        "n": len(results),
    })

    # Base-rate baseline: league-average frequencies, same for every match.
    rates = actual.value_counts(normalize=True)
    flat = pd.DataFrame({oc: [rates.get(oc, 0.0)] * len(results) for oc in OUTCOMES})
    rows.append({
        "model": "Base rates only",
        "brier": brier(flat, actual),
        "log_loss": log_loss(flat, actual),
        "accuracy": accuracy(flat, actual),
        "n": len(results),
    })

    # The real benchmark.
    has_odds = results.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    if len(has_odds) > 50:
        market = pd.DataFrame([
            odds_to_probs(r.odds_home, r.odds_draw, r.odds_away)
            for r in has_odds.itertuples()
        ], index=has_odds.index)
        rows.append({
            "model": "Bookmaker closing line",
            "brier": brier(market, has_odds["actual"]),
            "log_loss": log_loss(market, has_odds["actual"]),
            "accuracy": accuracy(market, has_odds["actual"]),
            "n": len(has_odds),
        })

    return pd.DataFrame(rows)


def tune_decay(matches: pd.DataFrame,
               grid: list[float] | None = None) -> pd.DataFrame:
    """Sweep the time-decay rate. The half-life is a finding, not a guess."""
    grid = grid or [0.0005, 0.001, 0.0018, 0.003, 0.005]
    out = []
    for xi in grid:
        res = walk_forward(matches, xi=xi)
        if res.empty:
            continue
        out.append({
            "xi": xi,
            "half_life_days": round(float(np.log(2) / xi)),
            "brier": brier(res, res["actual"]),
            "log_loss": log_loss(res, res["actual"]),
            "accuracy": accuracy(res, res["actual"]),
        })
    return pd.DataFrame(out).sort_values("brier").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    import logging

    from .ingest import DATA, load_matches

    ap = argparse.ArgumentParser(description="Walk-forward backtest.")
    ap.add_argument("--xi", type=float, default=0.0018,
                    help="Time-decay rate per day (default 0.0018 ~= 1y half-life).")
    ap.add_argument("--shrinkage", type=float, default=8.0)
    ap.add_argument("--step-days", type=int, default=14,
                    help="Refit cadence. Lower is slower but more realistic.")
    ap.add_argument("--tune", action="store_true",
                    help="Sweep the decay grid instead of a single fit. Slow.")
    ap.add_argument("--save", action="store_true",
                    help="Write per-match predictions to data/backtest_results.parquet")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    matches = load_matches()

    if args.tune:
        print("\nDecay sweep (this refits the whole history once per xi):\n")
        print(tune_decay(matches).to_string(index=False))
        return

    print(f"\nWalk-forward backtest: xi={args.xi}, step={args.step_days}d")
    print("This refits the model at every step and will take a few minutes.\n")

    res = walk_forward(matches, xi=args.xi, shrinkage=args.shrinkage,
                       step_days=args.step_days)
    if res.empty:
        print("No predictions produced -- not enough history?")
        return

    print(evaluate(res).round(4).to_string(index=False))
    print("\nCalibration (predicted vs observed):\n")
    print(calibration_table(res, res["actual"]).round(3).to_string(index=False))

    if args.save:
        out = DATA / "backtest_results.parquet"
        res.to_parquet(out, index=False)
        print(f"\nSaved {len(res)} predictions to {out}")


if __name__ == "__main__":
    _main()

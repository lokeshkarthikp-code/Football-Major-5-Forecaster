"""
Prediction ledger: a sealed, append-only record of forecasts.

WHY THIS EXISTS (and why the backtest is not enough)
-----------------------------------------------------
The walk-forward backtest is RETROSPECTIVE. It reconstructs what the model
would have said, by refitting on data available at each past date. It is
honest and leak-free, but it can be re-run with different parameters until
the numbers look good. Nobody watching can tell whether you swept fifty
configurations and reported the best.

This ledger is PROSPECTIVE. Every Wednesday it writes down what the model
predicts for matches that have not been played yet, timestamped, and never
edits that row again. Weeks later it fills in what actually happened.

That difference matters enormously for credibility. A retrospective backtest
is a claim; an accumulating prediction log is evidence. It is also the more
interesting artefact to show an interviewer: "here is every forecast I have
made since September, and here is how they turned out."

DESIGN
------
Append-only. A prediction row is never modified once written, except to fill
in the actual result. If you re-run on the same day, the existing row for
that (match, day) is kept rather than replaced -- no quiet rewriting of
history.

Multiple predictions per match are allowed and expected: a forecast made
three weeks out uses less information than one made two days out. Scoring
uses the LATEST prediction strictly before kickoff, which is what a real
forecaster would have had.

Model parameters are stored on every row. If you retune xi later, old
predictions stay attributable to the model that made them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent.parent / "data"
LEDGER = DATA / "predictions.parquet"

OUTCOMES = ["home_win", "draw", "away_win"]

COLUMNS = [
    "match_key", "predicted_at", "match_date", "season",
    "home_team", "away_team",
    "home_win", "draw", "away_win",
    "expected_home_goals", "expected_away_goals", "likely_score",
    "xi", "shrinkage", "is_cold_start",
    "actual_home_goals", "actual_away_goals", "actual", "settled_at",
]


def match_key(season, home: str, away: str) -> str:
    """Stable identity for a fixture.

    In a league season each ordered pairing occurs exactly once, so
    season+home+away is unique. Canonical team names make this safe --
    without them this key would be the single worst place for a name
    mismatch to hide.
    """
    return f"{season}:{home}:{away}"


def load_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(LEDGER)


def _empty_like() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def log_predictions(model, fixtures: pd.DataFrame, *,
                    xi: float, shrinkage: float,
                    now: datetime | None = None) -> pd.DataFrame:
    """Record predictions for unplayed fixtures. Append-only.

    Skips any (match, calendar day) already present, so re-running on the
    same day is a no-op rather than a duplicate.
    """
    now = now or datetime.now(timezone.utc)
    today = now.date().isoformat()

    if fixtures.empty:
        log.info("No fixtures to predict.")
        return load_ledger()

    existing = load_ledger()
    seen = set()
    if not existing.empty:
        seen = set(zip(
            existing["match_key"],
            pd.to_datetime(existing["predicted_at"]).dt.date.astype(str),
        ))

    rows = []
    for _, f in fixtures.iterrows():
        key = match_key(f.get("season"), f["home_team"], f["away_team"])
        if (key, today) in seen:
            continue

        p = model.predict(f["home_team"], f["away_team"])
        rows.append({
            "match_key": key,
            "predicted_at": now.isoformat(),
            "match_date": pd.to_datetime(f["date"]),
            "season": f.get("season"),
            "home_team": f["home_team"],
            "away_team": f["away_team"],
            "home_win": p["home_win"],
            "draw": p["draw"],
            "away_win": p["away_win"],
            "expected_home_goals": p["expected_home_goals"],
            "expected_away_goals": p["expected_away_goals"],
            "likely_score": p["likely_score"],
            "xi": xi,
            "shrinkage": shrinkage,
            "is_cold_start": bool(p["is_new_home"] or p["is_new_away"]),
            "actual_home_goals": np.nan,
            "actual_away_goals": np.nan,
            "actual": None,
            "settled_at": None,
        })

    if not rows:
        log.info("All fixtures already predicted today.")
        return existing

    new = pd.DataFrame(rows)
    out = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    out.to_parquet(LEDGER, index=False)
    log.info("Logged %d new prediction(s). Ledger now holds %d rows.",
             len(new), len(out))
    return out


def reconcile(results: pd.DataFrame, *, now: datetime | None = None) -> pd.DataFrame:
    """Fill in actual outcomes for predictions whose matches have been played.

    Only touches the result columns. The forecast itself is never rewritten.
    """
    ledger = load_ledger()
    if ledger.empty:
        return ledger

    played = results.dropna(subset=["home_goals", "away_goals"]).copy()
    if played.empty:
        return ledger

    played["match_key"] = [
        match_key(s, h, a) for s, h, a in
        zip(played["season"], played["home_team"], played["away_team"])
    ]
    lookup = played.set_index("match_key")[["home_goals", "away_goals"]].to_dict("index")

    now = now or datetime.now(timezone.utc)
    stamp = now.isoformat()
    filled = 0

    for i, row in ledger.iterrows():
        if pd.notna(row.get("actual")) and row.get("actual"):
            continue
        hit = lookup.get(row["match_key"])
        if hit is None:
            continue
        hg, ag = hit["home_goals"], hit["away_goals"]
        ledger.at[i, "actual_home_goals"] = hg
        ledger.at[i, "actual_away_goals"] = ag
        ledger.at[i, "actual"] = (
            "home_win" if hg > ag else "away_win" if hg < ag else "draw"
        )
        ledger.at[i, "settled_at"] = stamp
        filled += 1

    if filled:
        ledger.to_parquet(LEDGER, index=False)
        log.info("Settled %d prediction(s) against actual results.", filled)
    return ledger


def latest_before_kickoff(ledger: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per match: the last forecast made strictly before kickoff.

    A prediction made three weeks out is a weaker forecast than one made two
    days out. Scoring the most recent pre-kickoff prediction reflects what a
    forecaster would actually have used.
    """
    df = load_ledger() if ledger is None else ledger.copy()
    if df.empty:
        return df

    df["predicted_at"] = pd.to_datetime(df["predicted_at"], utc=True, errors="coerce")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    pa = df["predicted_at"].dt.tz_localize(None)
    df = df[pa < df["match_date"]]
    if df.empty:
        return df

    return (
        df.sort_values("predicted_at")
          .groupby("match_key", as_index=False)
          .last()
    )


def track_record(ledger: pd.DataFrame | None = None) -> dict:
    """Live scorecard over settled predictions.

    Deliberately reports the same metrics as the backtest, so the two are
    directly comparable. If the live numbers drift badly from the backtest,
    that is a signal worth investigating -- not something to paper over.
    """
    df = latest_before_kickoff(ledger)
    if df.empty:
        return {"n_settled": 0, "n_pending": 0}

    settled = df[df["actual"].notna() & (df["actual"] != "")]
    pending = df[df["actual"].isna() | (df["actual"] == "")]

    out: dict = {"n_settled": len(settled), "n_pending": len(pending)}
    if settled.empty:
        return out

    probs = settled[OUTCOMES]
    actual = settled["actual"]

    onehot = pd.get_dummies(actual).reindex(columns=OUTCOMES, fill_value=0)
    out["brier"] = float(
        ((probs.to_numpy() - onehot.to_numpy()) ** 2).sum(axis=1).mean()
    )
    p = np.clip(probs.to_numpy(), 1e-15, 1)
    out["log_loss"] = float(-(onehot.to_numpy() * np.log(p)).sum(axis=1).mean())
    out["accuracy"] = float(
        (probs.idxmax(axis=1).to_numpy() == actual.to_numpy()).mean()
    )
    out["first_prediction"] = str(settled["predicted_at"].min())
    return out


def update(xi: float = 0.0018, shrinkage: float = 8.0) -> pd.DataFrame:
    """One weekly cycle: settle what has been played, predict what is next.

    Order matters. Reconcile first so the ledger reflects reality, then fit
    on the freshest data, then log new forecasts.
    """
    from .ingest import load_fixtures, load_matches
    from .model import DixonColes

    matches = load_matches()
    reconcile(matches)

    model = DixonColes(xi=xi, shrinkage=shrinkage).fit(matches)
    fixtures = load_fixtures()

    # Only forecast the near horizon. Predicting May in September is noise:
    # squad and form information is worthless at that range, and it would
    # bloat the ledger with forecasts nobody would stand behind.
    if not fixtures.empty:
        fixtures = fixtures.copy()
        fixtures["date"] = pd.to_datetime(fixtures["date"])
        horizon = pd.Timestamp.now().normalize() + pd.Timedelta(days=14)
        fixtures = fixtures[fixtures["date"] <= horizon]

    return log_predictions(model, fixtures, xi=xi, shrinkage=shrinkage)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Update the prediction ledger.")
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--shrinkage", type=float, default=8.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    update(xi=args.xi, shrinkage=args.shrinkage)

    rec = track_record()
    print("\nLive track record")
    print(f"  settled: {rec['n_settled']}   pending: {rec['n_pending']}")
    if rec.get("n_settled"):
        print(f"  Brier:    {rec['brier']:.4f}")
        print(f"  log loss: {rec['log_loss']:.4f}")
        print(f"  accuracy: {rec['accuracy']:.1%}")


if __name__ == "__main__":
    _main()
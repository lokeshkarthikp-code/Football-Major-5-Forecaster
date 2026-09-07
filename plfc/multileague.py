"""
Five leagues, five models.

WHY THIS MODULE EXISTS
----------------------
Everything in `model.py` and `backtest.py` operates on one league. This is
the driver that runs them across all five and keeps the results separate.

The separation is not tidiness. Dixon-Coles estimates team strength from the
graph of who played whom. Inside a league every club meets every other twice,
so the graph is dense and strengths are well identified. Across leagues clubs
meet only in European competition -- a few dozen matches connecting several
hundred clubs. Fitting one model to the pooled table would produce a common
scale the data cannot support, and it would do it silently, with no error and
no obvious symptom. So: five fits, five backtests, five sets of ratings, and
a summary table that puts them side by side without ever mixing them.

WHAT GETS CACHED
----------------
One parquet of per-match walk-forward predictions, all leagues stacked with a
`league` column. Every downstream question -- calibration, per-league
comparison, and later the betting frequency tables -- reads that file instead
of refitting. A full run is expensive; it should happen once.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .backtest import evaluate, walk_forward
from .leagues import DEFAULT_LEAGUES, UI_ORDER, display_name
from .model import DixonColes

log = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent.parent / "data"
BACKTEST_CACHE = DATA / "backtest_all_leagues.parquet"


def available_leagues(matches: pd.DataFrame) -> list[str]:
    """Leagues present in the cached data, in display order."""
    if "league" not in matches.columns:
        return []
    present = set(matches["league"].dropna())
    ordered = [lg for lg in UI_ORDER if lg in present]
    return ordered + sorted(present - set(ordered))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_league(matches: pd.DataFrame, league: str, *,
               xi: float = 0.0018, shrinkage: float = 8.0,
               as_of: pd.Timestamp | None = None) -> DixonColes:
    """Fit one league's model on that league's matches only."""
    sub = matches[matches["league"] == league]
    if sub.empty:
        raise ValueError(f"No matches for league {league!r}.")
    return DixonColes(xi=xi, shrinkage=shrinkage).fit(sub, as_of=as_of)


def fit_all(matches: pd.DataFrame, *, leagues: list[str] | None = None,
            xi: float = 0.0018,
            shrinkage: float = 8.0) -> dict[str, DixonColes]:
    """Fit every league. A league that fails is logged and skipped.

    One league failing to fit should not cost you the other four -- the same
    reasoning as the schedule fetch in ingest.py.
    """
    leagues = leagues or available_leagues(matches) or DEFAULT_LEAGUES
    models: dict[str, DixonColes] = {}
    for lg in leagues:
        try:
            models[lg] = fit_league(matches, lg, xi=xi, shrinkage=shrinkage)
            log.info("Fitted %s: %d teams, home advantage %.3f, rho %.3f.",
                     display_name(lg), len(models[lg].teams),
                     models[lg].home_adv, models[lg].rho)
        except (ValueError, KeyError) as exc:
            log.warning("Could not fit %s (%s) -- skipping.", lg, exc)
    return models


def ratings_all(models: dict[str, DixonColes]) -> pd.DataFrame:
    """Every league's team-strength table, stacked with a league column.

    Ratings are comparable WITHIN a league and not across leagues. The column
    is there so the UI can filter, not so two leagues can be ranked together.
    """
    frames = []
    for lg, m in models.items():
        r = m.ratings()
        r.insert(0, "league", display_name(lg))
        frames.append(r)
    if not frames:
        return pd.DataFrame(columns=["league", "team", "attack", "defence",
                                     "weighted_matches", "overall"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def backtest_all(matches: pd.DataFrame, *, leagues: list[str] | None = None,
                 xi: float = 0.0018, shrinkage: float = 8.0,
                 step_days: int = 14, min_train_matches: int = 760,
                 warm_start: bool = True,
                 progress: bool = True) -> pd.DataFrame:
    """Walk-forward every league separately; return the stacked predictions.

    This is the expensive call. Each league refits roughly (seasons x 26)
    times. Warm-starting each fit from the previous one is what makes five
    leagues tractable rather than an overnight job.
    """
    leagues = leagues or available_leagues(matches) or DEFAULT_LEAGUES
    frames = []

    for lg in leagues:
        t0 = time.time()
        try:
            res = walk_forward(
                matches, league=lg, xi=xi, shrinkage=shrinkage,
                step_days=step_days, min_train_matches=min_train_matches,
                warm_start=warm_start, progress=False,
            )
        except ValueError as exc:
            log.warning("Backtest skipped for %s (%s).", lg, exc)
            continue

        if res.empty:
            log.warning("Backtest produced no predictions for %s.", lg)
            continue

        res["league"] = lg
        frames.append(res)
        if progress:
            print(f"  {display_name(lg):<16} {len(res):>5} predictions "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Per-league scorecard: model vs base rates vs the closing line.

    Each league is scored against its OWN market and its OWN base rates. A
    pooled row is appended for reference, but the per-league rows are the
    informative ones -- leagues differ in how predictable they are, and a
    pooled Brier hides that.
    """
    if results.empty:
        return pd.DataFrame()

    rows = []
    for lg, grp in results.groupby("league"):
        ev = evaluate(grp, label="Dixon-Coles")
        ev.insert(0, "league", display_name(lg))
        rows.append(ev)

    pooled = evaluate(results, label="Dixon-Coles")
    pooled.insert(0, "league", "ALL (reference only)")
    rows.append(pooled)

    out = pd.concat(rows, ignore_index=True)

    # Headline framing from the project history: what fraction of the
    # market's improvement over base rates does the model capture?
    edges = []
    for lg, grp in out.groupby("league", sort=False):
        d = grp.set_index("model")["brier"]
        try:
            base, model, market = (d["Base rates only"], d["Dixon-Coles"],
                                   d["Bookmaker closing line"])
            # The ratio is only meaningful when the market clears base rates
            # by a real margin. If the denominator is near zero the number
            # explodes and means nothing, so report nothing rather than a
            # figure that looks like a finding.
            share = ((base - model) / (base - market)
                     if (base - market) > 0.01 else float("nan"))
        except KeyError:
            share = float("nan")
        edges.append({"league": lg, "share_of_market_edge": share})
    share = pd.DataFrame(edges)

    return out.merge(share, on="league", how="left")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def save_backtest(results: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or BACKTEST_CACHE
    path.parent.mkdir(exist_ok=True)
    results.to_parquet(path, index=False)
    return path


def load_backtest(path: Path | None = None) -> pd.DataFrame:
    path = path or BACKTEST_CACHE
    if not path.exists():
        raise FileNotFoundError(
            "No cached multi-league backtest. Run: python -m plfc.multileague"
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    from .ingest import load_matches

    ap = argparse.ArgumentParser(
        description="Fit and walk-forward backtest every league separately.")
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--shrinkage", type=float, default=8.0)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--leagues", nargs="*", default=None,
                    help="League keys. Default: everything in the cache.")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="Disable warm starting. Slower; identical optimum.")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    matches = load_matches()

    leagues = args.leagues or available_leagues(matches)
    print(f"\nLeagues found: {', '.join(display_name(l) for l in leagues)}")
    print(f"Matches: {len(matches):,}\n")

    print("Fitting current models...")
    models = fit_all(matches, leagues=leagues, xi=args.xi,
                     shrinkage=args.shrinkage)

    print("\nWalk-forward backtest (this is the slow part):\n")
    res = backtest_all(matches, leagues=leagues, xi=args.xi,
                       shrinkage=args.shrinkage, step_days=args.step_days,
                       warm_start=not args.no_warm_start)

    if res.empty:
        print("\nNo predictions produced. Not enough history per league?")
        return

    print("\nPer-league scorecard:\n")
    print(summarise(res).round(4).to_string(index=False))

    if not args.no_save:
        p = save_backtest(res)
        print(f"\nSaved {len(res):,} predictions to {p}")

    print(f"\nFitted models: {len(models)}/{len(leagues)} leagues.")


if __name__ == "__main__":
    _main()
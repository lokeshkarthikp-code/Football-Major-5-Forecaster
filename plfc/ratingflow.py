"""
Rating movement: how each result moved a team's rating, and what that
did to the prediction for their next fixture.

    python -m plfc.ratingflow               # current season, all leagues
    python -m plfc.ratingflow --seasons 2   # current and previous season

HOW IT WORKS
------------
The model is refitted after every matchday of the season, each fit using
only matches before that date. For every match, two fits matter:

  before  fitted on everything up to the day before the match
  after   fitted on everything up to the next matchday, so it includes
          this result

Comparing them gives, per team per match:

  rating before and after, and the change
  the rating gap to the opponent going in
  the win chance the model gave going in, and the points it expected
  surprise = points won - points expected (a loss as a big favourite is
             a large negative surprise; that is what should move a rating)
  the next fixture's win chance under the before-model and the
  after-model -- the direct effect of this result on the next prediction

WHAT MOVES A RATING
-------------------
Surprise, not the result alone. Losing to a much weaker side moves a
rating far more than losing to an equal one. Goals matter too: the model
rates attack and defence from goals, so losing 0-3 moves it more than
losing 0-1. Older matches fade with time decay, so each new result
carries a little more weight than the one before.

Two things share the movement between fits. Every match played on the
same day is added at once, so an opponent's other results feed in
through the schedule. And the ratings are refitted jointly, so a rival's
result can shift everyone slightly. For a team's own match, its own
result dominates its own change.

Ratings are on each league's own scale. A 0.9 in La Liga and a 0.9 in the
Bundesliga are not the same strength.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .model import DixonColes

log = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA / "rating_history.parquet"


def _overall(m: DixonColes, team: str) -> tuple[float, float, float]:
    a, d = m._strength(team)
    return float(a), float(d), float(a + d)


def _p_win(m: DixonColes, team: str, opp: str, venue: str) -> float:
    p = m.predict(team, opp) if venue == "H" else m.predict(opp, team)
    return p["home_win"] if venue == "H" else p["away_win"]


def league_history(matches: pd.DataFrame, league: str, seasons: list[int], *,
                   fixtures: pd.DataFrame | None = None,
                   xi: float = 0.0018, shrinkage: float = 8.0,
                   progress: bool = True) -> pd.DataFrame:
    """One row per (match, team) for the chosen seasons of one league."""
    lg = matches[matches["league"] == league].copy()
    lg["date"] = pd.to_datetime(lg["date"]).dt.normalize()
    played = lg.dropna(subset=["home_goals", "away_goals"])
    target = played[played["season"].isin(seasons)].sort_values("date")
    if target.empty:
        return pd.DataFrame()

    # Full schedule (played + upcoming) to find each team's next fixture.
    sched = [target[["date", "home_team", "away_team"]]]
    if fixtures is not None and not fixtures.empty:
        f = fixtures[fixtures["league"] == league].copy()
        f["date"] = pd.to_datetime(f["date"]).dt.normalize()
        sched.append(f[["date", "home_team", "away_team"]])
    sched = pd.concat(sched).drop_duplicates()
    appearances = pd.concat([
        sched.rename(columns={"home_team": "team", "away_team": "opp"}).assign(venue="H"),
        sched.rename(columns={"away_team": "team", "home_team": "opp"}).assign(venue="A"),
    ]).sort_values("date")

    # One fit per matchday, plus one after the last, each warm-started.
    dates = sorted(target["date"].unique())
    fit_dates = dates + [dates[-1] + pd.Timedelta(days=1)]
    snaps: dict[pd.Timestamp, DixonColes] = {}
    prev, t0 = None, time.time()
    for i, d in enumerate(fit_dates):
        prev = DixonColes(xi=xi, shrinkage=shrinkage).fit(played, as_of=d, warm_start=prev)
        snaps[d] = prev
        if progress and (i % 20 == 0 or i == len(fit_dates) - 1):
            print(f"    {league}: fit {i + 1}/{len(fit_dates)} ({time.time() - t0:.0f}s)",
                  flush=True)

    rows = []
    for i, d in enumerate(dates):
        before, after = snaps[d], snaps[fit_dates[i + 1]]
        for _, r in target[target["date"] == d].iterrows():
            for team, opp, venue, gf, ga in (
                (r["home_team"], r["away_team"], "H", r["home_goals"], r["away_goals"]),
                (r["away_team"], r["home_team"], "A", r["away_goals"], r["home_goals"]),
            ):
                pred = before.predict(team, opp) if venue == "H" else before.predict(opp, team)
                pw = pred["home_win"] if venue == "H" else pred["away_win"]
                pl = pred["away_win"] if venue == "H" else pred["home_win"]
                pdraw = pred["draw"]
                pts = 3 if gf > ga else 1 if gf == ga else 0

                a0, d0, o0 = _overall(before, team)
                a1, d1, o1 = _overall(after, team)
                _, _, opp0 = _overall(before, opp)

                nxt = appearances[(appearances["team"] == team) & (appearances["date"] > d)].head(1)
                if len(nxt):
                    n = nxt.iloc[0]
                    nb = _p_win(before, team, n["opp"], n["venue"])
                    na = _p_win(after, team, n["opp"], n["venue"])
                    nxt_vals = (n["date"], n["opp"], n["venue"], nb, na, na - nb)
                else:
                    nxt_vals = (pd.NaT, None, None, np.nan, np.nan, np.nan)

                rows.append({
                    "date": d, "league": league, "season": r["season"],
                    "team": team, "opponent": opp, "venue": venue,
                    "rating_before": o0, "opp_rating_before": opp0,
                    "rating_gap": o0 - opp0,
                    "p_win": pw, "p_draw": pdraw, "p_loss": pl,
                    "expected_points": 3 * pw + pdraw,
                    "goals_for": int(gf), "goals_against": int(ga),
                    "result": "W" if pts == 3 else "D" if pts == 1 else "L",
                    "points": pts, "surprise": pts - (3 * pw + pdraw),
                    "attack_change": a1 - a0, "defence_change": d1 - d0,
                    "rating_after": o1, "rating_change": o1 - o0,
                    "next_date": nxt_vals[0], "next_opponent": nxt_vals[1],
                    "next_venue": nxt_vals[2], "next_p_win_before": nxt_vals[3],
                    "next_p_win_after": nxt_vals[4], "next_p_win_change": nxt_vals[5],
                })
    return pd.DataFrame(rows)


def build(matches: pd.DataFrame, *, seasons: list[int] | None = None,
          fixtures: pd.DataFrame | None = None, xi: float = 0.0018,
          shrinkage: float = 8.0) -> pd.DataFrame:
    seasons = seasons or [int(matches["season"].max())]
    frames = []
    for lg in sorted(matches["league"].dropna().unique()):
        try:
            part = league_history(matches, lg, seasons, fixtures=fixtures,
                                  xi=xi, shrinkage=shrinkage)
        except ValueError as exc:
            log.warning("Skipped %s (%s).", lg, exc)
            continue
        if not part.empty:
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_history() -> pd.DataFrame:
    return pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()


def _main() -> None:
    import argparse

    from .ingest import load_fixtures, load_matches

    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, default=1,
                    help="How many recent seasons to cover (default: current only).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    matches = load_matches()
    last = int(matches["season"].max())
    seasons = list(range(last - args.seasons + 1, last + 1))
    print(f"Rating movement for seasons {seasons}")
    h = build(matches, seasons=seasons, fixtures=load_fixtures())
    DATA.mkdir(exist_ok=True)
    h.to_parquet(CACHE, index=False)
    print(f"Saved {len(h):,} rows to {CACHE}")


if __name__ == "__main__":
    _main()
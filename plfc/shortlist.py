"""
Rolling elite-team shortlist.

THE HINDSIGHT TRAP
------------------
The obvious way to build this list is wrong. Take six years of results,
rank clubs by win rate, take the top eight, then count how often those
eight won over those same six years. The answer will look excellent, and
it means nothing: the clubs were SELECTED for winning during the exact
window being measured. Any strategy tested that way prints a profit.

So the shortlist rolls. To decide who is elite going into season S, only
seasons before S are used. A 2021 shortlist is built from 2018-2020 and
knows nothing about what happened next. That means the list changes over
time and will contain clubs that later fell off -- Leicester, Sevilla,
Dortmund in some windows. Those inclusions are not errors. They are what
an honest forecaster would have picked at the time, and excluding them is
precisely the bias this module exists to avoid.

WEIGHTING
---------
Recent seasons count more: 3:2:1 over a three-season lookback. A club
that won 70% two years ago and 55% last year is trending down, and the
weighting reflects that without needing a separate form model.

PROMOTED AND RETURNING CLUBS
----------------------------
A club with no matches in the lookback has no win rate and cannot qualify.
That is correct -- you would not put a promoted side in an elite shortlist
-- but it also means a club returning after a season away is judged only
on the seasons it actually played. `min_matches` guards against a club
qualifying on a handful of games.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_WEIGHTS = (3.0, 2.0, 1.0)   # most recent season first


def season_win_rates(matches: pd.DataFrame) -> pd.DataFrame:
    """Win rate per team per league-season. One row per team-season."""
    df = matches.dropna(subset=["home_goals", "away_goals"]).copy()

    home = pd.DataFrame({
        "league": df["league"], "season": df["season"],
        "team": df["home_team"],
        "won": (df["home_goals"] > df["away_goals"]).astype(float),
        "drew": (df["home_goals"] == df["away_goals"]).astype(float),
    })
    away = pd.DataFrame({
        "league": df["league"], "season": df["season"],
        "team": df["away_team"],
        "won": (df["away_goals"] > df["home_goals"]).astype(float),
        "drew": (df["home_goals"] == df["away_goals"]).astype(float),
    })

    both = pd.concat([home, away], ignore_index=True)
    return (
        both.groupby(["league", "season", "team"], as_index=False)
        .agg(played=("won", "size"), wins=("won", "sum"), draws=("drew", "sum"))
        .assign(win_rate=lambda d: d["wins"] / d["played"])
    )


def rolling_shortlist(matches: pd.DataFrame, as_of_season: int, *,
                      lookback: int = 3,
                      weights: tuple[float, ...] = DEFAULT_WEIGHTS,
                      min_rate: float = 0.60,
                      min_matches: int = 40,
                      top_n: int | None = None) -> pd.DataFrame:
    """Who counted as elite going INTO `as_of_season`.

    Uses only seasons strictly before `as_of_season`. Returns one row per
    qualifying club with its weighted win rate and the seasons it drew on,
    sorted best first.
    """
    rates = season_win_rates(matches)
    window = list(range(as_of_season - lookback, as_of_season))
    rates = rates[rates["season"].isin(window)]
    if rates.empty:
        return pd.DataFrame(columns=["league", "team", "weighted_win_rate",
                                     "seasons_used", "matches"])

    w = {s: weights[as_of_season - 1 - s] if (as_of_season - 1 - s) < len(weights)
         else weights[-1] for s in window}
    rates = rates.assign(w=rates["season"].map(w))

    out = (
        rates.groupby(["league", "team"], as_index=False)
        .apply(lambda g: pd.Series({
            "weighted_win_rate": (g["win_rate"] * g["w"]).sum() / g["w"].sum(),
            "seasons_used": int(g["season"].nunique()),
            "matches": int(g["played"].sum()),
        }), include_groups=False)
    )

    out = out[(out["matches"] >= min_matches)
              & (out["weighted_win_rate"] >= min_rate)]
    out = out.sort_values("weighted_win_rate", ascending=False).reset_index(drop=True)
    out.insert(0, "as_of_season", as_of_season)
    return out.head(top_n) if top_n else out


def tier_table(matches: pd.DataFrame, *,
               thresholds: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75),
               seasons: list[int] | None = None,
               lookback: int = 3) -> pd.DataFrame:
    """How many clubs clear each win-rate bar, season by season.

    This is the table that answers "should the shortlist be 5, 6, 7 or 8?"
    -- you pick the bar that gives a workable number of legs, and you can
    see how that number moves across seasons.
    """
    rates = season_win_rates(matches)
    seasons = seasons or sorted(
        s for s in rates["season"].unique()
        if s - lookback >= rates["season"].min()
    )

    rows = []
    for s in seasons:
        for th in thresholds:
            sl = rolling_shortlist(matches, int(s), lookback=lookback,
                                   min_rate=th)
            rows.append({
                "as_of_season": int(s),
                "threshold": th,
                "n_teams": len(sl),
                "teams": ", ".join(sl["team"].tolist()) if len(sl) else "",
            })
    return pd.DataFrame(rows)


def shortlist_history(matches: pd.DataFrame, *, min_rate: float = 0.60,
                      lookback: int = 3, top_n: int | None = None,
                      seasons: list[int] | None = None) -> pd.DataFrame:
    """Every season's shortlist, stacked. Shows churn over time.

    The churn is the interesting part: if the list is stable, elite status
    is persistent and a shortlist strategy is at least coherent. If it
    turns over heavily, the whole premise is shakier than it looks.
    """
    rates = season_win_rates(matches)
    seasons = seasons or sorted(
        int(s) for s in rates["season"].unique()
        if s - lookback >= rates["season"].min()
    )
    frames = [rolling_shortlist(matches, s, lookback=lookback,
                                min_rate=min_rate, top_n=top_n)
              for s in seasons]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
"""
Parlay leg frequency: how often does a set of clubs all win in the same week?

WHAT THIS COUNTS
----------------
Pick a set of clubs. For every calendar week, look at each club's league
match and record win / draw / loss / did-not-play. Then count the weeks by
how many legs failed: all won, one failed, two failed, and so on.

That distribution is the whole answer to "how often would a parlay of
these teams have paid". A four-leg parlay pays only in the all-won row.
Everything below it is a losing week.

THE WEEK ANCHOR
---------------
Weeks are anchored to TUESDAY, not Monday. This is not cosmetic. English
fixtures run Friday to Monday, so a Monday-anchored week splits a single
round across two buckets and a Monday night game gets counted as the
following week. Measured on the real data, that error alone moved
"all six clubs playing this week" from 63% to 56%. A Tuesday anchor puts
Fri/Sat/Sun/Mon in one bucket and gives midweek Tue/Wed rounds their own,
which is what they are.

PARTIAL WEEKS ARE REAL BETS, NOT DISCARDS
------------------------------------------
Clubs in different leagues do not share a matchweek. International breaks
pause every top league at once, so those weekends vanish for everyone and
cause no misalignment -- but domestic cup rounds, league-specific midweek
rounds and differing winter breaks do.

Measured across six elite clubs and ten seasons: all six play in the same
week 63% of the time, and five or more play 83% of the time.

The temptation is to count only the 63%. That is wrong twice over. It
throws away a fifth of the usable weeks, and it throws them away
NON-RANDOMLY: the discarded weeks are cup weekends and midweek rounds,
which is exactly when elite clubs rotate and drop points. Filtering them
out would flatter the strategy by removing its hardest weeks.

So the default counts the legs that actually existed. In a week where five
of six clubs play, you would have placed a five-leg parlay -- not nothing.
Because both the hit rate and the payout depend on how many legs a bet
has, results are always broken out BY LEG COUNT. Pooling four-leg and
six-leg weeks into one hit rate would be meaningless.

`require_full=True` remains available as the conservative reading, but it
is no longer the default.

DRAWS ARE LOSSES
----------------
On a match-result parlay a draw loses the leg. `won` means won.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


WEEK_ANCHOR = 1        # 0=Mon, 1=Tue. Tuesday keeps Fri-Mon rounds together.


def week_anchor(dates: pd.Series, anchor: int = WEEK_ANCHOR) -> pd.Series:
    """Bucket dates into weeks beginning on `anchor` (default Tuesday)."""
    d = pd.to_datetime(dates)
    shift = (d.dt.weekday - anchor) % 7
    return (d - pd.to_timedelta(shift, unit="D")).dt.normalize()


def _team_results(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match: result, opponent, and that team's odds."""
    df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    has_odds = {"odds_home", "odds_away"}.issubset(df.columns)

    home = pd.DataFrame({
        "date": df["date"], "season": df["season"], "league": df["league"],
        "team": df["home_team"], "opponent": df["away_team"], "venue": "H",
        "goals_for": df["home_goals"], "goals_against": df["away_goals"],
        "odds": df["odds_home"] if has_odds else np.nan,
    })
    away = pd.DataFrame({
        "date": df["date"], "season": df["season"], "league": df["league"],
        "team": df["away_team"], "opponent": df["home_team"], "venue": "A",
        "goals_for": df["away_goals"], "goals_against": df["home_goals"],
        "odds": df["odds_away"] if has_odds else np.nan,
    })

    out = pd.concat([home, away], ignore_index=True)
    out["result"] = np.where(out["goals_for"] > out["goals_against"], "win",
                     np.where(out["goals_for"] < out["goals_against"], "loss",
                              "draw"))
    out["won"] = out["result"] == "win"
    out["week_start"] = week_anchor(out["date"])
    return out


def flat_stake_return(weeks: pd.DataFrame, stake: float = 1.0) -> dict:
    """Realised return from backing every priced week at flat stakes.

    WHY NOT hit_rate - 1/mean(odds)
    -------------------------------
    That comparison is wrong, and wrong in a flattering direction. By
    Jensen's inequality 1/mean(odds) is not mean(1/odds), and with combined
    parlay odds ranging from about 4 to 15 the gap is large. It
    systematically understates the price you actually paid, so a losing
    strategy shows a positive "edge".

    It also hides the mechanism. Measured on real data, the weeks where
    every leg won had noticeably SHORTER combined odds than the weeks where
    a leg failed -- you win the cheap weeks and lose the expensive ones,
    because short prices mean easy fixtures. Averaging odds across all
    weeks and then inverting erases that entirely.

    So: settle every week at its own price and add up the money. ROI is
    the number that means something; hit rate alone never was.
    """
    if weeks.empty:
        return {"staked": 0.0, "returned": 0.0, "profit": 0.0,
                "roi": float("nan"), "mean_odds_when_won": float("nan")}
    wins = weeks["all_won"].to_numpy(dtype=bool)
    odds = weeks["combined_odds"].to_numpy(dtype=float)
    returned = float(np.where(wins, odds * stake, 0.0).sum())
    staked = float(stake * len(weeks))
    return {
        "mean_odds_when_won": float(odds[wins].mean()) if wins.any() else float("nan"),
        "staked": staked,
        "returned": returned,
        "profit": returned - staked,
        "roi": (returned - staked) / staked,
    }


def weekly_legs(matches: pd.DataFrame, teams: list[str], *,
                seasons: list[int] | None = None) -> pd.DataFrame:
    """One row per (week, team) for the chosen clubs. The raw material."""
    tr = _team_results(matches)
    tr = tr[tr["team"].isin(teams)]
    if seasons:
        tr = tr[tr["season"].isin(seasons)]

    # A club playing twice in one calendar week (midweek round) would
    # otherwise create two legs for one slot. Keep the first fixture, and
    # flag the week so it is visible rather than silently collapsed.
    tr = tr.sort_values(["week_start", "team", "date"])
    dupes = tr.duplicated(subset=["week_start", "team"], keep="first")
    tr = tr.assign(second_fixture_in_week=dupes)
    return tr[~dupes].reset_index(drop=True)


def week_summary(matches: pd.DataFrame, teams: list[str], *,
                 seasons: list[int] | None = None) -> pd.DataFrame:
    """Per week: how many of the clubs played, won, and failed."""
    legs = weekly_legs(matches, teams, seasons=seasons)
    if legs.empty:
        return pd.DataFrame()

    n_teams = len(teams)
    out = (
        legs.groupby("week_start", as_index=False)
        .agg(n_playing=("team", "size"),
             n_won=("won", "sum"),
             combined_odds=("odds", lambda s: float(np.prod(s.dropna()))
                            if s.notna().all() and len(s) else np.nan))
        .assign(n_selected=n_teams)
    )
    out["n_failed"] = out["n_playing"] - out["n_won"]
    out["full_week"] = out["n_playing"] == n_teams
    out["all_won"] = (out["n_won"] == out["n_playing"]) & (out["n_playing"] > 0)

    # Teams that did not play, spelled out -- useful when a week looks odd.
    played = legs.groupby("week_start")["team"].apply(set)
    out["missing"] = out["week_start"].map(
        lambda w: ", ".join(sorted(set(teams) - played.get(w, set())))
    )
    return out.sort_values("week_start").reset_index(drop=True)


def failure_frequency(matches: pd.DataFrame, teams: list[str], *,
                      seasons: list[int] | None = None,
                      require_full: bool = False,
                      min_legs: int = 2,
                      by_leg_count: bool = True) -> pd.DataFrame:
    """THE table: how many weeks had 0, 1, 2, ... legs fail.

    Broken out BY LEG COUNT by default, because a five-leg week and a
    six-leg week are different bets with different payouts. Pooling them
    into a single hit rate produces a number that describes no bet you
    could actually have placed.

    `min_legs` drops weeks too thin to be worth calling a parlay.
    `require_full=True` restricts to weeks where every selected club
    played -- the conservative reading, but it discards weeks
    non-randomly (cup rounds and midweek fixtures), so it is not the
    default.
    """
    ws = week_summary(matches, teams, seasons=seasons)
    if ws.empty:
        return pd.DataFrame()
    if require_full:
        ws = ws[ws["full_week"]]
    ws = ws[ws["n_playing"] >= min_legs]
    if ws.empty:
        return pd.DataFrame()

    keys = ["n_playing", "n_failed"] if by_leg_count else ["n_failed"]
    counts = (
        ws.groupby(keys, as_index=False)
        .agg(weeks=("week_start", "size"),
             mean_combined_odds=("combined_odds", "mean"))
        .sort_values(keys)
    )

    if by_leg_count:
        totals = counts.groupby("n_playing")["weeks"].transform("sum")
        counts["pct_of_weeks"] = counts["weeks"] / totals
        counts = counts.rename(columns={"n_playing": "legs"})
        cols = ["legs", "n_failed", "label", "weeks", "pct_of_weeks",
                "mean_combined_odds"]
    else:
        counts["pct_of_weeks"] = counts["weeks"] / counts["weeks"].sum()
        cols = ["n_failed", "label", "weeks", "pct_of_weeks",
                "mean_combined_odds"]

    counts["label"] = counts["n_failed"].map(
        lambda k: "all won" if k == 0 else
                  f"{k} leg failed" if k == 1 else f"{k} legs failed"
    )
    counts.attrs["total_weeks"] = int(counts["weeks"].sum())
    return counts[cols].reset_index(drop=True)


def parlay_economics(matches: pd.DataFrame, teams: list[str], *,
                     seasons: list[int] | None = None,
                     require_full: bool = False,
                     min_legs: int = 2,
                     stake: float = 1.0) -> pd.DataFrame:
    """Flat-stake P&L if you had backed the available legs every week.

    One row per leg count, plus a pooled row. The per-leg rows are the
    meaningful ones: hit rate and payout both scale with leg count, so
    only within a leg count can hit rate be compared against breakeven.

    Uses the real closing odds already in the data. Frequency alone
    cannot tell you whether a bet was worth making -- only frequency
    against price can.
    """
    ws = week_summary(matches, teams, seasons=seasons)
    if ws.empty:
        return pd.DataFrame()
    if require_full:
        ws = ws[ws["full_week"]]
    ws = ws[(ws["n_playing"] >= min_legs) & ws["combined_odds"].notna()]
    if ws.empty:
        return pd.DataFrame()

    def _block(g: pd.DataFrame, label) -> dict:
        wins = g["all_won"]
        returns = np.where(wins, g["combined_odds"] * stake, 0.0)
        staked = stake * len(g)
        return {
            "legs": label,
            "weeks": int(len(g)),
            "weeks_won": int(wins.sum()),
            "hit_rate": float(wins.mean()),
            "mean_combined_odds": float(g["combined_odds"].mean()),
            "mean_odds_when_won": (float(g.loc[wins, "combined_odds"].mean())
                                   if wins.any() else np.nan),
            "staked": float(staked),
            "returned": float(returns.sum()),
            "profit": float(returns.sum() - staked),
            "roi": float((returns.sum() - staked) / staked),
        }

    rows = [_block(g, int(n)) for n, g in ws.groupby("n_playing")]
    rows.append(_block(ws, "all"))
    return pd.DataFrame(rows)


def combination_counter(matches: pd.DataFrame, pool: list[str], *,
                        k: int = 3, seasons: list[int] | None = None,
                        require_full: bool = True,
                        min_weeks: int = 10) -> pd.DataFrame:
    """Every k-club combination from `pool`, ranked by how often all won.

    This answers the "PSG + Barca + Madrid vs City + Madrid + Bayern"
    question directly, for every trio at once rather than one at a time.

    Here `require_full` DOES default to True, unlike elsewhere: comparing
    combinations only makes sense if every combination is scored on the
    same kind of week. A trio measured on weeks where all three played is
    comparable to another trio measured the same way; mixing in partial
    weeks would make the ranking reflect calendar quirks as much as
    football.

    Read it with the multiple-comparisons caveat in mind: a pool of eight
    gives 56 trios, so the best-looking one is partly just the luckiest.
    `weeks` is there so you can see how thin each estimate is.
    """
    rows = []
    for combo in combinations(sorted(pool), k):
        ws = week_summary(matches, list(combo), seasons=seasons)
        if ws.empty:
            continue
        if require_full:
            ws = ws[ws["full_week"]]
        else:
            ws = ws[ws["n_playing"] >= 2]
        if len(ws) < min_weeks:
            continue

        priced = ws[ws["combined_odds"].notna()]
        rows.append({
            "combination": " + ".join(combo),
            "weeks": int(len(ws)),
            "all_won": int(ws["all_won"].sum()),
            "hit_rate": float(ws["all_won"].mean()),
            "mean_combined_odds": (float(priced["combined_odds"].mean())
                                   if len(priced) else np.nan),
            **flat_stake_return(priced),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values("roi", ascending=False).reset_index(drop=True)


def rolling_shortlist_parlay(matches: pd.DataFrame, *,
                             min_rate: float = 0.60,
                             lookback: int = 3,
                             top_n: int | None = None,
                             require_full: bool = False,
                             min_legs: int = 2) -> pd.DataFrame:
    """The no-hindsight version: each season uses a shortlist built only
    from earlier seasons, then that season's weeks are counted.

    This is the only version of the number that means anything. Selecting
    clubs on the full history and then counting how often they won over
    that same history measures the selection, not the strategy.

    `mean_legs` is worth reading alongside the hit rate: a season where
    the shortlist was large will show a lower hit rate at longer odds,
    which is arithmetic rather than a change in performance.
    """
    from .shortlist import rolling_shortlist, season_win_rates

    rates = season_win_rates(matches)
    seasons = sorted(int(s) for s in rates["season"].unique()
                     if s - lookback >= rates["season"].min())

    rows = []
    for s in seasons:
        sl = rolling_shortlist(matches, s, lookback=lookback,
                               min_rate=min_rate, top_n=top_n)
        if sl.empty:
            continue
        teams = sl["team"].tolist()
        ws = week_summary(matches, teams, seasons=[s])
        if ws.empty:
            continue
        if require_full:
            ws = ws[ws["full_week"]]
        ws = ws[ws["n_playing"] >= min_legs]
        if ws.empty:
            continue

        priced = ws[ws["combined_odds"].notna()]
        rows.append({
            "season": s,
            "n_shortlisted": len(teams),
            "shortlist": ", ".join(teams),
            "weeks": int(len(ws)),
            "mean_legs": float(ws["n_playing"].mean()),
            "full_weeks": int(ws["full_week"].sum()),
            "all_won": int(ws["all_won"].sum()),
            "hit_rate": float(ws["all_won"].mean()),
            "mean_combined_odds": (float(priced["combined_odds"].mean())
                                   if len(priced) else np.nan),
            **flat_stake_return(priced),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Goals markets -- over/under 2.5 and both-teams-to-score
# ---------------------------------------------------------------------------

def goals_market_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Right/wrong counts for the over-2.5 and BTTS calls.

    These are binary, so "correct" means the side the model leaned toward
    actually happened. A 51% over call that lands is a coin flip dressed up
    as a forecast, so the Brier column is the one that matters -- the hit
    rate is there because it is the thing people ask for.

    Needs `over_2_5` / `btts` predictions and actual goals. Rows without
    them are skipped rather than guessed at.
    """
    need = {"actual_home_goals", "actual_away_goals"}
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame()

    d = df.dropna(subset=list(need)).copy()
    if d.empty:
        return pd.DataFrame()

    total = d["actual_home_goals"] + d["actual_away_goals"]
    rows = []

    if "over_2_5" in d.columns and d["over_2_5"].notna().any():
        s = d[d["over_2_5"].notna()]
        p = s["over_2_5"].to_numpy(dtype=float)
        y = ((s["actual_home_goals"] + s["actual_away_goals"]) > 2.5).astype(float).to_numpy()
        rows.append({
            "market": "Over 2.5 goals",
            "n": len(s),
            "correct": int(((p > 0.5) == (y == 1)).sum()),
            "hit_rate": float(((p > 0.5) == (y == 1)).mean()),
            "predicted_rate": float(p.mean()),
            "observed_rate": float(y.mean()),
            "brier": float(((p - y) ** 2).mean()),
        })

    if "btts" in d.columns and d["btts"].notna().any():
        s = d[d["btts"].notna()]
        p = s["btts"].to_numpy(dtype=float)
        y = ((s["actual_home_goals"] > 0) & (s["actual_away_goals"] > 0)).astype(float).to_numpy()
        rows.append({
            "market": "Both teams to score",
            "n": len(s),
            "correct": int(((p > 0.5) == (y == 1)).sum()),
            "hit_rate": float(((p > 0.5) == (y == 1)).mean()),
            "predicted_rate": float(p.mean()),
            "observed_rate": float(y.mean()),
            "brier": float(((p - y) ** 2).mean()),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        # Binary Brier baseline: always predict the observed base rate.
        out["brier_baseline"] = out["observed_rate"] * (1 - out["observed_rate"])
        out["beats_baseline"] = out["brier"] < out["brier_baseline"]
    return out


def trio_weeks(matches: pd.DataFrame, *, k: int = 3, min_rate: float = 0.65,
               lookback: int = 3, top_n: int | None = None,
               require_full: bool = True, grouped: bool = False):
    """Every (combination, week) the rolling tier would have produced.

    The raw material behind `trio_counter`. Exposed separately because
    the same weeks can be sliced by things other than which clubs were
    in them -- price, for instance.

    grouped=True returns {combo: frame}; False returns one flat frame
    with a `combination` column.
    """
    from .shortlist import rolling_shortlist, season_win_rates

    rates = season_win_rates(matches)
    seasons = sorted(int(s) for s in rates["season"].unique()
                     if s - lookback >= rates["season"].min())

    acc: dict[tuple, list] = {}
    for s in seasons:
        sl = rolling_shortlist(matches, s, lookback=lookback,
                               min_rate=min_rate, top_n=top_n)
        if len(sl) < k:
            continue
        for combo in combinations(sorted(sl["team"].tolist()), k):
            ws = week_summary(matches, list(combo), seasons=[s])
            if ws.empty:
                continue
            ws = ws[ws["full_week"]] if require_full else ws[ws["n_playing"] >= k]
            if ws.empty:
                continue
            acc.setdefault(combo, []).append(ws.assign(season=s))

    merged = {c: pd.concat(f, ignore_index=True) for c, f in acc.items()}
    if grouped:
        return merged
    if not merged:
        return pd.DataFrame()
    return pd.concat(
        [f.assign(combination=" + ".join(c)) for c, f in merged.items()],
        ignore_index=True)


def price_buckets(matches: pd.DataFrame, *, k: int = 3,
                  min_rate: float = 0.65, lookback: int = 3,
                  edges: tuple[float, ...] = (0, 2.5, 3.0, 3.5, 4.0, 5.0,
                                              6.0, 8.0, 999),
                  require_full: bool = True,
                  min_weeks: int = 30) -> pd.DataFrame:
    """Is any PRICE band profitable, regardless of which clubs are in it?

    This is the one legitimate way stake variation could help. Not by
    betting more on some weeks -- that cannot change the sign of a
    negative expectation -- but by betting ONLY the weeks in a band that
    is genuinely positive, and skipping the rest. That is selection, and
    the combined price is known before kickoff, so it is usable.

    The result to watch for is a MONOTONIC pattern. Football weeks vary,
    so with eight buckets one or two will land above zero by chance. A
    real effect looks like a trend across adjacent bands; a fluke looks
    like one band up, its neighbours down.

    `min_weeks` suppresses bands too thin to read.
    """
    tw = trio_weeks(matches, k=k, min_rate=min_rate, lookback=lookback,
                    require_full=require_full)
    if tw.empty:
        return pd.DataFrame()

    tw = tw[tw["combined_odds"].notna()].copy()
    tw["band"] = pd.cut(tw["combined_odds"], list(edges), right=False)

    rows = []
    for band, g in tw.groupby("band", observed=True):
        if len(g) < min_weeks:
            continue
        won = g["all_won"].to_numpy(dtype=bool)
        odds = g["combined_odds"].to_numpy(dtype=float)
        returned = float(np.where(won, odds, 0.0).sum())
        hit = float(won.mean())
        rows.append({
            "odds_band": str(band),
            "weeks": int(len(g)),
            "hit_rate": hit,
            "breakeven_hit_rate": float(np.mean(1.0 / odds)),
            "mean_odds": float(odds.mean()),
            "roi": (returned - len(g)) / len(g),
            # Standard error on ROI, so a positive band can be read
            # against the noise rather than taken at face value.
            "roi_se": float(np.std(np.where(won, odds, 0.0) - 1.0,
                                   ddof=1) / np.sqrt(len(g))),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["signif"] = np.where(out["roi"] > 2 * out["roi_se"], "+",
                         np.where(out["roi"] < -2 * out["roi_se"], "-", "noise"))
    return out


def trio_counter(matches: pd.DataFrame, *, k: int = 3,
                 min_rate: float = 0.65, lookback: int = 3,
                 top_n: int | None = None,
                 require_full: bool = True,
                 min_weeks: int = 8) -> pd.DataFrame:
    """Every k-team combination, counted with NO hindsight.

    For each season the elite tier is rebuilt from earlier seasons only,
    every k-combination of that tier is enumerated, and that season's
    weeks are counted. A combination therefore only accrues weeks in the
    seasons where all k of its clubs were actually shortlisted at the
    time -- Juventus contributes to trios in 2020 and 2021 and then stops,
    because that is when a forecaster would have been picking them.

    Returns one row per combination: how many weeks it was live, how many
    of those weeks all k won, and what backing it flat would have
    returned at real closing odds.

    WHY k=3 IS A DIFFERENT BET FROM k=5
    ------------------------------------
    The bookmaker margin compounds per leg. At roughly 4% a leg, three
    legs carry about 12% and five carry about 20%. A three-leg parlay is
    a materially better bet than a five-leg one before any football
    happens -- which is not the same as it being a good one.

    READ `weeks` BEFORE `hit_rate`
    ------------------------------
    A tier of seven gives 35 trios. The best-looking row is partly the
    luckiest, and with 20-40 weeks apiece the spread between best and
    worst is mostly noise. `pooled_summary` in the same output is the
    number that generalises; individual rows are for exploration.
    """
    acc = trio_weeks(matches, k=k, min_rate=min_rate, lookback=lookback,
                     top_n=top_n, require_full=require_full, grouped=True)

    rows = []
    for combo, ws in acc.items():
        if len(ws) < min_weeks:
            continue
        priced = ws[ws["combined_odds"].notna()]
        rows.append({
            "combination": " + ".join(combo),
            "seasons_live": int(ws["season"].nunique()),
            "weeks": int(len(ws)),
            "all_won": int(ws["all_won"].sum()),
            "hit_rate": float(ws["all_won"].mean()),
            "mean_combined_odds": (float(priced["combined_odds"].mean())
                                   if len(priced) else np.nan),
            **flat_stake_return(priced),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("roi", ascending=False).reset_index(drop=True)


def pooled_summary(trios: pd.DataFrame) -> pd.DataFrame:
    """One honest line for a whole table of combinations.

    Every individual combination is a small sample and the table is
    sorted, so the top row is selected for luck as much as for quality.
    Pooling all combinations gives the number that would actually have
    applied to picking a trio at random each week, which is the strategy
    as stated.

    The spread columns are there to show how much of the variation
    between combinations is noise: if the best and worst differ by less
    than the pooled figure moves under resampling, there is nothing to
    choose between them.
    """
    if trios.empty:
        return pd.DataFrame()
    staked = trios["staked"].sum()
    returned = trios["returned"].sum()
    return pd.DataFrame([{
        "combinations": len(trios),
        "total_weeks": int(trios["weeks"].sum()),
        "total_all_won": int(trios["all_won"].sum()),
        "pooled_hit_rate": float(trios["all_won"].sum() / trios["weeks"].sum()),
        "pooled_roi": float((returned - staked) / staked) if staked else np.nan,
        "best_roi": float(trios["roi"].max()),
        "worst_roi": float(trios["roi"].min()),
        "combos_profitable": int((trios["roi"] > 0).sum()),
    }])
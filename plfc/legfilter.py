"""
Fixture-aware leg selection.

THE GAP THIS FILLS
------------------
The shortlist strategy backs every elite club every week regardless of who
they are playing. Bayern at home to a promoted side and Bayern away at
Leverkusen are treated identically, which is plainly wrong. This module
adds the missing half: keep the club shortlist, then filter to the
fixtures where the model actually likes the leg.

TWO WAYS TO FILTER, AND THEY ARE NOT THE SAME
----------------------------------------------
probability
    Keep legs where the model says the club wins with probability >= p.
    This selects EASY fixtures. It does not select MISPRICED ones -- an
    easy fixture is priced as an easy fixture, so this can raise the hit
    rate without improving returns at all. Worth measuring precisely
    because the intuition that it should help is so strong.

expected value
    Keep legs where model_probability * odds - 1 >= threshold. This is
    the real test: it selects fixtures the model thinks are underpriced,
    regardless of whether they are easy. Note this formula has no Jensen
    problem -- it settles each leg at its own price rather than comparing
    a hit rate against an averaged one.

The prior, given that the model lost to the closing line in all five
leagues, is that neither filter produces positive returns. The EV filter
is more interesting because it will select the legs where the model most
disagrees with the market, and if the model has no information the market
lacks, those are exactly the legs where the model is most wrong.

BACKTEST FIRST, THEN LIVE
-------------------------
`sweep_thresholds` answers the question on 12,000+ historical matches
with real closing odds. `suggest_legs` applies the same logic to this
week's fixtures. Do not deploy the second before reading the first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .parlay import week_anchor


# ---------------------------------------------------------------------------
# Reshape backtest output to one row per shortlisted club per match
# ---------------------------------------------------------------------------

def team_legs(results: pd.DataFrame, teams: list[str] | None = None,
              *, seasons: list[int] | None = None) -> pd.DataFrame:
    """One row per (match, team-of-interest) with that team's win
    probability and that team's price.

    `results` is the walk-forward backtest cache: model probabilities and
    closing odds for matches the model had not seen.
    """
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])

    home = pd.DataFrame({
        "date": df["date"], "league": df.get("league"), "season": df.get("season"),
        "team": df["home_team"], "opponent": df["away_team"], "venue": "H",
        "model_p": df["home_win"], "odds": df.get("odds_home"),
        "won": df["actual"] == "home_win",
    })
    away = pd.DataFrame({
        "date": df["date"], "league": df.get("league"), "season": df.get("season"),
        "team": df["away_team"], "opponent": df["home_team"], "venue": "A",
        "model_p": df["away_win"], "odds": df.get("odds_away"),
        "won": df["actual"] == "away_win",
    })

    out = pd.concat([home, away], ignore_index=True)
    if teams is not None:
        out = out[out["team"].isin(teams)]
    if seasons:
        out = out[out["season"].isin(seasons)]

    out = out.dropna(subset=["model_p", "odds"])
    out["market_p"] = 1.0 / out["odds"]          # includes the margin
    out["ev"] = out["model_p"] * out["odds"] - 1.0
    out["week_start"] = week_anchor(out["date"])
    return out.reset_index(drop=True)


def shortlisted_legs(results: pd.DataFrame, matches: pd.DataFrame, *,
                     min_rate: float = 0.65, lookback: int = 3,
                     top_n: int | None = None) -> pd.DataFrame:
    """Every backtested leg for clubs that were elite GOING INTO that season.

    The shortlist is rebuilt per season from earlier seasons only, so a
    club's inclusion in 2023 never depends on how 2023 turned out.
    """
    from .shortlist import rolling_shortlist

    frames = []
    for s in sorted(results["season"].dropna().unique()):
        sl = rolling_shortlist(matches, int(s), lookback=lookback,
                               min_rate=min_rate, top_n=top_n)
        if sl.empty:
            continue
        legs = team_legs(results, sl["team"].tolist(), seasons=[int(s)])
        if not legs.empty:
            frames.append(legs)
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame())


# ---------------------------------------------------------------------------
# Does filtering help?
# ---------------------------------------------------------------------------

def singles_return(legs: pd.DataFrame, stake: float = 1.0) -> dict:
    """Flat-stake return from backing each qualifying leg on its own.

    Singles isolate the question. If there is no edge on a single leg
    there cannot be one on a parlay of them -- the parlay only multiplies
    the margin.
    """
    if legs.empty:
        return {"legs": 0, "hit_rate": np.nan, "roi": np.nan}
    won = legs["won"].to_numpy(dtype=bool)
    odds = legs["odds"].to_numpy(dtype=float)
    returned = float(np.where(won, odds * stake, 0.0).sum())
    staked = float(stake * len(legs))
    return {
        "legs": int(len(legs)),
        "hit_rate": float(won.mean()),
        "mean_odds": float(odds.mean()),
        "mean_model_p": float(legs["model_p"].mean()),
        "staked": staked,
        "returned": returned,
        "profit": returned - staked,
        "roi": (returned - staked) / staked,
    }


def weekly_parlay_return(legs: pd.DataFrame, *, min_legs: int = 2,
                         max_legs: int | None = None,
                         stake: float = 1.0) -> pd.DataFrame:
    """Combine each week's qualifying legs into one parlay. Per leg count.

    Split by leg count because both hit rate and payout scale with it --
    a pooled figure describes no bet you could have placed.
    """
    if legs.empty:
        return pd.DataFrame()

    wk = (legs.groupby("week_start")
          .agg(n_legs=("won", "size"),
               all_won=("won", "all"),
               combined_odds=("odds", lambda s: float(np.prod(s))))
          .reset_index())
    wk = wk[wk["n_legs"] >= min_legs]
    if max_legs:
        wk = wk[wk["n_legs"] <= max_legs]
    if wk.empty:
        return pd.DataFrame()

    def block(g, label):
        won = g["all_won"].to_numpy(dtype=bool)
        odds = g["combined_odds"].to_numpy(dtype=float)
        returned = float(np.where(won, odds * stake, 0.0).sum())
        staked = float(stake * len(g))
        return {
            "legs": label, "weeks": int(len(g)),
            "weeks_won": int(won.sum()), "hit_rate": float(won.mean()),
            "mean_combined_odds": float(odds.mean()),
            "staked": staked, "returned": returned,
            "profit": returned - staked,
            "roi": (returned - staked) / staked,
        }

    rows = [block(g, int(n)) for n, g in wk.groupby("n_legs")]
    rows.append(block(wk, "all"))
    return pd.DataFrame(rows)


def sweep_thresholds(legs: pd.DataFrame, *, by: str = "model_p",
                     grid: tuple[float, ...] | None = None) -> pd.DataFrame:
    """THE table. Does filtering legs by `by` improve returns?

    by="model_p"  filters to easy fixtures
    by="ev"       filters to fixtures the model thinks are underpriced

    Read `roi_singles` first. A filter that raises the hit rate while
    leaving ROI negative has selected easy fixtures, not profitable ones,
    and the market had already priced the ease in.
    """
    if legs.empty:
        return pd.DataFrame()

    if grid is None:
        grid = ((0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
                if by == "model_p" else
                (-1.0, -0.10, -0.05, 0.0, 0.02, 0.05, 0.10, 0.20))

    rows = []
    for th in grid:
        sub = legs[legs[by] >= th]
        if sub.empty:
            continue
        s = singles_return(sub)
        par = weekly_parlay_return(sub)
        pooled = par[par["legs"] == "all"] if not par.empty else pd.DataFrame()
        rows.append({
            "threshold": th,
            "legs_kept": s["legs"],
            "pct_kept": s["legs"] / len(legs),
            "hit_rate": s["hit_rate"],
            "mean_odds": s["mean_odds"],
            "roi_singles": s["roi"],
            "parlay_weeks": int(pooled["weeks"].iloc[0]) if len(pooled) else 0,
            "roi_parlay": float(pooled["roi"].iloc[0]) if len(pooled) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Live suggestions
# ---------------------------------------------------------------------------

def suggest_legs(models: dict, fixtures: pd.DataFrame, shortlist: list[str],
                 *, min_p: float = 0.65, horizon_days: int = 8,
                 odds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank this week's shortlisted clubs by how the model likes the fixture.

    `models` maps league key -> fitted DixonColes. Legs are scored on the
    actual opponent and venue, so the same club can qualify one week and
    not the next -- which was the whole point.

    Without live odds this ranks by model probability only. That is a
    fixture-difficulty screen, not an edge screen: it cannot tell you a
    leg is underpriced, only that it looks easy. `sweep_thresholds` on
    historical data is what tells you whether that distinction matters.
    """
    if fixtures.empty:
        return pd.DataFrame()

    fx = fixtures.copy()
    fx["date"] = pd.to_datetime(fx["date"])
    now = pd.Timestamp.now().normalize()
    fx = fx[(fx["date"] >= now) & (fx["date"] <= now + pd.Timedelta(days=horizon_days))]

    rows = []
    for _, f in fx.iterrows():
        lg = f.get("league")
        model = models.get(lg)
        if model is None:
            continue
        for team, opp, venue in ((f["home_team"], f["away_team"], "H"),
                                 (f["away_team"], f["home_team"], "A")):
            if team not in shortlist:
                continue
            p = model.predict(f["home_team"], f["away_team"])
            prob = p["home_win"] if venue == "H" else p["away_win"]
            rows.append({
                "date": f["date"], "league": lg, "team": team,
                "opponent": opp, "venue": venue,
                "model_p": prob,
                "qualifies": prob >= min_p,
                "cold_start": bool(p["is_new_home"] or p["is_new_away"]),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if odds is not None and not odds.empty:
        out = out.merge(odds, on=["date", "team"], how="left")
        if "odds" in out.columns:
            out["ev"] = out["model_p"] * out["odds"] - 1.0

    return out.sort_values("model_p", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Week selection: bet some weeks, sit out others -- tested honestly
# ---------------------------------------------------------------------------

def select_weeks(legs: pd.DataFrame, *, k: int = 3, min_p: float = 0.0,
                 min_ev: float = -1.0, max_leg_odds: float = 99.0,
                 home_only: bool = False, skip_elite_clash: bool = True,
                 rank_by: str = "model_p") -> pd.DataFrame:
    """Apply one betting rule. Returns one row per week actually bet.

    Each week: drop clubs facing another shortlisted club (Madrid v
    Barcelona removes both), drop legs failing the thresholds, rank what
    is left, take the top k. Fewer than k qualifying legs -> sit out.
    """
    q = legs
    if skip_elite_clash:
        elite = set(legs["team"])
        q = q[~q["opponent"].isin(elite)]
    q = q[(q["model_p"] >= min_p) & (q["ev"] >= min_ev)
          & (q["odds"] <= max_leg_odds)]
    if home_only:
        q = q[q["venue"] == "H"]
    if q.empty:
        return pd.DataFrame()

    q = (q.sort_values(["week_start", rank_by], ascending=[True, False])
           .drop_duplicates(["week_start", "team"]))
    top = q.groupby("week_start").head(k)
    wk = (top.groupby("week_start")
             .agg(legs=("won", "size"), all_won=("won", "all"),
                  odds=("odds", "prod"), season=("season", "first"))
             .reset_index())
    return wk[wk["legs"] == k].reset_index(drop=True)


def _returns(wk: pd.DataFrame) -> dict:
    if wk.empty:
        return {"bets": 0, "hit_rate": np.nan, "mean_odds": np.nan,
                "roi": np.nan, "roi_se": np.nan}
    pay = np.where(wk["all_won"], wk["odds"], 0.0) - 1.0
    return {
        "bets": int(len(wk)),
        "hit_rate": float(wk["all_won"].mean()),
        "mean_odds": float(wk["odds"].mean()),
        "roi": float(pay.mean()),
        "roi_se": (float(pay.std(ddof=1) / np.sqrt(len(pay)))
                   if len(pay) > 1 else np.nan),
    }


def rule_grid() -> list[dict]:
    """Every rule combination tried. 576 in total."""
    grid = []
    for k in (2, 3):
        for min_p in (0.0, 0.55, 0.60, 0.65, 0.70, 0.75):
            for min_ev in (-1.0, -0.05, 0.0, 0.05):
                for max_odds in (99.0, 1.5, 1.3):
                    for home_only in (False, True):
                        for rank_by in ("model_p", "ev"):
                            grid.append(dict(k=k, min_p=min_p, min_ev=min_ev,
                                             max_leg_odds=max_odds,
                                             home_only=home_only,
                                             rank_by=rank_by))
    return grid


def fair_test(legs: pd.DataFrame, *, first_test_season: int = 2023,
              min_train_bets: int = 50) -> dict:
    """Tune on early seasons, run the chosen rule untouched on later ones.

    Every rule is scored on TRAIN seasons only. The best one (with at least
    `min_train_bets`, so a rule that fired three times cannot win) is then
    applied to TEST seasons, which played no part in choosing it. The test
    ROI is the only number that answers whether week selection works.

    Also reported: correlation between train and test ROI across rules.
    If picking weeks is a real skill, rules that did well early should
    tend to do well later. If it is noise, it sits near zero.
    """
    train = legs[legs["season"] < first_test_season]
    test = legs[legs["season"] >= first_test_season]

    rows = []
    for r in rule_grid():
        tr = _returns(select_weeks(train, **r))
        te = _returns(select_weeks(test, **r))
        rows.append({**r,
                     **{f"train_{k}": v for k, v in tr.items()},
                     **{f"test_{k}": v for k, v in te.items()}})
    allr = pd.DataFrame(rows)

    eligible = allr[allr["train_bets"] >= min_train_bets]
    ranked = eligible.sort_values("train_roi", ascending=False)
    both = eligible.dropna(subset=["train_roi", "test_roi"])
    corr = (float(both["train_roi"].corr(both["test_roi"]))
            if len(both) > 2 else np.nan)

    return {
        "rules_tested": len(allr),
        "rules_eligible": len(eligible),
        "profitable_on_train": int((eligible["train_roi"] > 0).sum()),
        "profitable_on_test": int((eligible["test_roi"] > 0).sum()),
        "train_test_correlation": corr,
        "best_rule": ranked.iloc[0] if len(ranked) else None,
        "top10": ranked.head(10),
        "train_seasons": sorted(int(s) for s in train["season"].dropna().unique()),
        "test_seasons": sorted(int(s) for s in test["season"].dropna().unique()),
    }


def _main() -> None:
    from .ingest import load_matches
    from .multileague import load_backtest

    pd.set_option("display.width", 220)
    legs = shortlisted_legs(load_backtest(), load_matches(), min_rate=0.60)
    r = fair_test(legs)
    b = r["best_rule"]
    rule_cols = ["k", "min_p", "min_ev", "max_leg_odds", "home_only", "rank_by"]

    print(f"Tuned on seasons  {r['train_seasons']}")
    print(f"Tested on seasons {r['test_seasons']}  (never seen during tuning)\n")
    print(f"Rules tested: {r['rules_tested']}   with enough bets: "
          f"{r['rules_eligible']}")
    print(f"Profitable on train: {r['profitable_on_train']}   "
          f"on test: {r['profitable_on_test']}")
    print(f"Train vs test ROI correlation across rules: "
          f"{r['train_test_correlation']:+.2f}\n")

    if b is None:
        print("No rule had enough bets on the training seasons.")
        return

    print("=== Best rule on train ===")
    print(b[rule_cols].to_string())
    print(f"\n  TRAIN  {int(b['train_bets'])} bets  hit {b['train_hit_rate']:.1%}  "
          f"ROI {b['train_roi']:+.1%}")
    print(f"  TEST   {int(b['test_bets'])} bets  hit {b['test_hit_rate']:.1%}  "
          f"ROI {b['test_roi']:+.1%}  (SE {b['test_roi_se']:.1%})")

    tb, tr, se = int(b["test_bets"]), b["test_roi"], b["test_roi_se"]
    if tb < 30:
        verdict = "too few test bets to read either way"
    elif tr > 2 * se:
        verdict = "profitable on unseen seasons, beyond two standard errors"
    elif tr < -2 * se:
        verdict = "loses on unseen seasons, beyond two standard errors"
    else:
        verdict = "inside two standard errors of zero on unseen seasons"
    print(f"  -> {verdict}")

    print("\n=== Top 10 rules on train, and what they did on test ===")
    cols = rule_cols + ["train_bets", "train_roi", "test_bets", "test_roi"]
    print(r["top10"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    _main()
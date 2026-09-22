"""
Does staking by confidence beat a flat stake?

    python -m plfc.sizing

THE QUESTION
------------
Stake more on high-confidence picks, less on low ones. Does that return
more per unit staked than betting the same amount on everything, on
seasons the rules never saw?

WHAT DECIDES IT
---------------
Return per unit, by confidence tier. Sizing can only help if high-tier
picks return more per unit than low-tier ones. If every tier returns
about the same, sizing changes the swings and nothing else. That table
prints first.

DESIGN, FIXED BEFORE ANY RESULT WAS SEEN
----------------------------------------
Legs        elite clubs, tier rebuilt each season from earlier seasons
            (0.65 win-rate bar), with walk-forward model probabilities
            and closing odds
Confidence  A model win chance
            B model edge  (win chance x odds - 1)
            C this season's win rate before the match (fewer than 3
              matches played -> neutral, placed in the middle tier)
            D agreement   min(model win chance, price-implied chance)
Tiers       low / medium / high at the terciles of the TRAIN seasons
Stakes      flat          1 / 1 / 1
            tiered        0.5 / 1 / 2
            aggressive    0 / 1 / 3   (skip low)
            proportional  2 x train-percentile of the score (mean ~1)
Bets        singles, and a weekly 3-leg parlay of that week's three
            highest-confidence legs, staked by the parlay's tier
Split       train 2019-2022, test 2023 onward. The best combination is
            chosen on train (return minus one standard error, so small
            lucky samples cannot win) and reported on test, compared
            against flat staking on the same bets, with a bootstrap range.

WHAT THIS CANNOT TEST
---------------------
Players, injuries and lineups (no data yet), and the user's own
intuition, which can only be measured forward. Closing odds are used;
prices taken earlier in the week may be better or worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEST_FROM = 2023
TIERS = ["low", "medium", "high"]
SCHEMES = {
    "flat": {"low": 1.0, "medium": 1.0, "high": 1.0},
    "tiered": {"low": 0.5, "medium": 1.0, "high": 2.0},
    "aggressive": {"low": 0.0, "medium": 1.0, "high": 3.0},
    "proportional": None,          # stake from the score's percentile
}
DEFINITIONS = {
    "A_model_win_chance": "model_p",
    "B_model_edge": "ev",
    "C_season_form": "season_form",
    "D_model_market_agree": "agree",
}


# ---------------------------------------------------------------------------
# Confidence scores
# ---------------------------------------------------------------------------

def season_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Each club's win rate this season from matches BEFORE each date."""
    d = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    long = pd.concat([
        pd.DataFrame({"date": d["date"], "season": d["season"], "team": d["home_team"],
                      "win": (d["home_goals"] > d["away_goals"]).astype(float)}),
        pd.DataFrame({"date": d["date"], "season": d["season"], "team": d["away_team"],
                      "win": (d["away_goals"] > d["home_goals"]).astype(float)}),
    ]).sort_values("date")
    g = long.groupby(["season", "team"])["win"]
    long["played_before"] = g.cumcount()
    long["wins_before"] = g.cumsum() - long["win"]
    long["season_form"] = np.where(long["played_before"] >= 3,
                                   long["wins_before"] / long["played_before"].clip(lower=1),
                                   np.nan)
    return long[["date", "team", "season_form"]].drop_duplicates(["date", "team"])


def add_scores(legs: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    out = legs.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(season_form(matches), on=["date", "team"], how="left")
    out["agree"] = np.minimum(out["model_p"], 1.0 / out["odds"])
    return out


def _cutoffs(train_scores: pd.Series) -> tuple[float, float]:
    s = train_scores.dropna()
    return float(s.quantile(1 / 3)), float(s.quantile(2 / 3))


def assign_tier(score: pd.Series, cuts: tuple[float, float]) -> pd.Series:
    lo, hi = cuts
    tier = np.where(score >= hi, "high", np.where(score >= lo, "medium", "low"))
    return pd.Series(np.where(score.isna(), "medium", tier), index=score.index)


def percentile_stake(score: pd.Series, train_scores: pd.Series) -> pd.Series:
    ref = np.sort(train_scores.dropna().to_numpy())
    pct = np.searchsorted(ref, score.fillna(np.median(ref)).to_numpy()) / max(len(ref), 1)
    return pd.Series(2.0 * pct, index=score.index)


# ---------------------------------------------------------------------------
# Bets
# ---------------------------------------------------------------------------

def singles(legs: pd.DataFrame, col: str) -> pd.DataFrame:
    b = legs[["week_start", "season", "odds", "won", col]].rename(columns={col: "score"})
    return b.assign(won=b["won"].astype(bool))


def weekly_parlays(legs: pd.DataFrame, col: str, k: int = 3) -> pd.DataFrame:
    """That week's k highest-confidence legs, combined. Score = their mean."""
    d = legs.dropna(subset=[col]) if col != "season_form" else legs.assign(
        season_form=legs["season_form"].fillna(legs["season_form"].median()))
    d = (d.sort_values(["week_start", col], ascending=[True, False])
          .drop_duplicates(["week_start", "team"]))
    top = d.groupby("week_start").head(k)
    wk = (top.groupby("week_start")
             .agg(n=("won", "size"), won=("won", "all"), odds=("odds", "prod"),
                  score=(col, "mean"), season=("season", "first"))
             .reset_index())
    return wk[wk["n"] == k].drop(columns="n")


def stake(bets: pd.DataFrame, scheme: str, train: pd.DataFrame) -> pd.Series:
    if scheme == "proportional":
        return percentile_stake(bets["score"], train["score"])
    tiers = assign_tier(bets["score"], _cutoffs(train["score"]))
    return tiers.map(SCHEMES[scheme]).astype(float)


def roi(bets: pd.DataFrame, stakes: pd.Series) -> dict:
    staked = float(stakes.sum())
    if staked <= 0:
        return {"bets": 0, "staked": 0.0, "profit": 0.0, "roi": np.nan, "roi_se": np.nan}
    pay = stakes.to_numpy() * np.where(bets["won"], bets["odds"] - 1.0, -1.0)
    profit = float(pay.sum())
    n = int((stakes > 0).sum())
    # ROI standard error: spread of per-bet profit, scaled by average stake.
    se = (float(pay.std(ddof=1) / np.sqrt(len(pay)) / stakes.mean())
          if len(pay) > 1 and stakes.mean() > 0 else np.nan)
    return {"bets": n, "staked": staked, "profit": profit,
            "roi": profit / staked, "roi_se": se}


def boot_diff(bets: pd.DataFrame, s_a: pd.Series, s_b: pd.Series,
              n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """2.5/97.5 percentiles of ROI(a) - ROI(b), resampling whole weeks."""
    rng = np.random.default_rng(seed)
    pay = np.where(bets["won"], bets["odds"] - 1.0, -1.0)
    wk = bets["week_start"].to_numpy()
    uniq = np.unique(wk)
    idx = {w: np.flatnonzero(wk == w) for w in uniq}
    a, b = s_a.to_numpy(), s_b.to_numpy()
    out = []
    for _ in range(n):
        rows = np.concatenate([idx[w] for w in rng.choice(uniq, len(uniq))])
        sa, sb = a[rows].sum(), b[rows].sum()
        if sa > 0 and sb > 0:
            out.append((a[rows] * pay[rows]).sum() / sa - (b[rows] * pay[rows]).sum() / sb)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def tier_table(legs: pd.DataFrame) -> pd.DataFrame:
    """Return per unit by tier, per definition, train and test separately."""
    rows = []
    for name, col in DEFINITIONS.items():
        b = singles(legs, col)
        train = b[b["season"] < TEST_FROM]
        tiers = assign_tier(b["score"], _cutoffs(train["score"]))
        for part, mask in (("train", b["season"] < TEST_FROM), ("test", b["season"] >= TEST_FROM)):
            for t in TIERS:
                g = b[mask & (tiers == t)]
                if g.empty:
                    continue
                pay = np.where(g["won"], g["odds"] - 1.0, -1.0)
                rows.append({"confidence": name, "part": part, "tier": t,
                             "legs": len(g), "won": g["won"].mean(),
                             "price_implied": (1 / g["odds"]).mean(),
                             "roi": pay.mean(),
                             "roi_se": pay.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan})
    return pd.DataFrame(rows)


def scheme_table(legs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in DEFINITIONS.items():
        for bet_type, builder in (("single", singles), ("parlay3", weekly_parlays)):
            b = builder(legs, col).reset_index(drop=True)
            train = b[b["season"] < TEST_FROM]
            test_mask = b["season"] >= TEST_FROM
            flat = stake(b, "flat", train)
            for scheme in SCHEMES:
                s = stake(b, scheme, train)
                tr = roi(b[~test_mask], s[~test_mask])
                te = roi(b[test_mask], s[test_mask])
                lo, hi = boot_diff(b[test_mask].reset_index(drop=True),
                                   s[test_mask].reset_index(drop=True),
                                   flat[test_mask].reset_index(drop=True),
                                   n=500) if scheme != "flat" else (0.0, 0.0)
                rows.append({"confidence": name, "bet": bet_type, "scheme": scheme,
                             "train_roi": tr["roi"], "train_se": tr["roi_se"],
                             "test_bets": te["bets"],
                             "test_staked": te["staked"], "test_roi": te["roi"],
                             "test_se": te["roi_se"],
                             "vs_flat_low": lo, "vs_flat_high": hi})
    t = pd.DataFrame(rows)
    flat_test = (t[t["scheme"] == "flat"]
                 .set_index(["confidence", "bet"])["test_roi"].rename("flat_test_roi"))
    return t.join(flat_test, on=["confidence", "bet"])


def run(legs: pd.DataFrame) -> dict:
    pd.set_option("display.width", 220)
    seasons = sorted(int(s) for s in legs["season"].dropna().unique())
    print(f"Legs: {len(legs):,}   train seasons {[s for s in seasons if s < TEST_FROM]}   "
          f"test seasons {[s for s in seasons if s >= TEST_FROM]}\n")

    tt = tier_table(legs)
    print("=" * 76)
    print("1 · Return per unit by confidence tier  (sizing works only if high > low)")
    print("=" * 76)
    for name in DEFINITIONS:
        sub = tt[tt["confidence"] == name]
        print(f"\n{name}")
        print(sub.drop(columns="confidence").round(3).to_string(index=False))

    st = scheme_table(legs)
    print("\n" + "=" * 76)
    print("2 · Staking plans: return per unit staked")
    print("=" * 76)
    print(st.drop(columns=["test_staked"]).round(3).to_string(index=False))

    # Chosen by train ROI minus one standard error, so a small lucky sample
    # (a parlay plan with ~100 bets) cannot beat a steady large one.
    cand = st[st["scheme"] != "flat"].dropna(subset=["train_roi", "train_se"]).copy()
    cand["train_score"] = cand["train_roi"] - cand["train_se"]
    best = cand.sort_values("train_score", ascending=False).iloc[0]
    print("\n" + "=" * 76)
    print("3 · Best combination on train, run on test")
    print("=" * 76)
    print(f"  {best['confidence']} · {best['bet']} · {best['scheme']}")
    print(f"  train ROI {best['train_roi']:+.1%}  (SE {best['train_se']:.1%})")
    print(f"  test  ROI {best['test_roi']:+.1%} (SE {best['test_se']:.1%}) on {int(best['test_bets'])} bets   "
          f"(flat staking on the same bets: {best['flat_test_roi']:+.1%})")
    print(f"  test ROI minus flat, 95% range: {best['vs_flat_low']:+.1%} to {best['vs_flat_high']:+.1%}")
    if best["vs_flat_low"] > 0:
        v = "sizing beat flat staking on unseen seasons"
    elif best["vs_flat_high"] < 0:
        v = "sizing did worse than flat staking on unseen seasons"
    else:
        v = "no reliable difference from flat staking on unseen seasons"
    print(f"  -> {v}")
    tr, se = best["test_roi"], best["test_se"]
    if tr > 2 * se:
        p = "the plan itself made money on unseen seasons, beyond two standard errors"
    elif tr < -2 * se:
        p = "the plan itself lost money on unseen seasons, beyond two standard errors"
    else:
        p = f"the plan's own return ({tr:+.1%}, SE {se:.1%}) is inside noise on unseen seasons"
    print(f"  -> {p}")
    return {"tiers": tt, "schemes": st, "best": best}


def _main() -> None:
    from .ingest import load_matches
    from .legfilter import shortlisted_legs
    from .multileague import load_backtest

    matches = load_matches()
    legs = shortlisted_legs(load_backtest(), matches, min_rate=0.65)
    run(add_scores(legs, matches))


if __name__ == "__main__":
    _main()
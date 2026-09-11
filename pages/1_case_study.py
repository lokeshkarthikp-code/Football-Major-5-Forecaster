"""
Case study page: did the shortlist strategy have an edge?

Reads data/case_study.json, written by build_case_study.py. This page
computes nothing -- the numbers are fixed at build time so they cannot
drift between visits, and the page loads instantly.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Case study", page_icon="📊", layout="wide")

CS = Path(__file__).resolve().parent.parent / "data" / "case_study.json"

if not CS.exists():
    st.error("No case study data. Run `python build_case_study.py` first.")
    st.stop()

d = json.loads(CS.read_text())
df = lambda k: pd.DataFrame(d[k])          # noqa: E731

sc = d["scope"]

st.title("Does a shortlist of elite clubs beat the market?")
st.caption(
    f"{sc['matches']:,} matches · {sc['leagues']} leagues · "
    f"{sc['seasons']} seasons · {sc['backtest_predictions']:,} "
    f"walk-forward predictions · built {d['built_at'][:10]}"
)

st.markdown(
    """
**The question.** Back a rolling shortlist of Europe's strongest clubs each
week. Does it produce legs with positive expected value?

**The answer, established five ways: no.** Not by team selection, not by
fixture filtering, not by model edge, not by price, and not by any staking
scheme. The section below is the evidence in the order it was gathered.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# 1. Where we started
# ---------------------------------------------------------------------------
st.header("1 · Where we started")
st.markdown(
    "A Dixon-Coles goals model, fitted separately per league, evaluated "
    "walk-forward against the bookmaker's closing line. Lower Brier is "
    "better."
)

b = df("brier_by_league")
long = b.melt(id_vars="league", var_name="source", value_name="brier")
c1, c2 = st.columns([3, 2])
with c1:
    st.altair_chart(
        alt.Chart(long).mark_bar().encode(
            y=alt.Y("league:N", title=None, sort="-x"),
            x=alt.X("brier:Q", title="Brier score",
                    scale=alt.Scale(domain=[0.55, 0.67])),
            color=alt.Color("source:N", title=None),
            yOffset="source:N",
            tooltip=["league", "source", "brier"],
        ).properties(height=280),
        use_container_width=True,
    )
with c2:
    st.markdown(
        "The model beat base rates comfortably and lost to the closing "
        "line in **all five leagues**. It captured 78–87% of the "
        "bookmaker's improvement over base rates using only goals and "
        "dates — respectable, and not an edge."
    )
    st.dataframe(df("edge_share").round(3), hide_index=True,
                 use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 2. The central finding
# ---------------------------------------------------------------------------
st.header("2 · The market's prices were right")

pb = df("price_buckets")
st.markdown(
    "Every three-leg parlay from the elite tier, bucketed by its combined "
    "price. The two lines are how often the trios **actually won** and how "
    "often they **needed to win** to break even."
)

melted = pb.melt(id_vars=["odds_band", "weeks"],
                 value_vars=["hit_rate", "breakeven_hit_rate"],
                 var_name="series", value_name="probability")
melted["series"] = melted["series"].map({
    "hit_rate": "Actually won",
    "breakeven_hit_rate": "Needed to break even"})

st.altair_chart(
    alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("odds_band:N", title="Combined odds band", sort=None),
        y=alt.Y("probability:Q", title="Probability"),
        color=alt.Color("series:N", title=None),
        tooltip=["odds_band", "series", "probability", "weeks"],
    ).properties(height=320),
    use_container_width=True,
)

st.markdown(
    "The lines track each other across every price range. When the market "
    "priced a trio at 3.24 it won 29.5% of the time against an implied "
    "30.9%. When it priced one at 12.20 it won 5.9% against an implied "
    "9.2%. **The bookmaker's estimate was accurate everywhere.** The "
    "persistent gap underneath is the margin, and no band clears it."
)

p = df("trio_pooled").iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trio-weeks tested", f"{int(p['total_weeks']):,}")
c2.metric("All three won", f"{p['pooled_hit_rate']:.1%}")
c3.metric("Pooled ROI", f"{p['pooled_roi']:+.1%}")
c4.metric("Profitable combinations", f"{int(p['combos_profitable'])} / "
                                      f"{int(p['combinations'])}")

with st.expander("Best and worst individual combinations — and why to ignore them"):
    st.markdown(
        "Sorted tables invite you to read the top row as a strategy. It "
        "isn't. Every large winner is a single season of ~30 weeks; the "
        "combinations with the most weeks sit near zero. That is "
        "regression, not skill."
    )
    st.dataframe(df("trio_best"), hide_index=True, use_container_width=True)
    st.dataframe(df("trio_worst"), hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 3. The sharpest result
# ---------------------------------------------------------------------------
st.header("3 · The model's disagreement was error, not insight")

sw = df("sweep_ev")
st.markdown(
    "Filtering legs to those where the model thought the market was most "
    "wrong. If the model had information the market lacked, returns should "
    "**rise** as the filter tightens."
)

st.altair_chart(
    alt.Chart(sw).mark_line(point=True).encode(
        x=alt.X("threshold:Q", title="Minimum expected value demanded"),
        y=alt.Y("roi_singles:Q", title="Return on investment",
                axis=alt.Axis(format="%")),
        tooltip=["threshold", "roi_singles", "legs_kept", "hit_rate"],
    ).properties(height=300)
    + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        strokeDash=[4, 4]).encode(y="y:Q"),
    use_container_width=True,
)

st.markdown(
    "Returns fall monotonically, from −4% unfiltered to −25% at the "
    "largest disagreements. **The legs where the model most disagreed with "
    "the market were the legs where the model was most wrong.** A model "
    "with no information would show a flat line. This slopes down, which "
    "says the market knows things the model cannot see — team news, "
    "injuries, rotation — and disagreement measures the size of that blind "
    "spot."
)

with st.expander("The other filter: model probability"):
    sp = df("sweep_prob")
    st.markdown(
        "Filtering to fixtures the model thinks are easy. Hit rate climbs "
        "from 67% to 84%; ROI never turns positive at any threshold. "
        "Selecting easy fixtures selects short prices, and the market had "
        "already priced the ease in."
    )
    st.dataframe(sp, hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 4. Correct score
# ---------------------------------------------------------------------------
if "correct_score" in d:
    cs = d["correct_score"]
    st.header("4 · On correct score, the grid barely beat a constant guess")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model's exact-score hit rate", f"{cs['model_hit_rate']:.2%}")
    c2.metric("Always guessing 1-1", f"{cs['always_1_1']:.2%}")
    c3.metric("Breakeven odds required", f"{cs['breakeven_odds']:.2f}")

    st.markdown(
        f"A gap of "
        f"{(cs['model_hit_rate'] - cs['always_1_1']) * 100:.2f} percentage "
        f"points over {cs['n']:,} predictions — within one standard error "
        f"of zero. **{cs['share_predicting_1_1']:.0%} of the model's picks "
        f"were 1-1**, the single most efficiently priced cell in the "
        f"market. Common scorelines typically pay 6.00–9.00; this needs "
        f"{cs['breakeven_odds']:.2f} just to break even."
    )

    bl = pd.DataFrame(cs["by_league"])
    st.altair_chart(
        alt.Chart(bl).mark_bar().encode(
            x=alt.X("hit_rate:Q", title="Exact-score hit rate",
                    axis=alt.Axis(format="%")),
            y=alt.Y("league:N", title=None, sort="-x"),
            tooltip=["league", "n", "hit_rate"],
        ).properties(height=200)
        + alt.Chart(pd.DataFrame({"x": [cs["always_1_1"]]})).mark_rule(
            color="red", strokeDash=[4, 4]).encode(x="x:Q"),
        use_container_width=True,
    )
    st.caption(
        "Dashed line: always guessing 1-1. Serie A and La Liga clear it; "
        "England, Germany and France do not."
    )

    st.divider()

# ---------------------------------------------------------------------------
# 5. Why staking cannot help
# ---------------------------------------------------------------------------
st.header("5 · Why no staking plan rescues it")
st.markdown(
    """
Total return is the sum of each stake multiplied by its expected value
per dollar. If every one of those expected values is negative, no set of
positive stakes makes the sum positive. Staking changes variance and how
fast you get there. It never changes the sign.

Simulated over 3,584 weeks from a $10,000 bankroll, on a series *milder*
than the one measured here:
    """
)
st.dataframe(
    pd.DataFrame([
        {"Staking plan": "Flat $100", "Final bankroll": "$0 (busted)"},
        {"Staking plan": "2% of bankroll", "Final bankroll": "$53"},
        {"Staking plan": "10% of bankroll", "Final bankroll": "$1 (busted)"},
        {"Staking plan": "Kelly, given the true win rate",
         "Final bankroll": "$1 (busted)"},
        {"Staking plan": "More on short prices", "Final bankroll": "$0 (busted)"},
        {"Staking plan": "More on long prices", "Final bankroll": "$0 (busted)"},
        {"Staking plan": "Martingale", "Final bankroll": "$0 (busted)"},
    ]), hide_index=True, use_container_width=True)

st.markdown(
    "Kelly is the row that matters. It is the mathematically optimal "
    "staking rule, it was given the true win probability, and it still "
    "went to zero — because with negative edge Kelly's correct answer is "
    "to stake nothing at all."
)

st.divider()

# ---------------------------------------------------------------------------
# What this cost, and what it means
# ---------------------------------------------------------------------------
st.header("Where that leaves it")
st.markdown(
    """
The bookmaker's margin on a single elite favourite is around 4%. Stack
three on one slip and it compounds: 0.96³ ≈ 0.88. The measured pooled ROI
of −10% is almost exactly that arithmetic. A parlay is not a way to win
more; it is a way to pay the margin three times on one bet.

Beating this would require information the market does not have. This
model reads goals and dates. The market reads injury reports, lineup
leaks, and the weight of money from people who know things first. On the
most heavily traded market in world sport, public data was never going to
be enough — and section 3 is the measurement of exactly how much was
missing.

**What went right.** The shortlist was rebuilt every season from earlier
seasons only, so clubs that later collapsed stayed in. Napoli qualified in
2023 off their title and appears in four of the five worst combinations.
A list built with hindsight would have quietly dropped them, and every
number on this page would look better and be false.
    """
)

with st.expander("A metric that had to be thrown out"):
    st.markdown(
        """
The first version of this analysis compared hit rate against
`1 / mean(odds)` and reported a **positive** edge of +3.5%. That formula
is wrong: by Jensen's inequality `1/mean(odds)` is not `mean(1/odds)`,
and with combined odds spanning 4 to 15 the gap is large enough to flip
the sign.

Tested against a synthetic book built to lose 12% by construction, the
broken metric reported +1.8% edge while the true return was −14.4%.

Every figure on this page settles each week at its own price and sums the
money instead. The corrected result is the one shown: the strategy loses.
        """
    )
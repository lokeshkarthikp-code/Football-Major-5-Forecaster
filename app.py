"""
Premier League match forecaster -- Streamlit front end.

Run:  streamlit run app.py

Reads only the cached parquet files. No network calls at request time, so
the app stays up even if a data source is down.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from plfc.backtest import OUTCOMES, calibration_table, evaluate, walk_forward
from plfc.ingest import last_refresh, load_fixtures, load_matches
from plfc.ledger import latest_before_kickoff, load_ledger, track_record
from plfc.model import DixonColes

st.set_page_config(page_title="PL Forecaster", page_icon="⚽", layout="wide")


@st.cache_data(ttl=3600)
def _matches() -> pd.DataFrame:
    return load_matches()


@st.cache_data(ttl=3600)
def _fixtures() -> pd.DataFrame:
    return load_fixtures()


@st.cache_data(ttl=600)
def _ledger() -> pd.DataFrame:
    return load_ledger()


@st.cache_resource
def _model(xi: float, shrinkage: float, data_version: str) -> DixonColes:
    """Fit the model.

    `data_version` is never read in the body -- it exists purely to key the
    cache. Without it, @st.cache_resource holds the fitted model for the life
    of the process, so a data refresh would update the ratings table while
    predictions quietly came from a stale fit. Silent staleness is exactly
    the failure mode this project is built to avoid.
    """
    return DixonColes(xi=xi, shrinkage=shrinkage).fit(_matches())


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


# ---------------------------------------------------------------------------

st.title("Premier League Match Forecaster")

stamp = last_refresh()
if stamp:
    age = datetime.now(timezone.utc) - datetime.fromisoformat(stamp)
    st.caption(
        f"Data refreshed {stamp[:16].replace('T', ' ')} UTC "
        f"({age.days}d {age.seconds // 3600}h ago). Auto-refresh runs Wednesdays."
    )

try:
    matches = _matches()
except FileNotFoundError:
    st.error("No cached data. Run `python -m plfc.ingest` first.")
    st.stop()

with st.sidebar:
    st.header("Model settings")
    xi = st.select_slider(
        "Time-decay rate (ξ)",
        options=[0.0005, 0.001, 0.0018, 0.003, 0.005],
        value=0.0018,
        help="How fast old matches lose influence. Tuned by backtest.",
    )
    shrinkage = st.slider(
        "Shrinkage toward prior", 0.0, 30.0, 8.0, 1.0,
        help="Higher = promoted and sparsely-observed teams pulled harder "
             "toward the league prior.",
    )
    st.caption(f"Half-life ≈ {int(0.693 / xi)} days")

model = _model(xi, shrinkage, stamp or "unknown")

tab_fix, tab_any, tab_rec, tab_rate, tab_val = st.tabs(
    ["This weekend", "Any fixture", "Track record", "Team ratings", "How good is it?"]
)


# --- upcoming fixtures -----------------------------------------------------
with tab_fix:
    fixtures = _fixtures()
    if fixtures.empty:
        st.info("No upcoming fixtures cached. Run a refresh.")
    else:
        fixtures["date"] = pd.to_datetime(fixtures["date"])
        now = pd.Timestamp.now().normalize()
        horizon = st.radio(
            "Window", ["Next 3 fixtures", "This weekend", "Next 14 days"],
            horizontal=True,
        )

        upcoming = fixtures[fixtures["date"] >= now].sort_values("date")
        if horizon == "Next 3 fixtures":
            window = upcoming.head(3)
        elif horizon == "This weekend":
            days_to_sat = (5 - now.weekday()) % 7
            sat = now + timedelta(days=days_to_sat)
            window = upcoming[
                (upcoming["date"] >= sat - timedelta(days=1))
                & (upcoming["date"] <= sat + timedelta(days=2))
            ]
        else:
            window = upcoming[upcoming["date"] <= now + timedelta(days=14)]

        if window.empty:
            st.info("Nothing scheduled in that window.")
        else:
            preds = model.predict_frame(window)
            for _, r in preds.iterrows():
                new_flag = ""
                if r["is_new_home"] or r["is_new_away"]:
                    new_flag = "  ⚠️ includes a team with no PL history — prior-based estimate"

                with st.container(border=True):
                    st.markdown(
                        f"**{r['home_team']} vs {r['away_team']}** — "
                        f"{pd.Timestamp(r['date']).strftime('%a %d %b')}{new_flag}"
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Home win", _pct(r["home_win"]))
                    c2.metric("Draw", _pct(r["draw"]))
                    c3.metric("Away win", _pct(r["away_win"]))
                    c4.metric("Likely score", r["likely_score"])
                    st.caption(
                        f"xG {r['expected_home_goals']:.2f} – "
                        f"{r['expected_away_goals']:.2f}  ·  "
                        f"Over 2.5: {_pct(r['over_2_5'])}  ·  "
                        f"BTTS: {_pct(r['btts'])}"
                    )


# --- arbitrary matchup -----------------------------------------------------
with tab_any:
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    away = c2.selectbox("Away", teams, index=1)

    if home == away:
        st.warning("Pick two different teams.")
    else:
        p = model.predict(home, away)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home} win", _pct(p["home_win"]))
        c2.metric("Draw", _pct(p["draw"]))
        c3.metric(f"{away} win", _pct(p["away_win"]))

        m = model.score_matrix(home, away)[:6, :6]
        grid = pd.DataFrame(
            m, index=[f"{i}" for i in range(6)], columns=[f"{i}" for i in range(6)]
        )
        st.markdown("**Scoreline probabilities** (home goals × away goals)")
        st.dataframe(
            grid.style.format("{:.1%}"),
            use_container_width=True,
        )


# --- live track record -----------------------------------------------------
with tab_rec:
    st.markdown(
        """
Every forecast this model has made, sealed with a timestamp before kickoff
and reconciled against the result afterwards.

This is different from the backtest in the next tab, and stronger evidence.
A backtest is **retrospective** — it reconstructs what the model would have
said, and can be re-run with different settings until the numbers flatter you.
This log is **prospective**: the prediction was written down before the match
was played and is never edited.
        """
    )

    led = _ledger()
    if led.empty:
        st.info(
            "No predictions logged yet. Run `python -m plfc.ledger` to record "
            "forecasts for upcoming fixtures. The log starts building from the "
            "first run — there is no way to backfill it, which is the point."
        )
    else:
        rec = track_record(led)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Settled", rec.get("n_settled", 0))
        c2.metric("Pending", rec.get("n_pending", 0))
        if rec.get("n_settled"):
            c3.metric("Brier", f"{rec['brier']:.4f}")
            c4.metric("Accuracy", f"{rec['accuracy']:.1%}")

        if rec.get("n_settled", 0) < 30:
            st.warning(
                f"Only {rec.get('n_settled', 0)} settled predictions so far. "
                "Football is high-variance — these numbers are dominated by "
                "noise until you have a few hundred. Report the backtest for "
                "now and let this accumulate."
            )

        view = latest_before_kickoff(led)
        if not view.empty:
            settled = view[view["actual"].notna() & (view["actual"] != "")]
            pending = view[view["actual"].isna() | (view["actual"] == "")]

            if not pending.empty:
                st.markdown("**Awaiting result**")
                st.dataframe(
                    pending[[
                        "match_date", "home_team", "away_team",
                        "home_win", "draw", "away_win", "likely_score",
                    ]].sort_values("match_date").round(3),
                    use_container_width=True, hide_index=True,
                )

            if not settled.empty:
                st.markdown("**Settled — prediction vs result**")
                show = settled.copy()
                show["predicted"] = show[OUTCOMES].idxmax(axis=1)
                show["hit"] = show["predicted"] == show["actual"]
                show["score"] = (
                    show["actual_home_goals"].astype("Int64").astype(str)
                    + "–"
                    + show["actual_away_goals"].astype("Int64").astype(str)
                )
                st.dataframe(
                    show[[
                        "match_date", "home_team", "away_team",
                        "home_win", "draw", "away_win",
                        "score", "actual", "hit",
                    ]].sort_values("match_date", ascending=False).round(3),
                    use_container_width=True, hide_index=True,
                )


# --- ratings ---------------------------------------------------------------
with tab_rate:
    st.markdown(
        "Attack and defence strengths on a log scale. Higher is better for both. "
        "`weighted_matches` reflects time decay — a team's recent matches count "
        "far more than old ones, so a promoted side shows a low value and is "
        "shrunk toward the prior."
    )
    st.dataframe(model.ratings().round(3), use_container_width=True, height=520)
    st.caption(
        f"Home advantage: {model.home_adv:.3f} (log scale) · "
        f"Low-score correction ρ: {model.rho:.3f} · "
        f"Half-life: {model.half_life_days:.0f} days"
    )


# --- honesty tab -----------------------------------------------------------
with tab_val:
    st.markdown(
        """
Evaluated by **walk-forward validation**: the model is refit on everything
strictly before each date and predicts forward. No shuffled cross-validation —
that would leak future matches into training and inflate every number here.

The benchmark that matters is the **bookmaker closing line**, not zero.
Football outcomes sit close to the noise ceiling; the market reaches roughly
53–55% accuracy on three-way results with far more information than this
model has. Getting close to it is the honest goal.
        """
    )
    if st.button("Run backtest (takes a minute)"):
        with st.spinner("Walking forward…"):
            res = walk_forward(matches, xi=xi, shrinkage=shrinkage, step_days=14)
            st.dataframe(evaluate(res).round(4), use_container_width=True)

            cal = calibration_table(res, res["actual"])
            st.markdown("**Calibration** — predicted vs observed frequency")
            st.line_chart(
                cal.set_index("mean_predicted")[["observed_rate"]],
                use_container_width=True,
            )
            st.caption(
                "A perfectly calibrated model traces the diagonal: when it says "
                "30%, the thing happens 30% of the time."
            )

    with st.expander("Known limitations"):
        st.markdown(
            """
- **No team-news.** Injuries, suspensions and rotation are invisible to the
  model. A missing key striker moves the true probability materially.
- **No European/cup fixture congestion.** Midweek matches affect weekend form.
- **Promoted teams are priors, not estimates**, until they accumulate matches.
- **Managerial changes** are not modelled; strength is assumed to drift
  smoothly, which a new appointment violates.
- **Odds normalisation is proportional**, which slightly distorts longshots
  (favourite-longshot bias).
- This is a forecasting exercise, not betting advice.
            """
        )
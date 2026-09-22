"""
European football match forecaster -- Streamlit front end.

Run:  streamlit run app.py

Reads only the cached parquet files. No network calls at request time, so
the app stays up even if a data source is down.

ONE MODEL PER LEAGUE. The league selector in the sidebar switches which
fitted model everything below is reading from. Ratings and strengths are
comparable within a league and never across leagues, so nothing here ever
puts two leagues in the same ranked table.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from plfc.backtest import OUTCOMES, calibration_table, evaluate, walk_forward
from plfc.ingest import last_refresh, load_fixtures, load_matches
from plfc.leagues import display_name
from plfc.ledger import latest_before_kickoff, load_ledger, track_record
from plfc.model import DixonColes
from plfc.multileague import available_leagues
from plfc.ratingflow import load_history

st.set_page_config(page_title="Football Forecaster", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

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
def _model(league: str, xi: float, shrinkage: float,
           data_version: str) -> DixonColes:
    """Fit one league's model.

    `data_version` is never read in the body -- it exists purely to key the
    cache. Without it, @st.cache_resource holds the fitted model for the life
    of the process, so a data refresh would update the ratings table while
    predictions quietly came from a stale fit. Silent staleness is exactly
    the failure mode this project is built to avoid.

    `league` is part of the key for the same reason: without it, switching
    leagues in the sidebar would show you a different league's model.
    """
    df = _matches()
    sub = df[df["league"] == league] if "league" in df.columns else df
    return DixonColes(xi=xi, shrinkage=shrinkage).fit(sub)


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _result_from_score(score: str) -> str | None:
    """Read win/draw/loss off a 'H-A' scoreline string.

    The model has two opinions about every match: the argmax of its
    win/draw/away vector, and the single most likely scoreline. They
    disagree often -- a draw probability rarely leads the vector, while
    1-1 is frequently the most likely cell -- so both are worth tracking.
    """
    try:
        h, a = str(score).split("-")
        h, a = int(h), int(a)
    except (ValueError, AttributeError):
        return None
    return "home_win" if h > a else "away_win" if h < a else "draw"


def _score_str(hg, ag) -> str:
    if pd.isna(hg) or pd.isna(ag):
        return ""
    return f"{int(hg)}-{int(ag)}"


def _filter_league(df: pd.DataFrame, league: str) -> pd.DataFrame:
    if df.empty or "league" not in df.columns:
        return df
    return df[df["league"] == league]


# ---------------------------------------------------------------------------
# Header and league selection
# ---------------------------------------------------------------------------

try:
    matches = _matches()
except FileNotFoundError:
    st.error("No cached data. Run `python -m plfc.ingest` first.")
    st.stop()

leagues = available_leagues(matches)
if not leagues:
    st.error("No `league` column in the cached data. Re-run the ingest.")
    st.stop()

stamp = last_refresh()

with st.sidebar:
    st.header("League")
    league = st.radio(
        "League", leagues, format_func=display_name, label_visibility="collapsed"
    )

    st.divider()
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
    st.caption("Each league is fitted separately. Strengths are not "
               "comparable between leagues.")

st.title(f"{display_name(league)} Match Forecaster")

if stamp:
    age = datetime.now(timezone.utc) - datetime.fromisoformat(stamp)
    st.caption(
        f"Data refreshed {stamp[:16].replace('T', ' ')} UTC "
        f"({age.days}d {age.seconds // 3600}h ago). Auto-refresh runs Wednesdays."
    )

league_matches = _filter_league(matches, league)
model = _model(league, xi, shrinkage, stamp or "unknown")

tab_fix, tab_any, tab_week, tab_rec, tab_rate, tab_move, tab_val = st.tabs(
    ["This weekend", "Any fixture", "Weekly scorecard", "Track record",
     "Team ratings", "Rating movement", "How good is it?"]
)


# --- upcoming fixtures -----------------------------------------------------
with tab_fix:
    fixtures = _filter_league(_fixtures(), league)
    if fixtures.empty:
        st.info("No upcoming fixtures cached for this league. Run a refresh.")
    else:
        fixtures = fixtures.copy()
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
                    new_flag = ("  ⚠️ includes a team with no history in this "
                                "league — prior-based estimate")

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
    teams = sorted(set(league_matches["home_team"]) | set(league_matches["away_team"]))
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home", teams, index=0)
    away = c2.selectbox("Away", teams, index=min(1, len(teams) - 1))

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
        st.dataframe(grid.style.format("{:.1%}"), use_container_width=True)


# --- weekly scorecard ------------------------------------------------------
with tab_week:
    st.markdown(
        "Forecasts sealed before kickoff, scored three ways. Backtest "
        "reference points over 12,413 predictions: result 52.3%, "
        "scoreline-implied result 41.7%, exact score 12.5%."
    )

    led = _filter_league(_ledger(), league)
    view = latest_before_kickoff(led) if not led.empty else pd.DataFrame()

    if view.empty:
        st.info(
            "No sealed predictions for this league yet. The log builds from "
            "the first run and cannot be backfilled."
        )
    else:
        settled = view[view["actual"].notna() & (view["actual"] != "")].copy()

        if settled.empty:
            st.info(
                f"{len(view)} prediction(s) logged, none settled yet. Results "
                "fill in on the next Wednesday run after the matches are played."
            )
        else:
            settled["match_date"] = pd.to_datetime(settled["match_date"])
            settled["predicted"] = settled[OUTCOMES].idxmax(axis=1)
            settled["result_hit"] = settled["predicted"] == settled["actual"]
            settled["actual_score"] = [
                _score_str(h, a) for h, a in
                zip(settled["actual_home_goals"], settled["actual_away_goals"])
            ]
            settled["score_hit"] = (
                settled["likely_score"].astype(str) == settled["actual_score"]
            )
            settled["score_result"] = settled["likely_score"].map(_result_from_score)
            settled["score_result_hit"] = (
                settled["score_result"] == settled["actual"]
            )
            settled["week_start"] = (
                settled["match_date"] - pd.to_timedelta(
                    settled["match_date"].dt.weekday, unit="D")
            ).dt.normalize()

            n = len(settled)
            n_res = int(settled["result_hit"].sum())
            n_scr = int(settled["score_hit"].sum())
            n_sr = int(settled["score_result_hit"].sum())

            st.markdown("#### Running totals")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predictions settled", n)
            c2.metric("Result correct", f"{n_res} / {n}",
                      delta=f"{n_res / n:.0%}", delta_color="off")
            c3.metric("Scoreline's result correct", f"{n_sr} / {n}",
                      delta=f"{n_sr / n:.0%}", delta_color="off")
            c4.metric("Exact score correct", f"{n_scr} / {n}",
                      delta=f"{n_scr / n:.0%}", delta_color="off")

            st.caption(
                "Result = highest-probability outcome. Scoreline's result = "
                "win/draw/loss read off the most likely scoreline. "
                "Exact score = that scoreline was right."
            )

            if n < 50:
                st.warning(
                    f"{n} settled predictions. Too few to read as evidence — "
                    "at 52% accuracy an 8-from-10 week occurs by chance "
                    "about once a month."
                )

            weekly = (
                settled.groupby("week_start")
                .agg(predictions=("result_hit", "size"),
                     result_correct=("result_hit", "sum"),
                     sr_correct=("score_result_hit", "sum"),
                     score_correct=("score_hit", "sum"))
                .reset_index()
                .sort_values("week_start", ascending=False)
            )
            weekly["result_%"] = (
                weekly["result_correct"] / weekly["predictions"]
            ).map("{:.0%}".format)
            weekly["sr_%"] = (
                weekly["sr_correct"] / weekly["predictions"]
            ).map("{:.0%}".format)
            weekly["score_%"] = (
                weekly["score_correct"] / weekly["predictions"]
            ).map("{:.0%}".format)
            weekly["week"] = weekly["week_start"].dt.strftime("w/c %d %b %Y")

            st.markdown("#### Week by week")
            st.dataframe(
                weekly[["week", "predictions", "result_correct", "result_%",
                        "sr_correct", "sr_%", "score_correct", "score_%"]],
                use_container_width=True, hide_index=True,
            )

            cols = ["match_date", "home_team", "away_team", "home_win", "draw",
                    "away_win", "predicted", "likely_score", "score_result",
                    "actual_score", "actual", "result_hit", "score_result_hit",
                    "score_hit"]

            with st.expander(f"Every settled prediction ({n} matches)"):
                st.dataframe(
                    settled[cols].sort_values("match_date", ascending=False)
                    .round(3),
                    use_container_width=True, hide_index=True, height=460,
                )

            with st.expander("Break it down week by week"):
                for wk in weekly["week_start"]:
                    grp = settled[settled["week_start"] == wk]
                    hits = int(grp["result_hit"].sum())
                    srh = int(grp["score_result_hit"].sum())
                    scr = int(grp["score_hit"].sum())
                    st.markdown(
                        f"**{wk.strftime('w/c %d %b %Y')}** — "
                        f"{hits}/{len(grp)} results, "
                        f"{srh}/{len(grp)} scoreline results, "
                        f"{scr}/{len(grp)} exact scores"
                    )
                    st.dataframe(
                        grp[cols].sort_values("match_date").round(3),
                        use_container_width=True, hide_index=True,
                    )

            with st.expander("Only the ones it got wrong"):
                miss = settled[~settled["result_hit"]]
                if miss.empty:
                    st.success("No incorrect results yet.")
                else:
                    miss = miss.copy()
                    miss["p_assigned"] = miss.apply(
                        lambda r: r[r["actual"]], axis=1)
                    st.caption(
                        "`p_assigned` = probability assigned to what actually "
                        "happened. Misses at 30% are expected; misses at 5% "
                        "are worth checking."
                    )
                    st.dataframe(
                        miss[cols + ["p_assigned"]]
                        .sort_values("p_assigned").round(3),
                        use_container_width=True, hide_index=True,
                    )

            with st.expander("Where the two routes disagreed"):
                dis = settled[settled["predicted"] != settled["score_result"]]
                if dis.empty:
                    st.info("The two routes agreed on every settled match.")
                else:
                    a = int(dis["result_hit"].sum())
                    b = int(dis["score_result_hit"].sum())
                    st.markdown(
                        f"{len(dis)} matches where the two routes diverged. "
                        f"Result route right {a}, scoreline route right {b}. "
                        "They diverge on 59.5% of backtested matches."
                    )
                    st.dataframe(
                        dis[cols].sort_values("match_date", ascending=False)
                        .round(3),
                        use_container_width=True, hide_index=True,
                    )

            with st.expander("Exact-score hits only"):
                got = settled[settled["score_hit"]]
                if got.empty:
                    st.info("No exact scorelines called correctly yet.")
                else:
                    st.dataframe(
                        got[cols].sort_values("match_date", ascending=False)
                        .round(3),
                        use_container_width=True, hide_index=True,
                    )


# --- live track record -----------------------------------------------------
with tab_rec:
    st.markdown(
        "Forecasts timestamped before kickoff, reconciled after. Append-only "
        "— a prediction is never edited once written. Unlike the backtest, "
        "this cannot be re-run with different settings."
    )

    led = _filter_league(_ledger(), league)
    if led.empty:
        st.info(
            "No predictions logged yet for this league. Run "
            "`python -m plfc.ledger` to record forecasts for upcoming fixtures."
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
                f"{rec.get('n_settled', 0)} settled. Dominated by noise below "
                "a few hundred; the backtest is the reportable figure for now."
            )

        view = latest_before_kickoff(led)
        if not view.empty:
            pending = view[view["actual"].isna() | (view["actual"] == "")]
            if not pending.empty:
                with st.expander(f"Awaiting result ({len(pending)})", expanded=True):
                    st.dataframe(
                        pending[[
                            "match_date", "home_team", "away_team",
                            "home_win", "draw", "away_win", "likely_score",
                        ]].sort_values("match_date").round(3),
                        use_container_width=True, hide_index=True,
                    )
            st.caption("Full settled history lives in the Weekly scorecard tab.")


# --- ratings ---------------------------------------------------------------
with tab_rate:
    st.markdown(
        "Attack and defence on a log scale, higher is better. "
        "`weighted_matches` reflects time decay, so promoted sides show a low "
        "value and are shrunk toward the prior."
    )
    st.dataframe(model.ratings().round(3), use_container_width=True, height=520)
    st.caption(
        f"Home advantage: {model.home_adv:.3f} (log scale) · "
        f"Low-score correction ρ: {model.rho:.3f} · "
        f"Half-life: {model.half_life_days:.0f} days · "
        f"{len(model.teams)} teams in {display_name(league)}"
    )
    st.caption(
        "Scale is estimated per league. Values are not comparable between "
        "leagues — clubs meet across them too rarely to place on a common scale."
    )


# --- rating movement -------------------------------------------------------
@st.cache_data(ttl=3600)
def _history(data_version: str) -> pd.DataFrame:
    return load_history()


with tab_move:
    hist = _history(stamp or "unknown")
    hist = hist[hist["league"] == league] if not hist.empty else hist

    if hist.empty:
        st.info("No rating history for this league yet. Run "
                "`python -m plfc.ratingflow`.")
    else:
        import altair as alt

        st.markdown(
            "How each result moved a team's rating, and what that did to the "
            "prediction for their next match. Ratings move on surprise: "
            "points won minus points the model expected."
        )

        latest = hist[hist["date"] == hist["date"].max()]
        movers = latest.reindex(
            latest["rating_change"].abs().sort_values(ascending=False).index).head(8)
        st.markdown(f"#### Biggest movers — {pd.Timestamp(hist['date'].max()):%d %b %Y}")
        st.dataframe(
            movers[["team", "opponent", "venue", "goals_for", "goals_against",
                    "p_win", "surprise", "rating_change", "next_opponent",
                    "next_p_win_before", "next_p_win_after"]].round(3),
            use_container_width=True, hide_index=True,
        )

        teams_ranked = (hist.sort_values("date").groupby("team")["rating_after"]
                        .last().sort_values(ascending=False).index.tolist())
        team = st.selectbox("Team", teams_ranked, key="move_team")
        t = hist[hist["team"] == team].sort_values("date")

        line = alt.Chart(t).mark_line(color="#999999").encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("rating_after:Q", title="Rating after match",
                    scale=alt.Scale(zero=False)))
        dots = alt.Chart(t).mark_point(size=90, filled=True, opacity=1).encode(
            x="date:T", y="rating_after:Q",
            color=alt.Color("result:N", title=None,
                            scale=alt.Scale(domain=["W", "D", "L"],
                                            range=["#2e7d32", "#9e9e9e", "#c62828"])),
            tooltip=["date:T", "opponent", "venue", "goals_for", "goals_against",
                     alt.Tooltip("p_win:Q", format=".0%"),
                     alt.Tooltip("surprise:Q", format="+.2f"),
                     alt.Tooltip("rating_change:Q", format="+.3f")])
        st.altair_chart(line + dots, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rating now", f"{t['rating_after'].iloc[-1]:.3f}",
                  delta=f"{t['rating_after'].iloc[-1] - t['rating_before'].iloc[0]:+.3f} this season")
        c2.metric("Points vs expected",
                  f"{t['points'].sum()} / {t['expected_points'].sum():.1f}")
        big = t.loc[t["surprise"].abs().idxmax()]
        c3.metric("Biggest surprise",
                  f"{big['result']} vs {big['opponent']}",
                  delta=f"{big['rating_change']:+.3f} rating", delta_color="off")

        show = t.assign(
            score=t["goals_for"].astype(str) + "-" + t["goals_against"].astype(str))
        st.dataframe(
            show[["date", "opponent", "venue", "rating_gap", "p_win", "score",
                  "result", "surprise", "attack_change", "defence_change",
                  "rating_change", "next_opponent", "next_p_win_before",
                  "next_p_win_after", "next_p_win_change"]]
            .sort_values("date", ascending=False).round(3),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "rating_gap = this team minus opponent going in. next_p_win "
            "before/after = the next match's win chance without and with this "
            "result. Other matches played the same day feed in too. Ratings are "
            "on this league's own scale."
        )


# --- honesty tab -----------------------------------------------------------
with tab_val:
    st.markdown(
        "Walk-forward: refit on everything strictly before each date, predict "
        "forward. Shuffled cross-validation would leak future matches into "
        "training. Benchmark is the closing line, not zero."
    )
    if st.button(f"Run backtest for {display_name(league)} (takes a minute)"):
        with st.spinner("Walking forward…"):
            res = walk_forward(matches, league=league, xi=xi,
                               shrinkage=shrinkage, step_days=14)
            st.dataframe(evaluate(res).round(4), use_container_width=True)

            cal = calibration_table(res, res["actual"])
            st.markdown("**Calibration** — predicted vs observed frequency")
            st.line_chart(
                cal.set_index("mean_predicted")[["observed_rate"]],
                use_container_width=True,
            )
            st.caption("Calibrated means tracing the diagonal.")

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
- **Refitting is not learning from errors.** Each refit re-estimates team
  strength from newer goals. It does not adjust the model's probabilities
  based on how its past probabilities turned out — that is a calibration
  layer, and it is not built yet.
- This is a forecasting exercise, not betting advice.
            """
        )
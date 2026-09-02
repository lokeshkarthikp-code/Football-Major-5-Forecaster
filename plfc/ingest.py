"""
Ingestion: pull, canonicalise, validate, cache.

Contract
--------
Everything downstream reads ONE tidy table: one row per match, canonical
team ids, no nulls in the key columns. If the raw data cannot satisfy that,
we raise here rather than let a bad row reach the model.

Sources
-------
Football-Data.co.uk (via soccerdata.MatchHistory)
    Results + closing betting odds, back to the 1990s. The odds are our
    benchmark: beating the closing line is the honest bar for "is this
    model any good", far more informative than raw accuracy.

football-data.org (via plfc.footballdata)
    Fixture schedule for the current season, including unplayed matches.
    This is what powers "show me this weekend's fixtures". A real REST API
    with a token -- replaced the FBref scraper, which was the one genuinely
    fragile source. Free tier has no odds and no match stats; we need
    neither here.

ClubElo (via soccerdata.ClubElo)
    Optional. Pre-computed strength ratings by date -- a strong baseline
    and a useful cold-start prior for promoted teams.

Caching
-------
Raw pulls are mirrored to parquet immediately. Once cached, the rest of the
project never touches the network. If a source changes in October, your
dataset is unaffected.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .names import resolve_series, write_soccerdata_config

log = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

LEAGUE = "ENG-Premier League"
MATCHES_PER_SEASON = 380          # 20 teams x 38 matchweeks. Non-negotiable.
TEAMS_PER_SEASON = 20

KEY_COLS = ["date", "season", "home_team", "away_team"]
RESULT_COLS = ["home_goals", "away_goals"]


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------

def current_season(today: datetime | None = None) -> int:
    """Return the season start year. PL seasons run Aug->May.

    A match in Jan 2026 belongs to season 2025 (the 2025/26 campaign).
    """
    today = today or datetime.now(timezone.utc)
    return today.year if today.month >= 7 else today.year - 1


def season_range(n_seasons: int = 10, today: datetime | None = None) -> list[int]:
    cur = current_season(today)
    return list(range(cur - n_seasons + 1, cur + 1))


# ---------------------------------------------------------------------------
# Validation -- loud failure by design
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when ingested data violates a structural invariant."""


def validate_matches(df: pd.DataFrame, *, allow_partial_season: bool = True) -> None:
    """Assert the structural invariants of a PL match table.

    These checks exist because the failure mode we fear is SILENT. A broken
    join does not raise -- it just drops rows. So we count them.
    """
    if df.empty:
        raise ValidationError("Match table is empty.")

    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing key columns: {missing}")

    for c in KEY_COLS:
        if df[c].isna().any():
            n = int(df[c].isna().sum())
            raise ValidationError(f"{n} null value(s) in key column {c!r}.")

    # Self-play is a classic symptom of a bad name merge (two aliases
    # collapsing onto the same canonical id incorrectly).
    self_play = df[df["home_team"] == df["away_team"]]
    if len(self_play):
        raise ValidationError(
            f"{len(self_play)} match(es) where a team plays itself -- "
            f"almost certainly a name-collapse bug. Sample:\n{self_play.head()}"
        )

    dupes = df.duplicated(subset=KEY_COLS).sum()
    if dupes:
        raise ValidationError(f"{dupes} duplicate match row(s) on {KEY_COLS}.")

    cur = current_season()
    problems: list[str] = []
    for season, grp in df.groupby("season"):
        n_teams = len(set(grp["home_team"]) | set(grp["away_team"]))
        if n_teams != TEAMS_PER_SEASON:
            problems.append(f"  season {season}: {n_teams} teams (expected {TEAMS_PER_SEASON})")

        played = grp.dropna(subset=RESULT_COLS) if all(
            c in grp.columns for c in RESULT_COLS
        ) else grp

        if season == cur and allow_partial_season:
            if len(grp) > MATCHES_PER_SEASON:
                problems.append(f"  season {season}: {len(grp)} matches (max {MATCHES_PER_SEASON})")
        elif len(played) != MATCHES_PER_SEASON:
            hint = "likely a dropped team from a bad name join"
            if n_teams > TEAMS_PER_SEASON:
                # More than 20 teams means two campaigns bled together --
                # usually a season assigned from match date rather than from
                # the source's season key. See _season_code_to_year.
                hint = ("more than 20 teams -- two seasons have merged; check "
                        "season assignment, not team names")
            problems.append(
                f"  season {season}: {len(played)} played matches "
                f"(expected {MATCHES_PER_SEASON}) -- {hint}"
            )

    if problems:
        raise ValidationError("Season integrity check failed:\n" + "\n".join(problems))

    log.info("Validation passed: %d matches across %d seasons.",
             len(df), df["season"].nunique())


# ---------------------------------------------------------------------------
# Pulls
# ---------------------------------------------------------------------------

def _seasons_as_strings(seasons: list[int]) -> list[str]:
    """soccerdata wants '1516' style season codes."""
    return [f"{s % 100:02d}{(s + 1) % 100:02d}" for s in seasons]


def _season_code_to_year(codes: pd.Series) -> pd.Series:
    """Convert soccerdata's '1920' season codes to the start year (2019).

    Authoritative, unlike inferring the season from a match date. Seasons do
    not always run Aug->May: the 2019/20 campaign was suspended in March 2020
    and finished on 26 July 2020, so any "month >= 7 means a new season" rule
    files those final matchweeks under the wrong campaign. The source already
    knows which season a match belongs to -- use that.
    """
    def one(c):
        s = str(c).strip()
        if len(s) == 4 and s.isdigit():
            start = int(s[:2])
            return 2000 + start if start < 90 else 1900 + start
        return pd.NA

    return codes.map(one).astype("Int64")


def fetch_results(seasons: list[int] | None = None) -> pd.DataFrame:
    """Historical results + closing odds from Football-Data.co.uk."""
    import soccerdata as sd

    write_soccerdata_config()          # normalise at the source
    seasons = seasons or season_range(10)

    mh = sd.MatchHistory(leagues=LEAGUE, seasons=_seasons_as_strings(seasons))
    raw = mh.read_games().reset_index()

    df = pd.DataFrame({
        "date": pd.to_datetime(raw["date"], errors="coerce"),
        "home_team": resolve_series(raw["home_team"]),
        "away_team": resolve_series(raw["away_team"]),
        "home_goals": pd.to_numeric(raw.get("FTHG"), errors="coerce"),
        "away_goals": pd.to_numeric(raw.get("FTAG"), errors="coerce"),
    })

    # Closing odds -- Bet365 columns are the most consistently populated.
    for src, dst in [("B365H", "odds_home"), ("B365D", "odds_draw"), ("B365A", "odds_away")]:
        if src in raw.columns:
            df[dst] = pd.to_numeric(raw[src], errors="coerce")

    # Season comes from the source's own key, never a date heuristic --
    # see _season_code_to_year for why (COVID broke the Aug->May assumption).
    if "season" in raw.columns:
        df["season"] = _season_code_to_year(raw["season"])
    else:
        log.warning("No season column in source; falling back to date heuristic.")
        df["season"] = df["date"].map(
            lambda d: current_season(d) if pd.notna(d) else pd.NA
        ).astype("Int64")
    df = df.dropna(subset=["date", "season"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_schedule(season: int | None = None,
                   competition: str = "PL") -> pd.DataFrame:
    """Full-season fixture list, INCLUDING unplayed matches.

    Sourced from football-data.org rather than scraped. Requires
    FOOTBALL_DATA_TOKEN -- see plfc/footballdata.py for setup.
    """
    from .footballdata import fetch_matches

    season = season or current_season()
    df = fetch_matches(competition=competition, season=season)

    keep = ["date", "home_team", "away_team", "home_goals", "away_goals",
            "matchweek", "season", "status"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # The API already stamps each match with its campaign, and fetch_matches
    # normalises that to a start year. Just make sure the dtype matches the
    # results feed so the two tables can be compared without surprises.
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    else:
        df["season"] = pd.Series([season] * len(df), dtype="Int64")

    return df.sort_values("date").reset_index(drop=True)


def fetch_elo(as_of: str | datetime | None = None) -> pd.DataFrame:
    """ClubElo ratings snapshot. Optional -- used as baseline and cold-start prior."""
    import soccerdata as sd

    write_soccerdata_config()
    as_of = as_of or datetime.now(timezone.utc).date().isoformat()
    if isinstance(as_of, datetime):
        as_of = as_of.date().isoformat()

    elo = sd.ClubElo()
    raw = elo.read_by_date(as_of).reset_index()

    raw["team"] = resolve_series(raw["team"], strict=False)
    out = raw.dropna(subset=["team"])[["team", "elo"]]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def refresh(n_seasons: int = 10, *, include_elo: bool = True) -> pd.DataFrame:
    """Full weekly refresh. Pull -> canonicalise -> validate -> cache.

    Safe to run repeatedly; soccerdata caches HTTP responses, and we only
    overwrite the parquet after validation passes. A failed run leaves the
    previous good dataset intact.
    """
    log.info("Refreshing Premier League data (%d seasons)...", n_seasons)

    results = fetch_results(season_range(n_seasons))
    validate_matches(results)

    try:
        schedule = fetch_schedule()
    except Exception as exc:                      # noqa: BLE001
        log.warning(
            "Schedule fetch failed (%s). Continuing without fixtures -- "
            "results and the model are unaffected. Check FOOTBALL_DATA_TOKEN.", exc)
        schedule = pd.DataFrame(columns=KEY_COLS)

    # Unplayed fixtures. The API's own status field is authoritative here --
    # more reliable than inferring "not yet played" from the results feed,
    # which lags by a day or two.
    if not schedule.empty and "status" in schedule.columns:
        fixtures = schedule[
            schedule["status"].isin(["SCHEDULED", "TIMED", "POSTPONED"])
        ].copy()
    elif not schedule.empty:
        played = set(zip(results["home_team"], results["away_team"], results["season"]))
        fixtures = schedule[[
            (h, a, s) not in played
            for h, a, s in zip(schedule["home_team"], schedule["away_team"],
                               schedule["season"])
        ]].copy()
    else:
        fixtures = pd.DataFrame(columns=KEY_COLS)

    stamp = datetime.now(timezone.utc).isoformat()
    results.attrs["refreshed_at"] = stamp

    results.to_parquet(DATA / "matches.parquet", index=False)
    fixtures.to_parquet(DATA / "fixtures.parquet", index=False)

    if include_elo:
        try:
            fetch_elo().to_parquet(DATA / "elo.parquet", index=False)
        except Exception as exc:                  # noqa: BLE001
            log.warning("Elo fetch failed (%s) -- non-fatal.", exc)

    (DATA / "LAST_REFRESH").write_text(stamp)
    log.info("Cached %d matches, %d upcoming fixtures.", len(results), len(fixtures))
    return results


def load_matches() -> pd.DataFrame:
    p = DATA / "matches.parquet"
    if not p.exists():
        raise FileNotFoundError("No cached data. Run: python -m plfc.ingest")
    return pd.read_parquet(p)


def load_fixtures() -> pd.DataFrame:
    p = DATA / "fixtures.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=KEY_COLS)


def last_refresh() -> str | None:
    p = DATA / "LAST_REFRESH"
    return p.read_text().strip() if p.exists() else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    refresh()
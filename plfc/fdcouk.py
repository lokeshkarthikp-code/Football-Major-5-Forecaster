"""
Football-Data.co.uk client -- direct CSV, no scraping.

WHY THIS REPLACED soccerdata
----------------------------
soccerdata is a good library, but it depends on seleniumbase, which pulls
in selenium and a full browser automation stack. That cost us three ways:

  1. seleniumbase registers a pytest plugin, so every test run imported a
     browser stack our tests never touch. Startup went from seconds to
     minutes.
  2. ~200MB of install weight on a Streamlit deploy that only reads parquet.
  3. A scraping dependency we did not actually need.

Because Football-Data.co.uk publishes plain CSV files at stable URLs, none
of that machinery was buying anything. This module downloads those files
with `requests` and parses them with pandas. No browser, no scraper, no
JavaScript.

URL FORMAT
----------
    https://football-data.co.uk/mmz4281/{season}/{div}.csv

where season is "2324" for 2023-24 and div is E0, SP1, I1, D1, F1.

NOTE THE MISSING "www." -- it matters. The www subdomain returns HTTP 503
while the bare domain serves normally. This cost a day of debugging: the
original URL was copied from another library's source rather than from an
actual link on the site, and every failure looked like an outage. Verify a
URL against the real thing before trusting it.

THE DATE TRAP
-------------
Football-Data.co.uk writes dates as dd/mm/yy in older files and dd/mm/yyyy
in newer ones. Both are day-first, which pandas does NOT assume by default
-- 03/04/2024 would silently parse as 4 March instead of 3 April. Every
date in this module is parsed with dayfirst=True, and the result is
range-checked against the season it claims to belong to.

CACHING AND RESILIENCE
----------------------
Raw CSVs are mirrored to data/raw/ on first download. Completed seasons
never change, so they are read from disk thereafter; only the current
season is re-fetched. That makes a refresh fast and keeps us from
hammering a free service.

The site is free infrastructure and does go down -- it returned 503 across
every league during development. Downloads therefore retry on 5xx with
exponential backoff, and fall back to a stale cached copy rather than
failing outright. Last week's data beats no data on an unattended run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://football-data.co.uk/mmz4281"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Bookmaker odds columns, in preference order. Bet365 is the most
# consistently populated across leagues and seasons; the others are
# fallbacks for older files where B365 columns are absent.
ODDS_PREFERENCE = [
    ("B365H", "B365D", "B365A"),      # Bet365
    ("BbAvH", "BbAvD", "BbAvA"),      # Betbrain average (older files)
    ("AvgH", "AvgD", "AvgA"),         # market average (newer files)
    ("PSH", "PSD", "PSA"),            # Pinnacle
    ("IWH", "IWD", "IWA"),            # Interwetten
]


class FootballDataCoUkError(RuntimeError):
    """Download or parse failure."""


def season_code(start_year: int) -> str:
    """2023 -> '2324'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _cache_path(div: str, season: int) -> Path:
    return RAW_DIR / f"{div}_{season_code(season)}.csv"


def _download(div: str, season: int, *, force: bool = False,
              retries: int = 4) -> Path:
    """Fetch one league-season CSV, caching completed seasons on disk.

    Retries on 5xx with exponential backoff. Football-Data.co.uk is a free
    service run on modest infrastructure and does go down; a transient 503
    should not kill an unattended Wednesday run. 404 is NOT retried -- a
    missing season will still be missing in eight seconds.
    """
    import time

    import requests

    path = _cache_path(div, season)
    if path.exists() and not force:
        return path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{season_code(season)}/{div}.csv"
    last = "unknown"

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60, headers={
                "User-Agent": "pl-forecaster/1.0 (portfolio project)"
            })
        except Exception as exc:                   # noqa: BLE001
            last = f"connection error: {exc}"
            log.warning("%s %s: %s (attempt %d/%d)",
                        div, season_code(season), last, attempt + 1, retries)
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200:
            # An HTML error page served with a 200 would otherwise be parsed
            # as if it were data. Check we actually got a CSV.
            head = resp.content[:200].lstrip().lower()
            if head.startswith(b"<!doctype") or head.startswith(b"<html"):
                last = "server returned an HTML page, not CSV"
                time.sleep(2 ** attempt)
                continue
            path.write_bytes(resp.content)
            log.info("Downloaded %s %s (%d KB)", div, season_code(season),
                     len(resp.content) // 1024)
            return path

        if resp.status_code == 404:
            raise FootballDataCoUkError(
                f"No data for {div} {season_code(season)} -- the season may "
                f"not exist yet, or the league did not run that year."
            )

        last = f"HTTP {resp.status_code}"
        if resp.status_code >= 500:
            # Their server, not our request. Back off and try again.
            wait = 2 ** attempt
            log.warning("%s %s: %s -- retrying in %ds (attempt %d/%d)",
                        div, season_code(season), last, wait,
                        attempt + 1, retries)
            time.sleep(wait)
            continue

        raise FootballDataCoUkError(f"{last} from {url}")

    # Fall back to a stale cache rather than failing outright: last week's
    # results are far more useful than none.
    if path.exists():
        log.warning("%s %s unreachable (%s) -- using cached copy.",
                    div, season_code(season), last)
        return path

    raise FootballDataCoUkError(
        f"{div} {season_code(season)} unreachable after {retries} attempts "
        f"({last}). If this is a 5xx, football-data.co.uk is likely down -- "
        f"check https://www.football-data.co.uk in a browser and retry later."
    )


def _parse(path: Path, season: int) -> pd.DataFrame:
    """Parse one CSV into tidy columns.

    Football-Data.co.uk files carry trailing blank rows and, in some
    seasons, stray unnamed columns. Rows without both team names are
    dropped -- those are footer artefacts, not matches.
    """
    # latin-1: some older files contain non-UTF8 bytes in team names.
    raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"], how="any")
    if raw.empty:
        raise FootballDataCoUkError(f"{path.name} contained no usable rows.")

    # dayfirst is essential: 03/04/2024 is 3 April, not 4 March.
    # format="mixed" because these files carry BOTH dd/mm/yy and dd/mm/yyyy,
    # sometimes within one league's history. Without it pandas warns and
    # silently falls back to per-element parsing.
    dates = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed",
                           errors="coerce")

    out = pd.DataFrame({
        "date": dates,
        "home_team_raw": raw["HomeTeam"].astype(str).str.strip(),
        "away_team_raw": raw["AwayTeam"].astype(str).str.strip(),
        "home_goals": pd.to_numeric(raw.get("FTHG"), errors="coerce"),
        "away_goals": pd.to_numeric(raw.get("FTAG"), errors="coerce"),
    })

    for h, d, a in ODDS_PREFERENCE:
        if h in raw.columns:
            oh = pd.to_numeric(raw[h], errors="coerce")
            od = pd.to_numeric(raw[d], errors="coerce")
            oa = pd.to_numeric(raw[a], errors="coerce")
            # Only accept a source that actually has values.
            if oh.notna().sum() > len(oh) * 0.5:
                out["odds_home"], out["odds_draw"], out["odds_away"] = oh, od, oa
                out.attrs["odds_source"] = h[:-1]
                break

    out["season"] = season
    out = out.dropna(subset=["date"])

    # Sanity-check the parsed dates against the season they claim to be in.
    # A day/month swap would show up here as dates outside the window.
    lo = pd.Timestamp(f"{season}-06-01")
    hi = pd.Timestamp(f"{season + 2}-01-01")
    stray = out[(out["date"] < lo) | (out["date"] > hi)]
    if len(stray) > len(out) * 0.05:
        raise FootballDataCoUkError(
            f"{path.name}: {len(stray)} of {len(out)} dates fall outside "
            f"{lo.date()}..{hi.date()}. Likely a date-format problem -- "
            f"check dayfirst parsing. Sample: {stray['date'].head(3).tolist()}"
        )
    out = out[(out["date"] >= lo) & (out["date"] <= hi)]

    return out.reset_index(drop=True)


def read_league_season(div: str, season: int, *,
                       force: bool = False) -> pd.DataFrame:
    """One league-season, tidy. Team names still RAW -- caller resolves them."""
    path = _download(div, season, force=force)
    return _parse(path, season)


def read_games(divisions: dict[str, str], seasons: list[int], *,
               refresh_current: bool = True,
               outage_threshold: int = 3) -> pd.DataFrame:
    """Read many league-seasons.

    Parameters
    ----------
    divisions
        Mapping of league key -> Football-Data.co.uk division code,
        e.g. {"ENG-Premier League": "E0", "ESP-La Liga": "SP1"}.
    seasons
        Season start years.
    refresh_current
        Re-download the most recent season even if cached. Completed
        seasons never change, so only the current one needs refetching.
    outage_threshold
        Give up after this many consecutive server-side failures across
        DIFFERENT league-seasons. See below.

    Resilience
    ----------
    A failure for one league-season is logged and skipped: partial data is
    far more useful than none, and the validator downstream will catch any
    league-season that came back incomplete.

    But consecutive 5xx failures across different files mean the HOST is
    down, not that one file is flaky. Without a circuit breaker, a full
    outage costs 15 seconds of backoff per league-season -- roughly twelve
    minutes of sleeping across five leagues and ten seasons, to learn that
    nothing worked. So we stop early and say so.

    Because data/raw/ is committed to the repo, completed seasons are read
    from disk and an outage only costs the current season's updates.
    """
    newest = max(seasons)
    frames = []
    consecutive_failures = 0
    aborted = False

    for league, div in divisions.items():
        if aborted:
            break
        for season in seasons:
            try:
                part = read_league_season(
                    div, season, force=(refresh_current and season == newest)
                )
            except FootballDataCoUkError as exc:
                consecutive_failures += 1
                log.warning("Skipping %s %s: %s", league, season_code(season), exc)
                if consecutive_failures >= outage_threshold:
                    log.error(
                        "%d consecutive failures across different files -- "
                        "football-data.co.uk appears to be down. Stopping here "
                        "and using whatever is cached locally.",
                        consecutive_failures,
                    )
                    aborted = True
                    break
                continue

            consecutive_failures = 0
            part["league"] = league
            frames.append(part)
            log.info("Parsed %s %s: %d matches", league, season_code(season), len(part))

    if not frames:
        raise FootballDataCoUkError(
            "No data could be read for any league-season requested, and no "
            "local copies were available. If data/raw/ is committed to your "
            "repo this should not happen -- check that those files are present."
        )

    if aborted:
        log.warning(
            "Returning partial data (%d league-seasons). The validator will "
            "reject any league-season that came back incomplete.", len(frames)
        )

    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
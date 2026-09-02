"""
football-data.org API client.

WHY THIS REPLACES THE FBref SCRAPER
------------------------------------
FBref was the one genuinely fragile source in the pipeline: HTML scraping,
aggressive rate limits, and a page layout that changes without notice. This
is a real REST API with a documented v4 contract, JSON responses, and an
auth token.

WHAT IT DOES *NOT* GIVE US
--------------------------
The free tier has no betting odds (a paid add-on from the Standard tier up)
and no match statistics -- corners, shots, possession. That is fine:

  * Dixon-Coles only needs goals, which the free tier provides.
  * Odds are our evaluation benchmark and continue to come from
    Football-Data.co.uk's free CSV files.

Do not be tempted to move odds here. Losing the closing-line benchmark would
gut the evaluation framework, which is the most defensible part of the
project.

RATE LIMITS
-----------
Free tier: 10 requests/minute. That is a per-minute limit, not a monthly
pool. We need roughly one request per season, so this is generous for our
use -- but the client throttles anyway and honours HTTP 429 Retry-After,
because a weekly cron job that gets banned is worse than one that is slow.

TERMS
-----
Free for non-commercial use. A portfolio project qualifies; if this ever
becomes commercial, the maintainer asks that you get in touch.

SETUP
-----
1. Register at https://www.football-data.org/client/register (email only,
   no card).
2. Put the token in your environment:

       export FOOTBALL_DATA_TOKEN="your_token_here"

   Or create a .env file in the repo root (it is gitignored):

       FOOTBALL_DATA_TOKEN=your_token_here

NEVER commit the token. It is a credential, and a public repo with a live
key in the history is a bad look on a portfolio project.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .names import resolve_series

log = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

# Competition codes available on the free tier. Adding a league here is the
# whole of "multi-league support" -- but see the README note on why depth
# beats breadth before you switch these on.
COMPETITIONS: dict[str, str] = {
    "ENG-Premier League": "PL",
    "ESP-La Liga": "PD",
    "ITA-Serie A": "SA",
    "GER-Bundesliga": "BL1",
    "FRA-Ligue 1": "FL1",
    "NED-Eredivisie": "DED",
    "POR-Primeira Liga": "PPL",
    "ENG-Championship": "ELC",
    "BRA-Serie A": "BSA",
    "UEFA-Champions League": "CL",
}

_MIN_INTERVAL = 6.5     # seconds between calls; 10/min with headroom
_last_call: float = 0.0


class FootballDataError(RuntimeError):
    """API returned something we cannot use."""


def _token() -> str:
    """Read the API token from env or a local .env file."""
    tok = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
    if tok:
        return tok

    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf8").splitlines():
            line = line.strip()
            if line.startswith("FOOTBALL_DATA_TOKEN="):
                return line.split("=", 1)[1].strip().strip("\"'")

    raise FootballDataError(
        "No API token found.\n"
        "  1. Register (free, email only) at "
        "https://www.football-data.org/client/register\n"
        "  2. export FOOTBALL_DATA_TOKEN='your_token'\n"
        "     ...or put FOOTBALL_DATA_TOKEN=your_token in a .env file "
        "at the repo root."
    )


def _get(path: str, params: dict | None = None, *, retries: int = 3) -> dict:
    """Throttled GET with 429 handling.

    Being a good citizen here is self-interested: the Wednesday cron job
    runs unattended, and an IP ban is far more expensive than a few seconds
    of sleeping.
    """
    import requests

    global _last_call

    headers = {"X-Auth-Token": _token()}
    url = f"{BASE_URL}{path}"

    for attempt in range(retries):
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)

        resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
        _last_call = time.monotonic()

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429:
            # The API tells us how long to wait; believe it.
            delay = int(resp.headers.get("Retry-After", 60))
            log.warning("Rate limited. Sleeping %ss (attempt %d/%d).",
                        delay, attempt + 1, retries)
            time.sleep(delay + 1)
            continue

        if resp.status_code in (401, 403):
            raise FootballDataError(
                f"HTTP {resp.status_code} -- token rejected or this resource "
                f"is not on the free tier. Note that odds and detailed match "
                f"statistics are paid add-ons. URL: {url}"
            )

        if resp.status_code >= 500:
            log.warning("Server error %s, retrying...", resp.status_code)
            time.sleep(5 * (attempt + 1))
            continue

        raise FootballDataError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")

    raise FootballDataError(f"Gave up on {url} after {retries} attempts.")


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def check_token() -> dict:
    """Verify the token works. Run this first -- fail fast, fail clearly."""
    data = _get("/competitions/PL")
    log.info("Token OK. Reached competition: %s", data.get("name", "?"))
    return data


def fetch_matches(competition: str = "PL", season: int | None = None) -> pd.DataFrame:
    """All matches for a competition-season, played and unplayed.

    The API returns one row per match with a `status` field. FINISHED matches
    carry a score; SCHEDULED and TIMED ones do not. We keep both -- played
    matches can cross-check the CSV feed, and unplayed ones are the fixture
    list the dashboard needs.

    Season is the START year: 2026 means the 2026/27 campaign.
    """
    params: dict[str, str | int] = {}
    if season is not None:
        params["season"] = season

    payload = _get(f"/competitions/{competition}/matches", params)
    matches = payload.get("matches", [])
    if not matches:
        raise FootballDataError(
            f"No matches returned for {competition} season={season}. "
            f"The free tier covers 12 competitions -- check the code is one "
            f"of them, and that the season exists."
        )

    rows = []
    for m in matches:
        score = (m.get("score") or {}).get("fullTime") or {}
        rows.append({
            "date": m.get("utcDate"),
            "home_team_raw": (m.get("homeTeam") or {}).get("name"),
            "away_team_raw": (m.get("awayTeam") or {}).get("name"),
            "home_goals": score.get("home"),
            "away_goals": score.get("away"),
            "status": m.get("status"),
            "matchweek": m.get("matchday"),
            "season_api": ((m.get("season") or {}).get("startDate") or "")[:4],
        })

    raw = pd.DataFrame(rows)

    # Resolve names LAST, on the unique set, so one bad name reports once
    # with every other offender rather than failing on the first row.
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.tz_localize(None),
        "home_team": resolve_series(raw["home_team_raw"]),
        "away_team": resolve_series(raw["away_team_raw"]),
        "home_goals": pd.to_numeric(raw["home_goals"], errors="coerce"),
        "away_goals": pd.to_numeric(raw["away_goals"], errors="coerce"),
        "matchweek": pd.to_numeric(raw["matchweek"], errors="coerce"),
        "status": raw["status"],
    })
    df["season"] = pd.to_numeric(raw["season_api"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["date", "home_team", "away_team"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_fixtures(competition: str = "PL", season: int | None = None) -> pd.DataFrame:
    """Unplayed matches only -- the forward fixture list.

    This is the direct replacement for the FBref scraper.
    """
    df = fetch_matches(competition, season)
    upcoming = df[df["status"].isin(["SCHEDULED", "TIMED", "POSTPONED"])].copy()
    return upcoming.drop(columns=["home_goals", "away_goals"], errors="ignore")


def probe_team_names(competition: str = "PL", season: int | None = None) -> list[str]:
    """Return raw API team names WITHOUT resolving them.

    Run this once per new competition before adding it. football-data.org
    writes full legal names -- "Nottingham Forest FC", "Wolverhampton
    Wanderers FC" -- which differ from both FBref and Football-Data.co.uk.
    This tells you exactly which aliases to add to CANONICAL_TEAMS.
    """
    params: dict[str, str | int] = {}
    if season is not None:
        params["season"] = season
    payload = _get(f"/competitions/{competition}/matches", params)

    names = set()
    for m in payload.get("matches", []):
        for side in ("homeTeam", "awayTeam"):
            n = (m.get(side) or {}).get("name")
            if n:
                names.add(n)
    return sorted(names)

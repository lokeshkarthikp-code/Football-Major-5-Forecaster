"""
Per-league configuration.

WHY THIS EXISTS
---------------
The original validator hardcoded 380 matches and 20 teams per season --
correct for the Premier League and wrong for two of the five leagues we
now cover. A validator that fires false alarms gets switched off, and a
switched-off validator is worse than none, so the invariants have to be
right per league.

The Ligue 1 case is the one that matters: it ran 20 teams / 380 matches
through 2022-23, then shrank to 18 teams / 306 matches from 2023-24. A
single fixed invariant would flag every season on one side of that line.
Season-dependent config is not over-engineering here; it is the minimum
needed to avoid crying wolf.

WHY MODELS ARE FITTED PER LEAGUE
--------------------------------
Dixon-Coles estimates team strengths from a graph of who played whom.
Within a league every team meets every other twice, so the graph is
dense and strengths are well identified. Across leagues, teams meet only
in European competition -- a handful of matches per season connecting
hundreds of clubs. That graph is far too sparse to place La Liga and
Bundesliga strengths on a common scale, so we do not try. Five leagues
means five models.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeagueConfig:
    """Structural facts about one league."""

    key: str                      # soccerdata league key
    code: str                     # Football-Data.co.uk code
    api_code: str                 # football-data.org competition code
    display: str
    country: str
    # Season-start-year -> (teams, matches). The entry with the largest
    # start year that is <= the season in question applies.
    format_by_season: dict[int, tuple[int, int]] = field(default_factory=dict)

    def expected(self, season: int) -> tuple[int, int]:
        """(teams, matches) expected for a completed season."""
        applicable = [s for s in self.format_by_season if s <= season]
        if not applicable:
            # Fall back to the earliest known format rather than guessing.
            key = min(self.format_by_season)
        else:
            key = max(applicable)
        return self.format_by_season[key]


LEAGUES: dict[str, LeagueConfig] = {
    "ENG-Premier League": LeagueConfig(
        key="ENG-Premier League", code="E0", api_code="PL",
        display="Premier League", country="England",
        format_by_season={2000: (20, 380)},
    ),
    "ESP-La Liga": LeagueConfig(
        key="ESP-La Liga", code="SP1", api_code="PD",
        display="La Liga", country="Spain",
        format_by_season={2000: (20, 380)},
    ),
    "ITA-Serie A": LeagueConfig(
        key="ITA-Serie A", code="I1", api_code="SA",
        display="Serie A", country="Italy",
        format_by_season={2004: (20, 380)},
    ),
    "GER-Bundesliga": LeagueConfig(
        key="GER-Bundesliga", code="D1", api_code="BL1",
        display="Bundesliga", country="Germany",
        format_by_season={1995: (18, 306)},
    ),
    "FRA-Ligue 1": LeagueConfig(
        key="FRA-Ligue 1", code="F1", api_code="FL1",
        display="Ligue 1", country="France",
        # Shrank from 20 to 18 clubs for 2023-24.
        format_by_season={2002: (20, 380), 2023: (18, 306)},
    ),
}

DEFAULT_LEAGUES = list(LEAGUES)

# Display order used in the UI -- roughly by size of following.
UI_ORDER = [
    "ENG-Premier League", "ESP-La Liga", "ITA-Serie A",
    "GER-Bundesliga", "FRA-Ligue 1",
]


def config(league: str) -> LeagueConfig:
    if league not in LEAGUES:
        raise KeyError(
            f"Unknown league {league!r}. Known: {', '.join(LEAGUES)}"
        )
    return LEAGUES[league]


def display_name(league: str) -> str:
    return LEAGUES[league].display if league in LEAGUES else league


def by_display(name: str) -> str:
    """Map a display name back to the league key."""
    for k, c in LEAGUES.items():
        if c.display == name:
            return k
    raise KeyError(f"No league with display name {name!r}")
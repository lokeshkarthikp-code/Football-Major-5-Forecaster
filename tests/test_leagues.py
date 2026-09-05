"""League configuration and multi-league validation."""

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from plfc.ingest import ValidationError, validate_matches
from plfc.leagues import DEFAULT_LEAGUES, LEAGUES, config, display_name
from plfc.names import resolve


class TestLeagueConfig:
    def test_all_five_present(self):
        assert len(DEFAULT_LEAGUES) == 5

    @pytest.mark.parametrize("league,season,teams,matches", [
        ("ENG-Premier League", 2024, 20, 380),
        ("ESP-La Liga",        2024, 20, 380),
        ("ITA-Serie A",        2024, 20, 380),
        ("GER-Bundesliga",     2024, 18, 306),
        # Ligue 1 shrank for 2023-24 -- the case a single constant gets wrong.
        ("FRA-Ligue 1",        2022, 20, 380),
        ("FRA-Ligue 1",        2023, 18, 306),
        ("FRA-Ligue 1",        2024, 18, 306),
    ])
    def test_expected_format(self, league, season, teams, matches):
        assert config(league).expected(season) == (teams, matches)

    def test_unknown_league_raises(self):
        with pytest.raises(KeyError):
            config("SCO-Premiership")


class TestContinentalNames:
    @pytest.mark.parametrize("raw,expected", [
        ("Ath Madrid", "Atletico Madrid"),
        ("Ath Bilbao", "Athletic Bilbao"),
        ("Espanol", "Espanyol"),
        ("Sociedad", "Real Sociedad"),
        ("Vallecano", "Rayo Vallecano"),
        ("M'gladbach", "Borussia Monchengladbach"),
        ("Ein Frankfurt", "Eintracht Frankfurt"),
        ("Bayern Munich", "Bayern Munich"),
        ("Dortmund", "Borussia Dortmund"),
        ("Paris SG", "Paris Saint-Germain"),
        ("St Etienne", "Saint-Etienne"),
        ("Inter", "Inter Milan"),
        ("Milan", "AC Milan"),
    ])
    def test_football_data_abbreviations(self, raw, expected):
        assert resolve(raw) == expected

    @pytest.mark.parametrize("a,b", [
        ("Ath Madrid", "Ath Bilbao"),
        ("Inter", "Milan"),
        ("Real Sociedad", "Real Madrid"),
        ("Real Betis", "Real Madrid"),
        ("Dortmund", "M'gladbach"),
    ])
    def test_lookalikes_stay_distinct(self, a, b):
        """One token apart, genuinely different clubs. A fuzzy matcher
        would merge these and corrupt a decade of data silently."""
        assert resolve(a) != resolve(b)


def _synth_league(league, season, n_teams, prefix, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"{prefix}{i:02d}" for i in range(n_teams)]
    atk = {t: rng.normal(0, .35) for t in teams}
    dfc = {t: rng.normal(0, .30) for t in teams}
    rows, d = [], pd.Timestamp(f"{season}-08-15")
    for h, a in permutations(teams, 2):
        lh = np.exp(atk[h] - dfc[a] + .26)
        la = np.exp(atk[a] - dfc[h])
        rows.append({
            "date": d, "season": season, "league": league,
            "home_team": h, "away_team": a,
            "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la),
        })
        d += pd.Timedelta(hours=8)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def multi_league():
    frames = []
    for i, (lg, cfg) in enumerate(LEAGUES.items()):
        for season in (2021, 2024):
            n, _ = cfg.expected(season)
            frames.append(_synth_league(lg, season, n, chr(65 + i), seed=i))
    return pd.concat(frames, ignore_index=True)


class TestMultiLeagueValidation:
    def test_passes_all_five(self, multi_league):
        validate_matches(multi_league, allow_partial_season=False)

    def test_ligue1_size_change_accepted(self, multi_league):
        sizes = multi_league[multi_league.league == "FRA-Ligue 1"].groupby("season").size()
        assert sizes[2021] == 380 and sizes[2024] == 306

    def test_catches_dropped_team_in_one_league(self, multi_league):
        bad = multi_league.drop(
            multi_league[(multi_league.league == "GER-Bundesliga")
                         & (multi_league.home_team == "D00")].index
        )
        with pytest.raises(ValidationError, match="GER-Bundesliga"):
            validate_matches(bad, allow_partial_season=False)

    def test_unknown_league_flagged(self, multi_league):
        bad = multi_league.copy()
        bad.loc[bad.index[:5], "league"] = "SCO-Premiership"
        with pytest.raises(ValidationError):
            validate_matches(bad, allow_partial_season=False)


def test_display_names():
    assert display_name("GER-Bundesliga") == "Bundesliga"
    assert display_name("FRA-Ligue 1") == "Ligue 1"
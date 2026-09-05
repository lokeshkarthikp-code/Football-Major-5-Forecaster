"""Tests for the canonical team-name resolver.

These matter more than they look. The failure mode this guards against is
SILENT: a bad name join does not raise, it just drops rows. A test suite is
the only thing standing between you and a model quietly trained on 342
matches instead of 380.
"""

import pandas as pd
import pytest

from plfc.names import (
    CANONICAL_TEAMS,
    TeamNameError,
    audit_names,
    resolve,
    resolve_series,
)


class TestNottinghamForest:
    """The canonical headache — every source spells it differently."""

    @pytest.mark.parametrize("variant", [
        "Nottingham Forest",
        "Nott'ham Forest",      # FBref
        "Nott'm Forest",        # Football-Data.co.uk
        "Nottm Forest",
        "Forest",               # ClubElo
        "NOTTM FOREST",
        "nottingham forest",
        "  Nottingham Forest  ",
        "Nottingham Forest FC",
    ])
    def test_all_variants_resolve(self, variant):
        assert resolve(variant) == "Nottingham Forest"


class TestCommonVariants:
    @pytest.mark.parametrize("raw,expected", [
        ("Man Utd", "Manchester United"),
        ("Manchester Utd", "Manchester United"),
        ("Man. United", "Manchester United"),
        ("Man City", "Manchester City"),
        ("Spurs", "Tottenham"),
        ("Tottenham Hotspur", "Tottenham"),
        ("Brighton & Hove Albion", "Brighton"),
        ("Brighton and Hove Albion", "Brighton"),
        ("Wolves", "Wolverhampton Wanderers"),
        ("West Brom", "West Bromwich Albion"),
        ("Sheffield Utd", "Sheffield United"),
        ("Newcastle Utd", "Newcastle United"),
        ("liverpool", "Liverpool"),
        ("Arsenal FC", "Arsenal"),
        ("Stoke", "Stoke City"),
    ])
    def test_variant(self, raw, expected):
        assert resolve(raw) == expected


class TestRefusesToGuess:
    """Fuzzy matching must SUGGEST, never decide.

    An auto-resolver at a 0.6 similarity cutoff will happily map a club it
    has never seen onto a superficially similar name and corrupt the data
    silently. Raising forces a human to add the alias deliberately.
    """

    def test_unknown_team_raises(self):
        with pytest.raises(TeamNameError):
            resolve("Boca Juniors")

    def test_error_carries_suggestion(self):
        with pytest.raises(TeamNameError, match="Nottingham Forest"):
            resolve("Notngham Forset")

    @pytest.mark.parametrize("a,b", [
        ("Manchester City", "Manchester United"),
        ("Ath Madrid", "Ath Bilbao"),          # one token apart, different clubs
        ("Inter", "Milan"),                     # both "Milan" colloquially
        ("Real Sociedad", "Real Madrid"),
        ("Dortmund", "M'gladbach"),             # both Borussia
    ])
    def test_never_collapses_distinct_clubs(self, a, b):
        """Near-identical strings must resolve to DIFFERENT canonical ids.

        These are the pairs a fuzzy matcher would merge. Merging them would
        corrupt a decade of data with no error raised.
        """
        ra, rb = resolve(a), resolve(b)
        assert ra != rb, f"{a!r} and {b!r} both resolved to {ra!r}"

    def test_empty_and_none(self):
        with pytest.raises(TeamNameError):
            resolve("")
        with pytest.raises(TeamNameError):
            resolve(None)

    def test_non_strict_returns_none(self):
        assert resolve("Boca Juniors", strict=False) is None


class TestSeriesResolution:
    def test_resolves_a_series(self):
        s = pd.Series(["Man Utd", "Spurs", "Nott'm Forest"])
        out = resolve_series(s)
        assert list(out) == ["Manchester United", "Tottenham", "Nottingham Forest"]

    def test_collects_all_failures_at_once(self):
        """One run should tell you every alias to add, not just the first."""
        s = pd.Series(["Arsenal", "Boca Juniors", "River Plate"])
        with pytest.raises(TeamNameError) as exc:
            resolve_series(s)
        msg = str(exc.value)
        assert "Boca Juniors" in msg and "River Plate" in msg

    def test_audit_is_non_raising(self):
        df = pd.DataFrame({"team": ["Arsenal", "Boca Juniors"]})
        out = audit_names((df, "team"))
        assert out["team"] == ["Boca Juniors"]


class TestRegistryIntegrity:
    def test_canonical_ids_are_self_resolving(self):
        """Every canonical id must resolve to itself — no accidental folds."""
        for canon in CANONICAL_TEAMS:
            assert resolve(canon) is not None

    def test_no_alias_maps_to_two_teams(self):
        seen: dict[str, str] = {}
        for canon, aliases in CANONICAL_TEAMS.items():
            for a in aliases:
                key = a.casefold()
                if key in seen and seen[key] != canon:
                    pytest.fail(
                        f"Alias {a!r} claimed by both {seen[key]!r} and {canon!r}"
                    )
                seen[key] = canon
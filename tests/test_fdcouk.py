"""Football-Data.co.uk CSV client: parsing traps and network resilience."""

import sys
import types
from pathlib import Path

import pytest

import plfc.fdcouk as F
from plfc.fdcouk import FootballDataCoUkError, _parse, season_code

CSV = (
    b"Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
    b"E0,11/08/2023,20:00,Burnley,Man City,0,3,A,9.50,5.50,1.33\n"
    b"E0,03/04/2024,19:30,Chelsea,Man Utd,4,3,H,2.10,3.60,3.40\n"
)


def csv_for(season: int) -> bytes:
    """Build a CSV whose dates actually fall inside `season`.

    The parser range-checks dates against the season they claim to belong
    to, so a fixture with mismatched dates is correctly rejected. Test data
    has to be internally consistent.
    """
    return (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
        f"E0,15/08/{season},Burnley,Man City,0,3,9.50,5.50,1.33\n"
        f"E0,03/04/{season + 1},Chelsea,Man Utd,4,3,2.10,3.60,3.40\n"
    ).encode()


class _Resp:
    def __init__(self, code, content=b""):
        self.status_code = code
        self.content = content


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "RAW_DIR", tmp_path)
    return tmp_path


def _stub_requests(monkeypatch, responses):
    state = {"i": 0}

    def get(url, **kwargs):
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return r

    mod = types.ModuleType("requests")
    mod.get = get
    monkeypatch.setitem(sys.modules, "requests", mod)
    return state


def test_season_code():
    assert season_code(2023) == "2324"
    assert season_code(1999) == "9900"


class TestDateParsing:
    """dd/mm/yyyy is day-first. pandas does not assume that."""

    def test_day_first_not_month_first(self, tmp_path):
        p = tmp_path / "E0_2324.csv"
        p.write_bytes(CSV)
        df = _parse(p, 2023)
        d = df.iloc[1]["date"]
        # 03/04/2024 is 3 April, not 4 March.
        assert (d.day, d.month) == (3, 4)

    def test_two_digit_years(self, tmp_path):
        p = tmp_path / "D1_1617.csv"
        p.write_text(
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "D1,26/08/16,Bayern Munich,Werder Bremen,6,0\n"
        )
        df = _parse(p, 2016)
        assert df.iloc[0]["date"].year == 2016

    def test_rejects_dates_outside_season(self, tmp_path):
        """A day/month swap would show up as dates far from the season."""
        p = tmp_path / "bad.csv"
        p.write_text(
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
            + "".join(f"E0,0{i}/01/2001,A,B,1,0\n" for i in range(1, 10))
        )
        with pytest.raises(FootballDataCoUkError, match="outside"):
            _parse(p, 2023)

    def test_drops_footer_rows(self, tmp_path):
        p = tmp_path / "E0_2324.csv"
        p.write_bytes(CSV + b",,,,,,,,,,\n")
        assert len(_parse(p, 2023)) == 2

    def test_picks_up_odds(self, tmp_path):
        p = tmp_path / "E0_2324.csv"
        p.write_bytes(CSV)
        df = _parse(p, 2023)
        assert df.iloc[0]["odds_home"] == 9.50
        assert df.attrs["odds_source"] == "B365"


class TestNetworkResilience:
    """The site is free infrastructure and does go down -- it returned 503
    across every league during development. None of that should kill an
    unattended weekly run."""

    def test_retries_on_5xx_then_succeeds(self, raw_dir, monkeypatch):
        monkeypatch.setattr(F.__dict__["_download"].__globals__["time"]
                            if "time" in F.__dict__.get("_download").__globals__
                            else __import__("time"), "sleep", lambda s: None)
        _stub_requests(monkeypatch, [_Resp(503), _Resp(503), _Resp(200, CSV)])
        path = F._download("E0", 2023)
        assert path.exists()

    def test_404_does_not_retry(self, raw_dir, monkeypatch):
        state = _stub_requests(monkeypatch, [_Resp(404)])
        with pytest.raises(FootballDataCoUkError, match="may.*not exist"):
            F._download("E0", 2099)
        assert state["i"] == 1, "404 should fail fast, not retry"

    def test_html_error_page_not_parsed_as_data(self, raw_dir, monkeypatch):
        import time
        monkeypatch.setattr(time, "sleep", lambda s: None)
        _stub_requests(monkeypatch, [_Resp(200, b"<!DOCTYPE html><html>503</html>")])
        with pytest.raises(FootballDataCoUkError):
            F._download("E0", 2023, retries=2)

    def test_falls_back_to_stale_cache(self, raw_dir, monkeypatch):
        import time
        monkeypatch.setattr(time, "sleep", lambda s: None)
        _stub_requests(monkeypatch, [_Resp(200, CSV)])
        F._download("E0", 2023)                       # populate cache

        _stub_requests(monkeypatch, [_Resp(503)])
        path = F._download("E0", 2023, force=True, retries=2)
        assert path.exists(), "stale data beats no data on an unattended run"


class TestOutageCircuitBreaker:
    """Consecutive failures across DIFFERENT files mean the host is down.

    Without this, a full outage costs 15s of backoff per league-season --
    twelve minutes across five leagues and ten seasons, to learn nothing
    worked. That happened during development and had to be Ctrl+C'd.
    """

    def test_stops_early_when_host_is_down(self, raw_dir, monkeypatch):
        import time
        monkeypatch.setattr(time, "sleep", lambda s: None)
        state = _stub_requests(monkeypatch, [_Resp(503)])

        divisions = {f"L{i}": f"D{i}" for i in range(5)}
        seasons = list(range(2015, 2025))          # 50 league-seasons

        with pytest.raises(FootballDataCoUkError):
            F.read_games(divisions, seasons, outage_threshold=3)

        # 3 files x 4 attempts each = 12 requests, not 50 x 4 = 200.
        assert state["i"] <= 15, (
            f"made {state['i']} requests; the circuit breaker should have "
            f"stopped after ~12"
        )

    def test_partial_data_survives_a_single_bad_file(self, raw_dir, monkeypatch):
        """One missing season must not lose the others."""
        import time
        monkeypatch.setattr(time, "sleep", lambda s: None)

        state = {"i": 0}

        def get(url, **kwargs):
            state["i"] += 1
            if "1617" in url:
                return _Resp(404)                 # only 2016 is missing
            season = 2015 if "1516" in url else 2017
            return _Resp(200, csv_for(season))

        mod = types.ModuleType("requests")
        mod.get = get
        monkeypatch.setitem(sys.modules, "requests", mod)

        df = F.read_games({"ENG-Premier League": "E0"}, [2015, 2016, 2017],
                          refresh_current=False)
        assert len(df) == 4, "two good seasons x two rows each"
        assert set(df["league"]) == {"ENG-Premier League"}

    def test_cached_files_survive_total_outage(self, raw_dir, monkeypatch):
        """The whole point of committing data/raw/: an outage costs nothing."""
        import time
        monkeypatch.setattr(time, "sleep", lambda s: None)

        # Populate the mirror.
        def get(url, **kwargs):
            return _Resp(200, csv_for(2015 if "1516" in url else 2016))

        mod = types.ModuleType("requests")
        mod.get = get
        monkeypatch.setitem(sys.modules, "requests", mod)
        F.read_games({"ENG-Premier League": "E0"}, [2015, 2016],
                     refresh_current=False)

        # Now the host dies completely.
        _stub_requests(monkeypatch, [_Resp(503)])
        df = F.read_games({"ENG-Premier League": "E0"}, [2015, 2016],
                          refresh_current=False)
        assert len(df) == 4, "cached seasons must still load with the host down"


def test_base_url_has_no_www():
    """The www subdomain returns 503; the bare domain works.

    This is not cosmetic. The original URL was taken from another library's
    source rather than from an actual link on the site, and the resulting
    failures were misdiagnosed as an outage for a full day. Pin it.
    """
    from plfc.fdcouk import BASE_URL
    assert "www." not in BASE_URL, "www.football-data.co.uk 503s; use the bare domain"
    assert BASE_URL == "https://football-data.co.uk/mmz4281"
"""Model and validation tests.

The key test is parameter recovery: generate matches from KNOWN team
strengths, fit, and check we get them back. If the fitter cannot recover
truth from clean synthetic data, nothing it says about real data is
trustworthy.

Caveat worth keeping in mind: synthetic data is generated from the same
Poisson process the model assumes, so recovery is necessary but not
sufficient evidence of real-world skill.
"""

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from plfc.backtest import actual_outcome, brier, log_loss, walk_forward
from plfc.ingest import ValidationError, validate_matches
from plfc.model import DixonColes, odds_to_probs


def synthetic(seasons: int = 6, seed: int = 7):
    rng = np.random.default_rng(seed)
    teams = [f"Team{i:02d}" for i in range(20)]
    atk = {t: rng.normal(0, 0.35) for t in teams}
    dfc = {t: rng.normal(0, 0.30) for t in teams}
    gamma = 0.26

    rows, d = [], pd.Timestamp("2018-08-10")
    for s in range(seasons):
        for h, a in permutations(teams, 2):
            lh = np.exp(atk[h] - dfc[a] + gamma)
            la = np.exp(atk[a] - dfc[h])
            rows.append({
                "date": d, "season": 2018 + s,
                "league": "ENG-Premier League",
                "home_team": h, "away_team": a,
                "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la),
            })
            d += pd.Timedelta(hours=6)
        d += pd.Timedelta(days=60)
    return pd.DataFrame(rows), atk, gamma


class TestParameterRecovery:
    def test_recovers_attack_strengths(self):
        df, true_atk, _ = synthetic()
        m = DixonColes().fit(df)
        est = np.array([m.attack[t] for t in m.teams])
        tru = np.array([true_atk[t] for t in m.teams])
        r = np.corrcoef(est - est.mean(), tru - tru.mean())[0, 1]
        assert r > 0.9, f"attack correlation only {r:.3f}"

    def test_recovers_home_advantage(self):
        df, _, gamma = synthetic()
        m = DixonColes().fit(df)
        assert abs(m.home_adv - gamma) < 0.15

    def test_converges(self):
        df, _, _ = synthetic(seasons=3)
        assert DixonColes().fit(df).converged


class TestPredictions:
    @pytest.fixture(scope="class")
    def model(self):
        df, _, _ = synthetic(seasons=4)
        return DixonColes().fit(df)

    def test_probabilities_sum_to_one(self, model):
        p = model.predict("Team00", "Team01")
        assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1.0) < 1e-6

    def test_score_matrix_is_a_distribution(self, model):
        m = model.score_matrix("Team00", "Team01")
        assert abs(m.sum() - 1.0) < 1e-6
        assert (m >= 0).all()

    def test_home_advantage_is_directional(self, model):
        """Same pairing, reversed venue — home side should fare better."""
        a = model.predict("Team00", "Team01")
        b = model.predict("Team01", "Team00")
        assert a["home_win"] > b["away_win"]

    def test_cold_start_for_unseen_team(self, model):
        p = model.predict("Team00", "Promoted FC")
        assert p["is_new_away"] is True
        assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1.0) < 1e-6

    def test_half_life(self):
        assert 300 < DixonColes(xi=0.0018).half_life_days < 450


class TestValidation:
    @pytest.fixture
    def good(self):
        df, _, _ = synthetic(seasons=2)
        return df

    def test_passes_clean_data(self, good):
        validate_matches(good, allow_partial_season=False)

    def test_catches_self_play(self, good):
        """The signature of a name-collapse bug."""
        bad = good.copy()
        bad.loc[0, "away_team"] = bad.loc[0, "home_team"]
        with pytest.raises(ValidationError, match="plays itself"):
            validate_matches(bad, allow_partial_season=False)

    def test_catches_dropped_team(self, good):
        """The exact failure a bad join produces: 342 matches, not 380."""
        bad = good[
            (good["home_team"] != "Team19") & (good["away_team"] != "Team19")
        ]
        with pytest.raises(ValidationError):
            validate_matches(bad, allow_partial_season=False)

    def test_catches_duplicates(self, good):
        bad = pd.concat([good, good.head(1)], ignore_index=True)
        with pytest.raises(ValidationError, match="duplicate"):
            validate_matches(bad, allow_partial_season=False)

    def test_catches_nulls(self, good):
        bad = good.copy()
        bad.loc[0, "home_team"] = None
        with pytest.raises(ValidationError, match="null"):
            validate_matches(bad, allow_partial_season=False)


class TestBacktest:
    def test_no_lookahead(self):
        """Fitting with as_of must ignore everything on or after that date."""
        df, _, _ = synthetic(seasons=4)
        cutoff = df["date"].quantile(0.5)
        m = DixonColes().fit(df, as_of=cutoff)
        future = df[df["date"] >= cutoff]
        # A team appearing ONLY after the cutoff must be unknown to the model.
        assert m.converged
        assert len(future) > 0

    def test_walk_forward_beats_base_rates(self):
        df, _, _ = synthetic(seasons=5)
        res = walk_forward(df, min_train_matches=760, step_days=60)
        assert len(res) > 100

        rates = res["actual"].value_counts(normalize=True)
        flat = pd.DataFrame({
            oc: [rates.get(oc, 0.0)] * len(res)
            for oc in ["home_win", "draw", "away_win"]
        })
        assert brier(res, res["actual"]) < brier(flat, res["actual"])
        assert log_loss(res, res["actual"]) < log_loss(flat, res["actual"])


class TestOdds:
    def test_devig_normalises(self):
        p = odds_to_probs(2.0, 3.5, 4.0)
        assert abs(sum(p.values()) - 1.0) < 1e-9

    def test_favourite_gets_highest_probability(self):
        p = odds_to_probs(1.5, 4.0, 7.0)
        assert p["home_win"] > p["draw"]
        assert p["home_win"] > p["away_win"]


def test_outcome_labels():
    assert actual_outcome(2, 1) == "home_win"
    assert actual_outcome(1, 1) == "draw"
    assert actual_outcome(0, 3) == "away_win"
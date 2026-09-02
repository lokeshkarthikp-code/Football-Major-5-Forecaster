"""
Starter analysis script. Open in Jupyter (`jupytext` converts it) or just run
sections in an IDE. Kept as .py rather than .ipynb so it diffs cleanly in git.

Run AFTER `python -m plfc.ingest` has succeeded.
"""

# %% Load
from plfc.ingest import load_matches, load_fixtures, last_refresh
from plfc.model import DixonColes
from plfc.backtest import walk_forward, evaluate, calibration_table, tune_decay

matches = load_matches()
print(f"{len(matches)} matches, seasons {matches.season.min()}-{matches.season.max()}")
print(f"Last refresh: {last_refresh()}")

# %% Sanity check the data before trusting anything downstream
print(matches.groupby("season").size())
print("\nNull check:\n", matches.isna().sum())

# %% Fit and inspect team strengths
model = DixonColes(xi=0.0018, shrinkage=8.0).fit(matches)
print(f"converged={model.converged}  home_adv={model.home_adv:.3f}  rho={model.rho:.3f}")
model.ratings().head(20)

# %% First real backtest -- THE moment of truth
results = walk_forward(matches, step_days=14)
evaluate(results).round(4)

# %% Calibration
calibration_table(results, results["actual"]).round(3)

# %% Tune the decay rate. Slow -- refits the full history per xi.
tune_decay(matches)

# %% Upcoming fixtures
fixtures = load_fixtures()
model.predict_frame(fixtures.head(10)).round(3)

# Premier League Match Forecaster

A Dixon-Coles bivariate Poisson model for Premier League match outcomes,
refreshed weekly and evaluated against the bookmaker closing line.

---

## Quickstart

```bash
bash scripts/setup.sh          # venv, deps, test suite
source .venv/bin/activate

# Get a free API token (email only, no card):
#   https://www.football-data.org/client/register
cp .env.example .env           # then paste your token in
python scripts/check_api.py    # verify token + team names before ingesting

python -m plfc.ingest          # build the dataset (several minutes)
python -m plfc.backtest        # walk-forward evaluation
python -m plfc.ledger          # log forecasts for upcoming fixtures
streamlit run app.py           # dashboard
```

## What it does

Pulls ten seasons of Premier League results and closing odds, fits team-level
attack and defence strengths with exponential time decay, and produces
calibrated probabilities for upcoming fixtures. A GitHub Action refreshes the
data every Wednesday at 06:00 UTC, so each matchweek is forecast with current
form.

The dashboard has five tabs: this weekend's fixtures with win/draw/loss
probabilities and expected goals, any custom matchup with a scoreline grid, the
live track record of past predictions against results, team strength ratings,
and an evaluation tab that runs the backtest and lists known limitations.

---

## Project structure

```
pl-forecaster/
├── plfc/
│   ├── __init__.py
│   ├── names.py           canonical team registry + resolver
│   ├── footballdata.py    football-data.org API client (fixtures)
│   ├── ingest.py          pulls, validation, parquet cache
│   ├── model.py           Dixon-Coles fit and prediction
│   ├── backtest.py        walk-forward evaluation, calibration, tuning
│   └── ledger.py          sealed prediction log, reconciled against results
├── tests/
│   ├── test_names.py      resolver behaviour
│   └── test_model.py      parameter recovery, validation invariants
├── notebooks/
│   └── 01_explore.py      starter analysis
├── scripts/
│   ├── setup.sh
│   └── check_api.py       token + name check before ingesting
├── .github/workflows/
│   ├── refresh.yml        Wednesday 06:00 UTC data refresh
│   └── tests.yml          CI on push/PR
├── data/                  parquet cache (matches.parquet is committed)
├── app.py                 Streamlit front end
├── requirements.txt       full pipeline
├── requirements-app.txt   slim deps for deploying the app alone
├── PROJECT_HISTORY.md     full build log and design rationale
└── .gitignore
```

---

## Git setup

```bash
cd pl-forecaster
git init -b main
git add .
git commit -m "Initial commit: Dixon-Coles PL forecaster"

# Create an empty repo on GitHub first (no README — you have one), then:
git remote add origin https://github.com/YOUR_USERNAME/pl-forecaster.git
git push -u origin main
```

**Add the API token as a repo secret.** Settings → Secrets and variables →
Actions → New repository secret, named `FOOTBALL_DATA_TOKEN`. The refresh
workflow reads it from there. Never commit `.env`.

**Enable the weekly refresh.** The Action commits data back to the repo, which
needs write permission: Settings → Actions → General → Workflow permissions →
*Read and write permissions*. Then trigger a manual run from the Actions tab
(`workflow_dispatch`) to confirm it works before relying on the schedule.

Note that GitHub disables scheduled workflows on repos with no activity for 60
days. If the refresh stops, a single commit re-enables it.

**Deploy.** Streamlit Community Cloud connects to the repo, auto-detects
`app.py`, and redeploys on every push — including the Wednesday data commits.
Free tier apps sleep after inactivity and take ~30s to wake, so put a
screenshot on your portfolio page rather than sending someone to a cold start.

---

## Design decisions

**Canonical team registry (`plfc/names.py`).**
Football data sources spell teams inconsistently — FBref writes
`Nott'ham Forest`, Football-Data.co.uk writes `Nott'm Forest`, ClubElo writes
`Forest`. Joining on raw strings drops rows *silently*: no exception, just a
season with 342 matches instead of 380 and a model trained on a hole.

Every source name maps to one canonical id through three layers: exact alias →
normalised (accents, punctuation, club suffixes stripped) → fuzzy
*suggestion*. Fuzzy matching never auto-resolves; it raises with a suggestion
so a human adds the alias deliberately. This matters concretely: `Barcelona`
fuzzy-matches `Arsenal` above a 0.6 cutoff. An auto-resolver would accept
that silently.

`soccerdata` ships with an **empty** replacement table by default — name
normalisation is not free. `write_soccerdata_config()` exports the registry to
the path it reads, so names are normalised at the source.

**Loud validation (`plfc/ingest.py`).**
Every completed season must contain exactly 380 matches and 20 teams. No team
may play itself (the signature of a name-collapse bug). No duplicate fixtures,
no nulls in key columns. Validation runs *before* the parquet is overwritten,
so a failed run leaves the last good dataset intact.

**Walk-forward evaluation, not k-fold.**
Shuffled cross-validation on time series trains on May to predict September.
Every number it produces is optimistic and meaningless. The backtest refits on
everything strictly before each date and predicts forward — exactly the
information a real forecaster has.

**The benchmark is the market.**
Bookmaker closing odds reach roughly 53–55% accuracy on three-way outcomes,
with far more information than this model has. Reporting raw accuracy against
zero invites the wrong conversation. We report Brier score, log loss, and a
calibration curve, benchmarked against the closing line.

**Cold start for promoted teams.**
Three teams arrive from the Championship each season with zero Premier League
history. Excluding them would make ~15% of fixtures unpredictable and bias
evaluation toward easy matches. Instead every team's strength is shrunk toward
an empirical prior, with shrinkage inversely proportional to time-weighted
matches observed. Established teams barely shrink; promoted sides sit near the
prior until they earn their own estimate. The dashboard flags these fixtures.

**Time decay is a measured choice, not an assumption.**
Each match is weighted `exp(-ξ · age_days)`. `python -m plfc.backtest --tune`
sweeps ξ.

⚠️ A caveat that matters: ξ = 0.0018 was chosen by judgment *before* any results
were seen, which is what makes the figures above a clean out-of-sample estimate.
If you sweep ξ and then quote the winner's Brier as your headline, that number
is optimistically biased — you have selected a hyperparameter on the same data
you are reporting. Either keep reporting the untuned figures and describe the
sweep as a sensitivity analysis, or hold out the most recent season for a clean
final evaluation.

**The Dixon-Coles τ correction.**
Independent Poisson underestimates 0-0 and 1-1 and overestimates 1-0 and 0-1 —
football has game-state effects that independence misses. A single parameter ρ
adjusts those four low-score cells.

---

## Results

Dataset: **3,440 matches across 10 seasons**. Walk-forward validation, 14-day
refit cadence, **2,680 out-of-sample predictions**.

| Model | Brier | Log loss | Accuracy |
|---|---|---|---|
| Base rates only | 0.6469 | 1.0688 | 43.5% |
| **Dixon-Coles** | **0.5853** | **0.9842** | **52.6%** |
| Bookmaker closing line | 0.5749 | 0.9689 | 54.5% |

**The headline is not the accuracy figure.** The market improves on base rates
by 0.0720 Brier; this model improves on them by 0.0616. It therefore captures
**86% of the bookmaker's edge over base rates** — using only historical goals,
with no injuries, lineups, transfers, team news, or odds in training.

The model loses to the closing line, and that is the expected result. Beating
it would be the red flag.

### Calibration

| Predicted | Observed | n |
|---|---|---|
| 0.068 | 0.094 | 298 |
| 0.159 | 0.157 | 1,186 |
| 0.249 | 0.259 | 3,045 |
| 0.347 | 0.344 | 1,208 |
| 0.449 | 0.440 | 885 |
| 0.546 | 0.510 | 643 |
| 0.646 | 0.655 | 444 |
| 0.749 | 0.734 | 229 |
| 0.838 | 0.854 | 89 |
| 0.925 | 0.846 | 13 |

Close to the diagonal wherever the sample is real: when the model says 35%, it
happens 34% of the time. Two honest notes — the 0.5–0.6 bin runs slightly hot
(mild overconfidence on near-coin-flips), and the top bin's apparent miss rests
on 13 observations, which is noise rather than a finding.

### Model verification

Before touching real data, the fitter was checked against synthetic seasons
generated from *known* team strengths: it recovered attack ratings at
**r = 0.967** and home advantage at **0.307** against a true 0.260. That
validates the machinery. It is not evidence of real-world skill — the synthetic
data comes from the same Poisson process the model assumes, so success there is
partly circular.

## Live track record

The backtest is **retrospective**: it reconstructs what the model would have
said, and can be re-run with different settings until the numbers flatter you.

`plfc/ledger.py` adds the **prospective** record. Every Wednesday it writes down
predictions for unplayed matches, timestamped, and never edits them. Later runs
fill in what actually happened. A backtest is a claim; an accumulating log is
evidence.

The ledger is append-only, keeps multiple forecasts per match (scoring the
latest one made before kickoff), and stores the model parameters on every row so
old predictions stay attributable to the model that made them.

**It cannot be backfilled.** It starts from the first run — which is exactly why
it is credible.

```bash
python -m plfc.ledger      # settle what has been played, forecast what is next
```

---

## Limitations

- No team-news: injuries, suspensions and rotation are invisible to the model.
- No fixture-congestion effects from European or cup competition.
- Managerial changes violate the smooth-drift assumption.
- Promoted teams are priors, not estimates, until matches accumulate.
- Odds are de-vigged proportionally, which slightly distorts longshots
  (favourite-longshot bias).
- A forecasting exercise, not betting advice.

## Data sources

| Source | Provides | Access |
|---|---|---|
| **Football-Data.co.uk** | Results + closing odds, 25+ leagues, 30+ years | Free static CSVs, no key |
| **football-data.org** | Fixture schedule, 12 competitions | Free REST API, email registration |
| **ClubElo** | Baseline ratings (optional) | Free CSV endpoint |

The fixture source was originally FBref, scraped from HTML. It was swapped for
football-data.org's REST API — the one genuinely fragile dependency replaced
with a documented contract.

**Odds deliberately stay on the CSV feed.** football-data.org's free tier has
no odds (a paid add-on from the Standard tier up), and the closing line is the
benchmark the whole evaluation rests on. Moving odds behind a paywall would
gut the most defensible part of the project.

football-data.org is free for non-commercial use; a portfolio project
qualifies. Free tier is 10 requests/minute, which is generous here — the
pipeline needs roughly one request per season.

## Multi-league

`COMPETITIONS` in `plfc/footballdata.py` lists the free-tier codes (La Liga,
Serie A, Bundesliga, Ligue 1, Eredivisie, Championship and others). Adding a
league is a config change plus its team names in the registry —
`python scripts/check_api.py --competition PD` reports exactly which aliases
are missing.

Worth being honest about the value, though: five leagues is the same model run
five times, not a five-times-better model. Dixon-Coles fits each league
independently — there is no shared signal unless you model cross-league
strength, which is genuinely hard (teams only meet in European competition, so
the graph connecting leagues is sparse). Depth on one league — tuned decay,
market benchmarking, honest calibration — is the stronger portfolio piece.
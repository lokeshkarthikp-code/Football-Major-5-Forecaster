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
streamlit run app.py           # dashboard
```

## What it does

Pulls ten seasons of Premier League results and closing odds, fits team-level
attack and defence strengths with exponential time decay, and produces
calibrated probabilities for upcoming fixtures. A GitHub Action refreshes the
data every Wednesday at 06:00 UTC, so each matchweek is forecast with current
form.

The dashboard shows this weekend's fixtures with win/draw/loss probabilities,
expected goals, a scoreline heatmap, team strength ratings, and a live
backtest against the market.

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
│   └── backtest.py        walk-forward evaluation, calibration, tuning
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
├── requirements.txt
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

**Time decay is tuned, not assumed.**
Each match is weighted `exp(-ξ · age_days)`. `python -m plfc.backtest --tune`
sweeps ξ. The fitted half-life is itself a reportable finding.

**The Dixon-Coles τ correction.**
Independent Poisson underestimates 0-0 and 1-1 and overestimates 1-0 and 0-1 —
football has game-state effects that independence misses. A single parameter ρ
adjusts those four low-score cells.

---

## Status

**Verified:** The model recovers known team strengths from synthetic data at
r = 0.967, with home advantage fitted at 0.307 against a true 0.260. The
walk-forward backtest runs end to end and beats base rates on Brier
(0.614 vs 0.646) and log loss. The test suite passes and is network-free.

**Not yet verified:** The ingestion code has never run against live sources —
the tests validate the machinery, not real-world skill. Synthetic data is
generated from the same Poisson process the model assumes, so good performance
there is partly circular. Expect to debug the first real
`python -m plfc.ingest`; column names in `read_games()` are the likeliest
break point, and `audit_names()` reports unresolved names without raising
while you sort it out.

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

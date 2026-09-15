# PL Forecaster — Complete Project History

*Everything that happened, why, and what the numbers mean. Written so you can
explain any part of this without the code in front of you.*

---

## Part 1: How we got here

### The starting point

You brought two public repositories as candidate portfolio projects, both from
the same author:

- `mar-antaya/2025_f1_predictions`
- `mar-antaya/2025_pl_predictions`

Both were assessed before any code was written. Neither survived as a base.

### Why the F1 repo was rejected

The structure was one standalone script per race (`prediction1.py` through
`prediction24.py`), each pulling last year's race session via FastF1, adding
hardcoded qualifying times, and fitting a `GradientBoostingRegressor` to
predict mean lap time.

Five problems, in order of severity:

**1. Target leakage.** The target `y` was each driver's mean 2024 lap time.
One of the features was `TotalSectorTime (s)` — the sum of that same driver's
mean sector times, from the same session. Sectors sum to lap time by
definition. The model was predicting a number from a near-identical copy of
itself. This invalidates every reported error figure.

**2. Sample size.** Roughly 13–20 rows per race. A 300-tree, depth-5 gradient
booster was being fit on about 10 training rows with a 2–3 row test set. The
reported MAE was noise.

**3. Predictions were fitted values.** The line
`clean_data["PredictedRaceTime (s)"] = model.predict(X)` predicts on the full
dataset, including the training rows. The published rankings were never
out-of-sample.

**4. An invented feature.** The `average_2025` dictionary contained values
88.0, 89.1, 89.2, 89.3, 89.4, 89.5 — incrementing by exactly 0.1 down the
list. That is a hand-written ranking of who the author thought was fast,
presented as data. The model was partly being told the answer.

**5. Wrong target.** Mean lap time does not determine a race winner. Strategy,
tyre degradation, safety cars and reliability do. None were modelled.

Plus: no baseline comparison (predicting the qualifying order is a strong
baseline in F1 and nothing tried to beat it), 24 near-duplicate scripts with no
shared module, and an API key placeholder committed to source.

### Why the PL repo was rejected

It contained a single 47-line file, `fbrefdata_example.py`. It requested
FBref's Premier League passing page, handed the HTML to `pd.read_html`, took
the first table, flattened the two-row header, and renamed columns that came
out as `Unnamed: 17_level_0_Ast`. Then it filtered to two hardcoded teams and
printed four columns.

No model. It was an example script, as the filename says.

### Why football, not F1

Three reasons decided it:

| | F1 | Football |
|---|---|---|
| Sample size | ~20 rows per race | 380 matches per season, ~3,400 over 10 |
| Target | "Mean lap time" as a proxy for winning | Home win / draw / away win, unambiguous |
| Benchmark | None available | Bookmaker closing odds, free and historical |

The benchmark was the deciding factor. Without one, any accuracy number you
report is uninterpretable.

---

## Part 2: The data journey

This changed three times. The history matters because each change was a
response to a specific problem.

### Attempt 1 — scraping FBref directly

The original PL repo's approach. Rejected before we built on it:

- `pd.read_html(...)[0]` assumes the table you want is first on the page. FBref
  pages often have several, and the ordering is not guaranteed.
- Hardcoded renames like `Unnamed: 17_level_0_Ast` break whenever FBref adds or
  reorders a column, which they do.
- FBref rate-limits aggressively. Building a historical dataset in a loop gets
  you blocked within the hour.
- The URL pointed at *current-season aggregates*, which are nearly useless for
  match prediction. You need one row per match with each team's form as of that
  date.

### Attempt 2 — `soccerdata` for everything

`soccerdata` is a maintained Python package wrapping FBref, Football-Data.co.uk,
ClubElo, Understat and others. It handles caching and rate limits and returns
consistent DataFrames. Much better than hand-rolled scraping.

We used it for three sources:

- **Football-Data.co.uk** — results + closing odds (static CSV downloads)
- **FBref** — the forward fixture list (HTML scraping)
- **ClubElo** — baseline ratings (CSV endpoint)

**A discovery worth remembering:** `soccerdata` ships with an *empty* team-name
replacement table by default. It reads overrides from
`~/soccerdata/config/teamname_replacements.json`, and if that file does not
exist it logs a notice and does nothing. Cross-source name normalisation is not
free — you have to supply it. That became our `names.py`.

### Attempt 3 — football-data.org API for fixtures (current)

You asked whether an API would be easier than scrapers. It is, but only
partially, and the detail matters.

**What we switched:** the fixture list, from FBref scraping to the
football-data.org REST API. That was the one genuinely fragile dependency.

**What we deliberately did not switch:** the odds. football-data.org's free
tier has no betting odds — they are a €15/month add-on available only from the
€49/month Standard tier upward. **That is the paywall you hit.** Moving odds
behind it would have destroyed the evaluation framework, which is the most
defensible part of the whole project.

**Current architecture:**

| Source | Provides | Access | Type |
|---|---|---|---|
| Football-Data.co.uk | Results + closing odds, 10 seasons | Free, no key | Static CSV |
| football-data.org | Fixture schedule | Free, email registration | REST API |
| ClubElo | Baseline ratings (optional) | Free, no key | CSV endpoint |

football-data.org is free for non-commercial use; a portfolio project
qualifies. The rate limit is 10 requests/minute, which is generous — the
pipeline makes roughly one call per ingest.

---

## Part 3: What the code actually does

Six modules. Here is the honest description of each.

### `plfc/names.py` — canonical team registry

**The problem it solves.** Football sources spell teams inconsistently:

| Source | How it writes Nottingham Forest |
|---|---|
| FBref | `Nott'ham Forest` |
| Football-Data.co.uk | `Nott'm Forest` |
| ClubElo | `Forest` |
| football-data.org | `Nottingham Forest FC` |

Joining on raw strings drops rows **silently**. No exception, no warning — just
a season with 342 matches instead of 380 and a model trained on a hole.

**How it works.** Three layers:

1. **Exact alias lookup** — a registry mapping every observed spelling to one
   canonical id.
2. **Normalised lookup** — strips accents, punctuation and club suffixes, then
   casefolds. Catches `NOTTM FOREST` and `  Arsenal FC  `.
3. **Fuzzy match — which only suggests, never decides.** It raises an error
   containing the closest known names so a human adds the alias deliberately.

Layer 3's refusal is the important design decision. In testing, `Barcelona`
fuzzy-matched `Arsenal` above a 0.6 similarity cutoff. An auto-resolver would
have accepted that silently and corrupted the dataset. **Fuzzy matching
proposes; a human disposes.**

### `plfc/ingest.py` — pull, validate, cache

Fetches from all three sources, canonicalises names, validates, writes parquet.

**The validation invariants** — every completed season must have:

- exactly 380 matches (20 teams × 38 matchweeks)
- exactly 20 teams
- no team playing itself (the signature of a name-collapse bug)
- no duplicate fixtures
- no nulls in key columns

Validation runs *before* the parquet is overwritten, so a failed run leaves the
last good dataset intact.

**Why caching matters.** After a successful ingest, everything downstream reads
local parquet. Modeling, backtesting and the dashboard never touch the network.
If a source breaks in October, your dataset is unaffected.

### `plfc/footballdata.py` — API client

Throttled to 6.5 seconds between calls (10/min with headroom). Honours HTTP 429
`Retry-After` headers rather than guessing. Gives clear errors when a resource
is behind the paywall instead of a confusing parse failure.

`probe_team_names()` returns raw API names *without* resolving them — that is
what `scripts/check_api.py` uses to tell you which aliases are missing before
you spend minutes on an ingest.

### `plfc/model.py` — Dixon-Coles

**The model.** Each team has an attack strength `a` and defence strength `d`.
For a match between home team i and away team j:

```
home goals ~ Poisson(exp(a_i - d_j + γ))
away goals ~ Poisson(exp(a_j - d_i))
```

where γ is home advantage. Goals are independent Poisson draws — **except for
low scores.**

**The τ correction.** Dixon & Coles (1997) observed that independent Poisson
underestimates 0-0 and 1-1 draws and overestimates 1-0 and 0-1. Football has
game-state effects: a team level at 0-0 late plays differently than
independence implies. A single parameter ρ adjusts those four cells.

*This is the first thing an interviewer will probe. Be able to say why goals
are approximately Poisson and where that assumption breaks.*

**Time decay.** Team strength drifts — the 2016 Leicester side and the 2026
Leicester side share a name and nothing else. Each match is weighted
`exp(-ξ × age_days)` in the likelihood. ξ is tuned by backtest, not assumed.

**Shrinkage for promoted teams.** Three teams arrive each season with zero
Premier League history. Excluding them would make ~15% of fixtures
unpredictable *and* bias evaluation toward easy matches. Instead, every team's
strength is shrunk toward a prior, with shrinkage inversely proportional to
time-weighted matches observed. Established teams barely shrink; promoted sides
sit near the prior until they earn their own estimate.

This is ridge regularisation with a non-zero centre. One-sentence version for
an interview: *"New teams don't get a free parameter until they've earned it."*

### `plfc/backtest.py` — walk-forward evaluation

**Why not k-fold.** Shuffled cross-validation on time series trains on May to
predict September. Every number it produces is optimistic and meaningless.

**What we do instead.** Refit on everything strictly before date D, predict
matches on D, advance by `step_days`. That is exactly the information a real
forecaster has. `step_days=14` means ~190 separate model fits across the
history.

**Metrics, and why accuracy is the weakest one:**

- **Brier score** — mean squared error of the probability vector. Rewards being
  right *and* appropriately uncertain. Lower is better.
- **Log loss** — punishes confident errors harshly. Sensitive to
  miscalibration.
- **Calibration curve** — of the matches where we said 30%, did roughly 30%
  happen? A model can be accurate and badly calibrated.
- **Accuracy** — reported because people ask, but nearly uninformative here.

### `plfc/ledger.py` — the prediction ledger

Added after the first successful backtest, to answer a question the backtest
cannot: *what did the model actually predict, and what happened?*

**The distinction that matters.** The backtest is **retrospective** — it
reconstructs what the model would have said by refitting on data available at
each past date. Leak-free and honest, but re-runnable with different settings
until the numbers flatter you. Nobody watching can tell whether you swept fifty
configurations and reported the best.

The ledger is **prospective**. Every Wednesday it writes down predictions for
matches not yet played, timestamped, and never edits them. Later runs fill in
what actually happened. A backtest is a claim; an accumulating log is evidence.

**Design decisions:**

- **Append-only.** Re-running on the same day is a no-op, not a rewrite. History
  is never quietly revised.
- **Multiple forecasts per match are kept.** One made three weeks out, one made
  two days out. Scoring uses the latest strictly before kickoff — what a real
  forecaster would have had.
- **Model parameters stored per row.** Retune ξ later and old predictions stay
  attributable to the model that made them.
- **14-day horizon only.** Forecasting May in September is noise.

**The honest limitation: it cannot be backfilled.** The log starts from the
first run. That is precisely why it is credible.

### `app.py` — Streamlit dashboard

Five tabs: this weekend's fixtures, any matchup, live track record, team
ratings, and an honesty tab that runs the backtest and lists limitations. Reads
only cached parquet, so it stays up even if a source is down.

---

## Part 4: Problems hit and fixed

Worth knowing in order — these are good interview material, because debugging
is the part most portfolio projects hide.

### 1. Duplicate alias caught by the test suite

The registry had a redundant `Stoke` entry claiming an alias that `Stoke City`
also owned. `test_no_alias_maps_to_two_teams` failed on its first run. That
ambiguity is exactly what corrupts a join. Removed the duplicate.

**The integrity test earned its keep immediately.**

### 2. `ModuleNotFoundError: No module named 'plfc'`

Running a script inside `scripts/` puts that folder on Python's path, not the
repo root. Fixed by inserting the project root into `sys.path` at the top of
`check_api.py`.

### 3. The COVID season bug — the interesting one

**Symptom:**

```
season 2019: 314 played matches (expected 380)
season 2020: 23 teams (expected 20)
season 2020: 446 played matches (expected 380)
```

**Diagnosis.** `current_season()` assigned seasons by date: month ≥ 7 means a
new campaign. That is correct in normal years. But the 2019/20 season was
suspended in March 2020 and resumed behind closed doors in June, finishing on
**26 July 2020**. Those final matchweeks were filed under season 2020 instead
of 2019.

The arithmetic confirms it exactly: 2019 short by 66 matches, 2020 holding
446 (380 + 66) and 23 teams — the extra three being clubs relegated in 2019/20
whose last matches fell in July.

**Fix.** Stop deriving the season from the date. `soccerdata` already returns
the authoritative season key (`1920`), so `_season_code_to_year()` converts
that to a start year. The source knows which campaign a match belongs to; a
date heuristic only guesses.

**Why this is a good story.** The validator did not just catch an error — the
*shape* of the error identified the cause. Fewer than 20 teams means a dropped
team from a name join. More than 20 means two campaigns merged, which is a
season-assignment problem. We then improved the error message to say so.

### 4. `NameError: name 'raw' is not defined`

The season fix was applied with a text pattern that matched in two places,
clobbering `fetch_schedule` — which has no `raw` variable. A self-inflicted
wound from careless patching. Fixed.

### 5. Stale-model cache bug

`_model()` in the Streamlit app used `@st.cache_resource` keyed only on
`(xi, shrinkage)`, with no TTL. Those values do not change when new data
arrives — so the model would fit once and stay cached forever. The ratings
table would show fresh data while predictions came from a stale fit.

Silent, and exactly the class of bug the whole project is built to avoid.

**Fix.** Add `last_refresh()` as a third argument to the cache key. It is never
read in the function body; it exists purely so a data refresh invalidates the
cache. In deployment the bug partly self-corrects (a redeploy restarts the
process and clears caches), but it bites hard locally.

### 6. ClubElo 502 Bad Gateway

Their server was down. Retried five times, gave up, logged a warning and
continued. **Not our bug, and by design not fatal** — Elo is an optional
baseline, not a model input.

---

## Part 5: The results

### Verification on synthetic data (before real data)

Generated seasons from *known* team strengths, then checked whether the fitter
recovered them:

- Attack strength correlation with truth: **r = 0.967**
- Home advantage: fitted **0.307** against a true **0.260**
- Probabilities sum to 1; cold-start for unseen teams works

**Important caveat.** Synthetic data was generated from the same Poisson
process the model assumes, so good performance there is partly circular. This
validated the *machinery*, not real-world skill.

### The real backtest

Dataset: **3,440 matches across 10 seasons** (nine complete seasons at 380
each, plus 20 matches of the young 2026/27 campaign). Walk-forward, 14-day
refit cadence, **2,680 out-of-sample predictions**.

| Model | Brier | Log loss | Accuracy |
|---|---|---|---|
| Base rates only | 0.6469 | 1.0688 | 43.5% |
| **Dixon-Coles** | **0.5853** | **0.9842** | **52.6%** |
| Bookmaker closing line | 0.5749 | 0.9689 | 54.5% |

**How to read this — and the number to lead with.**

Do not lead with 52.6% accuracy. Lead with this:

- The market improves on base rates by **0.0720** Brier
- Your model improves on base rates by **0.0616** Brier
- **0.0616 / 0.0720 = 86%**

**The model captures 86% of the bookmaker's edge over base rates, using only
historical goals — no injuries, no lineups, no transfers, no team news, and no
odds during training.**

That claim is defensible because you can show exactly how it was measured. The
residual 14% is roughly where you would expect it: bookmakers price in team
news that a goals-only model structurally cannot see.

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

Excellent through the middle, where the sample is real. When the model says
35%, it happens 34% of the time.

Two honest notes:

- The 0.5–0.6 bin runs slightly hot (predicts 54.6%, observes 51.0%) — mild
  overconfidence on near-coin-flips.
- The top bin looks bad at 0.925 vs 0.846, but **n = 13**. That is noise, not a
  finding. Say so rather than over-interpreting it.

---

## Part 6: What is done, and what is not

**Done and verified:**

- Ingestion running against live sources, 3,440 matches validated
- Name resolution across three sources converging on identical canonical ids
- Model fitted, walk-forward backtested, benchmarked against the market
- Test suite passing, network-free
- Calibration measured

**Also done since:**

- Fixtures ingesting: **356 upcoming fixtures** cached
- Streamlit app rendering correctly, all five tabs working
- Prediction ledger built, tested, and running in production
- Pushed to GitHub; CI passing; weekly Action verified end to end
- Deployed on Streamlit Community Cloud
- **20 predictions currently sealed and pending results**

**Not yet done:**

- Decay rate ξ is still the assumed 0.0018 (~1-year half-life), not tuned
  — **see the warning in the appendix before tuning it**
- ClubElo still enabled in `refresh()` despite being down and unused; setting
  `include_elo=False` would cut most of the weekly run time
- Ledger has no settled predictions yet — the first settle happens on the next
  Wednesday run
- Betting log with closing-line-value tracking (discussed, not built)

---

---

## Part 7: Deployment

Getting it running unattended surfaced four more problems worth recording.

### 1. Streamlit could not see the repository

"This repository does not exist" — a GitHub authorisation scope issue on
Streamlit's side, not a missing repo. Resolved by granting Streamlit repo
access.

### 2. Action failed with 403: write access not granted

The workflow declares `permissions: contents: write`, but a repo-level default
of read-only overrides it. Fixed in Settings → Actions → General → Workflow
permissions → Read and write.

Everything before the push had succeeded — ingest validated 3,440 matches in
the cloud, the ledger updated — only the final commit was rejected.

### 3. Health check failure: connection refused on 8501

`port = 8502` had been added to `.streamlit/config.toml` to avoid a local
conflict with another project. That file is committed, so Streamlit Cloud read
it too: the app bound to 8502 while the platform health-checked 8501.

**The general lesson.** Environment-specific settings belong in flags or
environment variables, never in files you deploy. Locally the right answer was
`streamlit run app.py --server.port 8502`.

### 4. ImportError: matplotlib

`Styler.background_gradient` requires matplotlib, which was not installed on
the deployment. It had never failed locally because the full `requirements.txt`
environment pulled matplotlib in transitively.

Fixed by dropping the gradient rather than adding the dependency — the heatmap
was decorative and the numbers carry the information.

**Worth internalising:** a transitive dependency you did not know you had is
invisible until you deploy somewhere clean.

### The finished pipeline

Every Wednesday at 06:00 UTC, entirely in the cloud:

1. GitHub Action pulls fresh results from Football-Data.co.uk and fixtures from
   football-data.org
2. Validates structural invariants; a failure leaves the last good dataset
   intact
3. Settles the previous week's ledger predictions against actual results
4. Logs new forecasts for the coming fortnight
5. Commits the updated parquet back to the repo
6. Streamlit Cloud sees the push and redeploys, which also clears its caches

No local machine is involved at any point.

---

## Part 8: Multi-league expansion (Phase 1)

The Premier League project became the foundation for a wider question:
whether a shortlist of elite clubs across Europe's top five leagues could
support a betting strategy. That required four more leagues.

### The validator had to become per-league

The original invariant -- 380 matches, 20 teams -- is Premier League
specific and wrong for two of the five:

| League | Teams | Matches |
|---|---|---|
| Premier League, La Liga, Serie A | 20 | 380 |
| Bundesliga | 18 | 306 |
| Ligue 1 (through 2022-23) | 20 | 380 |
| **Ligue 1 (2023-24 onward)** | **18** | **306** |

Ligue 1 shrank mid-dataset. A single fixed invariant would have flagged
every season on one side of that line. `plfc/leagues.py` now holds
season-dependent config, because **a validator that cries wolf gets
switched off, and a switched-off validator is worse than none.**

### Names got harder

`plfc/names_intl.py` adds ~140 clubs. Football-Data.co.uk abbreviates
aggressively and inconsistently:

    "Ath Madrid"    Atletico Madrid
    "Ath Bilbao"    Athletic Bilbao        <- one token apart, different club
    "Espanol"       Espanyol               <- their spelling, not a typo
    "M'gladbach"    Borussia Monchengladbach
    "Ein Frankfurt" Eintracht Frankfurt
    "Paris SG"      Paris Saint-Germain
    "Inter"         Internazionale
    "Milan"         AC Milan               <- and "Inter" is NOT Milan

The dangerous pairs are the point. `Ath Madrid`/`Ath Bilbao`,
`Inter`/`Milan`, `Real Sociedad`/`Real Madrid`, and the two Borussias are
each one token apart and genuinely distinct clubs. A fuzzy matcher would
merge them and corrupt a decade of data with no error raised. There are
now parametrised tests asserting each pair resolves differently.

### Models fit per league, never pooled

Dixon-Coles estimates strength from a graph of who played whom. Within a
league every team meets every other twice -- dense, well identified.
Across leagues, clubs meet only in European competition: a handful of
matches connecting hundreds of teams. That graph is far too sparse to put
La Liga and Bundesliga on a common scale, so five leagues means five
models.

---

## Part 9: Removing the scraping dependency entirely

### The symptom

`pytest` began taking minutes to start, then hanging outright. The
traceback showed pytest importing **seleniumbase**, which imports
selenium, which imports websocket.

### The cause

`soccerdata` depends on seleniumbase -- a full browser automation stack --
and seleniumbase registers itself as a pytest plugin. pytest auto-loads
every installed plugin at startup, so every test run imported a browser
stack the tests never touch.

It cost three ways: glacial test startup, ~200MB on a Streamlit deploy
that only reads parquet, and a scraping dependency for data that is not
scraped.

### The fix

Football-Data.co.uk publishes **plain CSV at stable URLs**:

    https://www.football-data.co.uk/mmz4281/{season}/{div}.csv

`plfc/fdcouk.py` downloads them with `requests` and parses with pandas.
No browser, no selenium, no scraper. ClubElo was dropped at the same time
-- it fed nothing into the model and its server had been down for hours.

**There is now no scraping anywhere in this project.** Two regression
tests assert `soccerdata`, `selenium` and `seleniumbase` appear in neither
`sys.modules` nor `requirements.txt`.

### Three data traps this surfaced

**1. Day-first dates.** Football-Data.co.uk writes `dd/mm/yy` in older
files and `dd/mm/yyyy` in newer ones. Both are day-first, which pandas
does NOT assume: `03/04/2024` silently parses as 4 March instead of
3 April. Every date parses with `dayfirst=True`, and a range check raises
if parsed dates fall outside the season they claim to belong to.

**2. HTML served with a 200.** A struggling server often returns an error
page with a success status. Without a guard, pandas would parse
`<!DOCTYPE html>` as data and produce garbage rows that pass validation.
The downloader now sniffs the first bytes and rejects anything HTML-shaped.
This mattered within days -- see below.

**3. Outages.** Downloads retry on 5xx with exponential backoff, and fall
back to a stale cached copy rather than failing outright. Last week's data
beats no data on an unattended Wednesday run. 404 is not retried: a
missing season will still be missing in eight seconds.

### The outage

Immediately after this work, football-data.co.uk began returning HTTP 503
across every league and season -- confirmed via curl, browser, and a
second device. Their nginx error page is exactly the HTML-with-200 case
the parser now guards against.

**The multi-league ingest has therefore not yet run.** Phase 1 is built
and tested (100 tests passing) but has only been exercised against
synthetic data for the four new leagues. The existing Premier League
parquet is unaffected, so the app and everything downstream still works.

---

## Part 10: The betting strategy investigation

A side thread that produced real findings and shaped Phase 2.

### The original proposal

Bet elite clubs (Real Madrid, Barcelona, Bayern, PSG, Arsenal, Man City,
Inter) as singles each week, parlay the same teams together, and add one
"upset" pick. Small stakes, many combinations, aiming to profit whenever
only one or two teams slip.

### What the maths said

**Parlays multiply the bookmaker's margin, they do not average it.** At a
typical 5% overround, four legs compound to ~22%.

**They are the opposite of risk reduction.** Four legs at 75% each: the
parlay loses 68.4% of the time. The same four picks as singles lose
everything only 0.39% of the time. Identical selections; the parlay turns
a 0.4% chance of total loss into 68%.

**Correlated exposure.** Betting Barcelona as a single *and* in the parlay
means one Barcelona defeat kills both. That is the same result purchased
twice, not diversification.

**Simulating the exact proposal** (3 singles + parlay of those three +
one upset, over a 38-week season): median −10 units, profitable 24% of
the time. Dropping just the parlay raised that to 34% -- the component
added *for* risk reduction was the most expensive part.

**Stake size does not change EV.** Betting $5 instead of $50 changes the
absolute loss, not the −15.3% expected return. Smaller stakes make a
strategy affordable, not profitable.

**Structure cannot change the mean.** The EV of a set of parlays is the
sum of each parlay's EV. Combination structure changes variance only. No
arrangement of negative-EV legs becomes positive.

### Where the argument turned

Two corrections, both from the user, both right:

**1. Fixture filtering was missing from the simulations.** Betting every
shortlisted team every week -- including Barcelona away at the Bernabeu --
is not the strategy. Selecting only favourable fixtures lifts the per-leg
win rate from a season average of ~68% toward 85-90%.

**2. Parlays amplify edge in BOTH directions.** Four legs at −5% each give
−18.5%. Four legs at **+5%** each give **+21.6%**. Parlays are not
inherently bad; they are a multiplier. The entire question is the sign of
the legs.

### The resulting position

The structure is sound *conditional on the legs having positive edge*.
That is an empirical question, and Phase 2 exists to answer it.

The honest prior: the Premier League backtest showed the model losing to
the closing line (Brier 0.5853 vs 0.5749). But that was aggregate
performance across all matches, not performance on the narrow subset of
short-priced elite favourites. Published evidence suggests favourites are
systematically better priced than longshots, so this niche may behave
differently from the market as a whole. Worth measuring.

**If the backtest says the edge is not there, the answer is that the
strategy does not work -- not that we need different parlay combinations.**

### Win-rate calibration

A proposed 75% floor for the shortlist would have excluded Arsenal (who
won the league at 68.4%) and Real Madrid (71.1%). Only Barcelona cleared
it. Elite seasons realistically land 65-75%; exceptional ones reach 84%.
The floor was set to 60%, with the top-8 ranking doing the actual
selection.

This surfaced a useful distinction:

- **Season win rate** selects *teams*. Barcelona's 81.6% includes away
  trips to Real Madrid.
- **Per-match probability** selects *bets*. Barcelona at home to a
  bottom-half side is closer to 88-92%.

You never bet a team's whole season. The per-fixture model does the real
work.

### The correct-score finding

A proposed alternative: back the exact scoreline on ~10 matches a week at
+500, needing 2 hits from 10 to profit.

The model answers this directly. Across 200 fixtures the **modal scoreline
carries a median 12.4% probability** (range 7.6% to 23.7%).

| Hits from 10 | Probability | Profit on $100 |
|---|---|---|
| 0 | 26.7% | −$100 |
| 1 | 37.7% | −$40 |
| 2 | 23.9% | +$20 |
| 3+ | 11.7% | +$80 or more |

P(2 or more) = 35.6%. **EV = −25.8%.**

A 12.4% shot is worth **+708**; +500 is offered. Correct-score carries
15-30% overround against ~5% on the 1X2 market -- it is priced that way
precisely because it looks attractive.

One caveat worth testing: the most predictable fixture in the sample sat
at 23.7%, where +500 would be +42% EV. If bookmakers apply a flat fat
margin to all correct-score markets regardless of how concentrated the
true distribution is, edge might exist specifically there. Phase 2 adds
correct-score to the edge buckets to find out.

---

## Part 11: Five leagues, five models

The 503 cleared. Football-Data.co.uk came back, the multi-league ingest ran,
and the name registry survived contact with the live feed: **16,210 matches
across five leagues, ten seasons, with closing odds.**

### The pooling bug that would have been silent

`walk_forward()` had no league parameter. Pointed at the multi-league
parquet it would have fitted a single Dixon-Coles across all five leagues --
exactly the pooling `leagues.py` argues at length must never happen -- and
it would have done so with no error and no obvious symptom. The ratings
table would have looked plausible. Every number downstream would have been
nonsense.

The fix is a guard that refuses rather than guesses:

```python
elif len(present) > 1:
    raise ValueError(
        "matches spans multiple leagues but no league was specified. "
        "Dixon-Coles strengths are only identified within a league."
    )
```

Same philosophy as the name resolver and the validator: when the code
cannot know, it stops.

### Warm starting

Five leagues × ~9 test seasons × biweekly refits is a few thousand model
fits. The optimiser was starting from a zero vector every single time,
throwing away a very good guess -- consecutive fits differ by two weeks of
matches, so the previous solution is nearly the answer.

`fit(..., warm_start=previous_model)` seeds `x0` from the prior fit.
Verified on synthetic data: max probability difference 0.0001. Same
objective, same optimum, far fewer iterations. Five leagues became a coffee
break instead of an overnight job.

### The per-league scorecard

| League | Model | Base rates | Market | Share of market edge |
|---|---|---|---|---|
| Premier League | 0.5853 | 0.6469 | 0.5749 | 85.5% |
| La Liga | 0.5875 | 0.6454 | 0.5773 | 85.0% |
| Serie A | 0.5838 | 0.6562 | 0.5725 | 86.5% |
| Bundesliga | 0.5931 | 0.6494 | 0.5772 | 78.1% |
| Ligue 1 | 0.6019 | 0.6501 | 0.5888 | 78.6% |

12,413 walk-forward predictions. **The model loses to the closing line in
all five leagues**, capturing 78-87% of the bookmaker's improvement over
base rates using nothing but goals and dates. Remarkably consistent, and
the right result: a goals-only model beating five independent markets
would have meant a bug.

The share-of-market-edge column is blanked when the market barely clears
base rates, because the ratio explodes and means nothing there.

---

## Part 12: The calibration layer that was built and rejected

Refitting weekly re-estimates team strength from newer goals. That is
learning about **teams**. Nothing in the pipeline ever noticed whether the
model's own 55% calls were coming in at 55%. That is a calibration layer,
and it was the obvious Phase 4 item.

Two methods, both deliberately tiny: temperature scaling (one parameter,
`p ∝ p^(1/T)`) and vector scaling (six). Anything richer -- isotonic,
per-bin lookups -- would fit noise in a few thousand matches.

The evaluation had to be walk-forward too. Fitting the calibrator on all
predictions and reporting improved metrics on those same predictions is
circular and always looks like it worked.

**Result on 2,251 out-of-sample Premier League predictions:**

| | Brier | log loss | ECE |
|---|---|---|---|
| Raw model | 0.5855 | 0.9844 | 0.0107 |
| Calibrated | 0.5859 | 0.9846 | 0.0088 |
| Change | +0.0004 | +0.0003 | −0.0019 |

Temperature came back at 1.09, close enough to 1.0 that it was mostly
fitting sampling noise in the training window and applying it as a
correction to the next block. ECE improved slightly; Brier got marginally
worse.

**Not deployed.** The model was already well calibrated -- the calibration
table tracks the diagonal -- so there was nothing for the layer to correct,
and an unnecessary parameter can only cost you.

`plfc/calibration.py` stays in the repo, imported by nothing. "I built a
calibration layer, tested it walk-forward, and it did not earn its place"
is a stronger thing to be able to say than shipping something that makes
the numbers slightly worse.

Sanity-checked in both directions first: on a synthetic model distorted to
be overconfident by 1.6×, the calibrator recovered T = 1.619 and improved
Brier by 0.0150. It works. There was just nothing here to fix.

---

## Part 13: Two bugs in the ledger

### Keys that could never match

`reconcile()` matches predictions to results on
`league:season:home:away`. Rows logged before the multi-league work carried
`league = NaN`, producing keys like `nan:2026:Ipswich Town:Liverpool` while
the results table produced `ENG-Premier League:2026:Ipswich Town:Liverpool`.

Those predictions could never settle. Not "would settle late" -- never, no
matter how many times the Wednesday job ran. The first ten sealed forecasts
sat pending indefinitely and it looked like data loss.

Fixed with a one-off backfill that infers each row's league by looking the
clubs up in the results table rather than assuming Premier League, then
rebuilds every `match_key`. It also collapsed a set of duplicate rows: ten
fixtures had been logged twice, once either side of the change, and once
the keys matched `latest_before_kickoff()` folded them correctly.

### A row that never existed

`latest_before_kickoff()` used `groupby().last()`. Pandas' `GroupBy.last()`
returns the last **non-null value per column**, not the last row. With two
forecasts for one match it was stitching together a row that never existed,
taking the later prediction's probabilities and silently back-filling any
null from the earlier one.

Benign at the time, and a live bug the moment prediction-time odds get
added for closing-line-value work, since those can legitimately be null on
the later row. Now `.tail(1)`.

---

## Part 14: Phase 2 -- the betting investigation

The question: does a rolling shortlist of elite clubs, filtered to
favourable fixtures, produce legs with positive expected value? The honest
prior was no, because the model lost to the closing line in aggregate. But
aggregate performance is dominated by mid-table coin flips; short-priced
elite favourites are a different population and had not been measured.

**The answer is no, established five separate ways.**

### The rolling shortlist

Picking clubs by their win rate over six years and then counting how often
those clubs won over those same six years is circular -- they were selected
for winning during the window being measured. So the tier rebuilds every
season from earlier seasons only: 3-season weighted win rate, weights 3:2:1
toward the most recent.

The tier table across seven seasons settled the threshold question:

| Bar | Clubs per season |
|---|---|
| 0.55 | 11-16 |
| 0.60 | 7-10 |
| **0.65** | **5-7** |
| 0.70 | 1-5 |
| 0.75 | 0-4 |

0.65 gives a workable parlay every year. 0.70 collapses to Manchester City
alone in 2023; 0.75 is empty in four of seven years.

**The churn is the proof it works.** Juventus qualifies for 2020-21 then
disappears. Liverpool for 2020-22. Napoli appears in 2023 straight off
their title and vanishes. Only PSG and Bayern clear 0.65 in every year.

Napoli is the single best illustration: a hindsight-built list would never
include them, and they went on to appear in **four of the five worst trio
combinations**. A forecaster standing in August 2023 would absolutely have
backed them. An honest backtest has to eat that loss.

### The week anchor -- a measurement error worth recording

Weeks were originally anchored to Monday. English fixtures run Friday to
Monday, so a Monday night game landed in the *following* week's bucket,
splitting a single round across two.

Measured on real data across six elite clubs and ten seasons:

| Anchor | All six playing | Five or more |
|---|---|---|
| Monday | 55.9% | 81.4% |
| **Tuesday** | **63.2%** | **83.4%** |

Seven percentage points of pure measurement error. Tuesday puts Fri/Sat/
Sun/Mon in one bucket and gives midweek rounds their own.

The same measurement killed an earlier design decision. `require_full=True`
-- counting only weeks where every shortlisted club played -- discards 37%
of weeks, and discards them **non-randomly**: the discarded weeks are cup
weekends and midweek rounds, exactly when elite clubs rotate and drop
points. Filtering them out would have flattered the strategy by removing
its hardest weeks. The default changed to counting the legs that actually
existed.

### ⚠️ The metric that lied

**This is the most important thing in Phase 2.**

The first version computed edge as `hit_rate − 1/mean(odds)` and reported
**+3.5% edge** on the shortlist strategy. Positive edge, negative ROI, in
every single row -- a contradiction that sat in the output for several
iterations before it got questioned.

The formula is wrong. By Jensen's inequality `1/mean(odds) ≠ mean(1/odds)`,
and with combined parlay odds spanning 4 to 15 the gap is large enough to
flip the sign.

Demonstrated on a synthetic book built to lose 12% by construction:

| | |
|---|---|
| hit rate | 0.1325 |
| `1/mean(odds)` → "edge" | **+0.0184** (wrong, looks profitable) |
| `mean(1/odds)` → edge | −0.0076 (correct sign) |
| realised ROI | **−0.1436** (the truth) |

It also hid the mechanism. Wins arrived at odds of 6.46 against an all-week
average of 8.76. **You win the cheap weeks and lose the expensive ones** --
short prices mean easy fixtures -- and averaging odds then inverting erases
that entirely.

Every figure now settles each week at its own price and sums the money.
`mean_odds_when_won` is reported alongside `mean_combined_odds` in every
table so the pattern stays visible.

### Result 1: parlays lose at every leg count

Five-team shortlist, 318 weeks, real closing odds:

| Legs | Weeks | Hit rate | Mean odds | Odds when won | ROI |
|---|---|---|---|---|---|
| 2 | 8 | 50.0% | 2.16 | 1.69 | −15.3% |
| 3 | 24 | 33.3% | 3.53 | 2.94 | −2.1% |
| 4 | 44 | 18.2% | 5.62 | 4.29 | −21.9% |
| 5 | 242 | 14.5% | 8.07 | 5.79 | −16.3% |
| **all** | **318** | 17.3% | 7.24 | 4.86 | **−16.0%** |

`mean_odds_when_won` is below `mean_combined_odds` in every row without
exception.

Season by season with no hindsight, every season loses: −83%, −66%, −52%,
−50%, −32%, −19%. The trend improves as the shortlist tightened and odds
shortened, but never crosses zero.

### Result 2: three legs is better and still loses

Three-leg parlays, every combination, rolling tier:

| Bar | Trios | Trio-weeks | All three won | Pooled ROI | Profitable |
|---|---|---|---|---|---|
| 0.60 | 252 | 17,101 | 26.7% | **−10.1%** | 63 / 252 |
| 0.65 | 72 | 3,584 | 27.9% | **−19.8%** | 13 / 72 |

Three legs compound the margin three times instead of five, so −10% instead
of −16%. Real improvement, wrong side of zero.

**Every large winner is a small sample.** Best trio at the 0.60 bar: one
season, 28 weeks, +87.6%. The trio with the most weight -- Barcelona +
Bayern + PSG, five seasons, 114 weeks -- returns +8.3%. That is regression,
not skill.

**The tighter bar loses more.** Better teams, worse returns: shorter prices
leave less room to absorb an upset while the compounded margin stays.
"Pick only the very best teams" is actively the wrong instinct.

### Result 3: fixture filtering does nothing

1,577 shortlisted legs with odds. Baseline: 67.3% hit rate at mean odds
1.53, **−4.05% ROI** -- approximately the bookmaker margin on short-priced
favourites.

Filtering to fixtures the model thinks are easy:

| Min model probability | Legs kept | Hit rate | ROI |
|---|---|---|---|
| none | 1,577 | 67.3% | −4.1% |
| 0.60 | 1,093 | 74.6% | −3.5% |
| 0.70 | 688 | 79.8% | −1.8% |
| 0.80 | 253 | 84.2% | −2.7% |

Hit rate climbs 17 points. **ROI never turns positive at any threshold.**
Selecting easy fixtures selects short prices; the market had already priced
the ease in.

### Result 4: the model's disagreement is anti-predictive

The sharpest finding in the project. Filtering to legs where the model
thinks the market is most wrong:

| Min expected value | Legs kept | ROI |
|---|---|---|
| none | 1,577 | −4.1% |
| ≥ 0.00 | 419 | −10.8% |
| ≥ 0.05 | 211 | −12.2% |
| ≥ 0.10 | 124 | −14.9% |
| ≥ 0.20 | 49 | −25.0% |

Monotonically worse. **The legs where the model most disagreed with the
market were the legs where the model was most wrong.**

A model with no information would show a flat line. This slopes down, which
says something specific: the market holds information the model cannot see
-- team news, injuries, rotation, weight of money -- and the fixtures where
they diverge most are precisely where that missing information matters
most. Disagreement was not insight. It was the size of the blind spot.

### Result 5: no price band is profitable

If expected value varied by something observable before the bet, you would
bet only that band. Combined price is known before kickoff, so it is
legitimately usable. Trio-weeks bucketed by combined odds, 0.60 bar:

| Odds band | Weeks | Won | Needed | ROI |
|---|---|---|---|---|
| < 2.5 | 3,304 | 43.5% | 47.9% | −8.4% |
| 2.5-3.0 | 2,671 | 35.0% | 36.4% | −3.9% |
| 3.0-3.5 | 2,376 | 29.5% | 30.9% | −4.8% |
| 3.5-4.0 | 1,885 | 25.7% | 26.8% | −3.8% |
| 4.0-5.0 | 2,428 | 21.3% | 22.5% | −5.7% |
| 5.0-6.0 | 1,354 | 15.7% | 18.4% | −14.7% |
| 6.0-8.0 | 1,426 | 12.3% | 14.6% | −16.0% |
| 8.0+ | 1,657 | 5.9% | 9.2% | −36.2% |

**This table is the whole answer.** Across 17,101 weeks, what the trios
actually did tracks what the price implied, in every band. When the market
said 30.9%, they won 29.5%. When it said 9.2%, they won 5.9%. The
bookmaker's estimate was accurate everywhere; the persistent gap underneath
is the margin, and no band clears it.

### Result 6: staking cannot help, by construction

Total return is `Σ (stake_i × EV_per_dollar_i)`. If every term is negative,
no set of positive stakes makes the sum positive. Staking changes variance
and speed, never sign.

Simulated over 3,584 weeks from $10,000, on a series *milder* than the one
measured (−4.7% vs −19.8%):

| Plan | Final |
|---|---|
| Flat $100 | $0 busted |
| 2% of bankroll | $53 |
| 10% of bankroll | $1 busted |
| **Kelly, given the true win rate** | **$1 busted** |
| More on short prices | $0 busted |
| More on long prices | $0 busted |
| Martingale | $0 busted |

Kelly is the row that matters. It is the mathematically optimal staking
rule, it was handed the true win probability, and it still went to zero --
because with negative edge Kelly's correct answer is to stake nothing.

Proportional staking survives longest only because it can never bet what
you do not have. It approaches zero asymptotically instead of hitting it.
That is a slower death, not a profit.

### The arithmetic underneath all of it

The margin on a single elite favourite is about 4%. Stack three on one
slip: 0.96³ ≈ 0.88. The measured pooled ROI of −10% is almost exactly that.
A parlay is not a way to win more; it is a way to pay the margin three
times on one bet.

---

## Part 15: Correct score, and the two routes

### The modal scoreline barely beats a constant guess

12,413 walk-forward predictions:

| | Hit rate | Breakeven odds |
|---|---|---|
| Model's most likely scoreline | **12.48%** | 8.01 |
| Always guessing 1-1 | **12.13%** | 8.24 |

A gap of **0.35 percentage points** -- inside one standard error of zero.
And **57.5% of the model's picks are 1-1**.

There is a flicker of signal: when the model picks 1-1 those land 13.32%,
above the 12.13% unconditional rate, so the grid does choose *which*
matches finish 1-1 slightly better than chance. Far too small to pay a 20%+
overround.

By predicted scoreline:

| Pick | Times | Hit rate | Breakeven |
|---|---|---|---|
| 1-1 | 7,142 | 13.32% | 7.51 |
| 1-0 | 1,581 | 13.16% | 7.60 |
| 2-0 | 1,351 | 10.88% | 9.19 |
| 2-1 | 429 | **9.32%** | 10.73 |

2-1 is the worst pick and prices like a common scoreline. Whenever the
model says 2-1, that is a bet to skip.

By league, and the spread is large enough to be real:

| League | Hit rate | Breakeven |
|---|---|---|
| Serie A | 14.03% | 7.13 |
| La Liga | 13.90% | 7.20 |
| Bundesliga | 11.71% | 8.54 |
| Ligue 1 | 11.25% | 8.89 |
| Premier League | 11.16% | 8.96 |

Lower-scoring, more predictable leagues concentrate probability into fewer
cells. If any version of correct-score betting is viable it is Italy and
Spain only, at 7.50+, and it is still thin.

**The weekly plan, costed.** 10 bets × $10 × 5 leagues = $500/week. At the
measured 12.48% you need 8.01 on every bet to break even. At 7.00 that is
−$63/week, roughly **−$2,400 a season**. The 8.00 row breaks even exactly
and assumes never taking a bad price.

Common scorelines price 6.00-9.00 and are the most efficiently priced part
of the correct-score market -- heaviest volume, priced by weight of
opinion. The model always bets into the tightest corner of the worst market
in football.

### Two routes, and why they disagree

The model holds two opinions about every match:

- **argmax** -- the largest of home/draw/away
- **scoreline** -- the result implied by the most likely scoreline

They disagree on **59.5%** of matches.

| | argmax | scoreline | actual |
|---|---|---|---|
| predicts home | 63.1% | 28.9% | 43.1% |
| predicts draw | **0.5%** | **60.0%** | 25.3% |
| predicts away | 36.4% | 11.1% | 31.6% |

| Accuracy | argmax **0.5233** | scoreline 0.4165 |
|---|---|---|

argmax wins by 10.7 points, consistently across all five leagues, far
beyond noise.

**Draw recall is the diagnostic.** argmax catches 0.6% of the 3,137 draws
that happened -- 67 draw calls out of 12,413. Not because the model is
blind to draws; it assigns them 25-29% routinely. A 29% draw simply never
outranks a 40% home win. The scoreline route catches 65.8% of draws and is
right on only 27.7% of the ones it calls.

**But the precision table complicates the story.** When the scoreline route
names a *winner* it is far more precise than argmax: 64.0% vs 53.9% on home
calls, 58.9% vs 50.0% on away. That looked like a hybrid rule waiting to
happen.

It was not. A hybrid -- take the scoreline when it names a winner, fall
back to argmax on draws -- came out **byte-identical to argmax**: same
accuracy, same 67 draw calls. Whenever the scoreline names a winner, argmax
already agrees. If the modal score is 2-0 the lambdas strongly favour home
and both routes say home. The hybrid had nothing to fix.

Which also explains the precision numbers. The scoreline route only names a
winner on lopsided matches -- it is more precise on a subset selected for
being easy. The same trap as filtering legs by model probability.

Draw-threshold rules were tested too (call a draw whenever `p_draw ≥ t`).
Every one loses, monotonically: 0.5252 → 0.5194 → 0.5149 → 0.5118 → 0.5023
as the bar drops. Out-of-sample on a time split, **no rule beats argmax**.

The draw blindness remains a genuine limitation worth stating rather than
hiding. The model is not wrong about draws; it just can never surface one
as a headline call.

---

## Part 16: What the app now consists of

### Structure

```
app.py                      league selector + five tabs
pages/1_Case_study.py       the Phase 2 write-up with charts
build_case_study.py         computes every case-study figure once -> JSON
plfc/
  ingest.py                 pull, canonicalise, validate, cache
  leagues.py                per-league structural config
  names.py                  resolver that refuses to guess
  model.py                  Dixon-Coles + warm start
  backtest.py               walk-forward, league-aware
  multileague.py            fit/backtest all five, cache predictions
  ledger.py                 sealed append-only forecast log
  shortlist.py              rolling elite tier, no hindsight
  parlay.py                 week anchoring, leg counting, trio counter
  legfilter.py              fixture-aware leg selection + threshold sweeps
  calibration.py            built, tested, not deployed
analysis/                   one-off runner scripts, kept as evidence
```

### The league selector

Sidebar radio across all five leagues. The model cache is keyed on league
as well as on `last_refresh()` -- without that, switching leagues would
show a different league's model, and a data refresh would update the
ratings table while predictions came from a stale fit.

Models are fitted lazily on page load, cached per league for the life of
the process. Nothing is serialised to disk; only the data is in the repo.

### The weekly scorecard

Three counters, because the model has more than one opinion:

- **Result correct** -- the highest-probability outcome
- **Scoreline's result correct** -- win/draw/loss read off the most likely
  scoreline (2-0 counts as a home win even if it finished 2-1)
- **Exact score correct** -- the scoreline itself

The first sealed week showed 3 / 8 / 2 on ten matches. The gap between 3
and 8 is entirely the draw structure described in Part 15: five of those
ten matches were drawn, argmax said home_win eight times, the scoreline
said 1-1 six times.

Week-by-week table, plus four expanders: every settled prediction, a
week-by-week breakdown, misses sorted by the probability assigned to what
actually happened (a miss at 30% is the model working; a miss at 5% is
worth looking at), and the matches where the two routes disagreed.

### Other tabs

Fixtures with cold-start warnings, arbitrary matchup with the full
scoreline grid, live track record against the sealed ledger, team ratings
with an explicit note that strengths are not comparable across leagues, and
the walk-forward backtest with calibration curve.

### The case study page

Five sections, four charts, built from a cached JSON so the page computes
nothing and the numbers cannot drift between visits:

1. Model vs base rates vs closing line across five leagues
2. The price-bucket chart -- what trios won against what they needed to win
3. Expected-value filter ROI sliding from −4% to −25%
4. Correct score against the always-1-1 baseline, per league
5. The staking simulation table

Two things sit in expanders rather than the main flow, deliberately: the
Napoli inclusion as proof the rolling shortlist works, and the Jensen error
that made the first version of the analysis report a profit.

---

## Part 17: Where things stand

**Complete and running:**

- Five league models, fitted and backtested separately, 16,210 matches
- Weekly scorecard with three counters, sealed prospective ledger
- Phase 2 investigation complete: the edge does not exist, five ways
- Case study page deployed
- GitHub Action refreshing and settling every Wednesday

**Built, tested, deliberately not deployed:**

- Calibration layer (`plfc/calibration.py`) -- no improvement to make
- `suggest_legs()` in `legfilter.py` -- works, but no threshold produces
  positive returns historically, so live suggestions would be a
  nicer-looking way to lose money

**Not built:**

- Player-level markets (Kane scored and Bayern won). Blocked on data:
  Football-Data.co.uk has no goalscorers and football-data.org's free tier
  has no match stats. Needs a new feed.
- Over/under 2.5 and BTTS evaluation. The predictions are now recorded in
  both the backtest and the ledger and `goals_market_scorecard()` exists,
  but it has not been run. **This is the cheapest untested idea in the
  project**: a different market rather than the same one sliced again,
  two-way so the margin is thinner, with real closing odds in the data.
- Champions League. No free odds anywhere.
- De-vigging beyond proportional normalisation. Shin and log-odds methods
  were specified and not built, because the EV filter result made the
  question moot -- the model has no edge to recover more precisely.

### The honest summary

The project asked whether a public-data goals model could find an edge in
the elite-favourite market. It cannot, and the investigation established
that from six independent directions: aggregate Brier, team selection,
fixture filtering, model edge, price banding, and staking theory.

The strongest artefacts are not the model. They are the walk-forward
design, the validator that fails loudly, the name resolver that refuses to
guess, the rolling shortlist that keeps Napoli in, the calibration layer
that was tested and rejected, and the edge metric that was found to be
lying and thrown out.

Most people running this analysis stop at the version that shows +3.5%.

---

## Appendix: how to talk about this

**Lead with methodology, not accuracy.** The strongest claims are: diagnosing
target leakage in existing work, building a resolver that refuses to guess,
validation that fails loudly, walk-forward evaluation, and benchmarking against
the market rather than against zero.

**The three things an interviewer will probe:**

1. *Why Dixon-Coles and not XGBoost?* Because it is interpretable, principled,
   and the τ correction addresses a real, documented failure of independent
   Poisson. Throwing a gradient booster at 3,400 rows would fit worse and
   explain nothing.
2. *Why walk-forward and not k-fold?* Because shuffled CV leaks the future.
3. *How do you handle promoted teams?* Shrinkage toward an empirical prior,
   with shrinkage decaying as matches accumulate.

### "Why did it work on the first try — is it overfitted?"

A fair suspicion, and the answer is worth understanding properly.

**First, reframe what happened.** The model did not work spectacularly. It
scored Brier 0.5853 against the market's 0.5749 — it *lost* to the closing
line. 52.6% accuracy against the market's 54.5%. **If it had beaten the
market, that would be the red flag.** Landing just short is the signature of
an honest model.

**Four structural reasons it is hard to overfit here:**

1. **Walk-forward makes leakage impossible by construction.** The model is
   refit ~190 times, each on data strictly before the test date. It never sees
   the match it is predicting. This is not a safeguard bolted on — it is the
   evaluation design.

2. **The model has almost no capacity.** Dixon-Coles fits 2×20 team parameters
   plus home advantage plus ρ — about 42 numbers — on ~3,400 matches. That is
   roughly 80 observations per parameter. Overfitting requires enough capacity
   to memorise noise, and there simply is not any here. This is the opposite
   situation to a neural network with millions of weights.

3. **It is well-specified for the domain.** Goals genuinely are approximately
   Poisson-distributed, and team attack/defence strength is a real, estimable
   quantity. The model is not discovering spurious structure; it is estimating
   a handful of things that actually exist.

4. **Shrinkage actively regularises.** Every team is pulled toward a prior in
   proportion to how little has been observed of them.

**Good calibration is itself evidence against overfitting.** An overfit model
is characteristically overconfident — it would predict 70% and be right 55% of
the time. The calibration table tracks the diagonal closely, which is what a
properly regularised model looks like.

### ⚠️ Where overfitting *could* still creep in — read before running `--tune`

Here is the real risk, and it is about to become live.

ξ = 0.0018 and shrinkage = 8.0 were chosen by judgment *before* seeing any
results. That makes the reported backtest a clean out-of-sample estimate.

`python -m plfc.backtest --tune` sweeps ξ across five values and reports the
Brier for each. **If you then pick the winner and quote its Brier as your
headline number, that figure is optimistically biased** — you have selected a
hyperparameter using the same data you are reporting performance on. It is a
mild form of overfitting, and a sharp interviewer will spot it.

**Two acceptable ways to handle it:**

- *Cheap and honest:* run the sweep, adopt the best ξ, but keep reporting the
  original untuned figures as your headline. Describe the sweep as a
  sensitivity analysis: "performance was stable across half-lives from 6 to 18
  months," which is a genuine finding and immune to this critique.
- *Proper:* hold out the most recent season entirely. Tune ξ on seasons
  2017–2024, then report performance on 2025–26 only, touched once. Costs you
  sample size but the number is clean.

Either is defensible. Quoting the best sweep result as if it were out-of-sample
is not.

**Be honest about the ceiling.** Football outcomes sit near the noise ceiling.
Bookmakers reach 53–55% on three-way results with vastly more information. A
model claiming 70% has a bug. Saying this unprompted signals that you
understand your problem domain.

### Three additions from Phase 2

**"Did you find a profitable strategy?"** No, and say it immediately. Then
explain that the interesting part is *how* it was established: six
independent lines of evidence, one of which required throwing out a metric
that had been reporting a profit. An interviewer who has run this analysis
themselves knows the answer is no. The signal is whether you got there
honestly.

**"What is the thing you are most proud of?"** The Jensen error. The first
version of the edge calculation reported +3.5% on a strategy that loses
16%, and the contradiction sat in the output -- positive edge, negative ROI
in every row -- for several iterations before it was questioned. Finding a
bug in your own favourable result is harder than finding one in an
unfavourable one, because nobody goes looking.

**"Why keep code you do not use?"** `plfc/calibration.py` is imported by
nothing. It exists because the question "does the model need calibrating?"
deserved a measured answer rather than an assumption, and the answer was
no. Deleting it would delete the evidence that the question was asked.

### The one number to lead with

Not accuracy. Not Brier. **78-87% of the bookmaker's edge over base rates,
captured with nothing but goals and dates, across five independent
markets.** It is honest about losing, specific about by how much, and it
frames the model against the right benchmark.
# Football Match Forecaster

A Dixon-Coles goals model for Europe's top five leagues, with walk-forward
evaluation, a sealed prediction ledger, and a published study on whether the
model can beat the bookmaker's closing line.

**It can't.** The study is the point.

🔗 **[Live app](https://pl-forecaster.streamlit.app)** ·
📊 **[Evaluation study](https://pl-forecaster.streamlit.app/case_study)**

---

## What it does

Fits a separate Dixon-Coles bivariate Poisson model to each of the Premier
League, La Liga, Serie A, Bundesliga and Ligue 1 — 16,210 matches across ten
seasons with historical closing odds. Produces outcome probabilities, expected
goals, scoreline distributions and derived markets for upcoming fixtures.

Every forecast is timestamped and sealed before kickoff in an append-only
ledger, then reconciled against the result. Data refreshes weekly via GitHub
Actions.

## Headline result

| League | Model Brier | Base rates | Closing line | Share of market edge |
|---|---|---|---|---|
| Premier League | 0.5853 | 0.6469 | 0.5749 | 85.5% |
| La Liga | 0.5875 | 0.6454 | 0.5773 | 85.0% |
| Serie A | 0.5838 | 0.6562 | 0.5725 | 86.5% |
| Bundesliga | 0.5931 | 0.6494 | 0.5772 | 78.1% |
| Ligue 1 | 0.6019 | 0.6501 | 0.5888 | 78.6% |

12,413 walk-forward predictions. The model captures **78–87% of the
bookmaker's improvement over base rates** using nothing but goals and dates,
and loses to the closing line in every league.

That is the expected result. A goals-only model on public data beating five
independent markets would mean a bug, not a discovery.

## The evaluation study

The app includes a study testing whether a rolling shortlist of elite clubs
produces positive expected value. The answer is no, established six ways:

- **Aggregate** — loses to the closing line in all five leagues
- **Team selection** — three-club combinations return −10% on stake across
  17,101 combination-weeks
- **Fixture filtering** — restricting to easy fixtures lifts the hit rate from
  67% to 84% and leaves returns negative at every threshold
- **Model edge** — restricting to fixtures the model reads as mispriced makes
  returns *monotonically worse*, from −4% to −25%. Disagreement with the
  market measures what the model can't see, not what it knows
- **Price banding** — observed win rates track the price-implied rates across
  every odds band; no band clears the margin
- **Stake sizing** — cannot reverse a negative expectation, by linearity of
  expectation

Methodology notes:

- The elite tier is rebuilt each season from **earlier seasons only**. Clubs
  that later declined stay in — Napoli qualified in 2023 off their title and
  appears in four of the five worst combinations. A tier selected with
  hindsight would improve every figure and be wrong.
- An earlier version of the edge calculation compared hit rate against
  `1/mean(odds)` and reported a **positive 3.5% edge** on a strategy that
  loses. By Jensen's inequality that isn't `mean(1/odds)`, and with combined
  odds spanning 4 to 15 the error is large enough to invert the conclusion.
  Every figure now settles each bet at its own price.

Research exercise, not betting advice.

## Design decisions worth knowing

**Walk-forward, never shuffled.** The model refits on everything strictly
before each test date. Shuffled cross-validation would train on May to predict
September and inflate every number.

**One model per league.** Dixon-Coles strengths are identified from the graph
of who played whom. Dense within a league, far too sparse across them.
`walk_forward()` raises rather than pooling silently.

**The validator fails loudly.** Structural invariants are checked per league
per season, because Bundesliga plays 306 matches and Ligue 1 changed from 380
to 306 in 2023-24. Known-incomplete seasons (Ligue 1 abandoned 2019-20) are
recorded explicitly with a reason, and still get checked for team count — a
name-join bug inside a curtailed season shouldn't hide behind the exemption.

**The name resolver refuses to guess.** Unknown club names raise rather than
fuzzy-match. A silent bad join drops rows without an error.

**A calibration layer was built, tested and rejected.** Temperature scaling
fitted walk-forward returned T ≈ 1.09 and changed Brier by +0.0004 over 2,251
out-of-sample predictions. The model was already well calibrated, so it isn't
deployed. `plfc/calibration.py` is imported by nothing and stays as evidence
the question was asked.

## Structure

```
app.py                    league selector, fixtures, scorecard, ratings
pages/1_case_study.py     the evaluation study
build_case_study.py       computes study figures once -> JSON
plfc/
  ingest.py               pull, canonicalise, validate, cache
  leagues.py              per-league structural config
  names.py                canonical club name resolver
  model.py                Dixon-Coles + warm start
  backtest.py             walk-forward, league-aware
  multileague.py          fit and backtest all five separately
  ledger.py               sealed append-only forecast log
  shortlist.py            rolling elite tier, no hindsight
  parlay.py               week anchoring, combination counting
  legfilter.py            fixture-aware leg selection
  calibration.py          built, tested, not deployed
analysis/                 one-off study scripts
```

## Running it

```bash
git clone https://github.com/lokeshkarthikp-code/pl-forecaster
pip install -r requirements.txt
streamlit run app.py    # reads cached data; no API key needed
```

Fixtures need a free [football-data.org](https://www.football-data.org) token
in `FOOTBALL_DATA_TOKEN`. Results and odds need no key.

## Data

- **[Football-Data.co.uk](https://www.football-data.co.uk)** — results and
  closing odds, read directly as CSV
- **[football-data.org](https://www.football-data.org)** — forward fixture
  schedule

No scraping anywhere in the project. An earlier version used a browser
automation stack to fetch files that were plain CSV at stable URLs; removing
it cut ~200MB from the deploy and eliminated the only genuinely fragile
source.

## Known limitations

- **No team news.** Injuries, suspensions and rotation are invisible to the
  model. Section 3 of the study is a direct measurement of how much that costs.
- **Draws are structurally under-called.** The model assigns draws 25–29%
  correctly, but a 29% draw never outranks a 40% home win, so the
  highest-probability outcome is a draw in 0.5% of matches against a 25% base
  rate. Alternative decision rules were tested; none beat the argmax
  out-of-sample.
- **Promoted teams are priors, not estimates**, until they accumulate matches.
- **Managerial changes** aren't modelled; strength is assumed to drift
  smoothly.
- **Odds normalisation is proportional**, which distorts longshots.

## Tests

```bash
pytest
```

108 tests, network-free — all sources are mocked.

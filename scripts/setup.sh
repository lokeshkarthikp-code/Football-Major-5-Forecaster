#!/usr/bin/env bash
# One-time project setup. Run from the repo root:  bash scripts/setup.sh
set -euo pipefail

echo "==> Creating virtual environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies"
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install pytest -q

echo "==> Running test suite (no network required)"
python -m pytest tests/ -q

cat <<'MSG'

Setup complete.

Next steps:

  source .venv/bin/activate

  1. Build the dataset (first run takes several minutes -- it pulls ~10
     seasons and caches HTTP responses under ~/soccerdata):

       python -m plfc.ingest

     Expect this to be where things break first. If a team name is
     unresolved it raises with the exact list to add to CANONICAL_TEAMS
     in plfc/names.py.

  2. Run the walk-forward backtest for real numbers:

       python -m plfc.backtest --step-days 14

  3. Launch the dashboard:

       streamlit run app.py

MSG

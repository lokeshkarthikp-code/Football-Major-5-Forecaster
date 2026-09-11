"""
Three-leg parlays from the elite tier, counted with no hindsight.

    python trio_run.py

Each season's tier is rebuilt from EARLIER seasons only, so a trio only
accrues weeks in the seasons where all three clubs were actually
shortlisted at the time.

Read pooled_roi before the per-trio table. With ~35 trios of 20-40 weeks
each, the top row is selected for luck as much as for quality.
"""

import pandas as pd
from plfc.ingest import load_matches
from plfc.parlay import trio_counter, pooled_summary

pd.set_option("display.width", 200)
m = load_matches()

for bar in (0.60, 0.65):
    t = trio_counter(m, k=3, min_rate=bar, require_full=True, min_weeks=8)
    if t.empty:
        print(f"\n=== {bar:.2f} bar: no trios with enough weeks ===")
        continue

    print(f"\n{'='*70}\n=== Elite bar {bar:.2f} — {len(t)} trios ===\n{'='*70}")
    print("\nPOOLED (the number that generalises):")
    print(pooled_summary(t).round(4).to_string(index=False))

    print("\nBest 8 trios:")
    print(t.head(8).round(3).to_string(index=False))
    print("\nWorst 5 trios:")
    print(t.tail(5).round(3).to_string(index=False))

    print(f"\nHow often did all three win?  "
          f"{t['all_won'].sum()} of {t['weeks'].sum()} trio-weeks "
          f"({t['all_won'].sum()/t['weeks'].sum():.1%})")
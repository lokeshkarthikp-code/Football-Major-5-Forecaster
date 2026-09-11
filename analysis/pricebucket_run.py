"""
Is any PRICE band of trio parlays profitable?

    python pricebucket_run.py

Not a staking question. Stake sizing cannot change the sign of a negative
expectation. This asks whether some band of combined odds is positive on
its own, in which case you would bet only that band and skip the rest --
selection, not stake variation. Combined price is known before kickoff,
so it is legitimately usable.

WHAT A REAL RESULT LOOKS LIKE
-----------------------------
A trend across ADJACENT bands. With eight buckets, one or two landing
above zero is expected from noise alone. The `signif` column marks bands
more than two standard errors from zero, but read the shape before the
stars: one band up with both neighbours down is a fluke, not an edge.
"""

import pandas as pd
from plfc.ingest import load_matches
from plfc.parlay import price_buckets

pd.set_option("display.width", 200)
m = load_matches()

for bar in (0.60, 0.65):
    print(f"\n{'='*72}")
    print(f"=== Elite bar {bar:.2f} — 3-leg parlays bucketed by combined odds ===")
    print(f"{'='*72}")
    pb = price_buckets(m, k=3, min_rate=bar, min_weeks=30)
    if pb.empty:
        print("Not enough weeks per band.")
        continue
    print(pb.round(4).to_string(index=False))

    pos = pb[pb["signif"] == "+"]
    if pos.empty:
        print("\nNo band is significantly profitable. "
              "Selection by price does not rescue it.")
    else:
        print(f"\n{len(pos)} band(s) above two standard errors: "
              f"{', '.join(pos['odds_band'])}")
        print("Check whether their neighbours point the same way before "
              "believing it.")
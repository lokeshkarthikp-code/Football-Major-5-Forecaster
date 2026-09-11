from plfc.ingest import load_matches
from plfc.multileague import load_backtest
from plfc.legfilter import shortlisted_legs, sweep_thresholds, singles_return

m = load_matches()
res = load_backtest()

legs = shortlisted_legs(res, m, min_rate=0.65)
print(f"Shortlisted legs with odds: {len(legs)}")
print("\nBaseline (no filter):", {k: round(v, 4) for k, v in
      singles_return(legs).items() if isinstance(v, float)})

print("\n=== Filter by model probability (easy fixtures) ===")
print(sweep_thresholds(legs, by="model_p").round(4).to_string(index=False))

print("\n=== Filter by expected value (model vs market) ===")
print(sweep_thresholds(legs, by="ev").round(4).to_string(index=False))

from plfc.ingest import load_matches
from plfc.parlay import rolling_shortlist_parlay, failure_frequency, parlay_economics
from plfc.shortlist import rolling_shortlist

m = load_matches()

print("=== No-hindsight, season by season (0.65 bar) ===")
r = rolling_shortlist_parlay(m, min_rate=0.65)
print(r.drop(columns=["shortlist"]).round(3).to_string(index=False))

teams = rolling_shortlist(m, 2026, min_rate=0.65)["team"].tolist()
print(f"\n=== Current shortlist: {teams} ===")

print("\n=== Failure frequency by leg count ===")
print(failure_frequency(m, teams).round(3).to_string(index=False))

print("\n=== Economics ===")
print(parlay_economics(m, teams).round(3).to_string(index=False))

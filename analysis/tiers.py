from plfc.ingest import load_matches
from plfc.shortlist import tier_table
print(tier_table(load_matches()).to_string(index=False))

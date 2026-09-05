#!/usr/bin/env python3
"""
Audit team names across all leagues BEFORE committing to a full ingest.

Why run this first: the registry in names_intl.py was written from known
Football-Data.co.uk conventions, not scraped from the live feed. Some
spellings will be missing -- especially for clubs promoted and relegated
years ago. This script finds them in one pass rather than failing an
ingest one name at a time.

    python scripts/audit_leagues.py                 # all leagues, 10 seasons
    python scripts/audit_leagues.py --seasons 3
    python scripts/audit_leagues.py --league ESP-La Liga

Any unresolved name is printed as a paste-ready line for names_intl.py.
Nothing is guessed or auto-added: a wrong alias is worse than a missing
one, because a missing alias raises and a wrong one silently corrupts.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

from plfc.ingest import _seasons_as_strings, season_range  # noqa: E402
from plfc.leagues import DEFAULT_LEAGUES, LEAGUES, config  # noqa: E402
from plfc.names import resolve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, default=10)
    ap.add_argument("--league", action="append", default=None,
                    help="Repeatable. Defaults to all five.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    import soccerdata as sd

    leagues = args.league or DEFAULT_LEAGUES
    seasons = _seasons_as_strings(season_range(args.seasons))

    all_unresolved: dict[str, set[str]] = {}
    total_names = 0

    for lg in leagues:
        cfg = config(lg)
        print(f"\n=== {cfg.display} ({cfg.code}) ===")
        try:
            mh = sd.MatchHistory(leagues=lg, seasons=seasons)
            raw = mh.read_games().reset_index()
        except Exception as exc:                       # noqa: BLE001
            print(f"  FETCH FAILED: {exc}")
            continue

        names = sorted(set(raw["home_team"]) | set(raw["away_team"]))
        total_names += len(names)
        unresolved = [n for n in names if resolve(n, strict=False) is None]

        print(f"  {len(names)} distinct team names, "
              f"{len(names) - len(unresolved)} resolve, {len(unresolved)} do not")

        if unresolved:
            all_unresolved[lg] = set(unresolved)
            for n in unresolved:
                print(f"    UNRESOLVED: {n!r}")

        # Also surface anything suspicious: two raw names collapsing onto one
        # canonical id is legitimate (aliases), but worth eyeballing once.
        mapping: dict[str, list[str]] = {}
        for n in names:
            got = resolve(n, strict=False)
            if got:
                mapping.setdefault(got, []).append(n)
        collisions = {k: v for k, v in mapping.items() if len(v) > 1}
        if collisions:
            print("  aliases collapsing to one club (verify these are correct):")
            for canon, raws in sorted(collisions.items()):
                print(f"    {canon}: {raws}")

    print("\n" + "=" * 66)
    if not all_unresolved:
        print(f"All {total_names} team names across {len(leagues)} league(s) resolve.")
        print("Safe to run: python -m plfc.ingest")
        return 0

    print("ADD THESE TO plfc/names_intl.py -- do not guess the canonical id,")
    print("look each club up if you are not certain which it is:\n")
    for lg, names in all_unresolved.items():
        print(f"  # {config(lg).display}")
        for n in sorted(names):
            print(f'  "CANONICAL NAME HERE": ["{n}"],')
    return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the repo root,
# so plfc would not be importable. Add the project root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plfc.footballdata import (
    FootballDataError, check_token, probe_team_names,
)
from plfc.names import resolve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default="PL",
                    help="Competition code (PL, PD, SA, BL1, FL1, ...)")
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("1. Checking token...")
    try:
        check_token()
        print("   OK\n")
    except FootballDataError as e:
        print(f"   FAILED\n\n{e}\n")
        return 1

    print(f"2. Fetching team names for {args.competition}...")
    try:
        names = probe_team_names(args.competition, args.season)
    except FootballDataError as e:
        print(f"   FAILED: {e}")
        return 1

    print(f"   {len(names)} teams returned.\n")

    unresolved = []
    for n in names:
        got = resolve(n, strict=False)
        flag = "OK " if got else "!! "
        print(f"   {flag}{n:38} -> {got or 'UNRESOLVED'}")
        if not got:
            unresolved.append(n)

    if unresolved:
        print(f"\n{len(unresolved)} name(s) need adding to CANONICAL_TEAMS "
              f"in plfc/names.py:\n")
        for n in unresolved:
            print(f'    "{n}",')
        return 1

    print("\nAll names resolve. Safe to run: python -m plfc.ingest")
    return 0


if __name__ == "__main__":
    sys.exit(main())

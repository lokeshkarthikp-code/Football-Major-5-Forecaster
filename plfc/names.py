"""
Canonical team identity for Premier League data.

WHY THIS EXISTS
---------------
Football data sources spell team names inconsistently. FBref writes
"Nott'ham Forest", Football-Data.co.uk writes "Nott'm Forest", ClubElo
writes "Forest", Wikipedia writes "Nottingham Forest F.C.". If you join
on raw strings, rows silently vanish -- no exception, no warning, just a
season with 342 matches instead of 380 and a model trained on a hole.

The fix is a canonical registry: every source name maps to one canonical
id, and ANY name we cannot resolve raises. This is the single place team
identity is decided -- no library does it for us. Silent failure is the enemy;
we would rather crash on Wednesday morning than serve wrong predictions
on Saturday.

DESIGN
------
1. Exact alias lookup (fast path, covers everything we have seen).
2. Normalised lookup -- strip punctuation/accents/suffixes, casefold.
3. Fuzzy match, but ONLY as a *suggestion* -- it never auto-resolves.
   It raises with the suggestion so a human adds the alias deliberately.

Step 3 matters. Auto-fuzzy-matching is how you end up mapping
"Manchester City" to "Manchester Utd" at 0.82 similarity and corrupting
a decade of data. Fuzzy matching proposes; a human disposes.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable

__all__ = [
    "CANONICAL_TEAMS",
    "TeamNameError",
    "resolve",
    "resolve_series",
    "audit_names",
]


class TeamNameError(KeyError):
    """Raised when a team name cannot be resolved to a canonical id."""


# ---------------------------------------------------------------------------
# The registry.
#
# Key   = canonical id (what we use everywhere downstream).
# Value = every spelling we have observed in the wild, across FBref,
#         Football-Data.co.uk, ClubElo, Understat and common hand-entry.
#
# Covers all clubs to appear in the Premier League since 2014/15, plus
# recent Championship sides likely to be promoted -- so a promotion does
# not break the pipeline in August.
# ---------------------------------------------------------------------------
CANONICAL_TEAMS: dict[str, list[str]] = {
    "Arsenal": ["Arsenal", "Arsenal FC", "Arsenal F.C."],
    "Aston Villa": ["Aston Villa", "Villa", "Aston Villa FC"],
    "Bournemouth": [
        "Bournemouth", "AFC Bournemouth", "A.F.C. Bournemouth", "Bourne mouth",
    ],
    "Brentford": ["Brentford", "Brentford FC"],
    "Brighton": [
        "Brighton", "Brighton & Hove Albion", "Brighton and Hove Albion",
        "Brighton & Hove Albion FC", "Brighton Hove Albion",
    ],
    "Burnley": ["Burnley", "Burnley FC"],
    "Cardiff City": ["Cardiff City", "Cardiff", "Cardiff City FC"],
    "Chelsea": ["Chelsea", "Chelsea FC", "Chelsea F.C."],
    "Crystal Palace": ["Crystal Palace", "Palace", "C Palace", "Crystal Palace FC"],
    "Everton": ["Everton", "Everton FC"],
    "Fulham": ["Fulham", "Fulham FC"],
    "Huddersfield": ["Huddersfield", "Huddersfield Town", "Huddersfield Town AFC"],
    "Hull City": ["Hull City", "Hull", "Hull City AFC"],
    "Ipswich Town": ["Ipswich Town", "Ipswich", "Ipswich Town FC"],
    "Leeds United": ["Leeds United", "Leeds", "Leeds Utd", "Leeds United FC"],
    "Leicester City": ["Leicester City", "Leicester", "Leicester City FC"],
    "Liverpool": ["Liverpool", "Liverpool FC", "Liverpool F.C."],
    "Luton Town": ["Luton Town", "Luton", "Luton Town FC"],
    "Manchester City": [
        "Manchester City", "Man City", "Man. City", "Manchester City FC", "ManCity",
    ],
    "Manchester United": [
        "Manchester United", "Manchester Utd", "Man United", "Man Utd",
        "Man. United", "Manchester United FC", "ManUtd",
    ],
    "Middlesbrough": ["Middlesbrough", "Middlesboro", "Middlesbrough FC"],
    "Newcastle United": [
        "Newcastle United", "Newcastle Utd", "Newcastle", "Newcastle United FC",
    ],
    "Norwich City": ["Norwich City", "Norwich", "Norwich City FC"],
    "Nottingham Forest": [
        # The canonical headache. Every source spells this differently.
        "Nottingham Forest", "Nott'ham Forest", "Nott'm Forest", "Nottm Forest",
        "Nottingham", "Forest", "Nott'ingham Forest", "Nottingham Forest FC",
        "Nottinghamshire Forest",
    ],
    "Sheffield United": [
        "Sheffield United", "Sheffield Utd", "Sheff United", "Sheff Utd",
        "Sheffield United FC",
    ],
    "Southampton": ["Southampton", "Soton", "Southampton FC"],
    "Stoke City": ["Stoke City", "Stoke", "Stoke City FC"],
    "Sunderland": ["Sunderland", "Sunderland AFC"],
    "Swansea City": ["Swansea City", "Swansea", "Swansea City AFC"],
    "Tottenham": [
        "Tottenham", "Tottenham Hotspur", "Spurs", "Tottenham Hotspur FC",
    ],
    "Watford": ["Watford", "Watford FC"],
    "West Bromwich Albion": [
        "West Bromwich Albion", "West Brom", "West Bromwich", "WBA", "West Brom Albion",
    "West Bromwich Albion FC",
    ],
    "West Ham United": [
        "West Ham United", "West Ham", "West Ham Utd", "West Ham United FC",
    ],
    "Wolverhampton Wanderers": [
        "Wolverhampton Wanderers", "Wolves", "Wolverhampton",
        "Wolverhampton Wanderers FC",
    ],
    # --- Championship sides kept warm so a promotion does not break August ---
    "Coventry City": ["Coventry City", "Coventry"],
    "Sheffield Wednesday": ["Sheffield Wednesday", "Sheffield Weds", "Sheff Wed"],
    "Millwall": ["Millwall", "Millwall FC"],
    "Preston North End": ["Preston North End", "Preston"],
    "Bristol City": ["Bristol City", "Bristol"],
    "Blackburn Rovers": ["Blackburn Rovers", "Blackburn"],
    "Birmingham City": ["Birmingham City", "Birmingham"],
    "Wrexham": ["Wrexham", "Wrexham AFC"],
    "Charlton Athletic": ["Charlton Athletic", "Charlton"],
    "Oxford United": ["Oxford United", "Oxford Utd", "Oxford"],
}

# No folds currently needed. Kept as the hook for future short-form collisions
# (e.g. if a source emits a bare surname that collides with a canonical id).

# The four continental leagues live in names_intl.py purely for file size.
# They are merged here so there is exactly ONE registry and one resolver --
# a second lookup path would be a second place for names to diverge.
from .names_intl import INTERNATIONAL_LEAGUES  # noqa: E402

for _canon, _aliases in INTERNATIONAL_LEAGUES.items():
    CANONICAL_TEAMS.setdefault(_canon, [])
    for _a in _aliases:
        if _a not in CANONICAL_TEAMS[_canon]:
            CANONICAL_TEAMS[_canon].append(_a)

_FOLD: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

_SUFFIXES = re.compile(
    r"\b(fc|afc|f\.c\.|a\.f\.c\.|football club|united|utd|city|town|"
    r"albion|wanderers|hotspur|rovers|athletic)\b"
)


def _normalise(name: str) -> str:
    """Aggressive normalisation for the fallback lookup.

    Strips accents, punctuation, common club suffixes and whitespace, then
    casefolds. "Nott'ham Forest F.C." and "nottingham forest" both collapse
    toward "notthamforest" / "nottinghamforest" -- not identical, which is
    exactly why we still need the explicit alias list. This layer catches
    punctuation and casing drift, not genuine abbreviation.
    """
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^\w\s]", "", s)          # drop apostrophes, dots, hyphens
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_suffixes(name: str) -> str:
    s = _normalise(name)
    s = _SUFFIXES.sub("", s)
    return re.sub(r"\s+", "", s)


def _build_index() -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    loose: dict[str, str] = {}
    for canon, aliases in CANONICAL_TEAMS.items():
        target = _FOLD.get(canon, canon)
        for alias in {*aliases, canon, target}:
            exact[alias] = target
            exact[alias.casefold()] = target
            loose.setdefault(_normalise(alias), target)
            loose.setdefault(_strip_suffixes(alias), target)
    return exact, loose


_EXACT, _LOOSE = _build_index()

_CANON_IDS = sorted({_FOLD.get(c, c) for c in CANONICAL_TEAMS})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve(name: str, *, strict: bool = True) -> str | None:
    """Resolve a raw team name to its canonical id.

    Parameters
    ----------
    name
        Raw team name from any source.
    strict
        If True (default) an unresolvable name raises TeamNameError with a
        fuzzy suggestion. If False, returns None instead.

    Raises
    ------
    TeamNameError
        When the name cannot be resolved. The message includes the closest
        known names so you can add the alias to CANONICAL_TEAMS.

    Notes
    -----
    Fuzzy matching NEVER auto-resolves. It only decorates the error. This is
    deliberate -- an automatic 0.8-similarity match between "Manchester City"
    and "Manchester United" would corrupt the dataset invisibly.
    """
    if name is None:
        if strict:
            raise TeamNameError("Cannot resolve team name: got None")
        return None

    raw = str(name).strip()
    if not raw:
        if strict:
            raise TeamNameError("Cannot resolve team name: empty string")
        return None

    # 1. exact
    if raw in _EXACT:
        return _EXACT[raw]
    if raw.casefold() in _EXACT:
        return _EXACT[raw.casefold()]

    # 2. normalised
    for key in (_normalise(raw), _strip_suffixes(raw)):
        if key in _LOOSE:
            return _LOOSE[key]

    # 3. suggest, never decide
    if not strict:
        return None

    suggestions = get_close_matches(_normalise(raw), list(_LOOSE), n=3, cutoff=0.6)
    hint = ""
    if suggestions:
        named = sorted({_LOOSE[s] for s in suggestions})
        hint = f" Did you mean: {', '.join(named)}?"
    raise TeamNameError(
        f"Unknown team name {raw!r}.{hint} "
        f"Add it to CANONICAL_TEAMS in plfc/names.py -- do not guess."
    )


def resolve_series(values: Iterable[str], *, strict: bool = True):
    """Vectorised resolve for a pandas Series or any iterable.

    Collects ALL failures before raising, so one run tells you every alias
    you need to add rather than making you fix them one at a time.
    """
    import pandas as pd

    s = pd.Series(list(values))
    uniques = s.dropna().unique()

    mapping: dict[str, str] = {}
    failures: list[str] = []
    for u in uniques:
        got = resolve(u, strict=False)
        if got is None:
            failures.append(str(u))
        else:
            mapping[u] = got

    if failures and strict:
        lines = []
        for f in sorted(failures):
            sug = get_close_matches(_normalise(f), list(_LOOSE), n=2, cutoff=0.6)
            named = sorted({_LOOSE[x] for x in sug}) if sug else []
            lines.append(f"  {f!r}" + (f"  -> maybe {', '.join(named)}" if named else ""))
        raise TeamNameError(
            f"{len(failures)} unresolved team name(s):\n" + "\n".join(lines) +
            "\n\nAdd these to CANONICAL_TEAMS in plfc/names.py."
        )

    return s.map(mapping)


def audit_names(*frames_and_cols) -> dict[str, list[str]]:
    """Report unresolved names across several (DataFrame, column) pairs.

    Non-raising -- for exploration when adding a new data source.

        audit_names((games, "home_team"), (elo, "team"))
    """
    out: dict[str, list[str]] = {}
    for df, col in frames_and_cols:
        bad = sorted({
            str(v) for v in df[col].dropna().unique()
            if resolve(v, strict=False) is None
        })
        if bad:
            out[col] = bad
    return out
#!/usr/bin/env python3
"""Collect archived repositories — the positive examples.

Script 01 seeds candidates from npm search, which is biased toward *living*
packages: archived repos rank poorly and rarely surface. Training on that alone
leaves almost nothing to learn from, since abandonment would be near-absent from
the data.

This script goes after archived repos directly.

Two constraints shape the approach:

  - Any single search query returns at most 1000 results, however many match.
    So the corpus is partitioned into star-count slices and queried once per
    slice, which multiplies the reachable total.
  - Search is limited to 30 requests/minute (versus 5000/hour for the core API),
    so this runs slower per request than it looks. The client handles the
    waiting.

Usage:
    python scripts/01b_collect_archived.py
    python scripts/01b_collect_archived.py --language python
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from driftlogg.collect import GitHubClient
from driftlogg.config import settings

logger = logging.getLogger(__name__)

STAR_SLICES = [
    (10, 25),
    (26, 50),
    (51, 100),
    (101, 250),
    (251, 500),
    (501, 1000),
    (1001, 5000),
    (5001, None),
]
"""Star ranges queried separately to work around the 1000-result cap per query.

Weighted toward the low end because that is where most archived repos live —
a package with 10k stars rarely gets archived quietly.
"""

MIN_STARS = 10
"""Below this, repos are mostly abandoned toy projects that were never alive."""


def build_query(language: str, low: int, high: int | None) -> str:
    """Compose a search query for one star slice."""
    stars = f"{low}..{high}" if high else f">={low}"
    return f"archived:true language:{language} stars:{stars}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        default="javascript",
        help="Match the ecosystem chosen in 01 (javascript for npm, python for PyPI).",
    )
    parser.add_argument(
        "--per-slice",
        type=int,
        default=500,
        help="Results to pull per star slice.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings.ensure_dirs()

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    with GitHubClient() as client:
        for low, high in STAR_SLICES:
            query = build_query(args.language, low, high)
            logger.info("Searching: %s", query)

            try:
                repos = client.search_repositories(query, max_results=args.per_slice)
            except Exception:
                logger.exception("Search failed for slice %s-%s — continuing.", low, high)
                continue

            logger.info("  found %d", len(repos))

            for repo in repos:
                owner_login = (repo.get("owner") or {}).get("login")
                name = repo.get("name")
                if not owner_login or not name:
                    continue

                key = (owner_login, name)
                if key in seen:
                    continue
                seen.add(key)

                rows.append(
                    {
                        # npm name is unknown here; the repo name is the best
                        # available identifier and is only used as a key.
                        "package": name,
                        "owner": owner_login,
                        "repo": name,
                        "stars": repo.get("stargazers_count", 0),
                        "archived": True,
                        "seed_query": f"archived:{low}-{high or 'max'}",
                    }
                )

    if not rows:
        raise SystemExit("No archived repos found. Check your token and network.")

    frame = pd.DataFrame(rows)
    output = settings.interim_dir / "candidates_archived.parquet"
    frame.to_parquet(output, index=False)

    logger.info("Wrote %d archived repos to %s", len(frame), output)
    logger.info("Star distribution:\n%s", frame["stars"].describe().to_string())
    logger.info("Next: python scripts/02_fetch_repos.py")


if __name__ == "__main__":
    main()

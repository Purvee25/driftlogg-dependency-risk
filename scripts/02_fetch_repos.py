#!/usr/bin/env python3
"""Fetch GitHub history for every candidate package.

This is the long one. With a token you get 5000 requests/hour and each package
costs roughly 5-15 requests, so a few thousand packages runs over several hours.
Every response is cached to disk, so the script is safe to interrupt and rerun —
it will skip whatever it already has.

Usage:
    python scripts/02_fetch_repos.py
    python scripts/02_fetch_repos.py --limit 50      # smoke test first
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from tqdm import tqdm

from driftlogg.collect import GitHubClient
from driftlogg.config import settings

logger = logging.getLogger(__name__)


def fetch_one(client: GitHubClient, owner: str, repo: str) -> dict | None:
    """Collect the full payload set for one repository.

    Returns:
        Repo metadata plus commits, issues, releases and contributors, or None
        if the repository no longer exists.
    """
    metadata = client.get_repo(owner, repo)
    if metadata is None:
        return None

    return {
        "repo": metadata,
        "commits": client.get_commits(owner, repo),
        "issues": client.get_issues(owner, repo),
        "releases": client.get_releases(owner, repo),
        "contributors": client.get_contributors(owner, repo),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings.ensure_dirs()

    candidates_path = settings.interim_dir / "candidates.parquet"
    if not candidates_path.exists():
        raise SystemExit("No candidates found. Run scripts/01_collect_candidates.py first.")

    candidates = pd.read_parquet(candidates_path)
    if args.limit:
        candidates = candidates.head(args.limit)

    output_dir = settings.raw_dir / "packages"
    output_dir.mkdir(parents=True, exist_ok=True)

    fetched = 0
    missing = 0

    with GitHubClient() as client:
        for row in tqdm(candidates.itertuples(), total=len(candidates), desc="repos"):
            target = output_dir / f"{row.owner}__{row.repo}.json"
            if target.exists():
                continue

            try:
                payload = fetch_one(client, row.owner, row.repo)
            except Exception:
                logger.exception("Failed on %s/%s — continuing.", row.owner, row.repo)
                continue

            if payload is None:
                missing += 1
                continue

            payload["package"] = row.package
            target.write_text(json.dumps(payload))
            fetched += 1

    logger.info("Fetched %d repos (%d no longer exist).", fetched, missing)
    logger.info("Next: python scripts/03_build_dataset.py")


if __name__ == "__main__":
    main()

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


HISTORY_START = "2022-01-01T00:00:00Z"
"""Earliest data any feature can need.

The earliest prediction date is 2023-01-01 and the trailing window is 365 days,
so nothing before 2022 is ever read. Bounding the request server-side keeps
pagination short on long-lived repos.
"""

COMMIT_PAGES = 5
ISSUE_PAGES = 3
RELEASE_PAGES = 2
"""Page caps, traded against request budget.

At 5000 requests/hour the corpus size is bound by requests-per-repo, so these
caps decide whether a full run takes 3 hours or 10. The cost is saturation:
a repo with >500 commits since 2022 has `commits_trailing` capped. That
compresses the top of the range but preserves the signal — the separation
between a busy repo and a dead one survives intact, and every saturated repo
is a negative anyway.
"""


def fetch_one(client: GitHubClient, owner: str, repo: str) -> dict | None:
    """Collect the payload set needed for feature extraction.

    Note:
        Contributors are deliberately not fetched. `features.py` derives
        contributor counts and the bus factor from commit authors, so the
        /contributors endpoint was pure request budget spent on nothing.

    Returns:
        Repo metadata plus commits, issues and releases, or None if the
        repository no longer exists.
    """
    metadata = client.get_repo(owner, repo)
    if metadata is None:
        return None

    return {
        "repo": metadata,
        "commits": client.get_commits(owner, repo, since=HISTORY_START, max_pages=COMMIT_PAGES),
        "issues": client.get_issues(owner, repo, since=HISTORY_START, max_pages=ISSUE_PAGES),
        "releases": client.get_releases(owner, repo, max_pages=RELEASE_PAGES),
    }


def load_candidates() -> pd.DataFrame:
    """Merge the live and archived candidate pools.

    Both sources are needed: script 01 supplies packages that are alive (the
    negatives) and 01b supplies archived ones (the positives). Training on
    either alone gives the model nothing to separate.

    Returns:
        Deduplicated candidates across both sources.

    Raises:
        SystemExit: If neither source has been collected yet.
    """
    sources = {
        "live": settings.interim_dir / "candidates.parquet",
        "archived": settings.interim_dir / "candidates_archived.parquet",
    }

    frames = []
    for name, path in sources.items():
        if path.exists():
            frame = pd.read_parquet(path)
            logger.info("Loaded %d %s candidates.", len(frame), name)
            frames.append(frame)
        else:
            logger.warning("Missing %s candidates (%s).", name, path.name)

    if not frames:
        raise SystemExit(
            "No candidates found. Run scripts/01_collect_candidates.py "
            "and scripts/01b_collect_archived.py first."
        )

    if len(frames) == 1:
        logger.warning(
            "Only one candidate source present — expect a severe class imbalance. "
            "Run both 01 and 01b before the full fetch."
        )

    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=["owner", "repo"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only fetch N (sampled).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings.ensure_dirs()

    candidates = load_candidates()
    if args.limit:
        # Sample rather than head() so a smoke test still covers both sources.
        candidates = candidates.sample(n=min(args.limit, len(candidates)), random_state=42)

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

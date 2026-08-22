#!/usr/bin/env python3
"""Build the candidate package list.

Produces a table of packages worth studying, each mapped to its GitHub repo.
This is the input to every later stage.

Aim for a few thousand packages with a deliberate mix:
  - popular and clearly alive
  - popular and since archived (your positive examples)
  - quiet but healthy (the negative controls that stop the model from simply
    learning "no commits means dead")

Usage:
    python scripts/01_collect_candidates.py --limit 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import re

import httpx
import pandas as pd

from driftlogg.config import settings

logger = logging.getLogger(__name__)

NPM_SEARCH_URL = "https://registry.npmjs.org/-/v1/search"
NPM_SEARCH_PAGE_SIZE = 250
GITHUB_REPO_RE = re.compile(r"github\.com[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


def extract_repo(package: dict) -> tuple[str, str] | None:
    """Pull (owner, repo) out of an npm search result, if it points at GitHub."""
    links = package.get("links", {})
    for key in ("repository", "homepage"):
        url = links.get(key)
        if not url:
            continue
        match = GITHUB_REPO_RE.search(url)
        if match:
            return match.group(1), match.group(2)
    return None


def search_npm(query: str, limit: int) -> list[dict]:
    """Page through npm search for one query.

    Args:
        query: Search text passed to the registry.
        limit: Maximum results to collect.

    Returns:
        Raw package objects from the registry.
    """
    collected: list[dict] = []
    with httpx.Client(timeout=settings.request_timeout_seconds) as client:
        for offset in range(0, limit, NPM_SEARCH_PAGE_SIZE):
            size = min(NPM_SEARCH_PAGE_SIZE, limit - offset)
            response = client.get(
                NPM_SEARCH_URL,
                params={"text": query, "size": size, "from": offset},
            )
            response.raise_for_status()
            objects = response.json().get("objects", [])
            if not objects:
                break
            collected.extend(obj["package"] for obj in objects)
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000, help="Packages per query.")
    parser.add_argument(
        "--queries",
        nargs="+",
        # Broad, unrelated terms give a more representative sample than one topic.
        default=["react", "cli", "parser", "logger", "test", "http", "config", "utils"],
        help="Search terms used to seed the candidate pool.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings.ensure_dirs()

    rows: list[dict] = []
    seen: set[str] = set()

    for query in args.queries:
        logger.info("Searching npm for %r...", query)
        for package in search_npm(query, args.limit // len(args.queries)):
            name = package.get("name")
            if not name or name in seen:
                continue
            seen.add(name)

            repo = extract_repo(package)
            if repo is None:
                continue

            rows.append(
                {
                    "package": name,
                    "owner": repo[0],
                    "repo": repo[1],
                    "npm_version": package.get("version"),
                    "seed_query": query,
                }
            )

    frame = pd.DataFrame(rows).drop_duplicates(subset=["owner", "repo"])
    output = settings.interim_dir / "candidates.parquet"
    frame.to_parquet(output, index=False)

    logger.info("Wrote %d candidates to %s", len(frame), output)
    logger.info("Next: python scripts/02_fetch_repos.py")

    # Keep a readable copy for eyeballing during development.
    (settings.interim_dir / "candidates_sample.json").write_text(json.dumps(rows[:20], indent=2))


if __name__ == "__main__":
    main()

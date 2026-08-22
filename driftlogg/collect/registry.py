"""Package registry lookups and repository resolution.

Shared by the collection scripts and the scoring API: both need to turn a
package name into the GitHub repository behind it.
"""

from __future__ import annotations

import logging
import re

import httpx

from driftlogg.config import settings

logger = logging.getLogger(__name__)

NPM_REGISTRY_URL = "https://registry.npmjs.org"
NPM_SEARCH_URL = f"{NPM_REGISTRY_URL}/-/v1/search"

GITHUB_REPO_RE = re.compile(r"github\.com[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")
"""Matches the GitHub owner/repo pair in a repository or homepage URL."""


def extract_github_repo(url: str | None) -> tuple[str, str] | None:
    """Pull (owner, repo) out of a URL, if it points at GitHub.

    Args:
        url: Repository or homepage URL, possibly None or non-GitHub.

    Returns:
        The owner and repo names, or None if the URL is not a GitHub repo.

    Example:
        >>> extract_github_repo("https://github.com/expressjs/express.git")
        ('expressjs', 'express')
    """
    if not url:
        return None
    match = GITHUB_REPO_RE.search(url.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def repo_from_package_metadata(package: dict) -> tuple[str, str] | None:
    """Find the GitHub repo for a package from its registry metadata.

    Checks the declared repository first, then falls back to the homepage —
    plenty of packages only fill in one of the two.

    Args:
        package: Registry metadata for a single package.

    Returns:
        The owner and repo names, or None if neither field points at GitHub.
    """
    repository = package.get("repository")
    if isinstance(repository, dict):
        found = extract_github_repo(repository.get("url"))
        if found:
            return found
    elif isinstance(repository, str):
        found = extract_github_repo(repository)
        if found:
            return found

    # npm search results nest the same fields under "links".
    links = package.get("links", {})
    for key in ("repository", "homepage"):
        found = extract_github_repo(links.get(key))
        if found:
            return found

    return extract_github_repo(package.get("homepage"))


class NpmClient:
    """Minimal npm registry client for package metadata lookups."""

    def __init__(self, timeout: float | None = None) -> None:
        self._client = httpx.Client(timeout=timeout or settings.request_timeout_seconds)

    def __enter__(self) -> NpmClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_package(self, name: str) -> dict | None:
        """Fetch registry metadata for one package.

        Args:
            name: Package name, scoped names included (e.g. "@scope/pkg").

        Returns:
            Registry metadata, or None if the package does not exist.
        """
        try:
            response = self._client.get(f"{NPM_REGISTRY_URL}/{name}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            logger.warning("Registry lookup failed for %s", name, exc_info=True)
            return None

    def resolve_repo(self, name: str) -> tuple[str, str] | None:
        """Resolve a package name to its GitHub repository.

        Args:
            name: Package name.

        Returns:
            The owner and repo names, or None if unresolvable.
        """
        metadata = self.get_package(name)
        if metadata is None:
            return None
        return repo_from_package_metadata(metadata)

"""GitHub API client with disk caching and rate-limit handling.

Every response is cached to disk before parsing. Collection runs take hours and
you will re-parse the same data many times while iterating on features — never
re-download it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from driftlogg.config import settings

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"
MAX_PAGE_SIZE = 100
RATE_LIMIT_BUFFER = 50
"""Stop and wait once this few requests remain, so parallel work doesn't tip us over."""


class RateLimitedError(RuntimeError):
    """Raised when GitHub reports the hourly quota is exhausted."""


class GitHubClient:
    """Thin GitHub REST wrapper that caches to disk and respects rate limits.

    Args:
        token: Personal access token. Falls back to settings.
        cache_dir: Where raw JSON responses are written. Falls back to settings.

    Example:
        >>> with GitHubClient() as gh:
        ...     repo = gh.get_repo("expressjs", "express")
        ...     print(repo["archived"])
    """

    def __init__(self, token: str | None = None, cache_dir: Path | None = None) -> None:
        self._token = token if token is not None else settings.github_token
        self._cache_dir = cache_dir or (settings.raw_dir / "github")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        else:
            logger.warning("No GitHub token set — you get 60 requests/hour instead of 5000.")

        self._client = httpx.Client(
            headers=headers,
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
        )

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ caching

    def _cache_path(self, path: str, params: dict[str, Any] | None) -> Path:
        """Build a stable, filesystem-safe cache path for a request."""
        key = json.dumps({"path": path, "params": params or {}}, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        safe_name = path.strip("/").replace("/", "_")[:80]
        return self._cache_dir / f"{safe_name}__{digest}.json"

    # ------------------------------------------------------------------ requests

    def _wait_for_rate_limit(self, response: httpx.Response) -> None:
        """Sleep until the quota resets if we are close to exhausting it."""
        remaining = response.headers.get("x-ratelimit-remaining")
        reset_at = response.headers.get("x-ratelimit-reset")
        if remaining is None or reset_at is None:
            return

        if int(remaining) > RATE_LIMIT_BUFFER:
            return

        sleep_seconds = max(0, int(reset_at) - int(time.time())) + 1
        logger.info("Rate limit nearly exhausted; sleeping %ss until reset.", sleep_seconds)
        time.sleep(sleep_seconds)

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, RateLimitedError)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(settings.max_retries),
        reraise=True,
    )
    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Issue one GET, honouring the cache and the rate limit."""
        cache_path = self._cache_path(path, params)
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        response = self._client.get(f"{API_ROOT}{path}", params=params)

        if response.status_code == 403 and "rate limit" in response.text.lower():
            self._wait_for_rate_limit(response)
            raise RateLimitedError(path)

        if response.status_code == 404:
            logger.debug("Not found: %s", path)
            cache_path.write_text(json.dumps(None))
            return None

        response.raise_for_status()
        payload = response.json()

        cache_path.write_text(json.dumps(payload))
        self._wait_for_rate_limit(response)
        return payload

    def _paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        max_pages: int = 10,
    ) -> list[dict[str, Any]]:
        """Collect up to `max_pages` pages of a list endpoint.

        The cap exists on purpose: a handful of pages is enough to measure
        recent activity, and unbounded paging on popular repos will burn the
        entire hourly quota on a single package.
        """
        results: list[dict[str, Any]] = []
        params = dict(params or {})
        params.setdefault("per_page", MAX_PAGE_SIZE)

        for page in range(1, max_pages + 1):
            params["page"] = page
            batch = self._request(path, params)
            if not batch:
                break
            results.extend(batch)
            if len(batch) < MAX_PAGE_SIZE:
                break

        return results

    # ------------------------------------------------------------------ endpoints

    def get_repo(self, owner: str, repo: str) -> dict[str, Any] | None:
        """Repository metadata. Carries the `archived` flag used for labelling."""
        return self._request(f"/repos/{owner}/{repo}")

    def get_commits(
        self,
        owner: str,
        repo: str,
        since: str | None = None,
        until: str | None = None,
        max_pages: int = 10,
    ) -> list[dict[str, Any]]:
        """Commits, optionally bounded by ISO-8601 timestamps.

        Pass `until` when building features so the request itself cannot return
        anything after the prediction cutoff.
        """
        params: dict[str, Any] = {}
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        return self._paginate(f"/repos/{owner}/{repo}/commits", params, max_pages)

    def get_issues(
        self,
        owner: str,
        repo: str,
        since: str | None = None,
        state: str = "all",
        max_pages: int = 10,
    ) -> list[dict[str, Any]]:
        """Issues and pull requests. Filter PRs out via the `pull_request` key."""
        params: dict[str, Any] = {"state": state, "sort": "created", "direction": "desc"}
        if since:
            params["since"] = since
        return self._paginate(f"/repos/{owner}/{repo}/issues", params, max_pages)

    def get_contributors(self, owner: str, repo: str, max_pages: int = 3) -> list[dict[str, Any]]:
        """Contributors ordered by commit count. Used for the bus-factor feature."""
        return self._paginate(f"/repos/{owner}/{repo}/contributors", None, max_pages)

    def get_releases(self, owner: str, repo: str, max_pages: int = 3) -> list[dict[str, Any]]:
        """Published releases, newest first."""
        return self._paginate(f"/repos/{owner}/{repo}/releases", None, max_pages)

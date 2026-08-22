"""Leakage-safe feature extraction.

Every function here takes an `as_of` cutoff and must ignore all data at or after
it. This is the single most important invariant in the project: a feature that
peeks past the cutoff produces a model that scores brilliantly in testing and is
useless in production, because at prediction time that data does not exist yet.

`FeatureWindow.filter_events` is the chokepoint — route every timestamped
collection through it rather than filtering by hand at each call site.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

TRAILING_WINDOW_DAYS = 365
"""How far back to look when computing activity features."""

RECENT_WINDOW_DAYS = 90
"""Short window, compared against the trailing window to derive trends."""

NEVER_OBSERVED_DAYS = 99_999.0
"""Stands in for "this never happened" in days-since features.

A finite sentinel rather than `inf` for two reasons: infinities crash
`DataFrame.replace` on certain frame shapes, and NaN would break the baseline —
`nan >= 180` is False, so a repo with zero commits would read as healthy. A
large finite value keeps comparisons and ordering correct everywhere.
"""


class LeakageError(RuntimeError):
    """Raised when a feature computation is handed data from after the cutoff."""


@dataclass(frozen=True)
class FeatureWindow:
    """A time-bounded view of a package's history.

    Attributes:
        as_of: The prediction date. Nothing at or after this may be used.
        trailing_days: Size of the long lookback window.
    """

    as_of: datetime
    trailing_days: int = TRAILING_WINDOW_DAYS

    @property
    def start(self) -> datetime:
        return self.as_of - timedelta(days=self.trailing_days)

    @property
    def recent_start(self) -> datetime:
        return self.as_of - timedelta(days=RECENT_WINDOW_DAYS)

    def filter_events(
        self,
        events: list[dict[str, Any]],
        timestamp_key: str,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Keep only events inside the window, strictly before the cutoff.

        Args:
            events: Raw event dicts.
            timestamp_key: Key holding an ISO-8601 timestamp.
            since: Optional later lower bound (defaults to the window start).

        Returns:
            Events falling in [since, as_of).
        """
        lower = since or self.start
        kept = []
        for event in events:
            ts = parse_timestamp(event.get(timestamp_key))
            if ts is None:
                continue
            if lower <= ts < self.as_of:
                kept.append(event)
        return kept

    def assert_no_leakage(self, timestamps: list[datetime]) -> None:
        """Fail loudly if any timestamp is at or after the cutoff.

        Raises:
            LeakageError: If future data reached a feature computation.
        """
        future = [ts for ts in timestamps if ts >= self.as_of]
        if future:
            raise LeakageError(
                f"{len(future)} event(s) at or after cutoff {self.as_of.isoformat()}; "
                f"earliest offender {min(future).isoformat()}"
            )


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp to a naive UTC datetime, or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


@dataclass
class PackageFeatures:
    """Model-ready features for one package at one prediction date.

    Trends matter more than levels here: a package going from weekly commits to
    nothing is a far stronger signal than a package that has always been quiet.
    """

    package: str
    as_of: datetime

    # Activity
    commits_trailing: int = 0
    commits_recent: int = 0
    commit_velocity_ratio: float = 0.0
    """Recent commit rate over trailing rate. Below 1.0 means slowing down."""
    days_since_last_commit: float = NEVER_OBSERVED_DAYS

    # People
    active_contributors_trailing: int = 0
    active_contributors_recent: int = 0
    bus_factor_ratio: float = 0.0
    """Share of commits by the single top contributor. Near 1.0 is fragile."""

    # Responsiveness — usually the strongest predictor
    median_first_response_hours: float | None = None
    response_time_trend: float = 0.0
    """Recent median response time over trailing median. Above 1.0 means slowing."""

    # Backlog
    issues_opened_trailing: int = 0
    issues_closed_trailing: int = 0
    open_close_ratio: float = 0.0
    stale_open_issues: int = 0

    # Releases
    releases_trailing: int = 0
    days_since_last_release: float = NEVER_OBSERVED_DAYS

    # Ecosystem
    stars: int = 0
    forks: int = 0
    fork_star_ratio: float = 0.0

    # Meta
    has_funding_link: bool = False
    open_issues_count: int = 0

    extras: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Flatten to a single dict suitable for a DataFrame row."""
        row = asdict(self)
        row.pop("extras")
        row.update(self.extras)
        return row


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide without blowing up on a zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def _days_since(latest: datetime | None, as_of: datetime) -> float:
    """Days between a timestamp and the cutoff.

    Returns NEVER_OBSERVED_DAYS when the event never happened, so downstream
    comparisons treat "never" as "very long ago" rather than as missing.
    """
    if latest is None:
        return NEVER_OBSERVED_DAYS
    return (as_of - latest).total_seconds() / 86400.0


def compute_commit_features(
    commits: list[dict[str, Any]],
    window: FeatureWindow,
    features: PackageFeatures,
) -> None:
    """Populate commit-activity and contributor features in place.

    Args:
        commits: Raw commit payloads from the GitHub API.
        window: The time window and cutoff.
        features: Target object, mutated in place.
    """

    def commit_time(commit: dict[str, Any]) -> datetime | None:
        return parse_timestamp(commit.get("commit", {}).get("author", {}).get("date"))

    in_window = [c for c in commits if (t := commit_time(c)) and window.start <= t < window.as_of]
    timestamps = [t for c in in_window if (t := commit_time(c))]
    window.assert_no_leakage(timestamps)

    recent = [t for t in timestamps if t >= window.recent_start]

    features.commits_trailing = len(timestamps)
    features.commits_recent = len(recent)
    features.days_since_last_commit = _days_since(max(timestamps, default=None), window.as_of)

    # Normalise both counts to a per-day rate before comparing the windows.
    trailing_rate = _safe_ratio(len(timestamps), window.trailing_days)
    recent_rate = _safe_ratio(len(recent), RECENT_WINDOW_DAYS)
    features.commit_velocity_ratio = _safe_ratio(recent_rate, trailing_rate)

    def author_login(commit: dict[str, Any]) -> str | None:
        author = commit.get("author")
        return author.get("login") if isinstance(author, dict) else None

    authors = [login for c in in_window if (login := author_login(c))]
    features.active_contributors_trailing = len(set(authors))

    recent_authors = [
        login
        for c in in_window
        if (t := commit_time(c)) and t >= window.recent_start and (login := author_login(c))
    ]
    features.active_contributors_recent = len(set(recent_authors))

    if authors:
        top_author_commits = max(authors.count(a) for a in set(authors))
        features.bus_factor_ratio = _safe_ratio(top_author_commits, len(authors))


def compute_issue_features(
    issues: list[dict[str, Any]],
    window: FeatureWindow,
    features: PackageFeatures,
) -> None:
    """Populate backlog and responsiveness features in place.

    Pull requests are filtered out — GitHub returns them from the issues
    endpoint, and mixing the two distorts both the backlog and response metrics.

    Args:
        issues: Raw issue payloads from the GitHub API.
        window: The time window and cutoff.
        features: Target object, mutated in place.
    """
    real_issues = [i for i in issues if "pull_request" not in i]
    opened = window.filter_events(real_issues, "created_at")
    window.assert_no_leakage([t for i in opened if (t := parse_timestamp(i.get("created_at")))])

    closed = window.filter_events(real_issues, "closed_at")

    features.issues_opened_trailing = len(opened)
    features.issues_closed_trailing = len(closed)
    features.open_close_ratio = _safe_ratio(len(opened), len(closed), default=float(len(opened)))

    features.stale_open_issues = sum(
        1
        for issue in opened
        if issue.get("state") == "open"
        and (created := parse_timestamp(issue.get("created_at")))
        and (window.as_of - created).days > RECENT_WINDOW_DAYS
    )

    # TODO: median_first_response_hours and response_time_trend need per-issue
    # comment timelines (/issues/{n}/comments), which is one request per issue.
    # Fetch them only for a sampled subset of issues — see scripts/04.


def compute_repo_features(
    repo: dict[str, Any],
    releases: list[dict[str, Any]],
    window: FeatureWindow,
    features: PackageFeatures,
) -> None:
    """Populate repository-level and release-cadence features in place.

    Args:
        repo: Repository payload from the GitHub API.
        releases: Release payloads, newest first.
        window: The time window and cutoff.
        features: Target object, mutated in place.
    """
    features.stars = repo.get("stargazers_count", 0)
    features.forks = repo.get("forks_count", 0)
    features.fork_star_ratio = _safe_ratio(features.forks, features.stars)
    features.open_issues_count = repo.get("open_issues_count", 0)
    features.has_funding_link = bool(repo.get("has_sponsorships") or repo.get("funding_links"))

    published = window.filter_events(releases, "published_at")
    timestamps = [t for r in published if (t := parse_timestamp(r.get("published_at")))]
    window.assert_no_leakage(timestamps)

    features.releases_trailing = len(timestamps)
    features.days_since_last_release = _days_since(max(timestamps, default=None), window.as_of)


def build_features(
    package: str,
    as_of: datetime,
    repo: dict[str, Any],
    commits: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    releases: list[dict[str, Any]],
) -> PackageFeatures:
    """Assemble the full feature vector for one package at one date.

    Args:
        package: Registry name.
        as_of: Prediction date; no data at or after this is used.
        repo: Repository payload.
        commits: Commit payloads.
        issues: Issue payloads (PRs are filtered internally).
        releases: Release payloads.

    Returns:
        Populated features.

    Raises:
        LeakageError: If any input contains post-cutoff data.
    """
    window = FeatureWindow(as_of=as_of)
    features = PackageFeatures(package=package, as_of=as_of)

    compute_commit_features(commits, window, features)
    compute_issue_features(issues, window, features)
    compute_repo_features(repo, releases, window, features)

    return features


FEATURE_COLUMNS = [
    "commits_trailing",
    "commits_recent",
    "commit_velocity_ratio",
    "days_since_last_commit",
    "active_contributors_trailing",
    "active_contributors_recent",
    "bus_factor_ratio",
    "issues_opened_trailing",
    "issues_closed_trailing",
    "open_close_ratio",
    "stale_open_issues",
    "releases_trailing",
    "days_since_last_release",
    "stars",
    "forks",
    "fork_star_ratio",
    "has_funding_link",
    "open_issues_count",
]
"""Columns fed to the model.

Deliberately excludes `median_first_response_hours` and `response_time_trend`:
both are declared on PackageFeatures but not yet computed, so one is always
None and the other always 0.0. A never-populated column is not a feature — it
is a null column that LightGBM rejects outright and that would silently
contribute nothing even if it were coerced. Add them back here once
compute_issue_features actually fills them in.
"""

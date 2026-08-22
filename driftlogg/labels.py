"""Defining what counts as an abandoned package.

This module is the foundation of the project. If the label is wrong, every
downstream metric is meaningless — a model can only be as good as the thing it
is taught to predict.

Design decision: `archived_at` on the GitHub repo is the primary signal. When a
maintainer archives a repo they are explicitly declaring it finished, and the
event carries a timestamp. That gives an unambiguous, dated, machine-readable
positive label. Softer signals (deprecation notices, README wording) are
recorded as secondary evidence but are noisier and should be reviewed by hand
before being trusted.

The trap this module exists to avoid: a stable, feature-complete library with
no recent commits is NOT abandoned. Silence alone is not evidence of death.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

UNMAINTAINED_PATTERNS = (
    r"\bno longer maintained\b",
    r"\bunmaintained\b",
    r"\bdeprecated\b",
    r"\bthis project is dead\b",
    r"\blooking for (?:a )?(?:new )?maintainers?\b",
    r"\bnot actively maintained\b",
)
_UNMAINTAINED_RE = re.compile("|".join(UNMAINTAINED_PATTERNS), re.IGNORECASE)

DEFAULT_HORIZON_DAYS = 90
"""How far ahead the model predicts. Also the window used to assign labels."""


class LabelSource(str, Enum):
    """Where an abandonment label came from, strongest first."""

    ARCHIVED = "archived"
    """Repo explicitly archived on GitHub. Dated and unambiguous."""

    DEPRECATED = "deprecated"
    """Registry-level deprecation flag (npm `deprecated`, PyPI classifier)."""

    README_NOTICE = "readme_notice"
    """Maintainer wrote an unmaintained notice. No reliable date — review by hand."""

    NONE = "none"
    """No evidence of abandonment."""


@dataclass(frozen=True)
class AbandonmentLabel:
    """The outcome for one package at one prediction date.

    Attributes:
        package: Registry name of the package.
        as_of: Prediction date. Features may only use data from before this.
        is_abandoned: Whether abandonment happened within the horizon after `as_of`.
        event_date: When abandonment occurred, if known.
        source: Which signal produced the label.
        needs_review: True when the label rests on a noisy signal.
    """

    package: str
    as_of: datetime
    is_abandoned: bool
    event_date: datetime | None
    source: LabelSource
    needs_review: bool = False

    @property
    def lead_time_days(self) -> int | None:
        """Days between the prediction date and the abandonment event."""
        if self.event_date is None:
            return None
        return (self.event_date - self.as_of).days


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp into a naive UTC datetime."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def detect_abandonment_event(
    repo: dict[str, Any],
    readme_text: str | None = None,
) -> tuple[datetime | None, LabelSource]:
    """Find when (and how) a repository was abandoned.

    Args:
        repo: Repository payload from the GitHub API.
        readme_text: Decoded README contents, if fetched.

    Returns:
        The abandonment date (None if unknown or alive) and the signal used.

    Note:
        A README notice returns `None` for the date even when matched, because
        the text carries no timestamp. Those rows must be dated by hand — or
        excluded — before training.
    """
    if repo.get("archived"):
        # `archived_at` is not always populated on older repos; fall back to the
        # last push, which is the closest proxy for when work stopped.
        archived_at = _parse_timestamp(repo.get("archived_at"))
        fallback = _parse_timestamp(repo.get("pushed_at"))
        return archived_at or fallback, LabelSource.ARCHIVED

    if readme_text and _UNMAINTAINED_RE.search(readme_text):
        return None, LabelSource.README_NOTICE

    return None, LabelSource.NONE


def build_label(
    package: str,
    as_of: datetime,
    repo: dict[str, Any],
    readme_text: str | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> AbandonmentLabel:
    """Label one package at one prediction date.

    A package is positive only if abandonment happened strictly after `as_of`
    and within `horizon_days`. Packages already dead at `as_of` must be dropped
    upstream — predicting a death that already happened is not a prediction.

    Args:
        package: Registry name.
        as_of: The prediction date.
        repo: Repository payload from the GitHub API.
        readme_text: Decoded README, if available.
        horizon_days: Size of the prediction window.

    Returns:
        The label for this (package, date) pair.
    """
    event_date, source = detect_abandonment_event(repo, readme_text)
    needs_review = source is LabelSource.README_NOTICE

    if event_date is None:
        return AbandonmentLabel(
            package=package,
            as_of=as_of,
            is_abandoned=False,
            event_date=None,
            source=source if source is LabelSource.NONE else source,
            needs_review=needs_review,
        )

    horizon_end = as_of + timedelta(days=horizon_days)
    within_window = as_of < event_date <= horizon_end

    return AbandonmentLabel(
        package=package,
        as_of=as_of,
        is_abandoned=within_window,
        event_date=event_date,
        source=source,
        needs_review=needs_review,
    )


def is_already_dead(label: AbandonmentLabel) -> bool:
    """Whether the package was already abandoned before the prediction date.

    These rows leak the answer and must be removed from the dataset.
    """
    return label.event_date is not None and label.event_date <= label.as_of

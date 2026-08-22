"""Defining what counts as an abandoned package.

This module is the foundation of the project. If the label is wrong, every
downstream metric is meaningless — a model can only be as good as the thing it
is taught to predict.

**Why this was rewritten.** The first version labelled on GitHub's `archived`
flag. It was dated, explicit, and machine-readable, which made it look ideal.
Measuring the trained model exposed the flaw: the median package that got
archived had *already been silent for over a year* before anyone archived it.
Archiving is not the moment a package dies — it is the moment a maintainer
eventually gets around to clicking a button, often years later. Predicting that
means predicting administrative paperwork, which activity signals cannot see
coming. It also mislabelled genuinely dead packages as alive whenever nobody
had bothered to archive them.

**What replaced it.** The target is now sustained silence, which is both
forecastable and the thing that actually matters: silence is what stops
security patches arriving. The structure is a true 90-day-ahead forecast:

    features        [as_of - 365, as_of)     what the model sees
    grace period    [as_of, as_of + 90]      ignored, the forecast horizon
    verdict window  [+90, +90 + 180]         must be wholly silent to be positive

A package only enters the dataset if it was alive at `as_of` — forecasting the
death of something already dead is not forecasting. Rows whose verdict window
extends past the data collection date are right-censored and dropped, because
the silence cannot be observed yet.
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

HORIZON_DAYS = 90
"""How far ahead the forecast reaches before the verdict window opens."""

SUSTAINED_SILENCE_DAYS = 180
"""How long silence must persist to count as abandonment rather than a lull.

Plenty of healthy packages go quiet for a quarter. Six months of total silence
following the horizon is a much stronger signal that work has actually stopped.
"""

ACTIVE_WINDOW_DAYS = 365
"""A package counts as alive at `as_of` if it committed within this window."""

DEFAULT_HORIZON_DAYS = HORIZON_DAYS
"""Backwards-compatible alias."""


class LabelSource(str, Enum):
    """How a label was determined."""

    SILENCE = "silence"
    """Observed sustained silence in the verdict window. The primary signal."""

    ARCHIVED = "archived"
    """Repo archived during the window. Corroborates silence."""

    ACTIVE = "active"
    """Commits continued through the verdict window."""

    README_NOTICE = "readme_notice"
    """Maintainer wrote an unmaintained notice. Undated, so review by hand."""


class ExclusionReason(str, Enum):
    """Why a (package, date) pair cannot be used for training."""

    NOT_ALIVE = "not_alive"
    """Already silent at `as_of`; there is nothing left to predict."""

    CENSORED = "censored"
    """Verdict window extends past the data we collected."""

    NONE = "none"


@dataclass(frozen=True)
class AbandonmentLabel:
    """The outcome for one package at one prediction date.

    Attributes:
        package: Registry name of the package.
        as_of: Prediction date. Features may only use data from before this.
        is_abandoned: Whether the package went sustainedly silent.
        event_date: Last observed commit before silence, when known.
        source: Which signal produced the label.
        horizon_days: Forecast horizon used, carried for lead-time reporting.
        exclusion: Why this row is unusable, if it is.
        needs_review: True when the label rests on a noisy signal.
    """

    package: str
    as_of: datetime
    is_abandoned: bool
    event_date: datetime | None
    source: LabelSource
    horizon_days: int = HORIZON_DAYS
    exclusion: ExclusionReason = ExclusionReason.NONE
    needs_review: bool = False

    @property
    def usable(self) -> bool:
        """Whether this row belongs in the dataset."""
        return self.exclusion is ExclusionReason.NONE

    @property
    def days_since_last_commit_at_prediction(self) -> int | None:
        """How long the package had been quiet when the forecast was made.

        A diagnostic, not a performance metric — always negative relative to
        `as_of` because the last commit necessarily precedes the prediction.
        """
        if self.event_date is None:
            return None
        return (self.as_of - self.event_date).days

    @property
    def lead_time_days(self) -> int | None:
        """Days of warning the forecast provides before silence is confirmed.

        Under silence labelling this equals the horizon by construction: the
        verdict window opens `horizon_days` after the prediction, so a correct
        positive is always exactly that far ahead. Reported for clarity rather
        than as a distribution to optimise — the earlier archival label made
        this vary, and its median was the honest thing to track; here the
        constant is the honest thing to state.
        """
        return self.horizon_days if self.is_abandoned else None


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp into a naive UTC datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def commit_timestamps(commits: list[dict[str, Any]]) -> list[datetime]:
    """Extract authored timestamps from raw commit payloads, ascending."""
    stamps = []
    for commit in commits:
        raw = commit.get("commit", {}).get("author", {}).get("date")
        parsed = _parse_timestamp(raw)
        if parsed is not None:
            stamps.append(parsed)
    return sorted(stamps)


def detect_archive_event(repo: dict[str, Any]) -> datetime | None:
    """When a repository was archived, if it was.

    Retained as a corroborating signal and for reporting. No longer the primary
    label — see the module docstring for why.
    """
    if not repo.get("archived"):
        return None
    return _parse_timestamp(repo.get("archived_at")) or _parse_timestamp(repo.get("pushed_at"))


def has_unmaintained_notice(readme_text: str | None) -> bool:
    """Whether a README declares the project unmaintained."""
    return bool(readme_text and _UNMAINTAINED_RE.search(readme_text))


def build_label(
    package: str,
    as_of: datetime,
    commits: list[dict[str, Any]],
    repo: dict[str, Any] | None = None,
    data_collected_at: datetime | None = None,
    horizon_days: int = HORIZON_DAYS,
    silence_days: int = SUSTAINED_SILENCE_DAYS,
    active_window_days: int = ACTIVE_WINDOW_DAYS,
) -> AbandonmentLabel:
    """Label one package at one prediction date.

    Unlike the feature code, this function *may* look past `as_of` — observing
    the future is what makes it a label. The cutoff discipline applies to
    features only.

    Args:
        package: Registry name.
        as_of: The prediction date.
        commits: Raw commit payloads spanning before and after `as_of`.
        repo: Repository payload, used for archive corroboration.
        data_collected_at: When collection ran, for the censoring check.
            Defaults to now.
        horizon_days: Grace period before the verdict window opens.
        silence_days: How long silence must persist inside the verdict window.
        active_window_days: Lookback used to decide the package was alive.

    Returns:
        The label, carrying an exclusion reason when the row is unusable.
    """
    collected_at = data_collected_at or datetime.utcnow()
    stamps = commit_timestamps(commits)

    verdict_start = as_of + timedelta(days=horizon_days)
    verdict_end = verdict_start + timedelta(days=silence_days)

    # Cannot judge silence we have not had the chance to observe.
    if verdict_end > collected_at:
        return AbandonmentLabel(
            package=package,
            as_of=as_of,
            is_abandoned=False,
            event_date=None,
            source=LabelSource.ACTIVE,
            horizon_days=horizon_days,
            exclusion=ExclusionReason.CENSORED,
        )

    active_start = as_of - timedelta(days=active_window_days)
    commits_before = [t for t in stamps if active_start <= t < as_of]

    # Already silent at prediction time — nothing left to forecast.
    if not commits_before:
        return AbandonmentLabel(
            package=package,
            as_of=as_of,
            is_abandoned=False,
            event_date=None,
            source=LabelSource.SILENCE,
            horizon_days=horizon_days,
            exclusion=ExclusionReason.NOT_ALIVE,
        )

    commits_in_verdict = [t for t in stamps if verdict_start <= t <= verdict_end]
    went_silent = not commits_in_verdict

    archived_at = detect_archive_event(repo or {})
    archived_in_window = archived_at is not None and as_of < archived_at <= verdict_end

    source = LabelSource.SILENCE if went_silent else LabelSource.ACTIVE
    if went_silent and archived_in_window:
        source = LabelSource.ARCHIVED

    return AbandonmentLabel(
        package=package,
        as_of=as_of,
        is_abandoned=went_silent,
        event_date=max(commits_before) if went_silent else None,
        source=source,
        horizon_days=horizon_days,
    )

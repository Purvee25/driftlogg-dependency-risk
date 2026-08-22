"""Tests for the abandonment label definition.

The label is now sustained silence rather than the `archived` flag. See the
module docstring in driftlogg/labels.py for why that changed — the short
version is that archiving happens long after a package actually dies, so it
was unforecastable and it mislabelled dead-but-unarchived packages as alive.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from driftlogg.labels import (
    ACTIVE_WINDOW_DAYS,
    HORIZON_DAYS,
    SUSTAINED_SILENCE_DAYS,
    ExclusionReason,
    LabelSource,
    build_label,
    commit_timestamps,
    detect_archive_event,
    has_unmaintained_notice,
)

AS_OF = datetime(2024, 6, 1)
COLLECTED_AT = datetime(2026, 8, 22)
"""Far enough past AS_OF that nothing is censored unless a test intends it."""


def commit(when: datetime) -> dict:
    """Build a commit payload shaped like the GitHub API response."""
    return {"commit": {"author": {"date": when.isoformat() + "Z"}}}


def alive_commits(as_of: datetime = AS_OF) -> list[dict]:
    """Commits that make a package count as alive at `as_of`."""
    return [commit(as_of - timedelta(days=d)) for d in (10, 40, 100)]


def label_for(commits: list[dict], **kwargs) -> object:
    """Build a label with the standard test settings."""
    params = {
        "package": "pkg",
        "as_of": AS_OF,
        "commits": commits,
        "repo": {},
        "data_collected_at": COLLECTED_AT,
    }
    params.update(kwargs)
    return build_label(**params)


class TestAliveness:
    def test_package_silent_at_prediction_time_is_excluded(self):
        """Forecasting the death of something already dead is not forecasting."""
        old = [commit(AS_OF - timedelta(days=500))]

        label = label_for(old)

        assert label.usable is False
        assert label.exclusion is ExclusionReason.NOT_ALIVE

    def test_package_with_no_commits_at_all_is_excluded(self):
        label = label_for([])

        assert label.usable is False
        assert label.exclusion is ExclusionReason.NOT_ALIVE

    def test_recently_active_package_is_usable(self):
        label = label_for(alive_commits())

        assert label.usable is True

    def test_commit_just_inside_active_window_counts_as_alive(self):
        edge = [commit(AS_OF - timedelta(days=ACTIVE_WINDOW_DAYS - 1))]

        assert label_for(edge).usable is True

    def test_commit_just_outside_active_window_does_not(self):
        edge = [commit(AS_OF - timedelta(days=ACTIVE_WINDOW_DAYS + 1))]

        assert label_for(edge).usable is False


class TestSilenceLabel:
    def test_silence_through_the_verdict_window_is_positive(self):
        label = label_for(alive_commits())

        assert label.is_abandoned is True
        assert label.source is LabelSource.SILENCE

    def test_commits_during_the_verdict_window_are_negative(self):
        active = alive_commits() + [
            commit(AS_OF + timedelta(days=HORIZON_DAYS + 30)),
            commit(AS_OF + timedelta(days=HORIZON_DAYS + 90)),
        ]

        label = label_for(active)

        assert label.is_abandoned is False
        assert label.source is LabelSource.ACTIVE

    def test_commits_inside_the_grace_period_do_not_prevent_a_positive(self):
        """The horizon is a grace period — only the verdict window decides."""
        commits = alive_commits() + [commit(AS_OF + timedelta(days=HORIZON_DAYS - 10))]

        label = label_for(commits)

        assert label.is_abandoned is True

    def test_a_single_commit_in_the_verdict_window_breaks_silence(self):
        verdict_middle = AS_OF + timedelta(days=HORIZON_DAYS + SUSTAINED_SILENCE_DAYS // 2)
        commits = alive_commits() + [commit(verdict_middle)]

        assert label_for(commits).is_abandoned is False

    def test_commits_after_the_verdict_window_do_not_matter(self):
        """A package that revives much later still went silent when it did."""
        revived = alive_commits() + [
            commit(AS_OF + timedelta(days=HORIZON_DAYS + SUSTAINED_SILENCE_DAYS + 60))
        ]

        assert label_for(revived).is_abandoned is True


class TestCensoring:
    def test_verdict_window_past_collection_date_is_excluded(self):
        """Silence cannot be judged before there was time to observe it."""
        recent_collection = AS_OF + timedelta(days=HORIZON_DAYS + 10)

        label = label_for(alive_commits(), data_collected_at=recent_collection)

        assert label.usable is False
        assert label.exclusion is ExclusionReason.CENSORED

    def test_verdict_window_exactly_covered_is_usable(self):
        just_enough = AS_OF + timedelta(days=HORIZON_DAYS + SUSTAINED_SILENCE_DAYS)

        label = label_for(alive_commits(), data_collected_at=just_enough)

        assert label.usable is True


class TestArchiveCorroboration:
    def test_archived_during_window_is_recorded_as_the_source(self):
        repo = {
            "archived": True,
            "archived_at": (AS_OF + timedelta(days=100)).isoformat() + "Z",
        }

        label = label_for(alive_commits(), repo=repo)

        assert label.is_abandoned is True
        assert label.source is LabelSource.ARCHIVED

    def test_archiving_does_not_override_continued_activity(self):
        """Commits in the verdict window mean the package is alive, flag or not."""
        repo = {
            "archived": True,
            "archived_at": (AS_OF + timedelta(days=100)).isoformat() + "Z",
        }
        commits = alive_commits() + [commit(AS_OF + timedelta(days=HORIZON_DAYS + 30))]

        label = label_for(commits, repo=repo)

        assert label.is_abandoned is False

    def test_detect_archive_event_falls_back_to_pushed_at(self):
        pushed = AS_OF + timedelta(days=10)
        repo = {"archived": True, "archived_at": None, "pushed_at": pushed.isoformat() + "Z"}

        assert detect_archive_event(repo) == pushed

    def test_detect_archive_event_returns_none_for_live_repo(self):
        assert detect_archive_event({"archived": False}) is None


class TestQuietButHealthyPackages:
    """The trap the label has to avoid in both directions.

    A finished library that goes quiet for a quarter is not abandoned, and a
    dead package nobody archived is not alive. Sustained silence separates the
    two where the archive flag could not.
    """

    def test_a_quiet_quarter_is_not_abandonment(self):
        commits = alive_commits() + [
            commit(AS_OF + timedelta(days=HORIZON_DAYS + SUSTAINED_SILENCE_DAYS - 10))
        ]

        assert label_for(commits).is_abandoned is False

    def test_dead_but_never_archived_is_still_positive(self):
        """The failure mode that motivated the rewrite."""
        label = label_for(alive_commits(), repo={"archived": False})

        assert label.is_abandoned is True


class TestHelpers:
    def test_commit_timestamps_are_sorted_ascending(self):
        commits = [
            commit(AS_OF),
            commit(AS_OF - timedelta(days=100)),
            commit(AS_OF - timedelta(days=50)),
        ]

        stamps = commit_timestamps(commits)

        assert stamps == sorted(stamps)
        assert len(stamps) == 3

    def test_commit_timestamps_skip_unparseable_entries(self):
        commits = [commit(AS_OF), {"commit": {"author": {"date": "garbage"}}}, {}]

        assert len(commit_timestamps(commits)) == 1

    def test_unmaintained_notice_detection(self):
        assert has_unmaintained_notice("This project is no longer maintained.")
        assert has_unmaintained_notice("DEPRECATED — use foo instead")
        assert not has_unmaintained_notice("A well maintained library.")
        assert not has_unmaintained_notice(None)


class TestLeadTime:
    """Lead time means warning given, not time since the last commit.

    The first version returned `event_date - as_of`, which is negative by
    construction because the last commit always precedes the prediction. That
    surfaced in the training report as "median lead -186 days", a number with
    no useful meaning.
    """

    def test_lead_time_equals_the_forecast_horizon(self):
        label = label_for(alive_commits())

        assert label.lead_time_days == HORIZON_DAYS

    def test_lead_time_follows_a_custom_horizon(self):
        label = label_for(alive_commits(), horizon_days=30)

        assert label.lead_time_days == 30

    def test_lead_time_is_never_negative(self):
        label = label_for(alive_commits())

        assert label.lead_time_days > 0

    def test_active_packages_have_no_lead_time(self):
        commits = alive_commits() + [commit(AS_OF + timedelta(days=HORIZON_DAYS + 30))]

        assert label_for(commits).lead_time_days is None

    def test_quiet_duration_is_available_separately_as_a_diagnostic(self):
        label = label_for(alive_commits())

        assert label.days_since_last_commit_at_prediction == 10

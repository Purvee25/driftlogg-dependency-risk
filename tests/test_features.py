"""Tests for leakage-safe feature extraction.

The leakage tests are the important ones. Everything else in this project can
be wrong and you will notice; leakage is silent and inflates your scores until
someone asks how the model performs in production.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from driftlogg.features import (
    NEVER_OBSERVED_DAYS,
    FeatureWindow,
    LeakageError,
    build_features,
    compute_commit_features,
    parse_timestamp,
)

AS_OF = datetime(2025, 6, 1)


def make_commit(when: datetime, author: str = "alice") -> dict:
    """Build a commit payload shaped like the GitHub API response."""
    return {
        "commit": {"author": {"date": when.isoformat() + "Z"}},
        "author": {"login": author},
    }


def make_issue(created: datetime, state: str = "open", closed: datetime | None = None) -> dict:
    """Build an issue payload shaped like the GitHub API response."""
    return {
        "created_at": created.isoformat() + "Z",
        "closed_at": closed.isoformat() + "Z" if closed else None,
        "state": state,
    }


class TestFeatureWindow:
    def test_filter_excludes_events_at_or_after_cutoff(self):
        window = FeatureWindow(as_of=AS_OF)
        events = [
            {"created_at": (AS_OF - timedelta(days=10)).isoformat() + "Z"},
            {"created_at": AS_OF.isoformat() + "Z"},  # exactly at cutoff — excluded
            {"created_at": (AS_OF + timedelta(days=5)).isoformat() + "Z"},  # future
        ]

        kept = window.filter_events(events, "created_at")

        assert len(kept) == 1

    def test_filter_excludes_events_before_window_start(self):
        window = FeatureWindow(as_of=AS_OF, trailing_days=30)
        events = [
            {"created_at": (AS_OF - timedelta(days=10)).isoformat() + "Z"},
            {"created_at": (AS_OF - timedelta(days=400)).isoformat() + "Z"},
        ]

        kept = window.filter_events(events, "created_at")

        assert len(kept) == 1

    def test_assert_no_leakage_raises_on_future_timestamp(self):
        window = FeatureWindow(as_of=AS_OF)

        with pytest.raises(LeakageError, match="at or after cutoff"):
            window.assert_no_leakage([AS_OF - timedelta(days=1), AS_OF + timedelta(days=1)])

    def test_assert_no_leakage_passes_on_clean_data(self):
        window = FeatureWindow(as_of=AS_OF)
        window.assert_no_leakage([AS_OF - timedelta(days=1), AS_OF - timedelta(days=100)])


class TestCommitFeatures:
    def test_future_commits_do_not_affect_features(self):
        """The whole point: adding future commits must change nothing."""
        from driftlogg.features import PackageFeatures

        past = [make_commit(AS_OF - timedelta(days=d)) for d in (5, 20, 100)]
        future = [make_commit(AS_OF + timedelta(days=d)) for d in (1, 30)]
        window = FeatureWindow(as_of=AS_OF)

        clean = PackageFeatures(package="p", as_of=AS_OF)
        compute_commit_features(past, window, clean)

        contaminated = PackageFeatures(package="p", as_of=AS_OF)
        compute_commit_features(past + future, window, contaminated)

        assert clean.commits_trailing == contaminated.commits_trailing
        assert clean.days_since_last_commit == contaminated.days_since_last_commit
        assert clean.commit_velocity_ratio == contaminated.commit_velocity_ratio

    def test_velocity_ratio_below_one_when_slowing(self):
        from driftlogg.features import PackageFeatures

        # Busy nine months ago, silent since — a decaying package.
        commits = [make_commit(AS_OF - timedelta(days=d)) for d in range(200, 300)]
        features = PackageFeatures(package="p", as_of=AS_OF)

        compute_commit_features(commits, FeatureWindow(as_of=AS_OF), features)

        assert features.commit_velocity_ratio < 1.0
        assert features.commits_recent == 0

    def test_bus_factor_high_when_single_maintainer(self):
        from driftlogg.features import PackageFeatures

        commits = [make_commit(AS_OF - timedelta(days=d), author="solo") for d in range(1, 30)]
        features = PackageFeatures(package="p", as_of=AS_OF)

        compute_commit_features(commits, FeatureWindow(as_of=AS_OF), features)

        assert features.bus_factor_ratio == pytest.approx(1.0)
        assert features.active_contributors_trailing == 1

    def test_no_commits_yields_never_observed_sentinel(self):
        from driftlogg.features import PackageFeatures

        features = PackageFeatures(package="p", as_of=AS_OF)

        compute_commit_features([], FeatureWindow(as_of=AS_OF), features)

        assert features.commits_trailing == 0
        assert features.days_since_last_commit == NEVER_OBSERVED_DAYS


class TestBuildFeatures:
    def test_end_to_end_ignores_future_data(self):
        repo = {"stargazers_count": 100, "forks_count": 10, "open_issues_count": 5}
        commits = [make_commit(AS_OF - timedelta(days=10))]
        issues = [make_issue(AS_OF - timedelta(days=20))]
        releases = [{"published_at": (AS_OF - timedelta(days=50)).isoformat() + "Z"}]

        future_commits = commits + [make_commit(AS_OF + timedelta(days=10))]
        future_releases = releases + [
            {"published_at": (AS_OF + timedelta(days=10)).isoformat() + "Z"}
        ]

        clean = build_features("p", AS_OF, repo, commits, issues, releases)
        contaminated = build_features("p", AS_OF, repo, future_commits, issues, future_releases)

        assert clean.to_row() == contaminated.to_row()

    def test_pull_requests_excluded_from_issue_counts(self):
        repo = {"stargazers_count": 1, "forks_count": 1, "open_issues_count": 1}
        issue = make_issue(AS_OF - timedelta(days=5))
        pull_request = {**make_issue(AS_OF - timedelta(days=5)), "pull_request": {"url": "..."}}

        features = build_features("p", AS_OF, repo, [], [issue, pull_request], [])

        assert features.issues_opened_trailing == 1


class TestDataFrameCompatibility:
    """Regression tests for features reaching pandas and the models.

    A package with no releases and no recorded response times used to emit
    infinities into a single-row frame, which crashed `DataFrame.replace`
    inside pandas' block manager — surfacing as a bare "IndexError: pop index
    out of range" with nothing in the traceback pointing here.
    """

    def test_package_with_no_releases_survives_dataframe_round_trip(self):
        import pandas as pd

        repo = {"stargazers_count": 10, "forks_count": 2, "open_issues_count": 1}
        features = build_features("p", AS_OF, repo, [], [], releases=[])

        frame = pd.DataFrame([features.to_row()])

        assert len(frame) == 1
        assert frame["days_since_last_release"].iloc[0] == NEVER_OBSERVED_DAYS

    def test_no_infinities_reach_the_frame(self):
        import numpy as np
        import pandas as pd

        repo = {"stargazers_count": 0, "forks_count": 0, "open_issues_count": 0}
        features = build_features("p", AS_OF, repo, [], [], [])

        frame = pd.DataFrame([features.to_row()])
        numeric = frame.select_dtypes(include=[np.number])

        assert not np.isinf(numeric.to_numpy(dtype=float)).any()

    def test_baseline_flags_a_package_that_never_committed(self):
        """`nan >= 180` is False, so a NaN sentinel would read as healthy."""
        import pandas as pd

        from driftlogg.model import InactivityBaseline

        repo = {"stargazers_count": 0, "forks_count": 0, "open_issues_count": 0}
        features = build_features("p", AS_OF, repo, [], [], [])
        frame = pd.DataFrame([features.to_row()])

        score = InactivityBaseline().predict_proba(frame)[:, 1][0]

        assert score == 1.0


class TestParseTimestamp:
    @pytest.mark.parametrize("value", [None, "", "not-a-date", 12345])
    def test_returns_none_on_bad_input(self, value):
        assert parse_timestamp(value) is None

    def test_parses_github_z_suffix(self):
        assert parse_timestamp("2025-06-01T12:00:00Z") == datetime(2025, 6, 1, 12, 0, 0)

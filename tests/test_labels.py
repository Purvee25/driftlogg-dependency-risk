"""Tests for the abandonment label definition."""

from __future__ import annotations

from datetime import datetime, timedelta

from driftlogg.labels import (
    LabelSource,
    build_label,
    detect_abandonment_event,
    is_already_dead,
)

AS_OF = datetime(2025, 6, 1)


class TestDetectAbandonmentEvent:
    def test_archived_repo_uses_archived_at(self):
        archived_at = AS_OF + timedelta(days=30)
        repo = {"archived": True, "archived_at": archived_at.isoformat() + "Z"}

        event_date, source = detect_abandonment_event(repo)

        assert source is LabelSource.ARCHIVED
        assert event_date == archived_at

    def test_archived_repo_falls_back_to_pushed_at(self):
        """Older archived repos have no archived_at; pushed_at is the proxy."""
        pushed_at = AS_OF + timedelta(days=10)
        repo = {"archived": True, "archived_at": None, "pushed_at": pushed_at.isoformat() + "Z"}

        event_date, source = detect_abandonment_event(repo)

        assert source is LabelSource.ARCHIVED
        assert event_date == pushed_at

    def test_readme_notice_is_flagged_but_undated(self):
        repo = {"archived": False}
        readme = "# my-lib\n\nThis project is no longer maintained. Use foo instead."

        event_date, source = detect_abandonment_event(repo, readme)

        assert source is LabelSource.README_NOTICE
        assert event_date is None, "README notices carry no timestamp"

    def test_healthy_repo_returns_none(self):
        repo = {"archived": False}

        event_date, source = detect_abandonment_event(repo, "# my-lib\n\nA great library.")

        assert source is LabelSource.NONE
        assert event_date is None


class TestBuildLabel:
    def test_positive_when_archived_inside_horizon(self):
        repo = {"archived": True, "archived_at": (AS_OF + timedelta(days=45)).isoformat() + "Z"}

        label = build_label("pkg", AS_OF, repo, horizon_days=90)

        assert label.is_abandoned is True
        assert label.lead_time_days == 45

    def test_negative_when_archived_beyond_horizon(self):
        """Died eventually, but not inside the window we predict for."""
        repo = {"archived": True, "archived_at": (AS_OF + timedelta(days=200)).isoformat() + "Z"}

        label = build_label("pkg", AS_OF, repo, horizon_days=90)

        assert label.is_abandoned is False

    def test_negative_when_already_dead_before_cutoff(self):
        repo = {"archived": True, "archived_at": (AS_OF - timedelta(days=10)).isoformat() + "Z"}

        label = build_label("pkg", AS_OF, repo, horizon_days=90)

        assert label.is_abandoned is False
        assert is_already_dead(label) is True, "must be dropped from the dataset upstream"

    def test_healthy_package_is_negative(self):
        label = build_label("pkg", AS_OF, {"archived": False})

        assert label.is_abandoned is False
        assert label.source is LabelSource.NONE
        assert label.needs_review is False

    def test_readme_notice_marked_for_review(self):
        label = build_label("pkg", AS_OF, {"archived": False}, readme_text="unmaintained, sorry")

        assert label.needs_review is True


class TestQuietButHealthyPackages:
    """The trap this project has to avoid.

    A finished, stable library with no recent commits is not abandoned. If these
    end up labelled positive, the model learns to flag every mature package and
    the tool becomes noise.
    """

    def test_quiet_stable_package_is_not_abandoned(self):
        repo = {
            "archived": False,
            "pushed_at": (AS_OF - timedelta(days=400)).isoformat() + "Z",
        }

        label = build_label("left-pad-ish", AS_OF, repo)

        assert label.is_abandoned is False, "silence alone is not death"

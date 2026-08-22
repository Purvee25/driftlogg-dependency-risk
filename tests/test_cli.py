"""Tests for the CI gate.

Scoring is stubbed throughout — the CLI's job is parsing, diffing, formatting,
and choosing an exit code. Whether the model is any good is measured elsewhere.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from driftlogg.api.schemas import FeatureContribution, PackageRisk, RiskBand
from driftlogg.cli import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_RISK_FOUND,
    build_parser,
    find_line_number,
    main,
)


def risk(package: str, score: float | None, error: str | None = None) -> PackageRisk:
    """Build a scored result."""
    return PackageRisk(
        package=package,
        repo=f"owner/{package}",
        score=score,
        band=RiskBand.from_score(score),
        reasons=(
            [
                FeatureContribution(
                    feature="days_since_last_commit", value=400.0, label="No commits in 400 days"
                )
            ]
            if score and score >= 0.6
            else []
        ),
        error=error,
    )


@pytest.fixture
def manifest(tmp_path):
    """A package.json with three dependencies."""
    path = tmp_path / "package.json"
    path.write_text(
        json.dumps(
            {"name": "app", "dependencies": {"express": "^4", "dead-pkg": "^1", "lodash": "^4"}},
            indent=2,
        )
    )
    return path


def run_with_scores(manifest, scores: dict[str, float | None], *extra: str) -> int:
    """Invoke the CLI with scoring stubbed out."""
    results = [risk(name, score) for name, score in scores.items()]

    with patch("driftlogg.cli.ScoringService") as service_cls:
        service = service_cls.return_value
        service.model_loaded = True
        service.model_kind = "lightgbm"
        service.score_packages.return_value = results
        return main(["check", str(manifest), *extra])


class TestExitCodes:
    def test_clean_manifest_exits_zero(self, manifest):
        code = run_with_scores(manifest, {"express": 0.05, "lodash": 0.10})

        assert code == EXIT_OK

    def test_risky_dependency_fails_the_build(self, manifest):
        code = run_with_scores(manifest, {"express": 0.05, "dead-pkg": 0.95})

        assert code == EXIT_RISK_FOUND

    def test_warn_only_never_fails(self, manifest):
        code = run_with_scores(manifest, {"dead-pkg": 0.99}, "--warn-only")

        assert code == EXIT_OK

    def test_score_below_threshold_passes(self, manifest):
        code = run_with_scores(manifest, {"dead-pkg": 0.55})

        assert code == EXIT_OK

    def test_threshold_is_inclusive(self, manifest):
        code = run_with_scores(manifest, {"dead-pkg": 0.60})

        assert code == EXIT_RISK_FOUND

    def test_custom_threshold_is_honoured(self, manifest):
        code = run_with_scores(manifest, {"dead-pkg": 0.35}, "--threshold", "0.3")

        assert code == EXIT_RISK_FOUND

    def test_unscorable_dependencies_do_not_fail_the_build(self, manifest):
        """A package we could not resolve is unknown, not risky."""
        results = [risk("mystery", None, error="No GitHub repository found")]

        with patch("driftlogg.cli.ScoringService") as service_cls:
            service_cls.return_value.model_loaded = True
            service_cls.return_value.score_packages.return_value = results
            code = main(["check", str(manifest)])

        assert code == EXIT_OK

    def test_missing_manifest_is_a_usage_error(self, tmp_path):
        assert main(["check", str(tmp_path / "nope.json")]) == EXIT_ERROR

    def test_unparseable_manifest_is_a_usage_error(self, tmp_path):
        bad = tmp_path / "package.json"
        bad.write_text("{ not json")

        assert main(["check", str(bad)]) == EXIT_ERROR


class TestChangedOnly:
    def test_only_added_dependencies_are_scored(self, manifest):
        """Pre-existing dead dependencies must not fail every future PR."""
        with (
            patch("driftlogg.cli.ScoringService") as service_cls,
            patch("driftlogg.cli.changed_dependencies", return_value={"express", "lodash"}),
        ):
            service_cls.return_value.model_loaded = True
            service_cls.return_value.score_packages.return_value = [risk("dead-pkg", 0.9)]

            main(["check", str(manifest), "--changed-only"])

            scored = service_cls.return_value.score_packages.call_args[0][0]

        assert scored == ["dead-pkg"]

    def test_no_new_dependencies_exits_early(self, manifest):
        existing = {"express", "dead-pkg", "lodash"}

        with (
            patch("driftlogg.cli.ScoringService") as service_cls,
            patch("driftlogg.cli.changed_dependencies", return_value=existing),
        ):
            code = main(["check", str(manifest), "--changed-only"])

            service_cls.return_value.score_packages.assert_not_called()

        assert code == EXIT_OK

    def test_unavailable_base_ref_scores_everything(self, manifest):
        """A new manifest has no baseline, so every dependency counts as added."""
        with (
            patch("driftlogg.cli.ScoringService") as service_cls,
            patch("driftlogg.cli.changed_dependencies", return_value=None),
        ):
            service_cls.return_value.model_loaded = True
            service_cls.return_value.score_packages.return_value = []

            main(["check", str(manifest), "--changed-only"])

            scored = service_cls.return_value.score_packages.call_args[0][0]

        assert len(scored) == 3


class TestOutputFormats:
    def test_github_format_emits_annotations(self, manifest, capsys):
        run_with_scores(manifest, {"dead-pkg": 0.95}, "--format", "github", "--warn-only")

        out = capsys.readouterr().out

        assert "::error" in out
        assert "file=" in out
        assert "dead-pkg" in out

    def test_elevated_risk_annotates_as_warning(self, manifest, capsys):
        run_with_scores(
            manifest, {"dead-pkg": 0.45}, "--format", "github", "--threshold", "0.3", "--warn-only"
        )

        out = capsys.readouterr().out

        assert "::warning" in out
        assert "::error" not in out

    def test_json_format_is_machine_readable(self, manifest, capsys):
        run_with_scores(manifest, {"dead-pkg": 0.95}, "--format", "json", "--warn-only")

        payload = json.loads(capsys.readouterr().out)

        assert payload["threshold"] == pytest.approx(0.60)
        assert payload["results"][0]["package"] == "dead-pkg"

    def test_text_format_lists_flagged_packages(self, manifest, capsys):
        run_with_scores(manifest, {"dead-pkg": 0.95}, "--warn-only")

        out = capsys.readouterr().out

        assert "dead-pkg" in out
        assert "95%" in out


class TestFindLineNumber:
    def test_locates_a_package_json_dependency(self, manifest):
        line = find_line_number(manifest, "dead-pkg")

        assert line is not None
        assert "dead-pkg" in manifest.read_text().splitlines()[line - 1]

    def test_locates_a_requirements_entry(self, tmp_path):
        path = tmp_path / "requirements.txt"
        path.write_text("# comment\nrequests==2.0\nflask>=3.0\n")

        assert find_line_number(path, "requests") == 2

    def test_returns_none_for_absent_package(self, manifest):
        assert find_line_number(manifest, "not-here") is None

    def test_returns_none_for_unreadable_file(self, tmp_path):
        assert find_line_number(tmp_path / "missing.json", "x") is None


class TestParser:
    def test_check_requires_a_manifest(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["check"])

    def test_defaults_are_sensible(self):
        args = build_parser().parse_args(["check", "package.json"])

        assert args.threshold == pytest.approx(0.60)
        assert args.format == "text"
        assert args.changed_only is False
        assert args.warn_only is False

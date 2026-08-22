"""Tests for manifest parsing, repo resolution, and the API surface."""

from __future__ import annotations

import json

import pytest

from driftlogg.api.manifests import (
    ManifestKind,
    ManifestParseError,
    detect_kind,
    parse_manifest,
    parse_package_json,
    parse_requirements_txt,
)
from driftlogg.api.schemas import RiskBand
from driftlogg.collect.registry import extract_github_repo, repo_from_package_metadata


class TestPackageJson:
    def test_extracts_both_dependency_sections(self):
        content = json.dumps(
            {
                "name": "app",
                "dependencies": {"express": "^4.0.0", "lodash": "~4.17.0"},
                "devDependencies": {"jest": "^29.0.0"},
            }
        )

        assert parse_package_json(content) == ["express", "lodash", "jest"]

    def test_can_exclude_dev_dependencies(self):
        content = json.dumps(
            {"dependencies": {"express": "^4"}, "devDependencies": {"jest": "^29"}}
        )

        assert parse_package_json(content, include_dev=False) == ["express"]

    def test_handles_missing_sections(self):
        assert parse_package_json(json.dumps({"name": "app"})) == []

    def test_preserves_scoped_names(self):
        content = json.dumps({"dependencies": {"@babel/core": "^7.0.0"}})

        assert parse_package_json(content) == ["@babel/core"]

    def test_raises_on_malformed_json(self):
        with pytest.raises(ManifestParseError, match="Invalid JSON"):
            parse_package_json("{not json")

    def test_raises_when_not_an_object(self):
        with pytest.raises(ManifestParseError):
            parse_package_json("[1, 2, 3]")


class TestRequirementsTxt:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("requests==2.31.0", "requests"),
            ("requests>=2.0", "requests"),
            ("requests", "requests"),
            ("requests[security]==2.31.0", "requests"),
            ("scikit-learn~=1.5", "scikit-learn"),
            ("pandas != 2.0", "pandas"),
            ('foo==1.0 ; python_version < "3.11"', "foo"),
        ],
    )
    def test_strips_version_specifiers(self, line, expected):
        assert parse_requirements_txt(line) == [expected]

    def test_skips_comments_and_blanks(self):
        content = "# a comment\n\nrequests==2.0\n   \n# another\nflask"

        assert parse_requirements_txt(content) == ["requests", "flask"]

    def test_strips_inline_comments(self):
        assert parse_requirements_txt("requests==2.0  # pinned for CI") == ["requests"]

    def test_skips_flags_and_includes(self):
        content = "-r base.txt\n--index-url https://example.com\n-e .\nrequests"

        assert parse_requirements_txt(content) == ["requests"]

    def test_skips_urls_and_local_paths(self):
        content = "https://example.com/pkg.tar.gz\n./local-pkg\nrequests"

        assert parse_requirements_txt(content) == ["requests"]

    def test_deduplicates_preserving_order(self):
        assert parse_requirements_txt("b\na\nb") == ["b", "a"]


class TestDetectKind:
    def test_detects_by_filename(self):
        assert detect_kind("package.json", "{}") is ManifestKind.PACKAGE_JSON
        assert detect_kind("requirements.txt", "x") is ManifestKind.REQUIREMENTS_TXT
        assert detect_kind("requirements-dev.txt", "x") is ManifestKind.REQUIREMENTS_TXT

    def test_falls_back_to_content(self):
        assert detect_kind("unknown", '{"dependencies": {}}') is ManifestKind.PACKAGE_JSON

    def test_raises_when_undeterminable(self):
        with pytest.raises(ManifestParseError, match="Cannot determine"):
            detect_kind("mystery", "some text")


class TestParseManifest:
    def test_dispatches_to_the_right_parser(self):
        kind, names = parse_manifest("package.json", json.dumps({"dependencies": {"a": "1"}}))
        assert kind is ManifestKind.PACKAGE_JSON
        assert names == ["a"]

        kind, names = parse_manifest("requirements.txt", "b==1.0")
        assert kind is ManifestKind.REQUIREMENTS_TXT
        assert names == ["b"]


class TestRepoResolution:
    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/expressjs/express",
            "https://github.com/expressjs/express.git",
            "git+https://github.com/expressjs/express.git",
            "git://github.com/expressjs/express.git",
            "https://github.com/expressjs/express/",
        ],
    )
    def test_extracts_owner_and_repo_from_url_forms(self, url):
        assert extract_github_repo(url) == ("expressjs", "express")

    @pytest.mark.parametrize("url", [None, "", "https://gitlab.com/foo/bar", "not a url"])
    def test_returns_none_for_non_github(self, url):
        assert extract_github_repo(url) is None

    def test_reads_repository_object(self):
        metadata = {"repository": {"type": "git", "url": "https://github.com/a/b.git"}}

        assert repo_from_package_metadata(metadata) == ("a", "b")

    def test_reads_repository_string(self):
        assert repo_from_package_metadata({"repository": "https://github.com/a/b"}) == ("a", "b")

    def test_falls_back_to_homepage(self):
        metadata = {"repository": {"url": "https://gitlab.com/x/y"},
                    "homepage": "https://github.com/a/b"}

        assert repo_from_package_metadata(metadata) == ("a", "b")

    def test_returns_none_when_nothing_resolves(self):
        assert repo_from_package_metadata({"name": "orphan"}) is None


class TestRiskBand:
    @pytest.mark.parametrize(
        ("score", "band"),
        [
            (0.95, RiskBand.HIGH),
            (0.60, RiskBand.HIGH),
            (0.45, RiskBand.ELEVATED),
            (0.30, RiskBand.ELEVATED),
            (0.10, RiskBand.LOW),
            (0.0, RiskBand.LOW),
            (None, RiskBand.UNKNOWN),
        ],
    )
    def test_bands_are_assigned_at_the_documented_cutoffs(self, score, band):
        assert RiskBand.from_score(score) is band


class TestAPIEndpoints:
    """Exercises the HTTP surface without touching the network.

    Scoring itself is not tested here — it depends on live GitHub data, which
    belongs in an integration test rather than the unit suite.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from driftlogg.api.main import app

        with TestClient(app) as test_client:
            yield test_client

    def test_health_reports_model_state(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "model_kind" in body

    def test_dashboard_is_served(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "DriftLogg" in response.text

    def test_manifest_upload_rejects_unparseable_file(self, client):
        response = client.post(
            "/score/manifest",
            files={"file": ("mystery.dat", b"not a manifest", "text/plain")},
        )

        assert response.status_code == 400

    def test_manifest_upload_rejects_empty_dependency_list(self, client):
        response = client.post(
            "/score/manifest",
            files={"file": ("package.json", b'{"name":"x"}', "application/json")},
        )

        assert response.status_code == 400
        assert "No dependencies" in response.json()["detail"]

    def test_score_rejects_empty_package_list(self, client):
        response = client.post("/score", json={"packages": []})

        assert response.status_code == 422


class TestSignalDescriptions:
    """The sentinel is an implementation detail and must never reach the UI."""

    def test_never_observed_commits_reads_in_plain_language(self):
        from driftlogg.api.service import _describe_signal
        from driftlogg.features import NEVER_OBSERVED_DAYS

        label = _describe_signal("days_since_last_commit", NEVER_OBSERVED_DAYS)

        assert label is not None
        assert "99999" not in label
        assert "No commits" in label

    def test_never_observed_release_reads_in_plain_language(self):
        from driftlogg.api.service import _describe_signal
        from driftlogg.features import NEVER_OBSERVED_DAYS

        label = _describe_signal("days_since_last_release", NEVER_OBSERVED_DAYS)

        assert label is not None
        assert "99999" not in label
        assert "273" not in label, "must not render the sentinel as years either"

    def test_real_day_counts_still_render_normally(self):
        from driftlogg.api.service import _describe_signal

        assert "220 days" in _describe_signal("days_since_last_commit", 220.0)

    def test_healthy_values_produce_no_reason(self):
        from driftlogg.api.service import _describe_signal

        assert _describe_signal("days_since_last_commit", 5.0) is None
        assert _describe_signal("bus_factor_ratio", 0.4) is None

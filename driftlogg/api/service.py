"""Scoring service: resolve packages, build features, predict.

Scoring a package live is expensive — resolving it to a repo and pulling enough
history costs several API calls and a few seconds. Results are cached in-process
so repeated scores of the same dependency are free within a session.

The service degrades honestly rather than failing: with no trained model on
disk it falls back to the inactivity baseline and reports which one produced
each score, so the dashboard never implies more confidence than it has.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from driftlogg.api.schemas import FeatureContribution, PackageRisk, RiskBand
from driftlogg.collect import GitHubClient
from driftlogg.collect.registry import Ecosystem, resolver_for
from driftlogg.config import settings
from driftlogg.features import (
    FEATURE_COLUMNS,
    NEVER_OBSERVED_DAYS,
    TRAILING_WINDOW_DAYS,
    build_features,
)
from driftlogg.labels import DEFAULT_HORIZON_DAYS
from driftlogg.model import InactivityBaseline

logger = logging.getLogger(__name__)

MAX_REASONS = 3
"""How many contributing signals to surface per package."""

BASELINE_KIND = "baseline (inactivity)"
MODEL_KIND = "lightgbm"


def _describe_signal(feature: str, value: float | None) -> str | None:
    """Turn a feature value into a human-readable reason.

    Returns None when the feature is not currently a cause for concern — the
    dashboard should show why a package looks risky, not list every input.
    """
    if value is None or pd.isna(value):
        return None

    # NEVER_OBSERVED_DAYS means the event is absent from the lookback window,
    # not that it literally happened 99,999 days ago. Say so in plain terms
    # rather than printing the sentinel.
    never_observed = value >= NEVER_OBSERVED_DAYS
    window_years = TRAILING_WINDOW_DAYS / 365

    match feature:
        case "days_since_last_commit" if never_observed:
            return f"No commits at all in the past {window_years:.0f} year(s)"
        case "days_since_last_commit" if value > 180:
            return f"No commits in {value:.0f} days"
        case "commit_velocity_ratio" if value < 0.3:
            return f"Commit rate down to {value:.0%} of its yearly average"
        case "bus_factor_ratio" if value > 0.9:
            return "Effectively a single maintainer"
        case "open_close_ratio" if value > 2:
            return f"Issues opening {value:.1f}x faster than they close"
        case "days_since_last_release" if never_observed:
            return f"No release in the past {window_years:.0f} year(s)"
        case "days_since_last_release" if value > 365:
            return f"No release in {value / 365:.1f} years"
        case "stale_open_issues" if value > 10:
            return f"{value:.0f} issues open longer than 90 days"
        case "active_contributors_recent" if value == 0:
            return "No active contributors in the last 90 days"
        case _:
            return None


class ScoringService:
    """Scores packages for abandonment risk.

    Args:
        model_path: Where to look for a trained model. Falls back to settings.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model = None
        self._feature_columns = FEATURE_COLUMNS
        self._cache: dict[str, PackageRisk] = {}

        candidates = (Path(model_path),) if model_path else settings.model_search_paths
        for path in candidates:
            if self._load_model(path):
                return

        if settings.model_url and self._download_model():
            return

        logger.warning(
            "No trained model found in %s — falling back to the inactivity "
            "baseline. Run scripts/04_train.py to produce one.",
            ", ".join(str(p) for p in candidates),
        )

    def _download_model(self) -> bool:
        """Fetch the model from `settings.model_url`, caching it to disk first.

        Caching before loading means a second cold start on the same warm
        instance hits the local file rather than downloading again, and a
        corrupt or partial download never reaches `pickle.load`.

        Returns:
            True if the model downloaded and loaded successfully.
        """
        import httpx

        target = settings.data_dir / "model.pkl"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            response = httpx.get(settings.model_url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            target.write_bytes(response.content)
        except Exception:
            logger.warning("Could not download model from %s.", settings.model_url, exc_info=True)
            return False

        return self._load_model(target)

    def _load_model(self, path: Path) -> bool:
        """Attempt to load a pickled model bundle.

        Returns:
            True if the model loaded, False if the file was absent or unusable.
        """
        try:
            with open(path, "rb") as handle:
                bundle = pickle.load(handle)
        except FileNotFoundError:
            return False
        except Exception:
            # A model pickled by a different library version is unusable but
            # must not take the whole service down with it.
            logger.warning("Could not load model at %s; skipping.", path, exc_info=True)
            return False

        self._model = bundle["model"]
        self._feature_columns = bundle.get("features", FEATURE_COLUMNS)
        logger.info("Loaded trained model from %s", path)
        return True

    @property
    def model_kind(self) -> str:
        """Which model is currently answering."""
        return MODEL_KIND if self._model is not None else BASELINE_KIND

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def score_packages(
        self,
        names: list[str],
        as_of: datetime | None = None,
        ecosystem: Ecosystem = Ecosystem.NPM,
    ) -> list[PackageRisk]:
        """Score a list of package names.

        Args:
            names: Package names to score.
            as_of: Scoring date; defaults to now.
            ecosystem: Which registry to resolve names against. Getting this
                wrong resolves to an unrelated project of the same name — see
                the note in driftlogg/collect/registry.py.

        Returns:
            One result per input name, in the same order. Failures are returned
            as entries carrying an `error` rather than raising, so one bad
            dependency cannot sink an entire manifest.
        """
        as_of = as_of or datetime.utcnow()
        results: list[PackageRisk] = []

        # Cache keys carry the ecosystem: the same name means different
        # packages in different registries.
        with GitHubClient() as github, resolver_for(ecosystem) as registry:
            for name in names:
                key = f"{ecosystem.value}:{name}"
                if key in self._cache:
                    results.append(self._cache[key])
                    continue

                risk = self._score_one(name, as_of, github, registry)
                self._cache[key] = risk
                results.append(risk)

        return results

    def _score_one(
        self,
        name: str,
        as_of: datetime,
        github: GitHubClient,
        registry,
    ) -> PackageRisk:
        """Score a single package, converting any failure into a result."""
        try:
            resolved = registry.resolve_repo(name)
            if resolved is None:
                return PackageRisk(package=name, error="No GitHub repository found")

            owner, repo_name = resolved
            repo = github.get_repo(owner, repo_name)
            if repo is None:
                return PackageRisk(
                    package=name,
                    repo=f"{owner}/{repo_name}",
                    error="Repository no longer exists",
                )

            # Already archived: this is a fact, not a prediction.
            if repo.get("archived"):
                return PackageRisk(
                    package=name,
                    repo=f"{owner}/{repo_name}",
                    score=1.0,
                    band=RiskBand.HIGH,
                    reasons=[
                        FeatureContribution(
                            feature="archived",
                            value=1.0,
                            label="Already archived by its maintainer",
                        )
                    ],
                )

            features = build_features(
                package=name,
                as_of=as_of,
                repo=repo,
                commits=github.get_commits(owner, repo_name, max_pages=5),
                issues=github.get_issues(owner, repo_name, max_pages=3),
                releases=github.get_releases(owner, repo_name, max_pages=2),
            )

            # Features emit NEVER_OBSERVED_DAYS rather than inf, so no
            # infinity scrubbing is needed here.
            frame = pd.DataFrame([features.to_row()])
            score = self._predict(frame)

            return PackageRisk(
                package=name,
                repo=f"{owner}/{repo_name}",
                score=score,
                band=RiskBand.from_score(score),
                reasons=self._build_reasons(features.to_row()),
            )

        except Exception as exc:
            logger.exception("Scoring failed for %s", name)
            return PackageRisk(package=name, error=f"{type(exc).__name__}: {exc}")

    def _predict(self, frame: pd.DataFrame) -> float:
        """Run whichever model is loaded."""
        for column in self._feature_columns:
            if column not in frame:
                frame[column] = pd.NA

        if self._model is not None:
            return float(self._model.predict_proba(frame)[:, 1][0])
        return float(InactivityBaseline().predict_proba(frame)[:, 1][0])

    def _build_reasons(self, row: dict) -> list[FeatureContribution]:
        """Surface the strongest concerning signals for one package."""
        reasons: list[FeatureContribution] = []

        for feature in self._feature_columns:
            value = row.get(feature)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue

            label = _describe_signal(feature, float(value))
            if label:
                reasons.append(
                    FeatureContribution(feature=feature, value=float(value), label=label)
                )

        return reasons[:MAX_REASONS]


HORIZON_DAYS = DEFAULT_HORIZON_DAYS

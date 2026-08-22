"""Tests for popularity matching.

The point of matching is that popularity stops carrying marginal information
about the label. These tests assert that property directly rather than checking
row counts, because the row counts are incidental and the property is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftlogg.sampling import (
    POPULARITY_FEATURES,
    STAR_BINS,
    match_on_popularity,
    popularity_balance_report,
)


def confounded_frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Build a frame where popularity predicts the label purely by sampling.

    Mirrors the real defect: high-star rows are drawn mostly from a pool that
    is enriched with positives, so stars proxies for the pool rather than for
    anything about package health.
    """
    rng = np.random.default_rng(seed)
    high = rng.random(n) < 0.4

    stars = np.where(high, rng.integers(1_000, 20_000, n), rng.integers(0, 200, n))
    positive = np.where(
        high,
        rng.random(n) < 0.80,  # popular pool: mostly positive
        rng.random(n) < 0.20,  # unpopular pool: mostly negative
    )

    return pd.DataFrame(
        {
            "package": [f"pkg-{i}" for i in range(n)],
            "stars": stars,
            "days_since_last_commit": rng.normal(100, 50, n),
            "is_abandoned": positive,
        }
    )


class TestMatchOnPopularity:
    def test_every_bin_becomes_class_balanced(self):
        matched = match_on_popularity(confounded_frame())

        report = popularity_balance_report(matched)
        rates = report["positive_rate"].to_numpy()

        assert np.allclose(rates, 0.5), f"bins not balanced: {rates}"

    def test_confound_is_present_before_matching(self):
        """Guards the fixture: if this fails the test below proves nothing."""
        report = popularity_balance_report(confounded_frame())
        rates = report["positive_rate"].dropna().to_numpy()

        assert rates.max() - rates.min() > 0.3, "fixture is not actually confounded"

    def test_matching_is_deterministic(self):
        frame = confounded_frame()

        first = match_on_popularity(frame, seed=7)
        second = match_on_popularity(frame, seed=7)

        assert first.equals(second)

    def test_different_seeds_draw_different_subsets(self):
        frame = confounded_frame()

        first = match_on_popularity(frame, seed=1)
        second = match_on_popularity(frame, seed=2)

        assert not first.equals(second)

    def test_single_class_bins_are_dropped_but_others_survive(self):
        """A bin holding only one class contributes nothing and is skipped."""
        frame = pd.DataFrame(
            {
                # 0-10: positives only, unmatchable.
                "stars": [5, 6, 7, 5_000, 6_000, 5_500, 7_000],
                # 1000-5000 and 5000+: both classes present.
                "is_abandoned": [True, True, True, True, True, False, False],
            }
        )

        matched = match_on_popularity(frame)

        assert matched["is_abandoned"].nunique() == 2
        assert matched["stars"].min() >= 1_000, "unmatchable low bin should be gone"

    def test_raises_when_no_bin_can_be_matched(self):
        """Silently returning an empty frame would hide the problem."""
        frame = pd.DataFrame(
            {
                "stars": [5, 6, 7, 5_000, 6_000],
                "is_abandoned": [True, True, True, False, False],
            }
        )

        with pytest.raises(ValueError, match="removed every row"):
            match_on_popularity(frame)

    def test_overall_balance_is_fifty_fifty(self):
        matched = match_on_popularity(confounded_frame())

        assert matched["is_abandoned"].mean() == pytest.approx(0.5)

    def test_preserves_other_columns(self):
        matched = match_on_popularity(confounded_frame())

        assert "days_since_last_commit" in matched.columns
        assert "package" in matched.columns

    def test_helper_bin_column_is_not_leaked(self):
        matched = match_on_popularity(confounded_frame())

        assert "_bin" not in matched.columns

    def test_raises_on_missing_columns(self):
        with pytest.raises(ValueError, match="Missing column"):
            match_on_popularity(pd.DataFrame({"stars": [1, 2]}))

        with pytest.raises(ValueError, match="Missing column"):
            match_on_popularity(pd.DataFrame({"is_abandoned": [True, False]}))


class TestBalanceReport:
    def test_reports_one_row_per_occupied_bin(self):
        report = popularity_balance_report(confounded_frame())

        assert len(report) <= len(STAR_BINS) - 1
        assert {"rows", "positive_rate"} == set(report.columns)

    def test_rates_stay_within_range(self):
        report = popularity_balance_report(confounded_frame())

        assert report["positive_rate"].between(0, 1).all()


class TestPopularityFeatureList:
    def test_named_features_exist_in_the_feature_set(self):
        from driftlogg.features import FEATURE_COLUMNS

        for feature in POPULARITY_FEATURES:
            assert feature in FEATURE_COLUMNS

    def test_ablation_leaves_decay_features_intact(self):
        from driftlogg.features import FEATURE_COLUMNS

        remaining = [c for c in FEATURE_COLUMNS if c not in POPULARITY_FEATURES]

        assert "days_since_last_commit" in remaining
        assert "commit_velocity_ratio" in remaining
        assert "bus_factor_ratio" in remaining
        assert len(remaining) >= 8


class TestFeatureValidation:
    """The variance warning is a training-time check.

    At inference the frame holds one row, where every column is constant by
    definition — warning there printed all 18 feature names on every single
    CLI invocation.
    """

    def test_single_row_inference_is_silent(self, caplog):
        import pandas as pd

        from driftlogg.model import _validate_features

        frame = pd.DataFrame([{"a": 1.0, "b": 2.0}])

        with caplog.at_level("WARNING"):
            _validate_features(frame, ["a", "b"])

        assert "Constant features" not in caplog.text

    def test_training_warns_about_genuinely_constant_columns(self, caplog):
        import pandas as pd

        from driftlogg.model import _validate_features

        frame = pd.DataFrame({"varies": [1.0, 2.0, 3.0], "flat": [7.0, 7.0, 7.0]})

        with caplog.at_level("WARNING"):
            _validate_features(frame, ["varies", "flat"], check_variance=True)

        assert "flat" in caplog.text
        assert "varies" not in caplog.text.split("Constant features")[-1]

    def test_all_null_column_raises_with_a_useful_message(self):
        import pandas as pd
        import pytest as pt

        from driftlogg.model import _validate_features

        frame = pd.DataFrame({"good": [1.0, 2.0], "empty": [None, None]})

        with pt.raises(ValueError, match="entirely null"):
            _validate_features(frame, ["good", "empty"])

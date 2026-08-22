"""Baseline and gradient-boosted models, plus the temporal split.

Order of work: establish the baseline, write its score down, then train the
model. If the model cannot beat "no commits in six months", the added
complexity has not earned its place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd

from driftlogg.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

BASELINE_INACTIVITY_DAYS = 180
"""The rule the model has to beat: silent for six months means dying."""

RANDOM_SEED = 42


class Classifier(Protocol):
    """Minimal interface shared by the baseline and the trained model."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


@dataclass
class TemporalSplit:
    """Train/test split drawn on a date boundary, never at random.

    A random split lets the model learn from the future and score far higher
    than it ever would in production. Splitting on time is the only honest
    option for a forecasting problem.

    Attributes:
        train: Rows with `as_of` before the boundary.
        test: Rows with `as_of` on or after the boundary.
        boundary: The split date.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    boundary: datetime

    def describe(self) -> str:
        """One-line summary of split sizes and class balance."""
        return (
            f"boundary={self.boundary.date()} "
            f"train={len(self.train)} (pos={int(self.train['is_abandoned'].sum())}) "
            f"test={len(self.test)} (pos={int(self.test['is_abandoned'].sum())})"
        )


def split_by_date(frame: pd.DataFrame, boundary: datetime) -> TemporalSplit:
    """Split a labelled feature table on a date boundary.

    Args:
        frame: Table with an `as_of` column and an `is_abandoned` column.
        boundary: Rows before this go to train, the rest to test.

    Returns:
        The split.

    Raises:
        ValueError: If either side is empty or has no positive examples.
    """
    as_of = pd.to_datetime(frame["as_of"])
    train = frame[as_of < boundary].copy()
    test = frame[as_of >= boundary].copy()

    if train.empty or test.empty:
        raise ValueError(f"Boundary {boundary} leaves an empty split.")

    for name, part in (("train", train), ("test", test)):
        if part["is_abandoned"].sum() == 0:
            raise ValueError(f"No positive examples in {name}; move the boundary.")

    return TemporalSplit(train=train, test=test, boundary=boundary)


class InactivityBaseline:
    """Predicts abandonment from commit silence alone.

    Deliberately naive. Its job is to set the bar the real model must clear.
    """

    def __init__(self, threshold_days: int = BASELINE_INACTIVITY_DAYS) -> None:
        self.threshold_days = threshold_days

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return a hard 0/1 as a two-column probability array."""
        silent = features["days_since_last_commit"] >= self.threshold_days
        positive = silent.astype(float).to_numpy()
        return np.column_stack([1.0 - positive, positive])


class GradientBoostedModel:
    """LightGBM classifier over the tabular feature set.

    Trees are the right call here over a neural net: the data is tabular and
    modest in size, and the per-prediction feature attributions are what let the
    dashboard explain *why* a package was flagged.
    """

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = {
            "objective": "binary",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "class_weight": "balanced",  # abandonment is rare
            "random_state": RANDOM_SEED,
            "verbose": -1,
            **params,
        }
        self._model: Any = None

    def fit(self, train: pd.DataFrame, feature_columns: list[str] | None = None) -> None:
        """Train on a labelled feature table.

        Args:
            train: Rows carrying features and an `is_abandoned` column.
            feature_columns: Override the default feature list.
        """
        import lightgbm as lgb

        columns = feature_columns or FEATURE_COLUMNS
        self.feature_columns = columns
        self._model = lgb.LGBMClassifier(**self.params)
        self._model.fit(train[columns], train["is_abandoned"].astype(int))
        logger.info("Trained on %d rows, %d features.", len(train), len(columns))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probability of abandonment within the horizon."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self._model.predict_proba(features[self.feature_columns])

    def feature_importance(self) -> pd.Series:
        """Gain-based importance per feature, descending."""
        if self._model is None:
            raise RuntimeError("Call fit() before feature_importance().")
        return pd.Series(
            self._model.feature_importances_,
            index=self.feature_columns,
        ).sort_values(ascending=False)

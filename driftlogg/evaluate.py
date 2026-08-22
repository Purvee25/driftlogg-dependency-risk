"""Evaluation metrics chosen for a rare-event forecasting problem.

Two deliberate choices:

1. PR-AUC, not ROC-AUC. With ~3% positives, ROC-AUC looks flattering no matter
   what the model does, because the huge negative class dominates the false
   positive rate. Precision-recall tells the truth on imbalanced data.

2. Lead time is reported alongside accuracy. A model that flags a package the
   day before it dies is accurate and worthless — the entire value proposition
   is warning you early enough to act.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationReport:
    """Scores for one model on one test set.

    Attributes:
        pr_auc: Area under the precision-recall curve. The headline metric.
        roc_auc: Reported for comparison only; do not lead with it.
        precision_at_k: Precision within the k highest-risk predictions.
        recall_at_k: Recall within the same k.
        k: Size of the top-k slice.
        median_lead_time_days: Median warning time on correctly flagged packages.
        n_samples: Test set size.
        n_positive: Number of true abandonments in the test set.
    """

    pr_auc: float
    roc_auc: float
    precision_at_k: float
    recall_at_k: float
    k: int
    median_lead_time_days: float | None
    n_samples: int
    n_positive: int

    def summary(self) -> str:
        """Human-readable block for logs and the README."""
        lead = (
            f"{self.median_lead_time_days:.0f} days"
            if self.median_lead_time_days is not None
            else "n/a"
        )
        return (
            f"PR-AUC          {self.pr_auc:.3f}\n"
            f"ROC-AUC         {self.roc_auc:.3f}  (reference only)\n"
            f"Precision@{self.k:<4}  {self.precision_at_k:.3f}\n"
            f"Recall@{self.k:<7}  {self.recall_at_k:.3f}\n"
            f"Median lead     {lead}\n"
            f"Test set        {self.n_samples} rows, {self.n_positive} positive "
            f"({self.n_positive / max(self.n_samples, 1):.1%})"
        )


def precision_recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Precision and recall restricted to the k highest-scoring rows.

    This mirrors how the tool is actually used: an engineer reviews the top of
    the risk board, not every dependency.

    Args:
        y_true: Binary ground truth.
        y_score: Predicted probabilities.
        k: How many top-ranked rows to consider.

    Returns:
        Precision and recall over the top-k slice.
    """
    k = min(k, len(y_score))
    if k == 0:
        return 0.0, 0.0

    top_indices = np.argsort(y_score)[::-1][:k]
    selected = np.zeros_like(y_true)
    selected[top_indices] = 1

    precision = precision_score(y_true, selected, zero_division=0)
    recall = recall_score(y_true, selected, zero_division=0)
    return float(precision), float(recall)


def median_lead_time(
    frame: pd.DataFrame,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float | None:
    """Median days of warning across correctly flagged abandonments.

    Args:
        frame: Test rows carrying `is_abandoned` and `lead_time_days`.
        y_score: Predicted probabilities.
        threshold: Score above which a package counts as flagged.

    Returns:
        Median lead time, or None if nothing was correctly flagged.
    """
    if "lead_time_days" not in frame.columns:
        return None

    caught = frame[(frame["is_abandoned"].astype(bool)) & (y_score >= threshold)]
    lead_times = caught["lead_time_days"].dropna()
    if lead_times.empty:
        return None
    return float(lead_times.median())


def evaluate(
    frame: pd.DataFrame,
    y_score: np.ndarray,
    k: int = 20,
    threshold: float = 0.5,
) -> EvaluationReport:
    """Score predictions against ground truth.

    Args:
        frame: Test rows with an `is_abandoned` column.
        y_score: Predicted probability of abandonment.
        k: Size of the top-k slice to report.
        threshold: Cutoff used for the lead-time calculation.

    Returns:
        The filled-in report.
    """
    y_true = frame["is_abandoned"].astype(int).to_numpy()

    precision_k, recall_k = precision_recall_at_k(y_true, y_score, k)

    return EvaluationReport(
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        precision_at_k=precision_k,
        recall_at_k=recall_k,
        k=min(k, len(y_score)),
        median_lead_time_days=median_lead_time(frame, y_score, threshold),
        n_samples=len(frame),
        n_positive=int(y_true.sum()),
    )


def operating_points(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """Precision/recall across every threshold.

    Use this to pick the alerting threshold: too low and engineers ignore the
    tool, too high and it misses the deaths it exists to catch.

    Args:
        y_true: Binary ground truth.
        y_score: Predicted probabilities.

    Returns:
        Threshold, precision, and recall per operating point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return pd.DataFrame(
        {
            "threshold": np.append(thresholds, 1.0),
            "precision": precision,
            "recall": recall,
        }
    )

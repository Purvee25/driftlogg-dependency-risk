#!/usr/bin/env python3
"""Train the model and score it against the baseline.

Runs four configurations, because a single number would hide the thing that
matters most about this dataset:

  1. BASELINE      "no commits in 180 days" — the bar to clear.
  2. FULL          Every feature.
  3. NO POPULARITY Drops stars/forks/issues. Measures how much of the full
                   model's performance rests on popularity.
  4. MATCHED       Popularity-matched train and test sets, so popularity cannot
                   proxy for the sampling source. This is the honest number.

Positives and negatives came from different pools (GitHub archived-search
versus npm search) with different popularity profiles *and* different base
rates. Configurations 3 and 4 exist to quantify that confound rather than
leave it as a caveat.

Usage:
    python scripts/04_train.py --boundary 2025-01-01
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime

import pandas as pd

from driftlogg.config import settings
from driftlogg.evaluate import evaluate, operating_points
from driftlogg.features import FEATURE_COLUMNS
from driftlogg.model import GradientBoostedModel, InactivityBaseline, split_by_date
from driftlogg.sampling import (
    POPULARITY_FEATURES,
    match_on_popularity,
    popularity_balance_report,
)

logger = logging.getLogger(__name__)


def banner(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def train_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    top_k: int,
) -> tuple[GradientBoostedModel, object]:
    """Fit a model and evaluate it on the held-out set."""
    model = GradientBoostedModel()
    model.fit(train, feature_columns=columns)
    scores = model.predict_proba(test)[:, 1]
    return model, evaluate(test, scores, k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary",
        default="2025-01-01",
        help="Temporal split date. Train before, test on or after.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Slice size for precision@k.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dataset_path = settings.processed_dir / "dataset.parquet"
    if not dataset_path.exists():
        raise SystemExit("No dataset found. Run scripts/03_build_dataset.py first.")

    frame = pd.read_parquet(dataset_path)
    boundary = datetime.fromisoformat(args.boundary)
    split = split_by_date(frame, boundary)
    logger.info("Split: %s", split.describe())

    banner("POPULARITY CONFOUND — positive rate by star bin")
    print("A flat rate means popularity carries no marginal signal.\n")
    print(popularity_balance_report(frame).round(3).to_string())

    banner("1. BASELINE — flag if no commits in 180 days")
    baseline_scores = InactivityBaseline().predict_proba(split.test)[:, 1]
    baseline_report = evaluate(split.test, baseline_scores, k=args.top_k)
    print(baseline_report.summary())

    banner("2. FULL — every feature")
    full_model, full_report = train_and_score(split.train, split.test, FEATURE_COLUMNS, args.top_k)
    print(full_report.summary())

    banner("3. NO POPULARITY — stars/forks/issues removed")
    lean_columns = [c for c in FEATURE_COLUMNS if c not in POPULARITY_FEATURES]
    _, lean_report = train_and_score(split.train, split.test, lean_columns, args.top_k)
    print(lean_report.summary())

    banner("4. MATCHED — popularity-matched train and test")
    print("Downsampled so every star bin is class-balanced.\n")
    matched_train = match_on_popularity(split.train)
    matched_test = match_on_popularity(split.test)
    matched_model, matched_report = train_and_score(
        matched_train, matched_test, FEATURE_COLUMNS, args.top_k
    )
    print()
    print(matched_report.summary())

    banner("SUMMARY")
    rows = [
        ("Baseline", baseline_report),
        ("Full", full_report),
        ("No popularity", lean_report),
        ("Matched", matched_report),
    ]
    print(
        f"{'Configuration':<16}{'PR-AUC':>9}{'base':>7}{'lift':>7}"
        f"{'ROC-AUC':>10}{'P@' + str(args.top_k):>8}"
    )
    for name, report in rows:
        base_rate = report.n_positive / max(report.n_samples, 1)
        lift = report.pr_auc / base_rate if base_rate else float("nan")
        print(
            f"{name:<16}{report.pr_auc:>9.3f}{base_rate:>7.1%}{lift:>7.2f}"
            f"{report.roc_auc:>10.3f}{report.precision_at_k:>8.3f}"
        )

    print(
        "\nPR-AUC and its lift both depend on the base rate, which differs by "
        "configuration\n(matching forces 50%). ROC-AUC does not, so it is the "
        "metric to compare across rows."
    )

    popularity_cost = full_report.roc_auc - lean_report.roc_auc
    matching_cost = full_report.roc_auc - matched_report.roc_auc
    print(
        f"\nROC-AUC given up by dropping popularity features: {popularity_cost:+.3f}"
        f"\nROC-AUC given up by matching away the confound:    {matching_cost:+.3f}"
    )
    if abs(matching_cost) < 0.05:
        print(
            "\nBoth costs are small, so the model was not primarily riding the "
            "sampling\nartefact — the decay features carry it."
        )
    else:
        print(
            "\nMatching costs real performance: a meaningful share of the full "
            "model's score\ncame from the sampling artefact, not from decay signal."
        )

    banner("TOP FEATURES (matched model)")
    print(matched_model.feature_importance().head(10).to_string())

    # The matched model is the one worth shipping: it cannot lean on the
    # sampling artefact, so its behaviour on real inputs is better founded.
    model_path = settings.processed_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"model": matched_model, "features": FEATURE_COLUMNS}, handle)
    logger.info("Saved matched model to %s", model_path)

    y_true = matched_test["is_abandoned"].astype(int).to_numpy()
    scores = matched_model.predict_proba(matched_test)[:, 1]
    operating_points(y_true, scores).to_csv(
        settings.processed_dir / "operating_points.csv", index=False
    )
    logger.info("Wrote operating points — use these to pick an alerting threshold.")


if __name__ == "__main__":
    main()

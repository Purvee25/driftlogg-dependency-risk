#!/usr/bin/env python3
"""Train the model and score it against the baseline.

Reports the baseline first, on purpose. If the trained model does not clearly
beat "no commits in six months", the extra complexity has not earned its place
and the honest move is to say so.

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

logger = logging.getLogger(__name__)


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
    frame = frame.replace([float("inf"), float("-inf")], pd.NA)

    # LightGBM handles NaN natively; the baseline needs a real number to compare.
    frame["days_since_last_commit"] = frame["days_since_last_commit"].fillna(9999)

    split = split_by_date(frame, datetime.fromisoformat(args.boundary))
    logger.info("Split: %s", split.describe())

    print("\n" + "=" * 60)
    print("BASELINE — flag if no commits in 180 days")
    print("=" * 60)
    baseline = InactivityBaseline()
    baseline_scores = baseline.predict_proba(split.test)[:, 1]
    baseline_report = evaluate(split.test, baseline_scores, k=args.top_k)
    print(baseline_report.summary())

    print("\n" + "=" * 60)
    print("MODEL — LightGBM")
    print("=" * 60)
    model = GradientBoostedModel()
    model.fit(split.train)
    model_scores = model.predict_proba(split.test)[:, 1]
    model_report = evaluate(split.test, model_scores, k=args.top_k)
    print(model_report.summary())

    lift = model_report.pr_auc - baseline_report.pr_auc
    print(f"\nPR-AUC lift over baseline: {lift:+.3f}")
    if lift <= 0:
        print("The model does not beat the baseline. Revisit features before tuning.")

    print("\nTop features by gain:")
    print(model.feature_importance().head(10).to_string())

    model_path = settings.processed_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "features": FEATURE_COLUMNS}, handle)
    logger.info("Saved model to %s", model_path)

    y_true = split.test["is_abandoned"].astype(int).to_numpy()
    points = operating_points(y_true, model_scores)
    points.to_csv(settings.processed_dir / "operating_points.csv", index=False)
    logger.info("Wrote operating points — use these to pick an alerting threshold.")


if __name__ == "__main__":
    main()

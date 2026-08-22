#!/usr/bin/env python3
"""Turn raw payloads into a labelled, leakage-free feature table.

For each package this samples several prediction dates rather than one. That
multiplies the dataset and — more importantly — teaches the model what a package
looks like at varying distances from its death, instead of only at one arbitrary
moment.

Rows where the package was already dead at the prediction date are dropped:
predicting a death that already happened is not a prediction.

Usage:
    python scripts/03_build_dataset.py
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from driftlogg.config import settings
from driftlogg.features import LeakageError, build_features
from driftlogg.labels import DEFAULT_HORIZON_DAYS, build_label, is_already_dead

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_DAYS = 90
"""Spacing between prediction dates sampled per package."""


def sample_dates(start: datetime, end: datetime, interval_days: int) -> list[datetime]:
    """Evenly spaced prediction dates across an observation period."""
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=interval_days)
    return dates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01", help="Earliest prediction date.")
    parser.add_argument("--end", default="2025-09-01", help="Latest prediction date.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_DAYS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings.ensure_dirs()

    package_dir = settings.raw_dir / "packages"
    payload_paths = sorted(package_dir.glob("*.json"))
    if not payload_paths:
        raise SystemExit("No raw payloads found. Run scripts/02_fetch_repos.py first.")

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    prediction_dates = sample_dates(start, end, SAMPLE_INTERVAL_DAYS)
    logger.info("Sampling %d prediction dates per package.", len(prediction_dates))

    rows: list[dict] = []
    skipped_dead = 0
    leakage_errors = 0

    for path in tqdm(payload_paths, desc="packages"):
        payload = json.loads(path.read_text())
        package = payload.get("package", path.stem)
        repo = payload["repo"]

        for as_of in prediction_dates:
            label = build_label(package, as_of, repo, horizon_days=args.horizon)

            # Already dead at prediction time — the answer is in the input.
            if is_already_dead(label):
                skipped_dead += 1
                continue

            # Labels resting on undated README notices need manual dating.
            if label.needs_review:
                continue

            try:
                features = build_features(
                    package=package,
                    as_of=as_of,
                    repo=repo,
                    commits=payload.get("commits", []),
                    issues=payload.get("issues", []),
                    releases=payload.get("releases", []),
                )
            except LeakageError:
                logger.exception("Leakage detected for %s at %s", package, as_of)
                leakage_errors += 1
                continue

            row = features.to_row()
            row["is_abandoned"] = label.is_abandoned
            row["lead_time_days"] = label.lead_time_days
            row["label_source"] = label.source.value
            rows.append(row)

    frame = pd.DataFrame(rows)
    output = settings.processed_dir / "dataset.parquet"
    frame.to_parquet(output, index=False)

    positives = int(frame["is_abandoned"].sum())
    logger.info("Wrote %d rows to %s", len(frame), output)
    logger.info(
        "Positives: %d (%.2f%%) | skipped already-dead: %d | leakage errors: %d",
        positives,
        100 * positives / max(len(frame), 1),
        skipped_dead,
        leakage_errors,
    )

    if positives < 50:
        logger.warning(
            "Very few positives. Collect more archived repos before training — "
            "the model has almost nothing to learn from."
        )

    logger.info("Next: python scripts/04_train.py")


if __name__ == "__main__":
    main()

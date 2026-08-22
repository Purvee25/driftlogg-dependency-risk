#!/usr/bin/env python3
"""Turn raw payloads into a labelled, leakage-free feature table.

For each package this samples several prediction dates rather than one. That
multiplies the dataset and — more importantly — teaches the model what a package
looks like at varying distances from going quiet, instead of only at one
arbitrary moment.

Two classes of row are dropped, and the counts are reported because they are
diagnostic:

  - `not_alive`: the package was already silent at the prediction date, so
    there is nothing left to forecast.
  - `censored`: the verdict window runs past the collection date, so the
    silence cannot be observed yet.

Usage:
    python scripts/03_build_dataset.py
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from driftlogg.config import settings
from driftlogg.features import LeakageError, build_features
from driftlogg.labels import (
    HORIZON_DAYS,
    SUSTAINED_SILENCE_DAYS,
    ExclusionReason,
    build_label,
)

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


def latest_collection_date(package_dir) -> datetime:
    """Infer when collection ran, from the newest payload file on disk.

    Used for the censoring check. Reading it from the data rather than assuming
    "now" keeps a rebuild reproducible weeks later.
    """
    paths = list(package_dir.glob("*.json"))
    if not paths:
        return datetime.utcnow()
    newest = max(path.stat().st_mtime for path in paths)
    return datetime.fromtimestamp(newest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01", help="Earliest prediction date.")
    parser.add_argument("--end", default="2025-09-01", help="Latest prediction date.")
    parser.add_argument("--horizon", type=int, default=HORIZON_DAYS)
    parser.add_argument("--silence", type=int, default=SUSTAINED_SILENCE_DAYS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings.ensure_dirs()

    package_dir = settings.raw_dir / "packages"
    payload_paths = sorted(package_dir.glob("*.json"))
    if not payload_paths:
        raise SystemExit("No raw payloads found. Run scripts/02_fetch_repos.py first.")

    collected_at = latest_collection_date(package_dir)
    logger.info("Treating %s as the collection date.", collected_at.date())

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    prediction_dates = sample_dates(start, end, SAMPLE_INTERVAL_DAYS)
    logger.info(
        "Sampling %d prediction dates per package | horizon=%dd silence=%dd",
        len(prediction_dates),
        args.horizon,
        args.silence,
    )

    rows: list[dict] = []
    excluded: Counter[str] = Counter()
    leakage_errors = 0

    for path in tqdm(payload_paths, desc="packages"):
        payload = json.loads(path.read_text())
        package = payload.get("package", path.stem)
        repo = payload["repo"]
        commits = payload.get("commits", [])

        for as_of in prediction_dates:
            label = build_label(
                package=package,
                as_of=as_of,
                commits=commits,
                repo=repo,
                data_collected_at=collected_at,
                horizon_days=args.horizon,
                silence_days=args.silence,
            )

            if not label.usable:
                excluded[label.exclusion.value] += 1
                continue

            try:
                features = build_features(
                    package=package,
                    as_of=as_of,
                    repo=repo,
                    commits=commits,
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

    if not rows:
        raise SystemExit(
            "No usable rows. Every candidate was excluded — check that the "
            "prediction window leaves room for the verdict window."
        )

    frame = pd.DataFrame(rows)
    output = settings.processed_dir / "dataset.parquet"
    frame.to_parquet(output, index=False)

    positives = int(frame["is_abandoned"].sum())
    logger.info("Wrote %d rows to %s", len(frame), output)
    logger.info(
        "Positives: %d (%.2f%%) | leakage errors: %d",
        positives,
        100 * positives / max(len(frame), 1),
        leakage_errors,
    )
    for reason in ExclusionReason:
        if excluded[reason.value]:
            logger.info("Excluded (%s): %d", reason.value, excluded[reason.value])

    if positives < 50:
        logger.warning("Very few positives — the model has almost nothing to learn from.")

    logger.info("Next: python scripts/04_train.py")


if __name__ == "__main__":
    main()

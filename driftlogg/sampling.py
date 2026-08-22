"""Matched sampling to remove popularity as a proxy for the sampling source.

**The problem this addresses.** Positives and negatives were drawn from two
different pools: a GitHub search for archived repositories, and an npm search
for live packages. The pools differ systematically in popularity (median 902
stars versus 201) *and* in label rate (64% positive versus 27%). Popularity is
therefore a usable proxy for "which pool did this come from", and pool
membership predicts the label — not because popular packages die more often,
but because of how the data was collected.

The archived pool was selected precisely *because* those repositories ended up
archived. That is selection on the outcome, and it is the more serious half of
the bias.

Note that the raw correlation between stars and the label is near zero
(r = -0.05), which makes this invisible to a linear check. A tree model can
still exploit it: split on stars to infer the pool, then use the pool's base
rate. Gain-based importance ranked `stars` second despite that near-zero
correlation, which is exactly the signature of a proxy variable.

**The fix.** Bin by popularity and downsample within each bin so every bin
carries the same class balance. Popularity then carries no marginal information
about the label by construction, and any performance that survives is coming
from genuine decay signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

POPULARITY_COLUMN = "stars"

STAR_BINS = [-1, 10, 50, 100, 500, 1_000, 5_000, np.inf]
"""Log-ish bin edges. Stars are heavy-tailed, so equal-width bins would put
almost everything in the first bucket."""

POPULARITY_FEATURES = [
    "stars",
    "forks",
    "fork_star_ratio",
    "open_issues_count",
]
"""Features that describe how popular a project is rather than how it is doing.

Used by the ablation in `scripts/04_train.py`: training without these measures
how much of the model's performance depends on popularity at all.
"""


def match_on_popularity(
    frame: pd.DataFrame,
    column: str = POPULARITY_COLUMN,
    bins: list[float] | None = None,
    label_column: str = "is_abandoned",
    seed: int = 42,
) -> pd.DataFrame:
    """Downsample within popularity bins so each bin is class-balanced.

    After this, `column` cannot indicate the label on its own: every bin holds
    equal numbers of positives and negatives, so learning "high stars means
    safe" (or the reverse) buys the model nothing.

    Args:
        frame: Labelled dataset.
        column: Popularity column to match on.
        bins: Bin edges. Defaults to STAR_BINS.
        label_column: Boolean label column.
        seed: Seed for the downsampling draw.

    Returns:
        A balanced subset. Expect to lose a substantial fraction of rows —
        that is the cost of removing the confound, and it is worth paying.

    Raises:
        ValueError: If the frame lacks the required columns.
    """
    for required in (column, label_column):
        if required not in frame.columns:
            raise ValueError(f"Missing column: {required}")

    edges = bins if bins is not None else STAR_BINS
    rng = np.random.default_rng(seed)

    binned = frame.assign(_bin=pd.cut(frame[column], bins=edges))
    kept: list[pd.DataFrame] = []

    for bin_label, group in binned.groupby("_bin", observed=True):
        positives = group[group[label_column]]
        negatives = group[~group[label_column]]

        keep_n = min(len(positives), len(negatives))
        if keep_n == 0:
            logger.debug("Bin %s has only one class; dropping %d rows.", bin_label, len(group))
            continue

        kept.append(_sample(positives, keep_n, rng))
        kept.append(_sample(negatives, keep_n, rng))

    if not kept:
        raise ValueError("Matching removed every row; check the bin edges.")

    matched = pd.concat(kept, ignore_index=True).drop(columns="_bin")

    logger.info(
        "Matched on %s: %d -> %d rows (%.1f%% retained), positive rate %.1f%% -> %.1f%%",
        column,
        len(frame),
        len(matched),
        100 * len(matched) / len(frame),
        100 * frame[label_column].mean(),
        100 * matched[label_column].mean(),
    )
    return matched


def _sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Take n rows without replacement, or all of them if there are fewer."""
    if len(frame) <= n:
        return frame
    indices = rng.choice(len(frame), size=n, replace=False)
    return frame.iloc[indices]


def popularity_balance_report(
    frame: pd.DataFrame,
    column: str = POPULARITY_COLUMN,
    bins: list[float] | None = None,
    label_column: str = "is_abandoned",
) -> pd.DataFrame:
    """Positive rate per popularity bin, for auditing the confound.

    A flat positive rate across bins means popularity carries no marginal
    information about the label. A sloped one means it does.

    Args:
        frame: Labelled dataset.
        column: Popularity column.
        bins: Bin edges. Defaults to STAR_BINS.
        label_column: Boolean label column.

    Returns:
        Row count and positive rate per bin.
    """
    edges = bins if bins is not None else STAR_BINS
    binned = frame.assign(_bin=pd.cut(frame[column], bins=edges))

    report = binned.groupby("_bin", observed=True)[label_column].agg(["size", "mean"])
    return report.rename(columns={"size": "rows", "mean": "positive_rate"})

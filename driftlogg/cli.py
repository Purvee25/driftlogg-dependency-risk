"""Command-line interface, built for use as a CI gate.

    python -m driftlogg check package.json
    python -m driftlogg check requirements.txt --changed-only --base-ref origin/main

Exit codes:
    0  no dependency exceeded the threshold
    1  at least one did — fail the build
    2  usage or configuration error

The `--changed-only` mode is what makes this practical in CI. Scoring every
dependency on every pull request means a repository with one long-dead package
fails forever, which trains people to ignore the check. Scoring only what the
branch *adds* keeps the signal tied to something the author can act on.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from driftlogg.api.manifests import ManifestParseError, parse_manifest
from driftlogg.api.schemas import PackageRisk, RiskBand
from driftlogg.api.service import ScoringService

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_RISK_FOUND = 1
EXIT_ERROR = 2

DEFAULT_THRESHOLD = 0.60
"""Matches the HIGH band cutoff in the API schema."""


def find_line_number(manifest_path: Path, package: str) -> int | None:
    """Locate the line declaring a package, for editor annotations.

    Args:
        manifest_path: Manifest to search.
        package: Package name to find.

    Returns:
        The 1-indexed line number, or None if it cannot be located.
    """
    try:
        lines = manifest_path.read_text().splitlines()
    except OSError:
        return None

    for index, line in enumerate(lines, start=1):
        # Quoted in package.json, bare at line start in requirements.txt.
        if f'"{package}"' in line or line.strip().split("==")[0].strip() == package:
            return index
    return None


def changed_dependencies(manifest_path: Path, base_ref: str) -> set[str] | None:
    """Dependencies added relative to a git ref.

    Args:
        manifest_path: Manifest to compare.
        base_ref: Git ref to diff against, e.g. "origin/main".

    Returns:
        Names present now but absent at `base_ref`, or None if the comparison
        could not be made (new file, missing ref, not a git repo).
    """
    try:
        previous = subprocess.run(
            ["git", "show", f"{base_ref}:{manifest_path}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        logger.info("No %s at %s — treating every dependency as new.", manifest_path, base_ref)
        return None

    try:
        _, old_names = parse_manifest(manifest_path.name, previous)
    except ManifestParseError:
        logger.warning("Could not parse %s at %s; comparing nothing.", manifest_path, base_ref)
        return None

    return set(old_names)


def emit_github_annotation(risk: PackageRisk, manifest: Path, line: int | None) -> None:
    """Print a GitHub Actions annotation for one risky dependency.

    Annotations surface inline on the pull request diff, which is the whole
    point of running this in CI rather than as a report nobody opens.
    """
    location = f"file={manifest}"
    if line:
        location += f",line={line}"

    reasons = "; ".join(reason.label for reason in risk.reasons) or "elevated risk"
    score = f"{risk.score:.0%}" if risk.score is not None else "unknown"
    message = f"{risk.package}: {score} chance of going unmaintained — {reasons}"

    level = "error" if risk.band is RiskBand.HIGH else "warning"
    print(f"::{level} {location}::{message}")


def render_text(results: list[PackageRisk], threshold: float) -> None:
    """Print a human-readable summary to stdout."""
    flagged = [r for r in results if r.score is not None and r.score >= threshold]
    scored = [r for r in results if r.score is not None]
    skipped = [r for r in results if r.score is None]

    print(f"\nChecked {len(results)} dependencies ({len(scored)} scored).\n")

    if flagged:
        print(f"{len(flagged)} at or above the {threshold:.0%} threshold:\n")
        for risk in sorted(flagged, key=lambda r: r.score or 0, reverse=True):
            print(f"  {risk.score:.0%}  {risk.package}  ({risk.repo or 'unresolved'})")
            for reason in risk.reasons:
                print(f"          {reason.label}")
    else:
        print("Nothing above the threshold.")

    if skipped:
        print(f"\n{len(skipped)} could not be scored:")
        for risk in skipped[:10]:
            print(f"  {risk.package}: {risk.error}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")


def command_check(args: argparse.Namespace) -> int:
    """Score a manifest and decide whether the build should fail."""
    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"No such manifest: {manifest}", file=sys.stderr)
        return EXIT_ERROR

    try:
        _, names = parse_manifest(manifest.name, manifest.read_text())
    except ManifestParseError as exc:
        print(f"Could not parse {manifest}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.changed_only:
        previous = changed_dependencies(manifest, args.base_ref)
        if previous is not None:
            added = [n for n in names if n not in previous]
            logger.info("%d of %d dependencies are new on this branch.", len(added), len(names))
            names = added

    if not names:
        print("No dependencies to check.")
        return EXIT_OK

    if len(names) > args.max_packages:
        print(f"Checking the first {args.max_packages} of {len(names)} dependencies.")
        names = names[: args.max_packages]

    service = ScoringService()
    if not service.model_loaded:
        print(
            "warning: no trained model found; scoring with the inactivity " "baseline instead.",
            file=sys.stderr,
        )

    results = service.score_packages(names)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "model": service.model_kind,
                    "threshold": args.threshold,
                    "results": [r.model_dump(mode="json") for r in results],
                },
                indent=2,
            )
        )
    elif args.format == "github":
        for risk in results:
            if risk.score is not None and risk.score >= args.threshold:
                emit_github_annotation(risk, manifest, find_line_number(manifest, risk.package))
        render_text(results, args.threshold)
    else:
        render_text(results, args.threshold)

    flagged = [r for r in results if r.score is not None and r.score >= args.threshold]
    if flagged and not args.warn_only:
        return EXIT_RISK_FOUND
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="driftlogg",
        description="Flag dependencies heading for abandonment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Score a dependency manifest.")
    check.add_argument("manifest", help="Path to package.json or requirements.txt.")
    check.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Risk score that fails the build (default {DEFAULT_THRESHOLD}).",
    )
    check.add_argument(
        "--changed-only",
        action="store_true",
        help="Only score dependencies added relative to --base-ref.",
    )
    check.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git ref to compare against (default origin/main).",
    )
    check.add_argument(
        "--format",
        choices=["text", "json", "github"],
        default="text",
        help="Output format. 'github' emits inline PR annotations.",
    )
    check.add_argument(
        "--warn-only",
        action="store_true",
        help="Report findings but always exit 0.",
    )
    check.add_argument(
        "--max-packages",
        type=int,
        default=100,
        help="Cap on dependencies scored per run (default 100).",
    )
    check.set_defaults(func=command_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

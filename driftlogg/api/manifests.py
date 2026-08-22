"""Parsing dependency manifests into plain package names.

Supports `package.json` and `requirements.txt`. Both formats carry version
constraints that are irrelevant here — DriftLogg scores the health of the
project behind a package, not the specific version pinned.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum

REQUIREMENT_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*(?:\[[^\]]+\])?\s*(?:[<>=!~;].*)?$")
"""Matches a requirement line, capturing the bare package name.

Handles extras (`pkg[extra]`), version specifiers, and environment markers.
"""


class ManifestKind(StrEnum):
    """Supported manifest formats."""

    PACKAGE_JSON = "package.json"
    REQUIREMENTS_TXT = "requirements.txt"


class ManifestParseError(ValueError):
    """Raised when a manifest cannot be parsed."""


def detect_kind(filename: str, content: str) -> ManifestKind:
    """Guess the manifest format from filename, falling back to content.

    Args:
        filename: Uploaded filename.
        content: Raw file contents.

    Returns:
        The detected format.

    Raises:
        ManifestParseError: If the format cannot be determined.
    """
    lowered = filename.lower()
    if lowered.endswith(".json"):
        return ManifestKind.PACKAGE_JSON
    if lowered.endswith(".txt") or "requirements" in lowered:
        return ManifestKind.REQUIREMENTS_TXT

    stripped = content.lstrip()
    if stripped.startswith("{"):
        return ManifestKind.PACKAGE_JSON

    raise ManifestParseError(
        f"Cannot determine manifest type for {filename!r}. "
        "Expected package.json or requirements.txt."
    )


def parse_package_json(content: str, include_dev: bool = True) -> list[str]:
    """Extract dependency names from a package.json.

    Args:
        content: Raw file contents.
        include_dev: Whether to include devDependencies.

    Returns:
        Dependency names, deduplicated, in a stable order.

    Raises:
        ManifestParseError: If the JSON is malformed.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ManifestParseError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestParseError("package.json must contain a JSON object.")

    sections = ["dependencies"]
    if include_dev:
        sections.append("devDependencies")

    names: list[str] = []
    for section in sections:
        block = data.get(section)
        if isinstance(block, dict):
            names.extend(block.keys())

    return _deduplicate(names)


def parse_requirements_txt(content: str) -> list[str]:
    """Extract package names from a requirements.txt.

    Comments, blank lines, editable installs, and `-r` includes are skipped —
    an include would need the referenced file, which an upload does not carry.

    Args:
        content: Raw file contents.

    Returns:
        Package names, deduplicated, in a stable order.
    """
    names: list[str] = []

    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue

        # A bare URL or local path has no resolvable package name.
        if "://" in line or line.startswith("."):
            continue

        match = REQUIREMENT_RE.match(line)
        if match:
            names.append(match.group(1))

    return _deduplicate(names)


def parse_manifest(
    filename: str,
    content: str,
    include_dev: bool = True,
) -> tuple[ManifestKind, list[str]]:
    """Parse any supported manifest into dependency names.

    Args:
        filename: Uploaded filename, used to detect the format.
        content: Raw file contents.
        include_dev: Whether to include dev dependencies (package.json only).

    Returns:
        The detected format and the dependency names.

    Raises:
        ManifestParseError: If the format is unknown or the content malformed.
    """
    kind = detect_kind(filename, content)

    if kind is ManifestKind.PACKAGE_JSON:
        return kind, parse_package_json(content, include_dev)
    return kind, parse_requirements_txt(content)


def _deduplicate(names: list[str]) -> list[str]:
    """Remove duplicates while preserving first-seen order."""
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique

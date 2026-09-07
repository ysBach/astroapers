"""Check matching versions and extract release notes before publishing.

Use Python 3.11+. Pass ``--tag vX.Y.Z`` for the stricter publication checks.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path

STABLE_VERSION = r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
LOGGER = logging.getLogger(__name__)


def release_notes(changelog: str, version: str, *, require_release: bool) -> str:
    """Return the dated version entry, rejecting unfinished tagged releases.

    Parameters
    ----------
    changelog : str
        Complete Markdown changelog.
    version : str
        Version shared by the package manifests.
    require_release : bool
        Require dated, nonempty notes and an empty `Unreleased` section.

    Returns
    -------
    str
        Release notes, or an empty string for an unfinished branch build.
    """
    headings = list(re.finditer(r"^## (.+)$", changelog, re.MULTILINE))
    entries: list[str] = []
    for index, heading in enumerate(headings):
        end = (
            headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
        )
        body = changelog[heading.end() : end].strip()
        content = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
        content = re.sub(r"^\s*#{1,6}\s+.*$", "", content, flags=re.MULTILINE).strip()
        title = heading.group(1).strip()
        if title == "Unreleased" and require_release and content:
            raise ValueError("Move all Unreleased changes into the dated release entry")
        match = re.fullmatch(
            rf"{re.escape(version)}(?: - (\d{{4}}-\d{{2}}-\d{{2}})| \((\d{{4}}-\d{{2}}-\d{{2}})\))",
            title,
        )
        if match:
            date.fromisoformat(match.group(1) or match.group(2))
            if not content:
                raise ValueError(f"Release {version} has empty notes")
            entries.append(body + "\n")
    if len(entries) > 1:
        raise ValueError(f"Duplicate changelog entries for {version}")
    if require_release and not entries:
        raise ValueError(f"Add a nonempty '## {version} - YYYY-MM-DD' changelog entry")
    return entries[0] if entries else ""


def check_release(root: Path, tag: str | None = None) -> tuple[str, str]:
    """Validate package metadata and return the version and release notes.

    Parameters
    ----------
    root : Path
        Repository root containing both manifests and lockfiles.
    tag : str or None, optional
        Exact stable release tag. `None` permits unfinished release notes.

    Returns
    -------
    tuple of str
        Package version and extracted Markdown notes.
    """
    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["package"]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    version = cargo["version"]
    if not re.fullmatch(STABLE_VERSION, version):
        raise ValueError(f"Version {version!r} must have stable X.Y.Z form")
    if cargo["name"] != "astroapers" or project["name"] != cargo["name"]:
        raise ValueError("Both manifests must name astroapers")
    if project["version"] != version:
        raise ValueError("Cargo.toml and pyproject.toml versions differ")
    for filename in ("Cargo.lock", "uv.lock"):
        packages = tomllib.loads((root / filename).read_text(encoding="utf-8"))[
            "package"
        ]
        roots = [
            package
            for package in packages
            if package["name"] == cargo["name"]
            and (
                "source" not in package
                if filename == "Cargo.lock"
                else package.get("source") == {"editable": "."}
            )
        ]
        if len(roots) != 1 or roots[0]["version"] != version:
            raise ValueError(f"{filename} root package version must be {version}")
    if tag is not None and (
        not re.fullmatch(rf"v{STABLE_VERSION}", tag) or tag != f"v{version}"
    ):
        raise ValueError(f"Tag {tag!r} must exactly match v{version}")
    dependencies = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        dependencies.extend(group)
    if any("@" in dependency for dependency in dependencies):
        raise ValueError("Published dependencies must use registry versions, not URLs")
    notes = release_notes(
        (root / "CHANGELOG.md").read_text(encoding="utf-8"),
        version,
        require_release=tag is not None,
    )
    return version, notes


def main() -> None:
    """Validate the checkout and optionally write CI outputs and release notes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--tag")
    parser.add_argument("--notes-file", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        version, notes = check_release(args.root, args.tag)
        if args.require_clean:
            status = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=args.root,
                text=True,
            )
            if status.strip():
                raise ValueError("Commit the intended changes and use a clean checkout")
        if args.notes_file:
            if not notes:
                raise ValueError(
                    "Cannot write release notes without a dated version entry"
                )
            args.notes_file.write_text(notes, encoding="utf-8")
        if output := os.environ.get("GITHUB_OUTPUT"):
            with Path(output).open("a", encoding="utf-8") as stream:
                stream.write(f"version={version}\n")
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release check failed: {error}\n")
    LOGGER.info("Release metadata is consistent: %s", version)


if __name__ == "__main__":
    main()

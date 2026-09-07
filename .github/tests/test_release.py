"""Test publication failures before credentials become available."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_release.py"
SPEC = importlib.util.spec_from_file_location("check_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create independent manifests and changelog for a release candidate."""
    for filename, section in (("Cargo.toml", "package"), ("pyproject.toml", "project")):
        (tmp_path / filename).write_text(
            f'[{section}]\nname = "astroapers"\nversion = "1.2.3"\n', encoding="utf-8"
        )
    for filename in ("Cargo.lock", "uv.lock"):
        source = '\nsource = { editable = "." }' if filename == "uv.lock" else ""
        (tmp_path / filename).write_text(
            f'[[package]]\nname = "astroapers"\nversion = "1.2.3"{source}\n',
            encoding="utf-8",
        )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## Unreleased\n\n## 1.2.3 - 2026-09-07\n\n- Fix aperture sums.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_tag_returns_only_its_release_notes(repository: Path) -> None:
    version, notes = release.check_release(repository, "v1.2.3")
    assert version == "1.2.3"
    assert notes == "- Fix aperture sums.\n"


@pytest.mark.parametrize(
    "filename", ["Cargo.toml", "pyproject.toml", "Cargo.lock", "uv.lock"]
)
def test_version_mismatch_blocks_release(repository: Path, filename: str) -> None:
    path = repository / filename
    path.write_text(
        path.read_text(encoding="utf-8").replace("1.2.3", "1.2.4"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version"):
        release.check_release(repository, "v1.2.3")


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2.4", "v1.2.3rc1", "v01.2.3", "v1.2.3\n"])
def test_tag_must_be_exact_stable_version(repository: Path, tag: str) -> None:
    with pytest.raises(ValueError, match="Tag"):
        release.check_release(repository, tag)


@pytest.mark.parametrize(
    "changelog",
    [
        "## Unreleased\n- Fix aperture sums.\n",
        "## 1.2.3\n- Fix aperture sums.\n",
        "## 1.2.3 - 2026-02-30\n- Fix aperture sums.\n",
        "## 1.2.3 - 2026-09-07\n### Fixed\n<!-- write notes here -->\n",
        "## 1.2.3 - 2026-09-07\n- One.\n## 1.2.3 (2026-09-07)\n- Two.\n",
    ],
)
def test_missing_invalid_or_duplicate_notes_block_release(
    repository: Path, changelog: str
) -> None:
    (repository / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    with pytest.raises(ValueError):
        release.check_release(repository, "v1.2.3")


def test_unreleased_changes_allowed_only_without_tag(repository: Path) -> None:
    path = repository / "CHANGELOG.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "## Unreleased", "## Unreleased\n- Next fix."
        ),
        encoding="utf-8",
    )
    assert release.check_release(repository)[0] == "1.2.3"
    with pytest.raises(ValueError, match="Unreleased"):
        release.check_release(repository, "v1.2.3")


def test_historical_parenthesized_date_is_supported(repository: Path) -> None:
    path = repository / "CHANGELOG.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(" - 2026-09-07", " (2026-09-07)"),
        encoding="utf-8",
    )
    assert release.check_release(repository, "v1.2.3")[1] == "- Fix aperture sums.\n"


def test_sibling_override_is_development_only(repository: Path) -> None:
    path = repository / "pyproject.toml"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            '\ndependencies = ["reducers>=0.3.0"]\n[tool.uv.sources]\nreducers = {path = "../reducers", editable = true}\n'
        )
    assert release.check_release(repository, "v1.2.3")[0] == "1.2.3"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "reducers>=0.3.0", "reducers @ file:///tmp/reducers"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="registry"):
        release.check_release(repository, "v1.2.3")

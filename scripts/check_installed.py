"""Verify the installed artifact, its compiled extension, and an aperture sum."""

from __future__ import annotations

import argparse
import importlib
import logging
import sysconfig
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

import astroapers

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Require installed packages and measure a radius-two circle on unit pixels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dist = distribution("astroapers")
    if dist.version != args.version:
        raise RuntimeError("Installed distribution version differs from the release")
    if dist.metadata.get("License-Expression") != "BSD-3-Clause":
        raise RuntimeError("Installed distribution must declare BSD-3-Clause")
    root = Path(__file__).resolve().parents[1]
    for name in ("LICENSE", "licenses/SEP-overlap-BSD-3-Clause.txt"):
        expected = (root / name).read_text(encoding="utf-8")
        if dist.read_text(f"licenses/{name}") != expected:
            raise RuntimeError(f"Installed notice differs from {name}")
    site_paths = {
        Path(sysconfig.get_path(kind)).resolve() for kind in ("purelib", "platlib")
    }
    for name in (
        "astroapers",
        "astroapers._rust",
        "reducers",
        "matplotlib",
        "photutils.aperture",
        "sep",
        "scipy",
        "astro_ndslice",
    ):
        module = importlib.import_module(name)
        location = Path(module.__file__).resolve()
        if not any(location.is_relative_to(site_path) for site_path in site_paths):
            raise RuntimeError(f"{name} loaded outside this environment: {location}")
    total, npix = astroapers.CircAp((5.0, 5.0), r=2.0).apsum_exact(np.ones((11, 11)))
    np.testing.assert_allclose([total, npix], [4 * np.pi, 4 * np.pi], rtol=1e-12)
    LOGGER.info("Installed astroapers %s passed the aperture smoke test", args.version)


if __name__ == "__main__":
    main()

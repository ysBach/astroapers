from __future__ import annotations

import numpy as np

from astroapers import kernels
from astroapers._containers import BoundingBox
from astroapers import _kernel_dispatch, _kernel_docs, _kernel_validation


def test_kernel_private_helpers_are_split_by_responsibility():
    assert hasattr(_kernel_dispatch, "_masked_apsum")
    assert hasattr(_kernel_dispatch, "_apsum")
    assert hasattr(_kernel_validation, "_validate_wedge")
    assert hasattr(_kernel_validation, "_shape2")
    assert hasattr(_kernel_docs, "apply_kernel_docstrings")


def test_public_kernel_docstrings_still_apply_after_helper_split():
    assert kernels.apsum_circ_exact.__doc__.startswith(
        "Return exact circular aperture sums"
    )
    assert kernels.weights_ellip_exact.__doc__.startswith(
        "Return exact elliptical-aperture bbox-tight aperture weights"
    )


def test_masked_sum_without_npix_allocates_only_sum_output(monkeypatch):
    full_calls = []
    original_full = np.full

    def tracked_full(*args, **kwargs):
        full_calls.append((args, kwargs))
        return original_full(*args, **kwargs)

    def weights_func(x, y, *params, validate=True):
        return [np.ones((1, 1), dtype=np.float64)], [BoundingBox(0, 1, 0, 1)]

    monkeypatch.setattr(_kernel_dispatch.np, "full", tracked_full)

    result = _kernel_dispatch._masked_apsum(
        weights_func,
        np.ones((3, 3), dtype=np.float64),
        None,
        np.array([1.0]),
        np.array([1.0]),
        return_npix=False,
    )

    assert result.shape == (1,)
    assert result[0] == 1.0
    assert len(full_calls) == 1

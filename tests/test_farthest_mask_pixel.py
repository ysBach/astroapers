from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import astroapers as apers
from astroapers import kernels


def test_farthest_mask_pixel_returns_radius_only_by_default():
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[1, 2, 3] = True

    radius = apers.farthest_mask_pixel(mask, center=(0.0, 0.0, 0.0))

    assert_allclose(radius, np.sqrt(14.0))


def test_farthest_mask_pixel_return_pos_returns_all_tied_positions():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[2, 2] = True

    radius, pos = apers.farthest_mask_pixel(
        mask,
        center=(1.0, 1.0),
        return_pos=True,
    )

    assert_allclose(radius, np.sqrt(2.0))
    assert_array_equal(pos, np.array([[0, 0], [2, 2]]))


def test_farthest_mask_pixel_rejects_empty_mask():
    mask = np.zeros((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="mask must contain at least one True pixel"):
        apers.farthest_mask_pixel(mask, center=(1.0, 1.0))


def test_farthest_mask_pixel_rejects_center_dimension_mismatch():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    with pytest.raises(ValueError, match="center length must match mask.ndim"):
        apers.farthest_mask_pixel(mask, center=(1.0,))


def test_farthest_mask_pixel_is_exported_from_kernels_namespace():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    radius = kernels.farthest_mask_pixel(mask, center=(0.0, 0.0))

    assert_allclose(radius, np.sqrt(2.0))

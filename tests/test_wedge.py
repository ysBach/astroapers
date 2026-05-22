from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astroapers as apers


def test_wedge_area_for_constant_width_annular_sector():
    aperture = apers.WedgeAp((10.0, 10.0), 2.0, 6.0, 0.3, 0.7)

    assert_allclose(aperture.area, 0.5 * 0.7 * (6.0**2 - 2.0**2))


def test_wedge_center_mask_selects_expected_east_sector_pixels():
    aperture = apers.WedgeAp((5.0, 5.0), 1.0, 3.1, 0.0, 0.5 * math.pi)
    weights = aperture.weights(method="center")[0]
    bbox = aperture.bboxes()[0]
    image = bbox.to_image(weights, (11, 11))

    assert image[5, 7] == 1.0
    assert image[5, 8] == 1.0
    assert image[5, 6] == 1.0
    assert image[7, 5] == 0.0
    assert image[5, 5] == 0.0


def test_wedge_exact_apsum_matches_mask_reduction_and_tracks_area():
    data = np.ones((64, 64), dtype=np.float64)
    aperture = apers.WedgeAp((32.0, 32.0), 4.0, 14.0, 0.4, 0.9)

    apsum, npix = aperture.apsum(data)
    weights = aperture.weights(method="exact")[0]
    bbox = aperture.bboxes()[0]

    assert_allclose(apsum, npix)
    assert_allclose(bbox.apsum(weights, data), (apsum, npix))
    assert_allclose(npix, aperture.area, rtol=0, atol=0.5)


def test_wedge_direct_kernels_match_object_for_tapered_wedge():
    data = np.arange(50 * 50, dtype=np.float64).reshape(50, 50)
    positions = np.array([[20.0, 20.0], [28.0, 22.0]])
    aperture = apers.WedgeAp(
        positions,
        3.0,
        10.0,
        0.2,
        0.4,
        theta_out=0.5,
        dtheta_out=0.8,
    )

    obj_apsum, obj_npix = aperture.apsum(data, method="center")
    got_apsum, got_npix = apers.apsum_wedge_center(
        data, positions[:, 0], positions[:, 1], 3.0, 10.0, 0.2, 0.4, 0.5, 0.8
    )

    assert_allclose(got_apsum, obj_apsum)
    assert_allclose(got_npix, obj_npix)


@pytest.mark.parametrize("method", ["exact", "center"])
def test_wedge_object_npix_and_weights(method):
    positions = np.array([[20.0, 20.0], [28.0, 22.0]])
    aperture = apers.WedgeAp(
        positions,
        3.0,
        10.0,
        0.2,
        0.4,
        theta_out=0.5,
        dtheta_out=0.8,
    )
    npix_func = apers.npix_wedge_exact if method == "exact" else apers.npix_wedge_center
    bad = np.zeros((50, 50), dtype=bool)
    bad[20, 20] = True

    weights = aperture.weights(method=method)
    boxes = aperture.bboxes()

    assert len(weights) == len(boxes) == len(positions)
    for weight, box in zip(weights, boxes, strict=True):
        assert weight.shape == box.shape
    assert_allclose(
        aperture.npix(bad.shape, method=method),
        npix_func(
            positions[:, 0],
            positions[:, 1],
            3.0,
            10.0,
            0.2,
            0.4,
            0.5,
            0.8,
            shape=bad.shape,
        ),
    )
    assert_allclose(
        aperture.npix(bad.shape, method=method, mask=bad),
        aperture.apsum(np.ones(bad.shape), method=method, mask=bad)[1],
    )


def test_wedge_tapered_exact_mode_is_available_and_tracks_area():
    aperture = apers.WedgeAp(
        (32.0, 32.0), 4.0, 14.0, 0.0, 0.5, theta_out=0.2, dtheta_out=0.7
    )
    data = np.ones((64, 64), dtype=np.float64)

    assert hasattr(apers, "apsum_wedge_exact")
    assert hasattr(apers, "npix_wedge_exact")
    exact_sum, exact_npix = aperture.apsum(data, method="exact")
    center_sum, center_npix = aperture.apsum(data, method="center")

    assert_allclose(exact_sum, exact_npix)
    assert abs(float(exact_npix) - aperture.area) < 0.5
    assert center_npix > 0.0
    assert abs(float(center_sum) - float(exact_sum)) < 10.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"r_in": 0.0}, "r_in"),
        ({"r_in": 3.0, "r_out": 3.0}, "smaller"),
        ({"dtheta_in": 0.0}, "dtheta_in"),
        ({"dtheta_in": 2.0 * math.pi}, "dtheta_in"),
        ({"dtheta_in": 2.0 * math.pi + 0.1}, "dtheta_in"),
        ({"dtheta_out": 2.0 * math.pi}, "dtheta_out"),
        ({"theta_in": math.nan}, "theta_in"),
    ],
)
def test_wedge_rejects_invalid_geometry(kwargs, match):
    params = {
        "r_in": 1.0,
        "r_out": 4.0,
        "theta_in": 0.0,
        "dtheta_in": 0.5,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        apers.WedgeAp((5.0, 5.0), **params)

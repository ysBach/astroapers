from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astroapers as apers


def _selected_coords(mask):
    yy, xx = np.nonzero(mask)
    return set(zip(xx.tolist(), yy.tolist(), strict=True))


def _assert_center_mask_matches_photutils(astro_aperture, photutils_aperture, shape):
    astro = astro_aperture.get_apmask("center").to_image(shape) > 0
    photutils = photutils_aperture.to_mask(method="center").to_image(shape).astype(bool)

    assert _selected_coords(astro) == _selected_coords(photutils)
    return astro


def test_circular_aperture_mask_matches_sum_kernel():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.CircAp((4.3, 5.1), r=2.4)

    apsum, npix = aperture.apsum(data)
    direct_apsum, direct_npix = apers.apsum_circ_exact(data, [4.3], [5.1], 2.4)

    assert_allclose(apsum, direct_apsum[0])
    assert_allclose(npix, direct_npix[0])
    apm = aperture.get_apmask()
    assert_allclose(apm.to_image(data.shape).sum(), npix)


def test_circular_aperture_mask_matches_photutils_overlap_grid():
    circular_overlap_grid = pytest.importorskip(
        "photutils.geometry"
    ).circular_overlap_grid
    aperture = apers.CircAp((4.3, 5.1), r=2.4)

    apm = aperture.get_apmask()
    bbox = apm.bbox
    expected = circular_overlap_grid(
        bbox.ixmin - 0.5 - 4.3,
        bbox.ixmax - 0.5 - 4.3,
        bbox.iymin - 0.5 - 5.1,
        bbox.iymax - 0.5 - 5.1,
        bbox.ixmax - bbox.ixmin,
        bbox.iymax - bbox.iymin,
        2.4,
        1,
        5,
    )

    assert_allclose(apm.weights, expected, rtol=0, atol=1e-14)


def test_get_apmask_exposes_raw_weights_on_wrapper():
    aperture = apers.CircAp((4.3, 5.1), r=2.4)

    apm = aperture.get_apmask()

    assert isinstance(apm.weights, np.ndarray)
    assert apm.weights.dtype == np.float64
    assert not hasattr(apm, "area")


def test_apmask_raw_weights_with_bbox_supports_general_sum():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.CircAp((4.3, 5.1), r=2.4)

    apm = aperture.get_apmask()
    weights = apm.weights
    bbox = apm.bbox
    raw_apsum, raw_npix = apers.apmask_apsum(weights, bbox, data)
    apm_apsum, apm_npix = apm.apsum(data)

    assert_allclose(raw_apsum, apm_apsum)
    assert_allclose(raw_npix, apm_npix)
    assert_allclose(
        apers.apmask_apsum(weights, bbox, data, return_npix=False), raw_apsum
    )
    assert_allclose(apm.apsum(data, return_npix=False), apm_apsum)


def test_apmask_npix_tracks_area_clipping_and_bad_pixel_mask():
    data_shape = (8, 8)
    aperture = apers.CircAp((1.0, 1.0), r=3.0)

    apm = aperture.get_apmask()
    full_area = apm.weights.sum()
    clipped_npix = apm.npix(data_shape)
    raw_clipped_npix = apers.apmask_npix(apm.weights, apm.bbox, data_shape)

    assert hasattr(apers, "apmask_npix")
    assert_allclose(full_area, apm.weights.sum())
    assert clipped_npix < full_area
    assert_allclose(raw_clipped_npix, clipped_npix)
    assert_allclose(apm.apsum(np.ones(data_shape))[1], clipped_npix)

    bad = np.zeros(data_shape, dtype=bool)
    bad[0, 0] = True
    assert apm.npix(data_shape, mask=bad) < clipped_npix
    assert_allclose(
        apm.apsum(np.ones(data_shape), mask=bad)[1], apm.npix(data_shape, mask=bad)
    )
    assert_allclose(
        apers.apmask_npix(apm.weights, apm.bbox, data_shape, mask=bad),
        apm.npix(data_shape, mask=bad),
    )

    all_bad = np.ones(data_shape, dtype=bool)
    assert apm.npix(data_shape, mask=all_bad) == 0.0
    assert apm.apsum(np.ones(data_shape), mask=all_bad) == (0.0, 0.0)


def test_apmask_npix_returns_zero_outside_image():
    weights = np.ones((2, 2), dtype=np.float64)
    bbox = apers.BoundingBox(ixmin=10, ixmax=12, iymin=10, iymax=12)

    assert apers.apmask_npix(weights, bbox, (5, 5)) == 0.0
    assert apers.ApMask(weights, bbox).npix((5, 5)) == 0.0
    assert apers.apmask_apsum(weights, bbox, np.ones((5, 5))) == (0.0, 0.0)
    assert apers.apmask_apsum(weights, bbox, np.ones((5, 5)), return_npix=False) == 0.0


def test_apmask_raw_weights_annulus_supports_weighted_values():
    data = np.arange(12 * 12, dtype=np.float64).reshape(12, 12)
    annulus = apers.CircAn((5.5, 6.0), r_in=2.0, r_out=4.0)

    apm = annulus.get_apmask(method="center")
    values = apers.apmask_weighted_values(apm.weights, apm.bbox, data)
    apm_values = apm.weighted_values(data)

    assert_allclose(values, apm_values)


def test_raw_weights_with_bbox_supports_general_to_image_and_multiply():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.RectAp((4.0, 5.0), w=4.0, h=3.0, theta=0.2)

    apm = aperture.get_apmask()
    weights = apm.weights
    bbox = apm.bbox

    assert hasattr(apers, "apmask_to_image")
    assert_allclose(
        apers.apmask_to_image(weights, bbox, data.shape),
        apm.to_image(data.shape),
    )
    assert not hasattr(apm, "to_ndarray")
    assert not hasattr(apers, "apmask_to_ndarray")
    assert_allclose(
        apers.apmask_weighted_cutout(weights, bbox, data),
        apm.weighted_cutout(data),
    )


def test_user_supplied_float32_apmask_preserves_float32_work_arrays():
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    weights = np.array([[0.25, 1.0], [0.5, 0.0]], dtype=np.float32)
    bbox = apers.BoundingBox(ixmin=1, ixmax=3, iymin=2, iymax=4)

    apm = apers.ApMask(weights, bbox)
    image = apm.to_image(data.shape)
    cutout = apm.weighted_cutout(data)
    values = apm.weighted_values(data)
    apsum, npix = apm.apsum(data)

    assert apm.weights.dtype == np.float32
    assert image.dtype == np.float32
    assert cutout.dtype == np.float32
    assert values.dtype == np.float32
    expected = data[2:4, 1:3] * weights
    assert_allclose(cutout, expected)
    assert_allclose(values, expected[weights > 0.0])
    assert_allclose(
        apsum,
        np.sum(data[2:4, 1:3].astype(np.float64) * weights.astype(np.float64)),
    )
    assert_allclose(npix, np.sum(weights, dtype=np.float64))


def test_float32_apmask_reductions_accumulate_in_float64():
    data = np.full((1, 100000), 1.1, dtype=np.float32)
    weights = np.full((1, 100000), 0.1, dtype=np.float32)
    bbox = apers.BoundingBox(ixmin=0, ixmax=100000, iymin=0, iymax=1)
    apm = apers.ApMask(weights, bbox)

    apsum, npix = apm.apsum(data)

    expected_apsum = np.sum(
        data.astype(np.float64) * weights.astype(np.float64),
        dtype=np.float64,
    )
    expected_npix = np.sum(weights, dtype=np.float64)
    float32_apsum = np.sum(data * weights, dtype=np.float32)
    float32_npix = np.sum(weights, dtype=np.float32)

    assert apm.weights.dtype == np.float32
    assert_allclose(apsum, expected_apsum, rtol=0, atol=0)
    assert_allclose(npix, expected_npix, rtol=0, atol=0)
    assert apsum != float(float32_apsum)
    assert npix != float(float32_npix)
    assert apm.apsum(data, return_npix=False) == apsum
    assert apm.npix(data.shape) == npix


def test_user_supplied_non_float32_float64_apmask_weights_convert_to_float64():
    dtype = getattr(np, "float128", np.longdouble)
    weights = np.array([[0.25, 1.0], [0.5, 0.0]], dtype=dtype)
    bbox = apers.BoundingBox(ixmin=1, ixmax=3, iymin=2, iymax=4)

    apm = apers.ApMask(weights, bbox)
    image = apm.to_image((5, 5))

    assert apm.weights.dtype == np.float64
    assert image.dtype == np.float64
    assert_allclose(apm.weights, weights.astype(np.float64))


@pytest.mark.parametrize("dtype", [np.bool_, np.uint16, np.int64])
def test_user_supplied_non_float_apmask_weights_convert_to_float64(dtype):
    weights = np.array([[0, 1], [1, 0]], dtype=dtype)
    bbox = apers.BoundingBox(ixmin=1, ixmax=3, iymin=2, iymax=4)

    apm = apers.ApMask(weights, bbox)

    assert apm.weights.dtype == np.float64
    assert_allclose(apm.weights, weights.astype(np.float64))


def test_vector_raw_masks_align_with_vector_bboxes():
    data = np.ones((16, 16), dtype=np.float64)
    aperture = apers.CircAp([(4.0, 5.0), (10.0, 8.0)], r=2.5)

    apms = aperture.get_apmask()

    assert isinstance(apms, list)
    assert len(apms) == 2
    for apm in apms:
        assert_allclose(
            apers.apmask_apsum(apm.weights, apm.bbox, data), apm.apsum(data)
        )


def test_vector_bboxes_match_scalar_bboxes_for_core_shapes():
    positions = np.array([(4.0, 5.0), (10.0, 8.0), (12.5, 3.5)])
    cases = [
        (apers.CircAp(positions, r=2.5), lambda x, y: apers.CircAp((x, y), r=2.5)),
        (
            apers.EllipAp(positions, a=4.0, b=2.0, theta=0.3),
            lambda x, y: apers.EllipAp((x, y), a=4.0, b=2.0, theta=0.3),
        ),
        (
            apers.RectAp(positions, w=5.0, h=3.0, theta=-0.2),
            lambda x, y: apers.RectAp((x, y), w=5.0, h=3.0, theta=-0.2),
        ),
        (
            apers.CircAn(positions, r_in=1.0, r_out=3.0),
            lambda x, y: apers.CircAn((x, y), r_in=1.0, r_out=3.0),
        ),
        (
            apers.EllipAn(
                positions, a_in=2.0, b_in=1.0, a_out=4.0, b_out=2.0, theta_in=0.3
            ),
            lambda x, y: apers.EllipAn(
                (x, y), a_in=2.0, b_in=1.0, a_out=4.0, b_out=2.0, theta_in=0.3
            ),
        ),
        (
            apers.RectAn(
                positions, w_in=2.0, h_in=1.0, w_out=5.0, h_out=3.0, theta_in=-0.2
            ),
            lambda x, y: apers.RectAn(
                (x, y), w_in=2.0, h_in=1.0, w_out=5.0, h_out=3.0, theta_in=-0.2
            ),
        ),
    ]

    for vector_aperture, scalar_factory in cases:
        assert vector_aperture.bbox == [
            scalar_factory(float(x), float(y)).bbox for x, y in positions
        ]


def test_scalar_bboxes_match_rust_vector_bboxes_at_boundaries():
    positions = np.array([(-0.5, -0.5), (0.0, 0.0), (4.5, 5.5), (11.25, 3.75)])
    cases = [
        (apers.CircAp(positions, r=2.5), lambda x, y: apers.CircAp((x, y), r=2.5)),
        (
            apers.EllipAp(positions, a=3.0, b=1.5, theta=0.0),
            lambda x, y: apers.EllipAp((x, y), a=3.0, b=1.5, theta=0.0),
        ),
        (
            apers.EllipAp(positions, a=3.0, b=1.5, theta=np.pi / 2.0 - 1.0e-12),
            lambda x, y: apers.EllipAp(
                (x, y), a=3.0, b=1.5, theta=np.pi / 2.0 - 1.0e-12
            ),
        ),
        (
            apers.EllipAp(positions, a=3.0, b=1.5, theta=0.37),
            lambda x, y: apers.EllipAp((x, y), a=3.0, b=1.5, theta=0.37),
        ),
        (
            apers.RectAp(positions, w=5.0, h=2.0, theta=0.0),
            lambda x, y: apers.RectAp((x, y), w=5.0, h=2.0, theta=0.0),
        ),
        (
            apers.RectAp(positions, w=5.0, h=2.0, theta=np.pi / 2.0 - 1.0e-12),
            lambda x, y: apers.RectAp(
                (x, y), w=5.0, h=2.0, theta=np.pi / 2.0 - 1.0e-12
            ),
        ),
        (
            apers.RectAp(positions, w=5.0, h=2.0, theta=-0.41),
            lambda x, y: apers.RectAp((x, y), w=5.0, h=2.0, theta=-0.41),
        ),
    ]

    for vector_aperture, scalar_factory in cases:
        assert vector_aperture.bbox == [
            scalar_factory(float(x), float(y)).bbox for x, y in positions
        ]


def test_vector_circle_apmask_exposes_weights():
    aperture = apers.CircAp([(4.3, 5.1), (8.7, 9.2), (12.4, 3.5)], r=2.4)

    apms = aperture.get_apmask()

    assert isinstance(apms, list)
    assert len(apms) == 3
    assert all(isinstance(apm.weights, np.ndarray) for apm in apms)


@pytest.mark.parametrize(
    ("aperture_cls", "params"),
    [
        (apers.CircAp, (2.4,)),
        (apers.EllipAp, (3.5, 1.8, 0.4)),
        (apers.RectAp, (4.0, 2.5, -0.3)),
        (apers.PillAp, (5.0, 1.6, 1.1, 0.35)),
    ],
)
@pytest.mark.parametrize("method", ["exact", "center"])
def test_vector_many_masks_match_scalar_masks(aperture_cls, params, method):
    positions = np.array([(4.3, 5.1), (8.7, 9.2), (12.4, 3.5)], dtype=np.float64)
    vector_aperture = aperture_cls(positions, *params)

    apms = vector_aperture.get_apmask(method=method)
    weights_list = [apm.weights for apm in apms]

    assert isinstance(weights_list, list)
    assert isinstance(apms, list)
    assert len(weights_list) == len(apms) == len(positions)
    assert vector_aperture.bbox == [apm.bbox for apm in apms]
    for weights, apm, (x, y) in zip(weights_list, apms, positions):
        scalar_apm = aperture_cls((float(x), float(y)), *params).get_apmask(
            method=method
        )
        assert weights.dtype == np.float64
        assert weights.flags.c_contiguous
        assert weights.shape == scalar_apm.weights.shape
        assert apm.bbox == scalar_apm.bbox
        assert_allclose(weights, scalar_apm.weights, rtol=0, atol=0)
        assert_allclose(apm.weights, scalar_apm.weights, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("annulus_cls", "params", "kwargs"),
    [
        (apers.CircAn, (1.3, 3.4), {}),
        (apers.EllipAn, (1.2, 0.8, 3.8, 2.4), {"theta_in": 0.4}),
        (apers.RectAn, (1.5, 0.9, 4.0, 2.7), {"theta_in": -0.3}),
        (apers.PillAn, (3.0, 0.8, 0.55, 5.0, 1.8, 1.2), {"theta_in": 0.35}),
    ],
)
@pytest.mark.parametrize("method", ["exact", "center"])
def test_vector_many_annulus_masks_match_scalar_masks(
    annulus_cls, params, kwargs, method
):
    positions = np.array([(4.3, 5.1), (8.7, 9.2), (12.4, 3.5)], dtype=np.float64)
    vector_annulus = annulus_cls(positions, *params, **kwargs)

    apms = vector_annulus.get_apmask(method=method)
    weights_list = [apm.weights for apm in apms]

    assert isinstance(weights_list, list)
    assert isinstance(apms, list)
    assert len(weights_list) == len(apms) == len(positions)
    assert vector_annulus.bbox == [apm.bbox for apm in apms]
    for weights, apm, (x, y) in zip(weights_list, apms, positions):
        scalar_apm = annulus_cls((float(x), float(y)), *params, **kwargs).get_apmask(
            method=method
        )
        assert weights.dtype == np.float64
        assert weights.flags.c_contiguous
        assert weights.shape == scalar_apm.weights.shape
        assert apm.bbox == scalar_apm.bbox
        assert_allclose(weights, scalar_apm.weights, rtol=0, atol=0)
        assert_allclose(apm.weights, scalar_apm.weights, rtol=0, atol=0)


def test_vector_get_apmask_returns_list_of_mask_objects():
    aperture = apers.RectAp([(4.0, 4.0), (8.5, 9.0)], w=3.0, h=2.0)

    masks = aperture.get_apmask()

    assert isinstance(masks, list)
    assert len(masks) == 2
    assert all(isinstance(apm.weights, np.ndarray) for apm in masks)


def test_get_apmask_center_exposes_center_weights():
    aperture = apers.CircAn((3.0, 3.0), r_in=1.1, r_out=2.1)

    assert aperture.get_apmask(method="center").weights.dtype == np.float64


def test_center_mask_marks_pixel_centers_inside_aperture():
    aperture = apers.CircAp((2.0, 2.0), r=1.1)
    center_apm = aperture.get_apmask(method="center")
    image = center_apm.to_image((5, 5))

    expected = np.zeros((5, 5), dtype=np.float64)
    expected[2, 2] = 1.0
    expected[1, 2] = 1.0
    expected[2, 1] = 1.0
    expected[2, 3] = 1.0
    expected[3, 2] = 1.0
    assert_allclose(image, expected)


def test_circular_center_mask_matches_photutils_boundary_convention():
    photutils_aperture = pytest.importorskip("photutils.aperture")
    shape = (32, 32)
    pos = (10, 10)
    radius = 5

    astro = apers.CircAp(pos, radius).get_apmask("center").to_image(shape) > 0
    photutils = (
        photutils_aperture.CircularAperture(pos, r=radius)
        .to_mask(method="center")
        .to_image(shape)
        .astype(bool)
    )

    assert _selected_coords(astro) == _selected_coords(photutils)
    assert not astro[10, 5]


def test_center_mask_marks_pixel_centers_inside_annulus():
    annulus = apers.CircAn((3.0, 3.0), r_in=1.1, r_out=2.1)
    image = annulus.get_apmask(method="center").to_image((7, 7))

    yy, xx = np.indices((7, 7), dtype=np.float64)
    r2 = (xx - 3.0) ** 2 + (yy - 3.0) ** 2
    expected = ((r2 < 2.1**2) & (r2 >= 1.1**2)).astype(np.float64)
    assert_allclose(image, expected)


def test_circular_annulus_center_mask_matches_photutils_boundary_convention():
    photutils_aperture = pytest.importorskip("photutils.aperture")
    shape = (32, 32)
    pos = (10, 10)
    r_in = 5
    r_out = 7

    astro = apers.CircAn(pos, r_in, r_out).get_apmask("center").to_image(shape) > 0
    photutils = (
        photutils_aperture.CircularAnnulus(pos, r_in=r_in, r_out=r_out)
        .to_mask(method="center")
        .to_image(shape)
        .astype(bool)
    )

    assert _selected_coords(astro) == _selected_coords(photutils)
    assert astro.sum() == 76
    assert astro[10, 5]
    assert not astro[10, 3]


def test_elliptical_center_mask_matches_photutils_boundary_convention():
    photutils_aperture = pytest.importorskip("photutils.aperture")
    shape = (32, 32)
    pos = (10, 10)
    a = 5
    b = 3

    astro = _assert_center_mask_matches_photutils(
        apers.EllipAp(pos, a, b, 0.0),
        photutils_aperture.EllipticalAperture(pos, a=a, b=b, theta=0.0),
        shape,
    )

    assert astro.sum() == 41
    assert not astro[10, 5]


def test_elliptical_annulus_center_mask_matches_photutils_boundary_convention():
    photutils_aperture = pytest.importorskip("photutils.aperture")
    shape = (32, 32)
    pos = (10, 10)
    a_in = 3
    b_in = 2
    a_out = 5
    b_out = 3

    astro = _assert_center_mask_matches_photutils(
        apers.EllipAn(pos, a_in, b_in, a_out, b_out, theta_in=0.0),
        photutils_aperture.EllipticalAnnulus(
            pos, a_in=a_in, b_in=b_in, a_out=a_out, b_out=b_out, theta=0.0
        ),
        shape,
    )

    assert astro.sum() == 26
    assert astro[10, 7]
    assert not astro[10, 5]


def test_rectangular_center_mask_matches_photutils_boundary_convention():
    photutils_aperture = pytest.importorskip("photutils.aperture")
    shape = (32, 32)
    pos = (10, 10)
    w = 6
    h = 4

    astro = _assert_center_mask_matches_photutils(
        apers.RectAp(pos, w, h, 0.0),
        photutils_aperture.RectangularAperture(pos, w=w, h=h, theta=0.0),
        shape,
    )

    assert astro.sum() == 15
    assert not astro[10, 7]


def test_rectangular_annulus_center_mask_matches_photutils_boundary_convention():
    photutils_aperture = pytest.importorskip("photutils.aperture")
    shape = (32, 32)
    pos = (10, 10)
    w_in = 4
    h_in = 2
    w_out = 6
    h_out = 4

    astro = _assert_center_mask_matches_photutils(
        apers.RectAn(pos, w_in, h_in, w_out, h_out, theta_in=0.0),
        photutils_aperture.RectangularAnnulus(
            pos, w_in=w_in, h_in=h_in, w_out=w_out, h_out=h_out, theta=0.0
        ),
        shape,
    )

    assert astro.sum() == 12
    assert astro[10, 8]
    assert not astro[10, 7]


def test_non_circular_center_masks_match_photutils_rotated_edge_cases():
    photutils_aperture = pytest.importorskip("photutils.aperture")
    shape = (32, 32)
    pos = (10, 10)
    theta = np.pi / 2.0

    cases = [
        (
            apers.EllipAp(pos, 5, 3, theta),
            photutils_aperture.EllipticalAperture(pos, a=5, b=3, theta=theta),
        ),
        (
            apers.EllipAn(pos, 3, 2, 5, 3, theta_in=theta),
            photutils_aperture.EllipticalAnnulus(
                pos, a_in=3, b_in=2, a_out=5, b_out=3, theta=theta
            ),
        ),
        (
            apers.RectAp(pos, 6, 4, theta),
            photutils_aperture.RectangularAperture(pos, w=6, h=4, theta=theta),
        ),
        (
            apers.RectAn(pos, 4, 2, 6, 4, theta_in=theta),
            photutils_aperture.RectangularAnnulus(
                pos, w_in=4, h_in=2, w_out=6, h_out=4, theta=theta
            ),
        ),
    ]

    for astro_aperture, photutils_aperture in cases:
        _assert_center_mask_matches_photutils(astro_aperture, photutils_aperture, shape)


def test_circular_center_sum_and_npix_follow_center_mask_boundary():
    data = np.ones((32, 32), dtype=np.float64)
    x = np.array([10.0])
    y = np.array([10.0])

    mask = apers.CircAp((10, 10), 5).get_apmask("center").to_image(data.shape)
    apsum, npix = apers.apsum_circ_center(data, x, y, 5)
    npix_only = apers.npix_circ_center(x, y, 5, shape=data.shape)

    assert_allclose(apsum[0], mask.sum())
    assert_allclose(npix[0], mask.sum())
    assert_allclose(npix_only[0], mask.sum())


def test_non_circular_center_sum_and_npix_follow_center_mask_boundary():
    data = np.ones((32, 32), dtype=np.float64)
    x = np.array([10.0])
    y = np.array([10.0])

    cases = [
        (
            apers.EllipAp((10, 10), 5, 3, 0.0),
            apers.apsum_ellip_center,
            apers.npix_ellip_center,
            (5, 3, 0.0),
        ),
        (
            apers.RectAp((10, 10), 6, 4, 0.0),
            apers.apsum_rect_center,
            apers.npix_rect_center,
            (6, 4, 0.0),
        ),
    ]

    for aperture, apsum_func, npix_func, params in cases:
        mask = aperture.get_apmask("center").to_image(data.shape)
        apsum, npix = apsum_func(data, x, y, *params)
        npix_only = npix_func(x, y, *params, shape=data.shape)

        assert_allclose(apsum[0], mask.sum())
        assert_allclose(npix[0], mask.sum())
        assert_allclose(npix_only[0], mask.sum())

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astroapers as apers
from astroapers import _containers


def _selected_coords(mask):
    yy, xx = np.nonzero(mask)
    return set(zip(xx.tolist(), yy.tolist(), strict=True))


def _weights_box(aperture, method: str = "exact", idx: int = 0):
    weights = (
        aperture.weights_exact() if method == "exact" else aperture.weights_center()
    )
    return weights[idx], aperture.bboxes()[idx]


def _to_image(aperture, shape, method: str = "exact", idx: int = 0):
    weights, bbox = _weights_box(aperture, method=method, idx=idx)
    return bbox.to_image(weights, shape)


def _assert_center_mask_matches_photutils(astro_aperture, photutils_aperture, shape):
    astro = _to_image(astro_aperture, shape, method="center") > 0
    photutils = photutils_aperture.to_mask(method="center").to_image(shape).astype(bool)

    assert _selected_coords(astro) == _selected_coords(photutils)
    return astro


def test_circular_aperture_mask_matches_sum_kernel():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.CircAp((4.3, 5.1), r=2.4)

    apsum, npix = aperture.apsum_exact(data)
    direct_apsum, direct_npix = apers.apsum_circ_exact(data, [4.3], [5.1], 2.4)

    assert_allclose(apsum, direct_apsum[0])
    assert_allclose(npix, direct_npix[0])
    assert_allclose(_to_image(aperture, data.shape).sum(), npix)


def test_circular_aperture_mask_matches_photutils_overlap_grid():
    circular_overlap_grid = pytest.importorskip(
        "photutils.geometry"
    ).circular_overlap_grid
    aperture = apers.CircAp((4.3, 5.1), r=2.4)

    weights, bbox = _weights_box(aperture)
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

    assert_allclose(weights, expected, rtol=0, atol=1e-14)


def test_weights_exposes_raw_weight_arrays():
    aperture = apers.CircAp((4.3, 5.1), r=2.4)

    weights = aperture.weights_exact()[0]

    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    assert not hasattr(weights, "area")


def test_bounding_box_raw_weights_support_general_sum():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.CircAp((4.3, 5.1), r=2.4)

    weights, bbox = _weights_box(aperture)
    raw_apsum, raw_npix = bbox.apsum(weights, data)
    object_apsum, object_npix = aperture.apsum_exact(data)

    assert_allclose(raw_apsum, object_apsum)
    assert_allclose(raw_npix, object_npix)
    assert_allclose(bbox.apsum(weights, data, return_npix=False), raw_apsum)
    assert_allclose(aperture.apsum_exact(data, return_npix=False), object_apsum)


def test_bounding_box_methods_apply_bbox_tight_weights_exact():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.CircAp((4.3, 5.1), r=2.4)

    weights, bbox = _weights_box(aperture)

    assert_allclose(
        bbox.apsum(weights, data, return_npix=False),
        aperture.apsum_exact(data, return_npix=False),
    )
    assert_allclose(bbox.apsum(weights, data), aperture.apsum_exact(data))
    assert_allclose(bbox.npix(weights, data.shape), aperture.npix_exact(data.shape))
    assert_allclose(bbox.to_image(weights, data.shape), _to_image(aperture, data.shape))
    assert_allclose(
        bbox.weighted_values(weights, data),
        aperture.weighted_values(data)[0],
    )


def test_bbox_npix_tracks_area_clipping_and_bad_pixel_mask():
    data_shape = (8, 8)
    aperture = apers.CircAp((1.0, 1.0), r=3.0)

    weights, bbox = _weights_box(aperture)
    full_area = weights.sum()
    clipped_npix = bbox.npix(weights, data_shape)
    raw_clipped_npix = bbox.npix(weights, data_shape)

    assert_allclose(full_area, weights.sum())
    assert clipped_npix < full_area
    assert_allclose(raw_clipped_npix, clipped_npix)
    assert_allclose(bbox.apsum(weights, np.ones(data_shape))[1], clipped_npix)

    bad = np.zeros(data_shape, dtype=bool)
    bad[0, 0] = True
    assert bbox.npix(weights, data_shape, mask=bad) < clipped_npix
    assert_allclose(
        bbox.apsum(weights, np.ones(data_shape), mask=bad)[1],
        bbox.npix(weights, data_shape, mask=bad),
    )
    assert_allclose(
        bbox.npix(weights, data_shape, mask=bad),
        aperture.npix_exact(data_shape, mask=bad),
    )

    all_bad = np.ones(data_shape, dtype=bool)
    assert bbox.npix(weights, data_shape, mask=all_bad) == 0.0
    assert bbox.apsum(weights, np.ones(data_shape), mask=all_bad) == (0.0, 0.0)


def test_bbox_npix_returns_zero_outside_image():
    weights = np.ones((2, 2), dtype=np.float64)
    bbox = apers.BoundingBox(ixmin=10, ixmax=12, iymin=10, iymax=12)

    assert bbox.npix(weights, (5, 5)) == 0.0
    assert bbox.apsum(weights, np.ones((5, 5))) == (0.0, 0.0)
    assert bbox.apsum(weights, np.ones((5, 5)), return_npix=False) == 0.0


def test_bbox_to_fits_section_returns_xy_1_indexed_inclusive_section():
    bbox = apers.BoundingBox(ixmin=1, ixmax=3, iymin=2, iymax=4)

    assert bbox.to_fits_section() == "[2:3,3:4]"


def test_bbox_to_fits_section_clips_to_shape():
    bbox = apers.BoundingBox(ixmin=-2, ixmax=3, iymin=4, iymax=8)

    assert bbox.to_fits_section((6, 10)) == "[1:3,5:6]"


def test_bbox_to_fits_section_requires_shape_for_negative_raw_box():
    bbox = apers.BoundingBox(ixmin=-1, ixmax=2, iymin=0, iymax=2)

    with pytest.raises(ValueError, match="shape is required"):
        bbox.to_fits_section()


def test_bbox_to_fits_section_rejects_no_overlap_and_degenerate_boxes():
    outside = apers.BoundingBox(ixmin=10, ixmax=12, iymin=10, iymax=12)
    degenerate = apers.BoundingBox(ixmin=2, ixmax=2, iymin=1, iymax=3)

    with pytest.raises(ValueError, match="does not overlap"):
        outside.to_fits_section((5, 5))
    with pytest.raises(ValueError, match="positive x and y extents"):
        degenerate.to_fits_section()


def test_bbox_to_fits_section_matches_astro_ndslice_when_available():
    astro_ndslice = pytest.importorskip("astro_ndslice")
    bbox = apers.BoundingBox(ixmin=-2, ixmax=3, iymin=4, iymax=8)
    data_slices, _ = bbox.overlap_slices((6, 10))

    assert bbox.to_fits_section((6, 10)) == astro_ndslice.slice_to_string(
        data_slices,
        fits_convention=True,
    )


def test_bbox_raw_weights_annulus_supports_weighted_values():
    data = np.arange(12 * 12, dtype=np.float64).reshape(12, 12)
    annulus = apers.CircAn((5.5, 6.0), r_in=2.0, r_out=4.0)

    weights, bbox = _weights_box(annulus, method="center")
    values = bbox.weighted_values(weights, data)
    object_values = annulus.sampled_values(data)[0]

    assert_allclose(values, object_values)


def test_raw_weights_with_bbox_supports_general_to_image_and_multiply():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.RectAp((4.0, 5.0), w=4.0, h=3.0, theta=0.2)

    weights, bbox = _weights_box(aperture)

    assert_allclose(
        bbox.to_image(weights, data.shape),
        _to_image(aperture, data.shape),
    )
    assert not hasattr(apers, "apmask_to_ndarray")
    assert_allclose(
        bbox.weighted_cutout(weights, data),
        bbox.weighted_cutout(weights, data),
    )


def test_bbox_weighted_cutout_uses_fill_value_for_bad_pixel_mask():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    weights = np.ones((3, 3), dtype=np.float64)
    bbox = apers.BoundingBox(ixmin=1, ixmax=4, iymin=1, iymax=4)
    bad = np.zeros_like(data, dtype=bool)
    bad[2, 2] = True

    cutout = bbox.weighted_cutout(weights, data, mask=bad, fill_value=-999.0)

    expected = data[1:4, 1:4].astype(np.float64)
    expected[1, 1] = -999.0
    assert_allclose(cutout, expected)


def test_user_supplied_float32_weights_preserve_float32_work_arrays():
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    weights = np.array([[0.25, 1.0], [0.5, 0.0]], dtype=np.float32)
    bbox = apers.BoundingBox(ixmin=1, ixmax=3, iymin=2, iymax=4)

    image = bbox.to_image(weights, data.shape)
    cutout = bbox.weighted_cutout(weights, data)
    values = bbox.weighted_values(weights, data)
    apsum, npix = bbox.apsum(weights, data)

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


def test_float32_bbox_reductions_accumulate_in_float64():
    data = np.full((1, 100000), 1.1, dtype=np.float32)
    weights = np.full((1, 100000), 0.1, dtype=np.float32)
    bbox = apers.BoundingBox(ixmin=0, ixmax=100000, iymin=0, iymax=1)

    apsum, npix = bbox.apsum(weights, data)

    expected_apsum = np.sum(
        data.astype(np.float64) * weights.astype(np.float64),
        dtype=np.float64,
    )
    expected_npix = np.sum(weights, dtype=np.float64)
    float32_apsum = np.sum(data * weights, dtype=np.float32)
    float32_npix = np.sum(weights, dtype=np.float32)

    eps = np.finfo(np.float64).eps
    assert_allclose(
        apsum, expected_apsum, rtol=0, atol=eps * data.size * abs(expected_apsum)
    )
    assert_allclose(
        npix, expected_npix, rtol=0, atol=eps * weights.size * abs(expected_npix)
    )
    assert apsum != float(float32_apsum)
    assert npix != float(float32_npix)
    assert bbox.apsum(weights, data, return_npix=False) == apsum
    assert bbox.npix(weights, data.shape) == npix


def test_bbox_apsum_uses_fused_reducers_weighted_sum(monkeypatch):
    bbox = apers.BoundingBox(1, 3, 2, 4)
    weights = np.ones(bbox.shape, dtype=np.float64)
    data = np.ones((6, 6), dtype=np.float64)
    calls = []

    def fake_sum(values, *, weights=None, return_sum_weights=False, validate):
        calls.append((values, weights, return_sum_weights, validate))
        weighted_sum = np.sum(values * weights, dtype=np.float64)
        if return_sum_weights:
            return weighted_sum, np.sum(weights, dtype=np.float64)
        return weighted_sum

    monkeypatch.setattr(_containers, "rd", type("FakeReducers", (), {"sum": fake_sum}))

    assert bbox.apsum(weights, data) == (4.0, 4.0)
    assert bbox.apsum(weights, data, return_npix=False) == 4.0
    assert len(calls) == 2
    assert calls[0][1] is not None
    assert calls[0][2] is True
    assert calls[0][3] is False
    assert calls[1][1] is not None
    assert calls[1][2] is False
    assert calls[1][3] is False


def test_user_supplied_non_float32_float64_weights_convert_to_float64():
    dtype = getattr(np, "float128", np.longdouble)
    weights = np.array([[0.25, 1.0], [0.5, 0.0]], dtype=dtype)
    bbox = apers.BoundingBox(ixmin=1, ixmax=3, iymin=2, iymax=4)

    image = bbox.to_image(weights, (5, 5))

    assert image.dtype == np.float64
    assert_allclose(image[2:4, 1:3], weights.astype(np.float64))


@pytest.mark.parametrize("dtype", [np.bool_, np.uint16, np.int64])
def test_user_supplied_non_float_weights_convert_to_float64(dtype):
    weights = np.array([[0, 1], [1, 0]], dtype=dtype)
    bbox = apers.BoundingBox(ixmin=1, ixmax=3, iymin=2, iymax=4)

    image = bbox.to_image(weights, (5, 5))

    assert image.dtype == np.float64
    assert_allclose(image[2:4, 1:3], weights.astype(np.float64))


def test_vector_raw_weights_align_with_vector_bboxes():
    data = np.ones((16, 16), dtype=np.float64)
    aperture = apers.CircAp([(4.0, 5.0), (10.0, 8.0)], r=2.5)

    weights = aperture.weights_exact()
    boxes = aperture.bboxes()

    assert isinstance(weights, list)
    assert isinstance(boxes, list)
    assert len(weights) == len(boxes) == 2
    for weight, bbox in zip(weights, boxes, strict=True):
        assert_allclose(bbox.apsum(weight, data), (weight.sum(), weight.sum()))


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
        assert vector_aperture.bboxes() == [
            scalar_factory(float(x), float(y)).bboxes()[0] for x, y in positions
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
        assert vector_aperture.bboxes() == [
            scalar_factory(float(x), float(y)).bboxes()[0] for x, y in positions
        ]


def test_vector_circle_weights_are_lists():
    aperture = apers.CircAp([(4.3, 5.1), (8.7, 9.2), (12.4, 3.5)], r=2.4)

    weights = aperture.weights_exact()

    assert isinstance(weights, list)
    assert len(weights) == 3
    assert all(isinstance(weight, np.ndarray) for weight in weights)


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
def test_vector_many_weights_match_scalar_weights(aperture_cls, params, method):
    positions = np.array([(4.3, 5.1), (8.7, 9.2), (12.4, 3.5)], dtype=np.float64)
    vector_aperture = aperture_cls(positions, *params)

    weights_list = (
        vector_aperture.weights_exact()
        if method == "exact"
        else vector_aperture.weights_center()
    )
    boxes = vector_aperture.bboxes()

    assert isinstance(weights_list, list)
    assert isinstance(boxes, list)
    assert len(weights_list) == len(boxes) == len(positions)
    for weights, bbox, (x, y) in zip(weights_list, boxes, positions, strict=True):
        scalar = aperture_cls((float(x), float(y)), *params)
        scalar_weights = (
            scalar.weights_exact() if method == "exact" else scalar.weights_center()
        )[0]
        scalar_bbox = scalar.bboxes()[0]
        assert weights.dtype == np.float64
        assert weights.flags.c_contiguous
        assert weights.shape == scalar_weights.shape
        assert bbox == scalar_bbox
        assert_allclose(weights, scalar_weights, rtol=0, atol=0)


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
def test_vector_many_annulus_weights_match_scalar_weights(
    annulus_cls, params, kwargs, method
):
    positions = np.array([(4.3, 5.1), (8.7, 9.2), (12.4, 3.5)], dtype=np.float64)
    vector_annulus = annulus_cls(positions, *params, **kwargs)

    weights_list = (
        vector_annulus.weights_exact()
        if method == "exact"
        else vector_annulus.weights_center()
    )
    boxes = vector_annulus.bboxes()

    assert isinstance(weights_list, list)
    assert isinstance(boxes, list)
    assert len(weights_list) == len(boxes) == len(positions)
    for weights, bbox, (x, y) in zip(weights_list, boxes, positions, strict=True):
        scalar = annulus_cls((float(x), float(y)), *params, **kwargs)
        scalar_weights = (
            scalar.weights_exact() if method == "exact" else scalar.weights_center()
        )[0]
        scalar_bbox = scalar.bboxes()[0]
        assert weights.dtype == np.float64
        assert weights.flags.c_contiguous
        assert weights.shape == scalar_weights.shape
        assert bbox == scalar_bbox
        assert_allclose(weights, scalar_weights, rtol=0, atol=0)


def test_vector_weights_returns_list_of_weight_arrays():
    aperture = apers.RectAp([(4.0, 4.0), (8.5, 9.0)], w=3.0, h=2.0)

    weights = aperture.weights_exact()

    assert isinstance(weights, list)
    assert len(weights) == 2
    assert all(isinstance(weight, np.ndarray) for weight in weights)


def test_weights_center_exposes_center_weights_exact():
    aperture = apers.CircAn((3.0, 3.0), r_in=1.1, r_out=2.1)

    assert aperture.weights_center()[0].dtype == np.float64


def test_center_mask_marks_pixel_centers_inside_aperture():
    aperture = apers.CircAp((2.0, 2.0), r=1.1)
    image = _to_image(aperture, (5, 5), method="center")

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

    astro = _to_image(apers.CircAp(pos, radius), shape, method="center") > 0
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
    image = _to_image(annulus, (7, 7), method="center")

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

    astro = _to_image(apers.CircAn(pos, r_in, r_out), shape, method="center") > 0
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

    mask = _to_image(apers.CircAp((10, 10), 5), data.shape, method="center")
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
        mask = _to_image(aperture, data.shape, method="center")
        apsum, npix = apsum_func(data, x, y, *params)
        npix_only = npix_func(x, y, *params, shape=data.shape)

        assert_allclose(apsum[0], mask.sum())
        assert_allclose(npix[0], mask.sum())
        assert_allclose(npix_only[0], mask.sum())

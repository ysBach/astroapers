from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astroapers as apers

STRICT_ATOL = 1.0e-12
AREA_ATOL = 1.0e-8
BENCHMARK_ATOL = 1.0e-5
BENCHMARK_RTOL = 1.0e-4
RECTANGLE_PHOTUTILS_RTOL = 2.0e-4
DTYPES = (np.float32, np.float64, np.int32, np.int16)
IMAGE_SHAPE = (72, 76)
POSITIONS = np.array(
    [
        (30.2, 31.7),
        (44.5, 45.2),
        (2.1, 3.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ApertureCase:
    name: str
    factory: Callable[..., object]
    exact_func: Callable[..., object]
    center_func: Callable[..., object]
    npix_exact_func: Callable[..., np.ndarray]
    npix_center_func: Callable[..., np.ndarray]
    params: tuple[float, ...]

    def aperture(self, positions=POSITIONS, *, validate: bool = True):
        return self.factory(positions, validate=validate)

    @property
    def area(self) -> float:
        return float(self.aperture(POSITIONS[:1]).area)


APERTURE_CASES = (
    ApertureCase(
        "circle",
        lambda positions, *, validate=True: apers.CircAp(
            positions, 4.0, validate=validate
        ),
        apers.apsum_circ_exact,
        apers.apsum_circ_center,
        apers.npix_circ_exact,
        apers.npix_circ_center,
        (4.0,),
    ),
    ApertureCase(
        "ellipse",
        lambda positions, *, validate=True: apers.EllipAp(
            positions, 5.0, 2.5, 0.35, validate=validate
        ),
        apers.apsum_ellip_exact,
        apers.apsum_ellip_center,
        apers.npix_ellip_exact,
        apers.npix_ellip_center,
        (5.0, 2.5, 0.35),
    ),
    ApertureCase(
        "rectangle",
        lambda positions, *, validate=True: apers.RectAp(
            positions, 7.0, 4.0, -0.25, validate=validate
        ),
        apers.apsum_rect_exact,
        apers.apsum_rect_center,
        apers.npix_rect_exact,
        apers.npix_rect_center,
        (7.0, 4.0, -0.25),
    ),
    ApertureCase(
        "pill",
        lambda positions, *, validate=True: apers.PillAp(
            positions, 6.0, 2.2, 1.4, 0.3, validate=validate
        ),
        apers.apsum_pill_exact,
        apers.apsum_pill_center,
        apers.npix_pill_exact,
        apers.npix_pill_center,
        (6.0, 2.2, 1.4, 0.3),
    ),
    ApertureCase(
        "circle_annulus",
        lambda positions, *, validate=True: apers.CircAn(
            positions, 2.0, 5.0, validate=validate
        ),
        apers.apsum_circ_ann_exact,
        apers.apsum_circ_ann_center,
        apers.npix_circ_ann_exact,
        apers.npix_circ_ann_center,
        (2.0, 5.0),
    ),
    ApertureCase(
        "ellipse_annulus",
        lambda positions, *, validate=True: apers.EllipAn(
            positions,
            2.0,
            1.0,
            5.0,
            2.5,
            theta_in=0.25,
            theta_out=-0.15,
            validate=validate,
        ),
        apers.apsum_ellip_ann_exact,
        apers.apsum_ellip_ann_center,
        apers.npix_ellip_ann_exact,
        apers.npix_ellip_ann_center,
        (2.0, 1.0, 5.0, 2.5, 0.25, -0.15),
    ),
    ApertureCase(
        "rectangle_annulus",
        lambda positions, *, validate=True: apers.RectAn(
            positions,
            2.0,
            1.0,
            7.0,
            4.0,
            theta_in=0.35,
            theta_out=-0.2,
            validate=validate,
        ),
        apers.apsum_rect_ann_exact,
        apers.apsum_rect_ann_center,
        apers.npix_rect_ann_exact,
        apers.npix_rect_ann_center,
        (2.0, 1.0, 7.0, 4.0, 0.35, -0.2),
    ),
    ApertureCase(
        "pill_annulus",
        lambda positions, *, validate=True: apers.PillAn(
            positions,
            2.0,
            0.8,
            0.55,
            6.0,
            2.2,
            1.4,
            theta_in=0.3,
            theta_out=-0.1,
            validate=validate,
        ),
        apers.apsum_pill_ann_exact,
        apers.apsum_pill_ann_center,
        apers.npix_pill_ann_exact,
        apers.npix_pill_ann_center,
        (2.0, 0.8, 0.55, 6.0, 2.2, 1.4, 0.3, -0.1),
    ),
)


def _sparse_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[31, 30] = True
    mask[45, 44] = True
    mask[:3, :4] = True
    return mask


def _expected_from_masks(
    aperture, data: np.ndarray, method: str, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    sums = []
    npixs = []
    weights = (
        aperture.weights_exact() if method == "exact" else aperture.weights_center()
    )
    for weight, bbox in zip(weights, aperture.bboxes(), strict=True):
        apsum, npix = bbox.apsum(weight, data, mask=mask)
        sums.append(apsum)
        npixs.append(npix)
    return np.asarray(sums), np.asarray(npixs)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: np.dtype(dtype).name)
@pytest.mark.parametrize("fill_value", [0, 1], ids=["zeros", "ones"])
@pytest.mark.parametrize("method", ["exact", "center"])
@pytest.mark.parametrize("mask_name", ["none", "sparse", "all_bad"])
@pytest.mark.parametrize("case", APERTURE_CASES, ids=lambda case: case.name)
def test_constant_images_cover_dtypes_methods_masks_and_shapes(
    case: ApertureCase,
    mask_name: str,
    method: str,
    fill_value: int,
    dtype,
):
    data = np.full(IMAGE_SHAPE, fill_value, dtype=dtype)
    x = POSITIONS[:, 0]
    y = POSITIONS[:, 1]
    mask = {
        "none": None,
        "sparse": _sparse_mask(IMAGE_SHAPE),
        "all_bad": np.ones(IMAGE_SHAPE, dtype=bool),
    }[mask_name]
    apsum_func = case.exact_func if method == "exact" else case.center_func
    npix_func = case.npix_exact_func if method == "exact" else case.npix_center_func
    aperture = case.aperture()

    expected_apsum, expected_npix = _expected_from_masks(aperture, data, method, mask)
    got_apsum, got_npix = apsum_func(data, x, y, *case.params, mask=mask)
    got_apsum_only = apsum_func(
        data, x.tolist(), y.tolist(), *case.params, mask=mask, return_npix=False
    )
    got_npix_only = npix_func(x, y, *case.params, shape=IMAGE_SHAPE)

    assert got_apsum.dtype == np.float64
    assert got_npix.dtype == np.float64
    assert_allclose(got_apsum, expected_apsum, rtol=0, atol=STRICT_ATOL)
    assert_allclose(got_npix, expected_npix, rtol=0, atol=STRICT_ATOL)
    assert_allclose(got_apsum_only, got_apsum, rtol=0, atol=STRICT_ATOL)
    if mask is None:
        assert_allclose(got_npix_only, got_npix, rtol=0, atol=STRICT_ATOL)
    else:
        assert_allclose(
            npix_func(x, y, *case.params, shape=IMAGE_SHAPE, mask=mask),
            got_npix,
            rtol=0,
            atol=STRICT_ATOL,
        )
    if fill_value == 0:
        assert_allclose(got_apsum, 0.0, rtol=0, atol=0.0)
    else:
        assert_allclose(got_apsum, got_npix, rtol=0, atol=STRICT_ATOL)
    if mask_name == "all_bad":
        assert_allclose(got_apsum, 0.0, rtol=0, atol=0.0)
        assert_allclose(got_npix, 0.0, rtol=0, atol=0.0)
    if method == "exact" and mask is None and "pill" not in case.name:
        assert_allclose(got_npix[:2], case.area, rtol=0, atol=AREA_ATOL)
        assert np.all(got_npix[2:] <= case.area)
    elif method == "exact" and mask is None:
        assert np.all(got_npix > 0.0)
        assert np.all(got_npix[2:] <= case.area)

    object_sum_func = (
        aperture.apsum_exact if method == "exact" else aperture.apsum_center
    )
    optimized_aperture = case.aperture(validate=False)
    optimized_sum_func = (
        optimized_aperture.apsum_exact
        if method == "exact"
        else optimized_aperture.apsum_center
    )
    object_apsum, object_npix = object_sum_func(data, mask=mask)
    optimized_apsum, optimized_npix = optimized_sum_func(data, mask=mask)
    assert_allclose(object_apsum, got_apsum, rtol=0, atol=STRICT_ATOL)
    assert_allclose(object_npix, got_npix, rtol=0, atol=STRICT_ATOL)
    assert_allclose(optimized_apsum, got_apsum, rtol=0, atol=STRICT_ATOL)
    assert_allclose(optimized_npix, got_npix, rtol=0, atol=STRICT_ATOL)

    object_npix_func = (
        aperture.npix_exact if method == "exact" else aperture.npix_center
    )
    object_npix_only = object_npix_func(IMAGE_SHAPE)
    assert_allclose(object_npix_only, got_npix_only, rtol=0, atol=STRICT_ATOL)
    if mask is not None:
        assert_allclose(
            object_npix_func(IMAGE_SHAPE, mask=mask),
            got_npix,
            rtol=0,
            atol=STRICT_ATOL,
        )


@pytest.mark.parametrize("method", ["exact", "center"])
@pytest.mark.parametrize("case", APERTURE_CASES, ids=lambda case: case.name)
def test_object_bboxes_and_weights_are_aligned_lists(case: ApertureCase, method: str):
    aperture = case.aperture()

    weights = (
        aperture.weights_exact() if method == "exact" else aperture.weights_center()
    )
    boxes = aperture.bboxes()

    assert isinstance(weights, list)
    assert isinstance(boxes, list)
    assert len(weights) == len(boxes) == len(POSITIONS)
    for weight, box in zip(weights, boxes, strict=True):
        assert weight.dtype == np.float64
        assert weight.shape == box.shape


@pytest.mark.parametrize("method", ["exact", "center"])
@pytest.mark.parametrize("case", APERTURE_CASES, ids=lambda case: case.name)
def test_scalar_object_api_preserves_scalar_sum_shapes_and_raw_weight_lists(
    case: ApertureCase, method: str
):
    data = np.ones(IMAGE_SHAPE, dtype=np.float64)
    aperture = case.aperture(positions=POSITIONS[0])

    sum_func = aperture.apsum_exact if method == "exact" else aperture.apsum_center
    npix_func = aperture.npix_exact if method == "exact" else aperture.npix_center
    apsum, npix = sum_func(data)
    weights = (
        aperture.weights_exact() if method == "exact" else aperture.weights_center()
    )
    boxes = aperture.bboxes()

    assert apsum.shape == ()
    assert npix.shape == ()
    assert npix_func(IMAGE_SHAPE).shape == ()
    assert len(weights) == len(boxes) == 1


def _realistic_scene(dtype) -> np.ndarray:
    yy, xx = np.indices((80, 84), dtype=np.float64)
    image = 37.0 + 0.07 * xx - 0.04 * yy
    image += 2.5 * np.sin(xx / 9.0) + 1.7 * np.cos(yy / 11.0)
    for amplitude, sigma, (x0, y0) in (
        (850.0, 1.4, (30.2, 31.7)),
        (420.0, 2.3, (54.6, 44.4)),
        (260.0, 1.8, (22.5, 58.2)),
    ):
        image += amplitude * np.exp(
            -0.5 * (((xx - x0) / sigma) ** 2 + ((yy - y0) / sigma) ** 2)
        )
    for amplitude, sx, sy, (x0, y0) in (
        (180.0, 4.0, 2.2, (63.0, 20.5)),
        (130.0, 3.2, 5.0, (41.5, 62.0)),
    ):
        image += amplitude * np.exp(
            -0.5 * (((xx - x0) / sx) ** 2 + ((yy - y0) / sy) ** 2)
        )
    return image.astype(dtype)


def _sky_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[25:38, 24:38] = True
    mask[42:48, 51:58] = True
    return mask


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: np.dtype(dtype).name)
@pytest.mark.parametrize("method", ["exact", "center"])
def test_realistic_scene_regresses_aperture_sum_and_sky_median(dtype, method: str):
    data = _realistic_scene(dtype)
    positions = np.array([[30.2, 31.7], [54.6, 44.4], [6.1, 6.2]], dtype=np.float64)
    source = apers.CircAp(positions, 3.2)
    sky = apers.CircAn(positions, 5.0, 8.0)
    expected = {
        "float32": {
            "exact": (
                np.array([10762.65178051, 9807.50184484, 1291.37545373]),
                np.array([33.50727450, 39.56990814, 37.82682912]),
                np.array([20.41486269, 39.66213226, 37.82682912]),
            ),
            "center": (
                np.array([10729.08079529, 9863.88818359, 1285.06106186]),
                np.array([36.11964417, 43.47990417, 40.04671097]),
                np.array([35.81173706, 43.47990417, 40.04671097]),
            ),
        },
        "float64": {
            "exact": (
                np.array([10762.65173842, 9807.50183531, 1291.37544614]),
                np.array([33.50727475, 39.56990917, 37.82682863]),
                np.array([20.41486284, 39.66213262, 37.82682863]),
            ),
            "center": (
                np.array([10729.08075552, 9863.88817579, 1285.06105253]),
                np.array([36.11964253, 43.47990339, 40.04671146]),
                np.array([35.81173662, 43.47990339, 40.04671146]),
            ),
        },
        "int32": {
            "exact": (
                np.array([10745.11980630, 9791.71722771, 1274.48164993]),
                np.array([33.22302725, 39.0, 37.0]),
                np.array([20.05291605, 39.0, 37.0]),
            ),
            "center": (
                np.array([10712.0, 9848.0, 1268.0]),
                np.array([36.0, 43.0, 40.0]),
                np.array([35.0, 43.0, 40.0]),
            ),
        },
        "int16": {
            "exact": (
                np.array([10745.11980630, 9791.71722771, 1274.48164993]),
                np.array([33.22302725, 39.0, 37.0]),
                np.array([20.05291605, 39.0, 37.0]),
            ),
            "center": (
                np.array([10712.0, 9848.0, 1268.0]),
                np.array([36.0, 43.0, 40.0]),
                np.array([35.0, 43.0, 40.0]),
            ),
        },
    }[np.dtype(dtype).name][method]
    expected_source, expected_sky, expected_masked_sky = expected

    source_sums = np.array(
        [
            bbox.apsum(weight, data)[0]
            for weight, bbox in zip(
                (
                    source.weights_exact()
                    if method == "exact"
                    else source.weights_center()
                ),
                source.bboxes(),
                strict=True,
            )
        ]
    )
    sky_median = np.array(
        [
            np.median(bbox.weighted_values(weight, data))
            for weight, bbox in zip(
                (sky.weights_exact() if method == "exact" else sky.weights_center()),
                sky.bboxes(),
                strict=True,
            )
        ]
    )
    masked_sky_median = np.array(
        [
            np.median(bbox.weighted_values(weight, data, mask=_sky_mask(data.shape)))
            for weight, bbox in zip(
                (sky.weights_exact() if method == "exact" else sky.weights_center()),
                sky.bboxes(),
                strict=True,
            )
        ]
    )

    assert_allclose(
        source_sums, expected_source, rtol=BENCHMARK_RTOL, atol=BENCHMARK_ATOL
    )
    assert_allclose(sky_median, expected_sky, rtol=BENCHMARK_RTOL, atol=BENCHMARK_ATOL)
    assert_allclose(
        masked_sky_median,
        expected_masked_sky,
        rtol=BENCHMARK_RTOL,
        atol=BENCHMARK_ATOL,
    )


@pytest.mark.parametrize("case", APERTURE_CASES, ids=lambda case: case.name)
def test_direct_kernels_preserve_scalar_list_vector_and_nd_position_inputs(
    case: ApertureCase,
):
    data = np.ones(IMAGE_SHAPE, dtype=np.float64)

    scalar_apsum, scalar_npix = case.exact_func(
        data, POSITIONS[0, 0], POSITIONS[0, 1], *case.params
    )
    list_apsum, list_npix = case.exact_func(
        data, POSITIONS[:, 0].tolist(), POSITIONS[:, 1].tolist(), *case.params
    )
    x2d = POSITIONS[:2, 0].reshape(1, 2)
    y2d = POSITIONS[:2, 1].reshape(1, 2)
    nd_apsum, nd_npix = case.exact_func(data, x2d, y2d, *case.params)
    nd_apsum_only = case.exact_func(data, x2d, y2d, *case.params, return_npix=False)

    assert scalar_apsum.shape == (1,)
    assert scalar_npix.shape == (1,)
    assert list_apsum.shape == (3,)
    assert list_npix.shape == (3,)
    assert nd_apsum.shape == x2d.shape
    assert nd_npix.shape == x2d.shape
    assert nd_apsum_only.shape == x2d.shape
    assert_allclose(scalar_apsum, scalar_npix, rtol=0, atol=STRICT_ATOL)
    assert_allclose(list_apsum, list_npix, rtol=0, atol=STRICT_ATOL)
    assert_allclose(nd_apsum, nd_npix, rtol=0, atol=STRICT_ATOL)
    assert_allclose(nd_apsum_only, nd_apsum, rtol=0, atol=STRICT_ATOL)


def test_rectangle_annulus_apsum_matches_photutils_subpixel_reference():
    photutils_aperture = pytest.importorskip("photutils.aperture")
    data = _realistic_scene(np.float64)
    positions = POSITIONS[:2]
    x = positions[:, 0]
    y = positions[:, 1]
    params = (2.0, 1.0, 7.0, 4.0, 0.35, 0.35)

    apsum, npix = apers.apsum_rect_ann_exact(data, x, y, *params)
    photutils_apsum, _ = photutils_aperture.RectangularAnnulus(
        positions,
        w_in=params[0],
        w_out=params[2],
        h_in=params[1],
        h_out=params[3],
        theta=params[4],
    ).do_photometry(data, method="subpixel", subpixels=128)
    photutils_masks = photutils_aperture.RectangularAnnulus(
        positions,
        w_in=params[0],
        w_out=params[2],
        h_in=params[1],
        h_out=params[3],
        theta=params[4],
    ).to_mask(method="subpixel", subpixels=128)
    photutils_npix = np.array(
        [mask.data.sum(dtype=np.float64) for mask in photutils_masks]
    )

    assert_allclose(
        photutils_apsum,
        apsum,
        rtol=RECTANGLE_PHOTUTILS_RTOL,
        atol=BENCHMARK_ATOL,
    )
    assert_allclose(
        photutils_npix,
        npix,
        rtol=RECTANGLE_PHOTUTILS_RTOL,
        atol=BENCHMARK_ATOL,
    )


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32], ids=lambda d: np.dtype(d).name
)
@pytest.mark.parametrize(
    "shape_name", ["circle", "ellipse", "circle_annulus", "ellipse_annulus"]
)
def test_sep_parity_runs_for_benchmark_dtypes(dtype, shape_name: str):
    sep = pytest.importorskip("sep")
    data = (_realistic_scene(np.float64) % 1000.0).astype(dtype)
    positions = POSITIONS[:2]
    x = positions[:, 0]
    y = positions[:, 1]

    if shape_name == "circle":
        apsum = apers.apsum_circ_exact(data, x, y, 4.0, return_npix=False)
        sep_apsum, _, flag = sep.sum_circle(data, x, y, 4.0, subpix=0)
    elif shape_name == "ellipse":
        apsum = apers.apsum_ellip_exact(data, x, y, 5.0, 2.5, 0.35, return_npix=False)
        sep_apsum, _, flag = sep.sum_ellipse(data, x, y, 5.0, 2.5, 0.35, subpix=0)
    elif shape_name == "circle_annulus":
        apsum = apers.apsum_circ_ann_exact(data, x, y, 2.0, 5.0, return_npix=False)
        sep_apsum, _, flag = sep.sum_circann(data, x, y, 2.0, 5.0, subpix=0)
    else:
        apsum = apers.apsum_ellip_ann_exact(
            data, x, y, 2.0, 1.0, 5.0, 2.5, 0.35, return_npix=False
        )
        sep_apsum, _, flag = sep.sum_ellipann(
            data, x, y, 5.0, 2.5, 0.35, 0.4, 1.0, subpix=0
        )

    assert np.all(flag == 0)
    assert_allclose(sep_apsum, apsum, rtol=BENCHMARK_RTOL, atol=BENCHMARK_ATOL)

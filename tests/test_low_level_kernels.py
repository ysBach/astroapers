from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astroapers as apers
import astroapers.kernels as aapk

SEP_PARITY_RTOL = 5e-9


def test_uniform_circle_area():
    data = np.ones((32, 32), dtype=np.float64)
    apsum, npix = apers.apsum_circ_exact(data, [16.0], [16.0], 5.0)
    assert_allclose(apsum, npix)
    assert_allclose(npix[0], np.pi * 25.0, rtol=0, atol=1e-8)


def test_uniform_ellipse_area():
    data = np.ones((64, 64), dtype=np.float64)
    apsum, npix = apers.apsum_ellip_exact(data, [32.2], [31.7], 7.0, 4.0, 0.4)
    assert_allclose(apsum, npix)
    assert_allclose(npix[0], np.pi * 28.0, rtol=0, atol=1e-7)


def test_uniform_rect_area():
    data = np.ones((64, 64), dtype=np.float64)
    apsum, npix = apers.apsum_rect_exact(data, [32.2], [31.7], 9.0, 5.0, 0.4)
    assert_allclose(apsum, npix)
    assert_allclose(npix[0], 45.0, rtol=0, atol=1e-12)


def test_center_apsum_functions_match_center_masks():
    data = np.arange(12 * 14, dtype=np.float64).reshape(12, 14)
    positions = np.array([[4.2, 5.1], [8.5, 6.4]])
    x = positions[:, 0]
    y = positions[:, 1]

    cases = [
        (apers.CircAp, (2.7,), apers.apsum_circ_center, apers.npix_circ_center),
        (
            apers.EllipAp,
            (3.0, 1.7, 0.4),
            apers.apsum_ellip_center,
            apers.npix_ellip_center,
        ),
        (
            apers.RectAp,
            (4.0, 2.5, 0.3),
            apers.apsum_rect_center,
            apers.npix_rect_center,
        ),
    ]

    for aperture_cls, params, apsum_func, npix_func in cases:
        aperture = aperture_cls(positions, *params)
        apsum, npix = apsum_func(data, x, y, *params)
        npix_only = npix_func(x, y, *params, shape=data.shape)
        expected_apsum = []
        expected_npix = []
        for apm in aperture.get_apmask(method="center"):
            expected_apsum.append(apm.apsum(data)[0])
            expected_npix.append(apm.weights.sum())
        assert_allclose(apsum, expected_apsum)
        assert_allclose(npix, expected_npix)
        assert_allclose(npix_only, expected_npix)
        assert_allclose(apsum_func(data, x, y, *params, return_npix=False), apsum)


def test_circular_annulus_npix_functions_match_apsum_and_masks():
    data = np.ones((24, 26), dtype=np.float64)
    x = np.array([4.3, 12.5, 22.2])
    y = np.array([5.1, 14.5, 2.2])
    r_in = 1.3
    r_out = 4.2

    exact_apsum, exact_npix = apers.apsum_circ_ann_exact(data, x, y, r_in, r_out)
    assert_allclose(
        apers.npix_circ_ann_exact(x, y, r_in, r_out, shape=data.shape),
        exact_npix,
    )
    assert_allclose(exact_apsum, exact_npix)

    annulus = apers.CircAn(np.column_stack([x, y]), r_in, r_out)
    expected_center = [
        apm.npix(data.shape) for apm in annulus.get_apmask(method="center")
    ]
    assert_allclose(
        apers.npix_circ_ann_center(x, y, r_in, r_out, shape=data.shape),
        expected_center,
    )


def test_circular_annulus_npix_accepts_zero_inner_radius():
    shape = (16, 18)
    x = np.array([4.0, 12.0])
    y = np.array([5.0, 10.0])
    r_out = 3.0

    assert_allclose(
        apers.npix_circ_ann_exact(x, y, 0.0, r_out, shape=shape),
        apers.npix_circ_exact(x, y, r_out, shape=shape),
    )
    assert_allclose(
        apers.npix_circ_ann_center(x, y, 0.0, r_out, shape=shape),
        apers.npix_circ_center(x, y, r_out, shape=shape),
    )


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int32, np.int16, np.uint16]
)
def test_direct_exact_apsum_accepts_mask(dtype):
    data = (np.arange(18 * 20).reshape(18, 20) % 200).astype(dtype)
    positions = np.array([[5.2, 6.1], [13.5, 9.8], [-2.0, 4.0]], dtype=np.float64)
    x = positions[:, 0]
    y = positions[:, 1]
    bad = np.zeros(data.shape, dtype=bool)
    bad[5:8, 4:7] = True
    bad[9, 13] = True

    cases = [
        (apers.CircAp(positions, 2.7), apers.apsum_circ_exact, (2.7,)),
        (
            apers.EllipAp(positions, 3.1, 1.8, 0.3),
            apers.apsum_ellip_exact,
            (3.1, 1.8, 0.3),
        ),
        (
            apers.RectAp(positions, 4.2, 2.5, -0.2),
            apers.apsum_rect_exact,
            (4.2, 2.5, -0.2),
        ),
        (
            apers.CircAn(positions, 1.2, 3.4),
            apers.apsum_circ_ann_exact,
            (1.2, 3.4),
        ),
    ]

    for aperture, func, params in cases:
        expected_apsum, expected_npix = aperture.apsum(data, mask=bad)
        apsum, npix = func(data, x, y, *params, mask=bad)

        assert_allclose(apsum, expected_apsum)
        assert_allclose(npix, expected_npix)
        assert_allclose(func(data, x, y, *params, mask=bad, return_npix=False), apsum)


def test_direct_masked_exact_apsum_accumulates_float32_data_in_float64():
    data = np.array([[1.0e8, 1.0, -1.0e8]], dtype=np.float32)
    x = np.array([1.0])
    y = np.array([0.0])
    bad = np.zeros(data.shape, dtype=bool)

    apsum, npix = apers.apsum_rect_exact(data, x, y, 3.0, 1.0, 0.0, mask=bad)
    weights, _ = aapk.weights_rect_exact(x, y, 3.0, 1.0, 0.0)
    expected_apsum = np.sum(
        data.astype(np.float64) * weights[0].astype(np.float64),
        dtype=np.float64,
    )
    expected_npix = np.sum(weights[0], dtype=np.float64)
    float32_apsum = np.sum(data * weights[0].astype(np.float32), dtype=np.float32)

    assert_allclose(apsum, [expected_apsum], rtol=0, atol=0)
    assert_allclose(npix, [expected_npix], rtol=0, atol=0)
    assert expected_apsum != float(float32_apsum)
    assert_allclose(
        apers.apsum_rect_exact(data, x, y, 3.0, 1.0, 0.0, mask=bad, return_npix=False),
        apsum,
    )


def test_direct_center_apsum_accepts_mask():
    data = np.arange(14 * 16, dtype=np.float64).reshape(14, 16)
    positions = np.array([[4.2, 5.1], [9.5, 7.4]], dtype=np.float64)
    x = positions[:, 0]
    y = positions[:, 1]
    bad = np.zeros(data.shape, dtype=bool)
    bad[4:7, 4:7] = True

    cases = [
        (apers.CircAp(positions, 2.7), apers.apsum_circ_center, (2.7,)),
        (
            apers.EllipAp(positions, 3.0, 1.7, 0.4),
            apers.apsum_ellip_center,
            (3.0, 1.7, 0.4),
        ),
        (
            apers.RectAp(positions, 4.0, 2.5, 0.3),
            apers.apsum_rect_center,
            (4.0, 2.5, 0.3),
        ),
    ]

    for aperture, func, params in cases:
        expected = [
            apm.apsum(data, mask=bad) for apm in aperture.get_apmask(method="center")
        ]
        expected_apsum = np.array([item[0] for item in expected])
        expected_npix = np.array([item[1] for item in expected])
        apsum, npix = func(data, x, y, *params, mask=bad)

        assert_allclose(apsum, expected_apsum)
        assert_allclose(npix, expected_npix)
        assert_allclose(func(data, x, y, *params, mask=bad, return_npix=False), apsum)


def test_direct_masked_apsum_preserves_shapes_and_nan_positions():
    data = np.ones((12, 12), dtype=np.float64)
    bad = np.zeros_like(data, dtype=np.uint8)
    bad[4, 4] = 1

    scalar_apsum, scalar_npix = apers.apsum_circ_exact(data, 4.0, 4.0, 2.0, mask=bad)
    assert scalar_apsum.shape == (1,)
    assert scalar_npix.shape == (1,)

    x = np.array([4.0, np.nan, 8.0])
    y = np.array([4.0, 5.0, np.inf])
    apsum, npix = apers.apsum_circ_exact(data, x, y, 2.0, mask=bad)

    assert np.isfinite(apsum[0])
    assert np.isfinite(npix[0])
    assert np.isnan(apsum[1])
    assert np.isnan(npix[1])
    assert np.isnan(apsum[2])
    assert np.isnan(npix[2])


def test_direct_masked_apsum_validates_mask_shape_and_all_bad_mask():
    data = np.ones((12, 12), dtype=np.float64)
    all_bad = np.ones_like(data, dtype=bool)

    apsum, npix = apers.apsum_circ_exact(data, [4.0], [4.0], 2.0, mask=all_bad)
    assert_allclose(apsum, [0.0])
    assert_allclose(npix, [0.0])

    with pytest.raises(ValueError, match="mask must have the same shape as data"):
        apers.apsum_circ_exact(
            data, [4.0], [4.0], 2.0, mask=np.zeros((12, 1), dtype=bool)
        )


@pytest.mark.parametrize(
    ("r_in", "r_out"),
    [
        (-1.0, 3.0),
        (1.0, 0.0),
        (3.0, 3.0),
        (4.0, 3.0),
        (np.nan, 3.0),
        (1.0, np.inf),
    ],
)
def test_circular_annulus_npix_rejects_invalid_radii(r_in, r_out):
    with pytest.raises(ValueError):
        apers.npix_circ_ann_exact([4.0], [5.0], r_in, r_out, shape=(12, 12))
    with pytest.raises(ValueError):
        apers.npix_circ_ann_center([4.0], [5.0], r_in, r_out, shape=(12, 12))


def test_circle_apsum_matches_sep_exact():
    sep = pytest.importorskip("sep")
    data = np.arange(64 * 48, dtype=np.float64).reshape(48, 64) / 10.0
    x = np.array([8.2, 31.5, 58.1])
    y = np.array([9.7, 20.2, 42.4])
    r = 4.3

    apsum, npix = apers.apsum_circ_exact(data, x, y, r)
    sep_apsum, _, sep_flag = sep.sum_circle(data, x, y, r, subpix=0)

    assert np.all(sep_flag == 0)
    assert_allclose(apsum, sep_apsum, rtol=SEP_PARITY_RTOL, atol=1e-6)
    expected_npix = [
        apers.CircAp((xi, yi), r=r).apsum(np.ones_like(data))[1] for xi, yi in zip(x, y)
    ]
    assert_allclose(npix, expected_npix)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int16])
def test_circle_apsum_accepts_supported_image_dtypes(dtype):
    data = (np.arange(64 * 48).reshape(48, 64) % 200).astype(dtype)
    x = np.array([8.2, 31.5, 58.1])
    y = np.array([9.7, 20.2, 42.4])
    r = 4.3

    apsum = apers.apsum_circ_exact(data, x, y, r, return_npix=False)
    tuple_apsum, _ = apers.apsum_circ_exact(data, x, y, r)

    assert_allclose(apsum, tuple_apsum)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int16])
def test_ellipse_apsum_accepts_supported_image_dtypes(dtype):
    data = (np.arange(72 * 80).reshape(72, 80) % 200).astype(dtype)
    x = np.array([12.3, 35.5, 68.2])
    y = np.array([10.1, 44.4, 60.0])
    a = 5.0
    b = 2.5
    theta = 0.35

    apsum = apers.apsum_ellip_exact(data, x, y, a, b, theta, return_npix=False)
    tuple_apsum, _ = apers.apsum_ellip_exact(data, x, y, a, b, theta)

    assert_allclose(apsum, tuple_apsum)


def test_ellipse_apsum_matches_sep_exact():
    sep = pytest.importorskip("sep")
    data = np.arange(72 * 80, dtype=np.float64).reshape(72, 80) / 7.0
    x = np.array([12.3, 35.5, 68.2])
    y = np.array([10.1, 44.4, 60.0])
    a = 5.0
    b = 2.5
    theta = 0.35

    apsum, _ = apers.apsum_ellip_exact(data, x, y, a, b, theta)
    sep_apsum, _, sep_flag = sep.sum_ellipse(data, x, y, a, b, theta, subpix=0)

    assert np.all(sep_flag == 0)
    assert_allclose(apsum, sep_apsum, rtol=SEP_PARITY_RTOL, atol=1e-6)


def test_circular_annulus_sum_matches_sep_exact():
    sep = pytest.importorskip("sep")
    data = np.arange(64 * 64, dtype=np.float64).reshape(64, 64) / 13.0
    x = np.array([16.2, 30.5, 48.7])
    y = np.array([15.5, 31.2, 47.8])
    r_in = 2.0
    r_out = 5.0

    apsum = np.array(
        [
            apers.CircAn((xi, yi), r_in=r_in, r_out=r_out).apsum(data)[0]
            for xi, yi in zip(x, y)
        ]
    )
    sep_apsum, _, sep_flag = sep.sum_circann(data, x, y, r_in, r_out, subpix=0)

    assert np.all(sep_flag == 0)
    assert_allclose(apsum, sep_apsum, rtol=SEP_PARITY_RTOL, atol=1e-6)


def test_elliptical_annulus_sum_matches_sep_exact():
    sep = pytest.importorskip("sep")
    data = np.arange(80 * 80, dtype=np.float64).reshape(80, 80) / 11.0
    x = np.array([18.2, 40.5, 60.7])
    y = np.array([17.5, 39.2, 58.8])
    a = 4.0
    b = 2.0
    theta = -0.25
    r_in = 0.5
    r_out = 1.8

    apsum = np.array(
        [
            apers.EllipAn(
                (xi, yi),
                a_in=a * r_in,
                b_in=b * r_in,
                a_out=a * r_out,
                b_out=b * r_out,
                theta_in=theta,
            ).apsum(data)[0]
            for xi, yi in zip(x, y)
        ]
    )
    sep_apsum, _, sep_flag = sep.sum_ellipann(
        data,
        x,
        y,
        a,
        b,
        theta,
        r_in,
        r_out,
        subpix=0,
    )

    assert np.all(sep_flag == 0)
    assert_allclose(apsum, sep_apsum, rtol=SEP_PARITY_RTOL, atol=1e-6)


def test_apsum_outputs_are_float64_for_builtin_kernels():
    for dtype in (np.float32, np.float64, np.int32, np.int16):
        data = np.arange(16, dtype=dtype).reshape(4, 4)

        direct_apsum, direct_npix = apers.apsum_circ_exact(data, [1.5], [1.5], 1.2)
        object_apsum, object_npix = apers.CircAp((1.5, 1.5), 1.2).apsum(data)

        assert direct_apsum.dtype == np.float64
        assert direct_npix.dtype == np.float64
        assert object_apsum.dtype == np.float64
        assert object_npix.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.bool_, np.uint16, np.int64])
def test_unmasked_direct_apsum_converts_unsupported_image_dtypes_to_float64(dtype):
    data = (np.arange(16).reshape(4, 4) % 7).astype(dtype)
    reference = data.astype(np.float64)
    x = np.array([1.5, 2.0])
    y = np.array([1.5, 2.0])

    apsum, npix = apers.apsum_circ_exact(data, x, y, 1.2)
    expected_apsum, expected_npix = apers.apsum_circ_exact(reference, x, y, 1.2)

    assert apsum.dtype == np.float64
    assert npix.dtype == np.float64
    assert_allclose(apsum, expected_apsum)
    assert_allclose(npix, expected_npix)


def test_direct_positions_are_converted_to_float64_without_broadcasting():
    data = np.ones((8, 8), dtype=np.float64)
    x = np.array([2, 4], dtype=np.int64)
    y = np.array([2, 4], dtype=np.int64)

    apsum, npix = apers.apsum_circ_exact(data, x, y, 1.5)
    expected_apsum, expected_npix = apers.apsum_circ_exact(
        data,
        x.astype(np.float64),
        y.astype(np.float64),
        1.5,
    )

    assert apsum.dtype == np.float64
    assert npix.dtype == np.float64
    assert_allclose(apsum, expected_apsum)
    assert_allclose(npix, expected_npix)


def test_object_core_apsum_matches_low_level_exact_kernels():
    data = (np.arange(28 * 32).reshape(28, 32) % 101).astype(np.int32)
    positions = np.array([(4.3, 5.1), (14.7, 19.2), (25.4, 3.5)], dtype=np.float64)
    x = positions[:, 0]
    y = positions[:, 1]

    cases = [
        (
            apers.CircAp(positions, 2.4),
            apers.apsum_circ_exact(data, x, y, 2.4),
        ),
        (
            apers.EllipAp(positions, 3.5, 1.8, 0.4),
            apers.apsum_ellip_exact(data, x, y, 3.5, 1.8, 0.4),
        ),
        (
            apers.RectAp(positions, 4.0, 2.5, -0.3),
            apers.apsum_rect_exact(data, x, y, 4.0, 2.5, -0.3),
        ),
    ]

    for aperture, expected in cases:
        apsum, npix = aperture.apsum(data)
        expected_apsum, expected_npix = expected
        assert_allclose(apsum, expected_apsum)
        assert_allclose(npix, expected_npix)


def test_object_annulus_apsum_subtracts_low_level_exact_kernels():
    data = np.arange(28 * 32, dtype=np.float64).reshape(28, 32)
    positions = np.array([(4.3, 5.1), (14.7, 19.2), (25.4, 3.5)], dtype=np.float64)
    x = positions[:, 0]
    y = positions[:, 1]

    circ = apers.CircAn(positions, 1.1, 3.2)
    circ_fused = apers.apsum_circ_ann_exact(data, x, y, 1.1, 3.2)
    circ_outer = apers.apsum_circ_exact(data, x, y, 3.2)
    circ_inner = apers.apsum_circ_exact(data, x, y, 1.1)
    assert_allclose(circ_fused[0], circ_outer[0] - circ_inner[0])
    assert_allclose(circ_fused[1], circ_outer[1] - circ_inner[1])
    assert_allclose(
        apers.apsum_circ_ann_exact(data, x, y, 1.1, 3.2, return_npix=False),
        circ_fused[0],
    )

    ellip = apers.EllipAn(positions, 1.2, 0.8, 3.8, 2.4, theta_in=0.4)
    ellip_outer = apers.apsum_ellip_exact(data, x, y, 3.8, 2.4, 0.4)
    ellip_inner = apers.apsum_ellip_exact(data, x, y, 1.2, 0.8, 0.4)

    rect = apers.RectAn(positions, 1.5, 0.9, 4.0, 2.7, theta_in=-0.3)
    rect_outer = apers.apsum_rect_exact(data, x, y, 4.0, 2.7, -0.3)
    rect_inner = apers.apsum_rect_exact(data, x, y, 1.5, 0.9, -0.3)

    for annulus, outer, inner in [
        (circ, circ_outer, circ_inner),
        (ellip, ellip_outer, ellip_inner),
        (rect, rect_outer, rect_inner),
    ]:
        apsum, npix = annulus.apsum(data)
        assert_allclose(apsum, outer[0] - inner[0])
        assert_allclose(npix, outer[1] - inner[1])


def test_low_level_functions_preserve_nd_position_shape():
    data = np.ones((32, 32), dtype=np.float64)
    x = np.array([[8.0, 10.0], [12.0, 14.0]])
    y = np.array([[9.0, 11.0], [13.0, 15.0]])

    apsum, npix = apers.apsum_circ_exact(data, x, y, 2.0)
    apsum_only = apers.apsum_circ_exact(data, x, y, 2.0, return_npix=False)
    npix_only = apers.npix_circ_exact(x, y, 2.0, shape=data.shape)
    npix_center = apers.npix_circ_center(x, y, 2.0, shape=data.shape)

    assert apsum.shape == x.shape
    assert npix.shape == x.shape
    assert apsum_only.shape == x.shape
    assert npix_only.shape == x.shape
    assert npix_center.shape == x.shape
    assert_allclose(apsum, apsum_only)
    assert_allclose(npix, npix_only)


def test_low_level_center_npix_handles_edges_and_nonfinite_positions():
    shape = (8, 9)
    x = np.array([0.0, 4.2, np.nan])
    y = np.array([0.0, 3.7, 2.0])

    circ = apers.npix_circ_center(x, y, 2.2, shape=shape)
    ellip = apers.npix_ellip_center(x, y, 2.4, 1.2, 0.3, shape=shape)
    rect = apers.npix_rect_center(x, y, 3.0, 2.0, -0.4, shape=shape)

    for values in (circ, ellip, rect):
        assert values[0] >= 0.0
        assert values[1] >= 0.0
        assert np.isnan(values[2])


def test_low_level_apsum_rejects_non_2d_data_before_dispatch():
    data = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="data must be a 2-D array"):
        apers.apsum_circ_exact(data, [1.0], [1.0], 2.0)


def test_low_level_apsum_returns_npix_by_default_and_can_skip_it():
    data = np.ones((16, 16), dtype=np.float64)
    x = np.array([8.0, 9.0])
    y = np.array([8.0, 9.0])

    apsum, npix = apers.apsum_circ_exact(data, x, y, 2.0)
    apsum_only = apers.apsum_circ_exact(data, x, y, 2.0, return_npix=False)

    assert_allclose(apsum_only, apsum)
    assert_allclose(apsum, npix)


def test_parallel_threshold_is_user_tunable():
    original = apers.get_parallel_threshold()
    try:
        apers.set_parallel_threshold(7)
        assert apers.get_parallel_threshold() == 7
    finally:
        apers.set_parallel_threshold(original)


def test_parallel_threshold_calibration_can_apply_threshold():
    original = apers.get_parallel_threshold()
    try:
        result = apers.calibrate_parallel_threshold(
            counts=(1, 2), repeats=1, image_size=64, apply=True
        )

        assert result.threshold > 0
        assert result.applied
        assert result.counts == (1, 2)
        assert apers.get_parallel_threshold() == result.threshold
    finally:
        apers.set_parallel_threshold(original)


def test_parallel_threshold_calibration_cli_reports_export(capsys):
    from astroapers._cli import main

    original = apers.get_parallel_threshold()
    try:
        status = main(
            [
                "calibrate-threshold",
                "--counts",
                "1",
                "2",
                "--repeats",
                "1",
                "--image-size",
                "64",
            ]
        )
    finally:
        apers.set_parallel_threshold(original)

    output = capsys.readouterr().out
    assert status == 0
    assert "recommended_threshold:" in output
    assert "export ASTROAPERS_PARALLEL_THRESHOLD=" in output
    assert "aap.set_parallel_threshold(" in output
    assert "--apply" not in output

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astroapers as apers


def test_object_scalar_apsum_preserves_zero_dimensional_return():
    data = np.ones((12, 12), dtype=np.float64)
    apsum, npix = apers.CircAp((4.3, 5.1), 2.4).apsum(data)

    assert apsum.shape == ()
    assert npix.shape == ()
    assert_allclose(apsum, npix)


def test_object_apsum_supports_return_npix_false_for_direct_and_mask_paths():
    data = np.ones((16, 16), dtype=np.float64)
    positions = np.array([(4.0, 5.0), (9.5, 8.5)])
    cases = [
        apers.CircAp(positions, r=2.4),
        apers.EllipAp(positions, a=3.0, b=1.5, theta=0.2),
        apers.RectAp(positions, w=4.0, h=2.0, theta=0.3),
        apers.CircAn(positions, r_in=1.0, r_out=3.0),
        apers.EllipAn(positions, a_in=1.0, b_in=0.5, a_out=3.0, b_out=2.0),
        apers.EllipAn(
            positions,
            a_in=0.6,
            b_in=0.4,
            a_out=3.0,
            b_out=2.0,
            theta_in=0.2,
            theta_out=0.7,
        ),
        apers.RectAn(positions, w_in=1.0, h_in=0.5, w_out=4.0, h_out=2.0),
        apers.RectAn(
            positions,
            w_in=0.7,
            h_in=0.5,
            w_out=4.0,
            h_out=2.0,
            theta_in=0.1,
            theta_out=0.5,
        ),
        apers.PillAp(positions, w=3.0, a=1.5, b=1.0, theta=0.2),
        apers.PillAn(
            positions,
            w_in=1.0,
            a_in=0.5,
            b_in=0.4,
            w_out=3.0,
            a_out=1.5,
            b_out=1.0,
            theta_in=0.2,
        ),
        apers.WedgeAp(positions, 2.0, 6.0, 0.4, 0.6),
    ]

    bad = np.zeros_like(data, dtype=bool)
    bad[4, 5] = True

    for aperture in cases:
        apsum, _ = aperture.apsum(data)
        assert_allclose(aperture.apsum(data, return_npix=False), apsum)
        masked_apsum, _ = aperture.apsum(data, mask=bad)
        assert_allclose(
            aperture.apsum(data, mask=bad, return_npix=False),
            masked_apsum,
        )


def test_vector_circle_weights_reject_unreasonably_large_radius():
    aperture = apers.CircAp([(0.0, 0.0), (1.0, 1.0)], r=1.0e20)

    with pytest.raises((MemoryError, ValueError)):
        aperture.weights()


def test_annulus_theta_keyword_is_not_supported():
    with pytest.raises(TypeError, match="unexpected keyword argument 'theta'"):
        apers.EllipAn((4.0, 5.0), 1.0, 0.5, 3.0, 2.0, theta=0.25)
    with pytest.raises(TypeError, match="unexpected keyword argument 'theta'"):
        apers.RectAn((4.0, 5.0), 1.0, 0.5, 3.0, 2.0, theta=0.25)
    with pytest.raises(TypeError, match="unexpected keyword argument 'theta'"):
        apers.PillAn(
            (4.0, 5.0),
            w_in=1.0,
            a_in=0.5,
            b_in=0.4,
            w_out=3.0,
            a_out=2.0,
            b_out=1.5,
            theta=0.25,
        )

    with pytest.raises(
        TypeError, match="takes 6 positional arguments but 7 were given"
    ):
        apers.EllipAn((4.0, 5.0), 1.0, 0.5, 3.0, 2.0, 0.25)
    with pytest.raises(
        TypeError, match="takes 6 positional arguments but 7 were given"
    ):
        apers.RectAn((4.0, 5.0), 1.0, 0.5, 3.0, 2.0, 0.25)
    with pytest.raises(
        TypeError, match="takes 8 positional arguments but 9 were given"
    ):
        apers.PillAn((4.0, 5.0), 1.0, 0.5, 0.4, 3.0, 2.0, 1.5, 0.25)

    aperture = apers.EllipAn((4.0, 5.0), 1.0, 0.5, 3.0, 2.0, theta_in=0.25)
    assert aperture.theta_in == 0.25
    assert aperture.theta_out == 0.25
    assert not hasattr(aperture, "theta")
    assert not hasattr(
        apers.RectAn((4.0, 5.0), 1.0, 0.5, 3.0, 2.0, theta_in=0.25),
        "theta",
    )
    assert not hasattr(
        apers.PillAn((4.0, 5.0), 1.0, 0.5, 0.4, 3.0, 2.0, 1.5, theta_in=0.25),
        "theta",
    )


def test_object_api_excludes_true_mask_pixels_from_apsum_and_npix():
    data = np.ones((9, 9), dtype=np.float64)
    aperture = apers.CircAp((4.0, 4.0), r=2.0)
    full_apsum, full_npix = aperture.apsum(data)

    bad = np.zeros_like(data, dtype=bool)
    bad[4, 4] = True
    masked_apsum, masked_npix = aperture.apsum(data, mask=bad)

    assert_allclose(full_apsum, full_npix)
    assert masked_apsum < full_apsum
    assert_allclose(masked_apsum, masked_npix)


def test_weighted_values_center_returns_unweighted_center_selected_data():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    aperture = apers.CircAp((2.0, 2.0), r=1.1)

    values = aperture.weighted_values(data)
    weights = aperture.weights(method="center")[0]
    bbox = aperture.bboxes()[0]
    selected = bbox.to_image(weights, data.shape).astype(bool)

    assert_allclose(values, data[selected].ravel())


def test_weighted_values_exact_returns_positive_weighted_data():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    data[2, 2] = 0.0
    aperture = apers.CircAp((2.1, 2.2), r=1.4)

    values = aperture.weighted_values(data, method="exact")
    weights = aperture.weights(method="exact")[0]
    bbox = aperture.bboxes()[0]
    weighted = bbox.weighted_cutout(weights, data)
    expected = weighted[weights > 0.0].ravel()

    assert_allclose(values, expected)


def test_weighted_values_vector_aperture_returns_list_of_arrays():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.RectAp([(2.0, 2.0), (7.0, 7.0)], w=3.0, h=3.0)

    values = aperture.weighted_values(data)

    assert isinstance(values, list)
    assert len(values) == 2
    assert all(value.ndim == 1 for value in values)
    assert_allclose(
        values[0], apers.RectAp((2.0, 2.0), w=3.0, h=3.0).weighted_values(data)
    )
    assert_allclose(
        values[1], apers.RectAp((7.0, 7.0), w=3.0, h=3.0).weighted_values(data)
    )


def test_weighted_values_returns_empty_array_for_out_of_frame_aperture():
    data = np.ones((5, 5), dtype=np.float64)
    aperture = apers.CircAp((-100.0, -100.0), r=1.0)

    values = aperture.weighted_values(data)

    assert values.shape == (0,)
    assert values.dtype == np.float64


def test_vector_aperture_returns_vector_weights_and_apsum():
    data = np.ones((16, 16), dtype=np.float64)
    aperture = apers.RectAp([(4.0, 4.0), (8.5, 9.0)], w=3.0, h=2.0)

    weights = aperture.weights()
    apsum, npix = aperture.apsum(data)

    assert len(weights) == 2
    assert apsum.shape == (2,)
    assert_allclose(apsum, npix)
    assert_allclose(npix, [6.0, 6.0], atol=1e-12)


def test_validate_false_reuses_position_array_and_skips_parameter_checks():
    positions = np.array([[4.0, 4.0], [8.5, 9.0]], dtype=np.float64)

    aperture = apers.CircAp(positions, r=2.0, validate=False)
    unchecked = apers.CircAp(positions, r=-2.0, validate=False)

    assert aperture.positions is positions
    assert not aperture.isscalar
    assert unchecked.r == -2.0
    with pytest.raises(ValueError, match="positive"):
        apers.CircAp(positions, r=-2.0)


@pytest.mark.parametrize("bad_position", [(np.nan, 4.0), (4.0, np.inf)])
def test_validate_true_rejects_nonfinite_scalar_positions(bad_position):
    with pytest.raises(ValueError, match="positions must be finite"):
        apers.CircAp(bad_position, r=2.0)


def test_validate_true_rejects_nonfinite_vector_positions():
    positions = np.array([[4.0, 4.0], [np.nan, 9.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="positions must be finite"):
        apers.RectAp(positions, w=3.0, h=2.0)


def test_validate_false_skips_position_finiteness_check():
    positions = np.array([[4.0, 4.0], [np.nan, 9.0]], dtype=np.float64)

    aperture = apers.CircAp(positions, r=2.0, validate=False)

    assert aperture.positions is positions


def test_object_sum_aliases_are_not_exposed():
    aperture = apers.CircAp((4.0, 4.0), r=2.0)

    assert not hasattr(aperture, "sum")
    assert not hasattr(aperture, "do_photometry")


def test_mask_method_validation_is_not_disabled_by_validate_false():
    data = np.ones((16, 16), dtype=np.float64)
    aperture = apers.CircAp((4.0, 4.0), r=2.0, validate=False)

    with pytest.raises(ValueError, match="method must be"):
        aperture.weights(method="not-a-method")
    with pytest.raises(ValueError, match="method must be"):
        aperture.weighted_values(data, method="not-a-method")


def test_validate_false_object_apsum_skips_kernel_parameter_validation(monkeypatch):
    data = np.ones((16, 16), dtype=np.float64)
    aperture = apers.EllipAn(
        (4.0, 4.0),
        a_in=1.0,
        b_in=0.5,
        a_out=2.0,
        b_out=1.5,
        validate=False,
    )

    def fail_validation(*args, **kwargs):
        raise AssertionError("kernel validation should be skipped")

    monkeypatch.setattr(apers.kernels, "_validate_inner_outer_axes", fail_validation)

    apsum, npix = aperture.apsum(data)

    assert np.isfinite(apsum)
    assert np.isfinite(npix)


def test_validate_false_object_npix_skips_mask_shape_validation(monkeypatch):
    aperture = apers.CircAp((4.0, 4.0), r=2.0, validate=False)
    mask = np.zeros((16, 16), dtype=bool)

    bad_mask = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="mask must have the same shape"):
        apers.CircAp((4.0, 4.0), r=2.0).npix((16, 16), mask=bad_mask)

    def fail_validation(*args, **kwargs):
        raise AssertionError("mask validation should be skipped")

    monkeypatch.setattr(apers.kernels, "validate_mask", fail_validation)

    aperture.npix((16, 16), mask=mask)


@pytest.mark.parametrize("bad_r_in", [-1.0, np.nan])
def test_circular_annulus_rejects_invalid_inner_radius(bad_r_in):
    with pytest.raises(ValueError, match="r_in must be a nonnegative finite scalar"):
        apers.CircAn((4.0, 4.0), r_in=bad_r_in, r_out=2.0)


def test_pillbox_custom_aperture_on_uniform_data_tracks_area():
    data = np.ones((64, 64), dtype=np.float64)
    aperture = apers.PillAp((32.0, 32.0), w=8.0, a=2.0, b=2.0, theta=0.3)

    apsum, npix = aperture.apsum(data)

    assert_allclose(apsum, npix)
    assert_allclose(npix, aperture.area, rtol=0, atol=0.2)


def test_pill_annulus_on_uniform_data_tracks_area_and_center_mask():
    data = np.ones((96, 96), dtype=np.float64)
    annulus = apers.PillAn(
        (48.0, 48.0),
        w_in=6.0,
        a_in=2.0,
        b_in=1.5,
        w_out=10.0,
        a_out=4.0,
        b_out=3.0,
        theta_in=0.2,
    )

    apsum, npix = annulus.apsum(data)
    center_weights = annulus.weights(method="center")[0]
    center_bbox = annulus.bboxes()[0]
    center = center_bbox.to_image(center_weights, data.shape)

    assert_allclose(apsum, npix)
    assert_allclose(npix, annulus.area, rtol=0, atol=0.5)
    assert center[48, 48] == 0.0
    assert np.count_nonzero(center) > 0


def test_public_top_level_exports_prefer_short_names():
    assert not hasattr(apers, "CircularAperture")
    assert not hasattr(apers, "CircularAnnulus")
    assert not hasattr(apers, "ApertureMask")
    assert apers.BoundingBox.__name__ == "BoundingBox"
    assert "CircAp" in apers.__all__
    assert "apsum_circ_ann_exact" in apers.__all__
    assert "apsum_pill_ann_exact" in apers.__all__
    assert "WedgeAp" in apers.__all__
    assert "apsum_wedge_exact" in apers.__all__
    assert "apsum_circ_center" in apers.__all__
    assert "npix_circ_ann_exact" in apers.__all__
    assert "npix_rect_ann_center" in apers.__all__
    assert "npix_circ_ann_center" in apers.__all__
    assert "kernels" in apers.__all__
    assert apers.kernels.apsum_circ_exact is apers.apsum_circ_exact
    removed_names = {
        "apmask_apsum",
        "apmask_npix",
        "apmask_to_image",
        "apmask_weighted_cutout",
        "apmask_weighted_values",
        "bbox_apsum",
        "bbox_npix",
        "bbox_to_image",
        "bbox_weighted_cutout",
        "bbox_weighted_values",
    }
    assert removed_names.isdisjoint(apers.__all__)
    assert all(not hasattr(apers, name) for name in removed_names)


def test_public_kernels_namespace_exports_direct_apsum_functions():
    import astroapers.kernels as aapk

    data = np.ones((8, 8), dtype=np.float64)
    apsum, npix = aapk.apsum_circ_exact(data, [4.0], [4.0], 2.0)
    shapes = {
        "circ",
        "circ_ann",
        "ellip",
        "ellip_ann",
        "rect",
        "rect_ann",
        "pill",
        "pill_ann",
        "wedge",
    }
    expected = {
        f"{operation}_{shape}_{method}"
        for operation in {"apsum", "npix"}
        for shape in shapes
        for method in {"exact", "center"}
    }

    assert expected.issubset(aapk.__all__)
    assert all(callable(getattr(aapk, name)) for name in expected)
    assert all(getattr(apers, name) is getattr(aapk, name) for name in expected)
    assert_allclose(apsum, npix)


def test_public_kernels_namespace_exports_weight_and_bbox_helpers():
    import astroapers.kernels as aapk

    shapes = {
        "circ",
        "circ_ann",
        "ellip",
        "ellip_ann",
        "rect",
        "rect_ann",
        "pill",
        "pill_ann",
        "wedge",
    }
    expected = {
        f"weights_{shape}_{method}"
        for shape in shapes
        for method in {"exact", "center"}
    } | {f"bboxes_{shape}" for shape in shapes}

    assert expected.issubset(aapk.__all__)
    assert all(callable(getattr(aapk, name)) for name in expected)
    assert not hasattr(aapk, "circ_bboxes")
    assert not hasattr(aapk, "ellip_bboxes")
    assert not hasattr(aapk, "rect_bboxes")
    assert "weights_exact" not in aapk.__all__
    assert "weights_center" not in aapk.__all__


def test_public_kernels_namespace_has_complete_apsum_surface():
    import astroapers.kernels as aapk

    y, x = np.indices((40, 40), dtype=np.float64)
    data = 1.0 + 0.01 * x + 0.02 * y
    positions = np.array([[16.2, 17.4], [22.6, 20.5]])
    xpos = positions[:, 0]
    ypos = positions[:, 1]
    shape = data.shape
    exact_cases = [
        (
            apers.CircAp(positions, 4.0),
            aapk.apsum_circ_exact,
            aapk.npix_circ_exact,
            (4.0,),
        ),
        (
            apers.CircAn(positions, 2.0, 5.0),
            aapk.apsum_circ_ann_exact,
            aapk.npix_circ_ann_exact,
            (2.0, 5.0),
        ),
        (
            apers.EllipAp(positions, 5.0, 2.5, 0.3),
            aapk.apsum_ellip_exact,
            aapk.npix_ellip_exact,
            (5.0, 2.5, 0.3),
        ),
        (
            apers.EllipAn(positions, 2.0, 1.0, 5.0, 2.5, theta_in=0.3),
            aapk.apsum_ellip_ann_exact,
            aapk.npix_ellip_ann_exact,
            (2.0, 1.0, 5.0, 2.5, 0.3),
        ),
        (
            apers.RectAp(positions, 6.0, 3.0, 0.2),
            aapk.apsum_rect_exact,
            aapk.npix_rect_exact,
            (6.0, 3.0, 0.2),
        ),
        (
            apers.RectAn(positions, 2.0, 1.0, 6.0, 3.0, theta_in=0.2),
            aapk.apsum_rect_ann_exact,
            aapk.npix_rect_ann_exact,
            (2.0, 1.0, 6.0, 3.0, 0.2),
        ),
        (
            apers.PillAp(positions, 5.0, 1.6, 1.1, 0.35),
            aapk.apsum_pill_exact,
            aapk.npix_pill_exact,
            (5.0, 1.6, 1.1, 0.35),
        ),
        (
            apers.PillAn(
                positions,
                2.0,
                0.8,
                0.55,
                5.0,
                1.6,
                1.1,
                theta_in=0.35,
            ),
            aapk.apsum_pill_ann_exact,
            aapk.npix_pill_ann_exact,
            (2.0, 0.8, 0.55, 5.0, 1.6, 1.1, 0.35),
        ),
        (
            apers.WedgeAp(positions, 2.0, 7.0, 0.3, 0.5),
            aapk.apsum_wedge_exact,
            aapk.npix_wedge_exact,
            (2.0, 7.0, 0.3, 0.5),
        ),
    ]

    center_cases = [
        (
            apers.CircAp(positions, 4.0),
            aapk.apsum_circ_center,
            aapk.npix_circ_center,
            (4.0,),
        ),
        (
            apers.CircAn(positions, 2.0, 5.0),
            aapk.apsum_circ_ann_center,
            aapk.npix_circ_ann_center,
            (2.0, 5.0),
        ),
        (
            apers.EllipAp(positions, 5.0, 2.5, 0.3),
            aapk.apsum_ellip_center,
            aapk.npix_ellip_center,
            (5.0, 2.5, 0.3),
        ),
        (
            apers.EllipAn(positions, 2.0, 1.0, 5.0, 2.5, theta_in=0.3),
            aapk.apsum_ellip_ann_center,
            aapk.npix_ellip_ann_center,
            (2.0, 1.0, 5.0, 2.5, 0.3),
        ),
        (
            apers.RectAp(positions, 6.0, 3.0, 0.2),
            aapk.apsum_rect_center,
            aapk.npix_rect_center,
            (6.0, 3.0, 0.2),
        ),
        (
            apers.RectAn(positions, 2.0, 1.0, 6.0, 3.0, theta_in=0.2),
            aapk.apsum_rect_ann_center,
            aapk.npix_rect_ann_center,
            (2.0, 1.0, 6.0, 3.0, 0.2),
        ),
        (
            apers.PillAp(positions, 5.0, 1.6, 1.1, 0.35),
            aapk.apsum_pill_center,
            aapk.npix_pill_center,
            (5.0, 1.6, 1.1, 0.35),
        ),
        (
            apers.PillAn(
                positions,
                2.0,
                0.8,
                0.55,
                5.0,
                1.6,
                1.1,
                theta_in=0.35,
            ),
            aapk.apsum_pill_ann_center,
            aapk.npix_pill_ann_center,
            (2.0, 0.8, 0.55, 5.0, 1.6, 1.1, 0.35),
        ),
        (
            apers.WedgeAp(positions, 2.0, 7.0, 0.3, 0.5),
            aapk.apsum_wedge_center,
            aapk.npix_wedge_center,
            (2.0, 7.0, 0.3, 0.5),
        ),
    ]

    def expected_from_masks(aperture, method, mask=None):
        sums = []
        npixs = []
        for weight, bbox in zip(
            aperture.weights(method=method), aperture.bboxes(), strict=True
        ):
            apsum, npix = bbox.apsum(weight, data, mask=mask)
            sums.append(apsum)
            npixs.append(npix)
        return np.asarray(sums), np.asarray(npixs)

    for method, cases in (("exact", exact_cases), ("center", center_cases)):
        bad = np.zeros(shape, dtype=bool)
        bad[17, 16] = True
        for aperture, apsum_func, npix_func, params in cases:
            expected_sum, expected_npix = expected_from_masks(aperture, method)
            got_sum, got_npix = apsum_func(data, xpos, ypos, *params)
            got_npix_only = npix_func(xpos, ypos, *params, shape=shape)
            assert_allclose(got_sum, expected_sum)
            assert_allclose(got_npix, expected_npix)
            assert_allclose(got_npix_only, expected_npix)
            assert_allclose(
                apsum_func(data, xpos, ypos, *params, return_npix=False),
                expected_sum,
            )

            masked_sum, masked_npix = apsum_func(data, xpos, ypos, *params, mask=bad)
            expected_masked_sum, expected_masked_npix = expected_from_masks(
                aperture, method, mask=bad
            )
            assert_allclose(masked_sum, expected_masked_sum)
            assert_allclose(masked_npix, expected_masked_npix)


def test_old_sum_exact_names_are_not_public_api():
    assert not hasattr(apers, "sum_circle_exact")
    assert not hasattr(apers, "sum_ellipse_exact")
    assert not hasattr(apers, "sum_rect_exact")


def test_stats_helpers_are_not_public_api():
    assert not hasattr(apers, "ApStats")
    assert not hasattr(apers.CircAp((4.0, 4.0), r=2.0), "stats")


def test_mask_wrapper_api_is_removed_from_public_surface():
    aperture = apers.CircAp((4.0, 4.0), r=2.0)

    assert not hasattr(apers, "ApMask")
    assert not hasattr(aperture, "get_apmask")
    assert not hasattr(aperture, "bbox")


def test_bboxes_and_weights_are_always_lists_for_scalar_aperture():
    aperture = apers.CircAp((4.0, 4.0), r=2.0)

    boxes = aperture.bboxes()
    weights = aperture.weights(method="exact")

    assert isinstance(boxes, list)
    assert isinstance(weights, list)
    assert len(boxes) == len(weights) == 1
    assert weights[0].shape == boxes[0].shape


def test_method_aware_object_apsum_uses_kernel_path_without_weights(monkeypatch):
    data = np.ones((32, 32), dtype=np.float64)
    aperture = apers.CircAp([(10.0, 11.0), (16.0, 17.0)], 3.0)

    def fail_weights(*args, **kwargs):
        raise AssertionError("optimized object apsum should not build weight arrays")

    monkeypatch.setattr(aperture, "weights", fail_weights)

    got_apsum, got_npix = aperture.apsum(data, method="center")
    expected_apsum, expected_npix = apers.apsum_circ_center(
        data, [10.0, 16.0], [11.0, 17.0], 3.0
    )

    assert_allclose(got_apsum, expected_apsum)
    assert_allclose(got_npix, expected_npix)


def test_masked_object_npix_uses_weight_sum_without_apsum(monkeypatch):
    aperture = apers.CircAp([(10.0, 11.0), (16.0, 17.0)], 3.0)
    bad = np.zeros((32, 32), dtype=bool)
    bad[11, 10] = True

    def fail_apsum(*args, **kwargs):
        raise AssertionError("masked npix should not allocate image data or call apsum")

    monkeypatch.setattr(aperture, "apsum", fail_apsum)

    got = aperture.npix(bad.shape, method="center", mask=bad)
    expected = np.array(
        [
            bbox.npix(weight, bad.shape, mask=bad)
            for weight, bbox in zip(
                aperture.weights(method="center"), aperture.bboxes(), strict=True
            )
        ]
    )

    assert_allclose(got, expected)

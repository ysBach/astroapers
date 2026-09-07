from __future__ import annotations

import copy
import pickle

import astroapers as apers
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_object_scalar_apsum_preserves_zero_dimensional_return():
    data = np.ones((12, 12), dtype=np.float64)
    apsum, npix = apers.CircAp((4.3, 5.1), 2.4).apsum_exact(data)

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
        apsum, _ = aperture.apsum_exact(data)
        assert_allclose(aperture.apsum_exact(data, return_npix=False), apsum)
        masked_apsum, _ = aperture.apsum_exact(data, mask=bad)
        assert_allclose(
            aperture.apsum_exact(data, mask=bad, return_npix=False),
            masked_apsum,
        )


def test_vector_circle_weights_reject_unreasonably_large_radius():
    aperture = apers.CircAp([(0.0, 0.0), (1.0, 1.0)], r=1.0e20)

    with pytest.raises((MemoryError, ValueError)):
        aperture.weights_exact()


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


def test_rectangular_annulus_allows_equal_width_when_height_grows():
    data = np.ones((32, 32), dtype=np.float64)
    annulus = apers.RectAn((16.0, 16.0), 1.0, 5.0, 1.0, 20.0, theta_in=0.1)

    apsum, npix = annulus.apsum_exact(data)

    assert_allclose(annulus.area, 15.0)
    assert_allclose(apsum, npix)
    assert_allclose(npix, annulus.area, atol=1e-12)


def test_object_api_excludes_true_mask_pixels_from_apsum_and_npix():
    data = np.ones((9, 9), dtype=np.float64)
    aperture = apers.CircAp((4.0, 4.0), r=2.0)
    full_apsum, full_npix = aperture.apsum_exact(data)

    bad = np.zeros_like(data, dtype=bool)
    bad[4, 4] = True
    masked_apsum, masked_npix = aperture.apsum_exact(data, mask=bad)

    assert_allclose(full_apsum, full_npix)
    assert masked_apsum < full_apsum
    assert_allclose(masked_apsum, masked_npix)


def test_sampled_values_scalar_returns_list_of_unweighted_center_selected_data():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    aperture = apers.CircAp((2.0, 2.0), r=1.1)

    values = aperture.sampled_values(data)
    weights = aperture.weights_center()[0]
    bbox = aperture.bboxes()[0]
    selected = bbox.to_image(weights, data.shape).astype(bool)

    assert isinstance(values, list)
    assert len(values) == 1
    assert_allclose(values[0], data[selected].ravel())


def test_sampled_values_applies_bad_pixel_mask():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    aperture = apers.CircAp((2.0, 2.0), r=1.1)
    bad = np.zeros_like(data, dtype=bool)
    bad[2, 2] = True

    values = aperture.sampled_values(data, mask=bad)
    weights = aperture.weights_center()[0]
    bbox = aperture.bboxes()[0]
    expected = bbox.weighted_values(weights, data, mask=bad)

    assert_allclose(values[0], expected)


def test_sampled_values_return_pix_matches_center_selected_value_order():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    aperture = apers.CircAp((2.0, 2.0), r=1.1)

    values, pix = aperture.sampled_values(data, return_pix=True)
    weights = aperture.weights_center()[0]
    bbox = aperture.bboxes()[0]
    selected = bbox.to_image(weights, data.shape).astype(bool)
    yy, xx = np.nonzero(selected)

    assert isinstance(values, list)
    assert isinstance(pix, list)
    assert len(values) == len(pix) == 1
    pix_y, pix_x = pix[0]
    assert_allclose(values[0], data[selected].ravel())
    assert_allclose(pix_y, yy)
    assert_allclose(pix_x, xx)


def test_sampled_values_return_pix_applies_bad_pixel_mask_to_pix():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    aperture = apers.CircAp((2.0, 2.0), r=1.1)
    bad = np.zeros_like(data, dtype=bool)
    bad[2, 2] = True

    values, pix = aperture.sampled_values(data, mask=bad, return_pix=True)
    pix_y, pix_x = pix[0]

    assert not np.any((pix_y == 2.0) & (pix_x == 2.0))
    assert values[0].size == pix_y.size == pix_x.size


def test_sampled_values_return_pix_flat_reconstructs_values_and_pix():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.RectAp([(2.0, 2.0), (7.0, 7.0)], w=3.0, h=3.0)

    values, pix = aperture.sampled_values(data, return_pix=True)
    flat_values, flat_pix, offsets = aperture.sampled_values(
        data, flat=True, return_pix=True
    )
    split_values = np.split(flat_values, offsets[1:-1])
    flat_pix_y, flat_pix_x = flat_pix
    split_pix_y = np.split(flat_pix_y, offsets[1:-1])
    split_pix_x = np.split(flat_pix_x, offsets[1:-1])

    assert flat_values.shape == flat_pix_y.shape == flat_pix_x.shape
    assert offsets[-1] == flat_values.size
    for got_values, want_values in zip(split_values, values, strict=True):
        assert_allclose(got_values, want_values)
    for got_y, got_x, want_pix in zip(split_pix_y, split_pix_x, pix, strict=True):
        want_y, want_x = want_pix
        assert_allclose(got_y, want_y)
        assert_allclose(got_x, want_x)


def test_weighted_values_returns_positive_weighted_data():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    data[2, 2] = 0.0
    aperture = apers.CircAp((2.1, 2.2), r=1.4)

    values = aperture.weighted_values(data)
    weights = aperture.weights_exact()[0]
    bbox = aperture.bboxes()[0]
    weighted = bbox.weighted_cutout(weights, data)
    expected = weighted[weights > 0.0].ravel()

    assert isinstance(values, list)
    assert len(values) == 1
    assert_allclose(values[0], expected)


def test_weighted_values_applies_bad_pixel_mask():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    aperture = apers.CircAp((2.1, 2.2), r=1.4)
    bad = np.zeros_like(data, dtype=bool)
    bad[2, 2] = True

    values = aperture.weighted_values(data, mask=bad)
    weights = aperture.weights_exact()[0]
    bbox = aperture.bboxes()[0]
    expected = bbox.weighted_values(weights, data, mask=bad)

    assert_allclose(values[0], expected)


def test_weighted_values_vector_aperture_returns_list_of_arrays():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.RectAp([(2.0, 2.0), (7.0, 7.0)], w=3.0, h=3.0)

    values = aperture.weighted_values(data)

    assert isinstance(values, list)
    assert len(values) == 2
    assert all(value.ndim == 1 for value in values)
    assert_allclose(
        values[0], apers.RectAp((2.0, 2.0), w=3.0, h=3.0).weighted_values(data)[0]
    )
    assert_allclose(
        values[1], apers.RectAp((7.0, 7.0), w=3.0, h=3.0).weighted_values(data)[0]
    )


def test_value_methods_return_empty_arrays_for_out_of_frame_aperture():
    data = np.ones((5, 5), dtype=np.float64)
    aperture = apers.CircAp((-100.0, -100.0), r=1.0)

    sampled = aperture.sampled_values(data)
    weighted = aperture.weighted_values(data)

    assert len(sampled) == len(weighted) == 1
    assert sampled[0].shape == (0,)
    assert weighted[0].shape == (0,)
    assert sampled[0].dtype == np.float64
    assert weighted[0].dtype == np.float64


@pytest.mark.parametrize("method_name", ["sampled_values", "weighted_values"])
def test_flat_values_reconstruct_default_list_output(method_name):
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.CircAp([(2.0, 2.0), (7.0, 7.0), (-100.0, -100.0)], r=1.5)

    expected = getattr(aperture, method_name)(data)
    flat_values, offsets = getattr(aperture, method_name)(data, flat=True)
    reconstructed = np.split(flat_values, offsets[1:-1])

    assert flat_values.ndim == 1
    assert offsets.shape == (len(aperture.positions) + 1,)
    assert np.issubdtype(offsets.dtype, np.integer)
    assert offsets[0] == 0
    assert offsets[-1] == flat_values.size
    assert offsets[-2] == offsets[-1]
    assert len(reconstructed) == len(expected)
    for got, want in zip(reconstructed, expected, strict=True):
        assert_allclose(got, want)


@pytest.mark.parametrize("method_name", ["sampled_values", "weighted_values"])
def test_flat_values_reconstruct_masked_and_fully_masked_outputs(method_name):
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    aperture = apers.RectAp([(2.0, 2.0), (7.0, 7.0)], w=3.0, h=3.0)
    partial_mask = np.zeros_like(data, dtype=bool)
    partial_mask[2, 2] = True
    full_mask = np.ones_like(data, dtype=bool)

    expected = getattr(aperture, method_name)(data, mask=partial_mask)
    flat_values, offsets = getattr(aperture, method_name)(
        data, mask=partial_mask, flat=True
    )
    reconstructed = np.split(flat_values, offsets[1:-1])

    for got, want in zip(reconstructed, expected, strict=True):
        assert_allclose(got, want)

    full_values = getattr(aperture, method_name)(data, mask=full_mask)
    full_flat, full_offsets = getattr(aperture, method_name)(
        data, mask=full_mask, flat=True
    )
    assert full_flat.shape == (0,)
    assert np.array_equal(full_offsets, np.zeros(len(aperture.positions) + 1))
    assert all(values.shape == (0,) for values in full_values)


def test_sampled_cutout_preserves_bbox_shape_and_uses_fill_value():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    aperture = apers.CircAp((2.0, 2.0), r=1.1)
    bad = np.zeros_like(data, dtype=bool)
    bad[2, 2] = True

    cutout = aperture.sampled_cutout(data, mask=bad, fill_value=-999.0)
    weights = aperture.weights_center()[0]
    bbox = aperture.bboxes()[0]
    expected = bbox.weighted_cutout(weights, data, mask=bad, fill_value=-999.0)

    assert isinstance(cutout, list)
    assert len(cutout) == 1
    assert cutout[0].shape == bbox.shape
    assert_allclose(cutout[0], expected)


def test_sampled_cutout_return_pix_matches_cutout_shape_and_mask():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    aperture = apers.CircAp((2.0, 2.0), r=1.1)
    bad = np.zeros_like(data, dtype=bool)
    bad[2, 2] = True

    cutout, pix = aperture.sampled_cutout(data, mask=bad, return_pix=True)
    pix_y, pix_x = pix[0]
    valid = np.isfinite(pix_y)

    assert cutout[0].shape == pix_y.shape == pix_x.shape
    assert not valid[1, 1]
    assert_allclose(pix_y[valid], [1.0, 2.0, 2.0, 3.0])
    assert_allclose(pix_x[valid], [2.0, 1.0, 3.0, 2.0])


def test_weighted_cutout_preserves_bbox_shape_and_uses_fill_value():
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    aperture = apers.CircAp((2.1, 2.2), r=1.4)
    bad = np.zeros_like(data, dtype=bool)
    bad[2, 2] = True

    cutout = aperture.weighted_cutout(data, mask=bad, fill_value=-999.0)
    weights = aperture.weights_exact()[0]
    bbox = aperture.bboxes()[0]
    expected = bbox.weighted_cutout(weights, data, mask=bad, fill_value=-999.0)

    assert isinstance(cutout, list)
    assert len(cutout) == 1
    assert cutout[0].shape == bbox.shape
    assert_allclose(cutout[0], expected)


def test_weighted_values_rejects_bad_pixel_mask_with_wrong_shape():
    data = np.ones((5, 5), dtype=np.float64)
    aperture = apers.CircAp((2.0, 2.0), r=1.0)

    with pytest.raises(ValueError, match="mask must have the same shape as data"):
        aperture.weighted_values(data, mask=np.zeros((4, 5), dtype=bool))


def test_vector_aperture_returns_vector_weights_and_apsum():
    data = np.ones((16, 16), dtype=np.float64)
    aperture = apers.RectAp([(4.0, 4.0), (8.5, 9.0)], w=3.0, h=2.0)

    weights = aperture.weights_exact()
    apsum, npix = aperture.apsum_exact(data)

    assert len(weights) == 2
    assert apsum.shape == (2,)
    assert_allclose(apsum, npix)
    assert_allclose(npix, [6.0, 6.0], atol=1e-12)


def _position_input(layout: str, dtype: type[np.generic]) -> np.ndarray:
    if layout == "c":
        return np.array([[2.0, 2.0], [7.0, 7.0]], dtype=dtype, order="C")
    if layout == "f":
        return np.array([[2.0, 2.0], [7.0, 7.0]], dtype=dtype, order="F")
    if layout == "strided":
        return np.array([[2.0, 2.0], [99.0, 99.0], [7.0, 7.0]], dtype=dtype)[::2]
    raise AssertionError(f"unknown position layout: {layout}")


@pytest.mark.parametrize(
    ("layout", "dtype"),
    [
        ("c", np.float64),
        ("f", np.float64),
        ("strided", np.float64),
        ("c", np.float32),
        ("c", np.int32),
    ],
)
@pytest.mark.parametrize("validate", [True, False])
def test_position_inputs_are_owned_float64_and_read_only(
    layout: str, dtype: type[np.generic], validate: bool
) -> None:
    source = _position_input(layout, dtype)

    aperture = apers.CircAp(source, r=0.1, validate=validate)

    assert aperture.positions.shape == (2, 2)
    assert aperture.positions.dtype == np.float64
    assert aperture.positions.flags.f_contiguous
    assert not aperture.positions.flags.writeable
    assert not np.shares_memory(aperture.positions, source)
    assert aperture._x.flags.c_contiguous
    assert aperture._y.flags.c_contiguous
    assert np.shares_memory(aperture.positions, aperture._x)
    assert np.shares_memory(aperture.positions, aperture._y)
    assert np.array_equal(aperture.positions, source)


@pytest.mark.parametrize("kind", ["scalar", "singleton", "empty"])
@pytest.mark.parametrize("validate", [True, False])
def test_position_shapes_preserve_scalar_and_vector_result_shapes(
    kind: str, validate: bool
) -> None:
    if kind == "scalar":
        positions = (2.0, 2.0)
        expected_shape = ()
        expected_length = 1
    elif kind == "singleton":
        positions = np.array([[2.0, 2.0]], dtype=np.float32)
        expected_shape = (1,)
        expected_length = 1
    else:
        positions = np.empty((0, 2), dtype=np.float64)
        expected_shape = (0,)
        expected_length = 0

    aperture = apers.CircAp(positions, r=0.1, validate=validate)
    apsum, npix = aperture.apsum_center(np.arange(12 * 12).reshape(12, 12))

    assert aperture.positions.shape == (expected_length, 2)
    assert aperture.isscalar is (kind == "scalar")
    assert apsum.shape == expected_shape
    assert npix.shape == expected_shape


@pytest.mark.parametrize("layout", ["c", "f", "strided"])
@pytest.mark.parametrize("validate", [True, False])
def test_position_source_mutation_does_not_desynchronize_aperture(
    layout: str, validate: bool
) -> None:
    source = _position_input(layout, np.float64)
    aperture = apers.CircAp(source, r=0.1, validate=validate)
    reference = apers.CircAp(np.array([[2.0, 2.0], [7.0, 7.0]]), r=0.1)
    data = np.arange(12 * 12, dtype=np.float64).reshape(12, 12)

    source[...] = [[8.0, 8.0], [9.0, 9.0]]

    assert np.array_equal(aperture.positions, reference.positions)
    assert aperture.bboxes() == reference.bboxes()
    for method in ("weights_center", "weights_exact"):
        got = getattr(aperture, method)()
        expected = getattr(reference, method)()
        for got_weights, expected_weights in zip(got, expected, strict=True):
            assert_allclose(got_weights, expected_weights)
    for method in ("npix_center", "npix_exact"):
        got = getattr(aperture, method)(data.shape)
        expected = getattr(reference, method)(data.shape)
        assert_allclose(got, expected)
    for method in ("apsum_center", "apsum_exact"):
        got = getattr(aperture, method)(data)
        expected = getattr(reference, method)(data)
        assert_allclose(got[0], expected[0])
        assert_allclose(got[1], expected[1])
    for method in ("sampled_values", "weighted_values"):
        got = getattr(aperture, method)(data)
        expected = getattr(reference, method)(data)
        assert len(got) == len(expected)
        for got_values, expected_values in zip(got, expected, strict=True):
            assert_allclose(got_values, expected_values)
    for method in ("sampled_cutout", "weighted_cutout"):
        got = getattr(aperture, method)(data)
        expected = getattr(reference, method)(data)
        for got_cutout, expected_cutout in zip(got, expected, strict=True):
            assert_allclose(got_cutout, expected_cutout)


def test_reconstructing_aperture_moves_position_dependent_methods() -> None:
    positions = np.array([[2.0, 2.0]])
    aperture = apers.CircAp(positions, r=0.1)
    data = np.arange(12 * 12, dtype=np.float64).reshape(12, 12)

    positions[...] = [[7.0, 7.0]]
    moved = apers.CircAp(positions, r=0.1)
    reference = apers.CircAp((7.0, 7.0), r=0.1)

    assert aperture.bboxes() != moved.bboxes()
    assert moved.bboxes() == reference.bboxes()
    for method in ("weights_center", "weights_exact"):
        got = getattr(moved, method)()
        expected = getattr(reference, method)()
        for got_weights, expected_weights in zip(got, expected, strict=True):
            assert_allclose(got_weights, expected_weights)
    for method in ("npix_center", "npix_exact"):
        got = getattr(moved, method)(data.shape)
        expected = getattr(reference, method)(data.shape)
        assert_allclose(got, expected)
    for method in ("apsum_center", "apsum_exact"):
        got = getattr(moved, method)(data)
        expected = getattr(reference, method)(data)
        assert_allclose(got[0], expected[0])
        assert_allclose(got[1], expected[1])
    for method in ("sampled_values", "weighted_values"):
        got = getattr(moved, method)(data)
        expected = getattr(reference, method)(data)
        for got_values, expected_values in zip(got, expected, strict=True):
            assert_allclose(got_values, expected_values)
    for method in ("sampled_cutout", "weighted_cutout"):
        got = getattr(moved, method)(data)
        expected = getattr(reference, method)(data)
        for got_cutout, expected_cutout in zip(got, expected, strict=True):
            assert_allclose(got_cutout, expected_cutout)


@pytest.mark.parametrize("positions", [(2.0, 2.0), np.array([[2.0, 2.0]])])
def test_positions_cannot_be_written_or_reassigned(
    positions: tuple[float, float] | np.ndarray,
) -> None:
    aperture = apers.CircAp(positions, r=0.1)

    with pytest.raises(ValueError, match="read-only"):
        aperture.positions[0, 0] = 3.0
    exposed = aperture.positions
    with pytest.raises(ValueError):
        exposed.flags.writeable = True
    with pytest.raises(AttributeError):
        aperture.positions = np.array([[3.0, 3.0]])


def test_positions_view_metadata_cannot_corrupt_aperture_storage() -> None:
    aperture = apers.CircAp(np.array([[2.0, 2.0], [7.0, 7.0]]), r=0.1)
    exposed = aperture.positions

    exposed.dtype = np.uint64

    assert aperture.positions.dtype == np.float64
    assert aperture.positions.shape == (2, 2)
    assert_allclose(aperture.positions, [[2.0, 2.0], [7.0, 7.0]])
    assert_allclose(aperture._x, [2.0, 7.0])
    assert_allclose(aperture._y, [2.0, 7.0])


@pytest.mark.parametrize("operation", ["copy", "deepcopy", "pickle"])
@pytest.mark.parametrize("kind", ["circ_scalar", "circ_vector", "path_vector"])
def test_position_copy_roundtrips_preserve_storage_and_results(
    operation: str, kind: str
) -> None:
    segments = [
        ("move", -1.0, -1.0),
        ("line", 1.0, -1.0),
        ("line", 1.0, 1.0),
        ("line", -1.0, 1.0),
        ("close",),
    ]
    if kind == "circ_scalar":
        aperture = apers.CircAp((2.0, 2.0), r=0.1)
    elif kind == "circ_vector":
        aperture = apers.CircAp(np.array([[2.0, 2.0], [7.0, 7.0]]), r=0.1)
    else:
        aperture = apers.PathAp(np.array([[2.0, 2.0], [7.0, 7.0]]), segments=segments)

    if operation == "copy":
        restored = copy.copy(aperture)
    elif operation == "deepcopy":
        restored = copy.deepcopy(aperture)
    else:
        restored = pickle.loads(pickle.dumps(aperture))

    assert restored is not aperture
    assert restored.isscalar is aperture.isscalar
    assert restored.positions.dtype == np.float64
    assert restored.positions.flags.f_contiguous
    assert not restored.positions.flags.writeable
    assert not np.shares_memory(restored.positions, aperture.positions)
    assert restored._x.flags.c_contiguous
    assert restored._y.flags.c_contiguous
    assert np.shares_memory(restored.positions, restored._x)
    assert np.shares_memory(restored.positions, restored._y)

    data = np.arange(12 * 12, dtype=np.float64).reshape(12, 12)
    assert restored.bboxes() == aperture.bboxes()
    for method in ("apsum_center", "apsum_exact"):
        got = getattr(restored, method)(data)
        expected = getattr(aperture, method)(data)
        assert got[0].shape == expected[0].shape
        assert got[1].shape == expected[1].shape
        assert_allclose(got[0], expected[0])
        assert_allclose(got[1], expected[1])


def test_validate_false_owns_positions_and_skips_parameter_checks() -> None:
    positions = np.array([[4.0, 4.0], [8.5, 9.0]], dtype=np.float64)

    aperture = apers.CircAp(positions, r=2.0, validate=False)
    unchecked = apers.CircAp(positions, r=-2.0, validate=False)

    assert not np.shares_memory(aperture.positions, positions)
    assert aperture.positions.dtype == np.float64
    assert not aperture.positions.flags.writeable
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


def test_validate_false_skips_position_finiteness_check() -> None:
    positions = np.array([[4.0, 4.0], [np.nan, 9.0]], dtype=np.float64)

    aperture = apers.CircAp(positions, r=2.0, validate=False)

    assert not np.shares_memory(aperture.positions, positions)
    assert np.isnan(aperture.positions[1, 0])


def test_object_sum_aliases_are_not_exposed():
    aperture = apers.CircAp((4.0, 4.0), r=2.0)

    assert not hasattr(aperture, "sum")
    assert not hasattr(aperture, "do_photometry")


def test_object_methods_do_not_accept_method_keyword():
    data = np.ones((16, 16), dtype=np.float64)
    aperture = apers.CircAp((4.0, 4.0), r=2.0, validate=False)

    with pytest.raises(TypeError):
        aperture.weights_exact(method="not-a-method")
    with pytest.raises(TypeError):
        aperture.apsum_exact(data, method="not-a-method")


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

    apsum, npix = aperture.apsum_exact(data)

    assert np.isfinite(apsum)
    assert np.isfinite(npix)


def test_validate_false_object_npix_skips_mask_shape_validation(monkeypatch):
    aperture = apers.CircAp((4.0, 4.0), r=2.0, validate=False)
    mask = np.zeros((16, 16), dtype=bool)

    bad_mask = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="mask must have the same shape"):
        apers.CircAp((4.0, 4.0), r=2.0).npix_exact((16, 16), mask=bad_mask)

    def fail_validation(*args, **kwargs):
        raise AssertionError("mask validation should be skipped")

    monkeypatch.setattr(apers.kernels, "validate_mask", fail_validation)

    aperture.npix_exact((16, 16), mask=mask)


@pytest.mark.parametrize("bad_r_in", [-1.0, np.nan])
def test_circular_annulus_rejects_invalid_inner_radius(bad_r_in):
    with pytest.raises(ValueError, match="r_in must be a nonnegative finite scalar"):
        apers.CircAn((4.0, 4.0), r_in=bad_r_in, r_out=2.0)


def test_pillbox_custom_aperture_on_uniform_data_tracks_area():
    data = np.ones((64, 64), dtype=np.float64)
    aperture = apers.PillAp((32.0, 32.0), w=8.0, a=2.0, b=2.0, theta=0.3)

    apsum, npix = aperture.apsum_exact(data)

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

    apsum, npix = annulus.apsum_exact(data)
    center_weights = annulus.weights_center()[0]
    center_bbox = annulus.bboxes()[0]
    center = center_bbox.to_image(center_weights, data.shape)

    assert_allclose(apsum, npix)
    assert_allclose(npix, annulus.area, rtol=0, atol=0.5)
    assert center[48, 48] == 0.0
    assert np.count_nonzero(center) > 0


def test_public_top_level_exports_prefer_short_names():
    import astroapers._rust as aapr

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
    assert "_rust" in apers.__all__
    assert "kernels" in apers.__all__
    assert aapr.apsum_circ_exact_sum is apers._rust.apsum_circ_exact_sum
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
        weights = (
            aperture.weights_exact() if method == "exact" else aperture.weights_center()
        )
        for weight, bbox in zip(weights, aperture.bboxes(), strict=True):
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
    weights = aperture.weights_exact()

    assert isinstance(boxes, list)
    assert isinstance(weights, list)
    assert len(boxes) == len(weights) == 1
    assert weights[0].shape == boxes[0].shape


def test_center_object_apsum_uses_kernel_path_without_weights(monkeypatch):
    data = np.ones((32, 32), dtype=np.float64)
    aperture = apers.CircAp([(10.0, 11.0), (16.0, 17.0)], 3.0)

    def fail_weights(*args, **kwargs):
        raise AssertionError("optimized object apsum should not build weight arrays")

    monkeypatch.setattr(aperture, "_weights_center", fail_weights)

    got_apsum, got_npix = aperture.apsum_center(data)
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

    monkeypatch.setattr(aperture, "_apsum_center", fail_apsum)

    got = aperture.npix_center(bad.shape, mask=bad)
    expected = np.array(
        [
            bbox.npix(weight, bad.shape, mask=bad)
            for weight, bbox in zip(
                aperture.weights_center(), aperture.bboxes(), strict=True
            )
        ]
    )

    assert_allclose(got, expected)

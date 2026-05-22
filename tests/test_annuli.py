from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astroapers as apers


def _weights_box(aperture, method: str = "exact", idx: int = 0):
    return aperture.weights(method=method)[idx], aperture.bboxes()[idx]


def _subtracted_annulus_reference(outer, inner):
    outer_weights, outer_bbox = outer
    inner_weights, inner_bbox = inner
    expected = outer_weights.copy()
    ixmin = max(outer_bbox.ixmin, inner_bbox.ixmin)
    ixmax = min(outer_bbox.ixmax, inner_bbox.ixmax)
    iymin = max(outer_bbox.iymin, inner_bbox.iymin)
    iymax = min(outer_bbox.iymax, inner_bbox.iymax)
    if ixmin < ixmax and iymin < iymax:
        outer_slice = (
            slice(iymin - outer_bbox.iymin, iymax - outer_bbox.iymin),
            slice(ixmin - outer_bbox.ixmin, ixmax - outer_bbox.ixmin),
        )
        inner_slice = (
            slice(iymin - inner_bbox.iymin, iymax - inner_bbox.iymin),
            slice(ixmin - inner_bbox.ixmin, ixmax - inner_bbox.ixmin),
        )
        expected[outer_slice] -= inner_weights[inner_slice]
    return np.clip(expected, 0.0, 1.0)


def _pill_component_reference(aperture, x, y, bbox, method):
    components = aperture._components(x, y)
    if method == "center":
        weights = [
            component._center_weights_one(*component.positions[0], bbox)
            for component in components
        ]
    else:
        weights = [
            component._exact_weights_one(*component.positions[0], bbox)
            for component in components
        ]
    return np.maximum.reduce(weights)


@pytest.mark.parametrize("method", ["exact", "center"])
def test_circular_annulus_fused_weights_match_subtracted_circle_reference(method):
    positions = np.array([(4.3, 5.1), (-1.2, 2.6), (12.4, 3.5)], dtype=np.float64)
    r_in = 1.7
    r_out = 4.2
    annulus = apers.CircAn(positions, r_in=r_in, r_out=r_out)

    weights = annulus.weights(method=method)
    boxes = annulus.bboxes()
    for weight, bbox, (x, y) in zip(weights, boxes, positions, strict=True):
        outer = _weights_box(apers.CircAp((float(x), float(y)), r_out), method)
        inner = _weights_box(apers.CircAp((float(x), float(y)), r_in), method)
        expected = _subtracted_annulus_reference(outer, inner)

        assert bbox == outer[1]
        assert_allclose(weight, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("method", ["exact", "center"])
def test_elliptical_annulus_fused_weights_match_subtracted_ellipse_reference(method):
    positions = np.array([(4.3, 5.1), (-1.2, 2.6), (12.4, 3.5)], dtype=np.float64)
    a_in, b_in, a_out, b_out, theta = 1.3, 0.8, 4.2, 2.7, 0.35
    annulus = apers.EllipAn(positions, a_in, b_in, a_out, b_out, theta_in=theta)

    weights = annulus.weights(method=method)
    boxes = annulus.bboxes()
    for weight, bbox, (x, y) in zip(weights, boxes, positions, strict=True):
        outer = _weights_box(
            apers.EllipAp((float(x), float(y)), a_out, b_out, theta), method
        )
        inner = _weights_box(
            apers.EllipAp((float(x), float(y)), a_in, b_in, theta), method
        )
        expected = _subtracted_annulus_reference(outer, inner)

        assert bbox == outer[1]
        assert_allclose(weight, expected, rtol=0, atol=5e-14)


@pytest.mark.parametrize("method", ["exact", "center"])
def test_rectangular_annulus_fused_weights_match_subtracted_rectangle_reference(method):
    positions = np.array([(4.3, 5.1), (-1.2, 2.6), (12.4, 3.5)], dtype=np.float64)
    w_in, h_in, w_out, h_out, theta = 1.5, 0.9, 4.8, 3.1, -0.35
    annulus = apers.RectAn(positions, w_in, h_in, w_out, h_out, theta_in=theta)

    weights = annulus.weights(method=method)
    boxes = annulus.bboxes()
    for weight, bbox, (x, y) in zip(weights, boxes, positions, strict=True):
        outer = _weights_box(
            apers.RectAp((float(x), float(y)), w_out, h_out, theta), method
        )
        inner = _weights_box(
            apers.RectAp((float(x), float(y)), w_in, h_in, theta), method
        )
        expected = _subtracted_annulus_reference(outer, inner)

        assert bbox == outer[1]
        assert_allclose(weight, expected, rtol=0, atol=5e-14)


@pytest.mark.parametrize("method", ["exact", "center"])
def test_pill_fused_weights_match_component_reference(method):
    positions = np.array([(4.3, 5.1), (-1.2, 2.6), (12.4, 3.5)], dtype=np.float64)
    aperture = apers.PillAp(positions, w=5.0, a=1.6, b=1.1, theta=0.35)

    weights = aperture.weights(method=method)
    boxes = aperture.bboxes()
    for weight, bbox, (x, y) in zip(weights, boxes, positions, strict=True):
        scalar = apers.PillAp((float(x), float(y)), 5.0, 1.6, 1.1, 0.35)
        expected = _pill_component_reference(scalar, float(x), float(y), bbox, method)

        assert bbox == scalar.bboxes()[0]
        assert_allclose(weight, expected, rtol=0, atol=5e-14)


@pytest.mark.parametrize("method", ["exact", "center"])
def test_pill_annulus_fused_weights_match_component_reference(method):
    positions = np.array([(4.3, 5.1), (-1.2, 2.6), (12.4, 3.5)], dtype=np.float64)
    w_in, a_in, b_in, w_out, a_out, b_out, theta = 3.0, 0.8, 0.55, 5.0, 1.8, 1.2, 0.35
    annulus = apers.PillAn(
        positions, w_in, a_in, b_in, w_out, a_out, b_out, theta_in=theta
    )

    weights = annulus.weights(method=method)
    boxes = annulus.bboxes()
    for weight, bbox, (x, y) in zip(weights, boxes, positions, strict=True):
        outer = apers.PillAp((float(x), float(y)), w_out, a_out, b_out, theta)
        inner = apers.PillAp((float(x), float(y)), w_in, a_in, b_in, theta)
        outer_weights = _pill_component_reference(
            outer, float(x), float(y), bbox, method
        )
        inner_weights = _pill_component_reference(
            inner, float(x), float(y), bbox, method
        )
        expected = np.clip(outer_weights - inner_weights, 0.0, 1.0)

        assert bbox == outer.bboxes()[0]
        assert_allclose(weight, expected, rtol=0, atol=5e-14)


@pytest.mark.parametrize(
    ("annulus", "atol"),
    [
        (apers.CircAn((32.2, 31.7), 1.4, 4.2), 1e-8),
        (apers.EllipAn((32.2, 31.7), 1.3, 0.8, 4.2, 2.7, theta_in=0.35), 1e-8),
        (apers.RectAn((32.2, 31.7), 1.5, 0.9, 4.8, 3.1, theta_in=-0.35), 1e-12),
    ],
)
def test_exact_annulus_weight_sum_tracks_analytic_area(annulus, atol):
    weight = annulus.weights(method="exact")[0]

    assert_allclose(weight.sum(), annulus.area, rtol=0, atol=atol)


@pytest.mark.parametrize("method", ["exact", "center"])
def test_zero_inner_circular_annulus_matches_circle_weights(method):
    position = (6.2, 7.4)
    annulus = apers.CircAn(position, r_in=0.0, r_out=3.5)
    circle = apers.CircAp(position, r=3.5)
    annulus_weight, annulus_bbox = _weights_box(annulus, method)
    circle_weight, circle_bbox = _weights_box(circle, method)

    assert annulus_bbox == circle_bbox
    assert_allclose(annulus_weight, circle_weight, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["exact", "center"])
def test_split_theta_annulus_masks_match_subtracted_references(method):
    position = (8.3, 9.1)
    ellip_ap = apers.EllipAn(
        position,
        a_in=1.0,
        b_in=0.7,
        a_out=4.0,
        b_out=2.5,
        theta_in=-0.4,
        theta_out=0.35,
    )
    ellip_weight, ellip_bbox = _weights_box(ellip_ap, method)
    ellip_outer = _weights_box(apers.EllipAp(position, 4.0, 2.5, 0.35), method)
    ellip_inner = _weights_box(apers.EllipAp(position, 1.0, 0.7, -0.4), method)
    assert ellip_bbox == ellip_outer[1]
    assert_allclose(
        ellip_weight,
        _subtracted_annulus_reference(ellip_outer, ellip_inner),
        rtol=0,
        atol=5e-14,
    )

    rect_ap = apers.RectAn(
        position,
        w_in=1.2,
        h_in=0.8,
        w_out=4.5,
        h_out=3.0,
        theta_in=-0.45,
        theta_out=0.25,
    )
    rect_weight, rect_bbox = _weights_box(rect_ap, method)
    rect_outer = _weights_box(apers.RectAp(position, 4.5, 3.0, 0.25), method)
    rect_inner = _weights_box(apers.RectAp(position, 1.2, 0.8, -0.45), method)
    assert rect_bbox == rect_outer[1]
    assert_allclose(
        rect_weight,
        _subtracted_annulus_reference(rect_outer, rect_inner),
        rtol=0,
        atol=5e-14,
    )

    pill_ap = apers.PillAn(
        position,
        w_in=2.0,
        a_in=0.9,
        b_in=0.6,
        w_out=5.0,
        a_out=1.8,
        b_out=1.2,
        theta_in=-0.25,
        theta_out=0.35,
    )
    pill_weight, pill_bbox = _weights_box(pill_ap, method)
    pill_outer = apers.PillAp(position, 5.0, 1.8, 1.2, 0.35)
    pill_inner = apers.PillAp(position, 2.0, 0.9, 0.6, -0.25)
    outer_weights = _pill_component_reference(
        pill_outer, position[0], position[1], pill_bbox, method
    )
    inner_weights = _pill_component_reference(
        pill_inner, position[0], position[1], pill_bbox, method
    )
    assert pill_bbox == pill_outer.bboxes()[0]
    assert_allclose(
        pill_weight,
        np.clip(outer_weights - inner_weights, 0.0, 1.0),
        rtol=0,
        atol=5e-14,
    )


def test_split_theta_annulus_apsum_uses_clipped_mask_path():
    data = np.arange(24 * 26, dtype=np.float64).reshape(24, 26)
    position = (10.2, 11.7)
    aperture = apers.EllipAn(
        position,
        a_in=3.4,
        b_in=1.0,
        a_out=4.0,
        b_out=2.0,
        theta_in=np.pi / 2.0,
        theta_out=0.0,
    )

    apsum, npix = aperture.apsum(data)
    weights, bbox = _weights_box(aperture)
    expected = bbox.apsum(weights, data)

    assert_allclose(apsum, expected[0])
    assert_allclose(npix, expected[1])


def test_annulus_exact_sum_tracks_area_on_uniform_data():
    data = np.ones((32, 32), dtype=np.float64)
    annulus = apers.CircAn((16.0, 16.0), r_in=3.0, r_out=5.0)
    apsum, npix = annulus.apsum(data)

    assert_allclose(apsum, npix)
    assert_allclose(npix, annulus.area, rtol=0, atol=1e-8)

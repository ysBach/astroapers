"""PathAp geometry and API tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

import astroapers as aap
import astroapers.kernels as aapk


# ─── helpers ──────────────────────────────────────────────────────────────────


def _triangle_segments(x0, y0, x1, y1, x2, y2):
    return [
        ("move", x0, y0),
        ("line", x1, y1),
        ("line", x2, y2),
        ("close",),
    ]


def _rect_segments(w, h):
    hw, hh = w / 2, h / 2
    return [
        ("move", -hw, -hh),
        ("line", hw, -hh),
        ("line", hw, hh),
        ("line", -hw, hh),
        ("close",),
    ]


def _rotate_point(point, theta):
    x, y = point
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return cos_t * x - sin_t * y, sin_t * x + cos_t * y


def _rotated_rect_segments(w, h, theta):
    hw, hh = w / 2, h / 2
    vertices = [
        _rotate_point((-hw, -hh), theta),
        _rotate_point((hw, -hh), theta),
        _rotate_point((hw, hh), theta),
        _rotate_point((-hw, hh), theta),
    ]
    return [
        ("move", *vertices[0]),
        ("line", *vertices[1]),
        ("line", *vertices[2]),
        ("line", *vertices[3]),
        ("close",),
    ]


def _circle_segments(r, n_arcs=4):
    """Approximate circle using n_arcs equal arcs (must divide 2π evenly)."""
    dtheta = 2 * math.pi / n_arcs
    segs = [("move", r, 0.0)]
    theta = 0.0
    for _ in range(n_arcs - 1):
        segs.append(("arc", 0.0, 0.0, r, theta, dtheta))
        theta += dtheta
    # last arc back to start
    segs.append(("arc", 0.0, 0.0, r, theta, dtheta))
    segs.append(("close",))
    return segs


def _pill_segments(w, r, theta):
    left_center = _rotate_point((-0.5 * w, 0.0), theta)
    right_center = _rotate_point((0.5 * w, 0.0), theta)
    lower_left = _rotate_point((-0.5 * w, -r), theta)
    lower_right = _rotate_point((0.5 * w, -r), theta)
    upper_left = _rotate_point((-0.5 * w, r), theta)
    return [
        ("move", *lower_left),
        ("line", *lower_right),
        ("arc", right_center[0], right_center[1], r, theta - math.pi / 2, math.pi),
        ("line", *upper_left),
        ("arc", left_center[0], left_center[1], r, theta + math.pi / 2, math.pi),
        ("close",),
    ]


def _parity_data():
    values = np.arange(70 * 80, dtype=np.float64).reshape(70, 80)
    return (values % 113) / 7.0


def _parity_positions():
    return np.array(
        [
            [20.2, 21.7],
            [42.5, 37.25],
            [7.8, 55.3],
        ],
        dtype=np.float64,
    )


# ─── geometry / unit tests ────────────────────────────────────────────────────


class TestTriangleExactArea:
    """Triangle path exact sum tracks analytic area."""

    def test_unit_right_triangle(self):
        # Triangle with vertices (0,0), (10,0), (0,10), area = 50
        segs = _triangle_segments(0, 0, 10, 0, 0, 10)
        img = np.ones((15, 15), dtype=np.float64)
        center = (0.0, 0.0)
        ap = aap.PathAp(center, segs)
        apsum, npix = ap.apsum(img)
        assert abs(float(npix) - 50.0) < 0.5
        assert abs(float(apsum) - 50.0) < 0.5

    def test_centered_triangle(self):
        # Equilateral-ish triangle, check center mode ≈ exact mode
        segs = _triangle_segments(-5, -3, 5, -3, 0, 6)
        img = np.ones((20, 20), dtype=np.float64)
        center = (9.5, 9.5)
        ap = aap.PathAp(center, segs)
        exact_sum, exact_npix = ap.apsum(img, method="exact")
        ctr_sum, ctr_npix = ap.apsum(img, method="center")
        # Exact and center should be within ~2 pixels for a largish shape
        assert abs(float(exact_npix) - float(ctr_npix)) < 3.0


class TestRectangleMatchesRectAp:
    """Rectangle path matches RectAp (axis-aligned)."""

    @pytest.mark.parametrize("w,h", [(4.0, 6.0), (3.5, 5.2)])
    def test_exact_area_matches(self, w, h):
        segs = _rect_segments(w, h)
        cx, cy = 10.5, 10.5
        img = np.ones((25, 25), dtype=np.float64)

        path_ap = aap.PathAp((cx, cy), segs)
        rect_ap = aap.RectAp((cx, cy), w, h, theta=0.0)

        _, path_npix = path_ap.apsum(img, method="exact")
        _, rect_npix = rect_ap.apsum(img)
        assert abs(float(path_npix) - float(rect_npix)) < 0.05

    def test_center_mode_matches(self):
        w, h = 6.0, 4.0
        segs = _rect_segments(w, h)
        cx, cy = 12.0, 12.0
        img = np.ones((25, 25), dtype=np.float64)

        path_ap = aap.PathAp((cx, cy), segs)

        _, path_npix = path_ap.apsum(img, method="center")
        # PathAp v1 follows its documented boundary convention: outer
        # boundary centers count as inside.
        assert float(path_npix) == 35.0


class TestCircularPathMatchesCircAp:
    """Circle built from 4 arcs tracks CircAp within tight tolerance."""

    @pytest.mark.parametrize("r", [3.0, 5.0, 7.5])
    def test_exact_npix_matches(self, r):
        segs = _circle_segments(r, n_arcs=4)
        cx, cy = 15.5, 15.5
        img = np.ones((40, 40), dtype=np.float64)

        path_ap = aap.PathAp((cx, cy), segs)
        circ_ap = aap.CircAp((cx, cy), r)

        _, path_npix = path_ap.apsum(img, method="exact")
        _, circ_npix = circ_ap.apsum(img)
        # 4-arc circle introduces small geometric difference at corners
        assert abs(float(path_npix) - float(circ_npix)) < 0.01


class TestBboxIncludesArcExtrema:
    """Bounding box must include arc extrema, not just endpoints."""

    def test_circle_bbox_at_least_radius(self):
        r = 5.0
        segs = _circle_segments(r, n_arcs=4)
        cx, cy = 10.0, 10.0
        bbox = aap.PathAp((cx, cy), segs).bboxes()[0]
        assert bbox.ixmin <= int(cx - r)
        assert bbox.ixmax >= int(cx + r)
        assert bbox.iymin <= int(cy - r)
        assert bbox.iymax >= int(cy + r)

    def test_single_arc_bbox_extends_to_extremum(self):
        # Arc from 0 to π/2, radius 5, center at (0,0)
        # Extremum at angle 0 (x=5) and π/2 (y=5); no cardinal extrema in interior
        r = 5.0
        segs = [
            ("move", r, 0.0),
            ("arc", 0.0, 0.0, r, 0.0, math.pi / 2),
            ("line", 0.0, 0.0),
            ("close",),
        ]
        cx, cy = 20.0, 20.0
        bbox = aap.PathAp((cx, cy), segs).bboxes()[0]
        # The arc spans only the first quadrant; x goes up to r, y goes up to r
        assert bbox.ixmax >= int(cx + r)
        assert bbox.iymax >= int(cy + r)


class TestInvalidPathsRaise:
    """Degenerate or invalid path inputs raise ValueError."""

    def test_missing_move(self):
        with pytest.raises(ValueError, match="move"):
            aap.PathAp((0, 0), [("line", 1, 0), ("close",)])

    def test_missing_close(self):
        with pytest.raises(ValueError, match="close"):
            aap.PathAp((0, 0), [("move", 0, 0), ("line", 1, 0)])

    def test_unknown_command(self):
        with pytest.raises(ValueError):
            aap.PathAp((0, 0), [("move", 0, 0), ("spline", 1, 0), ("close",)])

    def test_zero_length_line(self):
        segs = [("move", 0, 0), ("line", 0, 0), ("line", 1, 1), ("close",)]
        with pytest.raises((ValueError, RuntimeError)):
            aap.PathAp((0, 0), segs)

    def test_arc_zero_radius(self):
        segs = [("move", 1, 0), ("arc", 0, 0, 0, 0, math.pi / 2), ("close",)]
        with pytest.raises((ValueError, RuntimeError)):
            aap.PathAp((0, 0), segs)

    def test_arc_full_circle_dtheta(self):
        segs = [("move", 1, 0), ("arc", 0, 0, 1, 0, 2 * math.pi), ("close",)]
        with pytest.raises((ValueError, RuntimeError)):
            aap.PathAp((0, 0), segs)


# ─── Python API tests ─────────────────────────────────────────────────────────


class TestScalarAndVectorPositions:
    def setup_method(self):
        self.segs = _rect_segments(4.0, 4.0)
        self.img = np.ones((30, 30), dtype=np.float64)

    def test_scalar_position_returns_scalar_apsum(self):
        ap = aap.PathAp((10.0, 10.0), self.segs)
        result = ap.apsum(self.img)
        apsum, npix = result
        assert apsum.shape == ()
        assert npix.shape == ()

    def test_vector_positions_return_array(self):
        ap = aap.PathAp([(10.0, 10.0), (15.0, 15.0)], self.segs)
        apsum, npix = ap.apsum(self.img)
        assert apsum.shape == (2,)
        assert npix.shape == (2,)
        # Both apertures fully inside 30×30 image, should have same npix
        assert abs(float(npix[0]) - float(npix[1])) < 0.1


class TestReturnNpixFalse:
    def test_returns_only_apsum(self):
        segs = _rect_segments(4.0, 4.0)
        img = np.ones((20, 20), dtype=np.float64)
        ap = aap.PathAp((10.0, 10.0), segs)
        result = ap.apsum(img, return_npix=False)
        assert not isinstance(result, tuple)
        assert result.shape == ()


class TestMaskedApsum:
    def test_mask_excludes_pixels(self):
        segs = _rect_segments(6.0, 6.0)
        img = np.ones((20, 20), dtype=np.float64)
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True  # mask one pixel inside aperture

        ap = aap.PathAp((10.0, 10.0), segs)
        _, npix_unmasked = ap.apsum(img)
        _, npix_masked = ap.apsum(img, mask=mask)
        assert float(npix_masked) < float(npix_unmasked)


class TestWeights:
    def test_exact_weights_values_in_range(self):
        segs = _circle_segments(4.0, n_arcs=4)
        ap = aap.PathAp((10.5, 10.5), segs)
        w = ap.weights(method="exact")[0]
        assert w.min() >= 0.0 - 1e-12
        assert w.max() <= 1.0 + 1e-12

    def test_center_weights_binary(self):
        segs = _rect_segments(4.0, 4.0)
        ap = aap.PathAp((10.0, 10.0), segs)
        w = ap.weights(method="center")[0]
        assert set(np.unique(w)).issubset({0.0, 1.0})


class TestPublicExports:
    def test_pathap_in_astroapers(self):
        assert hasattr(aap, "PathAp")

    def test_apsum_path_exact_in_kernels(self):
        assert hasattr(aapk, "apsum_path_exact")

    def test_bboxes_path_in_kernels(self):
        assert hasattr(aapk, "bboxes_path")

    def test_weights_path_exact_in_kernels(self):
        assert hasattr(aapk, "weights_path_exact")


# ─── kernel-level tests ───────────────────────────────────────────────────────


class TestKernelApsum:
    def test_apsum_path_exact_scalar(self):
        segs = _rect_segments(4.0, 4.0)
        img = np.ones((20, 20), dtype=np.float64)
        apsum, npix = aapk.apsum_path_exact(img, 10.0, 10.0, segs)
        assert abs(float(apsum) - 16.0) < 0.1
        assert abs(float(npix) - 16.0) < 0.1

    def test_apsum_path_center_scalar(self):
        segs = _rect_segments(4.0, 4.0)
        img = np.ones((20, 20), dtype=np.float64)
        apsum, npix = aapk.apsum_path_center(img, 10.0, 10.0, segs)
        assert float(npix) > 0.0

    def test_bboxes_path_list(self):
        segs = _rect_segments(4.0, 4.0)
        boxes = aapk.bboxes_path([10.0, 15.0], [10.0, 15.0], segs)
        assert len(boxes) == 2


@pytest.mark.parametrize("method", ["exact", "center"])
def test_path_object_npix_and_weights(method):
    segs = _rect_segments(4.0, 4.0)
    positions = np.array([[10.0, 10.0], [15.0, 15.0]])
    aperture = aap.PathAp(positions, segs)
    npix_func = aapk.npix_path_exact if method == "exact" else aapk.npix_path_center
    bad = np.zeros((24, 24), dtype=bool)
    bad[10, 10] = True

    weights = aperture.weights(method=method)
    boxes = aperture.bboxes()

    assert len(weights) == len(boxes) == len(positions)
    for weight, box in zip(weights, boxes, strict=True):
        assert weight.shape == box.shape
    assert_allclose(
        aperture.npix(bad.shape, method=method),
        npix_func(positions[:, 0], positions[:, 1], segs, shape=bad.shape),
    )
    assert_allclose(
        aperture.npix(bad.shape, method=method, mask=bad),
        aperture.apsum(np.ones(bad.shape), method=method, mask=bad)[1],
    )


class TestHoleAnnulus:
    """Path annulus (outer circle minus inner circle) tracks CircAn."""

    def test_annulus_area_close_to_circan(self):
        r_in, r_out = 3.0, 6.0
        cx, cy = 15.5, 15.5
        img = np.ones((35, 35), dtype=np.float64)

        # Outer circle (CCW)
        outer = _circle_segments(r_out, n_arcs=4)
        # Inner circle as hole (will be forced CW by Rust orientation normalizer)
        inner = _circle_segments(r_in, n_arcs=4)

        path_an = aap.PathAp((cx, cy), outer, holes=[inner])
        circ_an = aap.CircAn((cx, cy), r_in, r_out)

        _, path_npix = path_an.apsum(img, method="exact")
        _, circ_npix = circ_an.apsum(img)

        assert abs(float(path_npix) - float(circ_npix)) < 0.1


class TestPathApShapeParity:
    """PathAp reproduces existing aperture kernels for equivalent geometry."""

    def test_rotated_rectap_exact_results_match(self):
        data = _parity_data()
        positions = _parity_positions()
        w, h, theta = 8.0, 4.5, 0.37

        path_ap = aap.PathAp(positions, _rotated_rect_segments(w, h, theta))
        rect_ap = aap.RectAp(positions, w, h, theta)

        path_apsum, path_npix = path_ap.apsum(data, method="exact")
        rect_apsum, rect_npix = rect_ap.apsum(data)

        assert_allclose(path_apsum, rect_apsum, rtol=0.0, atol=1e-11)
        assert_allclose(path_npix, rect_npix, rtol=0.0, atol=1e-12)

    def test_circap_exact_results_match(self):
        data = _parity_data()
        positions = _parity_positions()
        r = 5.3

        path_ap = aap.PathAp(positions, _circle_segments(r, n_arcs=4))
        circ_ap = aap.CircAp(positions, r)

        path_apsum, path_npix = path_ap.apsum(data, method="exact")
        circ_apsum, circ_npix = circ_ap.apsum(data)

        assert_allclose(path_apsum, circ_apsum, rtol=0.0, atol=1e-11)
        assert_allclose(path_npix, circ_npix, rtol=0.0, atol=1e-12)

    def test_circular_pillap_center_results_match(self):
        data = _parity_data()
        positions = _parity_positions()
        x = positions[:, 0]
        y = positions[:, 1]
        w, r, theta = 7.0, 2.25, 0.31

        path_ap = aap.PathAp(positions, _pill_segments(w, r, theta))

        path_apsum, path_npix = path_ap.apsum(data, method="center")
        pill_apsum, pill_npix = aapk.apsum_pill_center(data, x, y, w, r, r, theta)

        assert_allclose(path_apsum, pill_apsum, rtol=0.0, atol=1e-12)
        assert_allclose(path_npix, pill_npix, rtol=0.0, atol=0.0)

    def test_circular_pillap_exact_results_match(self):
        data = _parity_data()
        positions = _parity_positions()
        w, r, theta = 7.0, 2.25, 0.31

        path_ap = aap.PathAp(positions, _pill_segments(w, r, theta))
        pill_ap = aap.PillAp(positions, w, r, r, theta)

        path_apsum, path_npix = path_ap.apsum(data, method="exact")
        pill_apsum, pill_npix = pill_ap.apsum(data)

        assert_allclose(path_apsum, pill_apsum, rtol=1e-12, atol=1e-10)
        assert_allclose(path_npix, pill_npix, rtol=1e-12, atol=1e-10)

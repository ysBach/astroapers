"""Wedge aperture classes."""

from __future__ import annotations

import math

import numpy as np

from ._apertures import PixelAp, _shape_apsum_result
from ._containers import BoundingBox
from .kernels import (
    apsum_wedge_center,
    apsum_wedge_exact,
    bboxes_wedge,
    npix_wedge_center,
    npix_wedge_exact,
    weights_center,
    weights_exact,
    weights_wedge_center,
    weights_wedge_exact,
)
from ._utils import require_even_int_at_least, require_finite_float


class WedgeAp(PixelAp):
    """Annular wedge aperture in pixel coordinates.

    A wedge is bounded by inner and outer circular arcs, with straight sides
    connecting the corresponding arc endpoints. Angles are in radians, measured
    counterclockwise from the positive image x axis. ``dtheta_*`` values are
    full angular widths. If ``theta_out`` or ``dtheta_out`` is `None`, the
    corresponding inner value is used, giving the constant-width annular sector
    used by wedge-photometry workflows.

    Parameters
    ----------
    positions : tuple or array-like
        Aperture center or centers as ``(x, y)`` pixel coordinates. Multiple
        centers are accepted as an ``(N, 2)`` array-like object.
    r_in, r_out : float
        Inner and outer radii in pixels. ``r_out`` must be larger than
        ``r_in`` and ``r_in`` must be positive.
    theta_in, dtheta_in : float
        Center angle and full angular width at ``r_in``. ``dtheta_in`` must be
        in ``(0, 2*pi)``.
    theta_out, dtheta_out : float, optional
        Center angle and full angular width at ``r_out``. ``dtheta_out`` must
        be in ``(0, 2*pi)``. Missing values default to the corresponding inner
        value.
    plot_samples : int, optional
        Even number of samples used to draw the two circular arcs.
    validate : bool, optional
        If `False`, skip input validation and assume finite, ordered geometry
        and normalized positions.
    """

    def __init__(
        self,
        positions,
        r_in: float,
        r_out: float,
        theta_in: float,
        dtheta_in: float,
        theta_out: float | None = None,
        dtheta_out: float | None = None,
        plot_samples: int = 96,
        *,
        validate: bool = True,
    ):
        super().__init__(positions, validate=validate)
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.theta_in = float(theta_in)
        self.dtheta_in = float(dtheta_in)
        self.theta_out = self.theta_in if theta_out is None else float(theta_out)
        self.dtheta_out = self.dtheta_in if dtheta_out is None else float(dtheta_out)
        self.plot_samples = (
            require_even_int_at_least(plot_samples, "plot_samples", 4)
            if validate
            else int(plot_samples)
        )
        if validate:
            self._validate_geometry()

    def _validate_geometry(self) -> None:
        if not math.isfinite(self.r_in) or self.r_in <= 0.0:
            raise ValueError("r_in must be a positive finite scalar")
        if not math.isfinite(self.r_out) or self.r_out <= 0.0:
            raise ValueError("r_out must be a positive finite scalar")
        if self.r_in >= self.r_out:
            raise ValueError("r_in must be smaller than r_out")
        require_finite_float(self.theta_in, "theta_in")
        require_finite_float(self.theta_out, "theta_out")
        for name, value in (
            ("dtheta_in", self.dtheta_in),
            ("dtheta_out", self.dtheta_out),
        ):
            if not math.isfinite(value) or value <= 0.0 or value >= 2.0 * math.pi:
                raise ValueError(f"{name} must be a finite scalar in (0, 2*pi)")

    @property
    def area(self) -> float:
        """Analytic wedge area, before image clipping or bad-pixel masking."""
        bounds = _wedge_bounds(
            self.theta_in, self.dtheta_in, self.theta_out, self.dtheta_out
        )
        quad = np.array(
            [
                _polar_point(self.r_in, bounds[0]),
                _polar_point(self.r_out, bounds[2]),
                _polar_point(self.r_out, bounds[3]),
                _polar_point(self.r_in, bounds[1]),
            ],
            dtype=np.float64,
        )
        quad_area = _polygon_area(quad)
        outer_segment = (
            0.5 * self.r_out**2 * (self.dtheta_out - math.sin(self.dtheta_out))
        )
        inner_segment = 0.5 * self.r_in**2 * (self.dtheta_in - math.sin(self.dtheta_in))
        return quad_area + outer_segment - inner_segment

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        return bboxes_wedge(
            np.array([x]),
            np.array([y]),
            self.r_in,
            self.r_out,
            self.theta_in,
            self.dtheta_in,
            self.theta_out,
            self.dtheta_out,
            validate=self._validate,
        )[0]

    def bboxes(self) -> list[BoundingBox]:
        return bboxes_wedge(
            self._x,
            self._y,
            self.r_in,
            self.r_out,
            self.theta_in,
            self.dtheta_in,
            self.theta_out,
            self.dtheta_out,
            validate=self._validate,
        )

    def _weights_exact_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_exact(
            "weights_wedge_exact",
            bbox,
            (
                x,
                y,
                self.r_in,
                self.r_out,
                self.theta_in,
                self.dtheta_in,
                self.theta_out,
                self.dtheta_out,
            ),
        )

    def _weights_center_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_center(
            "weights_wedge_center",
            bbox,
            (
                x,
                y,
                self.r_in,
                self.r_out,
                self.theta_in,
                self.dtheta_in,
                self.theta_out,
                self.dtheta_out,
            ),
        )

    def _weights_with_method(self, method: str) -> list[np.ndarray]:
        method = self._weight_method(method)
        weights_func = (
            weights_wedge_exact if method == "exact" else weights_wedge_center
        )
        weights, _ = weights_func(
            self._x,
            self._y,
            self.r_in,
            self.r_out,
            self.theta_in,
            self.dtheta_in,
            self.theta_out,
            self.dtheta_out,
            validate=self._validate,
        )
        return weights

    def _apsum_with_method(
        self, data, mask=None, *, method: str, return_npix: bool = True
    ):
        method = self._weight_method(method)
        apsum_func = apsum_wedge_exact if method == "exact" else apsum_wedge_center
        result = apsum_func(
            data,
            self._x,
            self._y,
            self.r_in,
            self.r_out,
            self.theta_in,
            self.dtheta_in,
            self.theta_out,
            self.dtheta_out,
            mask=mask,
            return_npix=return_npix,
            validate=self._validate,
        )
        if not return_npix:
            return _shape_apsum_result(self, result)
        apsum, npix = result
        return _shape_apsum_result(self, apsum, npix)

    def _npix_with_method(self, shape: tuple[int, int], *, method: str, mask=None):
        method = self._weight_method(method)
        npix_func = npix_wedge_exact if method == "exact" else npix_wedge_center
        return _shape_apsum_result(
            self,
            npix_func(
                self._x,
                self._y,
                self.r_in,
                self.r_out,
                self.theta_in,
                self.dtheta_in,
                self.theta_out,
                self.dtheta_out,
                shape=shape,
                mask=mask,
                validate=self._validate,
            ),
        )

    def _weight_method(self, method: str) -> str:
        if method not in {"exact", "center"}:
            raise ValueError("method must be 'exact' or 'center'")
        return method

    def _patch_one(self, x: float, y: float, origin, **kwargs):
        import matplotlib.patches as mpatches

        plot_samples = require_even_int_at_least(
            kwargs.pop("plot_samples", self.plot_samples), "plot_samples", 4
        )
        kwargs.setdefault("fill", False)
        return mpatches.Polygon(
            _wedge_vertices(
                x,
                y,
                self.r_in,
                self.r_out,
                self.theta_in,
                self.dtheta_in,
                self.theta_out,
                self.dtheta_out,
                origin,
                plot_samples,
            ),
            closed=True,
            **kwargs,
        )


def _wedge_vertices(
    x: float,
    y: float,
    r_in: float,
    r_out: float,
    theta_in: float,
    dtheta_in: float,
    theta_out: float,
    dtheta_out: float,
    origin,
    plot_samples: int,
) -> np.ndarray:
    half = max(2, plot_samples // 2)
    right_in, left_in, right_out, left_out = _wedge_bounds(
        theta_in, dtheta_in, theta_out, dtheta_out
    )

    outer_theta = np.linspace(right_out, left_out, half)
    inner_theta = np.linspace(left_in, right_in, half)
    radii = np.concatenate([np.full(half, r_out), np.full(half, r_in)])
    angles = np.concatenate([outer_theta, inner_theta])
    pts = np.column_stack((x + radii * np.cos(angles), y + radii * np.sin(angles)))
    pts[:, 0] -= origin[0]
    pts[:, 1] -= origin[1]
    return pts


def _wedge_bounds(
    theta_in: float, dtheta_in: float, theta_out: float, dtheta_out: float
) -> tuple[float, float, float, float]:
    right_in = theta_in - 0.5 * dtheta_in
    left_in = right_in + dtheta_in
    right_out = right_in + _angular_signed_delta(right_in, theta_out - 0.5 * dtheta_out)
    left_out = right_out + dtheta_out
    return right_in, left_in, right_out, left_out


def _angular_signed_delta(reference: float, angle: float) -> float:
    return (angle - reference + math.pi) % (2.0 * math.pi) - math.pi


def _polar_point(r: float, theta: float) -> tuple[float, float]:
    return r * math.cos(theta), r * math.sin(theta)


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

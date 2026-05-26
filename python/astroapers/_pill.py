"""Composite pill aperture classes."""

from __future__ import annotations

import math

import numpy as np

from ._apertures import EllipAp, PixelAp, RectAp, _shape_apsum_result
from ._containers import BoundingBox
from .kernels import (
    apsum_pill_ann_center,
    apsum_pill_ann_exact,
    apsum_pill_center,
    apsum_pill_exact,
    bboxes_pill,
    bboxes_pill_ann,
    npix_pill_ann_center,
    npix_pill_ann_exact,
    npix_pill_center,
    npix_pill_exact,
    weights_pill_ann_center,
    weights_pill_ann_exact,
    weights_pill_center,
    weights_pill_exact,
    weights_center,
    weights_exact,
)
from ._utils import (
    compound_annulus_path,
    pill_vertices,
    require_even_int_at_least,
    require_finite_float,
    require_positive_float,
    resolve_theta_pair,
)


class PillAp(PixelAp):
    """Capsule-like aperture composed from one rectangle and two ellipses.

    The pill aperture has a central rectangle of length ``w`` and height
    ``2 * b`` with elliptical caps of semiaxes ``a`` and ``b`` centered on the
    rectangle ends. Its masks use fused Rust kernels that preserve the package's
    composite rectangle-plus-cap semantics.

    Parameters
    ----------
    positions : tuple or array-like
        Aperture center or centers as ``(x, y)`` pixel coordinates. Multiple
        centers are accepted as an ``(N, 2)`` array-like object.
    w : float
        Length of the central rectangle in pixels, measured along the pill
        major axis between the two cap centers.
    a : float
        Semimajor axis of each elliptical cap in pixels.
    b : float
        Semiminor axis of each elliptical cap in pixels. The central rectangle
        height is ``2 * b``.
    theta : float, optional
        Rotation angle in radians, measured counterclockwise from the positive
        x axis.
    plot_samples : int, optional
        Even number of samples used to draw the two elliptical caps when
        creating Matplotlib patches.
    validate : bool, optional
        If `False`, skip input validation and assume finite, positive geometry
        and normalized positions.

    Attributes
    ----------
    area : float
        Analytic geometric area, ``w * 2 * b + pi * a * b``. Image clipping and
        bad-pixel masks are accounted for by ``npix`` results, not by this
        property.
    """

    def __init__(
        self,
        positions,
        w: float,
        a: float,
        b: float,
        theta: float = 0.0,
        plot_samples: int = 96,
        *,
        validate: bool = True,
    ):
        super().__init__(positions, validate=validate)
        self.w = require_positive_float(w, "w") if validate else float(w)
        self.a = require_positive_float(a, "a") if validate else float(a)
        self.b = require_positive_float(b, "b") if validate else float(b)
        self.theta = require_finite_float(theta, "theta") if validate else float(theta)
        self.plot_samples = (
            require_even_int_at_least(plot_samples, "plot_samples", 4)
            if validate
            else int(plot_samples)
        )

    @property
    def area(self) -> float:
        """Analytic pill area, before image clipping or bad-pixel masking."""
        return self.w * (2.0 * self.b) + math.pi * self.a * self.b

    def _components(self, x: float, y: float):
        dx = 0.5 * self.w * math.cos(self.theta)
        dy = 0.5 * self.w * math.sin(self.theta)
        rect = RectAp((x, y), self.w, 2.0 * self.b, self.theta, validate=False)
        left = EllipAp((x - dx, y - dy), self.a, self.b, self.theta, validate=False)
        right = EllipAp((x + dx, y + dy), self.a, self.b, self.theta, validate=False)
        return rect, left, right

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        return bboxes_pill(
            np.array([x]),
            np.array([y]),
            self.w,
            self.a,
            self.b,
            self.theta,
            validate=self._validate,
        )[0]

    def bboxes(self) -> list[BoundingBox]:
        return bboxes_pill(
            self._x,
            self._y,
            self.w,
            self.a,
            self.b,
            self.theta,
            validate=self._validate,
        )

    def _weights_exact_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_exact(
            "weights_pill_exact", bbox, (x, y, self.w, self.a, self.b, self.theta)
        )

    def _weights_center_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_center(
            "weights_pill_center", bbox, (x, y, self.w, self.a, self.b, self.theta)
        )

    def _weights_with_method(self, method: str) -> list[np.ndarray]:
        method = self._weight_method(method)
        weights_func = weights_pill_exact if method == "exact" else weights_pill_center
        weights, _ = weights_func(
            self._x,
            self._y,
            self.w,
            self.a,
            self.b,
            self.theta,
            validate=self._validate,
        )
        return weights

    def _apsum_with_method(
        self, data, mask=None, *, method: str, return_npix: bool = True
    ):
        """Return pill-aperture sums, and npix by default.

        Parameters
        ----------
        data : array_like
            Two-dimensional image.
        mask : array_like of bool, optional
            Boolean image mask with the same shape as ``data``. ``True`` pixels
            are excluded from both ``apsum`` and ``npix``.
        return_npix : bool, optional
            If `True`, return ``(apsum, npix)``. If `False`, return only
            ``apsum``.
        """
        method = self._weight_method(method)
        apsum_func = apsum_pill_exact if method == "exact" else apsum_pill_center
        result = apsum_func(
            data,
            self._x,
            self._y,
            self.w,
            self.a,
            self.b,
            self.theta,
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
        npix_func = npix_pill_exact if method == "exact" else npix_pill_center
        return _shape_apsum_result(
            self,
            npix_func(
                self._x,
                self._y,
                self.w,
                self.a,
                self.b,
                self.theta,
                shape=shape,
                mask=mask,
                validate=self._validate,
            ),
        )

    def _patch_one(self, x: float, y: float, origin, **kwargs):
        import matplotlib.patches as mpatches
        import matplotlib.path as mpath

        plot_samples = require_even_int_at_least(
            kwargs.pop("plot_samples", self.plot_samples), "plot_samples", 4
        )
        kwargs.setdefault("fill", False)
        return mpatches.PathPatch(
            mpath.Path(
                pill_vertices(
                    x, y, self.w, self.a, self.b, self.theta, origin, plot_samples
                ),
                closed=True,
            ),
            **kwargs,
        )


class PillAn(PixelAp):
    """Capsule-like annulus with explicit inner and outer pills.

    The outer boundary is a pill with rectangle length ``w_out`` and cap
    semiaxes ``a_out`` and ``b_out``. The inner boundary uses explicit
    ``w_in``, ``a_in``, and ``b_in`` values. Its masks use fused Rust kernels
    that subtract the inner composite pill from the outer composite pill.
    If ``theta_out`` is `None`, it defaults to ``theta_in``.

    Parameters
    ----------
    positions : tuple or array-like
        Annulus center or centers as ``(x, y)`` pixel coordinates. Multiple
        centers are accepted as an ``(N, 2)`` array-like object.
    w_in : float
        Inner central-rectangle length in pixels.
    a_in : float
        Inner cap semimajor axis in pixels.
    b_in : float
        Inner cap semiminor axis in pixels.
    w_out : float
        Outer central-rectangle length in pixels. Must be larger than
        ``w_in``.
    a_out : float
        Outer cap semimajor axis in pixels. Must be larger than ``a_in``.
    b_out : float
        Outer cap semiminor axis in pixels. Must be larger than ``b_in``.
    theta_in : float, optional
        Inner pill rotation angle in radians, measured counterclockwise from
        the positive x axis.
    theta_out : float, optional
        Outer pill rotation angle in radians, measured counterclockwise from
        the positive x axis. If `None`, it defaults to ``theta_in``.
    plot_samples : int, optional
        Even number of samples used to draw each pill boundary when creating
        Matplotlib patches.
    validate : bool, optional
        If `False`, skip input validation and assume finite, positive geometry,
        ordered inner/outer dimensions, and normalized positions.

    Attributes
    ----------
    area : float
        Analytic geometric annulus area. Image clipping and bad-pixel masks are
        accounted for by ``npix`` results, not by this property.
    """

    def __init__(
        self,
        positions,
        w_in: float,
        a_in: float,
        b_in: float,
        w_out: float,
        a_out: float,
        b_out: float,
        *,
        theta_in: float = 0.0,
        theta_out: float | None = None,
        plot_samples: int = 96,
        validate: bool = True,
    ):
        super().__init__(positions, validate=validate)
        self.w_in = require_positive_float(w_in, "w_in") if validate else float(w_in)
        self.a_in = require_positive_float(a_in, "a_in") if validate else float(a_in)
        self.b_in = require_positive_float(b_in, "b_in") if validate else float(b_in)
        self.w_out = (
            require_positive_float(w_out, "w_out") if validate else float(w_out)
        )
        self.a_out = (
            require_positive_float(a_out, "a_out") if validate else float(a_out)
        )
        self.b_out = (
            require_positive_float(b_out, "b_out") if validate else float(b_out)
        )
        self.theta_in, self.theta_out = resolve_theta_pair(
            theta_in, theta_out, validate=validate
        )
        self.plot_samples = (
            require_even_int_at_least(plot_samples, "plot_samples", 4)
            if validate
            else int(plot_samples)
        )
        if validate and (
            self.w_in >= self.w_out
            or self.a_in >= self.a_out
            or self.b_in >= self.b_out
        ):
            raise ValueError(
                "inner pill dimensions must be smaller than outer dimensions"
            )

    @property
    def area(self) -> float:
        """Analytic pill-annulus area, before clipping or bad-pixel masking."""
        return (
            self.w_out * (2.0 * self.b_out)
            - self.w_in * (2.0 * self.b_in)
            + math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)
        )

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        return bboxes_pill_ann(
            np.array([x]),
            np.array([y]),
            self.w_in,
            self.a_in,
            self.b_in,
            self.w_out,
            self.a_out,
            self.b_out,
            self.theta_in,
            self.theta_out,
            validate=self._validate,
        )[0]

    def bboxes(self) -> list[BoundingBox]:
        return bboxes_pill_ann(
            self._x,
            self._y,
            self.w_in,
            self.a_in,
            self.b_in,
            self.w_out,
            self.a_out,
            self.b_out,
            self.theta_in,
            self.theta_out,
            validate=self._validate,
        )

    def _weights_exact_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_exact(
            "weights_pill_ann_exact",
            bbox,
            (
                x,
                y,
                self.w_in,
                self.a_in,
                self.b_in,
                self.w_out,
                self.a_out,
                self.b_out,
                self.theta_in,
                self.theta_out,
            ),
        )

    def _weights_center_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_center(
            "weights_pill_ann_center",
            bbox,
            (
                x,
                y,
                self.w_in,
                self.a_in,
                self.b_in,
                self.w_out,
                self.a_out,
                self.b_out,
                self.theta_in,
                self.theta_out,
            ),
        )

    def _weights_with_method(self, method: str) -> list[np.ndarray]:
        method = self._weight_method(method)
        weights_func = (
            weights_pill_ann_exact if method == "exact" else weights_pill_ann_center
        )
        weights, _ = weights_func(
            self._x,
            self._y,
            self.w_in,
            self.a_in,
            self.b_in,
            self.w_out,
            self.a_out,
            self.b_out,
            self.theta_in,
            self.theta_out,
            validate=self._validate,
        )
        return weights

    def _apsum_with_method(
        self, data, mask=None, *, method: str, return_npix: bool = True
    ):
        """Return pill-annulus sums, and npix by default.

        Parameters
        ----------
        data : array_like
            Two-dimensional image.
        mask : array_like of bool, optional
            Boolean image mask with the same shape as ``data``. ``True`` pixels
            are excluded from both ``apsum`` and ``npix``.
        return_npix : bool, optional
            If `True`, return ``(apsum, npix)``. If `False`, return only
            ``apsum``.
        """
        method = self._weight_method(method)
        apsum_func = (
            apsum_pill_ann_exact if method == "exact" else apsum_pill_ann_center
        )
        result = apsum_func(
            data,
            self._x,
            self._y,
            self.w_in,
            self.a_in,
            self.b_in,
            self.w_out,
            self.a_out,
            self.b_out,
            self.theta_in,
            self.theta_out,
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
        npix_func = npix_pill_ann_exact if method == "exact" else npix_pill_ann_center
        return _shape_apsum_result(
            self,
            npix_func(
                self._x,
                self._y,
                self.w_in,
                self.a_in,
                self.b_in,
                self.w_out,
                self.a_out,
                self.b_out,
                self.theta_in,
                self.theta_out,
                shape=shape,
                mask=mask,
                validate=self._validate,
            ),
        )

    def _patch_one(self, x: float, y: float, origin, **kwargs):
        import matplotlib.patches as mpatches
        import matplotlib.path as mpath

        plot_samples = require_even_int_at_least(
            kwargs.pop("plot_samples", self.plot_samples), "plot_samples", 4
        )
        kwargs.setdefault("fill", False)
        outer = mpath.Path(
            pill_vertices(
                x,
                y,
                self.w_out,
                self.a_out,
                self.b_out,
                self.theta_out,
                origin,
                plot_samples,
            ),
            closed=True,
        )
        inner = mpath.Path(
            pill_vertices(
                x,
                y,
                self.w_in,
                self.a_in,
                self.b_in,
                self.theta_in,
                origin,
                plot_samples,
            ),
            closed=True,
        )
        return mpatches.PathPatch(compound_annulus_path(outer, inner), **kwargs)

"""Aperture geometry classes.

This private module contains the object-oriented convenience layer. The
performance-critical bulk aperture-sum functions live in the public
:mod:`astroapers.kernels` namespace and call Rust directly; these classes
intentionally stay lightweight wrappers around geometry, masks, plotting, and
small-N convenience methods.
"""

from __future__ import annotations

import math

import numpy as np

from ._containers import BoundingBox
from .kernels import (
    apsum_circ_ann_center,
    apsum_circ_ann_exact,
    apsum_circ_center,
    apsum_circ_exact,
    apsum_ellip_ann_center,
    apsum_ellip_ann_exact,
    apsum_ellip_center,
    apsum_ellip_exact,
    apsum_rect_ann_center,
    apsum_rect_ann_exact,
    apsum_rect_center,
    apsum_rect_exact,
    bboxes_circ,
    bboxes_ellip,
    bboxes_rect,
    npix_circ_ann_center,
    npix_circ_ann_exact,
    npix_circ_center,
    npix_circ_exact,
    npix_ellip_ann_center,
    npix_ellip_ann_exact,
    npix_ellip_center,
    npix_ellip_exact,
    npix_rect_ann_center,
    npix_rect_ann_exact,
    npix_rect_center,
    npix_rect_exact,
    weights_circ_ann_center,
    weights_circ_ann_exact,
    weights_circ_center,
    weights_circ_exact,
    weights_ellip_ann_center,
    weights_ellip_ann_exact,
    weights_ellip_center,
    weights_ellip_exact,
    weights_rect_ann_center,
    weights_rect_ann_exact,
    weights_rect_center,
    weights_rect_exact,
    weights_center,
    weights_exact,
)
from ._utils import (
    bbox_from_extent,
    compound_annulus_path,
    normalize_positions,
    require_finite_float,
    require_positive_float,
    resolve_theta_pair,
)


class PixelAp:
    """Base class for pixel-coordinate apertures and annuli.

    Classes derived from `PixelAp` are lightweight geometry objects. The
    public method and attribute names are short but familiar for aperture
    work (`positions`, `area`, `weights()`, `apsum()`, `plot()`, and
    `to_patches()`), while heavy aperture summation routes through Rust-backed kernels.

    The intended performance split is:

    - use shape-specific `apsum_*_exact()` functions for fastest bulk aperture summation;
    - use `weights()` and `bboxes()` when bbox-tight weights are useful;
    - use `BoundingBox` methods when you already have raw weights and boxes.

    Constructors accept ``validate=False`` as an expert-only path for high-N
    workflows. It assumes positions and geometry parameters are already valid,
    skips constructor positivity/finite/order checks, and avoids copying an
    already-2D ndarray of positions. Runtime option strings such as
    ``method="exact"`` are still validated.
    """

    def __init__(self, positions, *, validate: bool = True):
        self._validate = bool(validate)
        if validate:
            self.positions, self.isscalar = normalize_positions(positions)
        else:
            arr = (
                positions
                if isinstance(positions, np.ndarray)
                else np.asarray(positions)
            )
            self.positions = arr.reshape(1, 2) if arr.ndim == 1 else arr
            self.isscalar = arr.ndim == 1
        self._x = np.ascontiguousarray(self.positions[:, 0], dtype=np.float64)
        self._y = np.ascontiguousarray(self.positions[:, 1], dtype=np.float64)

    @property
    def area(self) -> float:
        # Implementations return the analytic geometric area, not an
        # image-clipped area.
        raise NotImplementedError

    def bboxes(self) -> list[BoundingBox]:
        """Return bbox-tight bounding boxes for all aperture positions.

        The boxes are computed on demand. This method returns a list for both
        scalar and vector apertures so callers can handle both cases uniformly.
        It does not compute aperture weights; use :meth:`weights` for the
        corresponding bbox-tight weight arrays.
        """
        return [self._bbox_one(float(x), float(y)) for x, y in self.positions]

    def weights(self, method: str = "exact") -> list[np.ndarray]:
        """Return bbox-tight aperture weights for all positions.

        Parameters
        ----------
        method : {"exact", "center"}, optional
            Weighting method. ``"exact"`` returns fractional pixel overlap
            weights; ``"center"`` returns center-selected binary weights.

        Returns
        -------
        list of ndarray
            One bbox-tight weight array per aperture position. The returned
            list is aligned by index with :meth:`bboxes`.
        """
        method = self._weight_method(method)
        arrays = []
        for (x, y), bbox in zip(self.positions, self.bboxes(), strict=True):
            x = float(x)
            y = float(y)
            if method == "center":
                arrays.append(self._center_weights_one(x, y, bbox))
            else:
                arrays.append(self._exact_weights_one(x, y, bbox))
        return arrays

    def npix(self, shape: tuple[int, int], *, method: str = "exact", mask=None):
        """Return in-frame effective pixel counts.

        Parameters
        ----------
        shape : tuple[int, int]
            Image shape as ``(ny, nx)``.
        method : {"exact", "center"}, optional
            Weighting method.
        mask : array_like of bool, optional
            Boolean image mask with shape ``shape``. ``True`` pixels are
            excluded from the returned effective pixel count.
        """
        method = self._weight_method(method)
        weights = self.weights(method)
        boxes = self.bboxes()
        npix = np.empty(len(weights), dtype=np.float64)
        for idx, (weight, bbox) in enumerate(zip(weights, boxes, strict=True)):
            npix[idx] = bbox.npix(weight, shape, mask=mask, validate=self._validate)
        return _shape_apsum_result(self, npix)

    def apsum(
        self, data, mask=None, *, method: str = "exact", return_npix: bool = True
    ):
        """Return aperture apsum sums, and in-frame npix by default.

        This is the main object-level aperture-sum method. It uses fractional
        overlap masks. For maximum throughput with many circle, ellipse, or
        rectangle apertures, prefer the shape-specific bulk functions such as
        `apsum_circ_exact()`, `apsum_ellip_exact()`, and `apsum_rect_exact()`.
        """
        method = self._weight_method(method)
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("data must be a 2-D array")
        weights = self.weights(method)
        boxes = self.bboxes()
        apsum = np.empty(len(weights), dtype=np.float64)
        if return_npix:
            npix = np.empty(len(weights), dtype=np.float64)
            for idx, (weight, bbox) in enumerate(zip(weights, boxes, strict=True)):
                apsum[idx], npix[idx] = bbox.apsum(
                    weight, arr, mask=mask, return_npix=True, validate=self._validate
                )
            return _shape_apsum_result(self, apsum, npix)
        for idx, (weight, bbox) in enumerate(zip(weights, boxes, strict=True)):
            apsum[idx] = bbox.apsum(
                weight, arr, mask=mask, return_npix=False, validate=self._validate
            )
        return _shape_apsum_result(self, apsum)

    def weighted_values(self, data, method: str = "center"):
        """Return 1-D data values selected by the aperture.

        ``method="center"`` returns unweighted data values whose pixel centers
        are inside the aperture. ``method="exact"`` returns weighted values
        from ``data * fractional_mask`` for positive-weight mask pixels.
        """
        method = self._weight_method(method)
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("data must be a 2-D array")
        values = [
            self._weighted_values_one(arr, float(x), float(y), method)
            for x, y in self.positions
        ]
        return values[0] if self.isscalar else values

    def _weight_method(self, method: str) -> str:
        if method not in {"exact", "center"}:
            raise ValueError("method must be 'exact' or 'center'")
        return method

    def to_patches(self, origin=(0, 0), **kwargs):
        """Return Matplotlib patch objects without adding them to an axes.

        Returning the patch object instead of drawing directly lets callers
        customize artists before adding them to an axes.

        Parameters
        ----------
        origin : tuple[float, float], optional
            Coordinate offset subtracted from aperture positions before
            constructing patches. This is useful when plotting on a cutout
            image whose pixel origin differs from the parent image.
        **kwargs
            Passed directly to the Matplotlib patch constructor. By default
            patches are outlines (`fill=False`, `facecolor="none"`) with the
            standard Matplotlib first-cycle edge color.

        Returns
        -------
        matplotlib.patches.Patch or list[matplotlib.patches.Patch]
            A single patch for scalar-position apertures, otherwise one patch
            per aperture position.
        """
        kwargs.setdefault("fill", False)
        if "color" not in kwargs:
            kwargs.setdefault("facecolor", "none")
            kwargs.setdefault("edgecolor", "C0")
        patches = [
            self._patch_one(float(x), float(y), origin, **kwargs)
            for x, y in self.positions
        ]
        return patches[0] if self.isscalar else patches

    def plot(self, ax=None, origin=(0, 0), **kwargs):
        """Add aperture outlines to a Matplotlib axes and return the patches.

        This convenience wrapper delegates to :meth:`to_patches`, imports
        Matplotlib lazily, and has no effect on the Rust summation kernels.
        """
        from .plotting import plot_apertures

        return plot_apertures(self, ax=ax, origin=origin, **kwargs)

    def _weighted_values_one(
        self, data: np.ndarray, x: float, y: float, method: str
    ) -> np.ndarray:
        bbox = self._bbox_one(x, y)
        overlap = bbox.overlap_slices(data.shape)
        if overlap is None:
            return np.array([], dtype=data.dtype)
        data_slices, mask_slices = overlap
        if method == "center":
            selected = self._center_weights_one(x, y, bbox)[mask_slices] > 0.0
            return data[data_slices][selected].ravel()
        weights = self._exact_weights_one(x, y, bbox)[mask_slices]
        weighted = data[data_slices] * weights
        return weighted[weights > 0.0].ravel()

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        """Return one scalar bounding box.

        Fundamental vector aperture paths override `bboxes()` and call Rust.
        `_bbox_one()` stays scalar and Python-side for object convenience,
        plotting, weighted value extraction, and composite shapes where a Rust
        round trip per aperture would dominate this O(1) geometry.
        """
        raise NotImplementedError

    def _exact_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        raise NotImplementedError

    def _center_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        raise NotImplementedError

    def _patch_one(self, x: float, y: float, origin, **kwargs):
        raise NotImplementedError


def _shape_apsum_result(aperture: PixelAp, apsum, npix=None):
    # Rust aperture-sum kernels accumulate and return f64 for every image dtype.
    # This helper only preserves scalar-vs-vector object shape.
    apsum = np.asarray(apsum, dtype=np.float64)
    if npix is None:
        return apsum.reshape(()) if aperture.isscalar else apsum
    npix = np.asarray(npix, dtype=np.float64)
    if aperture.isscalar:
        return apsum.reshape(()), npix.reshape(())
    return apsum, npix


class CircAp(PixelAp):
    """Circular aperture in pixel coordinates.

    Parameters
    ----------
    positions : tuple or array-like
        Aperture center or centers as ``(x, y)`` pixel coordinates. Multiple
        centers are accepted as an ``(N, 2)`` array-like object.
    r : float
        Aperture radius in pixels.
    validate : bool, optional
        If `False`, skip input validation and assume finite, positive geometry
        and normalized positions. This is intended for expert high-throughput
        code paths.
    """

    def __init__(self, positions, r: float, *, validate: bool = True):
        super().__init__(positions, validate=validate)
        self.r = require_positive_float(r, "r") if validate else float(r)

    @property
    def area(self) -> float:
        return math.pi * self.r * self.r

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        return bbox_from_extent(x, y, self.r, self.r)

    def bboxes(self) -> list[BoundingBox]:
        """Override the default python for-loop with a Rust-accelerated batch function."""
        return bboxes_circ(
            self._x,
            self._y,
            self.r,
            validate=self._validate,
        )

    def weights(self, method: str = "exact") -> list[np.ndarray]:
        method = self._weight_method(method)
        weights_func = weights_circ_exact if method == "exact" else weights_circ_center
        weights, _ = weights_func(
            self._x,
            self._y,
            self.r,
            validate=self._validate,
        )
        return weights

    def apsum(
        self, data, mask=None, *, method: str = "exact", return_npix: bool = True
    ):
        method = self._weight_method(method)
        apsum_func = apsum_circ_exact if method == "exact" else apsum_circ_center
        result = apsum_func(
            data,
            self._x,
            self._y,
            self.r,
            mask=mask,
            return_npix=return_npix,
            validate=self._validate,
        )
        if not return_npix:
            return _shape_apsum_result(self, result)
        apsum, npix = result
        return _shape_apsum_result(self, apsum, npix)

    def npix(self, shape: tuple[int, int], *, method: str = "exact", mask=None):
        method = self._weight_method(method)
        npix_func = npix_circ_exact if method == "exact" else npix_circ_center
        return _shape_apsum_result(
            self,
            npix_func(
                self._x,
                self._y,
                self.r,
                shape=shape,
                mask=mask,
                validate=self._validate,
            ),
        )

    def _exact_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_exact("weights_circ_exact", bbox, (x, y, self.r))

    def _center_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_center("weights_circ_center", bbox, (x, y, self.r))

    def _patch_one(self, x: float, y: float, origin, **kwargs):
        import matplotlib.patches as mpatches

        kwargs.setdefault("fill", False)
        return mpatches.Circle((x - origin[0], y - origin[1]), self.r, **kwargs)


class EllipAp(PixelAp):
    """Elliptical aperture in pixel coordinates.

    Parameters
    ----------
    positions : tuple or array-like
        Aperture center or centers as ``(x, y)`` pixel coordinates. Multiple
        centers are accepted as an ``(N, 2)`` array-like object.
    a, b : float
        Semimajor and semiminor axis lengths in pixels.
    theta : float, optional
        Rotation angle in radians, measured counterclockwise from the positive
        x axis.
    validate : bool, optional
        If `False`, skip input validation and assume finite, positive geometry
        and normalized positions.
    """

    def __init__(
        self,
        positions,
        a: float,
        b: float,
        theta: float = 0.0,
        *,
        validate: bool = True,
    ):
        super().__init__(positions, validate=validate)
        self.a = require_positive_float(a, "a") if validate else float(a)
        self.b = require_positive_float(b, "b") if validate else float(b)
        self.theta = require_finite_float(theta, "theta") if validate else float(theta)

    @property
    def area(self) -> float:
        return math.pi * self.a * self.b

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        dx = math.sqrt((self.a * cos_t) ** 2 + (self.b * sin_t) ** 2)
        dy = math.sqrt((self.a * sin_t) ** 2 + (self.b * cos_t) ** 2)
        return bbox_from_extent(x, y, dx, dy)

    def bboxes(self) -> list[BoundingBox]:
        """Override the default python for-loop with a Rust-accelerated batch function."""
        return bboxes_ellip(
            self._x,
            self._y,
            self.a,
            self.b,
            self.theta,
            validate=self._validate,
        )

    def _exact_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_exact(
            "weights_ellip_exact", bbox, (x, y, self.a, self.b, self.theta)
        )

    def _center_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_center(
            "weights_ellip_center", bbox, (x, y, self.a, self.b, self.theta)
        )

    def weights(self, method: str = "exact") -> list[np.ndarray]:
        method = self._weight_method(method)
        weights_func = (
            weights_ellip_exact if method == "exact" else weights_ellip_center
        )
        weights, _ = weights_func(
            self._x,
            self._y,
            self.a,
            self.b,
            self.theta,
            validate=self._validate,
        )
        return weights

    def apsum(
        self, data, mask=None, *, method: str = "exact", return_npix: bool = True
    ):
        method = self._weight_method(method)
        apsum_func = apsum_ellip_exact if method == "exact" else apsum_ellip_center
        result = apsum_func(
            data,
            self._x,
            self._y,
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

    def npix(self, shape: tuple[int, int], *, method: str = "exact", mask=None):
        method = self._weight_method(method)
        npix_func = npix_ellip_exact if method == "exact" else npix_ellip_center
        return _shape_apsum_result(
            self,
            npix_func(
                self._x,
                self._y,
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

        kwargs.setdefault("fill", False)
        return mpatches.Ellipse(
            (x - origin[0], y - origin[1]),
            2.0 * self.a,
            2.0 * self.b,
            angle=math.degrees(self.theta),
            **kwargs,
        )


class RectAp(PixelAp):
    """Rotated rectangular aperture in pixel coordinates.

    Parameters
    ----------
    positions : tuple or array-like
        Aperture center or centers as ``(x, y)`` pixel coordinates. Multiple
        centers are accepted as an ``(N, 2)`` array-like object.
    w, h : float
        Full rectangle width and height in pixels. ``w`` is measured along the
        rectangle's local x axis, and ``h`` along its local y axis. At
        ``theta=0``, the width axis is aligned with the image x axis and the
        height axis is aligned with the image y axis.
    theta : float, optional
        Rotation angle in radians, measured counterclockwise from the positive
        image x axis to the rectangle's local width axis.
    validate : bool, optional
        If `False`, skip input validation and assume finite, positive geometry
        and normalized positions.
    """

    def __init__(
        self,
        positions,
        w: float,
        h: float,
        theta: float = 0.0,
        *,
        validate: bool = True,
    ):
        super().__init__(positions, validate=validate)
        self.w = require_positive_float(w, "w") if validate else float(w)
        self.h = require_positive_float(h, "h") if validate else float(h)
        self.theta = require_finite_float(theta, "theta") if validate else float(theta)

    @property
    def area(self) -> float:
        return self.w * self.h

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        dx = abs(0.5 * self.w * cos_t) + abs(0.5 * self.h * sin_t)
        dy = abs(0.5 * self.w * sin_t) + abs(0.5 * self.h * cos_t)
        return bbox_from_extent(x, y, dx, dy)

    def bboxes(self) -> list[BoundingBox]:
        """Override the default python for-loop with a Rust-accelerated batch function."""
        return bboxes_rect(
            self._x,
            self._y,
            self.w,
            self.h,
            self.theta,
            validate=self._validate,
        )

    def _exact_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_exact(
            "weights_rect_exact", bbox, (x, y, self.w, self.h, self.theta)
        )

    def _center_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_center(
            "weights_rect_center", bbox, (x, y, self.w, self.h, self.theta)
        )

    def weights(self, method: str = "exact") -> list[np.ndarray]:
        method = self._weight_method(method)
        weights_func = weights_rect_exact if method == "exact" else weights_rect_center
        weights, _ = weights_func(
            self._x,
            self._y,
            self.w,
            self.h,
            self.theta,
            validate=self._validate,
        )
        return weights

    def apsum(
        self, data, mask=None, *, method: str = "exact", return_npix: bool = True
    ):
        method = self._weight_method(method)
        apsum_func = apsum_rect_exact if method == "exact" else apsum_rect_center
        result = apsum_func(
            data,
            self._x,
            self._y,
            self.w,
            self.h,
            self.theta,
            mask=mask,
            return_npix=return_npix,
            validate=self._validate,
        )
        if not return_npix:
            return _shape_apsum_result(self, result)
        apsum, npix = result
        return _shape_apsum_result(self, apsum, npix)

    def npix(self, shape: tuple[int, int], *, method: str = "exact", mask=None):
        method = self._weight_method(method)
        npix_func = npix_rect_exact if method == "exact" else npix_rect_center
        return _shape_apsum_result(
            self,
            npix_func(
                self._x,
                self._y,
                self.w,
                self.h,
                self.theta,
                shape=shape,
                mask=mask,
                validate=self._validate,
            ),
        )

    def _patch_one(self, x: float, y: float, origin, **kwargs):
        import matplotlib.patches as mpatches

        kwargs.setdefault("fill", False)
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        corners = np.array(
            [
                [-0.5 * self.w, -0.5 * self.h],
                [0.5 * self.w, -0.5 * self.h],
                [0.5 * self.w, 0.5 * self.h],
                [-0.5 * self.w, 0.5 * self.h],
            ],
            dtype=np.float64,
        )
        vertices = np.column_stack(
            (
                x + cos_t * corners[:, 0] - sin_t * corners[:, 1] - origin[0],
                y + sin_t * corners[:, 0] + cos_t * corners[:, 1] - origin[1],
            )
        )
        return mpatches.Polygon(vertices, closed=True, **kwargs)


class CircAn(PixelAp):
    """Circular annulus in pixel coordinates.

    Parameters
    ----------
    positions : tuple or array-like
        Annulus center or centers as ``(x, y)`` pixel coordinates. Multiple
        centers are accepted as an ``(N, 2)`` array-like object.
    r_in, r_out : float
        Inner and outer radii in pixels. ``r_in`` may be zero and must be
        smaller than ``r_out``.
    validate : bool, optional
        If `False`, skip input validation and assume finite geometry, ordered
        radii, and normalized positions.
    """

    def __init__(self, positions, r_in: float, r_out: float, *, validate: bool = True):
        super().__init__(positions, validate=validate)
        self.r_in = float(r_in)
        if validate and (not math.isfinite(self.r_in) or self.r_in < 0.0):
            raise ValueError("r_in must be a nonnegative finite scalar")
        self.r_out = (
            require_positive_float(r_out, "r_out") if validate else float(r_out)
        )
        if validate and self.r_in >= self.r_out:
            raise ValueError("r_in must be smaller than r_out")

    @property
    def area(self) -> float:
        return math.pi * (self.r_out * self.r_out - self.r_in * self.r_in)

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        return bbox_from_extent(x, y, self.r_out, self.r_out)

    def bboxes(self) -> list[BoundingBox]:
        """Override the default python for-loop with a Rust-accelerated batch function."""
        return bboxes_circ(
            self._x,
            self._y,
            self.r_out,
            validate=self._validate,
        )

    def _exact_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_exact(
            "weights_circ_ann_exact", bbox, (x, y, self.r_in, self.r_out)
        )

    def _center_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_center(
            "weights_circ_ann_center", bbox, (x, y, self.r_in, self.r_out)
        )

    def weights(self, method: str = "exact") -> list[np.ndarray]:
        method = self._weight_method(method)
        weights_func = (
            weights_circ_ann_exact if method == "exact" else weights_circ_ann_center
        )
        weights, _ = weights_func(
            self._x,
            self._y,
            self.r_in,
            self.r_out,
            validate=self._validate,
        )
        return weights

    def apsum(
        self, data, mask=None, *, method: str = "exact", return_npix: bool = True
    ):
        method = self._weight_method(method)
        apsum_func = (
            apsum_circ_ann_exact if method == "exact" else apsum_circ_ann_center
        )
        result = apsum_func(
            data,
            self._x,
            self._y,
            self.r_in,
            self.r_out,
            mask=mask,
            return_npix=return_npix,
            validate=self._validate,
        )
        if not return_npix:
            return _shape_apsum_result(self, result)
        apsum, npix = result
        return _shape_apsum_result(self, apsum, npix)

    def npix(self, shape: tuple[int, int], *, method: str = "exact", mask=None):
        method = self._weight_method(method)
        npix_func = npix_circ_ann_exact if method == "exact" else npix_circ_ann_center
        return _shape_apsum_result(
            self,
            npix_func(
                self._x,
                self._y,
                self.r_in,
                self.r_out,
                shape=shape,
                mask=mask,
                validate=self._validate,
            ),
        )

    def _patch_one(self, x: float, y: float, origin, **kwargs):
        import matplotlib.patches as mpatches

        kwargs.setdefault("fill", False)
        outer = mpatches.Circle((x - origin[0], y - origin[1]), self.r_out)
        inner = mpatches.Circle((x - origin[0], y - origin[1]), self.r_in)
        return mpatches.PathPatch(
            compound_annulus_path(
                outer.get_path().transformed(outer.get_transform()),
                inner.get_path().transformed(inner.get_transform()),
            ),
            **kwargs,
        )


class EllipAn(PixelAp):
    """Elliptical annulus in pixel coordinates.

    Parameters
    ----------
    positions : tuple or array-like
        Annulus center or centers as ``(x, y)`` pixel coordinates. Multiple
        centers are accepted as an ``(N, 2)`` array-like object.
    a_in, b_in : float
        Inner ellipse semiaxes in pixels.
    a_out, b_out : float
        Outer ellipse semiaxes in pixels. Each outer axis must be larger than
        the corresponding inner axis.
    theta_in, theta_out : float, optional
        Inner and outer rotation angles in radians, measured counterclockwise
        from the positive x axis. If ``theta_out`` is `None`, it defaults to
        ``theta_in``.
    validate : bool, optional
        If `False`, skip input validation and assume finite, positive geometry,
        ordered axes, and normalized positions.
    """

    def __init__(
        self,
        positions,
        a_in: float,
        b_in: float,
        a_out: float,
        b_out: float,
        *,
        theta_in: float = 0.0,
        theta_out: float | None = None,
        validate: bool = True,
    ):
        super().__init__(positions, validate=validate)
        self.a_in = require_positive_float(a_in, "a_in") if validate else float(a_in)
        self.b_in = require_positive_float(b_in, "b_in") if validate else float(b_in)
        self.a_out = (
            require_positive_float(a_out, "a_out") if validate else float(a_out)
        )
        self.b_out = (
            require_positive_float(b_out, "b_out") if validate else float(b_out)
        )
        self.theta_in, self.theta_out = resolve_theta_pair(
            theta_in, theta_out, validate=validate
        )
        if validate and (self.a_in >= self.a_out or self.b_in >= self.b_out):
            raise ValueError("inner ellipse axes must be smaller than outer axes")

    @property
    def area(self) -> float:
        return math.pi * (self.a_out * self.b_out - self.a_in * self.b_in)

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        return EllipAp(
            (x, y), self.a_out, self.b_out, self.theta_out, validate=False
        )._bbox_one(x, y)

    def bboxes(self) -> list[BoundingBox]:
        """Override the default python for-loop with a Rust-accelerated batch function."""
        return bboxes_ellip(
            self._x,
            self._y,
            self.a_out,
            self.b_out,
            self.theta_out,
            validate=self._validate,
        )

    def _exact_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_exact(
            "weights_ellip_ann_exact",
            bbox,
            (
                x,
                y,
                self.a_in,
                self.b_in,
                self.a_out,
                self.b_out,
                self.theta_in,
                self.theta_out,
            ),
        )

    def _center_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_center(
            "weights_ellip_ann_center",
            bbox,
            (
                x,
                y,
                self.a_in,
                self.b_in,
                self.a_out,
                self.b_out,
                self.theta_in,
                self.theta_out,
            ),
        )

    def weights(self, method: str = "exact") -> list[np.ndarray]:
        method = self._weight_method(method)
        weights_func = (
            weights_ellip_ann_exact if method == "exact" else weights_ellip_ann_center
        )
        weights, _ = weights_func(
            self._x,
            self._y,
            self.a_in,
            self.b_in,
            self.a_out,
            self.b_out,
            self.theta_in,
            self.theta_out,
            validate=self._validate,
        )
        return weights

    def apsum(
        self, data, mask=None, *, method: str = "exact", return_npix: bool = True
    ):
        method = self._weight_method(method)
        apsum_func = (
            apsum_ellip_ann_exact if method == "exact" else apsum_ellip_ann_center
        )
        result = apsum_func(
            data,
            self._x,
            self._y,
            self.a_in,
            self.b_in,
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

    def npix(self, shape: tuple[int, int], *, method: str = "exact", mask=None):
        method = self._weight_method(method)
        npix_func = npix_ellip_ann_exact if method == "exact" else npix_ellip_ann_center
        return _shape_apsum_result(
            self,
            npix_func(
                self._x,
                self._y,
                self.a_in,
                self.b_in,
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

        kwargs.setdefault("fill", False)
        outer = EllipAp(
            (x, y), self.a_out, self.b_out, self.theta_out, validate=False
        )._patch_one(x, y, origin)
        inner = EllipAp(
            (x, y), self.a_in, self.b_in, self.theta_in, validate=False
        )._patch_one(x, y, origin)
        return mpatches.PathPatch(
            compound_annulus_path(
                outer.get_path().transformed(outer.get_transform()),
                inner.get_path().transformed(inner.get_transform()),
            ),
            **kwargs,
        )


class RectAn(PixelAp):
    """Rotated rectangular annulus in pixel coordinates.

    Parameters
    ----------
    positions : tuple or array-like
        Annulus center or centers as ``(x, y)`` pixel coordinates. Multiple
        centers are accepted as an ``(N, 2)`` array-like object.
    w_in, h_in : float
        Inner rectangle full width and height in pixels. ``w_in`` is measured
        along the inner rectangle's local x axis, and ``h_in`` along its local
        y axis.
    w_out, h_out : float
        Outer rectangle full width and height in pixels. ``w_out`` is measured
        along the outer rectangle's local x axis, and ``h_out`` along its local
        y axis. Each outer dimension must be larger than the corresponding
        inner dimension. At ``theta_out=0``, the outer width axis is aligned
        with the image x axis and the outer height axis with the image y axis.
    theta_in : float, optional
        Inner rectangle rotation angle in radians, measured counterclockwise
        from the positive image x axis to the inner rectangle's local width
        axis.
    theta_out : float, optional
        Outer rectangle rotation angle in radians, measured counterclockwise
        from the positive image x axis to the outer rectangle's local width
        axis. If `None`, it defaults to ``theta_in``.
    validate : bool, optional
        If `False`, skip input validation and assume finite, positive geometry,
        ordered dimensions, and normalized positions.
    """

    def __init__(
        self,
        positions,
        w_in: float,
        h_in: float,
        w_out: float,
        h_out: float,
        *,
        theta_in: float = 0.0,
        theta_out: float | None = None,
        validate: bool = True,
    ):
        super().__init__(positions, validate=validate)
        self.w_in = require_positive_float(w_in, "w_in") if validate else float(w_in)
        self.h_in = require_positive_float(h_in, "h_in") if validate else float(h_in)
        self.w_out = (
            require_positive_float(w_out, "w_out") if validate else float(w_out)
        )
        self.h_out = (
            require_positive_float(h_out, "h_out") if validate else float(h_out)
        )
        self.theta_in, self.theta_out = resolve_theta_pair(
            theta_in, theta_out, validate=validate
        )
        if validate and (self.w_in >= self.w_out or self.h_in >= self.h_out):
            raise ValueError(
                "inner rectangle dimensions must be smaller than outer dimensions"
            )

    @property
    def area(self) -> float:
        return self.w_out * self.h_out - self.w_in * self.h_in

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        return RectAp(
            (x, y), self.w_out, self.h_out, self.theta_out, validate=False
        )._bbox_one(x, y)

    def bboxes(self) -> list[BoundingBox]:
        """Override the default python for-loop with a Rust-accelerated batch function."""
        return bboxes_rect(
            self._x,
            self._y,
            self.w_out,
            self.h_out,
            self.theta_out,
            validate=self._validate,
        )

    def _exact_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_exact(
            "weights_rect_ann_exact",
            bbox,
            (
                x,
                y,
                self.w_in,
                self.h_in,
                self.w_out,
                self.h_out,
                self.theta_in,
                self.theta_out,
            ),
        )

    def _center_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_center(
            "weights_rect_ann_center",
            bbox,
            (
                x,
                y,
                self.w_in,
                self.h_in,
                self.w_out,
                self.h_out,
                self.theta_in,
                self.theta_out,
            ),
        )

    def weights(self, method: str = "exact") -> list[np.ndarray]:
        method = self._weight_method(method)
        weights_func = (
            weights_rect_ann_exact if method == "exact" else weights_rect_ann_center
        )
        weights, _ = weights_func(
            self._x,
            self._y,
            self.w_in,
            self.h_in,
            self.w_out,
            self.h_out,
            self.theta_in,
            self.theta_out,
            validate=self._validate,
        )
        return weights

    def apsum(
        self, data, mask=None, *, method: str = "exact", return_npix: bool = True
    ):
        method = self._weight_method(method)
        apsum_func = (
            apsum_rect_ann_exact if method == "exact" else apsum_rect_ann_center
        )
        result = apsum_func(
            data,
            self._x,
            self._y,
            self.w_in,
            self.h_in,
            self.w_out,
            self.h_out,
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

    def npix(self, shape: tuple[int, int], *, method: str = "exact", mask=None):
        method = self._weight_method(method)
        npix_func = npix_rect_ann_exact if method == "exact" else npix_rect_ann_center
        return _shape_apsum_result(
            self,
            npix_func(
                self._x,
                self._y,
                self.w_in,
                self.h_in,
                self.w_out,
                self.h_out,
                self.theta_in,
                self.theta_out,
                shape=shape,
                mask=mask,
                validate=self._validate,
            ),
        )

    def _patch_one(self, x: float, y: float, origin, **kwargs):
        import matplotlib.patches as mpatches

        kwargs.setdefault("fill", False)
        outer = RectAp(
            (x, y), self.w_out, self.h_out, self.theta_out, validate=False
        )._patch_one(x, y, origin)
        inner = RectAp(
            (x, y), self.w_in, self.h_in, self.theta_in, validate=False
        )._patch_one(x, y, origin)
        return mpatches.PathPatch(
            compound_annulus_path(
                outer.get_path().transformed(outer.get_transform()),
                inner.get_path().transformed(inner.get_transform()),
            ),
            **kwargs,
        )

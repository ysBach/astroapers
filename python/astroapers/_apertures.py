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

from ._aperture_kernel_ops import (
    CIRC_AN_OPS,
    CIRC_OPS,
    ELLIP_AN_OPS,
    ELLIP_OPS,
    RECT_AN_OPS,
    RECT_OPS,
)
from ._containers import BoundingBox, validate_mask
from .kernels import (
    _weights_many as _kernel_weights_many,
    _weights_one,
    bboxes_circ,
    bboxes_ellip,
    bboxes_rect,
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
    work (`positions`, `area`, `weights_exact()`, `apsum_exact()`, `plot()`,
    and `to_patches()`), while heavy aperture summation routes through
    Rust-backed kernels.

    The intended performance split is:

    - use shape-specific `apsum_*_exact()` functions for fastest bulk aperture summation;
    - use `weights_exact()` and `bboxes()` when bbox-tight weights are useful;
    - use `BoundingBox` methods when you already have raw weights and boxes.

    Constructors accept ``validate=False`` as an expert-only path for high-N
    workflows. It assumes positions and geometry parameters are already valid,
    skips constructor positivity/finite/order checks, and avoids copying an
    already-2D ndarray of positions. Private rasterization option strings
    are still validated.
    """

    _kernels = None

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
        It does not compute aperture weights; use :meth:`weights_exact` or
        :meth:`weights_center` for the corresponding bbox-tight weight
        arrays.
        """
        return [self._bbox_one(float(x), float(y)) for x, y in self.positions]

    def weights_exact(self) -> list[np.ndarray]:
        """Return bbox-tight fractional-overlap aperture weights."""
        return self._weights_exact()

    def weights_center(self) -> list[np.ndarray]:
        """Return bbox-tight center-sampled binary aperture weights."""
        return self._weights_center()

    def npix_exact(self, shape: tuple[int, int], *, mask=None):
        """Return in-frame fractional-overlap effective pixel counts."""
        return self._npix_exact(shape, mask=mask)

    def npix_center(self, shape: tuple[int, int], *, mask=None):
        """Return in-frame center-sampled pixel counts."""
        return self._npix_center(shape, mask=mask)

    def apsum_exact(self, data, mask=None, *, return_npix: bool = True):
        """Return fractional-overlap aperture sums, and npix by default."""
        return self._apsum_exact(
            data,
            mask=mask,
            return_npix=return_npix,
        )

    def apsum_center(self, data, mask=None, *, return_npix: bool = True):
        """Return center-sampled aperture sums, and sampled npix by default."""
        return self._apsum_center(
            data,
            mask=mask,
            return_npix=return_npix,
        )

    def _weights_exact(self) -> list[np.ndarray]:
        """Return bbox-tight fractional-overlap weights for all positions."""
        if self._kernels.weights_exact is not None:
            return self._weights_many(self._kernels.weights_exact)
        return [
            self._weights_exact_one(float(x), float(y), bbox)
            for (x, y), bbox in zip(self.positions, self.bboxes(), strict=True)
        ]

    def _weights_center(self) -> list[np.ndarray]:
        """Return bbox-tight center-sampled weights for all positions."""
        if self._kernels.weights_center is not None:
            return self._weights_many(self._kernels.weights_center)
        return [
            self._weights_center_one(float(x), float(y), bbox)
            for (x, y), bbox in zip(self.positions, self.bboxes(), strict=True)
        ]

    @property
    def _pars(self) -> tuple:
        raise NotImplementedError

    def _weights_many(self, func) -> list[np.ndarray]:
        weights, _ = _kernel_weights_many(
            func,
            self._x,
            self._y,
            *self._pars,
            validate=self._validate,
        )
        return weights

    def _npix_exact(self, shape: tuple[int, int], *, mask=None):
        if self._kernels.npix_exact is not None:
            return self._npix_many(self._kernels.npix_exact, shape, mask=mask)
        return self._npix_from_weights(self._weights_exact(), shape, mask=mask)

    def _npix_center(self, shape: tuple[int, int], *, mask=None):
        if self._kernels.npix_center is not None:
            return self._npix_many(self._kernels.npix_center, shape, mask=mask)
        return self._npix_from_weights(self._weights_center(), shape, mask=mask)

    def _npix_from_weights(self, weights: list[np.ndarray], shape, *, mask=None):
        """Return in-frame effective pixel counts.

        Parameters
        ----------
        shape : tuple[int, int]
            Image shape as ``(ny, nx)``.
        mask : array_like of bool, optional
            Boolean image mask with shape ``shape``. ``True`` pixels are
            excluded from the returned effective pixel count.
        """
        boxes = self.bboxes()
        npix = np.empty(len(weights), dtype=np.float64)
        for idx, (weight, bbox) in enumerate(zip(weights, boxes, strict=True)):
            npix[idx] = bbox.npix(weight, shape, mask=mask, validate=self._validate)
        return _shape_apsum_result(self, npix)

    def _apsum_exact(self, data, mask=None, *, return_npix: bool = True):
        if self._kernels.apsum_exact is not None:
            return self._apsum_many(
                self._kernels.apsum_exact,
                data,
                mask=mask,
                return_npix=return_npix,
            )
        return self._apsum_from_weights(
            data, self._weights_exact(), mask=mask, return_npix=return_npix
        )

    def _apsum_center(self, data, mask=None, *, return_npix: bool = True):
        if self._kernels.apsum_center is not None:
            return self._apsum_many(
                self._kernels.apsum_center,
                data,
                mask=mask,
                return_npix=return_npix,
            )
        return self._apsum_from_weights(
            data, self._weights_center(), mask=mask, return_npix=return_npix
        )

    def _apsum_many(self, func, data, mask=None, *, return_npix: bool = True):
        result = func(
            data,
            self._x,
            self._y,
            *self._pars,
            mask=mask,
            return_npix=return_npix,
            validate=self._validate,
        )
        if not return_npix:
            return _shape_apsum_result(self, result)
        apsum, npix = result
        return _shape_apsum_result(self, apsum, npix)

    def _npix_many(self, func, shape: tuple[int, int], *, mask=None):
        return _shape_apsum_result(
            self,
            func(
                self._x,
                self._y,
                *self._pars,
                shape=shape,
                mask=mask,
                validate=self._validate,
            ),
        )

    def _apsum_from_weights(
        self, data, weights: list[np.ndarray], mask=None, *, return_npix: bool = True
    ):
        """Return aperture apsum sums, and in-frame npix by default.

        This is the main object-level aperture-sum method. It uses fractional
        overlap masks. For maximum throughput with many circle, ellipse, or
        rectangle apertures, prefer the shape-specific bulk functions such as
        `apsum_circ_exact()`, `apsum_ellip_exact()`, and `apsum_rect_exact()`.
        """
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("data must be a 2-D array")
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

    def sampled_values(
        self, data, *, mask=None, flat: bool = False, return_pix: bool = False
    ):
        """Return unweighted values whose pixel centers fall in the aperture.

        This returns raw data values selected by :meth:`weights_center`, not
        the center weights themselves. Use :meth:`weights_center` when you need
        bbox-tight binary mask arrays. Pixels where ``mask`` is `True` are
        omitted. By default, this returns one 1-D array per aperture position,
        even for scalar apertures. With ``flat=True``, return
        ``(flat_values, offsets)`` where aperture ``i`` maps to
        ``flat_values[offsets[i]:offsets[i + 1]]``. With ``return_pix=True``,
        also return absolute pixel-center coordinate arrays aligned
        element-by-element with the returned values. For two-dimensional
        apertures, each coordinate tuple is ``(yy, xx)``. Subtract the aperture
        center to get relative offsets, and use ``np.hypot(dx, dy)`` to compute
        Euclidean radial distance.
        """
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("data must be a 2-D array")
        bad = None if mask is None else validate_mask(mask, arr.shape)
        if return_pix:
            pairs = [
                self._sampled_values_one(arr, float(x), float(y), bad, True)
                for x, y in self.positions
            ]
            values = [value for value, _pix in pairs]
            pix = [coord for _value, coord in pairs]
            if flat:
                flat_values, offsets = _flatten_value_list(values, arr.dtype)
                flat_pix = []
                for axis_coords in zip(*pix, strict=True):
                    flat_axis, axis_offsets = _flatten_value_list(
                        axis_coords, np.float64
                    )
                    if not np.array_equal(offsets, axis_offsets):
                        raise RuntimeError("sampled value and pix offsets differ")
                    flat_pix.append(flat_axis)
                return flat_values, tuple(flat_pix), offsets
            return values, pix
        values = [
            self._sampled_values_one(arr, float(x), float(y), bad, False)
            for x, y in self.positions
        ]
        if flat:
            return _flatten_value_list(values, arr.dtype)
        return values

    def weighted_values(self, data, *, mask=None, flat: bool = False):
        """Return fractional-overlap weighted aperture values.

        Returned values are ``data * fractional_weight`` for positive-weight
        pixels. Pixels where ``mask`` is `True` are omitted. By default, this
        returns one 1-D array per aperture position, even for scalar apertures.
        With ``flat=True``, return ``(flat_values, offsets)`` where aperture
        ``i`` maps to ``flat_values[offsets[i]:offsets[i + 1]]``.
        """
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("data must be a 2-D array")
        bad = None if mask is None else validate_mask(mask, arr.shape)
        values = [
            self._weighted_values_one(arr, float(x), float(y), bad)
            for x, y in self.positions
        ]
        if flat:
            return _flatten_value_list(values, arr.dtype)
        return values

    def sampled_cutout(
        self, data, mask=None, fill_value: float = np.nan, return_pix: bool = False
    ):
        """Return bbox-tight center-sampled cutouts.

        Selected pixels contain unweighted data values. Masked and off-image
        pixels contain ``fill_value``. This always returns one 2-D cutout per
        aperture position. With ``return_pix=True``, also return absolute
        pixel-center coordinate arrays with the same bbox-tight shapes as the
        cutouts. For two-dimensional apertures, each coordinate tuple is
        ``(yy, xx)``. Subtract the aperture center to get relative offsets, and
        use ``np.hypot(dx, dy)`` to compute Euclidean radial distance.
        """
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("data must be a 2-D array")
        bad = None if mask is None else validate_mask(mask, arr.shape)
        cutouts = [
            self._cutout_center_one(arr, float(x), float(y), bad, fill_value)
            for x, y in self.positions
        ]
        if return_pix:
            pix = [
                self._cutout_pix_one(arr, float(x), float(y), bad)
                for x, y in self.positions
            ]
            return cutouts, pix
        return cutouts

    def weighted_cutout(self, data, mask=None, fill_value: float = np.nan):
        """Return bbox-tight fractional-overlap weighted cutouts.

        Unmasked overlapping pixels contain ``data * fractional_weight``.
        Masked and off-image pixels contain ``fill_value``. This always
        returns one 2-D cutout per aperture position.
        """
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("data must be a 2-D array")
        bad = None if mask is None else validate_mask(mask, arr.shape)
        cutouts = [
            self._cutout_exact_one(arr, float(x), float(y), bad, fill_value)
            for x, y in self.positions
        ]
        return cutouts

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

    def _sampled_values_one(
        self, data: np.ndarray, x: float, y: float, mask, return_pix: bool
    ):
        bbox = self._bbox_one(x, y)
        overlap = bbox.overlap_slices(data.shape)
        if overlap is None:
            empty_values = np.array([], dtype=data.dtype)
            if return_pix:
                empty_axis = np.array([], dtype=np.float64)
                return empty_values, (empty_axis, empty_axis.copy())
            return empty_values
        data_slices, mask_slices = overlap
        selected = self._weights_center_one(x, y, bbox)[mask_slices] > 0.0
        if mask is not None:
            selected &= ~mask[data_slices]
        values = data[data_slices][selected].ravel()
        if not return_pix:
            return values
        rows, cols = np.nonzero(selected)
        origin_x = bbox.ixmin + mask_slices[1].start
        origin_y = bbox.iymin + mask_slices[0].start
        pix_y = rows.astype(np.float64) + origin_y
        pix_x = cols.astype(np.float64) + origin_x
        return values, (pix_y, pix_x)

    def _weighted_values_one(self, data: np.ndarray, x: float, y: float, mask):
        bbox = self._bbox_one(x, y)
        weights = self._weights_exact_one(x, y, bbox)
        return bbox.weighted_values(weights, data, mask=mask, validate=False)

    def _cutout_exact_one(
        self,
        data: np.ndarray,
        x: float,
        y: float,
        mask,
        fill_value: float,
    ):
        bbox = self._bbox_one(x, y)
        weights = self._weights_exact_one(x, y, bbox)
        return bbox.weighted_cutout(
            weights,
            data,
            mask=mask,
            fill_value=fill_value,
            validate=False,
        )

    def _cutout_center_one(
        self,
        data: np.ndarray,
        x: float,
        y: float,
        mask,
        fill_value: float,
    ):
        bbox = self._bbox_one(x, y)
        weights = self._weights_center_one(x, y, bbox)
        return bbox.weighted_cutout(
            weights,
            data,
            mask=mask,
            fill_value=fill_value,
            validate=False,
        )

    def _cutout_pix_one(self, data: np.ndarray, x: float, y: float, mask):
        bbox = self._bbox_one(x, y)
        pix_y = np.full(bbox.shape, np.nan, dtype=np.float64)
        pix_x = np.full(bbox.shape, np.nan, dtype=np.float64)
        overlap = bbox.overlap_slices(data.shape)
        if overlap is None:
            return pix_y, pix_x
        data_slices, mask_slices = overlap
        selected = self._weights_center_one(x, y, bbox)[mask_slices] > 0.0
        if mask is not None:
            selected &= ~mask[data_slices]
        rows, cols = np.nonzero(selected)
        pix_y_view = pix_y[mask_slices]
        pix_x_view = pix_x[mask_slices]
        pix_y_view[rows, cols] = rows.astype(np.float64) + data_slices[0].start
        pix_x_view[rows, cols] = cols.astype(np.float64) + data_slices[1].start
        return pix_y, pix_x

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        """Return one scalar bounding box.

        Fundamental vector aperture paths override `bboxes()` and call Rust.
        `_bbox_one()` stays scalar and Python-side for object convenience,
        plotting, weighted value extraction, and composite shapes where a Rust
        round trip per aperture would dominate this O(1) geometry.
        """
        raise NotImplementedError

    def _weights_exact_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        if self._kernels.weights_exact_one is None:
            raise NotImplementedError
        return _weights_one(self._kernels.weights_exact_one, bbox, (x, y, *self._pars))

    def _weights_center_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        if self._kernels.weights_center_one is None:
            raise NotImplementedError
        return _weights_one(self._kernels.weights_center_one, bbox, (x, y, *self._pars))

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


def _flatten_value_list(
    values: list[np.ndarray], dtype
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.empty(len(values) + 1, dtype=np.intp)
    offsets[0] = 0
    for idx, value in enumerate(values):
        offsets[idx + 1] = offsets[idx] + value.size
    if values:
        return np.concatenate(values), offsets
    return np.array([], dtype=dtype), offsets


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

    _kernels = CIRC_OPS

    def __init__(self, positions, r: float, *, validate: bool = True):
        super().__init__(positions, validate=validate)
        self.r = require_positive_float(r, "r") if validate else float(r)

    @property
    def _pars(self) -> tuple[float]:
        return (self.r,)

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
            *self._pars,
            validate=self._validate,
        )

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

    _kernels = ELLIP_OPS

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
    def _pars(self) -> tuple[float, float, float]:
        return self.a, self.b, self.theta

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
            *self._pars,
            validate=self._validate,
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

    _kernels = RECT_OPS

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
    def _pars(self) -> tuple[float, float, float]:
        return self.w, self.h, self.theta

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
            *self._pars,
            validate=self._validate,
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

    _kernels = CIRC_AN_OPS

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
    def _pars(self) -> tuple[float, float]:
        return self.r_in, self.r_out

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
        Outer ellipse semiaxes in pixels. Each outer axis must be at least the
        corresponding inner axis, and at least one outer axis must be larger.
    theta_in, theta_out : float, optional
        Inner and outer rotation angles in radians, measured counterclockwise
        from the positive x axis. If ``theta_out`` is `None`, it defaults to
        ``theta_in``.
    validate : bool, optional
        If `False`, skip input validation and assume finite, positive geometry,
        ordered axes, and normalized positions.
    """

    _kernels = ELLIP_AN_OPS

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
        if validate and (
            self.a_in > self.a_out
            or self.b_in > self.b_out
            or (self.a_in == self.a_out and self.b_in == self.b_out)
        ):
            raise ValueError(
                "inner ellipse axes must fit inside outer axes with at least "
                "one larger outer axis"
            )

    @property
    def _pars(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.a_in,
            self.b_in,
            self.a_out,
            self.b_out,
            self.theta_in,
            self.theta_out,
        )

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
        y axis. Each outer dimension must be at least the corresponding inner
        dimension, and at least one outer dimension must be larger. At
        ``theta_out=0``, the outer width axis is aligned with the image x axis
        and the outer height axis with the image y axis.
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

    _kernels = RECT_AN_OPS

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
        if validate and (
            self.w_in > self.w_out
            or self.h_in > self.h_out
            or (self.w_in == self.w_out and self.h_in == self.h_out)
        ):
            raise ValueError(
                "inner rectangle dimensions must fit inside outer dimensions "
                "with at least one larger outer dimension"
            )

    @property
    def _pars(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.w_in,
            self.h_in,
            self.w_out,
            self.h_out,
            self.theta_in,
            self.theta_out,
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

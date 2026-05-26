"""Python layer that calls the raw Rust aperture kernels.

Use ``import astroapers._rust as aapr`` for maximum performance. This module is
also useful as source-code reference for raw-call patterns.
"""

from __future__ import annotations

import numpy as np

from . import _rust as aapr
from ._containers import BoundingBox, validate_mask  # noqa: F401
from ._kernel_dispatch import (
    _apsum_or_masked,
    _npix_from_weights,
    _subtract_apsum,
    _weight_apsum,
)
from ._kernel_docs import apply_kernel_docstrings
from ._kernel_validation import (
    _boxes_from_tuple,
    _floatify,
    _positions,
    _shape2,
    _theta_pair,
    _validate_circ_ann_radii,
    _validate_inner_outer_axes,
    _validate_pill_ann,
    _validate_wedge,
    _wedge_outer_pair,
)
from ._path_encoding import _encode_path_commands

__all__ = [
    "apsum_circ_ann_exact",
    "apsum_circ_ann_center",
    "apsum_circ_center",
    "apsum_circ_exact",
    "apsum_ellip_ann_center",
    "apsum_ellip_ann_exact",
    "apsum_ellip_center",
    "apsum_ellip_exact",
    "apsum_pill_ann_center",
    "apsum_pill_ann_exact",
    "apsum_pill_center",
    "apsum_pill_exact",
    "apsum_rect_ann_center",
    "apsum_rect_ann_exact",
    "apsum_rect_center",
    "apsum_rect_exact",
    "apsum_wedge_center",
    "apsum_wedge_exact",
    "bboxes_circ",
    "bboxes_circ_ann",
    "bboxes_ellip",
    "bboxes_ellip_ann",
    "bboxes_pill",
    "bboxes_pill_ann",
    "bboxes_rect",
    "bboxes_rect_ann",
    "bboxes_wedge",
    "farthest_mask_pixel",
    "weights_circ_ann_center",
    "weights_circ_ann_exact",
    "weights_circ_center",
    "weights_circ_exact",
    "weights_ellip_ann_center",
    "weights_ellip_ann_exact",
    "weights_ellip_center",
    "weights_ellip_exact",
    "weights_pill_ann_center",
    "weights_pill_ann_exact",
    "weights_pill_center",
    "weights_pill_exact",
    "weights_rect_ann_center",
    "weights_rect_ann_exact",
    "weights_rect_center",
    "weights_rect_exact",
    "weights_wedge_center",
    "weights_wedge_exact",
    "npix_circ_ann_center",
    "npix_circ_ann_exact",
    "npix_circ_center",
    "npix_circ_exact",
    "npix_ellip_ann_center",
    "npix_ellip_ann_exact",
    "npix_ellip_center",
    "npix_ellip_exact",
    "npix_pill_ann_center",
    "npix_pill_ann_exact",
    "npix_pill_center",
    "npix_pill_exact",
    "npix_rect_ann_center",
    "npix_rect_ann_exact",
    "npix_rect_center",
    "npix_rect_exact",
    "npix_wedge_center",
    "npix_wedge_exact",
    "apsum_path_center",
    "apsum_path_exact",
    "bboxes_path",
    "npix_path_center",
    "npix_path_exact",
    "weights_path_center",
    "weights_path_exact",
    "get_parallel_threshold",
    "set_parallel_threshold",
]


def _raw_xy(x, y) -> bool:
    return (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and x.ndim == y.ndim == 1
        and x.flags.c_contiguous
        and y.flags.c_contiguous
    )


def _raw_apsum_args(data, x, y) -> bool:
    return isinstance(data, np.ndarray) and data.flags.c_contiguous and _raw_xy(x, y)


def farthest_mask_pixel(mask, center, *, return_pos: bool = False):
    """Return the farthest `True` mask pixel distance from ``center``.

    Parameters
    ----------
    mask : array_like of bool
        N-dimensional boolean mask. `True` pixels are candidates.
    center : sequence of float
        Center coordinate in NumPy index order. Its length must match
        ``mask.ndim``.
    return_pos : bool, optional
        If `True`, also return all farthest pixel positions as a
        ``(n_ties, mask.ndim)`` array.

    Returns
    -------
    radius : float
        Euclidean distance from ``center`` to the farthest `True` pixel.
    positions : ndarray, optional
        Returned only when ``return_pos=True``. A two-dimensional integer array
        containing every farthest coordinate in NumPy C-order.
    """
    mask_arr = np.ascontiguousarray(mask, dtype=bool)
    center_arr = np.ascontiguousarray(center, dtype=np.float64)
    if center_arr.ndim != 1:
        raise ValueError("center must be one-dimensional")

    radius, ndim, flat_positions = aapr._farthest_mask_pixel(
        mask_arr,
        center_arr,
        return_pos,
    )
    if not return_pos:
        return radius

    positions = np.asarray(flat_positions, dtype=np.intp)
    return radius, positions.reshape(-1, ndim)


def apsum_circ_exact(
    data, x, y, r: float, *, mask=None, return_npix: bool = True, validate=True
):
    if (
        mask is None
        and not return_npix
        and not validate
        and _raw_apsum_args(data, x, y)
    ):
        try:
            return aapr.apsum_circ_exact_sum(data, x, y, float(r))
        except TypeError:
            pass
    return _apsum_or_masked(
        aapr.apsum_circ_exact,
        weights_circ_exact,
        data,
        x,
        y,
        float(r),
        mask=mask,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_circ_ann_exact(
    data,
    x,
    y,
    r_in: float,
    r_out: float,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    if (
        mask is None
        and not return_npix
        and not validate
        and _raw_apsum_args(data, x, y)
    ):
        try:
            return aapr.apsum_circ_ann_exact_sum(data, x, y, *_floatify(r_in, r_out))
        except TypeError:
            pass
    return _apsum_or_masked(
        aapr.apsum_circ_ann_exact,
        weights_circ_ann_exact,
        data,
        x,
        y,
        *_floatify(r_in, r_out),
        mask=mask,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_circ_ann_center(
    data,
    x,
    y,
    r_in: float,
    r_out: float,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    return _weight_apsum(
        weights_circ_ann_center,
        data,
        mask,
        x,
        y,
        *_floatify(r_in, r_out),
        return_npix=return_npix,
        validate=validate,
    )


def apsum_circ_center(
    data, x, y, r: float, *, mask=None, return_npix: bool = True, validate=True
):
    if (
        mask is None
        and not return_npix
        and not validate
        and _raw_apsum_args(data, x, y)
    ):
        try:
            return aapr.apsum_circ_center(data, x, y, float(r))[0]
        except TypeError:
            pass
    return _apsum_or_masked(
        aapr.apsum_circ_center,
        weights_circ_center,
        data,
        x,
        y,
        float(r),
        mask=mask,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_ellip_exact(
    data,
    x,
    y,
    a: float,
    b: float,
    theta: float = 0.0,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    if (
        mask is None
        and not return_npix
        and not validate
        and _raw_apsum_args(data, x, y)
    ):
        try:
            return aapr.apsum_ellip_exact(data, x, y, *_floatify(a, b, theta))[0]
        except TypeError:
            pass
    return _apsum_or_masked(
        aapr.apsum_ellip_exact,
        weights_ellip_exact,
        data,
        x,
        y,
        *_floatify(a, b, theta),
        mask=mask,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_ellip_center(
    data,
    x,
    y,
    a: float,
    b: float,
    theta: float = 0.0,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    if (
        mask is None
        and not return_npix
        and not validate
        and _raw_apsum_args(data, x, y)
    ):
        try:
            return aapr.apsum_ellip_center(data, x, y, *_floatify(a, b, theta))[0]
        except TypeError:
            pass
    return _apsum_or_masked(
        aapr.apsum_ellip_center,
        weights_ellip_center,
        data,
        x,
        y,
        *_floatify(a, b, theta),
        mask=mask,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_ellip_ann_exact(
    data,
    x,
    y,
    a_in: float,
    b_in: float,
    a_out: float,
    b_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_inner_outer_axes(a_in, b_in, a_out, b_out, "ellipse")
    if theta_in == theta_out:
        outer = apsum_ellip_exact(
            data,
            x,
            y,
            a_out,
            b_out,
            theta_out,
            mask=mask,
            return_npix=return_npix,
            validate=validate,
        )
        inner = apsum_ellip_exact(
            data,
            x,
            y,
            a_in,
            b_in,
            theta_in,
            mask=mask,
            return_npix=return_npix,
            validate=validate,
        )
        return _subtract_apsum(outer, inner, return_npix=return_npix)
    return _weight_apsum(
        weights_ellip_ann_exact,
        data,
        mask,
        x,
        y,
        *_floatify(a_in, b_in, a_out, b_out),
        theta_in,
        theta_out,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_ellip_ann_center(
    data,
    x,
    y,
    a_in: float,
    b_in: float,
    a_out: float,
    b_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_inner_outer_axes(a_in, b_in, a_out, b_out, "ellipse")
    if theta_in == theta_out:
        outer = apsum_ellip_center(
            data,
            x,
            y,
            a_out,
            b_out,
            theta_out,
            mask=mask,
            return_npix=return_npix,
            validate=validate,
        )
        inner = apsum_ellip_center(
            data,
            x,
            y,
            a_in,
            b_in,
            theta_in,
            mask=mask,
            return_npix=return_npix,
            validate=validate,
        )
        return _subtract_apsum(outer, inner, return_npix=return_npix)
    return _weight_apsum(
        weights_ellip_ann_center,
        data,
        mask,
        x,
        y,
        *_floatify(a_in, b_in, a_out, b_out),
        theta_in,
        theta_out,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_rect_exact(
    data,
    x,
    y,
    w: float,
    h: float,
    theta: float = 0.0,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    if (
        mask is None
        and not return_npix
        and not validate
        and _raw_apsum_args(data, x, y)
    ):
        try:
            return aapr.apsum_rect_exact(data, x, y, *_floatify(w, h, theta))[0]
        except TypeError:
            pass
    return _apsum_or_masked(
        aapr.apsum_rect_exact,
        weights_rect_exact,
        data,
        x,
        y,
        *_floatify(w, h, theta),
        mask=mask,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_rect_ann_exact(
    data,
    x,
    y,
    w_in: float,
    h_in: float,
    w_out: float,
    h_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_inner_outer_axes(w_in, h_in, w_out, h_out, "rectangle")
    if theta_in == theta_out:
        outer = apsum_rect_exact(
            data,
            x,
            y,
            w_out,
            h_out,
            theta_out,
            mask=mask,
            return_npix=return_npix,
            validate=validate,
        )
        inner = apsum_rect_exact(
            data,
            x,
            y,
            w_in,
            h_in,
            theta_in,
            mask=mask,
            return_npix=return_npix,
            validate=validate,
        )
        return _subtract_apsum(outer, inner, return_npix=return_npix)
    return _weight_apsum(
        weights_rect_ann_exact,
        data,
        mask,
        x,
        y,
        *_floatify(w_in, h_in, w_out, h_out),
        theta_in,
        theta_out,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_rect_ann_center(
    data,
    x,
    y,
    w_in: float,
    h_in: float,
    w_out: float,
    h_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_inner_outer_axes(w_in, h_in, w_out, h_out, "rectangle")
    if theta_in == theta_out:
        outer = apsum_rect_center(
            data,
            x,
            y,
            w_out,
            h_out,
            theta_out,
            mask=mask,
            return_npix=return_npix,
            validate=validate,
        )
        inner = apsum_rect_center(
            data,
            x,
            y,
            w_in,
            h_in,
            theta_in,
            mask=mask,
            return_npix=return_npix,
            validate=validate,
        )
        return _subtract_apsum(outer, inner, return_npix=return_npix)
    return _weight_apsum(
        weights_rect_ann_center,
        data,
        mask,
        x,
        y,
        *_floatify(w_in, h_in, w_out, h_out),
        theta_in,
        theta_out,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_rect_center(
    data,
    x,
    y,
    w: float,
    h: float,
    theta: float = 0.0,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    if (
        mask is None
        and not return_npix
        and not validate
        and _raw_apsum_args(data, x, y)
    ):
        try:
            return aapr.apsum_rect_center(data, x, y, *_floatify(w, h, theta))[0]
        except TypeError:
            pass
    return _apsum_or_masked(
        aapr.apsum_rect_center,
        weights_rect_center,
        data,
        x,
        y,
        *_floatify(w, h, theta),
        mask=mask,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_pill_exact(
    data,
    x,
    y,
    w: float,
    a: float,
    b: float,
    theta: float = 0.0,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    return _weight_apsum(
        weights_pill_exact,
        data,
        mask,
        x,
        y,
        *_floatify(w, a, b, theta),
        return_npix=return_npix,
        validate=validate,
    )


def apsum_pill_center(
    data,
    x,
    y,
    w: float,
    a: float,
    b: float,
    theta: float = 0.0,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    return _weight_apsum(
        weights_pill_center,
        data,
        mask,
        x,
        y,
        *_floatify(w, a, b, theta),
        return_npix=return_npix,
        validate=validate,
    )


def apsum_pill_ann_exact(
    data,
    x,
    y,
    w_in: float,
    a_in: float,
    b_in: float,
    w_out: float,
    a_out: float,
    b_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    if validate:
        _validate_pill_ann(w_in, a_in, b_in, w_out, a_out, b_out)
    return _weight_apsum(
        weights_pill_ann_exact,
        data,
        mask,
        x,
        y,
        *_floatify(w_in, a_in, b_in, w_out, a_out, b_out),
        *_theta_pair(theta_in, theta_out, validate=validate),
        return_npix=return_npix,
        validate=validate,
    )


def apsum_pill_ann_center(
    data,
    x,
    y,
    w_in: float,
    a_in: float,
    b_in: float,
    w_out: float,
    a_out: float,
    b_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    if validate:
        _validate_pill_ann(w_in, a_in, b_in, w_out, a_out, b_out)
    return _weight_apsum(
        weights_pill_ann_center,
        data,
        mask,
        x,
        y,
        *_floatify(w_in, a_in, b_in, w_out, a_out, b_out),
        *_theta_pair(theta_in, theta_out, validate=validate),
        return_npix=return_npix,
        validate=validate,
    )


def apsum_wedge_exact(
    data,
    x,
    y,
    r_in: float,
    r_out: float,
    theta_in: float,
    dtheta_in: float,
    theta_out: float | None = None,
    dtheta_out: float | None = None,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    outer_angles = _wedge_outer_pair(theta_in, dtheta_in, theta_out, dtheta_out)
    if validate:
        _validate_wedge(r_in, r_out, theta_in, dtheta_in, *outer_angles)
    return _weight_apsum(
        weights_wedge_exact,
        data,
        mask,
        x,
        y,
        *_floatify(r_in, r_out, theta_in, dtheta_in),
        *outer_angles,
        return_npix=return_npix,
        validate=validate,
    )


def apsum_wedge_center(
    data,
    x,
    y,
    r_in: float,
    r_out: float,
    theta_in: float,
    dtheta_in: float,
    theta_out: float | None = None,
    dtheta_out: float | None = None,
    *,
    mask=None,
    return_npix: bool = True,
    validate=True,
):
    outer_angles = _wedge_outer_pair(theta_in, dtheta_in, theta_out, dtheta_out)
    if validate:
        _validate_wedge(r_in, r_out, theta_in, dtheta_in, *outer_angles)
    return _weight_apsum(
        weights_wedge_center,
        data,
        mask,
        x,
        y,
        *_floatify(r_in, r_out, theta_in, dtheta_in),
        *outer_angles,
        return_npix=return_npix,
        validate=validate,
    )


def npix_circ_exact(
    x, y, r: float, *, shape: tuple[int, int], mask=None, validate=True
):
    if mask is None and not validate and _raw_xy(x, y):
        try:
            return aapr.npix_circ_exact(x, y, float(r), shape[0], shape[1])
        except TypeError:
            pass
    if mask is not None:
        return _npix_from_weights(
            weights_circ_exact,
            x,
            y,
            float(r),
            shape=shape,
            mask=mask,
            validate=validate,
        )
    xs, ys = _positions(x, y, validate=validate)
    ny, nx = _shape2(shape, validate=validate)
    npix = aapr.npix_circ_exact(xs.reshape(-1), ys.reshape(-1), float(r), ny, nx)
    return np.asarray(npix, dtype=np.float64).reshape(xs.shape)


def npix_circ_ann_exact(
    x,
    y,
    r_in: float,
    r_out: float,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    if validate:
        r_in, r_out = _validate_circ_ann_radii(r_in, r_out)
    outer = npix_circ_exact(x, y, r_out, shape=shape, mask=mask, validate=validate)
    if r_in == 0.0:
        return outer
    inner = npix_circ_exact(x, y, r_in, shape=shape, mask=mask, validate=validate)
    return outer - inner


def npix_circ_center(
    x, y, r: float, *, shape: tuple[int, int], mask=None, validate=True
):
    if mask is None and not validate and _raw_xy(x, y):
        try:
            return aapr.npix_circ_center(x, y, float(r), shape[0], shape[1])
        except TypeError:
            pass
    if mask is not None:
        return _npix_from_weights(
            weights_circ_center,
            x,
            y,
            float(r),
            shape=shape,
            mask=mask,
            validate=validate,
        )
    xs, ys = _positions(x, y, validate=validate)
    ny, nx = _shape2(shape, validate=validate)
    npix = aapr.npix_circ_center(xs.reshape(-1), ys.reshape(-1), float(r), ny, nx)
    return np.asarray(npix, dtype=np.float64).reshape(xs.shape)


def npix_circ_ann_center(
    x,
    y,
    r_in: float,
    r_out: float,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    if validate:
        r_in, r_out = _validate_circ_ann_radii(r_in, r_out)
    outer = npix_circ_center(x, y, r_out, shape=shape, mask=mask, validate=validate)
    if r_in == 0.0:
        return outer
    inner = npix_circ_center(x, y, r_in, shape=shape, mask=mask, validate=validate)
    return outer - inner


def npix_ellip_exact(
    x,
    y,
    a: float,
    b: float,
    theta: float = 0.0,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    if mask is None and not validate and _raw_xy(x, y):
        try:
            return aapr.npix_ellip_exact(
                x, y, *_floatify(a, b, theta), shape[0], shape[1]
            )
        except TypeError:
            pass
    if mask is not None:
        return _npix_from_weights(
            weights_ellip_exact,
            x,
            y,
            *_floatify(a, b, theta),
            shape=shape,
            mask=mask,
            validate=validate,
        )
    xs, ys = _positions(x, y, validate=validate)
    ny, nx = _shape2(shape, validate=validate)
    npix = aapr.npix_ellip_exact(
        xs.reshape(-1), ys.reshape(-1), *_floatify(a, b, theta), ny, nx
    )
    return np.asarray(npix, dtype=np.float64).reshape(xs.shape)


def npix_ellip_center(
    x,
    y,
    a: float,
    b: float,
    theta: float = 0.0,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    if mask is None and not validate and _raw_xy(x, y):
        try:
            return aapr.npix_ellip_center(
                x, y, *_floatify(a, b, theta), shape[0], shape[1]
            )
        except TypeError:
            pass
    if mask is not None:
        return _npix_from_weights(
            weights_ellip_center,
            x,
            y,
            *_floatify(a, b, theta),
            shape=shape,
            mask=mask,
            validate=validate,
        )
    xs, ys = _positions(x, y, validate=validate)
    ny, nx = _shape2(shape, validate=validate)
    npix = aapr.npix_ellip_center(
        xs.reshape(-1), ys.reshape(-1), *_floatify(a, b, theta), ny, nx
    )
    return np.asarray(npix, dtype=np.float64).reshape(xs.shape)


def npix_ellip_ann_exact(
    x,
    y,
    a_in: float,
    b_in: float,
    a_out: float,
    b_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_inner_outer_axes(a_in, b_in, a_out, b_out, "ellipse")
    if theta_in == theta_out:
        outer = npix_ellip_exact(
            x, y, a_out, b_out, theta_out, shape=shape, mask=mask, validate=validate
        )
        inner = npix_ellip_exact(
            x, y, a_in, b_in, theta_in, shape=shape, mask=mask, validate=validate
        )
        return outer - inner
    return _npix_from_weights(
        weights_ellip_ann_exact,
        x,
        y,
        *_floatify(a_in, b_in, a_out, b_out),
        theta_in,
        theta_out,
        shape=shape,
        mask=mask,
        validate=validate,
    )


def npix_ellip_ann_center(
    x,
    y,
    a_in: float,
    b_in: float,
    a_out: float,
    b_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_inner_outer_axes(a_in, b_in, a_out, b_out, "ellipse")
    if theta_in == theta_out:
        outer = npix_ellip_center(
            x, y, a_out, b_out, theta_out, shape=shape, mask=mask, validate=validate
        )
        inner = npix_ellip_center(
            x, y, a_in, b_in, theta_in, shape=shape, mask=mask, validate=validate
        )
        return outer - inner
    return _npix_from_weights(
        weights_ellip_ann_center,
        x,
        y,
        *_floatify(a_in, b_in, a_out, b_out),
        theta_in,
        theta_out,
        shape=shape,
        mask=mask,
        validate=validate,
    )


def npix_rect_exact(
    x,
    y,
    w: float,
    h: float,
    theta: float = 0.0,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    if mask is None and not validate and _raw_xy(x, y):
        try:
            return aapr.npix_rect_exact(
                x, y, *_floatify(w, h, theta), shape[0], shape[1]
            )
        except TypeError:
            pass
    if mask is not None:
        return _npix_from_weights(
            weights_rect_exact,
            x,
            y,
            *_floatify(w, h, theta),
            shape=shape,
            mask=mask,
            validate=validate,
        )
    xs, ys = _positions(x, y, validate=validate)
    ny, nx = _shape2(shape, validate=validate)
    npix = aapr.npix_rect_exact(
        xs.reshape(-1), ys.reshape(-1), *_floatify(w, h, theta), ny, nx
    )
    return np.asarray(npix, dtype=np.float64).reshape(xs.shape)


def npix_rect_center(
    x,
    y,
    w: float,
    h: float,
    theta: float = 0.0,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    if mask is None and not validate and _raw_xy(x, y):
        try:
            return aapr.npix_rect_center(
                x, y, *_floatify(w, h, theta), shape[0], shape[1]
            )
        except TypeError:
            pass
    if mask is not None:
        return _npix_from_weights(
            weights_rect_center,
            x,
            y,
            *_floatify(w, h, theta),
            shape=shape,
            mask=mask,
            validate=validate,
        )
    xs, ys = _positions(x, y, validate=validate)
    ny, nx = _shape2(shape, validate=validate)
    npix = aapr.npix_rect_center(
        xs.reshape(-1), ys.reshape(-1), *_floatify(w, h, theta), ny, nx
    )
    return np.asarray(npix, dtype=np.float64).reshape(xs.shape)


def npix_rect_ann_exact(
    x,
    y,
    w_in: float,
    h_in: float,
    w_out: float,
    h_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_inner_outer_axes(w_in, h_in, w_out, h_out, "rectangle")
    if theta_in == theta_out:
        outer = npix_rect_exact(
            x, y, w_out, h_out, theta_out, shape=shape, mask=mask, validate=validate
        )
        inner = npix_rect_exact(
            x, y, w_in, h_in, theta_in, shape=shape, mask=mask, validate=validate
        )
        return outer - inner
    return _npix_from_weights(
        weights_rect_ann_exact,
        x,
        y,
        *_floatify(w_in, h_in, w_out, h_out),
        theta_in,
        theta_out,
        shape=shape,
        mask=mask,
        validate=validate,
    )


def npix_rect_ann_center(
    x,
    y,
    w_in: float,
    h_in: float,
    w_out: float,
    h_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_inner_outer_axes(w_in, h_in, w_out, h_out, "rectangle")
    if theta_in == theta_out:
        outer = npix_rect_center(
            x, y, w_out, h_out, theta_out, shape=shape, mask=mask, validate=validate
        )
        inner = npix_rect_center(
            x, y, w_in, h_in, theta_in, shape=shape, mask=mask, validate=validate
        )
        return outer - inner
    return _npix_from_weights(
        weights_rect_ann_center,
        x,
        y,
        *_floatify(w_in, h_in, w_out, h_out),
        theta_in,
        theta_out,
        shape=shape,
        mask=mask,
        validate=validate,
    )


def npix_pill_exact(
    x,
    y,
    w: float,
    a: float,
    b: float,
    theta: float = 0.0,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    return _npix_from_weights(
        weights_pill_exact,
        x,
        y,
        *_floatify(w, a, b, theta),
        shape=shape,
        mask=mask,
        validate=validate,
    )


def npix_pill_center(
    x,
    y,
    w: float,
    a: float,
    b: float,
    theta: float = 0.0,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    return _npix_from_weights(
        weights_pill_center,
        x,
        y,
        *_floatify(w, a, b, theta),
        shape=shape,
        mask=mask,
        validate=validate,
    )


def npix_pill_ann_exact(
    x,
    y,
    w_in: float,
    a_in: float,
    b_in: float,
    w_out: float,
    a_out: float,
    b_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    if validate:
        _validate_pill_ann(w_in, a_in, b_in, w_out, a_out, b_out)
    return _npix_from_weights(
        weights_pill_ann_exact,
        x,
        y,
        *_floatify(w_in, a_in, b_in, w_out, a_out, b_out),
        *_theta_pair(theta_in, theta_out, validate=validate),
        shape=shape,
        mask=mask,
        validate=validate,
    )


def npix_pill_ann_center(
    x,
    y,
    w_in: float,
    a_in: float,
    b_in: float,
    w_out: float,
    a_out: float,
    b_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    if validate:
        _validate_pill_ann(w_in, a_in, b_in, w_out, a_out, b_out)
    return _npix_from_weights(
        weights_pill_ann_center,
        x,
        y,
        *_floatify(w_in, a_in, b_in, w_out, a_out, b_out),
        *_theta_pair(theta_in, theta_out, validate=validate),
        shape=shape,
        mask=mask,
        validate=validate,
    )


def npix_wedge_exact(
    x,
    y,
    r_in: float,
    r_out: float,
    theta_in: float,
    dtheta_in: float,
    theta_out: float | None = None,
    dtheta_out: float | None = None,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    outer_angles = _wedge_outer_pair(theta_in, dtheta_in, theta_out, dtheta_out)
    if validate:
        _validate_wedge(r_in, r_out, theta_in, dtheta_in, *outer_angles)
    return _npix_from_weights(
        weights_wedge_exact,
        x,
        y,
        *_floatify(r_in, r_out, theta_in, dtheta_in),
        *outer_angles,
        shape=shape,
        mask=mask,
        validate=validate,
    )


def npix_wedge_center(
    x,
    y,
    r_in: float,
    r_out: float,
    theta_in: float,
    dtheta_in: float,
    theta_out: float | None = None,
    dtheta_out: float | None = None,
    *,
    shape: tuple[int, int],
    mask=None,
    validate=True,
):
    outer_angles = _wedge_outer_pair(theta_in, dtheta_in, theta_out, dtheta_out)
    if validate:
        _validate_wedge(r_in, r_out, theta_in, dtheta_in, *outer_angles)
    return _npix_from_weights(
        weights_wedge_center,
        x,
        y,
        *_floatify(r_in, r_out, theta_in, dtheta_in),
        *outer_angles,
        shape=shape,
        mask=mask,
        validate=validate,
    )


def _weights_one(func, bbox: BoundingBox, params: tuple[float, ...]) -> np.ndarray:
    """Return aperture weights for one bbox-tight bounding box."""
    data = func(
        *params,
        bbox.ixmin,
        bbox.iymin,
        bbox.shape[0],
        bbox.shape[1],
    )
    return np.asarray(data, dtype=np.float64).reshape(bbox.shape)


def _rust_positions(x, y, *, validate: bool):
    if (
        not validate
        and isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and x.ndim == y.ndim == 1
    ):
        return x, y
    xs, ys = _positions(x, y, validate=validate)
    return xs.reshape(-1), ys.reshape(-1)


def _weights_many(
    func, x, y, *params, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    xs, ys = _rust_positions(x, y, validate=validate)
    weights, ixmins, ixmaxs, iymins, iymaxs = func(xs, ys, *params)
    boxes = _boxes_from_tuple((ixmins, ixmaxs, iymins, iymaxs))
    arrays = [
        np.asarray(weight, dtype=np.float64).reshape(box.shape)
        for weight, box in zip(weights, boxes)
    ]
    return arrays, boxes


def weights_circ_exact(
    x, y, r: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(aapr.weights_circ_exact, x, y, float(r), validate=validate)


def weights_circ_center(
    x, y, r: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(aapr.weights_circ_center, x, y, float(r), validate=validate)


def weights_circ_ann_exact(
    x, y, r_in: float, r_out: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_circ_ann_exact,
        x,
        y,
        *_floatify(r_in, r_out),
        validate=validate,
    )


def weights_circ_ann_center(
    x, y, r_in: float, r_out: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_circ_ann_center,
        x,
        y,
        *_floatify(r_in, r_out),
        validate=validate,
    )


def weights_ellip_exact(
    x, y, a: float, b: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_ellip_exact,
        x,
        y,
        *_floatify(a, b, theta),
        validate=validate,
    )


def weights_ellip_center(
    x, y, a: float, b: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_ellip_center,
        x,
        y,
        *_floatify(a, b, theta),
        validate=validate,
    )


def weights_ellip_ann_exact(
    x,
    y,
    a_in: float,
    b_in: float,
    a_out: float,
    b_out: float,
    theta_in: float,
    theta_out: float,
    *,
    validate: bool = True,
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_ellip_ann_exact,
        x,
        y,
        *_floatify(a_in, b_in, a_out, b_out, theta_in, theta_out),
        validate=validate,
    )


def weights_ellip_ann_center(
    x,
    y,
    a_in: float,
    b_in: float,
    a_out: float,
    b_out: float,
    theta_in: float,
    theta_out: float,
    *,
    validate: bool = True,
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_ellip_ann_center,
        x,
        y,
        *_floatify(a_in, b_in, a_out, b_out, theta_in, theta_out),
        validate=validate,
    )


def weights_rect_exact(
    x, y, w: float, h: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_rect_exact,
        x,
        y,
        *_floatify(w, h, theta),
        validate=validate,
    )


def weights_rect_center(
    x, y, w: float, h: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_rect_center,
        x,
        y,
        *_floatify(w, h, theta),
        validate=validate,
    )


def weights_rect_ann_exact(
    x,
    y,
    w_in: float,
    h_in: float,
    w_out: float,
    h_out: float,
    theta_in: float,
    theta_out: float,
    *,
    validate: bool = True,
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_rect_ann_exact,
        x,
        y,
        *_floatify(w_in, h_in, w_out, h_out, theta_in, theta_out),
        validate=validate,
    )


def weights_rect_ann_center(
    x,
    y,
    w_in: float,
    h_in: float,
    w_out: float,
    h_out: float,
    theta_in: float,
    theta_out: float,
    *,
    validate: bool = True,
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_rect_ann_center,
        x,
        y,
        *_floatify(w_in, h_in, w_out, h_out, theta_in, theta_out),
        validate=validate,
    )


def weights_pill_exact(
    x, y, w: float, a: float, b: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_pill_exact,
        x,
        y,
        *_floatify(w, a, b, theta),
        validate=validate,
    )


def weights_pill_center(
    x, y, w: float, a: float, b: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_pill_center,
        x,
        y,
        *_floatify(w, a, b, theta),
        validate=validate,
    )


def weights_pill_ann_exact(
    x,
    y,
    w_in: float,
    a_in: float,
    b_in: float,
    w_out: float,
    a_out: float,
    b_out: float,
    theta_in: float,
    theta_out: float,
    *,
    validate: bool = True,
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_pill_ann_exact,
        x,
        y,
        *_floatify(w_in, a_in, b_in, w_out, a_out, b_out, theta_in, theta_out),
        validate=validate,
    )


def weights_pill_ann_center(
    x,
    y,
    w_in: float,
    a_in: float,
    b_in: float,
    w_out: float,
    a_out: float,
    b_out: float,
    theta_in: float,
    theta_out: float,
    *,
    validate: bool = True,
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_pill_ann_center,
        x,
        y,
        *_floatify(w_in, a_in, b_in, w_out, a_out, b_out, theta_in, theta_out),
        validate=validate,
    )


def weights_wedge_exact(
    x,
    y,
    r_in: float,
    r_out: float,
    theta_in: float,
    dtheta_in: float,
    theta_out: float,
    dtheta_out: float,
    *,
    validate: bool = True,
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_wedge_exact,
        x,
        y,
        *_floatify(r_in, r_out, theta_in, dtheta_in, theta_out, dtheta_out),
        validate=validate,
    )


def weights_wedge_center(
    x,
    y,
    r_in: float,
    r_out: float,
    theta_in: float,
    dtheta_in: float,
    theta_out: float,
    dtheta_out: float,
    *,
    validate: bool = True,
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        aapr.weights_wedge_center,
        x,
        y,
        *_floatify(r_in, r_out, theta_in, dtheta_in, theta_out, dtheta_out),
        validate=validate,
    )


def bboxes_circ(x, y, r: float, *, validate: bool = True):
    """Return exact circular aperture bounding boxes for one or many centers."""
    xs, ys = _rust_positions(x, y, validate=validate)
    return _boxes_from_tuple(aapr.bboxes_circ(xs, ys, float(r)))


def bboxes_circ_ann(x, y, r_in: float, r_out: float, *, validate: bool = True):
    """Return circular-annulus bounding boxes for one or many centers."""
    if validate:
        _validate_circ_ann_radii(r_in, r_out)
    return bboxes_circ(x, y, float(r_out), validate=validate)


def bboxes_ellip(x, y, a: float, b: float, theta: float, *, validate: bool = True):
    """Return exact elliptical aperture bounding boxes for one or many centers."""
    xs, ys = _rust_positions(x, y, validate=validate)
    return _boxes_from_tuple(aapr.bboxes_ellip(xs, ys, *_floatify(a, b, theta)))


def bboxes_ellip_ann(
    x,
    y,
    a_in: float,
    b_in: float,
    a_out: float,
    b_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    validate: bool = True,
):
    """Return elliptical-annulus bounding boxes for one or many centers."""
    if validate:
        _validate_inner_outer_axes(a_in, b_in, a_out, b_out, "ellipse")
    return bboxes_ellip(
        x,
        y,
        *_floatify(a_out, b_out),
        _theta_pair(theta_in, theta_out, validate=validate)[1],
        validate=validate,
    )


def bboxes_rect(x, y, w: float, h: float, theta: float, *, validate: bool = True):
    """Return exact rotated-rectangle aperture bounding boxes for one or many centers."""
    xs, ys = _rust_positions(x, y, validate=validate)
    return _boxes_from_tuple(aapr.bboxes_rect(xs, ys, *_floatify(w, h, theta)))


def bboxes_rect_ann(
    x,
    y,
    w_in: float,
    h_in: float,
    w_out: float,
    h_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    validate: bool = True,
):
    """Return rotated-rectangle-annulus bounding boxes for one or many centers."""
    if validate:
        _validate_inner_outer_axes(w_in, h_in, w_out, h_out, "rectangle")
    return bboxes_rect(
        x,
        y,
        *_floatify(w_out, h_out),
        _theta_pair(theta_in, theta_out, validate=validate)[1],
        validate=validate,
    )


def bboxes_pill(
    x, y, w: float, a: float, b: float, theta: float = 0.0, *, validate: bool = True
):
    """Return pill aperture bounding boxes for one or many centers."""
    xs, ys = _rust_positions(x, y, validate=validate)
    return _boxes_from_tuple(aapr.bboxes_pill(xs, ys, *_floatify(w, a, b, theta)))


def bboxes_pill_ann(
    x,
    y,
    w_in: float,
    a_in: float,
    b_in: float,
    w_out: float,
    a_out: float,
    b_out: float,
    theta_in: float = 0.0,
    theta_out: float | None = None,
    *,
    validate: bool = True,
):
    """Return pill-annulus bounding boxes for one or many centers."""
    if validate:
        _validate_pill_ann(w_in, a_in, b_in, w_out, a_out, b_out)
    xs, ys = _rust_positions(x, y, validate=validate)
    return _boxes_from_tuple(
        aapr.bboxes_pill_ann(
            xs,
            ys,
            *_floatify(w_in, a_in, b_in, w_out, a_out, b_out),
            *_theta_pair(theta_in, theta_out, validate=validate),
        )
    )


def _path_kernel_result(x, y, result, *, return_npix: bool):
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    if not scalar:
        return result
    if not return_npix:
        return np.asarray(result, dtype=np.float64).reshape(())
    apsum, npix = result
    return np.asarray(apsum, dtype=np.float64).reshape(()), np.asarray(
        npix, dtype=np.float64
    ).reshape(())


def _path_arrays(segments_or_kinds, holes_or_data=None):
    if isinstance(segments_or_kinds, np.ndarray) and segments_or_kinds.dtype == np.int8:
        return segments_or_kinds, holes_or_data
    return _encode_path_commands(segments_or_kinds, holes_or_data)


def weights_path_exact(
    x, y, segments_or_kinds, holes_or_data=None, *, validate: bool = True
):
    """Return exact path aperture weights for one or many centers.

    Path geometry is limited to closed contours made from ``"line"`` and
    circular ``"arc"`` commands. Bezier curves, splines, elliptical arcs, and
    Python callback masks are not supported.

    Accepts either ``(x, y, segments, holes=None)`` with Python command lists,
    or ``(x, y, kinds_array, data_array)`` with pre-encoded compact arrays.
    """
    kinds, data_arr = _path_arrays(segments_or_kinds, holes_or_data)
    return _weights_many(
        aapr.weights_path_exact, x, y, kinds, data_arr, validate=validate
    )


def weights_path_center(
    x, y, segments_or_kinds, holes_or_data=None, *, validate: bool = True
):
    """Return center-selected path aperture weights for one or many centers.

    Path geometry is limited to closed contours made from ``"line"`` and
    circular ``"arc"`` commands.
    """
    kinds, data_arr = _path_arrays(segments_or_kinds, holes_or_data)
    return _weights_many(
        aapr.weights_path_center, x, y, kinds, data_arr, validate=validate
    )


def apsum_path_exact(
    data,
    x,
    y,
    segments,
    holes=None,
    *,
    mask=None,
    return_npix: bool = True,
    validate: bool = True,
):
    """Return exact path aperture sums for one or many centers.

    Exact mode is analytic for the represented straight-line and circular-arc
    path. Other curve types are unsupported unless approximated before calling
    this function.
    """
    kinds, data_arr = _path_arrays(segments, holes)
    result = _weight_apsum(
        weights_path_exact,
        data,
        mask,
        x,
        y,
        kinds,
        data_arr,
        return_npix=return_npix,
        validate=validate,
    )
    return _path_kernel_result(x, y, result, return_npix=return_npix)


def apsum_path_center(
    data,
    x,
    y,
    segments,
    holes=None,
    *,
    mask=None,
    return_npix: bool = True,
    validate: bool = True,
):
    """Return center-selected path aperture sums for one or many centers.

    Path geometry is limited to closed contours made from ``"line"`` and
    circular ``"arc"`` commands.
    """
    kinds, data_arr = _path_arrays(segments, holes)
    result = _weight_apsum(
        weights_path_center,
        data,
        mask,
        x,
        y,
        kinds,
        data_arr,
        return_npix=return_npix,
        validate=validate,
    )
    return _path_kernel_result(x, y, result, return_npix=return_npix)


def npix_path_exact(
    x,
    y,
    segments,
    holes=None,
    *,
    shape: tuple[int, int],
    mask=None,
    validate: bool = True,
):
    """Return exact path aperture effective pixel counts.

    Exact mode is analytic for straight-line and circular-arc paths only.
    """
    kinds, data_arr = _path_arrays(segments, holes)
    result = _npix_from_weights(
        weights_path_exact,
        x,
        y,
        kinds,
        data_arr,
        shape=shape,
        mask=mask,
        validate=validate,
    )
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        return np.asarray(result, dtype=np.float64).reshape(())
    return result


def npix_path_center(
    x,
    y,
    segments,
    holes=None,
    *,
    shape: tuple[int, int],
    mask=None,
    validate: bool = True,
):
    """Return center-selected path aperture effective pixel counts.

    Path geometry is limited to straight line segments and circular arcs.
    """
    kinds, data_arr = _path_arrays(segments, holes)
    result = _npix_from_weights(
        weights_path_center,
        x,
        y,
        kinds,
        data_arr,
        shape=shape,
        mask=mask,
        validate=validate,
    )
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        return np.asarray(result, dtype=np.float64).reshape(())
    return result


def bboxes_path(x, y, segments, holes=None, *, validate: bool = True):
    """Return path aperture bounding boxes for one or many centers.

    Path geometry is limited to straight line segments and circular arcs.
    """
    kinds, data_arr = _path_arrays(segments, holes)
    xs, ys = _rust_positions(x, y, validate=validate)
    return _boxes_from_tuple(aapr.bboxes_path(xs, ys, kinds, data_arr))


def bboxes_wedge(
    x,
    y,
    r_in: float,
    r_out: float,
    theta_in: float,
    dtheta_in: float,
    theta_out: float | None = None,
    dtheta_out: float | None = None,
    *,
    validate: bool = True,
):
    """Return wedge aperture bounding boxes for one or many centers."""
    outer_angles = _wedge_outer_pair(theta_in, dtheta_in, theta_out, dtheta_out)
    if validate:
        _validate_wedge(r_in, r_out, theta_in, dtheta_in, *outer_angles)
    xs, ys = _rust_positions(x, y, validate=validate)
    return _boxes_from_tuple(
        aapr.bboxes_wedge(
            xs,
            ys,
            *_floatify(r_in, r_out, theta_in, dtheta_in),
            *outer_angles,
        )
    )


def get_parallel_threshold() -> int:
    """Return the aperture count where Rust kernels switch to Rayon."""
    return int(aapr.get_parallel_threshold())


def set_parallel_threshold(threshold: int) -> None:
    """Set the aperture count where Rust kernels switch to Rayon."""
    aapr.set_parallel_threshold(int(threshold))


apply_kernel_docstrings(globals())

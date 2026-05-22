"""Public direct aperture-sum kernels and Rust wrappers.

This module is the public namespace for direct catalog aperture-sum functions
such as ``astroapers.kernels.apsum_circ_exact``. The same functions are also
exported at top level as ``astroapers.apsum_*``. Functions with fused Rust
aperture-sum kernels use that fastest path for unmasked calls; the complete
surface for pill and split-angle annuli uses Rust-generated bbox-tight weights
plus the same ``BoundingBox`` reduction used by object methods. Calls with a
boolean mask route through bbox-tight aperture weights so bad pixels are
excluded from both ``apsum`` and ``npix``.
"""

from __future__ import annotations

import numpy as np

from . import _rust
from ._containers import BoundingBox, validate_mask

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


def apsum_circ_exact(
    data, x, y, r: float, *, mask=None, return_npix: bool = True, validate=True
):
    if mask is not None:
        return _masked_apsum(
            weights_circ_exact,
            data,
            mask,
            x,
            y,
            float(r),
            return_npix=return_npix,
            validate=validate,
        )
    return _apsum(
        "apsum_circ_exact",
        data,
        x,
        y,
        float(r),
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
    if mask is not None:
        return _masked_apsum(
            weights_circ_ann_exact,
            data,
            mask,
            x,
            y,
            float(r_in),
            float(r_out),
            return_npix=return_npix,
            validate=validate,
        )
    return _apsum(
        "apsum_circ_ann_exact",
        data,
        x,
        y,
        float(r_in),
        float(r_out),
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
        float(r_in),
        float(r_out),
        return_npix=return_npix,
        validate=validate,
    )


def apsum_circ_center(
    data, x, y, r: float, *, mask=None, return_npix: bool = True, validate=True
):
    if mask is not None:
        return _masked_apsum(
            weights_circ_center,
            data,
            mask,
            x,
            y,
            float(r),
            return_npix=return_npix,
            validate=validate,
        )
    return _apsum(
        "apsum_circ_center",
        data,
        x,
        y,
        float(r),
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
    if mask is not None:
        return _masked_apsum(
            weights_ellip_exact,
            data,
            mask,
            x,
            y,
            float(a),
            float(b),
            float(theta),
            return_npix=return_npix,
            validate=validate,
        )
    return _apsum(
        "apsum_ellip_exact",
        data,
        x,
        y,
        float(a),
        float(b),
        float(theta),
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
    if mask is not None:
        return _masked_apsum(
            weights_ellip_center,
            data,
            mask,
            x,
            y,
            float(a),
            float(b),
            float(theta),
            return_npix=return_npix,
            validate=validate,
        )
    return _apsum(
        "apsum_ellip_center",
        data,
        x,
        y,
        float(a),
        float(b),
        float(theta),
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
        float(a_in),
        float(b_in),
        float(a_out),
        float(b_out),
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
        float(a_in),
        float(b_in),
        float(a_out),
        float(b_out),
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
    if mask is not None:
        return _masked_apsum(
            weights_rect_exact,
            data,
            mask,
            x,
            y,
            float(w),
            float(h),
            float(theta),
            return_npix=return_npix,
            validate=validate,
        )
    return _apsum(
        "apsum_rect_exact",
        data,
        x,
        y,
        float(w),
        float(h),
        float(theta),
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
        float(w_in),
        float(h_in),
        float(w_out),
        float(h_out),
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
        float(w_in),
        float(h_in),
        float(w_out),
        float(h_out),
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
    if mask is not None:
        return _masked_apsum(
            weights_rect_center,
            data,
            mask,
            x,
            y,
            float(w),
            float(h),
            float(theta),
            return_npix=return_npix,
            validate=validate,
        )
    return _apsum(
        "apsum_rect_center",
        data,
        x,
        y,
        float(w),
        float(h),
        float(theta),
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
        float(w),
        float(a),
        float(b),
        float(theta),
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
        float(w),
        float(a),
        float(b),
        float(theta),
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
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_pill_ann(w_in, a_in, b_in, w_out, a_out, b_out)
    return _weight_apsum(
        weights_pill_ann_exact,
        data,
        mask,
        x,
        y,
        float(w_in),
        float(a_in),
        float(b_in),
        float(w_out),
        float(a_out),
        float(b_out),
        theta_in,
        theta_out,
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
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_pill_ann(w_in, a_in, b_in, w_out, a_out, b_out)
    return _weight_apsum(
        weights_pill_ann_center,
        data,
        mask,
        x,
        y,
        float(w_in),
        float(a_in),
        float(b_in),
        float(w_out),
        float(a_out),
        float(b_out),
        theta_in,
        theta_out,
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
    theta_out, dtheta_out = _wedge_outer_pair(
        theta_in, dtheta_in, theta_out, dtheta_out
    )
    if validate:
        _validate_wedge(r_in, r_out, theta_in, dtheta_in, theta_out, dtheta_out)
    return _weight_apsum(
        weights_wedge_exact,
        data,
        mask,
        x,
        y,
        float(r_in),
        float(r_out),
        float(theta_in),
        float(dtheta_in),
        theta_out,
        dtheta_out,
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
    theta_out, dtheta_out = _wedge_outer_pair(
        theta_in, dtheta_in, theta_out, dtheta_out
    )
    if validate:
        _validate_wedge(r_in, r_out, theta_in, dtheta_in, theta_out, dtheta_out)
    return _weight_apsum(
        weights_wedge_center,
        data,
        mask,
        x,
        y,
        float(r_in),
        float(r_out),
        float(theta_in),
        float(dtheta_in),
        theta_out,
        dtheta_out,
        return_npix=return_npix,
        validate=validate,
    )


def npix_circ_exact(
    x, y, r: float, *, shape: tuple[int, int], mask=None, validate=True
):
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
    npix = _rust.npix_circ_exact(xs.reshape(-1), ys.reshape(-1), float(r), ny, nx)
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
    npix = _rust.npix_circ_center(xs.reshape(-1), ys.reshape(-1), float(r), ny, nx)
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
    if mask is not None:
        return _npix_from_weights(
            weights_ellip_exact,
            x,
            y,
            float(a),
            float(b),
            float(theta),
            shape=shape,
            mask=mask,
            validate=validate,
        )
    xs, ys = _positions(x, y, validate=validate)
    ny, nx = _shape2(shape, validate=validate)
    npix = _rust.npix_ellip_exact(
        xs.reshape(-1), ys.reshape(-1), float(a), float(b), float(theta), ny, nx
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
    if mask is not None:
        return _npix_from_weights(
            weights_ellip_center,
            x,
            y,
            float(a),
            float(b),
            float(theta),
            shape=shape,
            mask=mask,
            validate=validate,
        )
    xs, ys = _positions(x, y, validate=validate)
    ny, nx = _shape2(shape, validate=validate)
    npix = _rust.npix_ellip_center(
        xs.reshape(-1), ys.reshape(-1), float(a), float(b), float(theta), ny, nx
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
        float(a_in),
        float(b_in),
        float(a_out),
        float(b_out),
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
        float(a_in),
        float(b_in),
        float(a_out),
        float(b_out),
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
    if mask is not None:
        return _npix_from_weights(
            weights_rect_exact,
            x,
            y,
            float(w),
            float(h),
            float(theta),
            shape=shape,
            mask=mask,
            validate=validate,
        )
    xs, ys = _positions(x, y, validate=validate)
    ny, nx = _shape2(shape, validate=validate)
    npix = _rust.npix_rect_exact(
        xs.reshape(-1), ys.reshape(-1), float(w), float(h), float(theta), ny, nx
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
    if mask is not None:
        return _npix_from_weights(
            weights_rect_center,
            x,
            y,
            float(w),
            float(h),
            float(theta),
            shape=shape,
            mask=mask,
            validate=validate,
        )
    xs, ys = _positions(x, y, validate=validate)
    ny, nx = _shape2(shape, validate=validate)
    npix = _rust.npix_rect_center(
        xs.reshape(-1), ys.reshape(-1), float(w), float(h), float(theta), ny, nx
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
        float(w_in),
        float(h_in),
        float(w_out),
        float(h_out),
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
        float(w_in),
        float(h_in),
        float(w_out),
        float(h_out),
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
        float(w),
        float(a),
        float(b),
        float(theta),
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
        float(w),
        float(a),
        float(b),
        float(theta),
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
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_pill_ann(w_in, a_in, b_in, w_out, a_out, b_out)
    return _npix_from_weights(
        weights_pill_ann_exact,
        x,
        y,
        float(w_in),
        float(a_in),
        float(b_in),
        float(w_out),
        float(a_out),
        float(b_out),
        theta_in,
        theta_out,
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
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_pill_ann(w_in, a_in, b_in, w_out, a_out, b_out)
    return _npix_from_weights(
        weights_pill_ann_center,
        x,
        y,
        float(w_in),
        float(a_in),
        float(b_in),
        float(w_out),
        float(a_out),
        float(b_out),
        theta_in,
        theta_out,
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
    theta_out, dtheta_out = _wedge_outer_pair(
        theta_in, dtheta_in, theta_out, dtheta_out
    )
    if validate:
        _validate_wedge(r_in, r_out, theta_in, dtheta_in, theta_out, dtheta_out)
    return _npix_from_weights(
        weights_wedge_exact,
        x,
        y,
        float(r_in),
        float(r_out),
        float(theta_in),
        float(dtheta_in),
        theta_out,
        dtheta_out,
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
    theta_out, dtheta_out = _wedge_outer_pair(
        theta_in, dtheta_in, theta_out, dtheta_out
    )
    if validate:
        _validate_wedge(r_in, r_out, theta_in, dtheta_in, theta_out, dtheta_out)
    return _npix_from_weights(
        weights_wedge_center,
        x,
        y,
        float(r_in),
        float(r_out),
        float(theta_in),
        float(dtheta_in),
        theta_out,
        dtheta_out,
        shape=shape,
        mask=mask,
        validate=validate,
    )


def weights_exact(
    symbol: str, bbox: BoundingBox, params: tuple[float, ...]
) -> np.ndarray:
    """Return exact aperture weights for one bbox-tight bounding box."""
    func = getattr(_rust, symbol)
    data = func(
        *params,
        bbox.ixmin,
        bbox.iymin,
        bbox.shape[0],
        bbox.shape[1],
    )
    return np.asarray(data, dtype=np.float64).reshape(bbox.shape)


def weights_center(
    symbol: str, bbox: BoundingBox, params: tuple[float, ...]
) -> np.ndarray:
    """Return center-selected aperture weights for one bbox-tight bounding box."""
    func = getattr(_rust, symbol)
    data = func(
        *params,
        bbox.ixmin,
        bbox.iymin,
        bbox.shape[0],
        bbox.shape[1],
    )
    return np.asarray(data, dtype=np.float64).reshape(bbox.shape)


def _weights_many(
    symbol: str, x, y, *params, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    xs, ys = _positions(x, y, validate=validate)
    weights, ixmins, ixmaxs, iymins, iymaxs = getattr(_rust, symbol)(
        xs.reshape(-1), ys.reshape(-1), *params
    )
    boxes = _boxes_from_tuple((ixmins, ixmaxs, iymins, iymaxs))
    arrays = [
        np.asarray(weight, dtype=np.float64).reshape(box.shape)
        for weight, box in zip(weights, boxes)
    ]
    return arrays, boxes


def weights_circ_exact(
    x, y, r: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many("weights_circ_exact_many", x, y, float(r), validate=validate)


def weights_circ_center(
    x, y, r: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many("weights_circ_center_many", x, y, float(r), validate=validate)


def weights_circ_ann_exact(
    x, y, r_in: float, r_out: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        "weights_circ_ann_exact_many",
        x,
        y,
        float(r_in),
        float(r_out),
        validate=validate,
    )


def weights_circ_ann_center(
    x, y, r_in: float, r_out: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        "weights_circ_ann_center_many",
        x,
        y,
        float(r_in),
        float(r_out),
        validate=validate,
    )


def weights_ellip_exact(
    x, y, a: float, b: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        "weights_ellip_exact_many",
        x,
        y,
        float(a),
        float(b),
        float(theta),
        validate=validate,
    )


def weights_ellip_center(
    x, y, a: float, b: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        "weights_ellip_center_many",
        x,
        y,
        float(a),
        float(b),
        float(theta),
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
        "weights_ellip_ann_exact_many",
        x,
        y,
        float(a_in),
        float(b_in),
        float(a_out),
        float(b_out),
        float(theta_in),
        float(theta_out),
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
        "weights_ellip_ann_center_many",
        x,
        y,
        float(a_in),
        float(b_in),
        float(a_out),
        float(b_out),
        float(theta_in),
        float(theta_out),
        validate=validate,
    )


def weights_rect_exact(
    x, y, w: float, h: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        "weights_rect_exact_many",
        x,
        y,
        float(w),
        float(h),
        float(theta),
        validate=validate,
    )


def weights_rect_center(
    x, y, w: float, h: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        "weights_rect_center_many",
        x,
        y,
        float(w),
        float(h),
        float(theta),
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
        "weights_rect_ann_exact_many",
        x,
        y,
        float(w_in),
        float(h_in),
        float(w_out),
        float(h_out),
        float(theta_in),
        float(theta_out),
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
        "weights_rect_ann_center_many",
        x,
        y,
        float(w_in),
        float(h_in),
        float(w_out),
        float(h_out),
        float(theta_in),
        float(theta_out),
        validate=validate,
    )


def weights_pill_exact(
    x, y, w: float, a: float, b: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        "weights_pill_exact_many",
        x,
        y,
        float(w),
        float(a),
        float(b),
        float(theta),
        validate=validate,
    )


def weights_pill_center(
    x, y, w: float, a: float, b: float, theta: float, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        "weights_pill_center_many",
        x,
        y,
        float(w),
        float(a),
        float(b),
        float(theta),
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
        "weights_pill_ann_exact_many",
        x,
        y,
        float(w_in),
        float(a_in),
        float(b_in),
        float(w_out),
        float(a_out),
        float(b_out),
        float(theta_in),
        float(theta_out),
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
        "weights_pill_ann_center_many",
        x,
        y,
        float(w_in),
        float(a_in),
        float(b_in),
        float(w_out),
        float(a_out),
        float(b_out),
        float(theta_in),
        float(theta_out),
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
        "weights_wedge_exact_many",
        x,
        y,
        float(r_in),
        float(r_out),
        float(theta_in),
        float(dtheta_in),
        float(theta_out),
        float(dtheta_out),
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
        "weights_wedge_center_many",
        x,
        y,
        float(r_in),
        float(r_out),
        float(theta_in),
        float(dtheta_in),
        float(theta_out),
        float(dtheta_out),
        validate=validate,
    )


def bboxes_circ(x, y, r: float, *, validate: bool = True):
    """Return exact circular aperture bounding boxes for one or many centers."""
    xs, ys = _positions(x, y, validate=validate)
    return _boxes_from_tuple(
        _rust.bboxes_circ_many(xs.reshape(-1), ys.reshape(-1), float(r))
    )


def bboxes_circ_ann(x, y, r_in: float, r_out: float, *, validate: bool = True):
    """Return circular-annulus bounding boxes for one or many centers."""
    if validate:
        _validate_circ_ann_radii(r_in, r_out)
    return bboxes_circ(x, y, float(r_out), validate=validate)


def bboxes_ellip(x, y, a: float, b: float, theta: float, *, validate: bool = True):
    """Return exact elliptical aperture bounding boxes for one or many centers."""
    xs, ys = _positions(x, y, validate=validate)
    return _boxes_from_tuple(
        _rust.bboxes_ellip_many(
            xs.reshape(-1), ys.reshape(-1), float(a), float(b), float(theta)
        )
    )


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
    _, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_inner_outer_axes(a_in, b_in, a_out, b_out, "ellipse")
    return bboxes_ellip(x, y, float(a_out), float(b_out), theta_out, validate=validate)


def bboxes_rect(x, y, w: float, h: float, theta: float, *, validate: bool = True):
    """Return exact rotated-rectangle aperture bounding boxes for one or many centers."""
    xs, ys = _positions(x, y, validate=validate)
    return _boxes_from_tuple(
        _rust.bboxes_rect_many(
            xs.reshape(-1), ys.reshape(-1), float(w), float(h), float(theta)
        )
    )


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
    _, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_inner_outer_axes(w_in, h_in, w_out, h_out, "rectangle")
    return bboxes_rect(x, y, float(w_out), float(h_out), theta_out, validate=validate)


def bboxes_pill(
    x, y, w: float, a: float, b: float, theta: float = 0.0, *, validate: bool = True
):
    """Return pill aperture bounding boxes for one or many centers."""
    xs, ys = _positions(x, y, validate=validate)
    return _boxes_from_tuple(
        _rust.bboxes_pill_many(
            xs.reshape(-1), ys.reshape(-1), float(w), float(a), float(b), float(theta)
        )
    )


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
    theta_in, theta_out = _theta_pair(theta_in, theta_out, validate=validate)
    if validate:
        _validate_pill_ann(w_in, a_in, b_in, w_out, a_out, b_out)
    xs, ys = _positions(x, y, validate=validate)
    return _boxes_from_tuple(
        _rust.bboxes_pill_ann_many(
            xs.reshape(-1),
            ys.reshape(-1),
            float(w_in),
            float(a_in),
            float(b_in),
            float(w_out),
            float(a_out),
            float(b_out),
            theta_in,
            theta_out,
        )
    )


def _encode_path(segments, holes):
    from ._path import _encode_path_commands

    return _encode_path_commands(segments, holes)


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
    if isinstance(segments_or_kinds, np.ndarray) and segments_or_kinds.dtype == np.int8:
        kinds, data_arr = segments_or_kinds, holes_or_data
    else:
        kinds, data_arr = _encode_path(segments_or_kinds, holes_or_data)
    return _weights_many(
        "weights_path_exact_many", x, y, kinds, data_arr, validate=validate
    )


def weights_path_center(
    x, y, segments_or_kinds, holes_or_data=None, *, validate: bool = True
):
    """Return center-selected path aperture weights for one or many centers.

    Path geometry is limited to closed contours made from ``"line"`` and
    circular ``"arc"`` commands.
    """
    if isinstance(segments_or_kinds, np.ndarray) and segments_or_kinds.dtype == np.int8:
        kinds, data_arr = segments_or_kinds, holes_or_data
    else:
        kinds, data_arr = _encode_path(segments_or_kinds, holes_or_data)
    return _weights_many(
        "weights_path_center_many", x, y, kinds, data_arr, validate=validate
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
    kinds, data_arr = _encode_path(segments, holes)
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
    kinds, data_arr = _encode_path(segments, holes)
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
    kinds, data_arr = _encode_path(segments, holes)
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
    kinds, data_arr = _encode_path(segments, holes)
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
    kinds, data_arr = _encode_path(segments, holes)
    xs, ys = _positions(x, y, validate=validate)
    return _boxes_from_tuple(
        _rust.bboxes_path_many(xs.reshape(-1), ys.reshape(-1), kinds, data_arr)
    )


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
    theta_out, dtheta_out = _wedge_outer_pair(
        theta_in, dtheta_in, theta_out, dtheta_out
    )
    if validate:
        _validate_wedge(r_in, r_out, theta_in, dtheta_in, theta_out, dtheta_out)
    xs, ys = _positions(x, y, validate=validate)
    return _boxes_from_tuple(
        _rust.bboxes_wedge_many(
            xs.reshape(-1),
            ys.reshape(-1),
            float(r_in),
            float(r_out),
            float(theta_in),
            float(dtheta_in),
            theta_out,
            dtheta_out,
        )
    )


def get_parallel_threshold() -> int:
    """Return the aperture count where Rust kernels switch to Rayon."""
    return int(_rust.get_parallel_threshold())


def set_parallel_threshold(threshold: int) -> None:
    """Set the aperture count where Rust kernels switch to Rayon."""
    _rust.set_parallel_threshold(int(threshold))


def _apsum(symbol: str, data, x, y, *params, return_npix: bool, validate: bool = True):
    arr = np.asarray(data)
    if validate and arr.ndim != 2:
        raise ValueError("data must be a 2-D array")
    sum_funcs = None if return_npix else _dtype_functions(f"{symbol}_sum", arr.dtype)
    funcs = _dtype_functions(symbol, arr.dtype)
    if funcs is not None:
        arr = np.ascontiguousarray(arr)
        func_one, func_many = funcs
    else:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        func_one = getattr(_rust, f"{symbol}_one", None)
        func_many = getattr(_rust, symbol)
        if not return_npix:
            sum_one = getattr(_rust, f"{symbol}_sum_one", None)
            sum_many = getattr(_rust, f"{symbol}_sum", None)
            if sum_many is not None:
                sum_funcs = (sum_one, sum_many)
    xs, ys = _positions(x, y, validate=validate)
    if not return_npix and sum_funcs is not None:
        sum_one, sum_many = sum_funcs
        if xs.size == 1 and sum_one is not None:
            apsum = sum_one(
                arr, float(xs.reshape(-1)[0]), float(ys.reshape(-1)[0]), *params
            )
            return np.asarray([apsum], dtype=np.float64).reshape(xs.shape)
        apsum = sum_many(arr, xs.reshape(-1), ys.reshape(-1), *params)
        return np.asarray(apsum, dtype=np.float64).reshape(xs.shape)
    if xs.size == 1 and func_one is not None:
        apsum, npix = func_one(
            arr, float(xs.reshape(-1)[0]), float(ys.reshape(-1)[0]), *params
        )
        apsum = np.asarray([apsum], dtype=np.float64).reshape(xs.shape)
        npix = np.asarray([npix], dtype=np.float64).reshape(xs.shape)
        return (apsum, npix) if return_npix else apsum
    apsum, npix = func_many(arr, xs.reshape(-1), ys.reshape(-1), *params)
    apsum = np.asarray(apsum, dtype=np.float64).reshape(xs.shape)
    npix = np.asarray(npix, dtype=np.float64).reshape(xs.shape)
    return (apsum, npix) if return_npix else apsum


def _weight_apsum(
    weights_func, data, mask, x, y, *params, return_npix: bool, validate: bool = True
):
    return _masked_apsum(
        weights_func,
        data,
        mask,
        x,
        y,
        *params,
        return_npix=return_npix,
        validate=validate,
    )


def _masked_apsum(
    weights_func, data, mask, x, y, *params, return_npix: bool, validate: bool = True
):
    arr = np.asarray(data)
    if validate and arr.ndim != 2:
        raise ValueError("data must be a 2-D array")
    bad = (
        None if mask is None else (validate_mask(mask, arr.shape) if validate else mask)
    )
    xs, ys = _positions(x, y, validate=validate)
    flat_x = xs.reshape(-1)
    flat_y = ys.reshape(-1)
    flat_apsum = np.full(flat_x.shape, np.nan, dtype=np.float64)
    flat_npix = np.full(flat_x.shape, np.nan, dtype=np.float64)
    if validate:
        finite = np.isfinite(flat_x) & np.isfinite(flat_y)
        if not np.any(finite):
            apsum = flat_apsum.reshape(xs.shape)
            if not return_npix:
                return apsum
            return apsum, flat_npix.reshape(xs.shape)
        weights, boxes = weights_func(
            flat_x[finite], flat_y[finite], *params, validate=validate
        )
        finite_indices = np.nonzero(finite)[0]
        for idx, weight, bbox in zip(finite_indices, weights, boxes, strict=True):
            if return_npix:
                flat_apsum[idx], flat_npix[idx] = bbox.apsum(
                    weight, arr, mask=bad, return_npix=True, validate=validate
                )
            else:
                flat_apsum[idx] = bbox.apsum(
                    weight, arr, mask=bad, return_npix=False, validate=validate
                )
    else:
        weights, boxes = weights_func(flat_x, flat_y, *params, validate=False)
        for idx, (weight, bbox) in enumerate(zip(weights, boxes, strict=True)):
            if return_npix:
                flat_apsum[idx], flat_npix[idx] = bbox.apsum(
                    weight, arr, mask=bad, return_npix=True, validate=False
                )
            else:
                flat_apsum[idx] = bbox.apsum(
                    weight, arr, mask=bad, return_npix=False, validate=False
                )
    apsum = flat_apsum.reshape(xs.shape)
    if not return_npix:
        return apsum
    return apsum, flat_npix.reshape(xs.shape)


def _npix_from_weights(
    weights_func,
    x,
    y,
    *params,
    shape: tuple[int, int],
    mask=None,
    validate: bool = True,
):
    shape = _shape2(shape, validate=validate)
    bad = None if mask is None else (validate_mask(mask, shape) if validate else mask)
    xs, ys = _positions(x, y, validate=validate)
    flat_x = xs.reshape(-1)
    flat_y = ys.reshape(-1)
    flat_npix = np.full(flat_x.shape, np.nan, dtype=np.float64)
    if validate:
        finite = np.isfinite(flat_x) & np.isfinite(flat_y)
        if not np.any(finite):
            return flat_npix.reshape(xs.shape)
        weights, boxes = weights_func(
            flat_x[finite], flat_y[finite], *params, validate=validate
        )
        finite_indices = np.nonzero(finite)[0]
        for idx, weight, bbox in zip(finite_indices, weights, boxes, strict=True):
            flat_npix[idx] = bbox.npix(weight, shape, mask=bad, validate=validate)
    else:
        weights, boxes = weights_func(flat_x, flat_y, *params, validate=False)
        for idx, (weight, bbox) in enumerate(zip(weights, boxes, strict=True)):
            flat_npix[idx] = bbox.npix(weight, shape, mask=bad, validate=False)
    return flat_npix.reshape(xs.shape)


def _subtract_apsum(outer, inner, *, return_npix: bool):
    if not return_npix:
        return outer - inner
    outer_sum, outer_npix = outer
    inner_sum, inner_npix = inner
    return outer_sum - inner_sum, outer_npix - inner_npix


def _positions(x, y, *, validate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if not validate:
        return x, y
    xs = np.ascontiguousarray(np.atleast_1d(x), dtype=np.float64)
    ys = np.ascontiguousarray(np.atleast_1d(y), dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError("x and y must have matching shapes")
    return xs, ys


def _validate_circ_ann_radii(r_in: float, r_out: float) -> tuple[float, float]:
    r_in = float(r_in)
    r_out = float(r_out)
    if not np.isfinite(r_in) or r_in < 0.0:
        raise ValueError("r_in must be a nonnegative finite scalar")
    if not np.isfinite(r_out) or r_out <= 0.0:
        raise ValueError("r_out must be a positive finite scalar")
    if r_in >= r_out:
        raise ValueError("r_in must be smaller than r_out")
    return r_in, r_out


def _theta_pair(
    theta_in: float, theta_out: float | None, *, validate: bool = True
) -> tuple[float, float]:
    theta_in = float(theta_in)
    theta_out = theta_in if theta_out is None else float(theta_out)
    if validate and (not np.isfinite(theta_in) or not np.isfinite(theta_out)):
        raise ValueError("theta values must be finite")
    return theta_in, theta_out


def _wedge_outer_pair(
    theta_in: float,
    dtheta_in: float,
    theta_out: float | None,
    dtheta_out: float | None,
) -> tuple[float, float]:
    theta_out = float(theta_in) if theta_out is None else float(theta_out)
    dtheta_out = float(dtheta_in) if dtheta_out is None else float(dtheta_out)
    return theta_out, dtheta_out


def _validate_wedge(
    r_in: float,
    r_out: float,
    theta_in: float,
    dtheta_in: float,
    theta_out: float,
    dtheta_out: float,
) -> tuple[float, float, float, float, float, float]:
    r_in = float(r_in)
    r_out = float(r_out)
    theta_in = float(theta_in)
    dtheta_in = float(dtheta_in)
    theta_out = float(theta_out)
    dtheta_out = float(dtheta_out)
    if not np.isfinite(r_in) or r_in <= 0.0:
        raise ValueError("r_in must be a positive finite scalar")
    if not np.isfinite(r_out) or r_out <= 0.0:
        raise ValueError("r_out must be a positive finite scalar")
    if r_in >= r_out:
        raise ValueError("r_in must be smaller than r_out")
    if not np.isfinite(theta_in) or not np.isfinite(theta_out):
        raise ValueError("theta values must be finite")
    if (
        not np.isfinite(dtheta_in)
        or not np.isfinite(dtheta_out)
        or dtheta_in <= 0.0
        or dtheta_out <= 0.0
        or dtheta_in >= 2.0 * np.pi
        or dtheta_out >= 2.0 * np.pi
    ):
        raise ValueError("dtheta values must be finite scalars in (0, 2*pi)")
    return r_in, r_out, theta_in, dtheta_in, theta_out, dtheta_out


def _validate_inner_outer_axes(
    in0: float, in1: float, out0: float, out1: float, shape_name: str
) -> tuple[float, float, float, float]:
    in0 = float(in0)
    in1 = float(in1)
    out0 = float(out0)
    out1 = float(out1)
    values = (in0, in1, out0, out1)
    if any((not np.isfinite(value) or value <= 0.0) for value in values):
        raise ValueError(f"{shape_name} dimensions must be positive finite scalars")
    if in0 >= out0 or in1 >= out1:
        raise ValueError(
            f"inner {shape_name} dimensions must be smaller than outer dimensions"
        )
    return in0, in1, out0, out1


def _validate_pill_ann(
    w_in: float,
    a_in: float,
    b_in: float,
    w_out: float,
    a_out: float,
    b_out: float,
) -> tuple[float, float, float, float, float, float]:
    w_in = float(w_in)
    a_in = float(a_in)
    b_in = float(b_in)
    w_out = float(w_out)
    a_out = float(a_out)
    b_out = float(b_out)
    values = (w_in, a_in, b_in, w_out, a_out, b_out)
    if any((not np.isfinite(value) or value <= 0.0) for value in values):
        raise ValueError("pill dimensions must be positive finite scalars")
    if w_in >= w_out or a_in >= a_out or b_in >= b_out:
        raise ValueError("inner pill dimensions must be smaller than outer dimensions")
    return w_in, a_in, b_in, w_out, a_out, b_out


def _boxes_from_tuple(raw) -> list[BoundingBox]:
    ixmins, ixmaxs, iymins, iymaxs = raw
    return [
        BoundingBox(int(ixmin), int(ixmax), int(iymin), int(iymax))
        for ixmin, ixmax, iymin, iymax in zip(ixmins, ixmaxs, iymins, iymaxs)
    ]


def _shape2(shape: tuple[int, int], *, validate: bool = True) -> tuple[int, int]:
    if validate and len(shape) != 2:
        raise ValueError("shape must be a 2-tuple")
    ny, nx = int(shape[0]), int(shape[1])
    if validate and (ny <= 0 or nx <= 0):
        raise ValueError("shape dimensions must be positive")
    return ny, nx


def _dtype_functions(symbol: str, dtype) -> tuple[object | None, object] | None:
    suffix = _DTYPE_SUFFIX.get(np.dtype(dtype))
    if suffix is None:
        return None
    one = getattr(_rust, f"{symbol}_one{suffix}", None)
    many = getattr(_rust, f"{symbol}{suffix}", None)
    if many is None:
        return None
    return one, many


_DTYPE_SUFFIX = {
    np.dtype(np.float64): "",
    np.dtype(np.float32): "_f32",
    np.dtype(np.int32): "_i32",
    np.dtype(np.int16): "_i16",
}


def _apsum_doc(summary: str, geometry_params: str, notes: str) -> str:
    return f"""Return {summary} for one or many aperture centers.

Parameters
----------
data : array_like
    Two-dimensional image. For ``mask=None``, ``float64``, ``float32``,
    ``int32``, and ``int16`` arrays dispatch to dtype-specialized Rust
    kernels when available; other dtypes are converted to contiguous
    ``float64`` before summation. Returned aperture sums and effective pixel
    counts are always ``float64`` arrays.
x, y : scalar or array_like
    Aperture center coordinates in pixel units. Inputs are converted to
    contiguous ``float64`` arrays. Shapes must match after ``numpy.atleast_1d``.
    The return shape matches that broadcast-free input shape, so scalar inputs
    return one-element arrays.
{geometry_params}
mask : array_like of bool, optional
    Boolean image mask with the same shape as ``data``. ``True`` pixels are
    excluded from both the aperture sum and effective pixel count. Values are
    converted to boolean. When this is omitted, the unmasked Rust summation path
    is used.
return_npix : bool, optional
    If `True`, return ``(apsum, npix)``. If `False`, return only ``apsum`` and
    use the sum-only Rust kernel for supported unmasked calls.

Returns
-------
apsum : ndarray
    Sum of the unmasked data values multiplied by the aperture weights.
npix : ndarray
    Effective in-frame pixel count, returned only when ``return_npix=True``.
    This is the sum of aperture weights after image clipping and mask
    exclusion.

Notes
-----
{notes}
"""


def _npix_doc(summary: str, geometry_params: str, notes: str) -> str:
    return f"""Return {summary} effective in-frame pixel counts.

Parameters
----------
x, y : scalar or array_like
    Aperture center coordinates in pixel units. Shapes must match after
    ``numpy.atleast_1d``. The return shape matches that broadcast-free input
    shape, so scalar inputs return one-element arrays.
{geometry_params}
shape : tuple[int, int]
    Image shape as ``(ny, nx)`` used to clip the aperture footprint.
mask : array_like of bool, optional
    Boolean image mask with shape ``shape``. ``True`` pixels are excluded from
    the returned effective pixel count.

Returns
-------
npix : ndarray
    Effective pixel count inside the image frame.

Notes
-----
{notes}
"""


def _weights_doc(summary: str, geometry_params: str, notes: str) -> str:
    return f"""Return {summary} bbox-tight aperture weights.

Parameters
----------
x, y : scalar or array_like
    Aperture center coordinates in pixel units. Shapes must match after
    ``numpy.atleast_1d``.
{geometry_params}

Returns
-------
weights : list[ndarray]
    One floating weight array per aperture center. Each array is bbox-tight,
    so its shape usually differs from the full image shape.
boxes : list[BoundingBox]
    Pixel bounding boxes that place each weight array back into image
    coordinates.

Notes
-----
{notes}
"""


def _bboxes_doc(summary: str, geometry_params: str, notes: str) -> str:
    return f"""Return {summary} bounding boxes for one or many aperture centers.

Parameters
----------
x, y : scalar or array_like
    Aperture center coordinates in pixel units. Shapes must match after
    ``numpy.atleast_1d``.
{geometry_params}

Returns
-------
boxes : list[BoundingBox]
    Pixel bounding boxes with inclusive lower and exclusive upper bounds.

Notes
-----
{notes}
"""


_CIRC_PARAMS = """r : float
    Aperture radius in pixels."""

_CIRC_ANN_PARAMS = """r_in, r_out : float
    Inner and outer annulus radii in pixels. ``r_in`` must be nonnegative and
    smaller than ``r_out``."""

_ELLIP_PARAMS = """a, b : float
    Semimajor and semiminor axes in pixels.
theta : float, optional
    Rotation angle in radians, measured counterclockwise from the positive
    x axis."""

_ELLIP_ANN_PARAMS = """a_in, b_in, a_out, b_out : float
    Inner and outer ellipse semiaxes in pixels. Inner axes must be smaller
    than the corresponding outer axes.
theta_in : float, optional
    Inner ellipse rotation angle in radians.
theta_out : float or None, optional
    Outer ellipse rotation angle in radians. If `None`, uses ``theta_in``."""

_RECT_PARAMS = """w, h : float
    Full rectangle width and height in pixels. ``w`` is measured along the
    rectangle's local x axis, and ``h`` along its local y axis. At ``theta=0``,
    the width axis is aligned with the image x axis and the height axis is
    aligned with the image y axis.
theta : float, optional
    Rotation angle in radians, measured counterclockwise from the positive
    image x axis to the rectangle's local width axis."""

_RECT_ANN_PARAMS = """w_in, h_in : float
    Inner rectangle full width and height in pixels. ``w_in`` is measured along
    the inner rectangle's local x axis, and ``h_in`` along its local y axis.
w_out, h_out : float
    Outer rectangle full width and height in pixels. ``w_out`` is measured
    along the outer rectangle's local x axis, and ``h_out`` along its local
    y axis. Each outer dimension must be larger than the corresponding inner
    dimension. At ``theta_out=0``, the outer width axis is aligned with the
    image x axis and the outer height axis with the image y axis.
theta_in : float, optional
    Inner rectangle rotation angle in radians, measured counterclockwise from
    the positive image x axis to the inner rectangle's local width axis.
theta_out : float or None, optional
    Outer rectangle rotation angle in radians, measured counterclockwise from
    the positive image x axis to the outer rectangle's local width axis. If
    `None`, uses ``theta_in``."""

_PILL_PARAMS = """w, a, b : float
    Pill rectangle length and cap semiaxes in pixels.
theta : float, optional
    Rotation angle in radians, measured counterclockwise from the positive
    x axis."""

_PILL_ANN_PARAMS = """w_in, a_in, b_in, w_out, a_out, b_out : float
    Inner and outer pill dimensions in pixels. Inner dimensions must be
    smaller than the corresponding outer dimensions.
theta_in : float, optional
    Inner pill rotation angle in radians.
theta_out : float or None, optional
    Outer pill rotation angle in radians. If `None`, uses ``theta_in``."""

_WEDGE_PARAMS = """r_in, r_out : float
    Inner and outer wedge radii in pixels. ``r_out`` must be larger than
    ``r_in`` and ``r_in`` must be positive.
theta_in, dtheta_in : float
    Center angle and full angular width at ``r_in``, in radians.
theta_out, dtheta_out : float or None, optional
    Center angle and full angular width at ``r_out``. If either value is
    `None`, the corresponding inner value is used."""

_EXACT_NOTE = (
    "``exact`` mode uses fractional pixel overlap weights for the analytic "
    "aperture footprint."
)

_CENTER_NOTE = (
    "``center`` mode uses binary weights selected by pixel-center inclusion. "
    "The boundary convention matches Photutils for supported shapes: pixels "
    "exactly on the outer boundary are excluded, and annuli include the inner "
    "boundary while excluding the outer boundary."
)

_CIRC_ANN_EXACT_NOTE = (
    "The annulus is computed as exact outer-circle coverage minus exact "
    "inner-circle coverage after validating the radii."
)

_CIRC_ANN_CENTER_NOTE = (
    "The annulus uses the same center convention as "
    "``npix_circ_center(r_out) - npix_circ_center(r_in)``: the inner boundary "
    "is included and the outer boundary is excluded."
)

_ANN_EXACT_NOTE = (
    "For matched inner and outer angles, this uses outer-minus-inner direct "
    "aperture summation. For split-angle annuli, it uses Rust-generated bbox-tight "
    "annulus weights and the same ``BoundingBox.apsum`` reduction."
)

_ANN_CENTER_NOTE = (
    "For matched inner and outer angles, this uses outer-minus-inner direct "
    "center aperture summation. For split-angle annuli, it uses Rust-generated "
    "bbox-tight annulus weights and the same center-boundary convention as "
    "bbox-tight weights."
)

_PILL_EXACT_NOTE = (
    "Pill aperture summation currently uses Rust-generated bbox-tight pill weights "
    "and the same ``BoundingBox.apsum`` reduction."
)

_PILL_CENTER_NOTE = (
    "Pill center-mode aperture summation currently uses Rust-generated bbox-tight binary "
    "pill weights and the same ``BoundingBox.apsum`` reduction."
)

_WEDGE_EXACT_NOTE = (
    "``exact`` mode computes analytic fractional pixel overlap for annular "
    "wedges. Constant-width wedges use a circular-sector path; generalized "
    "wedges connect inner and outer arc endpoints with straight sides."
)

_WEDGE_CENTER_NOTE = (
    "``center`` mode uses binary weights selected by pixel-center inclusion "
    "inside the annular wedge."
)

apsum_circ_exact.__doc__ = _apsum_doc(
    "exact circular aperture sums",
    _CIRC_PARAMS,
    _EXACT_NOTE,
)
apsum_circ_ann_exact.__doc__ = _apsum_doc(
    "exact circular-annulus aperture sums",
    _CIRC_ANN_PARAMS,
    _CIRC_ANN_EXACT_NOTE,
)
apsum_circ_center.__doc__ = _apsum_doc(
    "center-selected circular aperture sums",
    _CIRC_PARAMS,
    _CENTER_NOTE,
)
apsum_circ_ann_center.__doc__ = _apsum_doc(
    "center-selected circular-annulus aperture sums",
    _CIRC_ANN_PARAMS,
    _CIRC_ANN_CENTER_NOTE,
)
apsum_ellip_exact.__doc__ = _apsum_doc(
    "exact elliptical aperture sums",
    _ELLIP_PARAMS,
    _EXACT_NOTE,
)
apsum_ellip_center.__doc__ = _apsum_doc(
    "center-selected elliptical aperture sums",
    _ELLIP_PARAMS,
    _CENTER_NOTE,
)
apsum_ellip_ann_exact.__doc__ = _apsum_doc(
    "exact elliptical-annulus aperture sums",
    _ELLIP_ANN_PARAMS,
    _ANN_EXACT_NOTE,
)
apsum_ellip_ann_center.__doc__ = _apsum_doc(
    "center-selected elliptical-annulus aperture sums",
    _ELLIP_ANN_PARAMS,
    _ANN_CENTER_NOTE,
)
apsum_rect_exact.__doc__ = _apsum_doc(
    "exact rotated-rectangle aperture sums",
    _RECT_PARAMS,
    _EXACT_NOTE,
)
apsum_rect_center.__doc__ = _apsum_doc(
    "center-selected rotated-rectangle aperture sums",
    _RECT_PARAMS,
    _CENTER_NOTE,
)
apsum_rect_ann_exact.__doc__ = _apsum_doc(
    "exact rotated-rectangle-annulus aperture sums",
    _RECT_ANN_PARAMS,
    _ANN_EXACT_NOTE,
)
apsum_rect_ann_center.__doc__ = _apsum_doc(
    "center-selected rotated-rectangle-annulus aperture sums",
    _RECT_ANN_PARAMS,
    _ANN_CENTER_NOTE,
)
apsum_pill_exact.__doc__ = _apsum_doc(
    "exact pill aperture sums",
    _PILL_PARAMS,
    _PILL_EXACT_NOTE,
)
apsum_pill_center.__doc__ = _apsum_doc(
    "center-selected pill aperture sums",
    _PILL_PARAMS,
    _PILL_CENTER_NOTE,
)
apsum_pill_ann_exact.__doc__ = _apsum_doc(
    "exact pill-annulus aperture sums",
    _PILL_ANN_PARAMS,
    _PILL_EXACT_NOTE,
)
apsum_pill_ann_center.__doc__ = _apsum_doc(
    "center-selected pill-annulus aperture sums",
    _PILL_ANN_PARAMS,
    _PILL_CENTER_NOTE,
)
apsum_wedge_exact.__doc__ = _apsum_doc(
    "exact wedge aperture sums",
    _WEDGE_PARAMS,
    _WEDGE_EXACT_NOTE,
)
apsum_wedge_center.__doc__ = _apsum_doc(
    "center-selected wedge aperture sums",
    _WEDGE_PARAMS,
    _WEDGE_CENTER_NOTE,
)

npix_circ_exact.__doc__ = _npix_doc(
    "exact circular-aperture",
    _CIRC_PARAMS,
    _EXACT_NOTE,
)
npix_circ_ann_exact.__doc__ = _npix_doc(
    "exact circular-annulus",
    _CIRC_ANN_PARAMS,
    _CIRC_ANN_EXACT_NOTE,
)
npix_circ_center.__doc__ = _npix_doc(
    "center-selected circular-aperture",
    _CIRC_PARAMS,
    _CENTER_NOTE,
)
npix_circ_ann_center.__doc__ = _npix_doc(
    "center-selected circular-annulus",
    _CIRC_ANN_PARAMS,
    _CIRC_ANN_CENTER_NOTE,
)
npix_ellip_exact.__doc__ = _npix_doc(
    "exact elliptical-aperture",
    _ELLIP_PARAMS,
    _EXACT_NOTE,
)
npix_ellip_center.__doc__ = _npix_doc(
    "center-selected elliptical-aperture",
    _ELLIP_PARAMS,
    _CENTER_NOTE,
)
npix_ellip_ann_exact.__doc__ = _npix_doc(
    "exact elliptical-annulus",
    _ELLIP_ANN_PARAMS,
    _ANN_EXACT_NOTE,
)
npix_ellip_ann_center.__doc__ = _npix_doc(
    "center-selected elliptical-annulus",
    _ELLIP_ANN_PARAMS,
    _ANN_CENTER_NOTE,
)
npix_rect_exact.__doc__ = _npix_doc(
    "exact rotated-rectangle-aperture",
    _RECT_PARAMS,
    _EXACT_NOTE,
)
npix_rect_center.__doc__ = _npix_doc(
    "center-selected rotated-rectangle-aperture",
    _RECT_PARAMS,
    _CENTER_NOTE,
)
npix_rect_ann_exact.__doc__ = _npix_doc(
    "exact rotated-rectangle-annulus",
    _RECT_ANN_PARAMS,
    _ANN_EXACT_NOTE,
)
npix_rect_ann_center.__doc__ = _npix_doc(
    "center-selected rotated-rectangle-annulus",
    _RECT_ANN_PARAMS,
    _ANN_CENTER_NOTE,
)
npix_pill_exact.__doc__ = _npix_doc(
    "exact pill-aperture",
    _PILL_PARAMS,
    _PILL_EXACT_NOTE,
)
npix_pill_center.__doc__ = _npix_doc(
    "center-selected pill-aperture",
    _PILL_PARAMS,
    _PILL_CENTER_NOTE,
)
npix_pill_ann_exact.__doc__ = _npix_doc(
    "exact pill-annulus",
    _PILL_ANN_PARAMS,
    _PILL_EXACT_NOTE,
)
npix_pill_ann_center.__doc__ = _npix_doc(
    "center-selected pill-annulus",
    _PILL_ANN_PARAMS,
    _PILL_CENTER_NOTE,
)
npix_wedge_exact.__doc__ = _npix_doc(
    "exact wedge-aperture",
    _WEDGE_PARAMS,
    _WEDGE_EXACT_NOTE,
)
npix_wedge_center.__doc__ = _npix_doc(
    "center-selected wedge-aperture",
    _WEDGE_PARAMS,
    _WEDGE_CENTER_NOTE,
)

weights_circ_exact.__doc__ = _weights_doc(
    "exact circular-aperture",
    _CIRC_PARAMS,
    _EXACT_NOTE,
)
weights_circ_center.__doc__ = _weights_doc(
    "center-selected circular-aperture",
    _CIRC_PARAMS,
    _CENTER_NOTE,
)
weights_circ_ann_exact.__doc__ = _weights_doc(
    "exact circular-annulus",
    _CIRC_ANN_PARAMS,
    _CIRC_ANN_EXACT_NOTE,
)
weights_circ_ann_center.__doc__ = _weights_doc(
    "center-selected circular-annulus",
    _CIRC_ANN_PARAMS,
    _CIRC_ANN_CENTER_NOTE,
)
weights_ellip_exact.__doc__ = _weights_doc(
    "exact elliptical-aperture",
    _ELLIP_PARAMS,
    _EXACT_NOTE,
)
weights_ellip_center.__doc__ = _weights_doc(
    "center-selected elliptical-aperture",
    _ELLIP_PARAMS,
    _CENTER_NOTE,
)
weights_ellip_ann_exact.__doc__ = _weights_doc(
    "exact elliptical-annulus",
    _ELLIP_ANN_PARAMS,
    _ANN_EXACT_NOTE,
)
weights_ellip_ann_center.__doc__ = _weights_doc(
    "center-selected elliptical-annulus",
    _ELLIP_ANN_PARAMS,
    _ANN_CENTER_NOTE,
)
weights_rect_exact.__doc__ = _weights_doc(
    "exact rotated-rectangle-aperture",
    _RECT_PARAMS,
    _EXACT_NOTE,
)
weights_rect_center.__doc__ = _weights_doc(
    "center-selected rotated-rectangle-aperture",
    _RECT_PARAMS,
    _CENTER_NOTE,
)
weights_rect_ann_exact.__doc__ = _weights_doc(
    "exact rotated-rectangle-annulus",
    _RECT_ANN_PARAMS,
    _ANN_EXACT_NOTE,
)
weights_rect_ann_center.__doc__ = _weights_doc(
    "center-selected rotated-rectangle-annulus",
    _RECT_ANN_PARAMS,
    _ANN_CENTER_NOTE,
)
weights_pill_exact.__doc__ = _weights_doc(
    "exact pill-aperture",
    _PILL_PARAMS,
    _PILL_EXACT_NOTE,
)
weights_pill_center.__doc__ = _weights_doc(
    "center-selected pill-aperture",
    _PILL_PARAMS,
    _PILL_CENTER_NOTE,
)
weights_pill_ann_exact.__doc__ = _weights_doc(
    "exact pill-annulus",
    _PILL_ANN_PARAMS,
    _PILL_EXACT_NOTE,
)
weights_pill_ann_center.__doc__ = _weights_doc(
    "center-selected pill-annulus",
    _PILL_ANN_PARAMS,
    _PILL_CENTER_NOTE,
)
weights_wedge_exact.__doc__ = _weights_doc(
    "exact wedge-aperture",
    _WEDGE_PARAMS,
    _WEDGE_EXACT_NOTE,
)
weights_wedge_center.__doc__ = _weights_doc(
    "center-selected wedge-aperture",
    _WEDGE_PARAMS,
    _WEDGE_CENTER_NOTE,
)

bboxes_circ.__doc__ = _bboxes_doc(
    "circular-aperture",
    _CIRC_PARAMS,
    "The box encloses the aperture footprint and may extend outside a later "
    "image frame.",
)
bboxes_circ_ann.__doc__ = _bboxes_doc(
    "circular-annulus",
    _CIRC_ANN_PARAMS,
    "The box is determined by the outer circular radius after validating the "
    "inner and outer radii.",
)
bboxes_ellip.__doc__ = _bboxes_doc(
    "elliptical-aperture",
    _ELLIP_PARAMS,
    "The box encloses the rotated ellipse footprint and may extend outside a "
    "later image frame.",
)
bboxes_ellip_ann.__doc__ = _bboxes_doc(
    "elliptical-annulus",
    _ELLIP_ANN_PARAMS,
    "The box is determined by the outer ellipse and ``theta_out`` after "
    "validating the inner and outer axes.",
)
bboxes_rect.__doc__ = _bboxes_doc(
    "rotated-rectangle-aperture",
    _RECT_PARAMS,
    "The box encloses the rotated rectangle footprint and may extend outside a "
    "later image frame.",
)
bboxes_rect_ann.__doc__ = _bboxes_doc(
    "rotated-rectangle-annulus",
    _RECT_ANN_PARAMS,
    "The box is determined by the outer rectangle and ``theta_out`` after "
    "validating the inner and outer dimensions.",
)
bboxes_pill.__doc__ = _bboxes_doc(
    "pill-aperture",
    _PILL_PARAMS,
    "The box encloses the composite rectangle-plus-cap footprint and may "
    "extend outside a later image frame.",
)
bboxes_pill_ann.__doc__ = _bboxes_doc(
    "pill-annulus",
    _PILL_ANN_PARAMS,
    "The box encloses the complete outer-minus-inner composite pill annulus "
    "after validating the inner and outer dimensions.",
)
bboxes_wedge.__doc__ = _bboxes_doc(
    "wedge-aperture",
    _WEDGE_PARAMS,
    "The box encloses the annular wedge and may extend outside a later image frame.",
)

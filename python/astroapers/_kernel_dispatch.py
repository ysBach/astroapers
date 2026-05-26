"""Private dispatch helpers for :mod:`astroapers.kernels`."""

from __future__ import annotations

import numpy as np

from . import _rust
from ._containers import validate_mask
from ._kernel_validation import _positions, _shape2


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
    flat_npix = np.full(flat_x.shape, np.nan, dtype=np.float64) if return_npix else None
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

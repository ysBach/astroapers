"""Private dispatch helpers for :mod:`astroapers.kernels`."""

from __future__ import annotations

import numpy as np

from . import _rust as aapr
from ._containers import validate_mask
from ._kernel_validation import _positions, _shape2


def _apsum(func_many, data, x, y, *params, return_npix: bool, validate: bool = True):
    arr = np.asarray(data)
    if validate and arr.ndim != 2:
        raise ValueError("data must be a 2-D array")
    name = func_many.__name__
    fast = _fast_apsum(
        name,
        func_many,
        arr,
        x,
        y,
        *params,
        return_npix=return_npix,
        validate=validate,
    )
    if fast is not None:
        return fast
    sum_funcs = None if return_npix else _dtype_functions(f"{name}_sum", arr.dtype)
    funcs = _dtype_functions(name, arr.dtype)
    if funcs is not None:
        arr = np.ascontiguousarray(arr)
        func_one, func_many = funcs
    else:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        func_one = getattr(aapr, f"{name}_one", None)
        if not return_npix:
            sum_one = getattr(aapr, f"{name}_sum_one", None)
            sum_many = getattr(aapr, f"{name}_sum", None)
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


def _fast_apsum(
    name: str,
    func_many,
    arr: np.ndarray,
    x,
    y,
    *params,
    return_npix: bool,
    validate: bool,
):
    if validate or arr.ndim != 2 or not arr.flags.c_contiguous:
        return None
    if not (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and x.ndim == y.ndim == 1
        and x.dtype == y.dtype == np.float64
        and x.flags.c_contiguous
        and y.flags.c_contiguous
    ):
        return None
    if return_npix:
        return None
    sum_funcs = _dtype_functions(f"{name}_sum", arr.dtype)
    if sum_funcs is not None:
        _sum_one, sum_many = sum_funcs
        return sum_many(arr, x, y, *params)
    return None


def _apsum_or_masked(
    func_many,
    weights_func,
    data,
    x,
    y,
    *params,
    mask,
    return_npix: bool,
    validate: bool = True,
):
    if mask is not None:
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
    return _apsum(
        func_many,
        data,
        x,
        y,
        *params,
        return_npix=return_npix,
        validate=validate,
    )


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
    one = getattr(aapr, f"{symbol}_one{suffix}", None)
    many = getattr(aapr, f"{symbol}{suffix}", None)
    if many is None:
        return None
    return one, many


_DTYPE_SUFFIX = {
    np.dtype(np.float64): "",
    np.dtype(np.float32): "_f32",
    np.dtype(np.int32): "_i32",
    np.dtype(np.int16): "_i16",
}

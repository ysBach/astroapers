"""Private validation helpers for :mod:`astroapers.kernels`."""

from __future__ import annotations

import numpy as np

from ._containers import BoundingBox


def _floatify(*args) -> tuple[float, ...]:
    return tuple(float(arg) for arg in args)


def _positions(x, y, *, validate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if not validate:
        return x, y
    xs = np.ascontiguousarray(np.atleast_1d(x), dtype=np.float64)
    ys = np.ascontiguousarray(np.atleast_1d(y), dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError("x and y must have matching shapes")
    return xs, ys


def _validate_circ_ann_radii(r_in: float, r_out: float) -> tuple[float, float]:
    r_in, r_out = _floatify(r_in, r_out)
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
    theta_out, dtheta_out = _floatify(
        theta_in if theta_out is None else theta_out,
        dtheta_in if dtheta_out is None else dtheta_out,
    )
    return theta_out, dtheta_out


def _validate_wedge(
    r_in: float,
    r_out: float,
    theta_in: float,
    dtheta_in: float,
    theta_out: float,
    dtheta_out: float,
) -> tuple[float, float, float, float, float, float]:
    r_in, r_out, theta_in, dtheta_in, theta_out, dtheta_out = _floatify(
        r_in, r_out, theta_in, dtheta_in, theta_out, dtheta_out
    )
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
    in0, in1, out0, out1 = _floatify(in0, in1, out0, out1)
    values = (in0, in1, out0, out1)
    if any((not np.isfinite(value) or value <= 0.0) for value in values):
        raise ValueError(f"{shape_name} dimensions must be positive finite scalars")
    if in0 > out0 or in1 > out1 or (in0 == out0 and in1 == out1):
        raise ValueError(
            f"inner {shape_name} dimensions must fit inside outer dimensions "
            "with at least one larger outer dimension"
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
    w_in, a_in, b_in, w_out, a_out, b_out = _floatify(
        w_in, a_in, b_in, w_out, a_out, b_out
    )
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

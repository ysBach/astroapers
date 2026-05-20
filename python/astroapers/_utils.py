"""Shared geometry and validation helpers."""

from __future__ import annotations

import math

import numpy as np

from ._containers import BoundingBox


def normalize_positions(positions) -> tuple[np.ndarray, bool]:
    arr = np.asarray(positions, dtype=np.float64)
    if arr.shape == (2,):
        if not np.all(np.isfinite(arr)):
            raise ValueError("positions must be finite")
        return arr.reshape(1, 2), True
    if arr.ndim == 2 and arr.shape[1] == 2:
        if not np.all(np.isfinite(arr)):
            raise ValueError("positions must be finite")
        return np.ascontiguousarray(arr), False
    raise ValueError("positions must be a 2-tuple or an array with shape (N, 2)")


def compound_annulus_path(outer_path, inner_path):
    import matplotlib.path as mpath

    inner = reversed_path(inner_path)
    return mpath.Path(
        np.concatenate([outer_path.vertices, inner.vertices]),
        np.concatenate([outer_path.codes, inner.codes]),
    )


def reversed_path(path):
    import matplotlib.path as mpath

    vertices = np.asarray(path.vertices, dtype=np.float64)
    codes = path.codes
    if codes is None:
        return mpath.Path(vertices[::-1])

    segments = []
    start = None
    current = None
    i = 0
    while i < len(codes):
        code = codes[i]
        if code == mpath.Path.MOVETO:
            current = vertices[i]
            start = current
            i += 1
        elif code == mpath.Path.LINETO:
            end = vertices[i]
            segments.append((code, current, end))
            current = end
            i += 1
        elif code == mpath.Path.CURVE3:
            ctrl = vertices[i]
            end = vertices[i + 1]
            segments.append((code, current, ctrl, end))
            current = end
            i += 2
        elif code == mpath.Path.CURVE4:
            ctrl1 = vertices[i]
            ctrl2 = vertices[i + 1]
            end = vertices[i + 2]
            segments.append((code, current, ctrl1, ctrl2, end))
            current = end
            i += 3
        elif code == mpath.Path.CLOSEPOLY:
            if (
                start is not None
                and current is not None
                and not np.allclose(current, start)
            ):
                segments.append((mpath.Path.LINETO, current, start))
            i += 1
        else:
            raise ValueError(f"unsupported Matplotlib path code {code}")

    if not segments:
        return path

    reversed_vertices = [segments[-1][-1]]
    reversed_codes = [mpath.Path.MOVETO]
    for segment in reversed(segments):
        code = segment[0]
        if code == mpath.Path.LINETO:
            _, start_point, _end_point = segment
            reversed_vertices.append(start_point)
            reversed_codes.append(mpath.Path.LINETO)
        elif code == mpath.Path.CURVE3:
            _, start_point, ctrl, _end_point = segment
            reversed_vertices.extend([ctrl, start_point])
            reversed_codes.extend([mpath.Path.CURVE3, mpath.Path.CURVE3])
        elif code == mpath.Path.CURVE4:
            _, start_point, ctrl1, ctrl2, _end_point = segment
            reversed_vertices.extend([ctrl2, ctrl1, start_point])
            reversed_codes.extend(
                [mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4]
            )
    reversed_vertices.append(reversed_vertices[0])
    reversed_codes.append(mpath.Path.CLOSEPOLY)
    return mpath.Path(
        np.asarray(reversed_vertices), np.asarray(reversed_codes, dtype=np.uint8)
    )


def pill_vertices(
    x: float,
    y: float,
    w: float,
    a: float,
    b: float,
    theta: float,
    origin,
    plot_samples: int,
) -> np.ndarray:
    half = plot_samples // 2
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    right_angles = np.linspace(0.5 * math.pi, -0.5 * math.pi, half)
    left_angles = np.linspace(-0.5 * math.pi, -1.5 * math.pi, half)
    angles = np.concatenate([right_angles, left_angles])
    center_offsets = np.concatenate([np.full(half, 0.5 * w), np.full(half, -0.5 * w)])
    local_x = center_offsets + a * np.cos(angles)
    local_y = b * np.sin(angles)
    pts = np.column_stack(
        (
            x + cos_t * local_x - sin_t * local_y,
            y + sin_t * local_x + cos_t * local_y,
        )
    )
    pts[:, 0] -= origin[0]
    pts[:, 1] -= origin[1]
    return pts


def bbox_from_extent(x: float, y: float, dx: float, dy: float) -> BoundingBox:
    ixmin = math.floor((x - dx) + 0.5)
    ixmax = math.ceil((x + dx) + 0.5)
    iymin = math.floor((y - dy) + 0.5)
    iymax = math.ceil((y + dy) + 0.5)
    return BoundingBox(ixmin, ixmax, iymin, iymax)


def require_positive_float(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return value


def require_finite_float(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def resolve_theta_pair(
    theta_in: float,
    theta_out: float | None,
    *,
    validate: bool,
) -> tuple[float, float]:
    """Return validated inner and outer annulus rotations."""
    if theta_out is None:
        theta_out = theta_in
    if validate:
        return (
            require_finite_float(theta_in, "theta_in"),
            require_finite_float(theta_out, "theta_out"),
        )
    return float(theta_in), float(theta_out)


def require_even_int_at_least(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an even integer >= {minimum}")
    parsed = int(value)
    if parsed < minimum or parsed % 2:
        raise ValueError(f"{name} must be an even integer >= {minimum}")
    return parsed

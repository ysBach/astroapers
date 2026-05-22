"""PathAp: custom closed-shape aperture with line and circular-arc segments.

PathAp v1 supports only straight line segments and circular arcs. It does not
represent Bezier curves, splines, elliptical arcs, or Python callback masks.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ._apertures import PixelAp, _shape_apsum_result
from ._containers import BoundingBox
from ._utils import require_even_int_at_least
from .kernels import (
    _masked_apsum,
    _weights_many,
    weights_center,
    weights_exact,
)

# Segment kind constants matching Rust SEG_* values
_SEG_MOVE = np.int8(0)
_SEG_LINE = np.int8(1)
_SEG_ARC = np.int8(2)
_SEG_CLOSE = np.int8(3)


def _encode_path_commands(
    segments: Sequence,
    holes: Sequence[Sequence] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode path command tuples into compact kinds/data arrays.

    Parameters
    ----------
    segments : sequence of tuples
        Outer contour commands.
    holes : sequence of sequences of tuples, optional
        Zero or more hole contours.

    Returns
    -------
    kinds : ndarray of int8, shape (N,)
    data : ndarray of float64, shape (N, 6)
    """
    all_contours = [segments]
    if holes:
        all_contours.extend(holes)

    kinds_list: list[int] = []
    data_list: list[list[float]] = []

    for contour in all_contours:
        has_move = False
        has_close = False
        for cmd in contour:
            if not cmd:
                raise ValueError("path commands must be non-empty tuples")
            tag = cmd[0]
            if tag == "move":
                if len(cmd) != 3:
                    raise ValueError("'move' command must be ('move', x, y)")
                kinds_list.append(int(_SEG_MOVE))
                data_list.append([float(cmd[1]), float(cmd[2]), 0.0, 0.0, 0.0, 0.0])
                has_move = True
            elif tag == "line":
                if len(cmd) != 3:
                    raise ValueError("'line' command must be ('line', x, y)")
                kinds_list.append(int(_SEG_LINE))
                data_list.append([float(cmd[1]), float(cmd[2]), 0.0, 0.0, 0.0, 0.0])
            elif tag == "arc":
                if len(cmd) != 6:
                    raise ValueError(
                        "'arc' command must be ('arc', cx, cy, r, theta0, dtheta)"
                    )
                kinds_list.append(int(_SEG_ARC))
                data_list.append(
                    [
                        float(cmd[1]),  # cx
                        float(cmd[2]),  # cy
                        float(cmd[3]),  # r
                        float(cmd[4]),  # theta0
                        float(cmd[5]),  # dtheta
                        0.0,
                    ]
                )
            elif tag == "close":
                kinds_list.append(int(_SEG_CLOSE))
                data_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                has_close = True
            else:
                raise ValueError(f"unknown path command: {tag!r}")
        if not has_move:
            raise ValueError("contour must start with a ('move', x, y) command")
        if not has_close:
            raise ValueError("contour must end with a ('close',) command")

    kinds = np.array(kinds_list, dtype=np.int8)
    data = np.array(data_list, dtype=np.float64)
    return kinds, data


def _validate_segments(segments: Sequence) -> None:
    if not segments:
        raise ValueError("segments must be non-empty")
    if not all(isinstance(cmd, Sequence) and cmd for cmd in segments):
        raise ValueError("segments must be a sequence of command tuples")
    cmds = [cmd[0] for cmd in segments]
    if cmds[0] != "move":
        raise ValueError("segments must start with ('move', x, y)")
    if cmds[-1] != "close":
        raise ValueError("segments must end with ('close',)")
    for tag in cmds[1:-1]:
        if tag not in ("line", "arc"):
            raise ValueError(f"segment command {tag!r} must be 'line' or 'arc'")
    move_count = cmds.count("move")
    if move_count > 1:
        raise ValueError("segments must contain exactly one 'move' command")
    start: tuple[float, float] | None = None
    current: tuple[float, float] | None = None
    effective_edges = 0
    for cmd in segments:
        tag = cmd[0]
        if tag == "move":
            if len(cmd) != 3:
                raise ValueError("'move' command must be ('move', x, y)")
            x = float(cmd[1])
            y = float(cmd[2])
            if not np.isfinite([x, y]).all():
                raise ValueError("move point must be finite")
            start = (x, y)
            current = (x, y)
        elif tag == "line":
            if len(cmd) != 3:
                raise ValueError("'line' command must be ('line', x, y)")
            if current is None:
                raise ValueError("line command requires a current point")
            x = float(cmd[1])
            y = float(cmd[2])
            if not np.isfinite([x, y]).all():
                raise ValueError("line endpoint must be finite")
            if (x - current[0]) ** 2 + (y - current[1]) ** 2 <= 1.0e-20:
                raise ValueError("zero-length line segment")
            current = (x, y)
            effective_edges += 1
        elif tag == "arc":
            if len(cmd) != 6:
                raise ValueError(
                    "'arc' command must be ('arc', cx, cy, r, theta0, dtheta)"
                )
            cx, cy, r, theta0, dtheta = (float(v) for v in cmd[1:])
            if not np.isfinite([cx, cy, r, theta0, dtheta]).all():
                raise ValueError("arc parameters must be finite")
            if r <= 0.0:
                raise ValueError("arc radius must be positive")
            if abs(dtheta) <= 0.0 or abs(dtheta) >= 2.0 * math.pi:
                raise ValueError("arc |dtheta| must be in (0, 2*pi)")
            current = (
                cx + r * math.cos(theta0 + dtheta),
                cy + r * math.sin(theta0 + dtheta),
            )
            effective_edges += 1
        elif tag == "close":
            if len(cmd) != 1:
                raise ValueError("'close' command must be ('close',)")
            if start is not None and current is not None:
                if (start[0] - current[0]) ** 2 + (
                    start[1] - current[1]
                ) ** 2 > 1.0e-20:
                    effective_edges += 1
    if effective_edges < 3:
        raise ValueError("path must have at least three effective edges")


def _validate_with_rust(kinds: np.ndarray, data: np.ndarray) -> None:
    from . import _rust

    _rust.bboxes_path_many(
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        kinds,
        data,
    )


def weights_path_exact(
    x, y, kinds: np.ndarray, data: np.ndarray, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        "weights_path_exact_many", x, y, kinds, data, validate=validate
    )


def weights_path_center(
    x, y, kinds: np.ndarray, data: np.ndarray, *, validate: bool = True
) -> tuple[list[np.ndarray], list[BoundingBox]]:
    return _weights_many(
        "weights_path_center_many", x, y, kinds, data, validate=validate
    )


class PathAp(PixelAp):
    """Aperture defined by a closed path of line segments and circular arcs.

    `PathAp` supports only straight line segments and circular arcs. It does
    not accept Bezier curves, spline curves, elliptical arcs, Matplotlib
    ``Path`` objects, or arbitrary Python predicate/callback masks. If another
    curve type is flattened into line segments before construction, the
    resulting aperture is exact for that flattened line-segment path, not for
    the original curve.

    Coordinates in `segments` and `holes` are local offsets from each aperture
    center. Angles are radians counterclockwise from the positive x axis.

    Parameters
    ----------
    positions : tuple or array-like
        Aperture center(s) as ``(x, y)`` pixel coordinates. Multiple centers
        as ``(N, 2)`` array-like.
    segments : sequence of tuples
        Outer contour commands. Must start with ``('move', x, y)``, followed
        by any mix of ``('line', x, y)`` and
        ``('arc', cx, cy, r, theta0, dtheta)``, and end with ``('close',)``.
        These are the only supported geometry commands.
    holes : sequence of sequences, optional
        Zero or more hole contours, each using the same command language.
    plot_samples : int, optional
        Even number >= 4 used to sample arcs for plotting.
    validate : bool, optional
        If `False`, skip Python-side validation.
    """

    def __init__(
        self,
        positions,
        segments: Sequence,
        holes: Sequence[Sequence] | None = None,
        *,
        validate: bool = True,
        plot_samples: int = 96,
    ):
        super().__init__(positions, validate=validate)
        self.segments = list(segments)
        self.holes = list(holes) if holes else []
        self.plot_samples = (
            require_even_int_at_least(plot_samples, "plot_samples", 4)
            if validate
            else int(plot_samples)
        )
        if validate:
            _validate_segments(self.segments)
            for i, hole in enumerate(self.holes):
                _validate_segments(hole)
        self._kinds, self._data = _encode_path_commands(
            self.segments, self.holes or None
        )
        if validate:
            _validate_with_rust(self._kinds, self._data)

    def _bbox_one(self, x: float, y: float) -> BoundingBox:
        from . import _rust

        ixmins, ixmaxs, iymins, iymaxs = _rust.bboxes_path_many(
            np.array([x], dtype=np.float64),
            np.array([y], dtype=np.float64),
            self._kinds,
            self._data,
        )
        return BoundingBox(
            int(ixmins[0]), int(ixmaxs[0]), int(iymins[0]), int(iymaxs[0])
        )

    def bboxes(self) -> list[BoundingBox]:
        from . import _rust
        from .kernels import _boxes_from_tuple

        return _boxes_from_tuple(
            _rust.bboxes_path_many(
                self._x,
                self._y,
                self._kinds,
                self._data,
            )
        )

    def _exact_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_exact(
            "weights_path_exact",
            bbox,
            (x, y, self._kinds, self._data),
        )

    def _center_weights_one(self, x: float, y: float, bbox: BoundingBox) -> np.ndarray:
        return weights_center(
            "weights_path_center",
            bbox,
            (x, y, self._kinds, self._data),
        )

    def weights(self, method: str = "exact") -> list[np.ndarray]:
        method = self._weight_method(method)
        weights_func = weights_path_exact if method == "exact" else weights_path_center
        weights, _ = weights_func(
            self._x,
            self._y,
            self._kinds,
            self._data,
            validate=self._validate,
        )
        return weights

    def apsum(
        self, data, mask=None, *, method: str = "exact", return_npix: bool = True
    ):
        method = self._weight_method(method)
        weights_func = weights_path_exact if method == "exact" else weights_path_center
        result = _masked_apsum(
            weights_func,
            data,
            mask,
            self._x,
            self._y,
            self._kinds,
            self._data,
            return_npix=return_npix,
            validate=self._validate,
        )
        if not return_npix:
            return _shape_apsum_result(self, result)
        apsum_arr, npix = result
        return _shape_apsum_result(self, apsum_arr, npix)

    def _weight_method(self, method: str) -> str:
        if method not in {"exact", "center"}:
            raise ValueError("method must be 'exact' or 'center'")
        return method

    def _patch_one(self, x: float, y: float, origin, **kwargs):
        import matplotlib.patches as mpatches
        import matplotlib.path as mpath

        plot_samples = require_even_int_at_least(
            kwargs.pop("plot_samples", self.plot_samples), "plot_samples", 4
        )
        kwargs.setdefault("fill", False)
        paths = [self.segments, *self.holes]
        vertices: list[tuple[float, float]] = []
        codes: list[int] = []
        for path in paths:
            path_vertices = _path_vertices(path, x, y, origin, plot_samples)
            if len(path_vertices) == 0:
                continue
            vertices.extend(map(tuple, path_vertices))
            codes.extend(
                [mpath.Path.MOVETO, *([mpath.Path.LINETO] * (len(path_vertices) - 1))]
            )
            vertices.append(tuple(path_vertices[0]))
            codes.append(mpath.Path.CLOSEPOLY)
        return mpatches.PathPatch(mpath.Path(vertices, codes), **kwargs)


def _path_vertices(
    segments: Sequence,
    x: float,
    y: float,
    origin,
    plot_samples: int,
) -> np.ndarray:
    """Sample path commands into a vertex array for matplotlib plotting."""
    pts: list[tuple[float, float]] = []
    cur_x = cur_y = 0.0
    arc_steps = max(4, plot_samples // 4)

    for cmd in segments:
        tag = cmd[0]
        if tag == "move":
            cur_x, cur_y = float(cmd[1]), float(cmd[2])
            pts.append((x + cur_x - origin[0], y + cur_y - origin[1]))
        elif tag == "line":
            cur_x, cur_y = float(cmd[1]), float(cmd[2])
            pts.append((x + cur_x - origin[0], y + cur_y - origin[1]))
        elif tag == "arc":
            cx, cy, r, theta0, dtheta = (
                float(cmd[1]),
                float(cmd[2]),
                float(cmd[3]),
                float(cmd[4]),
                float(cmd[5]),
            )
            for t in np.linspace(0.0, 1.0, arc_steps + 1)[1:]:
                angle = theta0 + t * dtheta
                px_loc = cx + r * math.cos(angle)
                py_loc = cy + r * math.sin(angle)
                pts.append((x + px_loc - origin[0], y + py_loc - origin[1]))
            cur_x = cx + r * math.cos(theta0 + dtheta)
            cur_y = cy + r * math.sin(theta0 + dtheta)
        elif tag == "close":
            pass  # matplotlib closes automatically

    return np.array(pts, dtype=np.float64)

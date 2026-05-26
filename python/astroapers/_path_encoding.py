"""Private path command encoding helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Segment kind constants matching Rust SEG_* values.
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

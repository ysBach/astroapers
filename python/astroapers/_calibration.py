"""Runtime calibration helpers for astroapers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .kernels import apsum_circ_exact, get_parallel_threshold, set_parallel_threshold


@dataclass(frozen=True)
class ParallelThresholdCalibration:
    """Result from calibrating the Rayon parallelization threshold.

    Attributes
    ----------
    threshold : int
        Recommended aperture-count threshold for switching to Rayon.
    original_threshold : int
        Threshold value active before calibration started.
    applied : bool
        Whether ``threshold`` was applied to the current Python process.
    counts : tuple[int, ...]
        Aperture counts tested during calibration.
    serial_seconds, parallel_seconds : tuple[float, ...]
        Best measured runtime for each count with serial and forced-parallel
        execution.
    """

    threshold: int
    original_threshold: int
    applied: bool
    counts: tuple[int, ...]
    serial_seconds: tuple[float, ...]
    parallel_seconds: tuple[float, ...]


def calibrate_parallel_threshold(
    *,
    counts: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    repeats: int = 100,
    image_size: int = 256,
    radius: float = 5.0,
    apply: bool = True,
) -> ParallelThresholdCalibration:
    """Benchmark and optionally apply a machine-local parallel threshold.

    The calibration compares the forced-serial path against the forced-parallel
    path for representative circular aperture apsum measurements. The returned
    threshold minimizes total measured runtime across the tested counts. A
    threshold larger than the largest tested count means the serial path was
    fastest for the measured workload.
    """
    counts = tuple(_positive_int(value, "counts") for value in counts)
    if not counts:
        raise ValueError("counts must not be empty")
    counts = tuple(sorted(set(counts)))
    repeats = _positive_int(repeats, "repeats")
    image_size = _positive_int(image_size, "image_size")
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")

    data, x_all, y_all = _calibration_case(max(counts), image_size, radius)
    original = get_parallel_threshold()
    serial: list[float] = []
    parallel: list[float] = []
    try:
        for count in counts:
            x = x_all[:count]
            y = y_all[:count]
            serial.append(_best_time(data, x, y, radius, max(counts) + 1, repeats))
            parallel.append(_best_time(data, x, y, radius, 1, repeats))
    finally:
        set_parallel_threshold(original)

    threshold = _select_threshold(counts, serial, parallel)
    if apply:
        set_parallel_threshold(threshold)

    return ParallelThresholdCalibration(
        threshold=threshold,
        original_threshold=original,
        applied=apply,
        counts=counts,
        serial_seconds=tuple(serial),
        parallel_seconds=tuple(parallel),
    )


def _calibration_case(
    count: int, image_size: int, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin = max(2.0 * radius + 2.0, 16.0)
    if image_size <= 2.0 * margin:
        raise ValueError("image_size is too small for the requested radius")
    y_grid, x_grid = np.indices((image_size, image_size), dtype=np.float64)
    data = 100.0 + 0.01 * x_grid + 0.02 * y_grid
    rng = np.random.default_rng(20250501)
    x = rng.uniform(margin, image_size - margin, size=count).astype(np.float64)
    y = rng.uniform(margin, image_size - margin, size=count).astype(np.float64)
    return np.ascontiguousarray(data), x, y


def _best_time(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    threshold: int,
    repeats: int,
) -> float:
    set_parallel_threshold(threshold)
    apsum_circ_exact(data, x, y, radius, return_npix=False)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        apsum_circ_exact(data, x, y, radius, return_npix=False)
        best = min(best, time.perf_counter() - start)
    return best


def _select_threshold(
    counts: tuple[int, ...], serial: list[float], parallel: list[float]
) -> int:
    candidates = (*counts, counts[-1] + 1)
    best_threshold = candidates[-1]
    best_total = float("inf")
    for threshold in candidates:
        total = sum(
            parallel_time if count >= threshold else serial_time
            for count, serial_time, parallel_time in zip(counts, serial, parallel)
        )
        if total < best_total:
            best_threshold = threshold
            best_total = total
    return best_threshold


def _positive_int(value: int, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain positive integers") from exc
    if isinstance(value, bool) or parsed <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return parsed

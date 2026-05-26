"""Shared kernel operation bindings for aperture objects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import _rust as aapr
from .kernels import (
    apsum_circ_ann_center,
    apsum_circ_ann_exact,
    apsum_circ_center,
    apsum_circ_exact,
    apsum_ellip_ann_center,
    apsum_ellip_ann_exact,
    apsum_ellip_center,
    apsum_ellip_exact,
    apsum_path_center,
    apsum_path_exact,
    apsum_pill_ann_center,
    apsum_pill_ann_exact,
    apsum_pill_center,
    apsum_pill_exact,
    apsum_rect_ann_center,
    apsum_rect_ann_exact,
    apsum_rect_center,
    apsum_rect_exact,
    apsum_wedge_center,
    apsum_wedge_exact,
    bboxes_path,
    npix_circ_ann_center,
    npix_circ_ann_exact,
    npix_circ_center,
    npix_circ_exact,
    npix_ellip_ann_center,
    npix_ellip_ann_exact,
    npix_ellip_center,
    npix_ellip_exact,
    npix_path_center,
    npix_path_exact,
    npix_pill_ann_center,
    npix_pill_ann_exact,
    npix_pill_center,
    npix_pill_exact,
    npix_rect_ann_center,
    npix_rect_ann_exact,
    npix_rect_center,
    npix_rect_exact,
    npix_wedge_center,
    npix_wedge_exact,
)


@dataclass(frozen=True)
class ApertureKernelOps:
    weights_exact_one: Callable
    weights_center_one: Callable
    weights_exact: Callable
    weights_center: Callable
    apsum_exact: Callable
    apsum_center: Callable
    npix_exact: Callable
    npix_center: Callable
    bboxes: Callable | None = None


CIRC_OPS = ApertureKernelOps(
    weights_exact_one=aapr.weights_circ_exact_one,
    weights_center_one=aapr.weights_circ_center_one,
    weights_exact=aapr.weights_circ_exact,
    weights_center=aapr.weights_circ_center,
    apsum_exact=apsum_circ_exact,
    apsum_center=apsum_circ_center,
    npix_exact=npix_circ_exact,
    npix_center=npix_circ_center,
)

CIRC_AN_OPS = ApertureKernelOps(
    weights_exact_one=aapr.weights_circ_ann_exact_one,
    weights_center_one=aapr.weights_circ_ann_center_one,
    weights_exact=aapr.weights_circ_ann_exact,
    weights_center=aapr.weights_circ_ann_center,
    apsum_exact=apsum_circ_ann_exact,
    apsum_center=apsum_circ_ann_center,
    npix_exact=npix_circ_ann_exact,
    npix_center=npix_circ_ann_center,
)

ELLIP_OPS = ApertureKernelOps(
    weights_exact_one=aapr.weights_ellip_exact_one,
    weights_center_one=aapr.weights_ellip_center_one,
    weights_exact=aapr.weights_ellip_exact,
    weights_center=aapr.weights_ellip_center,
    apsum_exact=apsum_ellip_exact,
    apsum_center=apsum_ellip_center,
    npix_exact=npix_ellip_exact,
    npix_center=npix_ellip_center,
)

ELLIP_AN_OPS = ApertureKernelOps(
    weights_exact_one=aapr.weights_ellip_ann_exact_one,
    weights_center_one=aapr.weights_ellip_ann_center_one,
    weights_exact=aapr.weights_ellip_ann_exact,
    weights_center=aapr.weights_ellip_ann_center,
    apsum_exact=apsum_ellip_ann_exact,
    apsum_center=apsum_ellip_ann_center,
    npix_exact=npix_ellip_ann_exact,
    npix_center=npix_ellip_ann_center,
)

RECT_OPS = ApertureKernelOps(
    weights_exact_one=aapr.weights_rect_exact_one,
    weights_center_one=aapr.weights_rect_center_one,
    weights_exact=aapr.weights_rect_exact,
    weights_center=aapr.weights_rect_center,
    apsum_exact=apsum_rect_exact,
    apsum_center=apsum_rect_center,
    npix_exact=npix_rect_exact,
    npix_center=npix_rect_center,
)

RECT_AN_OPS = ApertureKernelOps(
    weights_exact_one=aapr.weights_rect_ann_exact_one,
    weights_center_one=aapr.weights_rect_ann_center_one,
    weights_exact=aapr.weights_rect_ann_exact,
    weights_center=aapr.weights_rect_ann_center,
    apsum_exact=apsum_rect_ann_exact,
    apsum_center=apsum_rect_ann_center,
    npix_exact=npix_rect_ann_exact,
    npix_center=npix_rect_ann_center,
)

PILL_OPS = ApertureKernelOps(
    weights_exact_one=aapr.weights_pill_exact_one,
    weights_center_one=aapr.weights_pill_center_one,
    weights_exact=aapr.weights_pill_exact,
    weights_center=aapr.weights_pill_center,
    apsum_exact=apsum_pill_exact,
    apsum_center=apsum_pill_center,
    npix_exact=npix_pill_exact,
    npix_center=npix_pill_center,
)

PILL_AN_OPS = ApertureKernelOps(
    weights_exact_one=aapr.weights_pill_ann_exact_one,
    weights_center_one=aapr.weights_pill_ann_center_one,
    weights_exact=aapr.weights_pill_ann_exact,
    weights_center=aapr.weights_pill_ann_center,
    apsum_exact=apsum_pill_ann_exact,
    apsum_center=apsum_pill_ann_center,
    npix_exact=npix_pill_ann_exact,
    npix_center=npix_pill_ann_center,
)

WEDGE_OPS = ApertureKernelOps(
    weights_exact_one=aapr.weights_wedge_exact_one,
    weights_center_one=aapr.weights_wedge_center_one,
    weights_exact=aapr.weights_wedge_exact,
    weights_center=aapr.weights_wedge_center,
    apsum_exact=apsum_wedge_exact,
    apsum_center=apsum_wedge_center,
    npix_exact=npix_wedge_exact,
    npix_center=npix_wedge_center,
)

PATH_OPS = ApertureKernelOps(
    weights_exact_one=aapr.weights_path_exact_one,
    weights_center_one=aapr.weights_path_center_one,
    weights_exact=aapr.weights_path_exact,
    weights_center=aapr.weights_path_center,
    apsum_exact=apsum_path_exact,
    apsum_center=apsum_path_center,
    npix_exact=npix_path_exact,
    npix_center=npix_path_center,
    bboxes=bboxes_path,
)

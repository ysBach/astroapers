"""Private generated docstrings for :mod:`astroapers.kernels`."""

from __future__ import annotations


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


def apply_kernel_docstrings(symbols: dict[str, object]) -> None:
    """Attach generated docstrings to public ``astroapers.kernels`` functions."""
    symbols["apsum_circ_exact"].__doc__ = _apsum_doc(
        "exact circular aperture sums",
        _CIRC_PARAMS,
        _EXACT_NOTE,
    )
    symbols["apsum_circ_ann_exact"].__doc__ = _apsum_doc(
        "exact circular-annulus aperture sums",
        _CIRC_ANN_PARAMS,
        _CIRC_ANN_EXACT_NOTE,
    )
    symbols["apsum_circ_center"].__doc__ = _apsum_doc(
        "center-selected circular aperture sums",
        _CIRC_PARAMS,
        _CENTER_NOTE,
    )
    symbols["apsum_circ_ann_center"].__doc__ = _apsum_doc(
        "center-selected circular-annulus aperture sums",
        _CIRC_ANN_PARAMS,
        _CIRC_ANN_CENTER_NOTE,
    )
    symbols["apsum_ellip_exact"].__doc__ = _apsum_doc(
        "exact elliptical aperture sums",
        _ELLIP_PARAMS,
        _EXACT_NOTE,
    )
    symbols["apsum_ellip_center"].__doc__ = _apsum_doc(
        "center-selected elliptical aperture sums",
        _ELLIP_PARAMS,
        _CENTER_NOTE,
    )
    symbols["apsum_ellip_ann_exact"].__doc__ = _apsum_doc(
        "exact elliptical-annulus aperture sums",
        _ELLIP_ANN_PARAMS,
        _ANN_EXACT_NOTE,
    )
    symbols["apsum_ellip_ann_center"].__doc__ = _apsum_doc(
        "center-selected elliptical-annulus aperture sums",
        _ELLIP_ANN_PARAMS,
        _ANN_CENTER_NOTE,
    )
    symbols["apsum_rect_exact"].__doc__ = _apsum_doc(
        "exact rotated-rectangle aperture sums",
        _RECT_PARAMS,
        _EXACT_NOTE,
    )
    symbols["apsum_rect_center"].__doc__ = _apsum_doc(
        "center-selected rotated-rectangle aperture sums",
        _RECT_PARAMS,
        _CENTER_NOTE,
    )
    symbols["apsum_rect_ann_exact"].__doc__ = _apsum_doc(
        "exact rotated-rectangle-annulus aperture sums",
        _RECT_ANN_PARAMS,
        _ANN_EXACT_NOTE,
    )
    symbols["apsum_rect_ann_center"].__doc__ = _apsum_doc(
        "center-selected rotated-rectangle-annulus aperture sums",
        _RECT_ANN_PARAMS,
        _ANN_CENTER_NOTE,
    )
    symbols["apsum_pill_exact"].__doc__ = _apsum_doc(
        "exact pill aperture sums",
        _PILL_PARAMS,
        _PILL_EXACT_NOTE,
    )
    symbols["apsum_pill_center"].__doc__ = _apsum_doc(
        "center-selected pill aperture sums",
        _PILL_PARAMS,
        _PILL_CENTER_NOTE,
    )
    symbols["apsum_pill_ann_exact"].__doc__ = _apsum_doc(
        "exact pill-annulus aperture sums",
        _PILL_ANN_PARAMS,
        _PILL_EXACT_NOTE,
    )
    symbols["apsum_pill_ann_center"].__doc__ = _apsum_doc(
        "center-selected pill-annulus aperture sums",
        _PILL_ANN_PARAMS,
        _PILL_CENTER_NOTE,
    )
    symbols["apsum_wedge_exact"].__doc__ = _apsum_doc(
        "exact wedge aperture sums",
        _WEDGE_PARAMS,
        _WEDGE_EXACT_NOTE,
    )
    symbols["apsum_wedge_center"].__doc__ = _apsum_doc(
        "center-selected wedge aperture sums",
        _WEDGE_PARAMS,
        _WEDGE_CENTER_NOTE,
    )

    symbols["npix_circ_exact"].__doc__ = _npix_doc(
        "exact circular-aperture",
        _CIRC_PARAMS,
        _EXACT_NOTE,
    )
    symbols["npix_circ_ann_exact"].__doc__ = _npix_doc(
        "exact circular-annulus",
        _CIRC_ANN_PARAMS,
        _CIRC_ANN_EXACT_NOTE,
    )
    symbols["npix_circ_center"].__doc__ = _npix_doc(
        "center-selected circular-aperture",
        _CIRC_PARAMS,
        _CENTER_NOTE,
    )
    symbols["npix_circ_ann_center"].__doc__ = _npix_doc(
        "center-selected circular-annulus",
        _CIRC_ANN_PARAMS,
        _CIRC_ANN_CENTER_NOTE,
    )
    symbols["npix_ellip_exact"].__doc__ = _npix_doc(
        "exact elliptical-aperture",
        _ELLIP_PARAMS,
        _EXACT_NOTE,
    )
    symbols["npix_ellip_center"].__doc__ = _npix_doc(
        "center-selected elliptical-aperture",
        _ELLIP_PARAMS,
        _CENTER_NOTE,
    )
    symbols["npix_ellip_ann_exact"].__doc__ = _npix_doc(
        "exact elliptical-annulus",
        _ELLIP_ANN_PARAMS,
        _ANN_EXACT_NOTE,
    )
    symbols["npix_ellip_ann_center"].__doc__ = _npix_doc(
        "center-selected elliptical-annulus",
        _ELLIP_ANN_PARAMS,
        _ANN_CENTER_NOTE,
    )
    symbols["npix_rect_exact"].__doc__ = _npix_doc(
        "exact rotated-rectangle-aperture",
        _RECT_PARAMS,
        _EXACT_NOTE,
    )
    symbols["npix_rect_center"].__doc__ = _npix_doc(
        "center-selected rotated-rectangle-aperture",
        _RECT_PARAMS,
        _CENTER_NOTE,
    )
    symbols["npix_rect_ann_exact"].__doc__ = _npix_doc(
        "exact rotated-rectangle-annulus",
        _RECT_ANN_PARAMS,
        _ANN_EXACT_NOTE,
    )
    symbols["npix_rect_ann_center"].__doc__ = _npix_doc(
        "center-selected rotated-rectangle-annulus",
        _RECT_ANN_PARAMS,
        _ANN_CENTER_NOTE,
    )
    symbols["npix_pill_exact"].__doc__ = _npix_doc(
        "exact pill-aperture",
        _PILL_PARAMS,
        _PILL_EXACT_NOTE,
    )
    symbols["npix_pill_center"].__doc__ = _npix_doc(
        "center-selected pill-aperture",
        _PILL_PARAMS,
        _PILL_CENTER_NOTE,
    )
    symbols["npix_pill_ann_exact"].__doc__ = _npix_doc(
        "exact pill-annulus",
        _PILL_ANN_PARAMS,
        _PILL_EXACT_NOTE,
    )
    symbols["npix_pill_ann_center"].__doc__ = _npix_doc(
        "center-selected pill-annulus",
        _PILL_ANN_PARAMS,
        _PILL_CENTER_NOTE,
    )
    symbols["npix_wedge_exact"].__doc__ = _npix_doc(
        "exact wedge-aperture",
        _WEDGE_PARAMS,
        _WEDGE_EXACT_NOTE,
    )
    symbols["npix_wedge_center"].__doc__ = _npix_doc(
        "center-selected wedge-aperture",
        _WEDGE_PARAMS,
        _WEDGE_CENTER_NOTE,
    )

    symbols["weights_circ_exact"].__doc__ = _weights_doc(
        "exact circular-aperture",
        _CIRC_PARAMS,
        _EXACT_NOTE,
    )
    symbols["weights_circ_center"].__doc__ = _weights_doc(
        "center-selected circular-aperture",
        _CIRC_PARAMS,
        _CENTER_NOTE,
    )
    symbols["weights_circ_ann_exact"].__doc__ = _weights_doc(
        "exact circular-annulus",
        _CIRC_ANN_PARAMS,
        _CIRC_ANN_EXACT_NOTE,
    )
    symbols["weights_circ_ann_center"].__doc__ = _weights_doc(
        "center-selected circular-annulus",
        _CIRC_ANN_PARAMS,
        _CIRC_ANN_CENTER_NOTE,
    )
    symbols["weights_ellip_exact"].__doc__ = _weights_doc(
        "exact elliptical-aperture",
        _ELLIP_PARAMS,
        _EXACT_NOTE,
    )
    symbols["weights_ellip_center"].__doc__ = _weights_doc(
        "center-selected elliptical-aperture",
        _ELLIP_PARAMS,
        _CENTER_NOTE,
    )
    symbols["weights_ellip_ann_exact"].__doc__ = _weights_doc(
        "exact elliptical-annulus",
        _ELLIP_ANN_PARAMS,
        _ANN_EXACT_NOTE,
    )
    symbols["weights_ellip_ann_center"].__doc__ = _weights_doc(
        "center-selected elliptical-annulus",
        _ELLIP_ANN_PARAMS,
        _ANN_CENTER_NOTE,
    )
    symbols["weights_rect_exact"].__doc__ = _weights_doc(
        "exact rotated-rectangle-aperture",
        _RECT_PARAMS,
        _EXACT_NOTE,
    )
    symbols["weights_rect_center"].__doc__ = _weights_doc(
        "center-selected rotated-rectangle-aperture",
        _RECT_PARAMS,
        _CENTER_NOTE,
    )
    symbols["weights_rect_ann_exact"].__doc__ = _weights_doc(
        "exact rotated-rectangle-annulus",
        _RECT_ANN_PARAMS,
        _ANN_EXACT_NOTE,
    )
    symbols["weights_rect_ann_center"].__doc__ = _weights_doc(
        "center-selected rotated-rectangle-annulus",
        _RECT_ANN_PARAMS,
        _ANN_CENTER_NOTE,
    )
    symbols["weights_pill_exact"].__doc__ = _weights_doc(
        "exact pill-aperture",
        _PILL_PARAMS,
        _PILL_EXACT_NOTE,
    )
    symbols["weights_pill_center"].__doc__ = _weights_doc(
        "center-selected pill-aperture",
        _PILL_PARAMS,
        _PILL_CENTER_NOTE,
    )
    symbols["weights_pill_ann_exact"].__doc__ = _weights_doc(
        "exact pill-annulus",
        _PILL_ANN_PARAMS,
        _PILL_EXACT_NOTE,
    )
    symbols["weights_pill_ann_center"].__doc__ = _weights_doc(
        "center-selected pill-annulus",
        _PILL_ANN_PARAMS,
        _PILL_CENTER_NOTE,
    )
    symbols["weights_wedge_exact"].__doc__ = _weights_doc(
        "exact wedge-aperture",
        _WEDGE_PARAMS,
        _WEDGE_EXACT_NOTE,
    )
    symbols["weights_wedge_center"].__doc__ = _weights_doc(
        "center-selected wedge-aperture",
        _WEDGE_PARAMS,
        _WEDGE_CENTER_NOTE,
    )

    symbols["bboxes_circ"].__doc__ = _bboxes_doc(
        "circular-aperture",
        _CIRC_PARAMS,
        "The box encloses the aperture footprint and may extend outside a later "
        "image frame.",
    )
    symbols["bboxes_circ_ann"].__doc__ = _bboxes_doc(
        "circular-annulus",
        _CIRC_ANN_PARAMS,
        "The box is determined by the outer circular radius after validating the "
        "inner and outer radii.",
    )
    symbols["bboxes_ellip"].__doc__ = _bboxes_doc(
        "elliptical-aperture",
        _ELLIP_PARAMS,
        "The box encloses the rotated ellipse footprint and may extend outside a "
        "later image frame.",
    )
    symbols["bboxes_ellip_ann"].__doc__ = _bboxes_doc(
        "elliptical-annulus",
        _ELLIP_ANN_PARAMS,
        "The box is determined by the outer ellipse and ``theta_out`` after "
        "validating the inner and outer axes.",
    )
    symbols["bboxes_rect"].__doc__ = _bboxes_doc(
        "rotated-rectangle-aperture",
        _RECT_PARAMS,
        "The box encloses the rotated rectangle footprint and may extend outside a "
        "later image frame.",
    )
    symbols["bboxes_rect_ann"].__doc__ = _bboxes_doc(
        "rotated-rectangle-annulus",
        _RECT_ANN_PARAMS,
        "The box is determined by the outer rectangle and ``theta_out`` after "
        "validating the inner and outer dimensions.",
    )
    symbols["bboxes_pill"].__doc__ = _bboxes_doc(
        "pill-aperture",
        _PILL_PARAMS,
        "The box encloses the composite rectangle-plus-cap footprint and may "
        "extend outside a later image frame.",
    )
    symbols["bboxes_pill_ann"].__doc__ = _bboxes_doc(
        "pill-annulus",
        _PILL_ANN_PARAMS,
        "The box encloses the complete outer-minus-inner composite pill annulus "
        "after validating the inner and outer dimensions.",
    )
    symbols["bboxes_wedge"].__doc__ = _bboxes_doc(
        "wedge-aperture",
        _WEDGE_PARAMS,
        "The box encloses the annular wedge and may extend outside a later image frame.",
    )

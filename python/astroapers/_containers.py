"""Mask and result containers for aperture operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OverlapSlices = tuple[tuple[slice, slice], tuple[slice, slice]]


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-index bounding box with exclusive upper bounds.

    Parameters
    ----------
    ixmin, ixmax : int
        Inclusive lower and exclusive upper x pixel indices.
    iymin, iymax : int
        Inclusive lower and exclusive upper y pixel indices.

    Notes
    -----
    Bounding boxes may extend outside an image. Use
    :meth:`overlap_slices` to obtain matching image and bbox-tight mask slices.
    """

    ixmin: int
    ixmax: int
    iymin: int
    iymax: int

    @property
    def shape(self) -> tuple[int, int]:
        """Mask-array shape as ``(ny, nx)``.

        The order follows NumPy image indexing, so the y extent comes before
        the x extent even though the bounding-box fields are named with x
        first.
        """
        return (self.iymax - self.iymin, self.ixmax - self.ixmin)

    def overlap_slices(self, shape: tuple[int, int]) -> OverlapSlices | None:
        """Return image and mask slices for the overlap with ``shape``.

        Parameters
        ----------
        shape : tuple[int, int]
            Full image shape as ``(ny, nx)``.

        Returns
        -------
        tuple of tuple of slice, or None
            ``(data_slices, mask_slices)`` when the bounding box overlaps the
            image. ``data_slices`` select the in-frame pixels from the full
            image; ``mask_slices`` select the matching region from the
            bbox-tight mask. Returns ``None`` when the bounding box is fully
            outside the image.
        """
        ny, nx = shape
        x0 = max(self.ixmin, 0)
        x1 = min(self.ixmax, nx)
        y0 = max(self.iymin, 0)
        y1 = min(self.iymax, ny)
        if x0 >= x1 or y0 >= y1:
            return None
        data_slices = (slice(y0, y1), slice(x0, x1))
        mask_slices = (
            slice(y0 - self.iymin, y1 - self.iymin),
            slice(x0 - self.ixmin, x1 - self.ixmin),
        )
        return data_slices, mask_slices


class ApMask:
    """Bbox-tight aperture overlap weights plus their pixel bounding box.

    `ApMask` is the convenience wrapper returned by `PixelAp.get_apmask()`. It is
    useful when the caller wants to place bbox-tight weights into a full image,
    multiply it by data, or extract weighted values. Use the `.weights`
    attribute when a top-level `apmask_*` helper needs raw bbox-tight weights. The
    wrapper preserves `float32` or `float64` aperture weights supplied by callers;
    other numeric or boolean dtypes are converted to `float64`.

    Common helpers include `npix()`, `to_image()`, `weighted_cutout()`,
    `weighted_values()`, and `apsum()`.
    """

    def __init__(self, weights, bbox: BoundingBox) -> None:
        self.bbox = bbox
        self.weights = _validate_apmask_weights(weights, bbox)

    def npix(self, shape: tuple[int, int], mask=None) -> float:
        """Return the in-frame sum of mask weights for ``shape``.

        Parameters
        ----------
        shape : tuple[int, int]
            Full image shape as ``(ny, nx)``. This controls image-edge clipping.
        mask : array_like of bool, optional
            Boolean image mask with shape ``shape``. ``True`` pixels are
            excluded from the returned effective pixel count.
        """
        return apmask_npix(self.weights, self.bbox, shape, mask=mask)

    def to_image(self, shape: tuple[int, int], fill_value: float = 0.0) -> np.ndarray:
        """Embed bbox-tight weights in a full image-shaped array.

        Parameters
        ----------
        shape : tuple[int, int]
            Output image shape as ``(ny, nx)``.
        fill_value : float, optional
            Value used outside the mask footprint.

        Returns
        -------
        ndarray
            Floating array with shape ``shape``. The dtype follows the bbox-tight
            aperture weights for floating weights. Pixels covered by this
            bounding box receive the fractional aperture weights; all other pixels
            are set to ``fill_value``. If the bounding box extends beyond the
            image, only the overlapping part is copied.
        """
        return apmask_to_image(self.weights, self.bbox, shape, fill_value=fill_value)

    def weighted_cutout(
        self, data, mask=None, fill_value: float = np.nan
    ) -> np.ndarray:
        """Return a bbox-tight weighted data cutout.

        Parameters
        ----------
        data : array_like
            Two-dimensional image sampled by this aperture mask.
        mask : array_like of bool, optional
            Boolean image mask with the same shape as ``data``. ``True`` pixels
            are excluded by setting their aperture weights to zero.
        fill_value : float, optional
            Value used for bbox-tight mask pixels that do not overlap ``data``.

        Returns
        -------
        ndarray
            Floating cutout with the same shape as ``self.weights``. In-frame
            pixels contain ``data * weight``; off-image pixels contain
            ``fill_value``.
        """
        return apmask_weighted_cutout(
            self.weights, self.bbox, data, mask=mask, fill_value=fill_value
        )

    def weighted_values(self, data, mask=None) -> np.ndarray:
        """Return weighted values for positive bbox-tight mask weights.

        Parameters
        ----------
        data : array_like
            Two-dimensional image sampled by this aperture mask.
        mask : array_like of bool, optional
            Boolean image mask with the same shape as ``data``. ``True`` pixels
            are excluded from the returned values.

        Returns
        -------
        ndarray
            One-dimensional floating array containing ``data * weight`` for
            pixels where the aperture weight is positive and the optional image
            mask is not set. Pixels outside ``data`` are ignored.
        """
        return apmask_weighted_values(self.weights, self.bbox, data, mask=mask)

    def apsum(self, data, mask=None, *, return_npix: bool = True):
        """Return aperture sum, and npix by default, for this mask.

        Parameters
        ----------
        data : array_like
            Two-dimensional image sampled by this aperture mask.
        mask : array_like of bool, optional
            Boolean image mask with the same shape as ``data``. ``True`` pixels
            contribute zero weight.
        return_npix : bool, optional
            If `True`, return ``(apsum, npix)``. If `False`, return only
            ``apsum``.

        Returns
        -------
        apsum : float
            Sum of ``data * weight`` over the in-frame, unmasked aperture
            footprint.
        npix : float
            Sum of the aperture weights that contributed to ``apsum``. Returned
            only when ``return_npix=True``.
        """
        return apmask_apsum(
            self.weights, self.bbox, data, mask=mask, return_npix=return_npix
        )


def _validate_apmask_weights(weights, bbox: BoundingBox) -> np.ndarray:
    """Return bbox-tight mask weights validated against ``bbox``.

    Parameters
    ----------
    weights : array_like
        Bbox-tight weights, normally stored in ``ApMask.weights`` or supplied by
        an external rasterizer.
    bbox : BoundingBox
        Pixel bounding box describing where ``weights`` lie in the parent image.

    Returns
    -------
    ndarray
        Floating ndarray view or copy of ``weights``. `float32` and `float64`
        are preserved; other numeric and boolean dtypes are converted to
        `float64`, including extended precision floating dtypes such as
        `float128`/`longdouble` where NumPy provides them.

    Raises
    ------
    ValueError
        If ``weights.shape`` does not match ``bbox.shape``.
    """
    arr = np.asarray(weights)
    if arr.shape != bbox.shape:
        raise ValueError(
            f"mask weights shape {arr.shape} does not match bbox shape {bbox.shape}"
        )
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr
    if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.float64, copy=False)
    raise TypeError(f"mask weights must be numeric, got dtype {arr.dtype}")


def _mask_dtype(weights: np.ndarray) -> np.dtype:
    if weights.dtype == np.float32:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _calculation_dtype(image: np.ndarray, weights: np.ndarray) -> np.dtype:
    result = np.result_type(image.dtype, weights.dtype)
    if result == np.dtype(np.float32):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def apmask_to_image(
    weights, bbox: BoundingBox, shape: tuple[int, int], fill_value: float = 0.0
) -> np.ndarray:
    """Embed bbox-tight mask weights in a full image array.

    This is the function equivalent of ``ApMask(weights, bbox).to_image(shape)``
    and is intended for raw weights stored on ``ApMask.weights`` or supplied by
    external rasterizers.

    Parameters
    ----------
    weights : array_like
        Bbox-tight aperture weights with shape ``bbox.shape``.
    bbox : BoundingBox
        Location of ``weights`` in the output image coordinate system.
    shape : tuple[int, int]
        Output image shape as ``(ny, nx)``.
    fill_value : float, optional
        Value assigned to pixels outside the aperture mask footprint.

    Returns
    -------
    ndarray
        Floating image with shape ``shape``. If ``bbox`` is partly outside the
        image, only the overlapping weights are copied. If it is fully outside,
        the returned image contains only ``fill_value``.
    """
    weights = _validate_apmask_weights(weights, bbox)
    image = np.full(shape, fill_value, dtype=_mask_dtype(weights))
    overlap = bbox.overlap_slices(shape)
    if overlap is None:
        return image
    data_slices, mask_slices = overlap
    image[data_slices] = weights[mask_slices]
    return image


def apmask_weighted_cutout(
    weights, bbox: BoundingBox, image, mask=None, fill_value: float = np.nan
) -> np.ndarray:
    """Return a bbox-tight weighted cutout for raw aperture weights and ``bbox``.

    Parameters
    ----------
    weights : array_like
        Bbox-tight aperture weights with shape ``bbox.shape``.
    bbox : BoundingBox
        Location of the bbox-tight weights in ``image`` coordinates.
    image : array_like
        Two-dimensional image to sample.
    mask : array_like of bool, optional
        Boolean image mask with the same shape as ``image``. ``True`` pixels
        are excluded by setting their aperture weights to zero before
        multiplication.
    fill_value : float, optional
        Value used in returned cutout pixels that do not overlap ``image``.

    Returns
    -------
    ndarray
        Floating cutout with shape ``bbox.shape``. Overlapping pixels contain
        ``image * weight``; non-overlapping pixels contain ``fill_value``.
    """
    weights_full = _validate_apmask_weights(weights, bbox)
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("data must be a 2-D array")
    dtype = _calculation_dtype(arr, weights_full)
    cutout = np.full(weights_full.shape, fill_value, dtype=dtype)
    overlap = bbox.overlap_slices(arr.shape)
    if overlap is None:
        return cutout
    data_slices, mask_slices = overlap
    weights = weights_full[mask_slices].astype(dtype, copy=False)
    if mask is not None:
        weights = weights.copy()
        bad = validate_mask(mask, arr.shape)[data_slices]
        weights[bad] = 0.0
    cutout[mask_slices] = arr[data_slices].astype(dtype, copy=False) * weights
    return cutout


def apmask_weighted_values(weights, bbox: BoundingBox, image, mask=None) -> np.ndarray:
    """Return weighted values for positive raw aperture weights and ``bbox``.

    Parameters
    ----------
    weights : array_like
        Bbox-tight aperture weights with shape ``bbox.shape``.
    bbox : BoundingBox
        Location of the bbox-tight weights in ``image`` coordinates.
    image : array_like
        Two-dimensional image to sample.
    mask : array_like of bool, optional
        Boolean image mask with the same shape as ``image``. ``True`` pixels
        are omitted from the returned vector.

    Returns
    -------
    ndarray
        One-dimensional floating array containing ``image * weight`` for
        in-frame pixels with positive aperture weight. Masked pixels and
        off-image pixels are omitted.
    """
    weights_full = _validate_apmask_weights(weights, bbox)
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("data must be a 2-D array")
    dtype = _calculation_dtype(arr, weights_full)
    overlap = bbox.overlap_slices(arr.shape)
    if overlap is None:
        return np.array([], dtype=dtype)
    data_slices, mask_slices = overlap
    weights = weights_full[mask_slices].astype(dtype, copy=False)
    valid = weights > 0.0
    if mask is not None:
        valid &= ~validate_mask(mask, arr.shape)[data_slices]
    return arr[data_slices].astype(dtype, copy=False)[valid] * weights[valid]


def apmask_npix(weights, bbox: BoundingBox, shape: tuple[int, int], mask=None) -> float:
    """Return the in-frame sum of raw aperture weights for ``shape``.

    Parameters
    ----------
    weights : array_like
        Bbox-tight aperture weights with shape ``bbox.shape``.
    bbox : BoundingBox
        Location of the bbox-tight weights in image coordinates.
    shape : tuple[int, int]
        Full image shape as ``(ny, nx)``. This controls image-edge clipping.
    mask : array_like of bool, optional
        Boolean image mask with shape ``shape``. ``True`` pixels are excluded
        from the returned effective pixel count.

    Returns
    -------
    npix : float
        Sum of in-frame, unmasked aperture weights. Accumulation uses
        `float64`, even when the stored weights are `float32`.
    """
    weights_full = _validate_apmask_weights(weights, bbox)
    shape = _validate_image_shape(shape)
    overlap = bbox.overlap_slices(shape)
    if overlap is None:
        return 0.0
    data_slices, mask_slices = overlap
    weights = weights_full[mask_slices]
    if mask is not None:
        weights = weights.copy()
        weights[validate_mask(mask, shape)[data_slices]] = 0.0
    return float(np.sum(weights, dtype=np.float64))


def apmask_apsum(
    weights, bbox: BoundingBox, image, mask=None, *, return_npix: bool = True
):
    """Return aperture sum, and npix by default, for raw aperture weights.

    Parameters
    ----------
    weights : array_like
        Bbox-tight aperture weights with shape ``bbox.shape``.
    bbox : BoundingBox
        Location of the bbox-tight weights in ``image`` coordinates.
    image : array_like
        Two-dimensional image to sample.
    mask : array_like of bool, optional
        Boolean image mask with the same shape as ``image``. ``True`` pixels
        contribute zero weight.
    return_npix : bool, optional
        If `True`, return ``(apsum, npix)``. If `False`, return only ``apsum``.

    Returns
    -------
    apsum : float
        Sum of ``image * weight`` over the overlapping, unmasked footprint.
        Accumulation uses `float64`, even when both the image and stored weights
        are `float32`.
    npix : float
        Sum of the contributing aperture weights. This can be smaller than the
        analytic aperture area when the aperture is clipped by the image edge
        or by ``mask``. Accumulation uses `float64`. Returned only when
        ``return_npix=True``.
    """
    weights_full = _validate_apmask_weights(weights, bbox)
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("data must be a 2-D array")
    overlap = bbox.overlap_slices(arr.shape)
    if overlap is None:
        return (0.0, 0.0) if return_npix else 0.0
    data_slices, mask_slices = overlap
    weights = weights_full[mask_slices].astype(np.float64, copy=False)
    if mask is not None:
        weights = weights.copy()
        weights[validate_mask(mask, arr.shape)[data_slices]] = 0.0
    data_values = arr[data_slices].astype(np.float64, copy=False)
    apsum = float(np.sum(data_values * weights, dtype=np.float64))
    if not return_npix:
        return apsum
    return apsum, float(np.sum(weights, dtype=np.float64))


def _validate_image_shape(shape: tuple[int, int]) -> tuple[int, int]:
    try:
        ny, nx = shape
    except (TypeError, ValueError) as exc:
        raise ValueError("shape must be a 2-tuple") from exc
    ny = int(ny)
    nx = int(nx)
    if ny < 0 or nx < 0:
        raise ValueError("shape dimensions must be nonnegative")
    return ny, nx


def validate_mask(mask, shape: tuple[int, int]) -> np.ndarray:
    """Return a boolean image mask validated against ``shape``.

    Parameters
    ----------
    mask : array_like
        Candidate mask. Values are converted with ``np.asarray(mask,
        dtype=bool)``.
    shape : tuple[int, int]
        Required image shape as ``(ny, nx)``.

    Returns
    -------
    ndarray
        Boolean mask with shape ``shape``.

    Raises
    ------
    ValueError
        If the converted mask does not have shape ``shape``.
    """
    bad = np.asarray(mask, dtype=bool)
    if bad.shape != shape:
        raise ValueError("mask must have the same shape as data")
    return bad

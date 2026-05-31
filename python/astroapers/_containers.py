"""Mask and result containers for aperture operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import reducers as rd

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

    def to_fits_section(self, shape: tuple[int, int] | None = None) -> str:
        """Return this box as a FITS image section string.

        Parameters
        ----------
        shape : tuple[int, int], optional
            Full image shape as ``(ny, nx)``. If supplied, the returned section
            is clipped to the image overlap. If omitted, the raw bounding box
            must already be valid in FITS coordinates.

        Returns
        -------
        str
            FITS-standard image section in 1-indexed, ``x,y`` order, with
            inclusive upper bounds.

        Raises
        ------
        ValueError
            If the raw bounding box has nonpositive extent, if an unclipped
            box would start before FITS pixel 1, or if the box does not
            overlap ``shape``.
        """
        if shape is None:
            if self.ixmax <= self.ixmin or self.iymax <= self.iymin:
                raise ValueError("bounding box must have positive x and y extents")
            if self.ixmin < 0 or self.iymin < 0:
                raise ValueError(
                    "shape is required to convert an off-image bounding box "
                    "to a valid FITS section"
                )
            return _slices_to_fits_section(
                slice(self.iymin, self.iymax), slice(self.ixmin, self.ixmax)
            )

        shape = _validate_imshape(shape)
        overlap = self.overlap_slices(shape)
        if overlap is None:
            raise ValueError("bounding box does not overlap shape")
        data_slices, _ = overlap
        return _slices_to_fits_section(data_slices[0], data_slices[1])

    def to_image(
        self,
        weights,
        shape: tuple[int, int],
        fill_value: float = 0.0,
        *,
        validate: bool = True,
    ) -> np.ndarray:
        """Embed bbox-tight mask weights in a full image array.

        Parameters
        ----------
        weights : array_like
            Bbox-tight aperture weights with shape ``self.shape``.
        shape : tuple[int, int]
            Output image shape as ``(ny, nx)``.
        fill_value : float, optional
            Value assigned to pixels outside the aperture mask footprint.
        validate : bool, optional
            If `False`, trust that ``weights`` and ``shape`` are already valid.

        Returns
        -------
        ndarray
            Floating image with shape ``shape``. If this bounding box is partly
            outside the image, only the overlapping weights are copied. If it
            is fully outside, the returned image contains only ``fill_value``.
        """
        weights = _validate_weights(weights, self) if validate else np.asarray(weights)
        shape = _validate_imshape(shape) if validate else shape
        image = np.full(shape, fill_value, dtype=_mask_dtype(weights))
        overlap = self.overlap_slices(shape)
        if overlap is None:
            return image
        data_slices, mask_slices = overlap
        image[data_slices] = weights[mask_slices]
        return image

    def weighted_cutout(
        self, weights, image, mask=None, fill_value: float = np.nan, *, validate=True
    ) -> np.ndarray:
        """Return a bbox-tight weighted cutout for raw aperture weights.

        Parameters
        ----------
        weights : array_like
            Bbox-tight aperture weights with shape ``self.shape``.
        image : array_like
            Two-dimensional image to sample.
        mask : array_like of bool, optional
            Boolean image mask with the same shape as ``image``. ``True``
            pixels are set to ``fill_value`` in the returned cutout.
        fill_value : float, optional
            Value used in returned cutout pixels that do not overlap ``image``.
        validate : bool, optional
            If `False`, trust that ``weights``, ``image``, and ``mask`` are
            already valid and aligned.

        Returns
        -------
        ndarray
            Floating cutout with shape ``self.shape``. Unmasked overlapping
            pixels contain ``image * weight``; masked and non-overlapping
            pixels contain ``fill_value``.
        """
        w_full = _validate_weights(weights, self) if validate else np.asarray(weights)
        arr = np.asarray(image)
        if validate and arr.ndim != 2:
            raise ValueError("data must be a 2-D array")
        dtype = _calculation_dtype(arr, w_full)
        cutout = np.full(w_full.shape, fill_value, dtype=dtype)
        overlap = self.overlap_slices(arr.shape)
        if overlap is None:
            return cutout
        data_slices, mask_slices = overlap
        weights = w_full[mask_slices].astype(dtype, copy=False)
        weighted = arr[data_slices].astype(dtype, copy=False) * weights
        if mask is not None:
            weighted = weighted.copy()
            bad = validate_mask(mask, arr.shape) if validate else mask
            weighted[bad[data_slices]] = fill_value
        cutout[mask_slices] = weighted
        return cutout

    def weighted_values(
        self, weights, image, mask=None, *, validate=True
    ) -> np.ndarray:
        """Return weighted values for positive raw aperture weights.

        Parameters
        ----------
        weights : array_like
            Bbox-tight aperture weights with shape ``self.shape``.
        image : array_like
            Two-dimensional image to sample.
        mask : array_like of bool, optional
            Boolean image mask with the same shape as ``image``. ``True``
            pixels are omitted from the returned vector.
        validate : bool, optional
            If `False`, trust that ``weights``, ``image``, and ``mask`` are
            already valid and aligned.

        Returns
        -------
        ndarray
            One-dimensional floating array containing ``image * weight`` for
            in-frame pixels with positive aperture weight. Masked pixels and
            off-image pixels are omitted.
        """
        w_full = _validate_weights(weights, self) if validate else np.asarray(weights)
        arr = np.asarray(image)
        if validate and arr.ndim != 2:
            raise ValueError("data must be a 2-D array")
        dtype = _calculation_dtype(arr, w_full)
        overlap = self.overlap_slices(arr.shape)
        if overlap is None:
            return np.array([], dtype=dtype)
        data_slices, mask_slices = overlap
        weights = w_full[mask_slices].astype(dtype, copy=False)
        valid = weights > 0.0
        if mask is not None:
            bad = validate_mask(mask, arr.shape) if validate else mask
            valid &= ~bad[data_slices]
        return arr[data_slices].astype(dtype, copy=False)[valid] * weights[valid]

    def npix(
        self, weights, shape: tuple[int, int], mask=None, *, validate: bool = True
    ) -> float:
        """Return the in-frame sum of raw aperture weights for ``shape``.

        Parameters
        ----------
        weights : array_like
            Bbox-tight aperture weights with shape ``self.shape``.
        shape : tuple[int, int]
            Full image shape as ``(ny, nx)``. This controls image-edge
            clipping.
        mask : array_like of bool, optional
            Boolean image mask with shape ``shape``. ``True`` pixels are
            excluded from the returned effective pixel count.
        validate : bool, optional
            If `False`, trust that ``weights``, ``shape``, and ``mask`` are
            already valid and aligned.

        Returns
        -------
        npix : float
            Sum of in-frame, unmasked aperture weights. Accumulation uses
            `float64`, even when the stored weights are `float32`.
        """
        w_full = _validate_weights(weights, self) if validate else np.asarray(weights)
        shape = _validate_imshape(shape) if validate else shape
        overlap = self.overlap_slices(shape)
        if overlap is None:
            return 0.0
        bad = (
            None if mask is None else (validate_mask(mask, shape) if validate else mask)
        )
        data_slices, mask_slices = overlap
        weights = w_full[mask_slices]
        if bad is not None:
            weights = weights.copy()
            weights[bad[data_slices]] = 0.0
        return _sum_float64(weights)

    def apsum(self, weights, image, mask=None, *, return_npix=True, validate=True):
        """Return aperture sum, and npix by default, for raw aperture weights.

        Parameters
        ----------
        weights : array_like
            Bbox-tight aperture weights with shape ``self.shape``.
        image : array_like
            Two-dimensional image to sample.
        mask : array_like of bool, optional
            Boolean image mask with the same shape as ``image``. ``True``
            pixels contribute zero weight.
        return_npix : bool, optional
            If `True`, return ``(apsum, npix)``. If `False`, return only
            ``apsum``.
        validate : bool, optional
            If `False`, trust that ``weights``, ``image``, and ``mask`` are
            already valid and aligned.

        Returns
        -------
        apsum : float
            Sum of ``image * weight`` over the overlapping, unmasked footprint.
            Accumulation uses `float64`, even when both the image and stored
            weights are `float32`.
        npix : float
            Sum of the contributing aperture weights. This can be smaller than
            the analytic aperture area when the aperture is clipped by the
            image edge or by ``mask``. Accumulation uses `float64`. Returned
            only when ``return_npix=True``.
        """
        w_full = _validate_weights(weights, self) if validate else np.asarray(weights)
        arr = np.asarray(image)
        if validate and arr.ndim != 2:
            raise ValueError("data must be a 2-D array")
        overlap = self.overlap_slices(arr.shape)
        if overlap is None:
            return (0.0, 0.0) if return_npix else 0.0
        bad = (
            None
            if mask is None
            else (validate_mask(mask, arr.shape) if validate else mask)
        )
        data_slices, mask_slices = overlap
        weights = w_full[mask_slices].astype(np.float64, copy=False)
        if bad is not None:
            weights = weights.copy()
            weights[bad[data_slices]] = 0.0
        data_values = arr[data_slices].astype(np.float64, copy=False)
        if not return_npix:
            return _weighted_sum_float64(data_values, weights, return_sum_weights=False)
        return _weighted_sum_float64(data_values, weights, return_sum_weights=True)


def _sum_float64(values: np.ndarray) -> float:
    return float(rd.sum(np.ravel(values), validate=False))


def _weighted_sum_float64(
    values: np.ndarray, weights: np.ndarray, *, return_sum_weights: bool
) -> float | tuple[float, float]:
    result = rd.sum(
        np.ravel(values),
        weights=np.ravel(weights),
        return_sum_weights=return_sum_weights,
        validate=False,
    )
    if return_sum_weights:
        return float(result[0]), float(result[1])
    return float(result)


def _validate_weights(weights, bbox: BoundingBox) -> np.ndarray:
    """Return bbox-tight mask weights validated against ``bbox``.

    Parameters
    ----------
    weights : array_like
        Bbox-tight weights supplied by an aperture rasterizer.
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


def _slices_to_fits_section(y_slice: slice, x_slice: slice) -> str:
    return (
        f"[{_fits_start(x_slice.start)}:{x_slice.stop},"
        f"{_fits_start(y_slice.start)}:{y_slice.stop}]"
    )


def _fits_start(start: int | None) -> int:
    if start is None:
        return 1
    return start + 1


def _mask_dtype(weights: np.ndarray) -> np.dtype:
    if weights.dtype == np.float32:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _calculation_dtype(image: np.ndarray, weights: np.ndarray) -> np.dtype:
    result = np.result_type(image.dtype, weights.dtype)
    if result == np.dtype(np.float32):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _validate_imshape(shape: tuple[int, int]) -> tuple[int, int]:
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

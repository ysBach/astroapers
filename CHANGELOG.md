# Changelog
Release candidates also appear here. See
[GitHub Releases](https://github.com/ysBach/astroapers/releases) for completed releases.

## Unreleased

## 0.2.1 - 2026-11-20

- Include the Photutils BSD-3-Clause notice for the circle–pixel overlap code
  adapted from SEP in every wheel, the Python source archive, and the Rust crate.

## 0.2.0 - 2026-09-07

### Breaking changes

- `ap.positions[...] = ...` and `ap.positions = ...` previously allowed edits
  (thus apsum is at old position and causes trouble); now raise errors. Move
  centers with a new object, e.g. `aap.CircAp(ap.positions + (0.5, 0),
  r=ap.r)`.

### Fixed

- Elliptical/rectangular annuli with `validate=False, return_npix=False` could
  raise `TypeError`; now return proper `float64` sums, including empty results.
- Raw `_rust` calls could misread Fortran-order data; now reject non-C-contiguous
  image/path/mask arrays with `ValueError`. Use `np.ascontiguousarray()`;
  public Python image APIs still accept Fortran-order arrays.
- A `1 x 1` mask for a `3 x 3` image was accepted by off-image `BoundingBox`
  helpers; now raises `ValueError`. `validate=False` still skips this check.
- The FITS-section test called unavailable `astro_ndslice.slice_to_string`;
  now checks round trips with the published `slice_from_string` API.

### Improved

- C-order positions needed separate x/y copies; now both columns are contiguous
  views of one Fortran-order coordinate buffer.
- `weighted_cutout(mask=...)` copied its result before masking; now masks it
  directly. Inputs and results are unchanged.
- Annulus subtraction specified `dtype=np.float64`; now infers the same dtype.
  Values and shapes, including empty results, are unchanged.

## 0.1.3 (2026-07-15)

- Simplified the README and docs.
- Reduced temporary allocation during path-hole validation.
- Minor internal update: Removed the unused `circ_area` C-ABI symbol.

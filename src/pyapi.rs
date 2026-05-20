//! Core aperture overlap and summation kernels.
//!
//! Pixel coordinates follow the SEP/Photutils convention: pixel `(i, j)` is
//! centered at `(i, j)` and covers `[i - 0.5, i + 0.5]` in x and y.

use numpy::{
    Element, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::env;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::geometry::{
    bbox_in_frame, circ_bbox, circ_pixel_area, ellip_bbox, ellip_inner_outer_radius2,
    ellip_pixel_area, extent_bbox, rect_pixel_area, Ellip, SQRT_HALF,
};

const DEFAULT_PARALLEL_THRESHOLD: usize = 64;
static PARALLEL_THRESHOLD: AtomicUsize = AtomicUsize::new(0);

type BBoxColumns = (Vec<isize>, Vec<isize>, Vec<isize>, Vec<isize>);
type WeightManyResult = (
    Vec<Py<PyAny>>,
    Vec<isize>,
    Vec<isize>,
    Vec<isize>,
    Vec<isize>,
);

mod apsum;
mod bbox;
mod npix;
mod parallel_api;
mod path;
mod raster;
mod summation;
mod support;
mod weights;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    apsum::register(m)?;
    npix::register(m)?;
    bbox::register(m)?;
    parallel_api::register(m)?;
    path::register(m)?;
    weights::register(m)?;
    Ok(())
}

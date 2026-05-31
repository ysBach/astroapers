//! Rust-backed aperture overlap and summation utilities.

pub mod aperture;

mod geometry;
#[cfg(feature = "python")]
mod pyapi;

pub use aperture::{
    circle_bbox, circle_exact_weight_at_pixel, ellipse_bbox, ellipse_exact_weight_at_pixel,
    rectangle_bbox, rectangle_exact_weight_at_pixel, BoundingBox, Ellipse, Rectangle,
};

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyapi::register(m)
}

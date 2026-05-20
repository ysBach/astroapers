//! Python extension entrypoint for astroapers.

use pyo3::prelude::*;

mod geometry;
mod pyapi;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyapi::register(m)
}

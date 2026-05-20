use super::support::*;
use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_parallel_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(set_parallel_threshold, m)?)?;
    Ok(())
}

#[pyfunction]
fn get_parallel_threshold() -> usize {
    parallel_threshold()
}

#[pyfunction]
fn set_parallel_threshold(threshold: usize) -> PyResult<()> {
    if threshold == 0 {
        return Err(PyValueError::new_err("parallel threshold must be positive"));
    }
    PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
    Ok(())
}

use super::summation::*;
use super::support::*;
use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(npix_circ_exact, m)?)?;
    m.add_function(wrap_pyfunction!(npix_circ_center, m)?)?;
    m.add_function(wrap_pyfunction!(npix_ellip_exact, m)?)?;
    m.add_function(wrap_pyfunction!(npix_ellip_center, m)?)?;
    m.add_function(wrap_pyfunction!(npix_rect_exact, m)?)?;
    m.add_function(wrap_pyfunction!(npix_rect_center, m)?)?;
    Ok(())
}

#[pyfunction]
fn npix_circ_exact(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    Ok(npix_circ_slice(xs, ys, r, ny, nx))
}

#[pyfunction]
fn npix_circ_center(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    Ok(npix_circ_center_slice(xs, ys, r, ny, nx))
}

#[pyfunction]
fn npix_ellip_exact(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    Ok(npix_ellip_slice(
        xs,
        ys,
        a,
        b,
        theta.cos(),
        theta.sin(),
        ny,
        nx,
    ))
}

#[pyfunction]
fn npix_ellip_center(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    Ok(npix_ellip_center_slice(
        xs,
        ys,
        a,
        b,
        theta.cos(),
        theta.sin(),
        ny,
        nx,
    ))
}

#[pyfunction]
fn npix_rect_exact(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    Ok(npix_rect_slice(xs, ys, w, h, theta, ny, nx))
}

#[pyfunction]
fn npix_rect_center(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    Ok(npix_rect_center_slice(xs, ys, w, h, theta, ny, nx))
}

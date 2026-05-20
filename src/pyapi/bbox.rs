use super::raster::*;
use super::support::*;
use super::*;
use crate::geometry::{wedge_bbox, Wedge};

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bboxes_circ_many, m)?)?;
    m.add_function(wrap_pyfunction!(bboxes_ellip_many, m)?)?;
    m.add_function(wrap_pyfunction!(bboxes_rect_many, m)?)?;
    m.add_function(wrap_pyfunction!(bboxes_wedge_many, m)?)?;
    m.add_function(wrap_pyfunction!(bboxes_pill_many, m)?)?;
    m.add_function(wrap_pyfunction!(bboxes_pill_ann_many, m)?)?;
    Ok(())
}

#[pyfunction]
fn bboxes_circ_many(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<BBoxColumns> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(1, 1, r, "invalid circle bbox parameters")?;
    let mut ixmins = Vec::with_capacity(xs.len());
    let mut ixmaxs = Vec::with_capacity(xs.len());
    let mut iymins = Vec::with_capacity(xs.len());
    let mut iymaxs = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            return Err(PyValueError::new_err("x and y must be finite"));
        }
        let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r);
        ixmins.push(ixmin);
        ixmaxs.push(ixmax);
        iymins.push(iymin);
        iymaxs.push(iymax);
    }
    Ok((ixmins, ixmaxs, iymins, iymaxs))
}

#[pyfunction]
fn bboxes_ellip_many(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<BBoxColumns> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(1, 1, a, b, theta, "invalid ellipse bbox parameters")?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let mut ixmins = Vec::with_capacity(xs.len());
    let mut ixmaxs = Vec::with_capacity(xs.len());
    let mut iymins = Vec::with_capacity(xs.len());
    let mut iymaxs = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            return Err(PyValueError::new_err("x and y must be finite"));
        }
        let (ixmin, ixmax, iymin, iymax) = ellip_bbox(Ellip {
            x0: x,
            y0: y,
            a,
            b,
            cos_t,
            sin_t,
        });
        ixmins.push(ixmin);
        ixmaxs.push(ixmax);
        iymins.push(iymin);
        iymaxs.push(iymax);
    }
    Ok((ixmins, ixmaxs, iymins, iymaxs))
}

#[pyfunction]
fn bboxes_rect_many(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<BBoxColumns> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(1, 1, w, h, theta, "invalid rectangle bbox parameters")?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    let dx = (half_w * cos_t).abs() + (half_h * sin_t).abs();
    let dy = (half_w * sin_t).abs() + (half_h * cos_t).abs();
    let mut ixmins = Vec::with_capacity(xs.len());
    let mut ixmaxs = Vec::with_capacity(xs.len());
    let mut iymins = Vec::with_capacity(xs.len());
    let mut iymaxs = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            return Err(PyValueError::new_err("x and y must be finite"));
        }
        let (ixmin, ixmax, iymin, iymax) = extent_bbox(x, y, dx, dy);
        ixmins.push(ixmin);
        ixmaxs.push(ixmax);
        iymins.push(iymin);
        iymaxs.push(iymax);
    }
    Ok((ixmins, ixmaxs, iymins, iymaxs))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn bboxes_wedge_many(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
    theta_in: f64,
    dtheta_in: f64,
    theta_out: f64,
    dtheta_out: f64,
) -> PyResult<BBoxColumns> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_wedge_params(
        1,
        1,
        r_in,
        r_out,
        theta_in,
        dtheta_in,
        theta_out,
        dtheta_out,
        "invalid wedge bbox parameters",
    )?;
    let mut ixmins = Vec::with_capacity(xs.len());
    let mut ixmaxs = Vec::with_capacity(xs.len());
    let mut iymins = Vec::with_capacity(xs.len());
    let mut iymaxs = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            return Err(PyValueError::new_err("x and y must be finite"));
        }
        let (ixmin, ixmax, iymin, iymax) = wedge_bbox(Wedge {
            x0: x,
            y0: y,
            r_in,
            r_out,
            theta_in,
            dtheta_in,
            theta_out,
            dtheta_out,
        });
        ixmins.push(ixmin);
        ixmaxs.push(ixmax);
        iymins.push(iymin);
        iymaxs.push(iymax);
    }
    Ok((ixmins, ixmaxs, iymins, iymaxs))
}

#[pyfunction]
fn bboxes_pill_many(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<BBoxColumns> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_pill_params(1, 1, w, a, b, theta, "invalid pill bbox parameters")?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let mut ixmins = Vec::with_capacity(xs.len());
    let mut ixmaxs = Vec::with_capacity(xs.len());
    let mut iymins = Vec::with_capacity(xs.len());
    let mut iymaxs = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            return Err(PyValueError::new_err("x and y must be finite"));
        }
        let (ixmin, ixmax, iymin, iymax) = pill_bbox(x, y, w, a, b, cos_t, sin_t);
        ixmins.push(ixmin);
        ixmaxs.push(ixmax);
        iymins.push(iymin);
        iymaxs.push(iymax);
    }
    Ok((ixmins, ixmaxs, iymins, iymaxs))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn bboxes_pill_ann_many(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w_in: f64,
    a_in: f64,
    b_in: f64,
    w_out: f64,
    a_out: f64,
    b_out: f64,
    theta_in: f64,
    theta_out: f64,
) -> PyResult<BBoxColumns> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_pill_ann_params(
        1,
        1,
        w_in,
        a_in,
        b_in,
        w_out,
        a_out,
        b_out,
        theta_in,
        theta_out,
        "invalid pill annulus bbox parameters",
    )?;
    let cos_out = theta_out.cos();
    let sin_out = theta_out.sin();
    let mut ixmins = Vec::with_capacity(xs.len());
    let mut ixmaxs = Vec::with_capacity(xs.len());
    let mut iymins = Vec::with_capacity(xs.len());
    let mut iymaxs = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            return Err(PyValueError::new_err("x and y must be finite"));
        }
        let (ixmin, ixmax, iymin, iymax) = pill_bbox(x, y, w_out, a_out, b_out, cos_out, sin_out);
        ixmins.push(ixmin);
        ixmaxs.push(ixmax);
        iymins.push(iymin);
        iymaxs.push(iymax);
    }
    Ok((ixmins, ixmaxs, iymins, iymaxs))
}

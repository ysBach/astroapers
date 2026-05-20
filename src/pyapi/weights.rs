#![allow(clippy::too_many_arguments)]

use super::raster::*;
use super::support::*;
use super::*;
use crate::geometry::{wedge_bbox, Wedge};

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(weights_circ_exact, m)?)?;
    m.add_function(wrap_pyfunction!(weights_circ_ann_exact, m)?)?;
    m.add_function(wrap_pyfunction!(weights_circ_exact_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_circ_ann_exact_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_ellip_exact_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_ellip_ann_exact_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_rect_exact_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_rect_ann_exact_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_wedge_exact_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_pill_exact_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_pill_ann_exact_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_circ_center_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_circ_ann_center_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_ellip_center_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_ellip_ann_center_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_rect_center_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_rect_ann_center_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_wedge_center_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_pill_center_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_pill_ann_center_many, m)?)?;
    m.add_function(wrap_pyfunction!(weights_ellip_exact, m)?)?;
    m.add_function(wrap_pyfunction!(weights_ellip_ann_exact, m)?)?;
    m.add_function(wrap_pyfunction!(weights_rect_exact, m)?)?;
    m.add_function(wrap_pyfunction!(weights_rect_ann_exact, m)?)?;
    m.add_function(wrap_pyfunction!(weights_wedge_exact, m)?)?;
    m.add_function(wrap_pyfunction!(weights_pill_exact, m)?)?;
    m.add_function(wrap_pyfunction!(weights_pill_ann_exact, m)?)?;
    m.add_function(wrap_pyfunction!(weights_circ_center, m)?)?;
    m.add_function(wrap_pyfunction!(weights_circ_ann_center, m)?)?;
    m.add_function(wrap_pyfunction!(weights_ellip_center, m)?)?;
    m.add_function(wrap_pyfunction!(weights_ellip_ann_center, m)?)?;
    m.add_function(wrap_pyfunction!(weights_rect_center, m)?)?;
    m.add_function(wrap_pyfunction!(weights_rect_ann_center, m)?)?;
    m.add_function(wrap_pyfunction!(weights_wedge_center, m)?)?;
    m.add_function(wrap_pyfunction!(weights_pill_center, m)?)?;
    m.add_function(wrap_pyfunction!(weights_pill_ann_center, m)?)?;
    Ok(())
}

#[pyfunction]
pub(in crate::pyapi) fn weights_circ_exact(
    x: f64,
    y: f64,
    r: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_circ_params(ny, nx, r, "invalid circle weights parameters")?;
    validate_weights_shape(x, y, ny, nx, "invalid circle weights shape")?;
    weights_circ_exact_values(x, y, r, x_min, y_min, ny, nx)
}

#[pyfunction]
fn weights_circ_ann_exact(
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_circ_ann_params(
        ny,
        nx,
        r_in,
        r_out,
        "invalid circular annulus weights parameters",
    )?;
    validate_weights_shape(x, y, ny, nx, "invalid circular annulus weights shape")?;
    weights_circ_ann_exact_values(x, y, r_in, r_out, x_min, y_min, ny, nx)
}

#[pyfunction]
fn weights_circ_exact_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid circle weights shape")?;
    validate_circ_params(1, 1, r, "invalid circle weights parameters")?;

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_circ_exact_values(x, y, r, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_circ_ann_exact_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid circular annulus weights shape")?;
    validate_circ_ann_params(
        1,
        1,
        r_in,
        r_out,
        "invalid circular annulus weights parameters",
    )?;

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r_out);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_circ_ann_exact_values(x, y, r_in, r_out, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_ellip_exact_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid ellipse weights shape")?;
    validate_ellip_params(1, 1, a, b, theta, "invalid ellipse weights parameters")?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    let build = |i: usize| -> PyResult<_> {
        let ell = Ellip {
            x0: xs[i],
            y0: ys[i],
            a,
            b,
            cos_t,
            sin_t,
        };
        let (ixmin, ixmax, iymin, iymax) = ellip_bbox(ell);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_ellip_exact_values(ell, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_ellip_ann_exact_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a_in: f64,
    b_in: f64,
    a_out: f64,
    b_out: f64,
    theta_in: f64,
    theta_out: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid elliptical annulus weights shape")?;
    validate_ellip_ann_params(
        1,
        1,
        a_in,
        b_in,
        a_out,
        b_out,
        theta_in,
        theta_out,
        "invalid elliptical annulus weights parameters",
    )?;
    let cos_in = theta_in.cos();
    let sin_in = theta_in.sin();
    let cos_out = theta_out.cos();
    let sin_out = theta_out.sin();

    let build = |i: usize| -> PyResult<_> {
        let outer = Ellip {
            x0: xs[i],
            y0: ys[i],
            a: a_out,
            b: b_out,
            cos_t: cos_out,
            sin_t: sin_out,
        };
        let inner = Ellip {
            x0: xs[i],
            y0: ys[i],
            a: a_in,
            b: b_in,
            cos_t: cos_in,
            sin_t: sin_in,
        };
        let (ixmin, ixmax, iymin, iymax) = ellip_bbox(outer);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_ellip_ann_exact_values(outer, inner, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_rect_exact_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid rectangle weights shape")?;
    validate_rect_params(1, 1, w, h, theta, "invalid rectangle weights parameters")?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    let dx = (half_w * cos_t).abs() + (half_h * sin_t).abs();
    let dy = (half_w * sin_t).abs() + (half_h * cos_t).abs();

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = extent_bbox(x, y, dx, dy);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_rect_exact_values(x, y, half_w, half_h, cos_t, sin_t, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_rect_ann_exact_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w_in: f64,
    h_in: f64,
    w_out: f64,
    h_out: f64,
    theta_in: f64,
    theta_out: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid rectangular annulus weights shape")?;
    validate_rect_ann_params(
        1,
        1,
        w_in,
        h_in,
        w_out,
        h_out,
        theta_in,
        theta_out,
        "invalid rectangular annulus weights parameters",
    )?;
    let cos_in = theta_in.cos();
    let sin_in = theta_in.sin();
    let cos_out = theta_out.cos();
    let sin_out = theta_out.sin();
    let half_w_in = 0.5 * w_in;
    let half_h_in = 0.5 * h_in;
    let half_w_out = 0.5 * w_out;
    let half_h_out = 0.5 * h_out;
    let dx = (half_w_out * cos_out).abs() + (half_h_out * sin_out).abs();
    let dy = (half_w_out * sin_out).abs() + (half_h_out * cos_out).abs();

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = extent_bbox(x, y, dx, dy);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_rect_ann_exact_values(
                x, y, half_w_in, half_h_in, half_w_out, half_h_out, cos_in, sin_in, cos_out,
                sin_out, ixmin, iymin, ny, nx,
            )?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn weights_wedge_exact_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
    theta_in: f64,
    dtheta_in: f64,
    theta_out: f64,
    dtheta_out: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid wedge weights shape")?;
    validate_wedge_params(
        1,
        1,
        r_in,
        r_out,
        theta_in,
        dtheta_in,
        theta_out,
        dtheta_out,
        "invalid wedge weights parameters",
    )?;

    let build = |i: usize| -> PyResult<_> {
        let wedge = Wedge {
            x0: xs[i],
            y0: ys[i],
            r_in,
            r_out,
            theta_in,
            dtheta_in,
            theta_out,
            dtheta_out,
        };
        let (ixmin, ixmax, iymin, iymax) = wedge_bbox(wedge);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_wedge_exact_values(wedge, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_pill_exact_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid pill weights shape")?;
    validate_pill_params(1, 1, w, a, b, theta, "invalid pill weights parameters")?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = pill_bbox(x, y, w, a, b, cos_t, sin_t);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_pill_exact_values(x, y, w, a, b, cos_t, sin_t, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_pill_ann_exact_many(
    py: Python<'_>,
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
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid pill annulus weights shape")?;
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
        "invalid pill annulus weights parameters",
    )?;
    let cos_in = theta_in.cos();
    let sin_in = theta_in.sin();
    let cos_out = theta_out.cos();
    let sin_out = theta_out.sin();

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = pill_bbox(x, y, w_out, a_out, b_out, cos_out, sin_out);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_pill_ann_exact_values(
                x, y, w_in, a_in, b_in, w_out, a_out, b_out, cos_in, sin_in, cos_out, sin_out,
                ixmin, iymin, ny, nx,
            )?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_circ_center_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid circle center weights shape")?;
    validate_circ_params(1, 1, r, "invalid circle center weights parameters")?;

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_circ_center_values(x, y, r, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_circ_ann_center_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid circular annulus center weights shape")?;
    validate_circ_ann_params(
        1,
        1,
        r_in,
        r_out,
        "invalid circular annulus center weights parameters",
    )?;

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r_out);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_circ_ann_center_values(x, y, r_in, r_out, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_ellip_center_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid ellipse center weights shape")?;
    validate_ellip_params(
        1,
        1,
        a,
        b,
        theta,
        "invalid ellipse center weights parameters",
    )?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    let build = |i: usize| -> PyResult<_> {
        let ell = Ellip {
            x0: xs[i],
            y0: ys[i],
            a,
            b,
            cos_t,
            sin_t,
        };
        let (ixmin, ixmax, iymin, iymax) = ellip_bbox(ell);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_ellip_center_values(ell, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_ellip_ann_center_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a_in: f64,
    b_in: f64,
    a_out: f64,
    b_out: f64,
    theta_in: f64,
    theta_out: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid elliptical annulus center weights shape")?;
    validate_ellip_ann_params(
        1,
        1,
        a_in,
        b_in,
        a_out,
        b_out,
        theta_in,
        theta_out,
        "invalid elliptical annulus center weights parameters",
    )?;
    let cos_in = theta_in.cos();
    let sin_in = theta_in.sin();
    let cos_out = theta_out.cos();
    let sin_out = theta_out.sin();

    let build = |i: usize| -> PyResult<_> {
        let outer = Ellip {
            x0: xs[i],
            y0: ys[i],
            a: a_out,
            b: b_out,
            cos_t: cos_out,
            sin_t: sin_out,
        };
        let inner = Ellip {
            x0: xs[i],
            y0: ys[i],
            a: a_in,
            b: b_in,
            cos_t: cos_in,
            sin_t: sin_in,
        };
        let (ixmin, ixmax, iymin, iymax) = ellip_bbox(outer);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_ellip_ann_center_values(outer, inner, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_rect_center_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid rectangle center weights shape")?;
    validate_rect_params(
        1,
        1,
        w,
        h,
        theta,
        "invalid rectangle center weights parameters",
    )?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    let dx = (half_w * cos_t).abs() + (half_h * sin_t).abs();
    let dy = (half_w * sin_t).abs() + (half_h * cos_t).abs();

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = extent_bbox(x, y, dx, dy);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_rect_center_values(x, y, half_w, half_h, cos_t, sin_t, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_rect_ann_center_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w_in: f64,
    h_in: f64,
    w_out: f64,
    h_out: f64,
    theta_in: f64,
    theta_out: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid rectangular annulus center weights shape")?;
    validate_rect_ann_params(
        1,
        1,
        w_in,
        h_in,
        w_out,
        h_out,
        theta_in,
        theta_out,
        "invalid rectangular annulus center weights parameters",
    )?;
    let cos_in = theta_in.cos();
    let sin_in = theta_in.sin();
    let cos_out = theta_out.cos();
    let sin_out = theta_out.sin();
    let half_w_in = 0.5 * w_in;
    let half_h_in = 0.5 * h_in;
    let half_w_out = 0.5 * w_out;
    let half_h_out = 0.5 * h_out;
    let dx = (half_w_out * cos_out).abs() + (half_h_out * sin_out).abs();
    let dy = (half_w_out * sin_out).abs() + (half_h_out * cos_out).abs();

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = extent_bbox(x, y, dx, dy);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_rect_ann_center_values(
                x, y, half_w_in, half_h_in, half_w_out, half_h_out, cos_in, sin_in, cos_out,
                sin_out, ixmin, iymin, ny, nx,
            )?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn weights_wedge_center_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
    theta_in: f64,
    dtheta_in: f64,
    theta_out: f64,
    dtheta_out: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid wedge center weights shape")?;
    validate_wedge_params(
        1,
        1,
        r_in,
        r_out,
        theta_in,
        dtheta_in,
        theta_out,
        dtheta_out,
        "invalid wedge center weights parameters",
    )?;

    let build = |i: usize| -> PyResult<_> {
        let wedge = Wedge {
            x0: xs[i],
            y0: ys[i],
            r_in,
            r_out,
            theta_in,
            dtheta_in,
            theta_out,
            dtheta_out,
        };
        let (ixmin, ixmax, iymin, iymax) = wedge_bbox(wedge);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_wedge_center_values(wedge, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_pill_center_many(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid pill center weights shape")?;
    validate_pill_params(
        1,
        1,
        w,
        a,
        b,
        theta,
        "invalid pill center weights parameters",
    )?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = pill_bbox(x, y, w, a, b, cos_t, sin_t);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_pill_center_values(x, y, w, a, b, cos_t, sin_t, ixmin, iymin, ny, nx)?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
fn weights_pill_ann_center_many(
    py: Python<'_>,
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
) -> PyResult<WeightManyResult> {
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs, ys, "invalid pill annulus center weights shape")?;
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
        "invalid pill annulus center weights parameters",
    )?;
    let cos_in = theta_in.cos();
    let sin_in = theta_in.sin();
    let cos_out = theta_out.cos();
    let sin_out = theta_out.sin();

    let build = |i: usize| -> PyResult<_> {
        let x = xs[i];
        let y = ys[i];
        let (ixmin, ixmax, iymin, iymax) = pill_bbox(x, y, w_out, a_out, b_out, cos_out, sin_out);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        Ok((
            weights_pill_ann_center_values(
                x, y, w_in, a_in, b_in, w_out, a_out, b_out, cos_in, sin_in, cos_out, sin_out,
                ixmin, iymin, ny, nx,
            )?,
            ixmin,
            ixmax,
            iymin,
            iymax,
        ))
    };
    weights_from_builders(py, xs.len(), build)
}

#[pyfunction]
pub(in crate::pyapi) fn weights_ellip_exact(
    x: f64,
    y: f64,
    a: f64,
    b: f64,
    theta: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse weights parameters")?;
    validate_weights_shape(x, y, ny, nx, "invalid ellipse weights shape")?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let ell = Ellip {
        x0: x,
        y0: y,
        a,
        b,
        cos_t,
        sin_t,
    };
    weights_ellip_exact_values(ell, x_min, y_min, ny, nx)
}

#[pyfunction]
fn weights_ellip_ann_exact(
    x: f64,
    y: f64,
    a_in: f64,
    b_in: f64,
    a_out: f64,
    b_out: f64,
    theta_in: f64,
    theta_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_ellip_ann_params(
        ny,
        nx,
        a_in,
        b_in,
        a_out,
        b_out,
        theta_in,
        theta_out,
        "invalid elliptical annulus weights parameters",
    )?;
    validate_weights_shape(x, y, ny, nx, "invalid elliptical annulus weights shape")?;
    let outer = Ellip {
        x0: x,
        y0: y,
        a: a_out,
        b: b_out,
        cos_t: theta_out.cos(),
        sin_t: theta_out.sin(),
    };
    let inner = Ellip {
        x0: x,
        y0: y,
        a: a_in,
        b: b_in,
        cos_t: theta_in.cos(),
        sin_t: theta_in.sin(),
    };
    weights_ellip_ann_exact_values(outer, inner, x_min, y_min, ny, nx)
}

#[pyfunction]
pub(in crate::pyapi) fn weights_rect_exact(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    theta: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle weights parameters")?;
    validate_weights_shape(x, y, ny, nx, "invalid rectangle weights shape")?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    weights_rect_exact_values(x, y, half_w, half_h, cos_t, sin_t, x_min, y_min, ny, nx)
}

#[pyfunction]
fn weights_rect_ann_exact(
    x: f64,
    y: f64,
    w_in: f64,
    h_in: f64,
    w_out: f64,
    h_out: f64,
    theta_in: f64,
    theta_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_rect_ann_params(
        ny,
        nx,
        w_in,
        h_in,
        w_out,
        h_out,
        theta_in,
        theta_out,
        "invalid rectangular annulus weights parameters",
    )?;
    validate_weights_shape(x, y, ny, nx, "invalid rectangular annulus weights shape")?;
    weights_rect_ann_exact_values(
        x,
        y,
        0.5 * w_in,
        0.5 * h_in,
        0.5 * w_out,
        0.5 * h_out,
        theta_in.cos(),
        theta_in.sin(),
        theta_out.cos(),
        theta_out.sin(),
        x_min,
        y_min,
        ny,
        nx,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn weights_wedge_exact(
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
    theta_in: f64,
    dtheta_in: f64,
    theta_out: f64,
    dtheta_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_wedge_params(
        ny,
        nx,
        r_in,
        r_out,
        theta_in,
        dtheta_in,
        theta_out,
        dtheta_out,
        "invalid wedge weights parameters",
    )?;
    validate_weights_shape(x, y, ny, nx, "invalid wedge weights shape")?;
    weights_wedge_exact_values(
        Wedge {
            x0: x,
            y0: y,
            r_in,
            r_out,
            theta_in,
            dtheta_in,
            theta_out,
            dtheta_out,
        },
        x_min,
        y_min,
        ny,
        nx,
    )
}

#[pyfunction]
fn weights_pill_exact(
    x: f64,
    y: f64,
    w: f64,
    a: f64,
    b: f64,
    theta: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_pill_params(ny, nx, w, a, b, theta, "invalid pill weights parameters")?;
    validate_weights_shape(x, y, ny, nx, "invalid pill weights shape")?;
    weights_pill_exact_values(
        x,
        y,
        w,
        a,
        b,
        theta.cos(),
        theta.sin(),
        x_min,
        y_min,
        ny,
        nx,
    )
}

#[pyfunction]
fn weights_pill_ann_exact(
    x: f64,
    y: f64,
    w_in: f64,
    a_in: f64,
    b_in: f64,
    w_out: f64,
    a_out: f64,
    b_out: f64,
    theta_in: f64,
    theta_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_pill_ann_params(
        ny,
        nx,
        w_in,
        a_in,
        b_in,
        w_out,
        a_out,
        b_out,
        theta_in,
        theta_out,
        "invalid pill annulus weights parameters",
    )?;
    validate_weights_shape(x, y, ny, nx, "invalid pill annulus weights shape")?;
    weights_pill_ann_exact_values(
        x,
        y,
        w_in,
        a_in,
        b_in,
        w_out,
        a_out,
        b_out,
        theta_in.cos(),
        theta_in.sin(),
        theta_out.cos(),
        theta_out.sin(),
        x_min,
        y_min,
        ny,
        nx,
    )
}

#[pyfunction]
fn weights_circ_center(
    x: f64,
    y: f64,
    r: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_circ_params(ny, nx, r, "invalid circle center weights parameters")?;
    validate_weights_shape(x, y, ny, nx, "invalid circle center weights shape")?;
    weights_circ_center_values(x, y, r, x_min, y_min, ny, nx)
}

#[pyfunction]
fn weights_circ_ann_center(
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_circ_ann_params(
        ny,
        nx,
        r_in,
        r_out,
        "invalid circular annulus center weights parameters",
    )?;
    validate_weights_shape(
        x,
        y,
        ny,
        nx,
        "invalid circular annulus center weights shape",
    )?;
    weights_circ_ann_center_values(x, y, r_in, r_out, x_min, y_min, ny, nx)
}

#[pyfunction]
fn weights_ellip_center(
    x: f64,
    y: f64,
    a: f64,
    b: f64,
    theta: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_ellip_params(
        ny,
        nx,
        a,
        b,
        theta,
        "invalid ellipse center weights parameters",
    )?;
    validate_weights_shape(x, y, ny, nx, "invalid ellipse center weights shape")?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    weights_ellip_center_values(
        Ellip {
            x0: x,
            y0: y,
            a,
            b,
            cos_t,
            sin_t,
        },
        x_min,
        y_min,
        ny,
        nx,
    )
}

#[pyfunction]
fn weights_ellip_ann_center(
    x: f64,
    y: f64,
    a_in: f64,
    b_in: f64,
    a_out: f64,
    b_out: f64,
    theta_in: f64,
    theta_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_ellip_ann_params(
        ny,
        nx,
        a_in,
        b_in,
        a_out,
        b_out,
        theta_in,
        theta_out,
        "invalid elliptical annulus center weights parameters",
    )?;
    validate_weights_shape(
        x,
        y,
        ny,
        nx,
        "invalid elliptical annulus center weights shape",
    )?;
    weights_ellip_ann_center_values(
        Ellip {
            x0: x,
            y0: y,
            a: a_out,
            b: b_out,
            cos_t: theta_out.cos(),
            sin_t: theta_out.sin(),
        },
        Ellip {
            x0: x,
            y0: y,
            a: a_in,
            b: b_in,
            cos_t: theta_in.cos(),
            sin_t: theta_in.sin(),
        },
        x_min,
        y_min,
        ny,
        nx,
    )
}

#[pyfunction]
fn weights_rect_center(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    theta: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_rect_params(
        ny,
        nx,
        w,
        h,
        theta,
        "invalid rectangle center weights parameters",
    )?;
    validate_weights_shape(x, y, ny, nx, "invalid rectangle center weights shape")?;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    weights_rect_center_values(x, y, half_w, half_h, cos_t, sin_t, x_min, y_min, ny, nx)
}

#[pyfunction]
fn weights_rect_ann_center(
    x: f64,
    y: f64,
    w_in: f64,
    h_in: f64,
    w_out: f64,
    h_out: f64,
    theta_in: f64,
    theta_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_rect_ann_params(
        ny,
        nx,
        w_in,
        h_in,
        w_out,
        h_out,
        theta_in,
        theta_out,
        "invalid rectangular annulus center weights parameters",
    )?;
    validate_weights_shape(
        x,
        y,
        ny,
        nx,
        "invalid rectangular annulus center weights shape",
    )?;
    weights_rect_ann_center_values(
        x,
        y,
        0.5 * w_in,
        0.5 * h_in,
        0.5 * w_out,
        0.5 * h_out,
        theta_in.cos(),
        theta_in.sin(),
        theta_out.cos(),
        theta_out.sin(),
        x_min,
        y_min,
        ny,
        nx,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn weights_wedge_center(
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
    theta_in: f64,
    dtheta_in: f64,
    theta_out: f64,
    dtheta_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_wedge_params(
        ny,
        nx,
        r_in,
        r_out,
        theta_in,
        dtheta_in,
        theta_out,
        dtheta_out,
        "invalid wedge center weights parameters",
    )?;
    validate_weights_shape(x, y, ny, nx, "invalid wedge center weights shape")?;
    weights_wedge_center_values(
        Wedge {
            x0: x,
            y0: y,
            r_in,
            r_out,
            theta_in,
            dtheta_in,
            theta_out,
            dtheta_out,
        },
        x_min,
        y_min,
        ny,
        nx,
    )
}

#[pyfunction]
fn weights_pill_center(
    x: f64,
    y: f64,
    w: f64,
    a: f64,
    b: f64,
    theta: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_pill_params(
        ny,
        nx,
        w,
        a,
        b,
        theta,
        "invalid pill center weights parameters",
    )?;
    validate_weights_shape(x, y, ny, nx, "invalid pill center weights shape")?;
    weights_pill_center_values(
        x,
        y,
        w,
        a,
        b,
        theta.cos(),
        theta.sin(),
        x_min,
        y_min,
        ny,
        nx,
    )
}

#[pyfunction]
fn weights_pill_ann_center(
    x: f64,
    y: f64,
    w_in: f64,
    a_in: f64,
    b_in: f64,
    w_out: f64,
    a_out: f64,
    b_out: f64,
    theta_in: f64,
    theta_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    validate_pill_ann_params(
        ny,
        nx,
        w_in,
        a_in,
        b_in,
        w_out,
        a_out,
        b_out,
        theta_in,
        theta_out,
        "invalid pill annulus center weights parameters",
    )?;
    validate_weights_shape(x, y, ny, nx, "invalid pill annulus center weights shape")?;
    weights_pill_ann_center_values(
        x,
        y,
        w_in,
        a_in,
        b_in,
        w_out,
        a_out,
        b_out,
        theta_in.cos(),
        theta_in.sin(),
        theta_out.cos(),
        theta_out.sin(),
        x_min,
        y_min,
        ny,
        nx,
    )
}

#![allow(clippy::too_many_arguments)]

use super::raster::*;
use super::support::*;
use super::*;
use crate::geometry::{build_validated_path, path_bbox, path_center_weight, path_pixel_area};

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(weights_path_exact, m)?)?;
    m.add_function(wrap_pyfunction!(weights_path_center, m)?)?;
    m.add_function(wrap_pyfunction!(weights_path_exact_one, m)?)?;
    m.add_function(wrap_pyfunction!(weights_path_center_one, m)?)?;
    m.add_function(wrap_pyfunction!(bboxes_path, m)?)?;
    Ok(())
}

fn parse_path_inputs<'a>(
    kinds: &'a PyReadonlyArray1<'a, i8>,
    data: &'a PyReadonlyArray2<'a, f64>,
) -> PyResult<(Vec<i8>, Vec<[f64; 6]>)> {
    let kinds_slice = kinds
        .as_slice()
        .map_err(|_| PyValueError::new_err("kinds must be a contiguous int8 array"))?;
    let data_flat = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be a contiguous float64 array"))?;
    let n = kinds_slice.len();
    let shape = data.shape();
    if shape[0] != n || shape[1] != 6 {
        return Err(PyValueError::new_err(
            "data must have shape (N, 6) where N == len(kinds)",
        ));
    }
    let kinds_vec: Vec<i8> = kinds_slice.to_vec();
    let mut rows: Vec<[f64; 6]> = Vec::with_capacity(n);
    for i in 0..n {
        rows.push([
            data_flat[i * 6],
            data_flat[i * 6 + 1],
            data_flat[i * 6 + 2],
            data_flat[i * 6 + 3],
            data_flat[i * 6 + 4],
            data_flat[i * 6 + 5],
        ]);
    }
    Ok((kinds_vec, rows))
}

#[pyfunction]
fn weights_path_exact(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    kinds: PyReadonlyArray1<'_, i8>,
    data: PyReadonlyArray2<'_, f64>,
) -> PyResult<WeightManyResult> {
    let xs_slice = vector_slice(&xs, "x")?;
    let ys_slice = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs_slice, ys_slice, "x and y must be finite float64 arrays")?;
    let (kinds_vec, rows) = parse_path_inputs(&kinds, &data)?;
    let path = build_validated_path(&kinds_vec, &rows).map_err(PyValueError::new_err)?;
    let n = xs_slice.len();
    let build = |i: usize| -> PyResult<(Vec<f64>, isize, isize, isize, isize)> {
        let x = xs_slice[i];
        let y = ys_slice[i];
        let (ixmin, ixmax, iymin, iymax) = path_bbox(&path, x, y);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        let mut weights = vec![0.0_f64; ny * nx];
        for row in 0..ny {
            for col in 0..nx {
                let ix = ixmin + col as isize;
                let iy = iymin + row as isize;
                weights[row * nx + col] = path_pixel_area(&path, x, y, ix, iy);
            }
        }
        Ok((weights, ixmin, ixmax, iymin, iymax))
    };
    weights_from_builders(py, n, build)
}

#[pyfunction]
fn weights_path_center(
    py: Python<'_>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    kinds: PyReadonlyArray1<'_, i8>,
    data: PyReadonlyArray2<'_, f64>,
) -> PyResult<WeightManyResult> {
    let xs_slice = vector_slice(&xs, "x")?;
    let ys_slice = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs_slice, ys_slice, "x and y must be finite float64 arrays")?;
    let (kinds_vec, rows) = parse_path_inputs(&kinds, &data)?;
    let path = build_validated_path(&kinds_vec, &rows).map_err(PyValueError::new_err)?;
    let n = xs_slice.len();
    let build = |i: usize| -> PyResult<(Vec<f64>, isize, isize, isize, isize)> {
        let x = xs_slice[i];
        let y = ys_slice[i];
        let (ixmin, ixmax, iymin, iymax) = path_bbox(&path, x, y);
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        let mut weights = vec![0.0_f64; ny * nx];
        for row in 0..ny {
            for col in 0..nx {
                let ix = ixmin + col as isize;
                let iy = iymin + row as isize;
                weights[row * nx + col] = path_center_weight(&path, x, y, ix, iy);
            }
        }
        Ok((weights, ixmin, ixmax, iymin, iymax))
    };
    weights_from_builders(py, n, build)
}

#[pyfunction]
fn weights_path_exact_one(
    x: f64,
    y: f64,
    kinds: PyReadonlyArray1<'_, i8>,
    data: PyReadonlyArray2<'_, f64>,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let (kinds_vec, rows) = parse_path_inputs(&kinds, &data)?;
    let path = build_validated_path(&kinds_vec, &rows).map_err(PyValueError::new_err)?;
    let mut weights = vec![0.0_f64; ny * nx];
    for row in 0..ny {
        for col in 0..nx {
            let ix = x_min + col as isize;
            let iy = y_min + row as isize;
            weights[row * nx + col] = path_pixel_area(&path, x, y, ix, iy);
        }
    }
    Ok(weights)
}

#[pyfunction]
fn weights_path_center_one(
    x: f64,
    y: f64,
    kinds: PyReadonlyArray1<'_, i8>,
    data: PyReadonlyArray2<'_, f64>,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let (kinds_vec, rows) = parse_path_inputs(&kinds, &data)?;
    let path = build_validated_path(&kinds_vec, &rows).map_err(PyValueError::new_err)?;
    let mut weights = vec![0.0_f64; ny * nx];
    for row in 0..ny {
        for col in 0..nx {
            let ix = x_min + col as isize;
            let iy = y_min + row as isize;
            weights[row * nx + col] = path_center_weight(&path, x, y, ix, iy);
        }
    }
    Ok(weights)
}

#[pyfunction]
fn bboxes_path(
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    kinds: PyReadonlyArray1<'_, i8>,
    data: PyReadonlyArray2<'_, f64>,
) -> PyResult<BBoxColumns> {
    let xs_slice = vector_slice(&xs, "x")?;
    let ys_slice = vector_slice(&ys, "y")?;
    validate_many_weights_inputs(xs_slice, ys_slice, "x and y must be finite float64 arrays")?;
    let (kinds_vec, rows) = parse_path_inputs(&kinds, &data)?;
    let path = build_validated_path(&kinds_vec, &rows).map_err(PyValueError::new_err)?;
    let n = xs_slice.len();
    let mut ixmins = vec![0isize; n];
    let mut ixmaxs = vec![0isize; n];
    let mut iymins = vec![0isize; n];
    let mut iymaxs = vec![0isize; n];
    for i in 0..n {
        let (ixmin, ixmax, iymin, iymax) = path_bbox(&path, xs_slice[i], ys_slice[i]);
        ixmins[i] = ixmin;
        ixmaxs[i] = ixmax;
        iymins[i] = iymin;
        iymaxs[i] = iymax;
    }
    Ok((ixmins, ixmaxs, iymins, iymaxs))
}

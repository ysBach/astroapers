use super::summation::*;
use super::support::*;
use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apsum_circ_exact, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_sum, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_sum_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_sum_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_sum_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_one, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_sum_one, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_sum_one_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_sum_one_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_sum_one_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_sum, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_sum_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_sum_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_sum_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_one, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_one_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_one_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_one_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_sum_one, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_sum_one_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_sum_one_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_ann_exact_sum_one_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_one_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_one_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_exact_one_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_center, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_center_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_center_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_circ_center_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_exact, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_exact_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_exact_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_exact_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_exact_one, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_exact_one_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_exact_one_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_exact_one_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_center, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_center_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_center_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_ellip_center_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_exact, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_exact_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_exact_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_exact_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_exact_one, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_exact_one_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_exact_one_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_exact_one_i16, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_center, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_center_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_center_i32, m)?)?;
    m.add_function(wrap_pyfunction!(apsum_rect_center_i16, m)?)?;
    Ok(())
}

#[pyfunction]
fn apsum_circ_exact(
    data: PyReadonlyArray2<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_exact_f32(
    data: PyReadonlyArray2<'_, f32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_exact_i32(
    data: PyReadonlyArray2<'_, i32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_exact_i16(
    data: PyReadonlyArray2<'_, i16>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_exact_sum(
    data: PyReadonlyArray2<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<Vec<f64>> {
    let (data, ny, nx) = data_slice(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_only_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_exact_sum_f32(
    data: PyReadonlyArray2<'_, f32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<Vec<f64>> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_only_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_exact_sum_i32(
    data: PyReadonlyArray2<'_, i32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<Vec<f64>> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_only_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_exact_sum_i16(
    data: PyReadonlyArray2<'_, i16>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<Vec<f64>> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_only_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_exact_one(
    data: PyReadonlyArray2<'_, f64>,
    x: f64,
    y: f64,
    r: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice(&data)?;
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_one(data, ny, nx, x, y, r)
}

#[pyfunction]
fn apsum_circ_exact_sum_one(
    data: PyReadonlyArray2<'_, f64>,
    x: f64,
    y: f64,
    r: f64,
) -> PyResult<f64> {
    let (data, ny, nx) = data_slice(&data)?;
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_only_one(data, ny, nx, x, y, r)
}

#[pyfunction]
fn apsum_circ_exact_sum_one_f32(
    data: PyReadonlyArray2<'_, f32>,
    x: f64,
    y: f64,
    r: f64,
) -> PyResult<f64> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_only_one(data, ny, nx, x, y, r)
}

#[pyfunction]
fn apsum_circ_exact_sum_one_i32(
    data: PyReadonlyArray2<'_, i32>,
    x: f64,
    y: f64,
    r: f64,
) -> PyResult<f64> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_only_one(data, ny, nx, x, y, r)
}

#[pyfunction]
fn apsum_circ_exact_sum_one_i16(
    data: PyReadonlyArray2<'_, i16>,
    x: f64,
    y: f64,
    r: f64,
) -> PyResult<f64> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_only_one(data, ny, nx, x, y, r)
}

#[pyfunction]
fn apsum_circ_ann_exact(
    data: PyReadonlyArray2<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_slice(data, ny, nx, xs, ys, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_f32(
    data: PyReadonlyArray2<'_, f32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_slice(data, ny, nx, xs, ys, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_i32(
    data: PyReadonlyArray2<'_, i32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_slice(data, ny, nx, xs, ys, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_i16(
    data: PyReadonlyArray2<'_, i16>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_slice(data, ny, nx, xs, ys, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_sum(
    data: PyReadonlyArray2<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
) -> PyResult<Vec<f64>> {
    let (data, ny, nx) = data_slice(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_only_slice(data, ny, nx, xs, ys, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_sum_f32(
    data: PyReadonlyArray2<'_, f32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
) -> PyResult<Vec<f64>> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_only_slice(data, ny, nx, xs, ys, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_sum_i32(
    data: PyReadonlyArray2<'_, i32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
) -> PyResult<Vec<f64>> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_only_slice(data, ny, nx, xs, ys, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_sum_i16(
    data: PyReadonlyArray2<'_, i16>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r_in: f64,
    r_out: f64,
) -> PyResult<Vec<f64>> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_only_slice(data, ny, nx, xs, ys, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_one(
    data: PyReadonlyArray2<'_, f64>,
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice(&data)?;
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_one(data, ny, nx, x, y, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_one_f32(
    data: PyReadonlyArray2<'_, f32>,
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_one(data, ny, nx, x, y, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_one_i32(
    data: PyReadonlyArray2<'_, i32>,
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_one(data, ny, nx, x, y, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_one_i16(
    data: PyReadonlyArray2<'_, i16>,
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_one(data, ny, nx, x, y, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_sum_one(
    data: PyReadonlyArray2<'_, f64>,
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
) -> PyResult<f64> {
    let (data, ny, nx) = data_slice(&data)?;
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_only_one(data, ny, nx, x, y, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_sum_one_f32(
    data: PyReadonlyArray2<'_, f32>,
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
) -> PyResult<f64> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_only_one(data, ny, nx, x, y, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_sum_one_i32(
    data: PyReadonlyArray2<'_, i32>,
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
) -> PyResult<f64> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_only_one(data, ny, nx, x, y, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_ann_exact_sum_one_i16(
    data: PyReadonlyArray2<'_, i16>,
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
) -> PyResult<f64> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    validate_circ_ann_params(ny, nx, r_in, r_out, "invalid circular annulus parameters")?;
    sum_circ_ann_only_one(data, ny, nx, x, y, r_in, r_out)
}

#[pyfunction]
fn apsum_circ_exact_one_f32(
    data: PyReadonlyArray2<'_, f32>,
    x: f64,
    y: f64,
    r: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_one(data, ny, nx, x, y, r)
}

#[pyfunction]
fn apsum_circ_exact_one_i32(
    data: PyReadonlyArray2<'_, i32>,
    x: f64,
    y: f64,
    r: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_one(data, ny, nx, x, y, r)
}

#[pyfunction]
fn apsum_circ_exact_one_i16(
    data: PyReadonlyArray2<'_, i16>,
    x: f64,
    y: f64,
    r: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_one(data, ny, nx, x, y, r)
}

#[pyfunction]
fn apsum_circ_center(
    data: PyReadonlyArray2<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_center_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_center_f32(
    data: PyReadonlyArray2<'_, f32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_center_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_center_i32(
    data: PyReadonlyArray2<'_, i32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_center_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_circ_center_i16(
    data: PyReadonlyArray2<'_, i16>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_circ_params(ny, nx, r, "invalid circle parameters")?;
    sum_circ_center_slice(data, ny, nx, xs, ys, r)
}

#[pyfunction]
fn apsum_ellip_exact(
    data: PyReadonlyArray2<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_slice(data, ny, nx, xs, ys, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_exact_f32(
    data: PyReadonlyArray2<'_, f32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_slice(data, ny, nx, xs, ys, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_exact_i32(
    data: PyReadonlyArray2<'_, i32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_slice(data, ny, nx, xs, ys, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_exact_i16(
    data: PyReadonlyArray2<'_, i16>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_slice(data, ny, nx, xs, ys, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_exact_one(
    data: PyReadonlyArray2<'_, f64>,
    x: f64,
    y: f64,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice(&data)?;
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_one(data, ny, nx, x, y, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_exact_one_f32(
    data: PyReadonlyArray2<'_, f32>,
    x: f64,
    y: f64,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_one(data, ny, nx, x, y, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_exact_one_i32(
    data: PyReadonlyArray2<'_, i32>,
    x: f64,
    y: f64,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_one(data, ny, nx, x, y, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_exact_one_i16(
    data: PyReadonlyArray2<'_, i16>,
    x: f64,
    y: f64,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_one(data, ny, nx, x, y, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_center(
    data: PyReadonlyArray2<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_center_slice(data, ny, nx, xs, ys, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_center_f32(
    data: PyReadonlyArray2<'_, f32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_center_slice(data, ny, nx, xs, ys, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_center_i32(
    data: PyReadonlyArray2<'_, i32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_center_slice(data, ny, nx, xs, ys, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_ellip_center_i16(
    data: PyReadonlyArray2<'_, i16>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_ellip_params(ny, nx, a, b, theta, "invalid ellipse parameters")?;
    sum_ellip_center_slice(data, ny, nx, xs, ys, a, b, theta.cos(), theta.sin())
}

#[pyfunction]
fn apsum_rect_exact(
    data: PyReadonlyArray2<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("data shape is too large"));
    }
    sum_rect_slice(data, ny, nx, xs, ys, w, h, theta)
}

#[pyfunction]
fn apsum_rect_exact_f32(
    data: PyReadonlyArray2<'_, f32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("data shape is too large"));
    }
    sum_rect_slice(data, ny, nx, xs, ys, w, h, theta)
}

#[pyfunction]
fn apsum_rect_exact_i32(
    data: PyReadonlyArray2<'_, i32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("data shape is too large"));
    }
    sum_rect_slice(data, ny, nx, xs, ys, w, h, theta)
}

#[pyfunction]
fn apsum_rect_exact_i16(
    data: PyReadonlyArray2<'_, i16>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("data shape is too large"));
    }
    sum_rect_slice(data, ny, nx, xs, ys, w, h, theta)
}

#[pyfunction]
fn apsum_rect_exact_one(
    data: PyReadonlyArray2<'_, f64>,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice(&data)?;
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    sum_rect_one(data, ny, nx, x, y, w, h, theta)
}

#[pyfunction]
fn apsum_rect_exact_one_f32(
    data: PyReadonlyArray2<'_, f32>,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    sum_rect_one(data, ny, nx, x, y, w, h, theta)
}

#[pyfunction]
fn apsum_rect_exact_one_i32(
    data: PyReadonlyArray2<'_, i32>,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    sum_rect_one(data, ny, nx, x, y, w, h, theta)
}

#[pyfunction]
fn apsum_rect_exact_one_i16(
    data: PyReadonlyArray2<'_, i16>,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(f64, f64)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    sum_rect_one(data, ny, nx, x, y, w, h, theta)
}

#[pyfunction]
fn apsum_rect_center(
    data: PyReadonlyArray2<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("data shape is too large"));
    }
    sum_rect_center_slice(data, ny, nx, xs, ys, w, h, theta)
}

#[pyfunction]
fn apsum_rect_center_f32(
    data: PyReadonlyArray2<'_, f32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_f32(&data)?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("data shape is too large"));
    }
    sum_rect_center_slice(data, ny, nx, xs, ys, w, h, theta)
}

#[pyfunction]
fn apsum_rect_center_i32(
    data: PyReadonlyArray2<'_, i32>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int32")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("data shape is too large"));
    }
    sum_rect_center_slice(data, ny, nx, xs, ys, w, h, theta)
}

#[pyfunction]
fn apsum_rect_center_i16(
    data: PyReadonlyArray2<'_, i16>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (data, ny, nx) = data_slice_typed(&data, "int16")?;
    let xs = vector_slice(&xs, "x")?;
    let ys = vector_slice(&ys, "y")?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    validate_rect_params(ny, nx, w, h, theta, "invalid rectangle parameters")?;
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("data shape is too large"));
    }
    sum_rect_center_slice(data, ny, nx, xs, ys, w, h, theta)
}

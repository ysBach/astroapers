use super::support::*;
use super::*;

type FarthestMaskResult = (f64, usize, Vec<usize>);

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_farthest_mask_pixel, m)?)?;
    m.add_function(wrap_pyfunction!(_selected_pixel_distances, m)?)?;
    Ok(())
}

#[pyfunction]
fn _farthest_mask_pixel(
    mask: PyReadonlyArrayDyn<'_, bool>,
    center: PyReadonlyArray1<'_, f64>,
    return_pos: bool,
) -> PyResult<FarthestMaskResult> {
    let shape = mask.shape();
    let ndim = shape.len();
    if ndim == 0 {
        return Err(PyValueError::new_err(
            "mask must have at least one dimension",
        ));
    }
    let center = vector_slice(&center, "center")?;
    if center.len() != ndim {
        return Err(PyValueError::new_err("center length must match mask.ndim"));
    }
    if center.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("center values must be finite"));
    }
    let mask = mask
        .as_slice()
        .map_err(|_| PyValueError::new_err("mask must be a contiguous bool array"))?;

    let mut best_dist2 = f64::NEG_INFINITY;
    let mut found = false;
    let mut positions = Vec::new();
    let mut coords = vec![0_usize; ndim];

    for (linear_index, &selected) in mask.iter().enumerate() {
        if !selected {
            continue;
        }
        unravel_c_order(linear_index, shape, &mut coords);
        let dist2 = squared_distance(&coords, center);
        if dist2 > best_dist2 {
            found = true;
            best_dist2 = dist2;
            if return_pos {
                positions.clear();
                positions.extend_from_slice(&coords);
            }
        } else if return_pos && dist2 == best_dist2 {
            positions.extend_from_slice(&coords);
        }
    }

    if !found {
        return Err(PyValueError::new_err(
            "mask must contain at least one True pixel",
        ));
    }
    Ok((best_dist2.sqrt(), ndim, positions))
}

fn unravel_c_order(mut linear_index: usize, shape: &[usize], coords: &mut [usize]) {
    for axis in (0..shape.len()).rev() {
        let size = shape[axis];
        coords[axis] = linear_index % size;
        linear_index /= size;
    }
}

fn squared_distance(coords: &[usize], center: &[f64]) -> f64 {
    coords
        .iter()
        .zip(center.iter())
        .map(|(&coord, &origin)| {
            let delta = coord as f64 - origin;
            delta * delta
        })
        .sum()
}

#[pyfunction]
fn _selected_pixel_distances(
    selected: PyReadonlyArray2<'_, bool>,
    x0: f64,
    y0: f64,
    x_min: isize,
    y_min: isize,
) -> PyResult<Vec<f64>> {
    if !x0.is_finite() || !y0.is_finite() {
        return Err(PyValueError::new_err("center values must be finite"));
    }
    let shape = selected.shape();
    let ny = shape[0];
    let nx = shape[1];
    let selected = selected
        .as_slice()
        .map_err(|_| PyValueError::new_err("selected must be a contiguous bool array"))?;

    let mut distances = Vec::new();
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid selected extent"))?;
        for xx in 0..nx {
            if !selected[yy * nx + xx] {
                continue;
            }
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid selected extent"))?;
            let dx = px as f64 - x0;
            let dy = py as f64 - y0;
            distances.push((dx * dx + dy * dy).sqrt());
        }
    }
    Ok(distances)
}

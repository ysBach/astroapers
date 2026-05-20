use super::*;

pub(in crate::pyapi) fn data_slice<'py>(
    data: &'py PyReadonlyArray2<'py, f64>,
) -> PyResult<(&'py [f64], usize, usize)> {
    let shape = data.shape();
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be a contiguous float64 array"))?;
    Ok((slice, shape[0], shape[1]))
}

pub(in crate::pyapi) fn data_slice_f32<'py>(
    data: &'py PyReadonlyArray2<'py, f32>,
) -> PyResult<(&'py [f32], usize, usize)> {
    let shape = data.shape();
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be a contiguous float32 array"))?;
    Ok((slice, shape[0], shape[1]))
}

pub(in crate::pyapi) fn data_slice_typed<'py, T: Element>(
    data: &'py PyReadonlyArray2<'py, T>,
    dtype: &str,
) -> PyResult<(&'py [T], usize, usize)> {
    let shape = data.shape();
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("data must be a contiguous {dtype} array")))?;
    Ok((slice, shape[0], shape[1]))
}

#[inline(always)]
pub(in crate::pyapi) fn data_value<T>(data: &[T], index: usize) -> f64
where
    T: Copy + Into<f64>,
{
    unsafe { (*data.get_unchecked(index)).into() }
}

pub(in crate::pyapi) fn vector_slice<'py>(
    array: &'py PyReadonlyArray1<'py, f64>,
    name: &str,
) -> PyResult<&'py [f64]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float64 array")))
}

pub(in crate::pyapi) fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

pub(in crate::pyapi) fn validate_circ_params(
    ny: usize,
    nx: usize,
    r: f64,
    message: &'static str,
) -> PyResult<()> {
    if ny == 0 || nx == 0 || !positive(r) {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

pub(in crate::pyapi) fn validate_circ_ann_params(
    ny: usize,
    nx: usize,
    r_in: f64,
    r_out: f64,
    message: &'static str,
) -> PyResult<()> {
    if ny == 0 || nx == 0 {
        return Err(PyValueError::new_err("shape dimensions must be positive"));
    }
    if !r_in.is_finite() || !r_out.is_finite() || r_in < 0.0 || r_out <= 0.0 || r_in >= r_out {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

pub(in crate::pyapi) fn validate_ellip_params(
    ny: usize,
    nx: usize,
    a: f64,
    b: f64,
    theta: f64,
    message: &'static str,
) -> PyResult<()> {
    if ny == 0 || nx == 0 || !positive(a) || !positive(b) || !theta.is_finite() {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn validate_ellip_ann_params(
    ny: usize,
    nx: usize,
    a_in: f64,
    b_in: f64,
    a_out: f64,
    b_out: f64,
    theta_in: f64,
    theta_out: f64,
    message: &'static str,
) -> PyResult<()> {
    if ny == 0 || nx == 0 {
        return Err(PyValueError::new_err("shape dimensions must be positive"));
    }
    if !positive(a_in)
        || !positive(b_in)
        || !positive(a_out)
        || !positive(b_out)
        || !theta_in.is_finite()
        || !theta_out.is_finite()
        || a_in >= a_out
        || b_in >= b_out
    {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

pub(in crate::pyapi) fn validate_rect_params(
    ny: usize,
    nx: usize,
    w: f64,
    h: f64,
    theta: f64,
    message: &'static str,
) -> PyResult<()> {
    if ny == 0 || nx == 0 || !positive(w) || !positive(h) || !theta.is_finite() {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn validate_rect_ann_params(
    ny: usize,
    nx: usize,
    w_in: f64,
    h_in: f64,
    w_out: f64,
    h_out: f64,
    theta_in: f64,
    theta_out: f64,
    message: &'static str,
) -> PyResult<()> {
    if ny == 0 || nx == 0 {
        return Err(PyValueError::new_err("shape dimensions must be positive"));
    }
    if !positive(w_in)
        || !positive(h_in)
        || !positive(w_out)
        || !positive(h_out)
        || !theta_in.is_finite()
        || !theta_out.is_finite()
        || w_in >= w_out
        || h_in >= h_out
    {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn validate_wedge_params(
    ny: usize,
    nx: usize,
    r_in: f64,
    r_out: f64,
    theta_in: f64,
    dtheta_in: f64,
    theta_out: f64,
    dtheta_out: f64,
    message: &'static str,
) -> PyResult<()> {
    if ny == 0 || nx == 0 {
        return Err(PyValueError::new_err("shape dimensions must be positive"));
    }
    if !positive(r_in)
        || !positive(r_out)
        || r_out <= r_in
        || !theta_in.is_finite()
        || !theta_out.is_finite()
        || !positive(dtheta_in)
        || !positive(dtheta_out)
        || dtheta_in >= 2.0 * PI
        || dtheta_out >= 2.0 * PI
    {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

pub(in crate::pyapi) fn validate_pill_params(
    ny: usize,
    nx: usize,
    w: f64,
    a: f64,
    b: f64,
    theta: f64,
    message: &'static str,
) -> PyResult<()> {
    if ny == 0 || nx == 0 || !positive(w) || !positive(a) || !positive(b) || !theta.is_finite() {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn validate_pill_ann_params(
    ny: usize,
    nx: usize,
    w_in: f64,
    a_in: f64,
    b_in: f64,
    w_out: f64,
    a_out: f64,
    b_out: f64,
    theta_in: f64,
    theta_out: f64,
    message: &'static str,
) -> PyResult<()> {
    if ny == 0 || nx == 0 {
        return Err(PyValueError::new_err("shape dimensions must be positive"));
    }
    if !positive(w_in)
        || !positive(a_in)
        || !positive(b_in)
        || !positive(w_out)
        || !positive(a_out)
        || !positive(b_out)
        || !theta_in.is_finite()
        || !theta_out.is_finite()
        || w_in >= w_out
        || a_in >= a_out
        || b_in >= b_out
    {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

pub(in crate::pyapi) fn validate_weights_shape(
    x: f64,
    y: f64,
    ny: usize,
    nx: usize,
    message: &'static str,
) -> PyResult<()> {
    if !x.is_finite() || !y.is_finite() || ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

pub(in crate::pyapi) fn validate_many_weights_inputs(
    xs: &[f64],
    ys: &[f64],
    message: &'static str,
) -> PyResult<()> {
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have matching shapes"));
    }
    if xs.iter().any(|value| !value.is_finite()) || ys.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

pub(in crate::pyapi) fn parallel_threshold() -> usize {
    let configured = PARALLEL_THRESHOLD.load(Ordering::Relaxed);
    if configured != 0 {
        return configured;
    }

    let threshold = env::var("ASTROAPERS_PARALLEL_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PARALLEL_THRESHOLD);
    let _ = PARALLEL_THRESHOLD.compare_exchange(0, threshold, Ordering::Relaxed, Ordering::Relaxed);
    PARALLEL_THRESHOLD.load(Ordering::Relaxed)
}

pub(in crate::pyapi) fn run_parallel<F>(
    n: usize,
    xs: &[f64],
    ys: &[f64],
    apsum: &mut [f64],
    area: &mut [f64],
    worker: F,
) where
    F: Fn(usize, &[f64], &[f64], &mut [f64], &mut [f64]) + Sync,
{
    let nthreads = rayon::current_num_threads().min(n);
    if nthreads <= 1 {
        worker(0, xs, ys, apsum, area);
        return;
    }

    let chunk = n.div_ceil(nthreads);
    apsum
        .par_chunks_mut(chunk)
        .zip(area.par_chunks_mut(chunk))
        .zip(xs.par_chunks(chunk).zip(ys.par_chunks(chunk)))
        .enumerate()
        .for_each(|(chunk_index, ((fchunk, achunk), (xchunk, ychunk)))| {
            worker(chunk_index * chunk, xchunk, ychunk, fchunk, achunk);
        });
}

pub(in crate::pyapi) fn run_parallel_sum_only<F>(
    n: usize,
    xs: &[f64],
    ys: &[f64],
    apsum: &mut [f64],
    worker: F,
) where
    F: Fn(&[f64], &[f64], &mut [f64]) + Sync,
{
    let nthreads = rayon::current_num_threads().min(n);
    if nthreads <= 1 {
        worker(xs, ys, apsum);
        return;
    }

    let chunk = n.div_ceil(nthreads);
    apsum
        .par_chunks_mut(chunk)
        .zip(xs.par_chunks(chunk).zip(ys.par_chunks(chunk)))
        .for_each(|(fchunk, (xchunk, ychunk))| {
            worker(xchunk, ychunk, fchunk);
        });
}

#[cfg(test)]
mod tests {
    use super::super::weights::{weights_circ_exact, weights_ellip_exact, weights_rect_exact};
    use super::*;

    #[test]
    fn centered_circ_area_tracks_analytic() {
        let ell = Ellip {
            x0: 8.0,
            y0: 8.0,
            a: 5.0,
            b: 5.0,
            cos_t: 1.0,
            sin_t: 0.0,
        };
        let mut area = 0.0;
        for y in 2..=14 {
            for x in 2..=14 {
                area += ellip_pixel_area(ell, x, y);
            }
        }
        assert!((area - PI * 25.0).abs() < 1.0e-8);
    }

    #[test]
    fn axis_aligned_rect_area_tracks_analytic() {
        let mut area = 0.0;
        for y in 5..=11 {
            for x in 4..=12 {
                area += rect_pixel_area(8.0, 8.0, x, y, 3.0, 2.0, 1.0, 0.0);
            }
        }
        assert!((area - 24.0).abs() < 1.0e-12);
    }

    #[test]
    fn weights_circ_fill_tracks_analytic_area() {
        let weights = weights_circ_exact(5.0, 5.0, 3.0, 0, 0, 11, 11).unwrap();
        let area: f64 = weights.iter().sum();
        assert!((area - PI * 9.0).abs() < 1.0e-8);
    }

    #[test]
    fn weights_circ_fill_accepts_negative_bbox() {
        let weights = weights_circ_exact(0.0, 0.0, 3.0, -4, -4, 8, 8).unwrap();
        let area: f64 = weights.iter().sum();
        assert!((area - PI * 9.0).abs() < 1.0e-8);
    }

    #[test]
    fn weights_ellip_and_rect_fill_track_analytic_area() {
        let ellip = weights_ellip_exact(10.0, 10.0, 4.0, 2.0, 0.3, 0, 0, 20, 20).unwrap();
        let ellip_area: f64 = ellip.iter().sum();
        assert!((ellip_area - PI * 8.0).abs() < 1.0e-8);

        let rect = weights_rect_exact(10.0, 10.0, 6.0, 3.0, 0.3, 0, 0, 20, 20).unwrap();
        let rect_area: f64 = rect.iter().sum();
        assert!((rect_area - 18.0).abs() < 1.0e-12);
    }
}

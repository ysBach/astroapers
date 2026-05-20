#![allow(clippy::too_many_arguments)]

use super::support::*;
use super::*;

pub(in crate::pyapi) fn sum_circ_center_slice<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    xs: &[f64],
    ys: &[f64],
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)>
where
    T: Copy + Into<f64> + Sync,
{
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("array shape is too large"));
    }
    let mut apsum = vec![0.0; xs.len()];
    let mut npix = vec![0.0; xs.len()];
    let r2 = r * r;

    let worker = |_: usize, xs: &[f64], ys: &[f64], apsum: &mut [f64], npix: &mut [f64]| {
        for i in 0..xs.len() {
            let x = xs[i];
            let y = ys[i];
            if !x.is_finite() || !y.is_finite() {
                apsum[i] = f64::NAN;
                npix[i] = f64::NAN;
                continue;
            }
            let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r);
            let xmin = ixmin.max(0) as usize;
            let xmax = ixmax.min(nx as isize).max(0) as usize;
            let ymin = iymin.max(0) as usize;
            let ymax = iymax.min(ny as isize).max(0) as usize;
            let mut f = 0.0;
            let mut n = 0.0;
            for yy in ymin..ymax {
                for xx in xmin..xmax {
                    let dx = xx as f64 - x;
                    let dy = yy as f64 - y;
                    if dx * dx + dy * dy < r2 {
                        f += data_value(data, yy * nx + xx);
                        n += 1.0;
                    }
                }
            }
            apsum[i] = f;
            npix[i] = n;
        }
    };

    if xs.len() >= parallel_threshold() {
        run_parallel(xs.len(), xs, ys, &mut apsum, &mut npix, worker);
    } else {
        worker(0, xs, ys, &mut apsum, &mut npix);
    }
    Ok((apsum, npix))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn sum_ellip_center_slice<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    xs: &[f64],
    ys: &[f64],
    a: f64,
    b: f64,
    cos_t: f64,
    sin_t: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)>
where
    T: Copy + Into<f64> + Sync,
{
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("array shape is too large"));
    }
    let mut apsum = vec![0.0; xs.len()];
    let mut npix = vec![0.0; xs.len()];

    let worker = |_: usize, xs: &[f64], ys: &[f64], apsum: &mut [f64], npix: &mut [f64]| {
        for i in 0..xs.len() {
            let x = xs[i];
            let y = ys[i];
            if !x.is_finite() || !y.is_finite() {
                apsum[i] = f64::NAN;
                npix[i] = f64::NAN;
                continue;
            }
            let ell = Ellip {
                x0: x,
                y0: y,
                a,
                b,
                cos_t,
                sin_t,
            };
            let (ixmin, ixmax, iymin, iymax) = ellip_bbox(ell);
            let xmin = ixmin.max(0) as usize;
            let xmax = ixmax.min(nx as isize).max(0) as usize;
            let ymin = iymin.max(0) as usize;
            let ymax = iymax.min(ny as isize).max(0) as usize;
            let mut f = 0.0;
            let mut n = 0.0;
            for yy in ymin..ymax {
                for xx in xmin..xmax {
                    let dx = xx as f64 - x;
                    let dy = yy as f64 - y;
                    let u = (cos_t * dx + sin_t * dy) / a;
                    let v = (-sin_t * dx + cos_t * dy) / b;
                    if u * u + v * v < 1.0 {
                        f += data_value(data, yy * nx + xx);
                        n += 1.0;
                    }
                }
            }
            apsum[i] = f;
            npix[i] = n;
        }
    };

    if xs.len() >= parallel_threshold() {
        run_parallel(xs.len(), xs, ys, &mut apsum, &mut npix, worker);
    } else {
        worker(0, xs, ys, &mut apsum, &mut npix);
    }
    Ok((apsum, npix))
}

pub(in crate::pyapi) fn sum_rect_center_slice<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    xs: &[f64],
    ys: &[f64],
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)>
where
    T: Copy + Into<f64> + Sync,
{
    let mut apsum = vec![0.0; xs.len()];
    let mut npix = vec![0.0; xs.len()];

    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    let dx = (half_w * cos_t).abs() + (half_h * sin_t).abs();
    let dy = (half_w * sin_t).abs() + (half_h * cos_t).abs();

    let worker = |_: usize, xs: &[f64], ys: &[f64], apsum: &mut [f64], npix: &mut [f64]| {
        for i in 0..xs.len() {
            let x = xs[i];
            let y = ys[i];
            if !x.is_finite() || !y.is_finite() {
                apsum[i] = f64::NAN;
                npix[i] = f64::NAN;
                continue;
            }
            let (ixmin, ixmax, iymin, iymax) = extent_bbox(x, y, dx, dy);
            let xmin = ixmin.max(0) as usize;
            let xmax = ixmax.min(nx as isize).max(0) as usize;
            let ymin = iymin.max(0) as usize;
            let ymax = iymax.min(ny as isize).max(0) as usize;
            let mut f = 0.0;
            let mut n = 0.0;
            for yy in ymin..ymax {
                for xx in xmin..xmax {
                    let ddx = xx as f64 - x;
                    let ddy = yy as f64 - y;
                    let u = cos_t * ddx + sin_t * ddy;
                    let v = -sin_t * ddx + cos_t * ddy;
                    if u.abs() < half_w && v.abs() < half_h {
                        f += data_value(data, yy * nx + xx);
                        n += 1.0;
                    }
                }
            }
            apsum[i] = f;
            npix[i] = n;
        }
    };

    if xs.len() >= parallel_threshold() {
        run_parallel(xs.len(), xs, ys, &mut apsum, &mut npix, worker);
    } else {
        worker(0, xs, ys, &mut apsum, &mut npix);
    }
    Ok((apsum, npix))
}

pub(in crate::pyapi) fn sum_rect_slice<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    xs: &[f64],
    ys: &[f64],
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)>
where
    T: Copy + Into<f64> + Sync,
{
    let n = xs.len();
    let mut apsum = vec![0.0; n];
    let mut area = vec![0.0; n];

    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    let dx = (half_w * cos_t).abs() + (half_h * sin_t).abs();
    let dy = (half_w * sin_t).abs() + (half_h * cos_t).abs();
    let ext = 0.5 * (cos_t.abs() + sin_t.abs());

    let worker = |_: usize, xs: &[f64], ys: &[f64], apsum: &mut [f64], area: &mut [f64]| {
        for i in 0..xs.len() {
            let x = xs[i];
            let y = ys[i];
            if !x.is_finite() || !y.is_finite() {
                apsum[i] = f64::NAN;
                area[i] = f64::NAN;
                continue;
            }
            let (ixmin, ixmax, iymin, iymax) = extent_bbox(x, y, dx, dy);
            let xmin = ixmin.max(0) as usize;
            let xmax = ixmax.min(nx as isize).max(0) as usize;
            let ymin = iymin.max(0) as usize;
            let ymax = iymax.min(ny as isize).max(0) as usize;
            let mut f = 0.0;
            let mut ar = 0.0;
            for yy in ymin..ymax {
                for xx in xmin..xmax {
                    let ddx = xx as f64 - x;
                    let ddy = yy as f64 - y;
                    let u = cos_t * ddx + sin_t * ddy;
                    let v = -sin_t * ddx + cos_t * ddy;
                    let au = u.abs();
                    let av = v.abs();
                    let overlap = if au <= half_w - ext && av <= half_h - ext {
                        1.0
                    } else if au >= half_w + ext || av >= half_h + ext {
                        0.0
                    } else {
                        rect_pixel_area(
                            x,
                            y,
                            xx as isize,
                            yy as isize,
                            half_w,
                            half_h,
                            cos_t,
                            sin_t,
                        )
                    };
                    if overlap > 0.0 {
                        f += overlap * data_value(data, yy * nx + xx);
                        ar += overlap;
                    }
                }
            }
            if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
                ar = w * h;
            }
            apsum[i] = f;
            area[i] = ar;
        }
    };

    if n >= parallel_threshold() {
        run_parallel(n, xs, ys, &mut apsum, &mut area, worker);
    } else {
        worker(0, xs, ys, &mut apsum, &mut area);
    }
    Ok((apsum, area))
}

pub(in crate::pyapi) fn sum_circ_one<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    x: f64,
    y: f64,
    r: f64,
) -> PyResult<(f64, f64)>
where
    T: Copy + Into<f64>,
{
    if !x.is_finite() || !y.is_finite() {
        return Ok((f64::NAN, f64::NAN));
    }
    let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r);
    let xmin = ixmin.max(0) as usize;
    let xmax = ixmax.min(nx as isize).max(0) as usize;
    let ymin = iymin.max(0) as usize;
    let ymax = iymax.min(ny as isize).max(0) as usize;
    let r_in = (r - SQRT_HALF).max(0.0);
    let r_in2 = r_in * r_in;
    let r_out = r + SQRT_HALF;
    let r_out2 = r_out * r_out;
    let mut apsum = 0.0;
    let mut area = 0.0;
    for yy in ymin..ymax {
        for xx in xmin..xmax {
            let dx = xx as f64 - x;
            let dy = yy as f64 - y;
            let rpix2 = dx * dx + dy * dy;
            let overlap = if rpix2 < r_in2 {
                1.0
            } else if rpix2 > r_out2 {
                0.0
            } else {
                circ_pixel_area(x, y, xx as isize, yy as isize, r)
            };
            if overlap > 0.0 {
                apsum += overlap * data_value(data, yy * nx + xx);
                area += overlap;
            }
        }
    }
    if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
        area = PI * r * r;
    }
    Ok((apsum, area))
}

pub(in crate::pyapi) fn sum_circ_only_one<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    x: f64,
    y: f64,
    r: f64,
) -> PyResult<f64>
where
    T: Copy + Into<f64>,
{
    if !x.is_finite() || !y.is_finite() {
        return Ok(f64::NAN);
    }
    let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r);
    let xmin = ixmin.max(0) as usize;
    let xmax = ixmax.min(nx as isize).max(0) as usize;
    let ymin = iymin.max(0) as usize;
    let ymax = iymax.min(ny as isize).max(0) as usize;
    let r_in = (r - SQRT_HALF).max(0.0);
    let r_in2 = r_in * r_in;
    let r_out = r + SQRT_HALF;
    let r_out2 = r_out * r_out;
    let mut apsum = 0.0;
    for yy in ymin..ymax {
        for xx in xmin..xmax {
            let overlap = circ_overlap_at(x, y, xx, yy, r, r_in2, r_out2);
            if overlap > 0.0 {
                apsum += overlap * data_value(data, yy * nx + xx);
            }
        }
    }
    Ok(apsum)
}

#[inline(always)]
pub(in crate::pyapi) fn circ_overlap_at(
    x: f64,
    y: f64,
    xx: usize,
    yy: usize,
    r: f64,
    r_inner2: f64,
    r_outer2: f64,
) -> f64 {
    let dx = xx as f64 - x;
    let dy = yy as f64 - y;
    let rpix2 = dx * dx + dy * dy;
    if rpix2 < r_inner2 {
        1.0
    } else if rpix2 > r_outer2 {
        0.0
    } else {
        circ_pixel_area(x, y, xx as isize, yy as isize, r)
    }
}

pub(in crate::pyapi) fn sum_circ_ann_one<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
) -> PyResult<(f64, f64)>
where
    T: Copy + Into<f64>,
{
    if !x.is_finite() || !y.is_finite() {
        return Ok((f64::NAN, f64::NAN));
    }
    let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r_out);
    let xmin = ixmin.max(0) as usize;
    let xmax = ixmax.min(nx as isize).max(0) as usize;
    let ymin = iymin.max(0) as usize;
    let ymax = iymax.min(ny as isize).max(0) as usize;
    let out_inner = (r_out - SQRT_HALF).max(0.0);
    let out_inner2 = out_inner * out_inner;
    let out_outer = r_out + SQRT_HALF;
    let out_outer2 = out_outer * out_outer;
    let in_inner = (r_in - SQRT_HALF).max(0.0);
    let in_inner2 = in_inner * in_inner;
    let in_outer = r_in + SQRT_HALF;
    let in_outer2 = in_outer * in_outer;
    let mut apsum = 0.0;
    let mut area = 0.0;
    for yy in ymin..ymax {
        for xx in xmin..xmax {
            let outer = circ_overlap_at(x, y, xx, yy, r_out, out_inner2, out_outer2);
            if outer <= 0.0 {
                continue;
            }
            let inner = if r_in == 0.0 {
                0.0
            } else {
                circ_overlap_at(x, y, xx, yy, r_in, in_inner2, in_outer2)
            };
            let overlap = (outer - inner).clamp(0.0, 1.0);
            if overlap > 0.0 {
                apsum += overlap * data_value(data, yy * nx + xx);
                area += overlap;
            }
        }
    }
    if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
        area = PI * (r_out * r_out - r_in * r_in);
    }
    Ok((apsum, area))
}

pub(in crate::pyapi) fn sum_circ_ann_only_one<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
) -> PyResult<f64>
where
    T: Copy + Into<f64>,
{
    if !x.is_finite() || !y.is_finite() {
        return Ok(f64::NAN);
    }
    let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r_out);
    let xmin = ixmin.max(0) as usize;
    let xmax = ixmax.min(nx as isize).max(0) as usize;
    let ymin = iymin.max(0) as usize;
    let ymax = iymax.min(ny as isize).max(0) as usize;
    let out_inner = (r_out - SQRT_HALF).max(0.0);
    let out_inner2 = out_inner * out_inner;
    let out_outer = r_out + SQRT_HALF;
    let out_outer2 = out_outer * out_outer;
    let in_inner = (r_in - SQRT_HALF).max(0.0);
    let in_inner2 = in_inner * in_inner;
    let in_outer = r_in + SQRT_HALF;
    let in_outer2 = in_outer * in_outer;
    let mut apsum = 0.0;
    for yy in ymin..ymax {
        for xx in xmin..xmax {
            let outer = circ_overlap_at(x, y, xx, yy, r_out, out_inner2, out_outer2);
            if outer <= 0.0 {
                continue;
            }
            let inner = if r_in == 0.0 {
                0.0
            } else {
                circ_overlap_at(x, y, xx, yy, r_in, in_inner2, in_outer2)
            };
            let overlap = (outer - inner).clamp(0.0, 1.0);
            if overlap > 0.0 {
                apsum += overlap * data_value(data, yy * nx + xx);
            }
        }
    }
    Ok(apsum)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn sum_ellip_one<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    x: f64,
    y: f64,
    a: f64,
    b: f64,
    cos_t: f64,
    sin_t: f64,
) -> PyResult<(f64, f64)>
where
    T: Copy + Into<f64>,
{
    if !x.is_finite() || !y.is_finite() {
        return Ok((f64::NAN, f64::NAN));
    }
    let ell = Ellip {
        x0: x,
        y0: y,
        a,
        b,
        cos_t,
        sin_t,
    };
    let (ixmin, ixmax, iymin, iymax) = ellip_bbox(ell);
    let xmin = ixmin.max(0) as usize;
    let xmax = ixmax.min(nx as isize).max(0) as usize;
    let ymin = iymin.max(0) as usize;
    let ymax = iymax.min(ny as isize).max(0) as usize;
    let (r_in2, r_out2) = ellip_inner_outer_radius2(a, b, cos_t, sin_t);
    let mut apsum = 0.0;
    let mut area = 0.0;
    for yy in ymin..ymax {
        for xx in xmin..xmax {
            let dx = xx as f64 - x;
            let dy = yy as f64 - y;
            let u = (cos_t * dx + sin_t * dy) / a;
            let v = (-sin_t * dx + cos_t * dy) / b;
            let rpix2 = u * u + v * v;
            let overlap = if rpix2 < r_in2 {
                1.0
            } else if rpix2 > r_out2 {
                0.0
            } else {
                ellip_pixel_area(ell, xx as isize, yy as isize)
            };
            if overlap > 0.0 {
                apsum += overlap * data_value(data, yy * nx + xx);
                area += overlap;
            }
        }
    }
    if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
        area = PI * a * b;
    }
    Ok((apsum, area))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn sum_rect_one<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    theta: f64,
) -> PyResult<(f64, f64)>
where
    T: Copy + Into<f64>,
{
    if !x.is_finite() || !y.is_finite() {
        return Ok((f64::NAN, f64::NAN));
    }
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    let dx = (half_w * cos_t).abs() + (half_h * sin_t).abs();
    let dy = (half_w * sin_t).abs() + (half_h * cos_t).abs();
    let ext = 0.5 * (cos_t.abs() + sin_t.abs());
    let (ixmin, ixmax, iymin, iymax) = extent_bbox(x, y, dx, dy);
    let xmin = ixmin.max(0) as usize;
    let xmax = ixmax.min(nx as isize).max(0) as usize;
    let ymin = iymin.max(0) as usize;
    let ymax = iymax.min(ny as isize).max(0) as usize;
    let mut apsum = 0.0;
    let mut area = 0.0;
    for yy in ymin..ymax {
        for xx in xmin..xmax {
            let ddx = xx as f64 - x;
            let ddy = yy as f64 - y;
            let u = cos_t * ddx + sin_t * ddy;
            let v = -sin_t * ddx + cos_t * ddy;
            let au = u.abs();
            let av = v.abs();
            let overlap = if au <= half_w - ext && av <= half_h - ext {
                1.0
            } else if au >= half_w + ext || av >= half_h + ext {
                0.0
            } else {
                rect_pixel_area(x, y, xx as isize, yy as isize, half_w, half_h, cos_t, sin_t)
            };
            if overlap > 0.0 {
                apsum += overlap * data_value(data, yy * nx + xx);
                area += overlap;
            }
        }
    }
    if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
        area = w * h;
    }
    Ok((apsum, area))
}

pub(in crate::pyapi) fn npix_circ_slice(
    xs: &[f64],
    ys: &[f64],
    r: f64,
    ny: usize,
    nx: usize,
) -> Vec<f64> {
    let full_area = PI * r * r;
    let mut areas = vec![0.0; xs.len()];
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            areas[i] = f64::NAN;
            continue;
        }
        let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r);
        if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
            areas[i] = full_area;
            continue;
        }
        let xmin = ixmin.max(0) as usize;
        let xmax = ixmax.min(nx as isize).max(0) as usize;
        let ymin = iymin.max(0) as usize;
        let ymax = iymax.min(ny as isize).max(0) as usize;
        let mut area = 0.0;
        for yy in ymin..ymax {
            for xx in xmin..xmax {
                area += circ_pixel_area(x, y, xx as isize, yy as isize, r);
            }
        }
        areas[i] = area;
    }
    areas
}

pub(in crate::pyapi) fn npix_circ_center_slice(
    xs: &[f64],
    ys: &[f64],
    r: f64,
    ny: usize,
    nx: usize,
) -> Vec<f64> {
    let r2 = r * r;
    let mut areas = vec![0.0; xs.len()];
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            areas[i] = f64::NAN;
            continue;
        }
        let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r);
        let xmin = ixmin.max(0) as usize;
        let xmax = ixmax.min(nx as isize).max(0) as usize;
        let ymin = iymin.max(0) as usize;
        let ymax = iymax.min(ny as isize).max(0) as usize;
        let mut area = 0.0;
        for yy in ymin..ymax {
            for xx in xmin..xmax {
                let dx = xx as f64 - x;
                let dy = yy as f64 - y;
                if dx * dx + dy * dy < r2 {
                    area += 1.0;
                }
            }
        }
        areas[i] = area;
    }
    areas
}

pub(in crate::pyapi) fn npix_ellip_slice(
    xs: &[f64],
    ys: &[f64],
    a: f64,
    b: f64,
    cos_t: f64,
    sin_t: f64,
    ny: usize,
    nx: usize,
) -> Vec<f64> {
    let full_area = PI * a * b;
    let mut areas = vec![0.0; xs.len()];
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            areas[i] = f64::NAN;
            continue;
        }
        let ell = Ellip {
            x0: x,
            y0: y,
            a,
            b,
            cos_t,
            sin_t,
        };
        let (ixmin, ixmax, iymin, iymax) = ellip_bbox(ell);
        if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
            areas[i] = full_area;
            continue;
        }
        let xmin = ixmin.max(0) as usize;
        let xmax = ixmax.min(nx as isize).max(0) as usize;
        let ymin = iymin.max(0) as usize;
        let ymax = iymax.min(ny as isize).max(0) as usize;
        let mut area = 0.0;
        for yy in ymin..ymax {
            for xx in xmin..xmax {
                area += ellip_pixel_area(ell, xx as isize, yy as isize);
            }
        }
        areas[i] = area;
    }
    areas
}

pub(in crate::pyapi) fn npix_ellip_center_slice(
    xs: &[f64],
    ys: &[f64],
    a: f64,
    b: f64,
    cos_t: f64,
    sin_t: f64,
    ny: usize,
    nx: usize,
) -> Vec<f64> {
    let mut areas = vec![0.0; xs.len()];
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            areas[i] = f64::NAN;
            continue;
        }
        let ell = Ellip {
            x0: x,
            y0: y,
            a,
            b,
            cos_t,
            sin_t,
        };
        let (ixmin, ixmax, iymin, iymax) = ellip_bbox(ell);
        let xmin = ixmin.max(0) as usize;
        let xmax = ixmax.min(nx as isize).max(0) as usize;
        let ymin = iymin.max(0) as usize;
        let ymax = iymax.min(ny as isize).max(0) as usize;
        let mut area = 0.0;
        for yy in ymin..ymax {
            for xx in xmin..xmax {
                let dx = xx as f64 - x;
                let dy = yy as f64 - y;
                let u = (cos_t * dx + sin_t * dy) / a;
                let v = (-sin_t * dx + cos_t * dy) / b;
                if u * u + v * v < 1.0 {
                    area += 1.0;
                }
            }
        }
        areas[i] = area;
    }
    areas
}

pub(in crate::pyapi) fn npix_rect_slice(
    xs: &[f64],
    ys: &[f64],
    w: f64,
    h: f64,
    theta: f64,
    ny: usize,
    nx: usize,
) -> Vec<f64> {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    let dx = (half_w * cos_t).abs() + (half_h * sin_t).abs();
    let dy = (half_w * sin_t).abs() + (half_h * cos_t).abs();
    let full_area = w * h;
    let mut areas = vec![0.0; xs.len()];
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            areas[i] = f64::NAN;
            continue;
        }
        let (ixmin, ixmax, iymin, iymax) = extent_bbox(x, y, dx, dy);
        if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
            areas[i] = full_area;
            continue;
        }
        let xmin = ixmin.max(0) as usize;
        let xmax = ixmax.min(nx as isize).max(0) as usize;
        let ymin = iymin.max(0) as usize;
        let ymax = iymax.min(ny as isize).max(0) as usize;
        let mut area = 0.0;
        for yy in ymin..ymax {
            for xx in xmin..xmax {
                area +=
                    rect_pixel_area(x, y, xx as isize, yy as isize, half_w, half_h, cos_t, sin_t);
            }
        }
        areas[i] = area;
    }
    areas
}

pub(in crate::pyapi) fn npix_rect_center_slice(
    xs: &[f64],
    ys: &[f64],
    w: f64,
    h: f64,
    theta: f64,
    ny: usize,
    nx: usize,
) -> Vec<f64> {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half_w = 0.5 * w;
    let half_h = 0.5 * h;
    let dx = (half_w * cos_t).abs() + (half_h * sin_t).abs();
    let dy = (half_w * sin_t).abs() + (half_h * cos_t).abs();
    let mut areas = vec![0.0; xs.len()];
    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            areas[i] = f64::NAN;
            continue;
        }
        let (ixmin, ixmax, iymin, iymax) = extent_bbox(x, y, dx, dy);
        let xmin = ixmin.max(0) as usize;
        let xmax = ixmax.min(nx as isize).max(0) as usize;
        let ymin = iymin.max(0) as usize;
        let ymax = iymax.min(ny as isize).max(0) as usize;
        let mut area = 0.0;
        for yy in ymin..ymax {
            for xx in xmin..xmax {
                let ddx = xx as f64 - x;
                let ddy = yy as f64 - y;
                let u = cos_t * ddx + sin_t * ddy;
                let v = -sin_t * ddx + cos_t * ddy;
                if u.abs() < half_w && v.abs() < half_h {
                    area += 1.0;
                }
            }
        }
        areas[i] = area;
    }
    areas
}

pub(in crate::pyapi) fn sum_circ_slice<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    xs: &[f64],
    ys: &[f64],
    r: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)>
where
    T: Copy + Into<f64> + Sync,
{
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("array shape is too large"));
    }
    let mut apsum = vec![0.0; xs.len()];
    let mut area = vec![0.0; xs.len()];
    let r_in = (r - SQRT_HALF).max(0.0);
    let r_in2 = r_in * r_in;
    let r_out = r + SQRT_HALF;
    let r_out2 = r_out * r_out;

    let worker = |_: usize, xs: &[f64], ys: &[f64], apsum: &mut [f64], area: &mut [f64]| {
        for i in 0..xs.len() {
            let x = xs[i];
            let y = ys[i];
            if !x.is_finite() || !y.is_finite() {
                apsum[i] = f64::NAN;
                area[i] = f64::NAN;
                continue;
            }
            let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r);
            let xmin = ixmin.max(0) as usize;
            let xmax = ixmax.min(nx as isize).max(0) as usize;
            let ymin = iymin.max(0) as usize;
            let ymax = iymax.min(ny as isize).max(0) as usize;
            let mut f = 0.0;
            let mut ar = 0.0;
            for yy in ymin..ymax {
                for xx in xmin..xmax {
                    let dx = xx as f64 - x;
                    let dy = yy as f64 - y;
                    let rpix2 = dx * dx + dy * dy;
                    let overlap = if rpix2 < r_in2 {
                        1.0
                    } else if rpix2 > r_out2 {
                        0.0
                    } else {
                        circ_pixel_area(x, y, xx as isize, yy as isize, r)
                    };
                    if overlap > 0.0 {
                        f += overlap * data_value(data, yy * nx + xx);
                        ar += overlap;
                    }
                }
            }
            if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
                ar = PI * r * r;
            }
            apsum[i] = f;
            area[i] = ar;
        }
    };

    if xs.len() >= parallel_threshold() {
        run_parallel(xs.len(), xs, ys, &mut apsum, &mut area, worker);
    } else {
        worker(0, xs, ys, &mut apsum, &mut area);
    }
    Ok((apsum, area))
}

pub(in crate::pyapi) fn sum_circ_only_slice<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    xs: &[f64],
    ys: &[f64],
    r: f64,
) -> PyResult<Vec<f64>>
where
    T: Copy + Into<f64> + Sync,
{
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("array shape is too large"));
    }
    let mut apsum = vec![0.0; xs.len()];
    let r_in = (r - SQRT_HALF).max(0.0);
    let r_in2 = r_in * r_in;
    let r_out = r + SQRT_HALF;
    let r_out2 = r_out * r_out;

    let worker = |xs: &[f64], ys: &[f64], apsum: &mut [f64]| {
        for i in 0..xs.len() {
            let x = xs[i];
            let y = ys[i];
            if !x.is_finite() || !y.is_finite() {
                apsum[i] = f64::NAN;
                continue;
            }
            let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r);
            let xmin = ixmin.max(0) as usize;
            let xmax = ixmax.min(nx as isize).max(0) as usize;
            let ymin = iymin.max(0) as usize;
            let ymax = iymax.min(ny as isize).max(0) as usize;
            let mut f = 0.0;
            for yy in ymin..ymax {
                for xx in xmin..xmax {
                    let overlap = circ_overlap_at(x, y, xx, yy, r, r_in2, r_out2);
                    if overlap > 0.0 {
                        f += overlap * data_value(data, yy * nx + xx);
                    }
                }
            }
            apsum[i] = f;
        }
    };

    if xs.len() >= parallel_threshold() {
        run_parallel_sum_only(xs.len(), xs, ys, &mut apsum, worker);
    } else {
        worker(xs, ys, &mut apsum);
    }
    Ok(apsum)
}

pub(in crate::pyapi) fn sum_circ_ann_slice<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    xs: &[f64],
    ys: &[f64],
    r_in: f64,
    r_out: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)>
where
    T: Copy + Into<f64> + Sync,
{
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("array shape is too large"));
    }
    let mut apsum = vec![0.0; xs.len()];
    let mut area = vec![0.0; xs.len()];
    let full_area = PI * (r_out * r_out - r_in * r_in);
    let out_inner = (r_out - SQRT_HALF).max(0.0);
    let out_inner2 = out_inner * out_inner;
    let out_outer = r_out + SQRT_HALF;
    let out_outer2 = out_outer * out_outer;
    let in_inner = (r_in - SQRT_HALF).max(0.0);
    let in_inner2 = in_inner * in_inner;
    let in_outer = r_in + SQRT_HALF;
    let in_outer2 = in_outer * in_outer;

    let worker = |_: usize, xs: &[f64], ys: &[f64], apsum: &mut [f64], area: &mut [f64]| {
        for i in 0..xs.len() {
            let x = xs[i];
            let y = ys[i];
            if !x.is_finite() || !y.is_finite() {
                apsum[i] = f64::NAN;
                area[i] = f64::NAN;
                continue;
            }
            let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r_out);
            let xmin = ixmin.max(0) as usize;
            let xmax = ixmax.min(nx as isize).max(0) as usize;
            let ymin = iymin.max(0) as usize;
            let ymax = iymax.min(ny as isize).max(0) as usize;
            let mut f = 0.0;
            let mut ar = 0.0;
            for yy in ymin..ymax {
                for xx in xmin..xmax {
                    let outer = circ_overlap_at(x, y, xx, yy, r_out, out_inner2, out_outer2);
                    if outer <= 0.0 {
                        continue;
                    }
                    let inner = if r_in == 0.0 {
                        0.0
                    } else {
                        circ_overlap_at(x, y, xx, yy, r_in, in_inner2, in_outer2)
                    };
                    let overlap = (outer - inner).clamp(0.0, 1.0);
                    if overlap > 0.0 {
                        f += overlap * data_value(data, yy * nx + xx);
                        ar += overlap;
                    }
                }
            }
            if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
                ar = full_area;
            }
            apsum[i] = f;
            area[i] = ar;
        }
    };

    if xs.len() >= parallel_threshold() {
        run_parallel(xs.len(), xs, ys, &mut apsum, &mut area, worker);
    } else {
        worker(0, xs, ys, &mut apsum, &mut area);
    }
    Ok((apsum, area))
}

pub(in crate::pyapi) fn sum_circ_ann_only_slice<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    xs: &[f64],
    ys: &[f64],
    r_in: f64,
    r_out: f64,
) -> PyResult<Vec<f64>>
where
    T: Copy + Into<f64> + Sync,
{
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("array shape is too large"));
    }
    let mut apsum = vec![0.0; xs.len()];
    let out_inner = (r_out - SQRT_HALF).max(0.0);
    let out_inner2 = out_inner * out_inner;
    let out_outer = r_out + SQRT_HALF;
    let out_outer2 = out_outer * out_outer;
    let in_inner = (r_in - SQRT_HALF).max(0.0);
    let in_inner2 = in_inner * in_inner;
    let in_outer = r_in + SQRT_HALF;
    let in_outer2 = in_outer * in_outer;

    let worker = |xs: &[f64], ys: &[f64], apsum: &mut [f64]| {
        for i in 0..xs.len() {
            let x = xs[i];
            let y = ys[i];
            if !x.is_finite() || !y.is_finite() {
                apsum[i] = f64::NAN;
                continue;
            }
            let (ixmin, ixmax, iymin, iymax) = circ_bbox(x, y, r_out);
            let xmin = ixmin.max(0) as usize;
            let xmax = ixmax.min(nx as isize).max(0) as usize;
            let ymin = iymin.max(0) as usize;
            let ymax = iymax.min(ny as isize).max(0) as usize;
            let mut f = 0.0;
            for yy in ymin..ymax {
                for xx in xmin..xmax {
                    let outer = circ_overlap_at(x, y, xx, yy, r_out, out_inner2, out_outer2);
                    if outer <= 0.0 {
                        continue;
                    }
                    let inner = if r_in == 0.0 {
                        0.0
                    } else {
                        circ_overlap_at(x, y, xx, yy, r_in, in_inner2, in_outer2)
                    };
                    let overlap = (outer - inner).clamp(0.0, 1.0);
                    if overlap > 0.0 {
                        f += overlap * data_value(data, yy * nx + xx);
                    }
                }
            }
            apsum[i] = f;
        }
    };

    if xs.len() >= parallel_threshold() {
        run_parallel_sum_only(xs.len(), xs, ys, &mut apsum, worker);
    } else {
        worker(xs, ys, &mut apsum);
    }
    Ok(apsum)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn sum_ellip_slice<T>(
    data: &[T],
    ny: usize,
    nx: usize,
    xs: &[f64],
    ys: &[f64],
    a: f64,
    b: f64,
    cos_t: f64,
    sin_t: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)>
where
    T: Copy + Into<f64> + Sync,
{
    if ny.checked_mul(nx).is_none() {
        return Err(PyValueError::new_err("array shape is too large"));
    }
    let mut apsum = vec![0.0; xs.len()];
    let mut area = vec![0.0; xs.len()];
    let (r_in2, r_out2) = ellip_inner_outer_radius2(a, b, cos_t, sin_t);

    let worker = |_: usize, xs: &[f64], ys: &[f64], apsum: &mut [f64], area: &mut [f64]| {
        for i in 0..xs.len() {
            let x = xs[i];
            let y = ys[i];
            if !x.is_finite() || !y.is_finite() {
                apsum[i] = f64::NAN;
                area[i] = f64::NAN;
                continue;
            }
            let ell = Ellip {
                x0: x,
                y0: y,
                a,
                b,
                cos_t,
                sin_t,
            };
            let (ixmin, ixmax, iymin, iymax) = ellip_bbox(ell);
            let xmin = ixmin.max(0) as usize;
            let xmax = ixmax.min(nx as isize).max(0) as usize;
            let ymin = iymin.max(0) as usize;
            let ymax = iymax.min(ny as isize).max(0) as usize;
            let mut f = 0.0;
            let mut ar = 0.0;
            for yy in ymin..ymax {
                for xx in xmin..xmax {
                    let dx = xx as f64 - x;
                    let dy = yy as f64 - y;
                    let u = (cos_t * dx + sin_t * dy) / a;
                    let v = (-sin_t * dx + cos_t * dy) / b;
                    let rpix2 = u * u + v * v;
                    let overlap = if rpix2 < r_in2 {
                        1.0
                    } else if rpix2 > r_out2 {
                        0.0
                    } else {
                        ellip_pixel_area(ell, xx as isize, yy as isize)
                    };
                    if overlap > 0.0 {
                        f += overlap * data_value(data, yy * nx + xx);
                        ar += overlap;
                    }
                }
            }
            if bbox_in_frame(ixmin, ixmax, iymin, iymax, ny, nx) {
                ar = PI * a * b;
            }
            apsum[i] = f;
            area[i] = ar;
        }
    };

    if xs.len() >= parallel_threshold() {
        run_parallel(xs.len(), xs, ys, &mut apsum, &mut area, worker);
    } else {
        worker(0, xs, ys, &mut apsum, &mut area);
    }
    Ok((apsum, area))
}

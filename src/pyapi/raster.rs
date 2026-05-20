#![allow(clippy::too_many_arguments)]

use super::support::*;
use super::*;
use crate::geometry::{
    circular_pill_path, path_pixel_area, wedge_center_weight_at_pixel, wedge_exact_weight_at_pixel,
    Wedge,
};

pub(in crate::pyapi) fn weights_from_builders<F>(
    py: Python<'_>,
    len: usize,
    build: F,
) -> PyResult<WeightManyResult>
where
    F: Fn(usize) -> PyResult<(Vec<f64>, isize, isize, isize, isize)> + Send + Sync,
{
    let items: Vec<_> = if len >= parallel_threshold() {
        (0..len).into_par_iter().map(build).collect()
    } else {
        (0..len).map(build).collect()
    };

    let mut masks = Vec::with_capacity(items.len());
    let mut ixmins = Vec::with_capacity(items.len());
    let mut ixmaxs = Vec::with_capacity(items.len());
    let mut iymins = Vec::with_capacity(items.len());
    let mut iymaxs = Vec::with_capacity(items.len());
    for item in items {
        let (weights, ixmin, ixmax, iymin, iymax) = item?;
        let (ny, nx) = checked_weights_dimensions(ixmin, ixmax, iymin, iymax)?;
        let array = PyArray1::from_vec(py, weights).reshape([ny, nx])?;
        masks.push(array.into_any().unbind());
        ixmins.push(ixmin);
        ixmaxs.push(ixmax);
        iymins.push(iymin);
        iymaxs.push(iymax);
    }
    Ok((masks, ixmins, ixmaxs, iymins, iymaxs))
}

pub(in crate::pyapi) fn checked_weights_dimensions(
    ixmin: isize,
    ixmax: isize,
    iymin: isize,
    iymax: isize,
) -> PyResult<(usize, usize)> {
    let nx = ixmax
        .checked_sub(ixmin)
        .filter(|value| *value > 0)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
    let ny = iymax
        .checked_sub(iymin)
        .filter(|value| *value > 0)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
    ny.checked_mul(nx)
        .ok_or_else(|| PyMemoryError::new_err("weights allocation is too large"))?;
    Ok((ny, nx))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn pill_bbox(
    x: f64,
    y: f64,
    w: f64,
    a: f64,
    b: f64,
    cos_t: f64,
    sin_t: f64,
) -> (isize, isize, isize, isize) {
    let half_w = 0.5 * w;
    let rect_dx = (half_w * cos_t).abs() + (b * sin_t).abs();
    let rect_dy = (half_w * sin_t).abs() + (b * cos_t).abs();
    let (rixmin, rixmax, riymin, riymax) = extent_bbox(x, y, rect_dx, rect_dy);
    let cap_dx = half_w * cos_t;
    let cap_dy = half_w * sin_t;
    let left = Ellip {
        x0: x - cap_dx,
        y0: y - cap_dy,
        a,
        b,
        cos_t,
        sin_t,
    };
    let right = Ellip {
        x0: x + cap_dx,
        y0: y + cap_dy,
        a,
        b,
        cos_t,
        sin_t,
    };
    let (lixmin, lixmax, liymin, liymax) = ellip_bbox(left);
    let (qixmin, qixmax, qiymin, qiymax) = ellip_bbox(right);
    (
        rixmin.min(lixmin).min(qixmin),
        rixmax.max(lixmax).max(qixmax),
        riymin.min(liymin).min(qiymin),
        riymax.max(liymax).max(qiymax),
    )
}

pub(in crate::pyapi) fn allocate_weights(ny: usize, nx: usize) -> PyResult<Vec<f64>> {
    let size = ny
        .checked_mul(nx)
        .ok_or_else(|| PyMemoryError::new_err("weights allocation is too large"))?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(size)
        .map_err(|_| PyMemoryError::new_err("weights allocation failed"))?;
    weights.resize(size, 0.0);
    Ok(weights)
}

pub(in crate::pyapi) fn weights_circ_exact_values(
    x: f64,
    y: f64,
    r: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
            weights[yy * nx + xx] = circ_exact_weight_at_pixel(x, y, r, px, py);
        }
    }
    Ok(weights)
}

pub(in crate::pyapi) fn weights_circ_ann_exact_values(
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
            let outer = circ_exact_weight_at_pixel(x, y, r_out, px, py);
            let inner = if r_in == 0.0 {
                0.0
            } else {
                circ_exact_weight_at_pixel(x, y, r_in, px, py)
            };
            weights[yy * nx + xx] = (outer - inner).clamp(0.0, 1.0);
        }
    }
    Ok(weights)
}

#[inline(always)]
pub(in crate::pyapi) fn circ_exact_weight_at_pixel(
    x: f64,
    y: f64,
    r: f64,
    px: isize,
    py: isize,
) -> f64 {
    let r_inner = (r - SQRT_HALF).max(0.0);
    let r_inner2 = r_inner * r_inner;
    let r_outer = r + SQRT_HALF;
    let r_outer2 = r_outer * r_outer;
    let dx = px as f64 - x;
    let dy = py as f64 - y;
    let rpix2 = dx * dx + dy * dy;
    if rpix2 < r_inner2 {
        1.0
    } else if rpix2 > r_outer2 {
        0.0
    } else {
        circ_pixel_area(x, y, px, py, r)
    }
}

pub(in crate::pyapi) fn weights_ellip_exact_values(
    ell: Ellip,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
            weights[yy * nx + xx] = ellip_pixel_area(ell, px, py);
        }
    }
    Ok(weights)
}

pub(in crate::pyapi) fn weights_ellip_ann_exact_values(
    outer: Ellip,
    inner: Ellip,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
            let outer_weight = ellip_pixel_area(outer, px, py);
            let inner_weight = ellip_pixel_area(inner, px, py);
            weights[yy * nx + xx] = (outer_weight - inner_weight).clamp(0.0, 1.0);
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn weights_rect_exact_values(
    x: f64,
    y: f64,
    half_w: f64,
    half_h: f64,
    cos_t: f64,
    sin_t: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
            weights[yy * nx + xx] = rect_pixel_area(x, y, px, py, half_w, half_h, cos_t, sin_t);
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn weights_rect_ann_exact_values(
    x: f64,
    y: f64,
    half_w_in: f64,
    half_h_in: f64,
    half_w_out: f64,
    half_h_out: f64,
    cos_in: f64,
    sin_in: f64,
    cos_out: f64,
    sin_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
            let outer_weight =
                rect_pixel_area(x, y, px, py, half_w_out, half_h_out, cos_out, sin_out);
            let inner_weight = rect_pixel_area(x, y, px, py, half_w_in, half_h_in, cos_in, sin_in);
            weights[yy * nx + xx] = (outer_weight - inner_weight).clamp(0.0, 1.0);
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn weights_wedge_exact_values(
    wedge: Wedge,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
            weights[yy * nx + xx] = wedge_exact_weight_at_pixel(wedge, px, py);
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn weights_pill_exact_values(
    x: f64,
    y: f64,
    w: f64,
    a: f64,
    b: f64,
    cos_t: f64,
    sin_t: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    let circular_path = if (a - b).abs() <= 1.0e-14 * a.max(b).max(1.0) {
        Some(circular_pill_path(w, a, sin_t.atan2(cos_t)))
    } else {
        None
    };
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
            weights[yy * nx + xx] = if let Some(path) = &circular_path {
                path_pixel_area(path, x, y, px, py)
            } else {
                pill_exact_weight_at_pixel(x, y, w, a, b, cos_t, sin_t, px, py)
            };
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn weights_pill_ann_exact_values(
    x: f64,
    y: f64,
    w_in: f64,
    a_in: f64,
    b_in: f64,
    w_out: f64,
    a_out: f64,
    b_out: f64,
    cos_in: f64,
    sin_in: f64,
    cos_out: f64,
    sin_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
            let outer =
                pill_exact_weight_at_pixel(x, y, w_out, a_out, b_out, cos_out, sin_out, px, py);
            let inner = pill_exact_weight_at_pixel(x, y, w_in, a_in, b_in, cos_in, sin_in, px, py);
            weights[yy * nx + xx] = (outer - inner).clamp(0.0, 1.0);
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(in crate::pyapi) fn pill_exact_weight_at_pixel(
    x: f64,
    y: f64,
    w: f64,
    a: f64,
    b: f64,
    cos_t: f64,
    sin_t: f64,
    px: isize,
    py: isize,
) -> f64 {
    let half_w = 0.5 * w;
    let dx = half_w * cos_t;
    let dy = half_w * sin_t;
    let rect = rect_pixel_area(x, y, px, py, half_w, b, cos_t, sin_t);
    let left = ellip_pixel_area(
        Ellip {
            x0: x - dx,
            y0: y - dy,
            a,
            b,
            cos_t,
            sin_t,
        },
        px,
        py,
    );
    let right = ellip_pixel_area(
        Ellip {
            x0: x + dx,
            y0: y + dy,
            a,
            b,
            cos_t,
            sin_t,
        },
        px,
        py,
    );
    rect.max(left).max(right).clamp(0.0, 1.0)
}

pub(in crate::pyapi) fn weights_circ_center_values(
    x: f64,
    y: f64,
    r: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let r2 = r * r;
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
            as f64;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
                as f64;
            let dx = px - x;
            let dy = py - y;
            weights[yy * nx + xx] = if dx * dx + dy * dy < r2 { 1.0 } else { 0.0 };
        }
    }
    Ok(weights)
}

pub(in crate::pyapi) fn weights_circ_ann_center_values(
    x: f64,
    y: f64,
    r_in: f64,
    r_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let r_in2 = r_in * r_in;
    let r_out2 = r_out * r_out;
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
            as f64;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
                as f64;
            let dx = px - x;
            let dy = py - y;
            let rpix2 = dx * dx + dy * dy;
            weights[yy * nx + xx] = if rpix2 < r_out2 && rpix2 >= r_in2 {
                1.0
            } else {
                0.0
            };
        }
    }
    Ok(weights)
}

pub(in crate::pyapi) fn weights_ellip_center_values(
    ell: Ellip,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
            as f64;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
                as f64;
            let dx = px - ell.x0;
            let dy = py - ell.y0;
            let u = (ell.cos_t * dx + ell.sin_t * dy) / ell.a;
            let v = (-ell.sin_t * dx + ell.cos_t * dy) / ell.b;
            weights[yy * nx + xx] = if u * u + v * v < 1.0 { 1.0 } else { 0.0 };
        }
    }
    Ok(weights)
}

pub(in crate::pyapi) fn weights_ellip_ann_center_values(
    outer: Ellip,
    inner: Ellip,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
            as f64;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
                as f64;
            let dx = px - outer.x0;
            let dy = py - outer.y0;
            let outer_u = (outer.cos_t * dx + outer.sin_t * dy) / outer.a;
            let outer_v = (-outer.sin_t * dx + outer.cos_t * dy) / outer.b;
            let inner_u = (inner.cos_t * dx + inner.sin_t * dy) / inner.a;
            let inner_v = (-inner.sin_t * dx + inner.cos_t * dy) / inner.b;
            let inside_outer = outer_u * outer_u + outer_v * outer_v < 1.0;
            let inside_inner = inner_u * inner_u + inner_v * inner_v < 1.0;
            weights[yy * nx + xx] = if inside_outer && !inside_inner {
                1.0
            } else {
                0.0
            };
        }
    }
    Ok(weights)
}

pub(in crate::pyapi) fn weights_wedge_center_values(
    wedge: Wedge,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?;
            weights[yy * nx + xx] = wedge_center_weight_at_pixel(wedge, px, py);
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn weights_rect_center_values(
    x: f64,
    y: f64,
    half_w: f64,
    half_h: f64,
    cos_t: f64,
    sin_t: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
            as f64;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
                as f64;
            let dx = px - x;
            let dy = py - y;
            let u = cos_t * dx + sin_t * dy;
            let v = -sin_t * dx + cos_t * dy;
            weights[yy * nx + xx] = if u.abs() < half_w && v.abs() < half_h {
                1.0
            } else {
                0.0
            };
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn weights_rect_ann_center_values(
    x: f64,
    y: f64,
    half_w_in: f64,
    half_h_in: f64,
    half_w_out: f64,
    half_h_out: f64,
    cos_in: f64,
    sin_in: f64,
    cos_out: f64,
    sin_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
            as f64;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
                as f64;
            let dx = px - x;
            let dy = py - y;
            let outer_u = cos_out * dx + sin_out * dy;
            let outer_v = -sin_out * dx + cos_out * dy;
            let inner_u = cos_in * dx + sin_in * dy;
            let inner_v = -sin_in * dx + cos_in * dy;
            let inside_outer = outer_u.abs() < half_w_out && outer_v.abs() < half_h_out;
            let inside_inner = inner_u.abs() < half_w_in && inner_v.abs() < half_h_in;
            weights[yy * nx + xx] = if inside_outer && !inside_inner {
                1.0
            } else {
                0.0
            };
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn weights_pill_center_values(
    x: f64,
    y: f64,
    w: f64,
    a: f64,
    b: f64,
    cos_t: f64,
    sin_t: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
            as f64;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
                as f64;
            weights[yy * nx + xx] = if pill_center_contains(x, y, w, a, b, cos_t, sin_t, px, py) {
                1.0
            } else {
                0.0
            };
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::pyapi) fn weights_pill_ann_center_values(
    x: f64,
    y: f64,
    w_in: f64,
    a_in: f64,
    b_in: f64,
    w_out: f64,
    a_out: f64,
    b_out: f64,
    cos_in: f64,
    sin_in: f64,
    cos_out: f64,
    sin_out: f64,
    x_min: isize,
    y_min: isize,
    ny: usize,
    nx: usize,
) -> PyResult<Vec<f64>> {
    let mut weights = allocate_weights(ny, nx)?;
    for yy in 0..ny {
        let py = y_min
            .checked_add(yy as isize)
            .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
            as f64;
        for xx in 0..nx {
            let px = x_min
                .checked_add(xx as isize)
                .ok_or_else(|| PyValueError::new_err("invalid weights extent"))?
                as f64;
            let outer = pill_center_contains(x, y, w_out, a_out, b_out, cos_out, sin_out, px, py);
            let inner = pill_center_contains(x, y, w_in, a_in, b_in, cos_in, sin_in, px, py);
            weights[yy * nx + xx] = if outer && !inner { 1.0 } else { 0.0 };
        }
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(in crate::pyapi) fn pill_center_contains(
    x: f64,
    y: f64,
    w: f64,
    a: f64,
    b: f64,
    cos_t: f64,
    sin_t: f64,
    px: f64,
    py: f64,
) -> bool {
    let half_w = 0.5 * w;
    let dx = px - x;
    let dy = py - y;
    let u = cos_t * dx + sin_t * dy;
    let v = -sin_t * dx + cos_t * dy;
    if u.abs() < half_w && v.abs() < b {
        return true;
    }
    let cap_dx = half_w * cos_t;
    let cap_dy = half_w * sin_t;
    let left_dx = px - (x - cap_dx);
    let left_dy = py - (y - cap_dy);
    let left_u = (cos_t * left_dx + sin_t * left_dy) / a;
    let left_v = (-sin_t * left_dx + cos_t * left_dy) / b;
    if left_u * left_u + left_v * left_v < 1.0 {
        return true;
    }
    let right_dx = px - (x + cap_dx);
    let right_dy = py - (y + cap_dy);
    let right_u = (cos_t * right_dx + sin_t * right_dy) / a;
    let right_v = (-sin_t * right_dx + cos_t * right_dy) / b;
    right_u * right_u + right_v * right_v < 1.0
}

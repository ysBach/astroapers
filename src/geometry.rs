//! Pure geometry and pixel-overlap helpers.
#![cfg_attr(not(feature = "python"), allow(dead_code))]

use std::f64::consts::{FRAC_1_SQRT_2, PI};

pub(crate) const SQRT_HALF: f64 = FRAC_1_SQRT_2;
const CLIP_EPS: f64 = 1.0e-15;

#[derive(Clone, Copy)]
pub(crate) struct Ellip {
    pub(crate) x0: f64,
    pub(crate) y0: f64,
    pub(crate) a: f64,
    pub(crate) b: f64,
    pub(crate) cos_t: f64,
    pub(crate) sin_t: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct Wedge {
    pub(crate) x0: f64,
    pub(crate) y0: f64,
    pub(crate) r_in: f64,
    pub(crate) r_out: f64,
    pub(crate) theta_in: f64,
    pub(crate) dtheta_in: f64,
    pub(crate) theta_out: f64,
    pub(crate) dtheta_out: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Point {
    x: f64,
    y: f64,
}

#[inline(always)]
pub(crate) fn circ_bbox(x: f64, y: f64, r: f64) -> (isize, isize, isize, isize) {
    extent_bbox(x, y, r, r)
}

#[inline(always)]
pub(crate) fn wedge_bbox(wedge: Wedge) -> (isize, isize, isize, isize) {
    if wedge.dtheta_in >= 2.0 * PI || wedge.dtheta_out >= 2.0 * PI {
        return circ_bbox(wedge.x0, wedge.y0, wedge.r_out);
    }

    let bounds = wedge_bounds(wedge);

    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;

    for (angle, r) in [
        (bounds.lower_in, wedge.r_in),
        (bounds.upper_in, wedge.r_in),
        (bounds.lower_out, wedge.r_out),
        (bounds.upper_out, wedge.r_out),
    ] {
        let x = wedge.x0 + r * angle.cos();
        let y = wedge.y0 + r * angle.sin();
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    for k in -4..=4 {
        let angle = k as f64 * 0.5 * PI;
        if angle_in_interval(angle, bounds.lower_out, bounds.upper_out) {
            let x = wedge.x0 + wedge.r_out * angle.cos();
            let y = wedge.y0 + wedge.r_out * angle.sin();
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
        if angle_in_interval(angle, bounds.lower_in, bounds.upper_in) {
            let x = wedge.x0 + wedge.r_in * angle.cos();
            let y = wedge.y0 + wedge.r_in * angle.sin();
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
    }
    let margin = SQRT_HALF;
    (
        (xmin - margin + 0.5).floor() as isize,
        (xmax + margin + 0.5).ceil() as isize,
        (ymin - margin + 0.5).floor() as isize,
        (ymax + margin + 0.5).ceil() as isize,
    )
}

#[inline(always)]
pub(crate) fn extent_bbox(x: f64, y: f64, dx: f64, dy: f64) -> (isize, isize, isize, isize) {
    let ixmin = ((x - dx) + 0.5).floor() as isize;
    let ixmax = ((x + dx) + 0.5).ceil() as isize;
    let iymin = ((y - dy) + 0.5).floor() as isize;
    let iymax = ((y + dy) + 0.5).ceil() as isize;
    (ixmin, ixmax, iymin, iymax)
}

#[inline(always)]
pub(crate) fn bbox_in_frame(
    ixmin: isize,
    ixmax: isize,
    iymin: isize,
    iymax: isize,
    ny: usize,
    nx: usize,
) -> bool {
    ixmin >= 0 && iymin >= 0 && ixmax <= nx as isize && iymax <= ny as isize
}

#[inline(always)]
pub(crate) fn ellip_inner_outer_radius2(a: f64, b: f64, cos_t: f64, sin_t: f64) -> (f64, f64) {
    let mut delta = 0.0_f64;
    for dx in [-0.5_f64, 0.5] {
        for dy in [-0.5_f64, 0.5] {
            let u = (cos_t * dx + sin_t * dy) / a;
            let v = (-sin_t * dx + cos_t * dy) / b;
            delta = delta.max((u * u + v * v).sqrt());
        }
    }
    let r_in = (1.0 - delta).max(0.0);
    let r_out = 1.0 + delta;
    (r_in * r_in, r_out * r_out)
}

#[inline(always)]
pub(crate) fn ellip_bbox(ell: Ellip) -> (isize, isize, isize, isize) {
    let dx = ((ell.a * ell.cos_t).powi(2) + (ell.b * ell.sin_t).powi(2)).sqrt();
    let dy = ((ell.a * ell.sin_t).powi(2) + (ell.b * ell.cos_t).powi(2)).sqrt();
    extent_bbox(ell.x0, ell.y0, dx, dy)
}
#[inline(always)]
pub(crate) fn circ_pixel_area(x0: f64, y0: f64, xx: isize, yy: isize, r: f64) -> f64 {
    let xlo = xx as f64 - 0.5;
    let xhi = xx as f64 + 0.5;
    let ylo = yy as f64 - 0.5;
    let yhi = yy as f64 + 0.5;
    circ_rect_overlap(xlo - x0, ylo - y0, xhi - x0, yhi - y0, r).clamp(0.0, 1.0)
}

// Specialized rectangle-circle overlap adapted from SEP's BSD-licensed overlap.h.
#[inline(always)]
fn circ_rect_overlap(xmin: f64, ymin: f64, xmax: f64, ymax: f64, r: f64) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    if xmin >= 0.0 {
        if ymin >= 0.0 {
            circ_rect_overlap_core(xmin, ymin, xmax, ymax, r)
        } else if ymax <= 0.0 {
            circ_rect_overlap_core(-ymax, xmin, -ymin, xmax, r)
        } else {
            circ_rect_overlap(xmin, ymin, xmax, 0.0, r)
                + circ_rect_overlap(xmin, 0.0, xmax, ymax, r)
        }
    } else if xmax <= 0.0 {
        if ymin >= 0.0 {
            circ_rect_overlap_core(-xmax, ymin, -xmin, ymax, r)
        } else if ymax <= 0.0 {
            circ_rect_overlap_core(-xmax, -ymax, -xmin, -ymin, r)
        } else {
            circ_rect_overlap(xmin, ymin, xmax, 0.0, r)
                + circ_rect_overlap(xmin, 0.0, xmax, ymax, r)
        }
    } else if ymin >= 0.0 || ymax <= 0.0 {
        circ_rect_overlap(xmin, ymin, 0.0, ymax, r) + circ_rect_overlap(0.0, ymin, xmax, ymax, r)
    } else {
        circ_rect_overlap(xmin, ymin, 0.0, 0.0, r)
            + circ_rect_overlap(0.0, ymin, xmax, 0.0, r)
            + circ_rect_overlap(xmin, 0.0, 0.0, ymax, r)
            + circ_rect_overlap(0.0, 0.0, xmax, ymax, r)
    }
}

#[inline(always)]
fn circ_rect_overlap_core(xmin: f64, ymin: f64, xmax: f64, ymax: f64, r: f64) -> f64 {
    let xmin2 = xmin * xmin;
    let ymin2 = ymin * ymin;
    let r2 = r * r;
    if xmin2 + ymin2 > r2 {
        return 0.0;
    }

    let xmax2 = xmax * xmax;
    let ymax2 = ymax * ymax;
    if xmax2 + ymax2 < r2 {
        return (xmax - xmin) * (ymax - ymin);
    }

    let corner1 = xmax2 + ymin2;
    let corner2 = xmin2 + ymax2;
    if corner1 < r2 && corner2 < r2 {
        let x1 = (r2 - ymax2).sqrt();
        let y1 = ymax;
        let x2 = xmax;
        let y2 = (r2 - xmax2).sqrt();
        return (xmax - xmin) * (ymax - ymin) - tri_area(x1, y1, x2, y2, xmax, ymax)
            + circ_arc_area(x1, y1, x2, y2, r);
    }

    if corner1 < r2 {
        let x1 = xmin;
        let y1 = (r2 - xmin2).sqrt();
        let x2 = xmax;
        let y2 = (r2 - xmax2).sqrt();
        return circ_arc_area(x1, y1, x2, y2, r)
            + tri_area(x1, y1, x1, ymin, xmax, ymin)
            + tri_area(x1, y1, x2, ymin, x2, y2);
    }

    if corner2 < r2 {
        let x1 = (r2 - ymin2).sqrt();
        let y1 = ymin;
        let x2 = (r2 - ymax2).sqrt();
        let y2 = ymax;
        return circ_arc_area(x1, y1, x2, y2, r)
            + tri_area(x1, y1, xmin, y1, xmin, ymax)
            + tri_area(x1, y1, xmin, y2, x2, y2);
    }

    let x1 = (r2 - ymin2).sqrt();
    let y1 = ymin;
    let x2 = xmin;
    let y2 = (r2 - xmin2).sqrt();
    circ_arc_area(x1, y1, x2, y2, r) + tri_area(x1, y1, x2, y2, xmin, ymin)
}

#[inline(always)]
fn circ_arc_area(x1: f64, y1: f64, x2: f64, y2: f64, r: f64) -> f64 {
    let chord = ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)).sqrt();
    let theta = 2.0 * (0.5 * chord / r).asin();
    0.5 * r * r * (theta - theta.sin())
}

#[inline(always)]
fn tri_area(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) -> f64 {
    0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)).abs()
}

#[inline(always)]
pub(crate) fn wedge_contains_point(wedge: Wedge, x: f64, y: f64) -> bool {
    if wedge.dtheta_in >= 2.0 * PI && wedge.dtheta_out >= 2.0 * PI {
        let dx = x - wedge.x0;
        let dy = y - wedge.y0;
        let r2 = dx * dx + dy * dy;
        return r2 >= wedge.r_in * wedge.r_in && r2 < wedge.r_out * wedge.r_out;
    }

    let p = Point {
        x: x - wedge.x0,
        y: y - wedge.y0,
    };
    let parts = wedge_parts(wedge);
    (point_in_polygon(p, &parts.quad)
        || point_in_circular_segment(p, wedge.r_out, parts.lower_out, wedge.dtheta_out))
        && !point_in_circular_segment(p, wedge.r_in, parts.lower_in, wedge.dtheta_in)
}

#[inline(always)]
pub(crate) fn wedge_center_weight_at_pixel(wedge: Wedge, px: isize, py: isize) -> f64 {
    if wedge_contains_point(wedge, px as f64, py as f64) {
        1.0
    } else {
        0.0
    }
}

#[inline(always)]
pub(crate) fn wedge_exact_weight_at_pixel(wedge: Wedge, px: isize, py: isize) -> f64 {
    if wedge.dtheta_in >= 2.0 * PI && wedge.dtheta_out >= 2.0 * PI {
        let outer = circ_pixel_area(wedge.x0, wedge.y0, px, py, wedge.r_out);
        let inner = circ_pixel_area(wedge.x0, wedge.y0, px, py, wedge.r_in);
        return (outer - inner).clamp(0.0, 1.0);
    }

    let xlo = px as f64 - 0.5 - wedge.x0;
    let xhi = px as f64 + 0.5 - wedge.x0;
    let ylo = py as f64 - 0.5 - wedge.y0;
    let yhi = py as f64 + 0.5 - wedge.y0;
    let square = [
        Point { x: xlo, y: ylo },
        Point { x: xhi, y: ylo },
        Point { x: xhi, y: yhi },
        Point { x: xlo, y: yhi },
    ];

    if (wedge.theta_out - wedge.theta_in).abs() <= 1.0e-14
        && (wedge.dtheta_out - wedge.dtheta_in).abs() <= 1.0e-14
    {
        return annular_sector_pixel_area(
            &square,
            wedge.r_in,
            wedge.r_out,
            wedge.theta_in - 0.5 * wedge.dtheta_in,
            wedge.dtheta_in,
        );
    }

    let parts = wedge_parts(wedge);
    let quad = polygon_rect_overlap_area(&parts.quad, xlo, xhi, ylo, yhi);
    let outer_sector = sector_pixel_area(&square, wedge.r_out, parts.lower_out, wedge.dtheta_out);
    let inner_sector = sector_pixel_area(&square, wedge.r_in, parts.lower_in, wedge.dtheta_in);
    let outer_triangle = polygon_rect_overlap_area(&parts.outer_triangle, xlo, xhi, ylo, yhi);
    let inner_triangle = polygon_rect_overlap_area(&parts.inner_triangle, xlo, xhi, ylo, yhi);
    let outer_sign = wedge.dtheta_out.sin().signum();
    let inner_sign = wedge.dtheta_in.sin().signum();

    (quad + outer_sector - outer_sign * outer_triangle - inner_sector + inner_sign * inner_triangle)
        .clamp(0.0, 1.0)
}

#[inline(always)]
fn angle_in_interval(angle: f64, start: f64, end: f64) -> bool {
    (angle - start).rem_euclid(2.0 * PI) <= (end - start).rem_euclid(2.0 * PI) + CLIP_EPS
}

#[inline(always)]
fn angular_signed_delta(reference: f64, angle: f64) -> f64 {
    (angle - reference + PI).rem_euclid(2.0 * PI) - PI
}

#[derive(Clone, Copy)]
struct WedgeBounds {
    lower_in: f64,
    upper_in: f64,
    lower_out: f64,
    upper_out: f64,
}

#[derive(Clone, Copy)]
struct WedgeParts {
    lower_in: f64,
    lower_out: f64,
    quad: [Point; 4],
    outer_triangle: [Point; 3],
    inner_triangle: [Point; 3],
}

#[inline(always)]
fn wedge_bounds(wedge: Wedge) -> WedgeBounds {
    let lower_in = wedge.theta_in - 0.5 * wedge.dtheta_in;
    let upper_in = lower_in + wedge.dtheta_in;
    let lower_out =
        lower_in + angular_signed_delta(lower_in, wedge.theta_out - 0.5 * wedge.dtheta_out);
    let upper_out = lower_out + wedge.dtheta_out;
    WedgeBounds {
        lower_in,
        upper_in,
        lower_out,
        upper_out,
    }
}

#[inline(always)]
fn wedge_parts(wedge: Wedge) -> WedgeParts {
    let bounds = wedge_bounds(wedge);
    let inner_lower = polar_point(wedge.r_in, bounds.lower_in);
    let outer_lower = polar_point(wedge.r_out, bounds.lower_out);
    let outer_upper = polar_point(wedge.r_out, bounds.upper_out);
    let inner_upper = polar_point(wedge.r_in, bounds.upper_in);
    let origin = Point { x: 0.0, y: 0.0 };
    WedgeParts {
        lower_in: bounds.lower_in,
        lower_out: bounds.lower_out,
        quad: [inner_lower, outer_lower, outer_upper, inner_upper],
        outer_triangle: [origin, outer_lower, outer_upper],
        inner_triangle: [origin, inner_lower, inner_upper],
    }
}

#[inline(always)]
fn polar_point(r: f64, theta: f64) -> Point {
    Point {
        x: r * theta.cos(),
        y: r * theta.sin(),
    }
}

#[inline(always)]
fn annular_sector_pixel_area(
    square: &[Point; 4],
    r_in: f64,
    r_out: f64,
    angle0: f64,
    dtheta: f64,
) -> f64 {
    if dtheta >= 2.0 * PI {
        let outer = circle_polygon_overlap_area_scaled(square, r_out);
        let inner = circle_polygon_overlap_area_scaled(square, r_in);
        return (outer - inner).max(0.0);
    }
    let mut total = 0.0;
    let mut start = angle0;
    let mut remaining = dtheta;
    while remaining > 1.0e-14 {
        let width = remaining.min(0.5 * PI);
        let end = start + width;
        let poly = clip_polygon_to_angle_interval(square, start, end);
        if poly.len() >= 3 {
            let outer = circle_polygon_overlap_area_scaled(&poly, r_out);
            let inner = circle_polygon_overlap_area_scaled(&poly, r_in);
            total += (outer - inner).max(0.0);
        }
        start = end;
        remaining -= width;
    }
    total
}

#[inline(always)]
fn sector_pixel_area(square: &[Point; 4], r: f64, angle0: f64, dtheta: f64) -> f64 {
    annular_sector_pixel_area(square, 0.0, r, angle0, dtheta)
}

fn polygon_rect_overlap_area(poly: &[Point], xlo: f64, xhi: f64, ylo: f64, yhi: f64) -> f64 {
    let clipped = clip_polygon_to_rect(poly, xlo, xhi, ylo, yhi);
    polygon_area(&clipped)
}

fn clip_polygon_to_rect(poly: &[Point], xlo: f64, xhi: f64, ylo: f64, yhi: f64) -> Vec<Point> {
    let clipped = clip_polygon_by_axis(poly, true, xlo, true);
    let clipped = clip_polygon_by_axis(&clipped, true, xhi, false);
    let clipped = clip_polygon_by_axis(&clipped, false, ylo, true);
    clip_polygon_by_axis(&clipped, false, yhi, false)
}

fn clip_polygon_by_axis(poly: &[Point], use_x: bool, bound: f64, keep_greater: bool) -> Vec<Point> {
    if poly.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(poly.len() + 2);
    let mut prev = poly[poly.len() - 1];
    let mut prev_inside = axis_inside(prev, use_x, bound, keep_greater);
    for &cur in poly {
        let cur_inside = axis_inside(cur, use_x, bound, keep_greater);
        if cur_inside {
            if !prev_inside {
                out.push(intersect_segment_axis(prev, cur, use_x, bound));
            }
            out.push(cur);
        } else if prev_inside {
            out.push(intersect_segment_axis(prev, cur, use_x, bound));
        }
        prev = cur;
        prev_inside = cur_inside;
    }
    out
}

#[inline(always)]
fn axis_inside(p: Point, use_x: bool, bound: f64, keep_greater: bool) -> bool {
    let value = if use_x { p.x } else { p.y };
    if keep_greater {
        value >= bound - CLIP_EPS
    } else {
        value <= bound + CLIP_EPS
    }
}

#[inline(always)]
fn intersect_segment_axis(a: Point, b: Point, use_x: bool, bound: f64) -> Point {
    let denom = if use_x { b.x - a.x } else { b.y - a.y };
    if denom.abs() <= CLIP_EPS {
        return a;
    }
    let value = if use_x { a.x } else { a.y };
    let t = (bound - value) / denom;
    Point {
        x: a.x + t * (b.x - a.x),
        y: a.y + t * (b.y - a.y),
    }
}

#[inline(always)]
fn polygon_area(poly: &[Point]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut twice_area = 0.0;
    for i in 0..poly.len() {
        twice_area += cross(poly[i], poly[(i + 1) % poly.len()]);
    }
    0.5 * twice_area.abs()
}

fn point_in_polygon(p: Point, poly: &[Point]) -> bool {
    let mut inside = false;
    let mut prev = poly[poly.len() - 1];
    for &cur in poly {
        let crosses = (cur.y > p.y) != (prev.y > p.y);
        if crosses {
            let x = (prev.x - cur.x) * (p.y - cur.y) / (prev.y - cur.y) + cur.x;
            if p.x < x {
                inside = !inside;
            }
        }
        prev = cur;
    }
    inside
}

fn point_in_circular_segment(p: Point, r: f64, angle0: f64, dtheta: f64) -> bool {
    if point_norm2(p) >= r * r - CLIP_EPS {
        return false;
    }
    let in_sector = angle_in_interval(p.y.atan2(p.x), angle0, angle0 + dtheta);
    let a = polar_point(r, angle0);
    let b = polar_point(r, angle0 + dtheta);
    let in_triangle = point_in_polygon(p, &[Point { x: 0.0, y: 0.0 }, a, b]);
    if dtheta.sin() >= 0.0 {
        in_sector && !in_triangle
    } else {
        in_sector || in_triangle
    }
}

fn clip_polygon_to_angle_interval(poly: &[Point], angle0: f64, angle1: f64) -> Vec<Point> {
    let start = Point {
        x: angle0.cos(),
        y: angle0.sin(),
    };
    let end = Point {
        x: angle1.cos(),
        y: angle1.sin(),
    };
    let clipped = clip_polygon_by_origin_line(poly, start, true);
    clip_polygon_by_origin_line(&clipped, end, false)
}

fn clip_polygon_by_origin_line(poly: &[Point], dir: Point, keep_left: bool) -> Vec<Point> {
    if poly.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(poly.len() + 2);
    let mut prev = poly[poly.len() - 1];
    let mut prev_inside = origin_line_inside(dir, prev, keep_left);
    for &cur in poly {
        let cur_inside = origin_line_inside(dir, cur, keep_left);
        if cur_inside {
            if !prev_inside {
                out.push(intersect_segment_origin_line(prev, cur, dir));
            }
            out.push(cur);
        } else if prev_inside {
            out.push(intersect_segment_origin_line(prev, cur, dir));
        }
        prev = cur;
        prev_inside = cur_inside;
    }
    out
}

#[inline(always)]
fn origin_line_inside(dir: Point, p: Point, keep_left: bool) -> bool {
    let value = cross(dir, p);
    if keep_left {
        value >= -CLIP_EPS
    } else {
        value <= CLIP_EPS
    }
}

#[inline(always)]
fn intersect_segment_origin_line(a: Point, b: Point, dir: Point) -> Point {
    let ab = Point {
        x: b.x - a.x,
        y: b.y - a.y,
    };
    let denom = cross(dir, ab);
    if denom.abs() <= CLIP_EPS {
        return a;
    }
    let t = -cross(dir, a) / denom;
    Point {
        x: a.x + t * ab.x,
        y: a.y + t * ab.y,
    }
}

fn circle_polygon_overlap_area_scaled(poly: &[Point], r: f64) -> f64 {
    if r <= 0.0 || poly.len() < 3 {
        return 0.0;
    }
    let scaled: Vec<_> = poly
        .iter()
        .map(|p| Point {
            x: p.x / r,
            y: p.y / r,
        })
        .collect();
    circ_polygon_overlap_area(&scaled) * r * r
}

#[inline(always)]
pub(crate) fn ellip_pixel_area(ell: Ellip, xx: isize, yy: isize) -> f64 {
    let xlo = xx as f64 - 0.5;
    let xhi = xx as f64 + 0.5;
    let ylo = yy as f64 - 0.5;
    let yhi = yy as f64 + 0.5;
    let poly = [
        ellip_normalized_point(ell, xlo, ylo),
        ellip_normalized_point(ell, xhi, ylo),
        ellip_normalized_point(ell, xhi, yhi),
        ellip_normalized_point(ell, xlo, yhi),
    ];
    if poly.iter().all(|&p| point_norm2(p) <= 1.0) {
        return 1.0;
    }
    (circ_polygon_overlap_area(&poly) * ell.a * ell.b).clamp(0.0, 1.0)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rect_pixel_area(
    x0: f64,
    y0: f64,
    xx: isize,
    yy: isize,
    half_w: f64,
    half_h: f64,
    cos_t: f64,
    sin_t: f64,
) -> f64 {
    let mut ax = [0.0_f64; 16];
    let mut ay = [0.0_f64; 16];
    let mut bx = [0.0_f64; 16];
    let mut by = [0.0_f64; 16];

    let corners = [
        (xx as f64 - 0.5, yy as f64 - 0.5),
        (xx as f64 + 0.5, yy as f64 - 0.5),
        (xx as f64 + 0.5, yy as f64 + 0.5),
        (xx as f64 - 0.5, yy as f64 + 0.5),
    ];
    for (i, (xw, yw)) in corners.into_iter().enumerate() {
        let dx = xw - x0;
        let dy = yw - y0;
        ax[i] = cos_t * dx + sin_t * dy;
        ay[i] = -sin_t * dx + cos_t * dy;
    }

    let mut n = 4;
    n = clip_x_fixed(&ax, &ay, n, &mut bx, &mut by, -half_w, true);
    n = clip_x_fixed(&bx, &by, n, &mut ax, &mut ay, half_w, false);
    n = clip_y_fixed(&ax, &ay, n, &mut bx, &mut by, -half_h, true);
    n = clip_y_fixed(&bx, &by, n, &mut ax, &mut ay, half_h, false);
    polygon_area_fixed(&ax, &ay, n).clamp(0.0, 1.0)
}

#[inline(always)]
fn clip_x_fixed(
    in_x: &[f64; 16],
    in_y: &[f64; 16],
    n_in: usize,
    out_x: &mut [f64; 16],
    out_y: &mut [f64; 16],
    bound: f64,
    keep_greater: bool,
) -> usize {
    clip_fixed(
        in_x,
        in_y,
        n_in,
        out_x,
        out_y,
        |x, _| {
            if keep_greater {
                x >= bound
            } else {
                x <= bound
            }
        },
        |x0, y0, x1, y1| {
            let denom = x1 - x0;
            if denom.abs() < CLIP_EPS {
                (bound, 0.5 * (y0 + y1))
            } else {
                let t = (bound - x0) / denom;
                (bound, y0 + t * (y1 - y0))
            }
        },
    )
}

#[inline(always)]
fn clip_y_fixed(
    in_x: &[f64; 16],
    in_y: &[f64; 16],
    n_in: usize,
    out_x: &mut [f64; 16],
    out_y: &mut [f64; 16],
    bound: f64,
    keep_greater: bool,
) -> usize {
    clip_fixed(
        in_x,
        in_y,
        n_in,
        out_x,
        out_y,
        |_, y| {
            if keep_greater {
                y >= bound
            } else {
                y <= bound
            }
        },
        |x0, y0, x1, y1| {
            let denom = y1 - y0;
            if denom.abs() < CLIP_EPS {
                (0.5 * (x0 + x1), bound)
            } else {
                let t = (bound - y0) / denom;
                (x0 + t * (x1 - x0), bound)
            }
        },
    )
}

fn clip_fixed<Inside, Intersect>(
    in_x: &[f64; 16],
    in_y: &[f64; 16],
    n_in: usize,
    out_x: &mut [f64; 16],
    out_y: &mut [f64; 16],
    inside: Inside,
    intersect: Intersect,
) -> usize
where
    Inside: Fn(f64, f64) -> bool,
    Intersect: Fn(f64, f64, f64, f64) -> (f64, f64),
{
    if n_in == 0 {
        return 0;
    }
    let mut n_out = 0;
    let mut px = in_x[n_in - 1];
    let mut py = in_y[n_in - 1];
    let mut prev_inside = inside(px, py);
    for i in 0..n_in {
        let cx = in_x[i];
        let cy = in_y[i];
        let cur_inside = inside(cx, cy);
        if cur_inside {
            if !prev_inside {
                let (x, y) = intersect(px, py, cx, cy);
                out_x[n_out] = x;
                out_y[n_out] = y;
                n_out += 1;
            }
            out_x[n_out] = cx;
            out_y[n_out] = cy;
            n_out += 1;
        } else if prev_inside {
            let (x, y) = intersect(px, py, cx, cy);
            out_x[n_out] = x;
            out_y[n_out] = y;
            n_out += 1;
        }
        px = cx;
        py = cy;
        prev_inside = cur_inside;
    }
    n_out
}

#[inline(always)]
fn polygon_area_fixed(xs: &[f64; 16], ys: &[f64; 16], n: usize) -> f64 {
    if n < 3 {
        return 0.0;
    }
    let mut twice_area = 0.0;
    for i in 0..n {
        let j = if i + 1 == n { 0 } else { i + 1 };
        twice_area += xs[i] * ys[j] - xs[j] * ys[i];
    }
    0.5 * twice_area.abs()
}

#[inline(always)]
fn ellip_normalized_point(ell: Ellip, x: f64, y: f64) -> Point {
    let dx = x - ell.x0;
    let dy = y - ell.y0;
    Point {
        x: (ell.cos_t * dx + ell.sin_t * dy) / ell.a,
        y: (-ell.sin_t * dx + ell.cos_t * dy) / ell.b,
    }
}

#[inline(always)]
fn point_norm2(p: Point) -> f64 {
    p.x * p.x + p.y * p.y
}

#[inline(always)]
fn circ_polygon_overlap_area(poly: &[Point]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut signed_area = 0.0;
    for i in 0..poly.len() {
        signed_area += circ_edge_area(poly[i], poly[(i + 1) % poly.len()]);
    }
    signed_area.abs().min(PI)
}

#[inline(always)]
fn circ_edge_area(a: Point, b: Point) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let aa = dx * dx + dy * dy;
    let bb = 2.0 * (a.x * dx + a.y * dy);
    let cc = a.x * a.x + a.y * a.y - 1.0;
    let disc = bb * bb - 4.0 * aa * cc;
    let mut ts = [0.0_f64; 4];
    let mut n = 2;
    ts[0] = 0.0;
    ts[1] = 1.0;
    if aa > 0.0 && disc > 0.0 {
        let sqrt_disc = disc.sqrt();
        let t1 = (-bb - sqrt_disc) / (2.0 * aa);
        let t2 = (-bb + sqrt_disc) / (2.0 * aa);
        if t1 > 0.0 && t1 < 1.0 {
            ts[n] = t1;
            n += 1;
        }
        if t2 > 0.0 && t2 < 1.0 {
            ts[n] = t2;
            n += 1;
        }
    }
    ts[..n].sort_by(|x, y| x.total_cmp(y));

    let mut area = 0.0;
    for i in 0..(n - 1) {
        let t0 = ts[i];
        let t1 = ts[i + 1];
        if t1 <= t0 {
            continue;
        }
        let p = Point {
            x: a.x + dx * t0,
            y: a.y + dy * t0,
        };
        let q = Point {
            x: a.x + dx * t1,
            y: a.y + dy * t1,
        };
        let mid = Point {
            x: a.x + dx * (0.5 * (t0 + t1)),
            y: a.y + dy * (0.5 * (t0 + t1)),
        };
        if point_norm2(mid) <= 1.0 {
            area += 0.5 * cross(p, q);
        } else {
            area += 0.5 * cross(p, q).atan2(dot(p, q));
        }
    }
    area
}

#[inline(always)]
fn cross(a: Point, b: Point) -> f64 {
    a.x * b.y - a.y * b.x
}

#[inline(always)]
fn dot(a: Point, b: Point) -> f64 {
    a.x * b.x + a.y * b.y
}

// ─────────────────────────────────────────────────────────────────────────────
// Path aperture geometry (line + circular-arc segments, Green's theorem)
// ─────────────────────────────────────────────────────────────────────────────

const PATH_EPS: f64 = 1.0e-10;
pub(crate) const SEG_MOVE: i8 = 0;
pub(crate) const SEG_LINE: i8 = 1;
pub(crate) const SEG_ARC: i8 = 2;
pub(crate) const SEG_CLOSE: i8 = 3;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ArcData {
    pub(crate) center: Point,
    pub(crate) r: f64,
    pub(crate) theta0: f64,
    pub(crate) dtheta: f64,
}

#[derive(Clone, Debug)]
pub(crate) enum PathSeg {
    Line { p0: Point, p1: Point },
    Arc { arc: ArcData, p0: Point, p1: Point },
}

#[derive(Clone, Debug)]
pub(crate) struct PathContour {
    pub(crate) segs: Vec<PathSeg>,
    pub(crate) signed_area: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedPath {
    pub(crate) contours: Vec<PathContour>,
}

impl PathSeg {
    pub(crate) fn p0(&self) -> Point {
        match self {
            PathSeg::Line { p0, .. } => *p0,
            PathSeg::Arc { p0, .. } => *p0,
        }
    }

    fn point_at(&self, t: f64) -> Point {
        match self {
            PathSeg::Line { p0, p1 } => Point {
                x: p0.x + t * (p1.x - p0.x),
                y: p0.y + t * (p1.y - p0.y),
            },
            PathSeg::Arc { arc, .. } => {
                let theta = arc.theta0 + t * arc.dtheta;
                Point {
                    x: arc.center.x + arc.r * theta.cos(),
                    y: arc.center.y + arc.r * theta.sin(),
                }
            }
        }
    }

    fn green_integral(&self) -> f64 {
        match self {
            PathSeg::Line { p0, p1 } => 0.5 * cross(*p0, *p1),
            PathSeg::Arc { arc, p0, p1 } => {
                let dp = Point {
                    x: p1.x - p0.x,
                    y: p1.y - p0.y,
                };
                0.5 * (cross(arc.center, dp) + arc.r * arc.r * arc.dtheta)
            }
        }
    }

    fn sub_green(&self, t0: f64, t1: f64) -> f64 {
        let q0 = self.point_at(t0);
        let q1 = self.point_at(t1);
        match self {
            PathSeg::Line { .. } => 0.5 * cross(q0, q1),
            PathSeg::Arc { arc, .. } => {
                let dtheta_sub = (t1 - t0) * arc.dtheta;
                let dp = Point {
                    x: q1.x - q0.x,
                    y: q1.y - q0.y,
                };
                0.5 * (cross(arc.center, dp) + arc.r * arc.r * dtheta_sub)
            }
        }
    }

    fn t_at_x(&self, val: f64) -> Vec<f64> {
        match self {
            PathSeg::Line { p0, p1 } => {
                let dx = p1.x - p0.x;
                if dx.abs() < PATH_EPS {
                    return vec![];
                }
                let t = (val - p0.x) / dx;
                if t > PATH_EPS && t < 1.0 - PATH_EPS {
                    vec![t]
                } else {
                    vec![]
                }
            }
            PathSeg::Arc { arc, .. } => {
                let cos_val = (val - arc.center.x) / arc.r;
                if cos_val.abs() > 1.0 + PATH_EPS {
                    return vec![];
                }
                let cos_val = cos_val.clamp(-1.0, 1.0);
                let base = cos_val.acos();
                let mut ts = vec![];
                for &angle in &[base, -base] {
                    let t = arc_t_of_angle(arc, angle);
                    if t > PATH_EPS && t < 1.0 - PATH_EPS {
                        ts.push(t);
                    }
                }
                ts
            }
        }
    }

    fn t_at_y(&self, val: f64) -> Vec<f64> {
        match self {
            PathSeg::Line { p0, p1 } => {
                let dy = p1.y - p0.y;
                if dy.abs() < PATH_EPS {
                    return vec![];
                }
                let t = (val - p0.y) / dy;
                if t > PATH_EPS && t < 1.0 - PATH_EPS {
                    vec![t]
                } else {
                    vec![]
                }
            }
            PathSeg::Arc { arc, .. } => {
                let sin_val = (val - arc.center.y) / arc.r;
                if sin_val.abs() > 1.0 + PATH_EPS {
                    return vec![];
                }
                let sin_val = sin_val.clamp(-1.0, 1.0);
                let base = sin_val.asin();
                let other = PI - base;
                let mut ts = vec![];
                for &angle in &[base, other] {
                    let t = arc_t_of_angle(arc, angle);
                    if t > PATH_EPS && t < 1.0 - PATH_EPS {
                        ts.push(t);
                    }
                }
                ts
            }
        }
    }

    fn winding_contribution(&self, px: f64, py: f64) -> i32 {
        match self {
            PathSeg::Line { p0, p1 } => {
                let dy = p1.y - p0.y;
                if dy.abs() < PATH_EPS {
                    return 0;
                }
                let t = (py - p0.y) / dy;
                if !(0.0..1.0).contains(&t) {
                    return 0;
                }
                let cross_x = p0.x + t * (p1.x - p0.x);
                if cross_x <= px {
                    return 0;
                }
                if dy > 0.0 {
                    1
                } else {
                    -1
                }
            }
            PathSeg::Arc { arc, .. } => {
                let sin_val = (py - arc.center.y) / arc.r;
                if sin_val.abs() > 1.0 + PATH_EPS {
                    return 0;
                }
                let sin_val = sin_val.clamp(-1.0, 1.0);
                let base = sin_val.asin();
                let other = PI - base;
                let mut contrib = 0i32;
                for &angle in &[base, other] {
                    let t = arc_t_of_angle(arc, angle);
                    if !(0.0..1.0).contains(&t) {
                        continue;
                    }
                    let cross_x = arc.center.x + arc.r * (arc.theta0 + t * arc.dtheta).cos();
                    if cross_x <= px {
                        continue;
                    }
                    let theta = arc.theta0 + t * arc.dtheta;
                    let dy_dt = arc.r * arc.dtheta * theta.cos();
                    if dy_dt.abs() < PATH_EPS {
                        continue;
                    }
                    if dy_dt > 0.0 {
                        contrib += 1;
                    } else {
                        contrib -= 1;
                    }
                }
                contrib
            }
        }
    }

    fn contains_point_on_boundary(&self, p: Point) -> bool {
        match self {
            PathSeg::Line { p0, p1 } => {
                let v = Point {
                    x: p1.x - p0.x,
                    y: p1.y - p0.y,
                };
                let w = Point {
                    x: p.x - p0.x,
                    y: p.y - p0.y,
                };
                if cross(v, w).abs() > PATH_EPS {
                    return false;
                }
                let dot = w.x * v.x + w.y * v.y;
                let len2 = v.x * v.x + v.y * v.y;
                dot >= -PATH_EPS && dot <= len2 + PATH_EPS
            }
            PathSeg::Arc { arc, .. } => {
                let dx = p.x - arc.center.x;
                let dy = p.y - arc.center.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if (dist - arc.r).abs() > PATH_EPS {
                    return false;
                }
                let t = arc_t_of_angle(arc, dy.atan2(dx));
                (-PATH_EPS..=1.0 + PATH_EPS).contains(&t)
            }
        }
    }
}

fn arc_t_of_angle(arc: &ArcData, angle: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let normalized = if arc.dtheta > 0.0 {
        (angle - arc.theta0).rem_euclid(two_pi)
    } else {
        let r = (angle - arc.theta0).rem_euclid(two_pi);
        if r > PATH_EPS {
            r - two_pi
        } else {
            r
        }
    };
    normalized / arc.dtheta
}

fn contour_signed_area(segs: &[PathSeg]) -> f64 {
    segs.iter().map(|s| s.green_integral()).sum()
}

fn reverse_seg(seg: &PathSeg) -> PathSeg {
    match seg {
        PathSeg::Line { p0, p1 } => PathSeg::Line { p0: *p1, p1: *p0 },
        PathSeg::Arc { arc, p0, p1 } => PathSeg::Arc {
            arc: ArcData {
                center: arc.center,
                r: arc.r,
                theta0: arc.theta0 + arc.dtheta,
                dtheta: -arc.dtheta,
            },
            p0: *p1,
            p1: *p0,
        },
    }
}

fn parse_one_contour(kinds: &[i8], data: &[[f64; 6]]) -> Result<PathContour, String> {
    if kinds.is_empty() || kinds[0] != SEG_MOVE {
        return Err("contour must start with move".to_string());
    }
    if *kinds.last().unwrap() != SEG_CLOSE {
        return Err("contour must end with close".to_string());
    }
    let move_x = data[0][0];
    let move_y = data[0][1];
    if !move_x.is_finite() || !move_y.is_finite() {
        return Err("move point must be finite".to_string());
    }

    let mut segs: Vec<PathSeg> = Vec::with_capacity(kinds.len());
    let mut current = Point {
        x: move_x,
        y: move_y,
    };

    for i in 1..kinds.len() {
        let k = kinds[i];
        let d = &data[i];
        match k {
            SEG_LINE => {
                let p1 = Point { x: d[0], y: d[1] };
                if !p1.x.is_finite() || !p1.y.is_finite() {
                    return Err("line endpoint must be finite".to_string());
                }
                let ddx = p1.x - current.x;
                let ddy = p1.y - current.y;
                if ddx * ddx + ddy * ddy < PATH_EPS * PATH_EPS {
                    return Err("zero-length line segment".to_string());
                }
                segs.push(PathSeg::Line { p0: current, p1 });
                current = p1;
            }
            SEG_ARC => {
                let (cx, cy, r, theta0, dtheta) = (d[0], d[1], d[2], d[3], d[4]);
                if !cx.is_finite()
                    || !cy.is_finite()
                    || !r.is_finite()
                    || !theta0.is_finite()
                    || !dtheta.is_finite()
                {
                    return Err("arc parameters must be finite".to_string());
                }
                if r <= 0.0 {
                    return Err("arc radius must be positive".to_string());
                }
                if dtheta == 0.0 || dtheta.abs() >= 2.0 * PI {
                    return Err("arc |dtheta| must be in (0, 2π)".to_string());
                }
                let center = Point { x: cx, y: cy };
                let p0_arc = Point {
                    x: cx + r * theta0.cos(),
                    y: cy + r * theta0.sin(),
                };
                let p1_arc = Point {
                    x: cx + r * (theta0 + dtheta).cos(),
                    y: cy + r * (theta0 + dtheta).sin(),
                };
                let ddx = p0_arc.x - current.x;
                let ddy = p0_arc.y - current.y;
                if ddx * ddx + ddy * ddy > 1e-6 {
                    return Err("arc start does not connect to current point".to_string());
                }
                segs.push(PathSeg::Arc {
                    arc: ArcData {
                        center,
                        r,
                        theta0,
                        dtheta,
                    },
                    p0: p0_arc,
                    p1: p1_arc,
                });
                current = p1_arc;
            }
            SEG_CLOSE => {
                let ddx = move_x - current.x;
                let ddy = move_y - current.y;
                if ddx * ddx + ddy * ddy > PATH_EPS * PATH_EPS {
                    segs.push(PathSeg::Line {
                        p0: current,
                        p1: Point {
                            x: move_x,
                            y: move_y,
                        },
                    });
                }
                break;
            }
            SEG_MOVE => {
                return Err("multiple move commands in one contour".to_string());
            }
            _ => {
                return Err(format!("unknown segment kind: {k}"));
            }
        }
    }

    if segs.len() < 3 {
        return Err("contour must have at least 3 edges".to_string());
    }
    let signed_area = contour_signed_area(&segs);
    Ok(PathContour { segs, signed_area })
}

fn segs_properly_intersect(s1: &PathSeg, s2: &PathSeg) -> bool {
    match (s1, s2) {
        (PathSeg::Line { p0: a0, p1: a1 }, PathSeg::Line { p0: b0, p1: b1 }) => {
            line_line_intersect(*a0, *a1, *b0, *b1)
        }
        (PathSeg::Line { p0, p1 }, PathSeg::Arc { arc, .. }) => line_arc_intersect(*p0, *p1, arc),
        (PathSeg::Arc { arc, .. }, PathSeg::Line { p0, p1 }) => line_arc_intersect(*p0, *p1, arc),
        (PathSeg::Arc { arc: arc1, .. }, PathSeg::Arc { arc: arc2, .. }) => {
            arc_arc_intersect(arc1, arc2)
        }
    }
}

fn line_line_intersect(a0: Point, a1: Point, b0: Point, b1: Point) -> bool {
    let da = Point {
        x: a1.x - a0.x,
        y: a1.y - a0.y,
    };
    let db = Point {
        x: b1.x - b0.x,
        y: b1.y - b0.y,
    };
    let denom = cross(da, db);
    if denom.abs() < PATH_EPS {
        return false;
    }
    let diff = Point {
        x: b0.x - a0.x,
        y: b0.y - a0.y,
    };
    let t = cross(diff, db) / denom;
    let s = cross(diff, da) / denom;
    t > PATH_EPS && t < 1.0 - PATH_EPS && s > PATH_EPS && s < 1.0 - PATH_EPS
}

fn line_arc_intersect(p0: Point, p1: Point, arc: &ArcData) -> bool {
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let fx = p0.x - arc.center.x;
    let fy = p0.y - arc.center.y;
    let a = dx * dx + dy * dy;
    if a < PATH_EPS * PATH_EPS {
        return false;
    }
    let b = 2.0 * (fx * dx + fy * dy);
    let c = fx * fx + fy * fy - arc.r * arc.r;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return false;
    }
    let sqrt_disc = disc.sqrt();
    for &t in &[(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] {
        if t > PATH_EPS && t < 1.0 - PATH_EPS {
            let pt = Point {
                x: p0.x + t * dx,
                y: p0.y + t * dy,
            };
            let angle = (pt.y - arc.center.y).atan2(pt.x - arc.center.x);
            let s = arc_t_of_angle(arc, angle);
            if s > PATH_EPS && s < 1.0 - PATH_EPS {
                return true;
            }
        }
    }
    false
}

fn arc_arc_intersect(arc1: &ArcData, arc2: &ArcData) -> bool {
    let dx = arc2.center.x - arc1.center.x;
    let dy = arc2.center.y - arc1.center.y;
    let d2 = dx * dx + dy * dy;
    let d = d2.sqrt();
    if d < PATH_EPS {
        return false;
    }
    let sum_r = arc1.r + arc2.r;
    let diff_r = (arc1.r - arc2.r).abs();
    if d > sum_r + PATH_EPS || d < diff_r - PATH_EPS {
        return false;
    }
    let a = (arc1.r * arc1.r - arc2.r * arc2.r + d2) / (2.0 * d);
    let h2 = arc1.r * arc1.r - a * a;
    if h2 < 0.0 {
        return false;
    }
    let h = h2.sqrt();
    let mid = Point {
        x: arc1.center.x + a * dx / d,
        y: arc1.center.y + a * dy / d,
    };
    let perp = Point {
        x: h * dy / d,
        y: -h * dx / d,
    };
    for &pt in &[
        Point {
            x: mid.x + perp.x,
            y: mid.y + perp.y,
        },
        Point {
            x: mid.x - perp.x,
            y: mid.y - perp.y,
        },
    ] {
        let angle1 = (pt.y - arc1.center.y).atan2(pt.x - arc1.center.x);
        let angle2 = (pt.y - arc2.center.y).atan2(pt.x - arc2.center.x);
        let t1 = arc_t_of_angle(arc1, angle1);
        let t2 = arc_t_of_angle(arc2, angle2);
        if t1 > PATH_EPS && t1 < 1.0 - PATH_EPS && t2 > PATH_EPS && t2 < 1.0 - PATH_EPS {
            return true;
        }
    }
    false
}

fn check_self_intersections(contour: &PathContour) -> Result<(), String> {
    let n = contour.segs.len();
    for i in 0..n {
        for j in (i + 2)..n {
            if i == 0 && j == n - 1 {
                continue; // first and last share the close point
            }
            if segs_properly_intersect(&contour.segs[i], &contour.segs[j]) {
                return Err("self-intersecting path contour".to_string());
            }
        }
    }
    Ok(())
}

pub(crate) fn build_validated_path(
    kinds: &[i8],
    data: &[[f64; 6]],
) -> Result<ValidatedPath, String> {
    if kinds.is_empty() {
        return Err("path must have at least one segment".to_string());
    }
    if kinds.len() != data.len() {
        return Err("kinds and data must have the same length".to_string());
    }

    let mut contour_starts: Vec<usize> = vec![];
    for (i, &k) in kinds.iter().enumerate() {
        if k == SEG_MOVE {
            contour_starts.push(i);
        }
    }
    if contour_starts.is_empty() {
        return Err("path must start with a move command".to_string());
    }

    let n = kinds.len();
    let mut contours: Vec<PathContour> = Vec::with_capacity(contour_starts.len());
    for (ci, &start) in contour_starts.iter().enumerate() {
        let end = if ci + 1 < contour_starts.len() {
            contour_starts[ci + 1]
        } else {
            n
        };
        let contour = parse_one_contour(&kinds[start..end], &data[start..end])?;
        contours.push(contour);
    }

    // Normalize orientation: outer CCW, holes CW
    for (i, contour) in contours.iter_mut().enumerate() {
        let want_positive = i == 0;
        if want_positive && contour.signed_area < 0.0 || !want_positive && contour.signed_area > 0.0
        {
            contour.segs.reverse();
            for seg in &mut contour.segs {
                *seg = reverse_seg(seg);
            }
            contour.signed_area = -contour.signed_area;
        }
        check_self_intersections(contour)?;
    }

    if contours[0].signed_area.abs() < PATH_EPS {
        return Err("outer contour has zero area".to_string());
    }

    // Hole containment and cross-contour intersection checks
    for hi in 1..contours.len() {
        // Every hole segment start must be inside the outer contour
        let outer_segs = &contours[0].segs;
        for seg in &contours[hi].segs {
            let p = seg.point_at(0.5);
            let w: i32 = outer_segs
                .iter()
                .map(|s| s.winding_contribution(p.x, p.y))
                .sum();
            if w == 0 {
                return Err("hole is not contained within the outer contour".to_string());
            }
        }
        // Outer and hole segments must not cross
        let outer_segs = &contours[0].segs;
        for sa in outer_segs {
            for sb in &contours[hi].segs {
                if segs_properly_intersect(sa, sb) {
                    return Err("outer contour and hole contour intersect".to_string());
                }
            }
        }
    }

    Ok(ValidatedPath { contours })
}

pub(crate) fn path_contains_point(path: &ValidatedPath, px: f64, py: f64) -> bool {
    if path.contours.is_empty() {
        return false;
    }
    let p = Point { x: px, y: py };
    if path.contours[0]
        .segs
        .iter()
        .any(|seg| seg.contains_point_on_boundary(p))
    {
        return true;
    }
    let winding_y = py + PATH_EPS;
    let outer_winding: i32 = path.contours[0]
        .segs
        .iter()
        .map(|s| s.winding_contribution(px, winding_y))
        .sum();
    if outer_winding == 0 {
        return false;
    }
    for hole in path.contours.iter().skip(1) {
        if hole
            .segs
            .iter()
            .any(|seg| seg.contains_point_on_boundary(p))
        {
            return false;
        }
        let hole_winding: i32 = hole
            .segs
            .iter()
            .map(|s| s.winding_contribution(px, winding_y))
            .sum();
        if hole_winding != 0 {
            return false;
        }
    }
    true
}

fn update_bbox_for_seg(
    seg: &PathSeg,
    min_x: &mut f64,
    max_x: &mut f64,
    min_y: &mut f64,
    max_y: &mut f64,
) {
    match seg {
        PathSeg::Line { p0, p1 } => {
            *min_x = min_x.min(p0.x).min(p1.x);
            *max_x = max_x.max(p0.x).max(p1.x);
            *min_y = min_y.min(p0.y).min(p1.y);
            *max_y = max_y.max(p0.y).max(p1.y);
        }
        PathSeg::Arc { arc, p0, p1 } => {
            *min_x = min_x.min(p0.x).min(p1.x);
            *max_x = max_x.max(p0.x).max(p1.x);
            *min_y = min_y.min(p0.y).min(p1.y);
            *max_y = max_y.max(p0.y).max(p1.y);
            for &angle in &[0.0_f64, PI / 2.0, PI, 3.0 * PI / 2.0] {
                let t = arc_t_of_angle(arc, angle);
                if t > 0.0 && t < 1.0 {
                    let ep = Point {
                        x: arc.center.x + arc.r * angle.cos(),
                        y: arc.center.y + arc.r * angle.sin(),
                    };
                    *min_x = min_x.min(ep.x);
                    *max_x = max_x.max(ep.x);
                    *min_y = min_y.min(ep.y);
                    *max_y = max_y.max(ep.y);
                }
            }
        }
    }
}

pub(crate) fn path_bbox(path: &ValidatedPath, x0: f64, y0: f64) -> (isize, isize, isize, isize) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for seg in &path.contours[0].segs {
        update_bbox_for_seg(seg, &mut min_x, &mut max_x, &mut min_y, &mut max_y);
    }
    let ixmin = (x0 + min_x - 0.5).floor() as isize;
    let ixmax = (x0 + max_x + 0.5).ceil() as isize;
    let iymin = (y0 + min_y - 0.5).floor() as isize;
    let iymax = (y0 + max_y + 0.5).ceil() as isize;
    (ixmin, ixmax, iymin, iymax)
}

pub(crate) fn circular_pill_path(w: f64, r: f64, theta: f64) -> ValidatedPath {
    let half_w = 0.5 * w;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let rotate = |x: f64, y: f64| Point {
        x: cos_t * x - sin_t * y,
        y: sin_t * x + cos_t * y,
    };

    let left_center = rotate(-half_w, 0.0);
    let right_center = rotate(half_w, 0.0);
    let lower_left = rotate(-half_w, -r);
    let lower_right = rotate(half_w, -r);
    let upper_left = rotate(-half_w, r);
    let upper_right = rotate(half_w, r);
    let segs = vec![
        PathSeg::Line {
            p0: lower_left,
            p1: lower_right,
        },
        PathSeg::Arc {
            arc: ArcData {
                center: right_center,
                r,
                theta0: theta - 0.5 * PI,
                dtheta: PI,
            },
            p0: lower_right,
            p1: upper_right,
        },
        PathSeg::Line {
            p0: upper_right,
            p1: upper_left,
        },
        PathSeg::Arc {
            arc: ArcData {
                center: left_center,
                r,
                theta0: theta + 0.5 * PI,
                dtheta: PI,
            },
            p0: upper_left,
            p1: lower_left,
        },
    ];
    let signed_area = contour_signed_area(&segs);
    ValidatedPath {
        contours: vec![PathContour { segs, signed_area }],
    }
}

fn contour_is_polygon(contour: &PathContour) -> bool {
    contour
        .segs
        .iter()
        .all(|seg| matches!(seg, PathSeg::Line { .. }))
}

fn contour_vertices(contour: &PathContour) -> Vec<Point> {
    contour.segs.iter().map(PathSeg::p0).collect()
}

fn polygon_path_pixel_area(path: &ValidatedPath, lx: f64, rx: f64, by: f64, ty: f64) -> f64 {
    let mut area = 0.0_f64;
    if let Some(outer) = path.contours.first() {
        area += polygon_rect_overlap_area(&contour_vertices(outer), lx, rx, by, ty);
    }
    for hole in path.contours.iter().skip(1) {
        area -= polygon_rect_overlap_area(&contour_vertices(hole), lx, rx, by, ty);
    }
    area.clamp(0.0, 1.0)
}

fn seg_pixel_contribution(seg: &PathSeg, lx: f64, rx: f64, by: f64, ty: f64) -> f64 {
    let mut ts: Vec<f64> = vec![0.0, 1.0];
    for t in seg.t_at_x(lx) {
        ts.push(t);
    }
    for t in seg.t_at_x(rx) {
        ts.push(t);
    }
    for t in seg.t_at_y(by) {
        ts.push(t);
    }
    for t in seg.t_at_y(ty) {
        ts.push(t);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts.dedup_by(|a, b| (*a - *b).abs() < PATH_EPS);

    let mut contribution = 0.0;
    for i in 0..ts.len().saturating_sub(1) {
        let t_mid = 0.5 * (ts[i] + ts[i + 1]);
        let mid = seg.point_at(t_mid);
        if mid.x > lx - PATH_EPS
            && mid.x < rx + PATH_EPS
            && mid.y > by - PATH_EPS
            && mid.y < ty + PATH_EPS
        {
            contribution += seg.sub_green(ts[i], ts[i + 1]);
        }
    }
    contribution
}

fn pixel_edge_contribution(
    path: &ValidatedPath,
    x0e: f64,
    y0e: f64,
    x1e: f64,
    y1e: f64,
    is_horiz: bool,
) -> f64 {
    let edge_val = if is_horiz { y0e } else { x0e };
    let mut ts: Vec<f64> = vec![0.0, 1.0];
    for contour in &path.contours {
        for seg in &contour.segs {
            let seg_ts = if is_horiz {
                seg.t_at_y(edge_val)
            } else {
                seg.t_at_x(edge_val)
            };
            for seg_t in seg_ts {
                let pt = seg.point_at(seg_t);
                let edge_t = if is_horiz {
                    let dx = x1e - x0e;
                    if dx.abs() < PATH_EPS {
                        continue;
                    }
                    (pt.x - x0e) / dx
                } else {
                    let dy = y1e - y0e;
                    if dy.abs() < PATH_EPS {
                        continue;
                    }
                    (pt.y - y0e) / dy
                };
                if edge_t > PATH_EPS && edge_t < 1.0 - PATH_EPS {
                    ts.push(edge_t);
                }
            }
        }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts.dedup_by(|a, b| (*a - *b).abs() < PATH_EPS);

    let mut contribution = 0.0;
    for i in 0..ts.len().saturating_sub(1) {
        let t_mid = 0.5 * (ts[i] + ts[i + 1]);
        let mid = Point {
            x: x0e + t_mid * (x1e - x0e),
            y: y0e + t_mid * (y1e - y0e),
        };
        if path_contains_point(path, mid.x, mid.y) {
            let p0e = Point {
                x: x0e + ts[i] * (x1e - x0e),
                y: y0e + ts[i] * (y1e - y0e),
            };
            let p1e = Point {
                x: x0e + ts[i + 1] * (x1e - x0e),
                y: y0e + ts[i + 1] * (y1e - y0e),
            };
            contribution += 0.5 * cross(p0e, p1e);
        }
    }
    contribution
}

pub(crate) fn path_pixel_area(path: &ValidatedPath, x0: f64, y0: f64, ix: isize, iy: isize) -> f64 {
    let local_px = (ix as f64) - x0;
    let local_py = (iy as f64) - y0;
    let lx = local_px - 0.5;
    let rx = local_px + 0.5;
    let by = local_py - 0.5;
    let ty = local_py + 0.5;

    if path.contours.iter().all(contour_is_polygon) {
        return polygon_path_pixel_area(path, lx, rx, by, ty);
    }

    let mut area = 0.0_f64;
    for contour in &path.contours {
        for seg in &contour.segs {
            area += seg_pixel_contribution(seg, lx, rx, by, ty);
        }
    }
    // Pixel edges: bottom, right, top, left (CCW order)
    area += pixel_edge_contribution(path, lx, by, rx, by, true);
    area += pixel_edge_contribution(path, rx, by, rx, ty, false);
    area += pixel_edge_contribution(path, rx, ty, lx, ty, true);
    area += pixel_edge_contribution(path, lx, ty, lx, by, false);
    area.clamp(0.0, 1.0)
}

pub(crate) fn path_center_weight(
    path: &ValidatedPath,
    x0: f64,
    y0: f64,
    ix: isize,
    iy: isize,
) -> f64 {
    let local_px = (ix as f64) - x0;
    let local_py = (iy as f64) - y0;
    if path_contains_point(path, local_px, local_py) {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wedge_contains_uses_straight_sides_between_arc_endpoints() {
        let wedge = Wedge {
            x0: 0.0,
            y0: 0.0,
            r_in: 2.0,
            r_out: 6.0,
            theta_in: 0.0,
            dtheta_in: PI / 2.0,
            theta_out: 0.0,
            dtheta_out: PI,
        };

        assert!(wedge_contains_point(wedge, 3.0, 0.0));
        assert!(wedge_contains_point(wedge, 4.0, 2.0));
        assert!(!wedge_contains_point(wedge, -1.0, 3.0));
        assert!(!wedge_contains_point(wedge, 7.0, 0.0));
    }

    #[test]
    fn wedge_center_weight_uses_center_sampling() {
        let wedge = Wedge {
            x0: 0.0,
            y0: 0.0,
            r_in: 0.5,
            r_out: 2.5,
            theta_in: 0.0,
            dtheta_in: PI / 2.0,
            theta_out: 0.0,
            dtheta_out: PI / 2.0,
        };

        assert_eq!(wedge_center_weight_at_pixel(wedge, 2, 1), 1.0);
        assert_eq!(wedge_center_weight_at_pixel(wedge, 0, 0), 0.0);
    }

    #[test]
    fn wedge_exact_constant_sector_tracks_area_on_grid() {
        let wedge = Wedge {
            x0: 0.0,
            y0: 0.0,
            r_in: 0.5,
            r_out: 4.0,
            theta_in: 0.0,
            dtheta_in: PI / 3.0,
            theta_out: 0.0,
            dtheta_out: PI / 3.0,
        };
        let mut area = 0.0;
        for py in -5..=5 {
            for px in -5..=5 {
                area += wedge_exact_weight_at_pixel(wedge, px, py);
            }
        }
        let expected =
            0.5 * wedge.dtheta_in * (wedge.r_out * wedge.r_out - wedge.r_in * wedge.r_in);

        assert!(
            (area - expected).abs() < 5.0e-2,
            "area={area} expected={expected}"
        );
    }

    #[test]
    fn wedge_exact_generalized_straight_sided_shape_tracks_area_on_grid() {
        let wedge = Wedge {
            x0: 0.0,
            y0: 0.0,
            r_in: 1.3,
            r_out: 5.0,
            theta_in: 0.0,
            dtheta_in: PI / 4.0,
            theta_out: 0.4,
            dtheta_out: PI / 2.0,
        };
        let mut area = 0.0;
        for py in -6..=6 {
            for px in -6..=6 {
                area += wedge_exact_weight_at_pixel(wedge, px, py);
            }
        }
        let parts = wedge_parts(wedge);
        let expected = polygon_area(&parts.quad)
            + 0.5 * wedge.r_out * wedge.r_out * (wedge.dtheta_out - wedge.dtheta_out.sin())
            - 0.5 * wedge.r_in * wedge.r_in * (wedge.dtheta_in - wedge.dtheta_in.sin());

        assert!(
            (area - expected).abs() < 5.0e-2,
            "area={area} expected={expected}"
        );
    }
}

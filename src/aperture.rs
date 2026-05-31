//! Public Rust aperture helpers.
//!
//! These functions are the small, stable Rust API over the same exact
//! pixel-overlap kernels used by the Python extension. They intentionally do
//! not perform Python-style validation; callers should pass finite coordinates
//! and positive aperture sizes.

use std::ops::Range;

use crate::geometry::{
    circ_bbox as raw_circle_bbox, circ_pixel_area, ellip_bbox as raw_ellipse_bbox,
    ellip_pixel_area, extent_bbox, rect_pixel_area, Ellip, SQRT_HALF,
};

/// Bbox-tight integer pixel extent.
///
/// Bounds are half-open: x pixels are in `ixmin..ixmax`, and y pixels are in
/// `iymin..iymax`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundingBox {
    pub ixmin: isize,
    pub ixmax: isize,
    pub iymin: isize,
    pub iymax: isize,
}

impl BoundingBox {
    /// Pixel x range covered by this bounding box.
    #[inline]
    pub fn x_range(self) -> Range<isize> {
        self.ixmin..self.ixmax
    }

    /// Pixel y range covered by this bounding box.
    #[inline]
    pub fn y_range(self) -> Range<isize> {
        self.iymin..self.iymax
    }

    /// Shape as `(ny, nx)`.
    #[inline]
    pub fn shape(self) -> (usize, usize) {
        (
            extent_len(self.iymin, self.iymax),
            extent_len(self.ixmin, self.ixmax),
        )
    }
}

impl From<(isize, isize, isize, isize)> for BoundingBox {
    #[inline]
    fn from((ixmin, ixmax, iymin, iymax): (isize, isize, isize, isize)) -> Self {
        Self {
            ixmin,
            ixmax,
            iymin,
            iymax,
        }
    }
}

#[inline]
fn extent_len(start: isize, end: isize) -> usize {
    end.checked_sub(start)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0)
}

/// Ellipse in pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ellipse {
    pub x0: f64,
    pub y0: f64,
    pub a: f64,
    pub b: f64,
    /// Rotation angle in radians.
    pub theta: f64,
}

impl Ellipse {
    #[inline]
    fn raw(self) -> Ellip {
        Ellip {
            x0: self.x0,
            y0: self.y0,
            a: self.a,
            b: self.b,
            cos_t: self.theta.cos(),
            sin_t: self.theta.sin(),
        }
    }
}

/// Rotated rectangle in pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rectangle {
    pub x0: f64,
    pub y0: f64,
    pub width: f64,
    pub height: f64,
    /// Rotation angle in radians.
    pub theta: f64,
}

/// Return the bbox-tight pixel extent for a circle.
#[inline]
pub fn circle_bbox(x: f64, y: f64, radius: f64) -> BoundingBox {
    raw_circle_bbox(x, y, radius).into()
}

/// Return the exact fractional overlap between a circle and one pixel.
#[inline]
pub fn circle_exact_weight_at_pixel(x: f64, y: f64, radius: f64, px: isize, py: isize) -> f64 {
    let r_inner = (radius - SQRT_HALF).max(0.0);
    let r_inner2 = r_inner * r_inner;
    let r_outer = radius + SQRT_HALF;
    let r_outer2 = r_outer * r_outer;
    let dx = px as f64 - x;
    let dy = py as f64 - y;
    let rpix2 = dx * dx + dy * dy;
    if rpix2 < r_inner2 {
        1.0
    } else if rpix2 > r_outer2 {
        0.0
    } else {
        circ_pixel_area(x, y, px, py, radius)
    }
}

/// Return the bbox-tight pixel extent for an ellipse.
#[inline]
pub fn ellipse_bbox(ellipse: Ellipse) -> BoundingBox {
    raw_ellipse_bbox(ellipse.raw()).into()
}

/// Return the exact fractional overlap between an ellipse and one pixel.
#[inline]
pub fn ellipse_exact_weight_at_pixel(ellipse: Ellipse, px: isize, py: isize) -> f64 {
    ellip_pixel_area(ellipse.raw(), px, py)
}

/// Return the bbox-tight pixel extent for a rectangle.
#[inline]
pub fn rectangle_bbox(rectangle: Rectangle) -> BoundingBox {
    let half_w = 0.5 * rectangle.width;
    let half_h = 0.5 * rectangle.height;
    let cos_t = rectangle.theta.cos();
    let sin_t = rectangle.theta.sin();
    let dx = (half_w * cos_t).abs() + (half_h * sin_t).abs();
    let dy = (half_w * sin_t).abs() + (half_h * cos_t).abs();
    extent_bbox(rectangle.x0, rectangle.y0, dx, dy).into()
}

/// Return the exact fractional overlap between a rectangle and one pixel.
#[inline]
pub fn rectangle_exact_weight_at_pixel(rectangle: Rectangle, px: isize, py: isize) -> f64 {
    rect_pixel_area(
        rectangle.x0,
        rectangle.y0,
        px,
        py,
        0.5 * rectangle.width,
        0.5 * rectangle.height,
        rectangle.theta.cos(),
        rectangle.theta.sin(),
    )
}

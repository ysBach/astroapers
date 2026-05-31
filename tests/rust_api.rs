use astroapers::{
    circle_bbox, circle_exact_weight_at_pixel, ellipse_bbox, ellipse_exact_weight_at_pixel,
    rectangle_bbox, rectangle_exact_weight_at_pixel, BoundingBox, Ellipse, Rectangle,
};

#[test]
fn circle_api_is_usable_without_python_features() {
    let weight = circle_exact_weight_at_pixel(0.0, 0.0, 0.5, 0, 0);

    assert!((weight - std::f64::consts::FRAC_PI_4).abs() <= 1.0e-14);
}

#[test]
fn bbox_ranges_cover_expected_pixels() {
    let bbox = circle_bbox(2.0, 3.0, 1.0);

    assert_eq!(bbox.x_range(), 1..4);
    assert_eq!(bbox.y_range(), 2..5);
    assert_eq!(bbox.shape(), (3, 3));
    assert_eq!(
        BoundingBox {
            ixmin: 2,
            ixmax: 1,
            iymin: 4,
            iymax: 3,
        }
        .shape(),
        (0, 0),
    );
}

#[test]
fn ellipse_and_rectangle_apis_return_exact_pixel_weights() {
    let ellipse = Ellipse {
        x0: 0.0,
        y0: 0.0,
        a: 1.0,
        b: 0.5,
        theta: 0.0,
    };
    let rectangle = Rectangle {
        x0: 0.0,
        y0: 0.0,
        width: 1.0,
        height: 1.0,
        theta: 0.0,
    };

    assert_eq!(ellipse_bbox(ellipse).shape(), (1, 3));
    assert_eq!(rectangle_bbox(rectangle).shape(), (1, 1));
    assert!(ellipse_exact_weight_at_pixel(ellipse, 0, 0) > 0.0);
    assert_eq!(rectangle_exact_weight_at_pixel(rectangle, 0, 0), 1.0);
}

from __future__ import annotations

import astroapers as apers
import numpy as np
import pytest
from numpy.testing import assert_allclose

POSITIONS = np.array(
    [
        (23.4, 24.1),
        (52.6, 40.3),
        (78.2, 71.7),
    ],
    dtype=np.float64,
)
CIRCLE_RADIUS = 5.0
ELLIPSE_A = 6.0
ELLIPSE_B = 3.0
ELLIPSE_THETA = 0.35
RECTANGLE_W = 9.0
RECTANGLE_H = 5.0
RECTANGLE_THETA = 0.4
CIRC_ANN_R_IN = 6.0
CIRC_ANN_R_OUT = 9.0
ELLIP_ANN_A = 5.0
ELLIP_ANN_B = 2.5
ELLIP_ANN_R_IN = 1.3
ELLIP_ANN_R_OUT = 2.1
ELLIP_ANN_THETA = -0.25
PHOTUTILS_RECTANGLE_SUBPIXELS = 128
SEP_EXACT_RTOL = 1.0e-8
SEP_EXACT_ATOL = 5.0e-4


def _image_cases() -> list[tuple[str, np.ndarray]]:
    shape = (96, 100)
    rng = np.random.default_rng(20250501)
    yy, xx = np.indices(shape, dtype=np.float64)
    noisy = rng.normal(loc=100.0, scale=0.25, size=shape)
    for amplitude, sigma, (x0, y0) in zip(
        (900.0, 650.0, 1100.0),
        (1.6, 2.1, 1.3),
        POSITIONS,
        strict=True,
    ):
        radius2 = ((xx - x0) / sigma) ** 2 + ((yy - y0) / sigma) ** 2
        noisy += amplitude * np.exp(-0.5 * radius2)
    return [
        ("zeros", np.zeros(shape, dtype=np.float64)),
        ("ones", np.ones(shape, dtype=np.float64)),
        ("noisy_stars", noisy.astype(np.float64)),
    ]


@pytest.mark.parametrize(("image_name", "image"), _image_cases())
def test_circle_apsum_exact_matches_photutils_and_sep(image_name, image):
    photutils_aperture = pytest.importorskip("photutils.aperture")
    sep = pytest.importorskip("sep")
    x = POSITIONS[:, 0]
    y = POSITIONS[:, 1]

    aap_apsum, aap_npix = apers.apsum_circ_exact(image, x, y, CIRCLE_RADIUS)
    object_apsum, object_npix = apers.CircAp(POSITIONS, CIRCLE_RADIUS).apsum_exact(
        image
    )
    photutils_apsum, _ = photutils_aperture.CircularAperture(
        POSITIONS, r=CIRCLE_RADIUS
    ).do_photometry(image, method="exact")
    sep_apsum, _, sep_flag = sep.sum_circle(image, x, y, CIRCLE_RADIUS, subpix=0)

    assert np.all(sep_flag == 0), image_name
    assert_allclose(object_apsum, aap_apsum, rtol=0, atol=1e-12)
    assert_allclose(object_npix, aap_npix, rtol=0, atol=1e-12)
    assert_allclose(photutils_apsum, aap_apsum, rtol=5e-10, atol=1e-7)
    assert_allclose(sep_apsum, aap_apsum, rtol=SEP_EXACT_RTOL, atol=SEP_EXACT_ATOL)
    if image_name == "ones":
        assert_allclose(aap_apsum, aap_npix, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("image_name", "image"), _image_cases())
def test_ellipse_apsum_exact_matches_photutils_and_sep(image_name, image):
    photutils_aperture = pytest.importorskip("photutils.aperture")
    sep = pytest.importorskip("sep")
    x = POSITIONS[:, 0]
    y = POSITIONS[:, 1]

    aap_apsum, aap_npix = apers.apsum_ellip_exact(
        image, x, y, ELLIPSE_A, ELLIPSE_B, ELLIPSE_THETA
    )
    object_apsum, object_npix = apers.EllipAp(
        POSITIONS, ELLIPSE_A, ELLIPSE_B, ELLIPSE_THETA
    ).apsum_exact(image)
    photutils_apsum, _ = photutils_aperture.EllipticalAperture(
        POSITIONS, a=ELLIPSE_A, b=ELLIPSE_B, theta=ELLIPSE_THETA
    ).do_photometry(image, method="exact")
    sep_apsum, _, sep_flag = sep.sum_ellipse(
        image, x, y, ELLIPSE_A, ELLIPSE_B, ELLIPSE_THETA, subpix=0
    )

    assert np.all(sep_flag == 0), image_name
    assert_allclose(object_apsum, aap_apsum, rtol=0, atol=1e-12)
    assert_allclose(object_npix, aap_npix, rtol=0, atol=1e-12)
    assert_allclose(photutils_apsum, aap_apsum, rtol=5e-10, atol=1e-7)
    assert_allclose(sep_apsum, aap_apsum, rtol=SEP_EXACT_RTOL, atol=SEP_EXACT_ATOL)
    if image_name == "ones":
        assert_allclose(aap_apsum, aap_npix, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("image_name", "image"), _image_cases())
def test_rectangle_apsum_exact_matches_photutils_subpixel_reference(image_name, image):
    photutils_aperture = pytest.importorskip("photutils.aperture")
    x = POSITIONS[:, 0]
    y = POSITIONS[:, 1]

    aap_apsum, aap_npix = apers.apsum_rect_exact(
        image, x, y, RECTANGLE_W, RECTANGLE_H, RECTANGLE_THETA
    )
    object_apsum, object_npix = apers.RectAp(
        POSITIONS, RECTANGLE_W, RECTANGLE_H, RECTANGLE_THETA
    ).apsum_exact(image)
    photutils_apsum, _ = photutils_aperture.RectangularAperture(
        POSITIONS,
        w=RECTANGLE_W,
        h=RECTANGLE_H,
        theta=RECTANGLE_THETA,
    ).do_photometry(
        image,
        method="subpixel",
        subpixels=PHOTUTILS_RECTANGLE_SUBPIXELS,
    )

    assert_allclose(object_apsum, aap_apsum, rtol=0, atol=1e-12)
    assert_allclose(object_npix, aap_npix, rtol=0, atol=1e-12)
    assert_allclose(photutils_apsum, aap_apsum, rtol=1e-4, atol=5e-3)
    if image_name == "ones":
        assert_allclose(aap_apsum, aap_npix, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("image_name", "image"), _image_cases())
def test_circular_annulus_apsum_exact_matches_photutils_and_sep(image_name, image):
    photutils_aperture = pytest.importorskip("photutils.aperture")
    sep = pytest.importorskip("sep")
    x = POSITIONS[:, 0]
    y = POSITIONS[:, 1]

    aap_apsum, aap_npix = apers.apsum_circ_ann_exact(
        image, x, y, CIRC_ANN_R_IN, CIRC_ANN_R_OUT
    )
    object_apsum, object_npix = apers.CircAn(
        POSITIONS, CIRC_ANN_R_IN, CIRC_ANN_R_OUT
    ).apsum_exact(image)
    photutils_apsum, _ = photutils_aperture.CircularAnnulus(
        POSITIONS, r_in=CIRC_ANN_R_IN, r_out=CIRC_ANN_R_OUT
    ).do_photometry(image, method="exact")
    sep_apsum, _, sep_flag = sep.sum_circann(
        image, x, y, CIRC_ANN_R_IN, CIRC_ANN_R_OUT, subpix=0
    )

    assert np.all(sep_flag == 0), image_name
    assert_allclose(object_apsum, aap_apsum, rtol=0, atol=1e-12)
    assert_allclose(object_npix, aap_npix, rtol=0, atol=1e-12)
    assert_allclose(photutils_apsum, aap_apsum, rtol=5e-10, atol=1e-7)
    assert_allclose(sep_apsum, aap_apsum, rtol=SEP_EXACT_RTOL, atol=SEP_EXACT_ATOL)
    if image_name == "ones":
        assert_allclose(aap_apsum, aap_npix, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("image_name", "image"), _image_cases())
def test_elliptical_annulus_apsum_exact_matches_photutils_and_sep(image_name, image):
    photutils_aperture = pytest.importorskip("photutils.aperture")
    sep = pytest.importorskip("sep")
    x = POSITIONS[:, 0]
    y = POSITIONS[:, 1]
    a_in = ELLIP_ANN_A * ELLIP_ANN_R_IN
    b_in = ELLIP_ANN_B * ELLIP_ANN_R_IN
    a_out = ELLIP_ANN_A * ELLIP_ANN_R_OUT
    b_out = ELLIP_ANN_B * ELLIP_ANN_R_OUT

    object_apsum, object_npix = apers.EllipAn(
        POSITIONS, a_in, b_in, a_out, b_out, theta_in=ELLIP_ANN_THETA
    ).apsum_exact(image)
    photutils_apsum, _ = photutils_aperture.EllipticalAnnulus(
        POSITIONS,
        a_in=a_in,
        a_out=a_out,
        b_in=b_in,
        b_out=b_out,
        theta=ELLIP_ANN_THETA,
    ).do_photometry(image, method="exact")
    sep_apsum, _, sep_flag = sep.sum_ellipann(
        image,
        x,
        y,
        ELLIP_ANN_A,
        ELLIP_ANN_B,
        ELLIP_ANN_THETA,
        ELLIP_ANN_R_IN,
        ELLIP_ANN_R_OUT,
        subpix=0,
    )

    assert np.all(sep_flag == 0), image_name
    assert_allclose(photutils_apsum, object_apsum, rtol=5e-10, atol=1e-7)
    assert_allclose(sep_apsum, object_apsum, rtol=SEP_EXACT_RTOL, atol=SEP_EXACT_ATOL)
    if image_name == "ones":
        assert_allclose(object_apsum, object_npix, rtol=0, atol=1e-12)

"""Benchmark astroapers against photutils and SEP.

The benchmark validates numerical agreement before reporting timings.  It uses
exact aperture modes where available and skips cases that do not exist.  The
``mask`` task materializes bbox-tight aperture weights for aperture-object
backends and validates them by comparing summed weights. The photutils.geometry
mask rows are low-level npix-style references, not reusable mask objects. Timing
samples are collected in global round-robin passes across benchmark cases rather
than exhausting one case before moving to the next.
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import astroapers as apers
import astroapers.kernels as aapk
import numpy as np
from numpy.testing import assert_allclose

AAP = "aap"
AAP_OPT = "aap_opt"

REQUIRED_ASTROAPERS_API = (
    "PillAp",
    "PillAn",
)
REQUIRED_KERNELS_API = (
    "npix_circ_exact",
    "npix_ellip_exact",
    "npix_rect_exact",
    "apsum_circ_exact",
    "apsum_ellip_ann_exact",
    "apsum_ellip_exact",
    "apsum_pill_ann_exact",
    "apsum_pill_exact",
    "apsum_rect_ann_exact",
    "apsum_rect_exact",
    "apsum_circ_center",
    "apsum_ellip_center",
    "apsum_rect_center",
)

missing_api = [name for name in REQUIRED_ASTROAPERS_API if not hasattr(apers, name)]
missing_kernels_api = [name for name in REQUIRED_KERNELS_API if not hasattr(aapk, name)]
if missing_api or missing_kernels_api:
    missing = ", ".join((*missing_api, *missing_kernels_api))
    raise SystemExit(
        f"astroapers native extension is stale or incomplete; missing {missing}. "
        "Rebuild it with: uv run --with maturin maturin develop --extras test"
    )

try:
    import sep
except ImportError:  # pragma: no cover - benchmark dependency check
    sep = None

try:
    from photutils.aperture import (
        CircularAperture,
        CircularAnnulus,
        EllipticalAperture,
        EllipticalAnnulus,
        RectangularAperture,
        RectangularAnnulus,
    )
    from photutils.geometry import circular_overlap_grid, elliptical_overlap_grid
except ImportError as exc:  # pragma: no cover - benchmark dependency check
    raise SystemExit("photutils is required for this benchmark") from exc


DEFAULT_COUNTS = (1, 100, 10_000)
DEFAULT_ATOL = 1.0e-5
DEFAULT_RTOL = 1.0e-4
DEFAULT_PHOTUTILS_RECTANGLE_SUBPIXELS = 32
DEFAULT_PHOTUTILS_RECTANGLE_ATOL = DEFAULT_ATOL
DEFAULT_PHOTUTILS_RECTANGLE_RTOL = 2.0e-4
SHAPE_COLUMN_WIDTH = 18
DEFAULT_SHAPES = (
    "circle",
    "ellipse",
    "rectangle",
    "pill",
    "circle_annulus",
    "ellipse_annulus",
    "rectangle_annulus",
    "pill_annulus",
)
DEFAULT_TASKS = ("mask", "apsum")


@dataclass(frozen=True)
class Case:
    shape: str
    dtype: str
    n_apertures: int
    image: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Timing:
    task: str
    shape: str
    dtype: str
    n_apertures: int
    library: str
    seconds: float
    speedup_vs_library: str | None = None
    speedup: float | None = None


@dataclass(frozen=True)
class BenchmarkGroup:
    task: str
    shape: str
    dtype: str
    n_apertures: int
    functions: dict[str, Callable[[], object]]


def main() -> None:
    args = parse_args()

    groups: list[BenchmarkGroup] = []
    first_dtype = args.dtypes[0]
    for dtype in args.dtypes:
        rng = np.random.default_rng(args.seed)
        data = make_image(args.image_size, dtype)
        for n_apertures in args.counts:
            case = make_case(data, dtype, n_apertures, rng)
            run_mask = dtype == first_dtype
            for shape in args.shapes:
                groups.extend(prepare_shape(case, shape, args, run_mask=run_mask))

    timings = time_benchmark_groups(groups, args)

    if args.format == "csv":
        print_timings_csv(timings, args)
    else:
        print_timings_table(timings, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="square image side length in pixels",
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_COUNTS),
        help="aperture counts",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float32", "float64", "int32", "int16"],
        choices=["float32", "float64", "int32", "int16"],
        help="image dtypes to benchmark",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=list(DEFAULT_SHAPES),
        choices=list(DEFAULT_SHAPES),
        help="aperture shapes to benchmark",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        choices=list(DEFAULT_TASKS),
        help="benchmark tasks to run",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help=(
            "global round-robin timing passes per benchmark; reported time is "
            "the median after dropping fastest/slowest when possible"
        ),
    )
    parser.add_argument(
        "--no-adaptive-repeats",
        action="store_true",
        help=(
            "disable slow-backend repeat caps so every backend receives exactly "
            "--repeats timing samples"
        ),
    )
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument(
        "--atol", type=float, default=DEFAULT_ATOL, help="absolute validation tolerance"
    )
    parser.add_argument(
        "--rtol", type=float, default=DEFAULT_RTOL, help="relative validation tolerance"
    )
    parser.add_argument(
        "--photutils-rectangle-subpixels",
        type=int,
        default=DEFAULT_PHOTUTILS_RECTANGLE_SUBPIXELS,
        help=(
            "subpixel grid used for Photutils rectangular Aperture rows; "
            "Photutils maps method='exact' to subpixels=32 for rectangles"
        ),
    )
    parser.add_argument(
        "--photutils-rectangle-atol",
        type=float,
        default=DEFAULT_PHOTUTILS_RECTANGLE_ATOL,
        help="absolute validation tolerance for Photutils rectangular Aperture rows",
    )
    parser.add_argument(
        "--photutils-rectangle-rtol",
        type=float,
        default=DEFAULT_PHOTUTILS_RECTANGLE_RTOL,
        help="relative validation tolerance for Photutils rectangular Aperture rows",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv"],
        default="table",
        help=("output format; table is human-readable and csv is machine-readable"),
    )
    return parser.parse_args()


def make_image(image_size: int, dtype: str) -> np.ndarray:
    y, x = np.indices((image_size, image_size), dtype=np.float64)
    data = 100.0
    data += 5.0 * np.sin(x / 127.0)
    data += 3.0 * np.cos(y / 113.0)
    data += 0.001 * x + 0.002 * y
    return data.astype(np.dtype(dtype))


def make_case(
    data: np.ndarray, dtype: str, n_apertures: int, rng: np.random.Generator
) -> Case:
    margin = 32.0
    height, width = data.shape
    x = rng.uniform(margin, width - margin, size=n_apertures).astype(np.float64)
    y = rng.uniform(margin, height - margin, size=n_apertures).astype(np.float64)
    return Case(shape="", dtype=dtype, n_apertures=n_apertures, image=data, x=x, y=y)


def prepare_shape(
    case: Case, shape: str, args: argparse.Namespace, *, run_mask: bool
) -> list[BenchmarkGroup]:
    mask_functions = mask_benchmarks(case, shape, args)
    apsum_functions = apsum_benchmarks(case, shape, args)
    groups: list[BenchmarkGroup] = []

    if "mask" in args.tasks and run_mask and mask_functions:
        validate_mask_results(shape, case.n_apertures, mask_functions, args)
        groups.append(
            BenchmarkGroup(
                "mask",
                shape,
                case.dtype,
                case.n_apertures,
                mask_functions,
            )
        )

    if "apsum" in args.tasks:
        validate_results("apsum", shape, case.n_apertures, apsum_functions, args)
        groups.append(
            BenchmarkGroup(
                "apsum",
                shape,
                case.dtype,
                case.n_apertures,
                apsum_functions,
            )
        )
    return groups


def mask_benchmarks(
    case: Case, shape: str, args: argparse.Namespace
) -> dict[str, Callable[[], object]]:
    if shape == "circle":
        radius = 5.0
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: apers.CircAp(positions, radius, validate=False).weights(),
            AAP: lambda: apers.CircAp(positions, radius).weights(),
            "photutils.geometry": lambda: circular_grid_mask_npix(
                case.x, case.y, radius
            ),
            "photutils.Aperture": lambda: as_mask_list(
                CircularAperture(positions, r=radius).to_mask(method="exact")
            ),
        }
    if shape == "ellipse":
        a, b, theta = 7.0, 3.0, 0.4
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: apers.EllipAp(
                positions, a, b, theta, validate=False
            ).weights(),
            AAP: lambda: apers.EllipAp(positions, a, b, theta).weights(),
            "photutils.geometry": lambda: elliptical_grid_mask_npix(
                case.x, case.y, a, b, theta
            ),
            "photutils.Aperture": lambda: as_mask_list(
                EllipticalAperture(positions, a=a, b=b, theta=theta).to_mask(
                    method="exact"
                )
            ),
        }
    if shape == "rectangle":
        w, h, theta = 9.0, 5.0, 0.4
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: apers.RectAp(
                positions, w, h, theta, validate=False
            ).weights(),
            AAP: lambda: apers.RectAp(positions, w, h, theta).weights(),
            "photutils.Aperture": lambda: as_mask_list(
                RectangularAperture(positions, w=w, h=h, theta=theta).to_mask(
                    method="subpixel", subpixels=args.photutils_rectangle_subpixels
                )
            ),
        }
    if shape == "pill":
        w, a, b, theta = 8.0, 3.0, 2.0, 0.4
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: apers.PillAp(
                positions, w, a, b, theta, validate=False
            ).weights(),
            AAP: lambda: apers.PillAp(positions, w, a, b, theta).weights(),
        }
    if shape == "circle_annulus":
        r_in, r_out = 2.0, 5.0
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: apers.CircAn(
                positions, r_in, r_out, validate=False
            ).weights(),
            AAP: lambda: apers.CircAn(positions, r_in, r_out).weights(),
            "photutils.geometry": lambda: circular_annulus_grid_mask_npix(
                case.x, case.y, r_in, r_out
            ),
            "photutils.Aperture": lambda: as_mask_list(
                CircularAnnulus(positions, r_in=r_in, r_out=r_out).to_mask(
                    method="exact"
                )
            ),
        }
    if shape == "ellipse_annulus":
        a, b, r_in, r_out, theta = 4.0, 2.0, 0.5, 1.8, -0.25
        a_in, b_in = a * r_in, b * r_in
        a_out, b_out = a * r_out, b * r_out
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: apers.EllipAn(
                positions, a_in, b_in, a_out, b_out, theta_in=theta, validate=False
            ).weights(),
            AAP: lambda: apers.EllipAn(
                positions, a_in, b_in, a_out, b_out, theta_in=theta
            ).weights(),
            "photutils.geometry": lambda: elliptical_annulus_grid_mask_npix(
                case.x, case.y, a, b, r_in, r_out, theta
            ),
            "photutils.Aperture": lambda: as_mask_list(
                EllipticalAnnulus(
                    positions,
                    a_in=a_in,
                    a_out=a_out,
                    b_in=b_in,
                    b_out=b_out,
                    theta=theta,
                ).to_mask(method="exact")
            ),
        }
    if shape == "rectangle_annulus":
        w_in, h_in, w_out, h_out, theta = 3.0, 2.0, 9.0, 5.0, 0.4
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: apers.RectAn(
                positions, w_in, h_in, w_out, h_out, theta_in=theta, validate=False
            ).weights(),
            AAP: lambda: apers.RectAn(
                positions, w_in, h_in, w_out, h_out, theta_in=theta
            ).weights(),
            "photutils.Aperture": lambda: as_mask_list(
                RectangularAnnulus(
                    positions,
                    w_in=w_in,
                    w_out=w_out,
                    h_in=h_in,
                    h_out=h_out,
                    theta=theta,
                ).to_mask(
                    method="subpixel", subpixels=args.photutils_rectangle_subpixels
                )
            ),
        }
    if shape == "pill_annulus":
        w_in, a_in, b_in, w_out, a_out, b_out, theta = 5.0, 1.5, 1.0, 8.0, 3.0, 2.0, 0.4
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: apers.PillAn(
                positions,
                w_in,
                a_in,
                b_in,
                w_out,
                a_out,
                b_out,
                theta_in=theta,
                validate=False,
            ).weights(),
            AAP: lambda: apers.PillAn(
                positions, w_in, a_in, b_in, w_out, a_out, b_out, theta_in=theta
            ).weights(),
        }
    raise ValueError(f"unsupported shape: {shape}")


def apsum_benchmarks(
    case: Case, shape: str, args: argparse.Namespace
) -> dict[str, Callable[[], np.ndarray]]:
    if shape == "circle":
        radius = 5.0
        positions = np.column_stack((case.x, case.y))
        functions = {
            AAP_OPT: lambda: aapk.apsum_circ_exact(
                case.image, case.x, case.y, radius, return_npix=False
            ),
            AAP: lambda: apers.CircAp(positions, radius).apsum(
                case.image, return_npix=False
            ),
            "photutils.geometry": lambda: grid_photometry(
                case.image,
                case.x,
                case.y,
                lambda xi, yi: circular_grid_mask(xi, yi, radius),
            ),
            "photutils.Aperture": lambda: photutils_photometry(
                CircularAperture(positions, r=radius), case.image
            ),
        }
        if sep is not None and case.image.dtype != np.int16:
            functions["sep"] = lambda: sep.sum_circle(
                case.image, case.x, case.y, radius, subpix=0
            )[0]
        return functions
    if shape == "ellipse":
        a, b, theta = 7.0, 3.0, 0.4
        positions = np.column_stack((case.x, case.y))
        functions = {
            AAP_OPT: lambda: aapk.apsum_ellip_exact(
                case.image, case.x, case.y, a, b, theta, return_npix=False
            ),
            AAP: lambda: apers.EllipAp(positions, a, b, theta).apsum(
                case.image, return_npix=False
            ),
            "photutils.geometry": lambda: grid_photometry(
                case.image,
                case.x,
                case.y,
                lambda xi, yi: elliptical_grid_mask(xi, yi, a, b, theta),
            ),
            "photutils.Aperture": lambda: photutils_photometry(
                EllipticalAperture(positions, a=a, b=b, theta=theta),
                case.image,
            ),
        }
        if sep is not None and case.image.dtype != np.int16:
            functions["sep"] = lambda: sep.sum_ellipse(
                case.image, case.x, case.y, a, b, theta, subpix=0
            )[0]
        return functions
    if shape == "rectangle":
        w, h, theta = 9.0, 5.0, 0.4
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: aapk.apsum_rect_exact(
                case.image, case.x, case.y, w, h, theta, return_npix=False
            ),
            AAP: lambda: apers.RectAp(positions, w, h, theta).apsum(
                case.image, return_npix=False
            ),
            "photutils.Aperture": lambda: photutils_photometry(
                RectangularAperture(positions, w=w, h=h, theta=theta),
                case.image,
                method="subpixel",
                subpixels=args.photutils_rectangle_subpixels,
            ),
        }
    if shape == "pill":
        w, a, b, theta = 8.0, 3.0, 2.0, 0.4
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: aapk.apsum_pill_exact(
                case.image, case.x, case.y, w, a, b, theta, return_npix=False
            ),
            AAP: lambda: apers.PillAp(positions, w, a, b, theta).apsum(
                case.image, return_npix=False
            ),
        }
    if shape == "circle_annulus":
        r_in, r_out = 2.0, 5.0
        positions = np.column_stack((case.x, case.y))
        functions = {
            AAP_OPT: lambda: aapk.apsum_circ_ann_exact(
                case.image, case.x, case.y, r_in, r_out, return_npix=False
            ),
            AAP: lambda: apers.CircAn(positions, r_in, r_out).apsum(
                case.image, return_npix=False
            ),
            "photutils.geometry": lambda: grid_photometry(
                case.image,
                case.x,
                case.y,
                lambda xi, yi: circular_annulus_grid_mask(xi, yi, r_in, r_out),
            ),
            "photutils.Aperture": lambda: photutils_photometry(
                CircularAnnulus(positions, r_in=r_in, r_out=r_out), case.image
            ),
        }
        if sep is not None and case.image.dtype != np.int16:
            functions["sep"] = lambda: sep.sum_circann(
                case.image, case.x, case.y, r_in, r_out, subpix=0
            )[0]
        return functions
    if shape == "ellipse_annulus":
        a, b, r_in, r_out, theta = 4.0, 2.0, 0.5, 1.8, -0.25
        a_in, b_in = a * r_in, b * r_in
        a_out, b_out = a * r_out, b * r_out
        positions = np.column_stack((case.x, case.y))
        functions = {
            AAP_OPT: lambda: aapk.apsum_ellip_ann_exact(
                case.image,
                case.x,
                case.y,
                a_in,
                b_in,
                a_out,
                b_out,
                theta,
                return_npix=False,
            ),
            AAP: lambda: apers.EllipAn(
                positions, a_in, b_in, a_out, b_out, theta_in=theta
            ).apsum(case.image, return_npix=False),
            "photutils.geometry": lambda: grid_photometry(
                case.image,
                case.x,
                case.y,
                lambda xi, yi: elliptical_annulus_grid_mask(
                    xi, yi, a, b, r_in, r_out, theta
                ),
            ),
            "photutils.Aperture": lambda: photutils_photometry(
                EllipticalAnnulus(
                    positions,
                    a_in=a_in,
                    a_out=a_out,
                    b_in=b_in,
                    b_out=b_out,
                    theta=theta,
                ),
                case.image,
            ),
        }
        if sep is not None and case.image.dtype != np.int16:
            functions["sep"] = lambda: sep.sum_ellipann(
                case.image, case.x, case.y, a, b, theta, r_in, r_out, subpix=0
            )[0]
        return functions
    if shape == "rectangle_annulus":
        w_in, h_in, w_out, h_out, theta = 3.0, 2.0, 9.0, 5.0, 0.4
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: aapk.apsum_rect_ann_exact(
                case.image,
                case.x,
                case.y,
                w_in,
                h_in,
                w_out,
                h_out,
                theta,
                return_npix=False,
            ),
            AAP: lambda: apers.RectAn(
                positions, w_in, h_in, w_out, h_out, theta_in=theta
            ).apsum(case.image, return_npix=False),
            "photutils.Aperture": lambda: photutils_photometry(
                RectangularAnnulus(
                    positions,
                    w_in=w_in,
                    w_out=w_out,
                    h_in=h_in,
                    h_out=h_out,
                    theta=theta,
                ),
                case.image,
                method="subpixel",
                subpixels=args.photutils_rectangle_subpixels,
            ),
        }
    if shape == "pill_annulus":
        w_in, a_in, b_in, w_out, a_out, b_out, theta = 5.0, 1.5, 1.0, 8.0, 3.0, 2.0, 0.4
        positions = np.column_stack((case.x, case.y))
        return {
            AAP_OPT: lambda: aapk.apsum_pill_ann_exact(
                case.image,
                case.x,
                case.y,
                w_in,
                a_in,
                b_in,
                w_out,
                a_out,
                b_out,
                theta,
                return_npix=False,
            ),
            AAP: lambda: apers.PillAn(
                positions, w_in, a_in, b_in, w_out, a_out, b_out, theta_in=theta
            ).apsum(case.image, return_npix=False),
        }
    raise ValueError(f"unsupported shape: {shape}")


def as_mask_list(masks) -> list:
    return masks if isinstance(masks, list) else [masks]


def mask_npix(masks) -> np.ndarray:
    if isinstance(masks, np.ndarray) and masks.ndim == 1:
        return np.asarray(masks, dtype=np.float64)
    mask_list = as_mask_list(masks)
    return np.array(
        [mask_weights(mask).sum(dtype=np.float64) for mask in mask_list],
        dtype=np.float64,
    )


def mask_weights(mask) -> np.ndarray:
    if hasattr(mask, "weights"):
        weights = mask.weights
    elif hasattr(mask, "data"):
        weights = mask.data
    else:
        weights = mask
    return np.asarray(weights, dtype=np.float64)


def is_npix_vector(masks) -> bool:
    return isinstance(masks, np.ndarray) and masks.ndim == 1


def photutils_photometry(
    aperture, data: np.ndarray, method: str = "exact", subpixels: int = 5
) -> np.ndarray:
    apsum, _ = aperture.do_photometry(data, method=method, subpixels=subpixels)
    return np.asarray(apsum, dtype=np.float64)


def circular_grid_mask_npix(x: np.ndarray, y: np.ndarray, radius: float) -> np.ndarray:
    return np.array(
        [
            circular_grid_mask(float(xi), float(yi), radius)[0].sum()
            for xi, yi in zip(x, y)
        ],
        dtype=np.float64,
    )


def circular_annulus_grid_mask_npix(
    x: np.ndarray, y: np.ndarray, r_in: float, r_out: float
) -> np.ndarray:
    return np.array(
        [
            circular_annulus_grid_mask(float(xi), float(yi), r_in, r_out)[0].sum()
            for xi, yi in zip(x, y)
        ],
        dtype=np.float64,
    )


def elliptical_grid_mask_npix(
    x: np.ndarray, y: np.ndarray, a: float, b: float, theta: float
) -> np.ndarray:
    return np.array(
        [
            elliptical_grid_mask(float(xi), float(yi), a, b, theta)[0].sum()
            for xi, yi in zip(x, y)
        ],
        dtype=np.float64,
    )


def elliptical_annulus_grid_mask_npix(
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    b: float,
    r_in: float,
    r_out: float,
    theta: float,
) -> np.ndarray:
    return np.array(
        [
            elliptical_annulus_grid_mask(
                float(xi), float(yi), a, b, r_in, r_out, theta
            )[0].sum()
            for xi, yi in zip(x, y)
        ],
        dtype=np.float64,
    )


def grid_photometry(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask_factory: Callable[[float, float], tuple[np.ndarray, int, int]],
) -> np.ndarray:
    apsum = np.empty(x.size, dtype=np.float64)
    for idx, (xi, yi) in enumerate(zip(x, y)):
        mask, ixmin, iymin = mask_factory(float(xi), float(yi))
        cutout = data[iymin : iymin + mask.shape[0], ixmin : ixmin + mask.shape[1]]
        apsum[idx] = np.sum(cutout * mask, dtype=np.float64)
    return apsum


def circular_grid_mask(
    x: float, y: float, radius: float
) -> tuple[np.ndarray, int, int]:
    ixmin, ixmax, iymin, iymax = bbox_from_extent(x, y, radius, radius)
    return (
        circular_grid_mask_on_bbox(x, y, radius, ixmin, ixmax, iymin, iymax),
        ixmin,
        iymin,
    )


def circular_annulus_grid_mask(
    x: float, y: float, r_in: float, r_out: float
) -> tuple[np.ndarray, int, int]:
    ixmin, ixmax, iymin, iymax = bbox_from_extent(x, y, r_out, r_out)
    outer = circular_grid_mask_on_bbox(x, y, r_out, ixmin, ixmax, iymin, iymax)
    inner = circular_grid_mask_on_bbox(x, y, r_in, ixmin, ixmax, iymin, iymax)
    return np.clip(outer - inner, 0.0, 1.0), ixmin, iymin


def circular_grid_mask_on_bbox(
    x: float, y: float, radius: float, ixmin: int, ixmax: int, iymin: int, iymax: int
) -> np.ndarray:
    mask = circular_overlap_grid(
        ixmin - 0.5 - x,
        ixmax - 0.5 - x,
        iymin - 0.5 - y,
        iymax - 0.5 - y,
        ixmax - ixmin,
        iymax - iymin,
        radius,
        1,
        5,
    )
    return np.asarray(mask, dtype=np.float64)


def elliptical_grid_mask(
    x: float, y: float, a: float, b: float, theta: float
) -> tuple[np.ndarray, int, int]:
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = float(np.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2))
    dy = float(np.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2))
    ixmin, ixmax, iymin, iymax = bbox_from_extent(x, y, dx, dy)
    return (
        elliptical_grid_mask_on_bbox(x, y, a, b, theta, ixmin, ixmax, iymin, iymax),
        ixmin,
        iymin,
    )


def elliptical_annulus_grid_mask(
    x: float,
    y: float,
    a: float,
    b: float,
    r_in: float,
    r_out: float,
    theta: float,
) -> tuple[np.ndarray, int, int]:
    a_in, b_in = a * r_in, b * r_in
    a_out, b_out = a * r_out, b * r_out
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = float(np.sqrt((a_out * cos_t) ** 2 + (b_out * sin_t) ** 2))
    dy = float(np.sqrt((a_out * sin_t) ** 2 + (b_out * cos_t) ** 2))
    ixmin, ixmax, iymin, iymax = bbox_from_extent(x, y, dx, dy)
    outer = elliptical_grid_mask_on_bbox(
        x, y, a_out, b_out, theta, ixmin, ixmax, iymin, iymax
    )
    inner = elliptical_grid_mask_on_bbox(
        x, y, a_in, b_in, theta, ixmin, ixmax, iymin, iymax
    )
    return np.clip(outer - inner, 0.0, 1.0), ixmin, iymin


def elliptical_grid_mask_on_bbox(
    x: float,
    y: float,
    a: float,
    b: float,
    theta: float,
    ixmin: int,
    ixmax: int,
    iymin: int,
    iymax: int,
) -> np.ndarray:
    mask = elliptical_overlap_grid(
        ixmin - 0.5 - x,
        ixmax - 0.5 - x,
        iymin - 0.5 - y,
        iymax - 0.5 - y,
        ixmax - ixmin,
        iymax - iymin,
        a,
        b,
        theta,
        1,
        5,
    )
    return np.asarray(mask, dtype=np.float64)


def bbox_from_extent(
    x: float, y: float, dx: float, dy: float
) -> tuple[int, int, int, int]:
    ixmin = int(np.floor((x - dx) + 0.5))
    ixmax = int(np.ceil((x + dx) + 0.5))
    iymin = int(np.floor((y - dy) + 0.5))
    iymax = int(np.ceil((y + dy) + 0.5))
    return ixmin, ixmax, iymin, iymax


def validate_results(
    task: str,
    shape: str,
    n_apertures: int,
    functions: dict[str, Callable[[], np.ndarray]],
    args: argparse.Namespace,
) -> None:
    reference_name = AAP_OPT
    reference = np.asarray(functions[reference_name](), dtype=np.float64)
    for name, function in functions.items():
        if name == reference_name:
            continue
        candidate = np.asarray(function(), dtype=np.float64)
        rtol, atol = validation_tolerances(shape, name, args)
        try:
            assert_allclose(candidate, reference, rtol=rtol, atol=atol)
        except AssertionError as exc:
            max_abs = float(np.max(np.abs(candidate - reference)))
            denom = np.maximum(np.abs(reference), atol)
            max_rel = float(np.max(np.abs(candidate - reference) / denom))
            raise AssertionError(
                f"{task} {shape} n={n_apertures}: {name} disagrees with {reference_name}; "
                f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, "
                f"rtol={rtol:.3g}, atol={atol:.3g}"
            ) from exc


def validate_mask_results(
    shape: str,
    n_apertures: int,
    functions: dict[str, Callable[[], object]],
    args: argparse.Namespace,
) -> None:
    reference_name = AAP_OPT
    reference_masks = functions[reference_name]()
    reference = mask_npix(reference_masks)
    for name, function in functions.items():
        if name == reference_name:
            continue
        candidate_masks = function()
        candidate = mask_npix(candidate_masks)
        rtol, atol = validation_tolerances(shape, name, args)
        try:
            assert_allclose(candidate, reference, rtol=rtol, atol=atol)
            validate_mask_samples(
                shape, n_apertures, name, candidate_masks, reference_masks, args
            )
        except AssertionError as exc:
            max_abs = float(np.max(np.abs(candidate - reference)))
            denom = np.maximum(np.abs(reference), atol)
            max_rel = float(np.max(np.abs(candidate - reference) / denom))
            raise AssertionError(
                f"mask {shape} n={n_apertures}: {name} disagrees with {reference_name}; "
                f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, "
                f"rtol={rtol:.3g}, atol={atol:.3g}"
            ) from exc


def validation_tolerances(
    shape: str, name: str, args: argparse.Namespace
) -> tuple[float, float]:
    if name == "photutils.Aperture" and shape in {"rectangle", "rectangle_annulus"}:
        return args.photutils_rectangle_rtol, args.photutils_rectangle_atol
    return args.rtol, args.atol


def validate_mask_samples(
    shape: str,
    n_apertures: int,
    name: str,
    candidate_masks: object,
    reference_masks: object,
    args: argparse.Namespace,
) -> None:
    if name != AAP:
        return
    if is_npix_vector(candidate_masks) or is_npix_vector(reference_masks):
        return
    candidate_list = as_mask_list(candidate_masks)
    reference_list = as_mask_list(reference_masks)
    if len(candidate_list) != len(reference_list):
        raise AssertionError(
            f"mask {shape} n={n_apertures}: {name} returned {len(candidate_list)} "
            f"masks, expected {len(reference_list)}"
        )
    sample_indices = sorted({0, len(reference_list) // 2, len(reference_list) - 1})
    for idx in sample_indices:
        candidate = mask_weights(candidate_list[idx])
        reference = mask_weights(reference_list[idx])
        if candidate.shape != reference.shape:
            raise AssertionError(
                f"mask {shape} n={n_apertures}: {name} mask {idx} shape "
                f"{candidate.shape} differs from reference {reference.shape}"
            )
        assert_allclose(candidate, reference, rtol=args.rtol, atol=args.atol)


def time_benchmark_groups(
    groups: list[BenchmarkGroup], args: argparse.Namespace
) -> list[Timing]:
    raw_by_group = global_round_robin_median_times(groups, args)

    timings: list[Timing] = []
    for group_idx, group in enumerate(groups):
        raw = raw_by_group[group_idx]
        timings.extend(
            Timing(
                group.task,
                group.shape,
                group.dtype,
                group.n_apertures,
                name,
                seconds,
            )
            for name, seconds in raw.items()
        )
        aap_opt_time = raw.get(AAP_OPT)
        if aap_opt_time is not None:
            for name, seconds in raw.items():
                if name == AAP_OPT:
                    continue
                timings.append(
                    Timing(
                        task=group.task,
                        shape=group.shape,
                        dtype=group.dtype,
                        n_apertures=group.n_apertures,
                        library=AAP_OPT,
                        seconds=aap_opt_time,
                        speedup_vs_library=name,
                        speedup=seconds / aap_opt_time,
                    )
                )
    return timings


def global_round_robin_median_times(
    groups: list[BenchmarkGroup], args: argparse.Namespace
) -> dict[int, dict[str, float]]:
    for group in groups:
        for function in group.functions.values():
            result = function()
            if result_size(result) == 0:
                raise RuntimeError("benchmark function returned an empty result")

    samples = {
        group_idx: {name: [] for name in group.functions}
        for group_idx, group in enumerate(groups)
    }
    repeat_counts = {
        group_idx: {
            name: repeat_count_for_backend(name, group.n_apertures, args)
            for name in group.functions
        }
        for group_idx, group in enumerate(groups)
    }

    max_repeats = max(
        (count for counts in repeat_counts.values() for count in counts.values()),
        default=0,
    )
    rng = random.Random(12345)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat_idx in range(max_repeats):
            group_order = list(enumerate(groups))
            rng.shuffle(group_order)
            for group_idx, group in group_order:
                library_order = list(group.functions)
                rng.shuffle(library_order)
                for name in library_order:
                    if repeat_idx >= repeat_counts[group_idx][name]:
                        continue
                    start = time.perf_counter()
                    result = group.functions[name]()
                    elapsed = time.perf_counter() - start
                    if result_size(result) == 0:
                        raise RuntimeError(
                            "benchmark function returned an empty result"
                        )
                    samples[group_idx][name].append(elapsed)
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        group_idx: {
            name: trimmed_median(sample)
            for name, sample in group_samples.items()
            if sample
        }
        for group_idx, group_samples in samples.items()
    }


def trimmed_median(samples: list[float]) -> float:
    if len(samples) >= 3:
        samples = sorted(samples)[1:-1]
    return float(np.median(samples))


def repeat_count_for_backend(
    library: str, n_apertures: int, args: argparse.Namespace
) -> int:
    if args.no_adaptive_repeats:
        return args.repeats
    return adaptive_repeats(library, n_apertures, args.repeats)


def adaptive_repeats(library: str, n_apertures: int, repeats: int) -> int:
    if repeats <= 1 or library in {AAP, AAP_OPT}:
        return repeats
    if library.startswith("photutils"):
        if n_apertures >= 10_000:
            return 1
        if n_apertures >= 1_000:
            return min(repeats, 2)
        if n_apertures >= 100:
            return min(repeats, 3)
    if library == "sep" and n_apertures >= 10_000:
        return min(repeats, 2)
    return repeats


def result_size(result) -> int:
    if isinstance(result, list):
        return len(result)
    return np.asarray(result).size


def print_notes(args: argparse.Namespace) -> None:
    print(f"# validation defaults: rtol={DEFAULT_RTOL:g}, atol={DEFAULT_ATOL:g}")
    print(
        "# timings report median elapsed seconds after dropping one fastest and "
        "one slowest repeat when at least three timing samples are available"
    )
    print(
        "# benchmark cases and backend timings are sampled in deterministic "
        "global round-robin order to reduce fixed-order cache and thread-pool "
        "warmup bias"
    )
    if args.no_adaptive_repeats:
        print(
            "# adaptive repeat caps are disabled; every backend uses --repeats samples"
        )
    else:
        print(
            "# adaptive repeat caps are enabled for slow large-n photutils/SEP rows; "
            "use --no-adaptive-repeats to disable"
        )
    print(
        "# Photutils rectangular Aperture validation: "
        f"rtol={args.photutils_rectangle_rtol:g}, "
        f"atol={args.photutils_rectangle_atol:g}"
    )
    print(
        "# `mask` rows run only for one dtype because mask weights are dtype-independent."
    )
    print(
        "# photu_opt: photutils.geometry low-level overlap kernels; for mask rows "
        "this reports summed temporary weights, not reusable mask objects"
    )
    print(
        "# photu_Ap.: default photutils Aperture objects and do_photometry/to_mask methods"
    )
    print(
        "# rectangle and rectangle_annulus photu_opt rows are skipped: "
        "exact rectangle geometry is absent"
    )
    print(
        "# photutils rectangular Aperture rows use method='subpixel', "
        f"subpixels={args.photutils_rectangle_subpixels}; Photutils maps "
        "method='exact' to subpixels=32 for rectangles"
    )
    print(
        "# aap_opt: astroapers speed path; direct kernels where available and "
        "validate=False object paths otherwise"
    )
    print(
        "# aap: normal object API; supported unmasked apsum rows delegate to "
        "the same kernels, so ratios near 1 are ties"
    )
    print("# ratios are backend_time / aap_opt_time")


def print_timings_csv(timings: list[Timing], args: argparse.Namespace) -> None:
    print_notes(args)
    print("task,shape,dtype,n_apertures,library,seconds,speedup_vs_library,speedup")
    for timing in timings:
        speedup = "" if timing.speedup is None else f"{timing.speedup:.3f}"
        print(
            f"{timing.task},{timing.shape},{timing.dtype},{timing.n_apertures},{timing.library},"
            f"{timing.seconds:.9f},{timing.speedup_vs_library or ''},{speedup}"
        )
    print_fastest_backends_csv(timings)


def print_timings_table(timings: list[Timing], args: argparse.Namespace) -> None:
    raw = [timing for timing in timings if timing.speedup_vs_library is None]
    print(
        f"{'dtype':<7} {'task':<6} {'shape':<{SHAPE_COLUMN_WIDTH}} {'n':>7} "
        f"{'aap_opt_us':>14} {'aap_us':>9} {'aap/aap_opt':>11} "
        f"{'photu_opt/aap_opt':>17} "
        f"{'photu_Ap./aap_opt':>23} {'sep/aap_opt':>11}"
    )
    print(
        f"{'-' * 7} {'-' * 6} {'-' * SHAPE_COLUMN_WIDTH} {'-' * 7} "
        f"{'-' * 14} {'-' * 9} {'-' * 11} "
        f"{'-' * 17} {'-' * 23} {'-' * 11}"
    )
    keys = sorted(
        {
            (timing.dtype, timing.task, timing.shape, timing.n_apertures)
            for timing in raw
        }
    )
    for dtype, task, shape, n_apertures in keys:
        group = [
            timing
            for timing in raw
            if (timing.dtype, timing.task, timing.shape, timing.n_apertures)
            == (dtype, task, shape, n_apertures)
        ]
        seconds_by_library = {timing.library: timing.seconds for timing in group}
        aap_opt_seconds = seconds_by_library.get(AAP_OPT)
        aap_ratio = format_ratio(seconds_by_library.get(AAP), aap_opt_seconds)
        photu_opt_ratio = format_ratio(
            seconds_by_library.get("photutils.geometry"), aap_opt_seconds
        )
        photu_ap_ratio = format_ratio(
            seconds_by_library.get("photutils.Aperture"), aap_opt_seconds
        )
        print(
            f"{dtype:<7} {task:<6} {shape:<{SHAPE_COLUMN_WIDTH}} {n_apertures:7d} "
            f"{format_us(aap_opt_seconds):>14} "
            f"{format_us(seconds_by_library.get(AAP)):>9} "
            f"{aap_ratio:>11} "
            f"{photu_opt_ratio:>17} "
            f"{photu_ap_ratio:>23} "
            f"{format_ratio(seconds_by_library.get('sep'), aap_opt_seconds):>11}"
        )
    print_notes(args)


def print_fastest_backends_csv(timings: list[Timing]) -> None:
    raw = [timing for timing in timings if timing.speedup_vs_library is None]
    keys = sorted(
        {
            (timing.task, timing.shape, timing.dtype, timing.n_apertures)
            for timing in raw
        }
    )
    print()
    print("# fastest backend summary")
    print("# fastest_speedup_vs_aap_opt > 1 means another backend was faster")
    print(
        "task,shape,dtype,n_apertures,fastest_library,fastest_seconds,"
        "aap_opt_seconds,fastest_speedup_vs_aap_opt"
    )
    for task, shape, dtype, n_apertures in keys:
        group = [
            timing
            for timing in raw
            if (timing.task, timing.shape, timing.dtype, timing.n_apertures)
            == (task, shape, dtype, n_apertures)
        ]
        fastest = min(group, key=lambda timing: timing.seconds)
        aap_opt = next(timing for timing in group if timing.library == AAP_OPT)
        speedup = aap_opt.seconds / fastest.seconds
        print(
            f"{task},{shape},{dtype},{n_apertures},{fastest.library},"
            f"{fastest.seconds:.9f},{aap_opt.seconds:.9f},{speedup:.3f}"
        )


def format_us(seconds: float | None) -> str:
    if seconds is None:
        return "    --   "
    return f"{seconds * 1.0e6:9.2f}"


def format_ratio(seconds: float | None, reference: float | None) -> str:
    if seconds is None or reference is None:
        return "    --   "
    return f"{seconds / reference:9.2f}"


if __name__ == "__main__":
    main()

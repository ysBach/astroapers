"""Command-line helpers for astroapers."""

from __future__ import annotations

import argparse

from ._calibration import ParallelThresholdCalibration, calibrate_parallel_threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astroapers", description="astroapers utilities"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser(
        "calibrate-threshold", help="benchmark and recommend the parallel threshold"
    )
    calibrate.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        help="aperture counts to test",
    )
    calibrate.add_argument("--repeats", type=int, default=1000, help="timing repeats")
    calibrate.add_argument(
        "--image-size", type=int, default=256, help="test image size"
    )
    calibrate.add_argument("--radius", type=float, default=5.0, help="circle radius")

    args = parser.parse_args(argv)
    if args.command == "calibrate-threshold":
        result = calibrate_parallel_threshold(
            counts=args.counts,
            repeats=args.repeats,
            image_size=args.image_size,
            radius=args.radius,
            apply=False,
        )
        print_calibration(result)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def print_calibration(result: ParallelThresholdCalibration) -> None:
    print(f"recommended_threshold: {result.threshold}")
    print(f"original_threshold: {result.original_threshold}")
    print("count serial_us parallel_us winner")
    for count, serial, parallel in zip(
        result.counts, result.serial_seconds, result.parallel_seconds
    ):
        winner = "parallel" if parallel < serial else "serial"
        print(f"{count:5d} {serial * 1e6:9.2f} {parallel * 1e6:11.2f} {winner}")
    print()
    print("Use this threshold in a shell before starting Python/IPython/Jupyter:")
    print(f"export ASTROAPERS_PARALLEL_THRESHOLD={result.threshold}")
    print()
    print(
        "To persist it for future shell sessions, add this line to your shell config:"
    )
    print(f"export ASTROAPERS_PARALLEL_THRESHOLD={result.threshold}")
    print()
    print("Or set it inside an already-running Python process:")
    print("import astroapers as aap")
    print(f"aap.set_parallel_threshold({result.threshold})")


if __name__ == "__main__":
    raise SystemExit(main())

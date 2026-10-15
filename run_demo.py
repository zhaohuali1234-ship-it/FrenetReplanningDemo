"""Run the stable Frenet replanning simulation with an optional delivery delay."""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    # Windows consoles may default to GBK, which cannot print units such as m/s².
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("base", "H01", "H05"), default="base")
    parser.add_argument("--delay-ms", type=float, default=0.0,
                        help="Simulated planning-result delivery delay (default: 0)")
    parser.add_argument("--show", action="store_true", help="Show Matplotlib animation")
    parser.add_argument("--expect-complete", action="store_true",
                        help="Exit with an error unless the vehicle reaches the goal")
    args = parser.parse_args()
    if not np.isfinite(args.delay_ms) or args.delay_ms < 0:
        parser.error("--delay-ms must be a non-negative finite number")
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    module_dir = Path(__file__).resolve().parent / "PathPlanning" / "FrenetOptimalTrajectory"
    sys.path.insert(0, str(module_dir))
    import frenet_tracking_oos_core as core

    shifts = {"base": (0.0, 0.0), "H01": (-4.0, 0.0), "H05": (-3.0, 1.0)}
    core.planner.OBSTACLES = np.asarray(core.planner.OBSTACLES, dtype=float) + shifts[args.case]
    core.planner.INITIAL_SPEED = 10.0 / 3.6
    result = core.main(show_animation=args.show, fixed_delay_ms=args.delay_ms)
    completed = bool(result["goal_reached"])
    print(f"\nCase: {args.case}; delivery delay: {args.delay_ms:g} ms")
    print(f"Outcome: {'COMPLETE' if completed else 'SAFE_BRAKE / INCOMPLETE'}")
    print(f"Minimum circular-model clearance: {float(result['clearance']):.3f} m")
    print(f"Reason: {result['status']}")
    if args.expect_complete and not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

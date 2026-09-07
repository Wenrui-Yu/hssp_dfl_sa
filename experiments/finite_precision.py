#!/usr/bin/env python3
"""Robustness to finite-precision errors -- paper Table 3 (Section 7.5).

Real DFL checkpoints are stored in float64 and integerised with a fixed-point
scale ``S = 10^gamma``.  Truncating instead of rounding introduces an error
``E = Y_observed - X A_int`` that the exact Nguyen-Stern Step 1 cannot absorb,
so the attack switches to the noise-tolerant Step 1 of Section 4.4:

    A_int      = quantize(S * client_checkpoint)          # t0.5 states
    Y_observed = quantize(S * beta * aggregate_checkpoint) # t1 aggregates
    E          = Y_observed - X * A_int

The attack only receives ``Y_observed mod Q`` and a conservative public bound
``rho``; the clean observation is used solely to score the recovery.  Each cell
of Table 3 is ``recall / number of retained candidate vectors``.

Paper configuration (Section 7.5 and Appendix G)
    n = 10, e = 20, eta = 0.6, mHLCP, batch size 1, 100-bit pseudo-prime Q
    gamma in {4, 6, 8, 10}  i.e.  S in {1e4, 1e6, 1e8, 1e10}
    r     in {1, 2, 4, 8}   noisy Step 1 observation columns
    exact-r = 4             the exact-arithmetic baseline
    1 trial per cell, up to 200 coordinate resamples

A bare invocation reproduces exactly that grid.  Requires SageMath.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hssp_dfl import paths  # noqa: E402

PAPER_SCALES = [10**4, 10**6, 10**8, 10**10]
PAPER_R = [1, 2, 4, 8]


def main():
    parser = argparse.ArgumentParser(
        description="Table 3: noisy mHSSP/mHLCP under fixed-point truncation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+",
                        default=["sentiment140", "purchase", "cifar"])
    parser.add_argument("--scales", nargs="+", type=int, default=PAPER_SCALES,
                        help="fixed-point scales S = 10^gamma")
    parser.add_argument("--r", nargs="+", type=int, default=PAPER_R,
                        dest="r_values", help="noisy Step 1 column counts")
    parser.add_argument("--exact-r", nargs="+", type=int, default=[4],
                        dest="exact_r", help="exact-arithmetic baseline columns")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-resample", type=int, default=200)
    parser.add_argument("--quantizers", nargs="+", default=["truncate"],
                        choices=["nearest", "truncate"])
    parser.add_argument("--trial-timeout", type=int, default=10)
    parser.add_argument("--output-dir", default=str(paths.RESULTS / "finite_precision"))
    parser.add_argument("--skip-tex", action="store_true",
                        help="do not render the LaTeX table afterwards")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "-m",
        "hssp_dfl.experiments.checkpoint_truncation_replay",
        "--datasets", *args.datasets,
        "--scales", *[str(s) for s in args.scales],
        "--quantizers", *args.quantizers,
        "--r", *[str(r) for r in args.r_values],
        "--exact-r", *[str(r) for r in args.exact_r],
        "--trials", str(args.trials),
        "--max-resample", str(args.max_resample),
        "--trial-timeout", str(args.trial_timeout),
        "--model-dir", str(paths.MODEL_DIR),
        "--network-path", str(paths.NETWORK_MAT),
        "--output-dir", str(out),
    ]
    print("[finite_precision] " + " ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)

    trials_csv = out / "checkpoint_replay_trials.csv"
    print(f"\n[finite_precision] per-trial results  -> {trials_csv}")
    print(f"[finite_precision] summary            -> "
          f"{out / 'checkpoint_replay_summary.csv'}")

    if args.skip_tex:
        return
    tex_command = [
        sys.executable, str(ROOT / "figures" / "make_table3_tex.py"),
        "--input", str(trials_csv),
        "--output", str(out / "table3.tex"),
        "--scales", *[str(s) for s in args.scales],
        "--r-values", *[str(r) for r in args.r_values],
        "--baseline-scale", str(max(args.scales)),
        "--baseline-exact-r", str(args.exact_r[-1]),
    ]
    print("[finite_precision] " + " ".join(tex_command))
    subprocess.run(tex_command, check=True, cwd=ROOT)
    print(f"[finite_precision] Table 3 LaTeX       -> {out / 'table3.tex'}")


if __name__ == "__main__":
    main()

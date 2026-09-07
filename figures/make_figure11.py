"""Exact vs. noise-tolerant Step 1 under DP -- paper Figure 11.

Overlays the recall of the exact Step 1 (which recovers nothing once
aggregation-level DP is applied) with the recall of the noise-tolerant Step 1
of Section 4.4 (which still recovers 25-50% of the ground-truth vectors).

Inputs: the two CSVs written by experiments/dp_defense.py with --step1 exact
and --step1 noisy.
"""


import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hssp_dfl import paths

_DP = paths.RESULTS / "dp_defense"
_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_parser.add_argument("--exact-csv", default=str(_DP / "dp_exact_stats.csv"))
_parser.add_argument("--noisy-csv", default=str(_DP / "dp_noisy_stats.csv"))
_parser.add_argument("--extended-noisy-csv",
                     default=str(_DP / "dp_noisy_extended_stats.csv"),
                     help="optional extra noisy sweep; skipped when absent")
_parser.add_argument("--comparison-csv",
                     default=str(_DP / "dp_exact_noisy_comparison.csv"))
_parser.add_argument("--output",
                     default=str(paths.RESULTS / "figures" /
                                 "figure11_dp_noisy_robustness.png"))
_ARGS = _parser.parse_args()

ROOT = paths.ROOT
EXACT_CSV = Path(_ARGS.exact_csv)
NOISY_CSV = Path(_ARGS.noisy_csv)
EXTENDED_NOISY_CSV = Path(_ARGS.extended_noisy_csv)
OUT_CSV = Path(_ARGS.comparison_csv)
OUT_FIG = Path(_ARGS.output)


def read_rows(path):
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["dp_mode"], float(row["dp_epsilon"])): row for row in rows}


def as_int(row, key):
    value = row.get(key) if row is not None else None
    if value in (None, ""):
        return None
    return int(float(value))


def build_comparison(exact_rows, noisy_rows, extended_noisy_rows=None):
    extended_noisy_rows = extended_noisy_rows or {}
    all_noisy_rows = dict(noisy_rows)
    all_noisy_rows.update(extended_noisy_rows)
    keys = sorted(set(exact_rows) | set(all_noisy_rows))
    comparison = []
    for mode, epsilon in keys:
        exact = exact_rows.get((mode, epsilon))
        noisy = all_noisy_rows.get((mode, epsilon))
        n_honest = as_int(noisy, "n_honest") or as_int(exact, "n_honest")
        if n_honest is None:
            continue
        comparison.append(
            {
                "dp_mode": mode,
                "dp_epsilon": epsilon,
                "n_honest": n_honest,
                "exact_status": exact["status"] if exact else "",
                "noisy_status": noisy["status"] if noisy else "",
                "exact_vectors_matched": as_int(exact, "true_vectors_matched"),
                "noisy_vectors_matched": as_int(noisy, "true_vectors_matched"),
                "exact_reconstruction_count": as_int(exact, "reconstruction_count"),
                "noisy_reconstruction_count": as_int(noisy, "reconstruction_count"),
                "rho_assumed": as_int(noisy, "rho_assumed"),
                "rho_true": as_int(noisy, "rho_true"),
                "rho_bound_valid": noisy["rho_bound_valid"] if noisy else "",
            }
        )
    return comparison


def write_comparison(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mode_series(rows, mode):
    baseline = next(row for row in rows if row["dp_mode"] == "none")
    mode_rows = sorted(
        (row for row in rows if row["dp_mode"] == mode),
        key=lambda row: row["dp_epsilon"],
    )
    selected = [baseline] + mode_rows
    x = [1.0] + [float(row["dp_epsilon"]) for row in mode_rows]
    labels = ["no DP"] + [f"{row['dp_epsilon']:g}" for row in mode_rows]
    exact = [
        None
        if row["exact_vectors_matched"] is None
        else 100.0 * row["exact_vectors_matched"] / row["n_honest"]
        for row in selected
    ]
    noisy = [
        None
        if row["noisy_vectors_matched"] is None
        else 100.0 * row["noisy_vectors_matched"] / row["n_honest"]
        for row in selected
    ]
    return x, labels, exact, noisy


def format_epsilon_tick(value):
    if value == 1.0:
        return "no DP"
    if value >= 1e4:
        return f"{value:.0e}"
    return f"{value:g}"


def plot(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    panels = (
        (axes[0], "exchange", "(a)"),
        (axes[1], "aggregate", "(b)"),
    )
    for ax, mode, title in panels:
        x, labels, exact, noisy = mode_series(rows, mode)
        ax.plot(x, exact, "o--", color="#6b7280", label="Exact Step 1")
        ax.plot(x, noisy, "s-", color="#d62728", label="Noisy Step 1")
        ax.set_xscale("log")
        tick_indices = [0] + list(range(2, len(x), 2))
        if tick_indices[-1] != len(x) - 1:
            tick_indices.append(len(x) - 1)
        tick_x = [x[index] for index in tick_indices]
        tick_labels = [format_epsilon_tick(x_value) for x_value in tick_x]
        ax.set_xticks(tick_x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(r"$\epsilon$")
        ax.set_xlim(0.7, max(x) * 1.6)
        ax.set_ylim(-5, 110)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.text(
            0.5,
            -0.42,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("recall (%)")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.40)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    exact_rows = read_rows(EXACT_CSV)
    noisy_rows = read_rows(NOISY_CSV)
    extended_rows = read_rows(EXTENDED_NOISY_CSV) if EXTENDED_NOISY_CSV.exists() else {}
    comparison = build_comparison(exact_rows, noisy_rows, extended_rows)
    if not comparison:
        raise RuntimeError("No aligned exact/noisy DP rows found")
    write_comparison(comparison, OUT_CSV)
    plot(comparison, OUT_FIG)
    print(f"Compared {len(comparison)} aligned rows")
    print(f"Saved comparison -> {OUT_CSV}")
    print(f"Saved figure -> {OUT_FIG}")


if __name__ == "__main__":
    main()

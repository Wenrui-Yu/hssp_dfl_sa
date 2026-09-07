"""Utility vs. attack success under DP -- paper Figure 5.

Joins the per-epsilon test accuracy written by training/train_cifar_dp.py with
the attack outcome written by experiments/dp_defense.py --step1 exact, and
plots both panels: (a) local DP (``exchange``), (b) aggregation-level DP.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hssp_dfl import paths

_DP = paths.RESULTS / "dp_defense"
_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_parser.add_argument("--attack-csv", default=str(_DP / "dp_exact_stats.csv"),
                     help="output of experiments/dp_defense.py --step1 exact")
_parser.add_argument("--fl-dir", default=None,
                     help="directory with the fl_dp_*.csv accuracy logs "
                          "(default: alongside --attack-csv)")
_parser.add_argument("--output",
                     default=str(paths.RESULTS / "figures" / "figure5_dp_defense.png"))
_ARGS = _parser.parse_args()

ROOT = paths.ROOT
ATTACK_CSV = Path(_ARGS.attack_csv)
FL_DIR = Path(_ARGS.fl_dir) if _ARGS.fl_dir else ATTACK_CSV.parent
FL_GLOB = "fl_dp_cifar10_cnn_ni500_N10_e10*.csv"
OUT_FIG = Path(_ARGS.output)


def _safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_attack_rows(path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            mode = r.get("dp_mode", "").strip()
            eps = _safe_float(r.get("dp_epsilon"))
            if mode == "" or eps is None:
                continue

            status = (r.get("status") or "").strip()
            ok = status == "ok"
            # Compatibility: user text mentions no_fund; treat both as failed.
            failed = status in {"no_found", "no_fund"}

            rows.append(
                {
                    "mode": mode,
                    "epsilon": eps,
                    "status": status,
                    "attack_success": 1 if ok else 0,
                    "attack_failed": 1 if failed else 0,
                    "gradient_mse": _safe_float(r.get("gradient_mse")),
                }
            )
    return rows


def read_fl_rows(table_dp_dir):
    rows = []
    for path in sorted(table_dp_dir.glob(FL_GLOB)):
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            file_rows = list(reader)

        if not file_rows:
            continue

        mode = file_rows[0].get("dp_mode", "").strip()
        eps = _safe_float(file_rows[0].get("dp_epsilon"))
        if mode == "" or eps is None:
            continue

        # Use the last non-NaN accuracy in this run.
        acc_vals = [_safe_float(r.get("test_acc_mean")) for r in file_rows]
        acc_vals = [x for x in acc_vals if x is not None]
        if not acc_vals:
            continue
        final_acc = acc_vals[-1]

        rows.append({"mode": mode, "epsilon": eps, "test_acc": final_acc, "file": path.name})

    return rows


def build_joined(attack_rows, fl_rows):
    attack_map = {(r["mode"], r["epsilon"]): r for r in attack_rows}

    joined = []
    for fr in fl_rows:
        key = (fr["mode"], fr["epsilon"])
        ar = attack_map.get(key)
        if ar is None:
            continue
        item = {
            "mode": fr["mode"],
            "epsilon": fr["epsilon"],
            "test_acc": fr["test_acc"],
            "status": ar["status"],
            "attack_success": ar["attack_success"],
            "gradient_mse": ar["gradient_mse"],
        }
        joined.append(item)

    # Utility drop (%) relative to best observed utility in the same mode.
    by_mode = {}
    for r in joined:
        by_mode.setdefault(r["mode"], []).append(r)

    return joined


def _extract_baseline(joined):
    for row in joined:
        if row["mode"] == "none":
            return row
    raise RuntimeError("No baseline row found for dp_mode=none.")


def _mode_series(joined, mode, baseline):
    rows = sorted([r for r in joined if r["mode"] == mode], key=lambda x: x["epsilon"], reverse=True)
    labels = ["no DP"] + [f"{r['epsilon']:g}" for r in rows]
    test_acc = [100.0 * baseline["test_acc"]] + [100.0 * r["test_acc"] for r in rows]
    attack_success = [100.0 * baseline["attack_success"]] + [100.0 * r["attack_success"] for r in rows]
    return labels, test_acc, attack_success


def plot(joined, out_path):
    baseline = _extract_baseline(joined)
    baseline_acc_pct = 100.0 * baseline["test_acc"]
    exchange_labels, exchange_acc, exchange_success = _mode_series(joined, "exchange", baseline)
    aggregate_labels, aggregate_acc, aggregate_success = _mode_series(joined, "aggregate", baseline)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    small_font = 9

    plots = [
        (axes[0], "(a)", exchange_labels, exchange_acc, exchange_success, "#1f77b4"),
        (axes[1], "(b)", aggregate_labels, aggregate_acc, aggregate_success, "#d62728"),
    ]

    for ax, title, labels, test_acc, attack_success, color in plots:
        x = list(range(len(labels)))
        ax2 = ax.twinx()

        ax.plot(x, test_acc, "o-", color=color, label="Test accuracy (%)")
        ax2.plot(x, attack_success, "s--", color="#2ca02c", label="Attack success (%)")

        ax.text(
            0.5,
            -0.25,
            f"{title}",
            transform=ax.transAxes,
            ha="center",
            va="top",
                fontsize=small_font+2
            )
        ax.set_ylabel("Test accuracy (%)", fontsize=small_font)
        ax2.set_ylabel("Attack success (%)", fontsize=small_font)
        ax.set_ylim(-5.0, baseline_acc_pct+5)
        ax2.set_ylim(-5.0, 105.0)
        ax.grid(True, linestyle="--", alpha=0.35)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=small_font)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(r"$\epsilon$", fontsize=small_font)
        ax.tick_params(axis='y', labelsize=small_font)
        ax2.tick_params(axis='y', labelsize=small_font)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28, wspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure -> {out_path}")


if __name__ == "__main__":
    attack_rows = read_attack_rows(ATTACK_CSV)
    fl_rows = read_fl_rows(FL_DIR)
    joined_rows = build_joined(attack_rows, fl_rows)

    print(f"Loaded attack rows: {len(attack_rows)}")
    print(f"Loaded FL rows: {len(fl_rows)}")
    print(f"Matched rows: {len(joined_rows)}")

    if not joined_rows:
        raise SystemExit(
            f"No epsilon appears in both the attack stats and the accuracy logs.\n"
            f"  attack stats  : {ATTACK_CSV}\n"
            f"  accuracy logs : {FL_DIR}/{FL_GLOB}\n\n"
            f"The accuracy logs are written by training/train_cifar_dp.py, into the\n"
            f"same results/dp_defense/ directory.  If you are attacking pre-trained\n"
            f"DP checkpoints without retraining, copy the published logs first:\n"
            f"  cp reference/tables/fl_dp_cifar10_cnn_ni500_N10_e10*.csv "
            f"results/dp_defense/"
        )

    plot(joined_rows, OUT_FIG)

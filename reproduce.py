#!/usr/bin/env python3
"""One entry point for every result in the paper.

    python reproduce.py --list                 # show all targets and their cost
    python reproduce.py table1                 # reproduce one result
    python reproduce.py figure2 figure8        # several at once
    python reproduce.py lattice                # a whole group
    python reproduce.py table7 --dry-run       # print the commands only

Each target expands to the exact command(s) that produced the published result.

Targets are grouped by cost:

    fast     seconds to a few minutes, no SageMath needed
    medium   minutes, needs SageMath
    slow     tens of minutes to hours
    heavy    hours, needs the optional `breaching` dependency or GPU training
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _exp(name):
    return [PY, str(ROOT / "experiments" / name)]


def _fig(name):
    return [PY, str(ROOT / "figures" / name)]


# The DP figures were produced from two separate sweeps.  The group set fixes
# the order in which checkpoints are attacked, which fixes the coordinates the
# noisy Step 1 samples -- so these patterns are what makes Figure 11
# reproducible rather than merely similar.
DP_MAIN_GROUPS = (
    r"^model_avg_ni500_N10_z0_e10"
    r"(_exchange_ep(10|50|100|200|500|1000|10000)p0"
    r"|_aggregate_ep(50|100|200|500|1000|10000)p0)?$"
)
DP_EXTENDED_GROUPS = (
    r"^model_avg_ni500_N10_z0_e10_aggregate_ep"
    r"(20000|50000|100000|1000000|100000000|10000000000)p0$"
)


# ---------------------------------------------------------------------------
# target -> {paper, cost, needs, commands}
# ---------------------------------------------------------------------------
TARGETS: dict[str, dict] = {
    # ---------------- topology structure (no SageMath needed) --------------
    "figure2": {
        "paper": "Figure 2 - honest-node recoverability vs eta, undirected",
        "cost": "fast",
        "needs": "python",
        "commands": [
            _exp("topology_recovery.py") + ["--graph", "undirected", "--nodes", "20",
                                            "--edges", "40", "--trials", "100"],
            _exp("topology_recovery.py") + ["--graph", "undirected", "--nodes", "20",
                                            "--edges", "60", "--trials", "100"],
        ],
    },
    "figure8": {
        "paper": "Figure 8 - honest-node recoverability vs eta, directed",
        "cost": "medium",
        "needs": "python",
        "commands": [
            _exp("topology_recovery.py") + ["--graph", "directed", "--nodes", "20",
                                            "--edges", str(e), "--trials", "100"]
            for e in (40, 60, 80, 100)
        ],
    },
    "figure6": {
        "paper": "Figure 6 - core sub-graph size distribution, large networks",
        "cost": "medium",
        "needs": "python",
        "commands": [
            _exp("core_subgraph_scale.py") + ["--nodes", str(n), "--edges", str(e),
                                              "--trials", "100"]
            for n, e in ((100, 150), (100, 200), (200, 300), (200, 400))
        ],
    },
    "figure7": {
        "paper": "Figure 7 - core sub-graph size across ER / ring / small-world / scale-free",
        "cost": "medium",
        "needs": "python",
        "commands": [_exp("topology_families.py")],
    },

    # ---------------- synthetic lattice attack (SageMath) ------------------
    "table1": {
        "paper": "Table 1 - mHLCP recall and candidate count, undirected",
        "cost": "slow",
        "needs": "sage",
        "commands": [
            _exp("lattice_attack_batch.py") + [
                "--problem", "mhlcp", "--nodes", str(n), "--edges", str(e),
                "--corrupt-ratio", str(r), "--topologies", "100",
                "--max-resample", "1", "--summary-only"]
            for n, e, r in ((10, 20, 0.6), (10, 20, 0.7), (10, 20, 0.8),
                            (20, 40, 0.7), (20, 40, 0.8))
        ] + [_fig("make_summary_tables.py")],
    },
    "table9": {
        "paper": "Table 9 - mHSSP recall and candidate count, undirected",
        "cost": "slow",
        "needs": "sage",
        "commands": [
            _exp("lattice_attack_batch.py") + [
                "--problem", "mhssp", "--nodes", str(n), "--edges", str(e),
                "--corrupt-ratio", str(r), "--topologies", "100",
                "--max-resample", "1", "--summary-only"]
            for n, e, r in ((10, 20, 0.6), (10, 20, 0.7), (10, 20, 0.8),
                            (20, 40, 0.6), (20, 40, 0.7), (20, 40, 0.8),
                            (30, 60, 0.6), (30, 60, 0.7), (30, 60, 0.8),
                            (40, 80, 0.6), (40, 80, 0.7), (40, 80, 0.8))
        ] + [_fig("make_summary_tables.py")],
    },
    "table7": {
        "paper": "Table 7 - per-topology mHLCP with Cases 1-3, eta=0.6",
        "cost": "medium",
        "needs": "sage",
        "commands": [
            _exp("lattice_attack_batch.py") + [
                "--problem", "mhlcp", "--nodes", "10", "--edges", "20",
                "--corrupt-ratio", "0.6", "--topologies", "10"],
        ],
    },
    "table8": {
        "paper": "Table 8 - per-topology mHLCP with Cases 1-3, eta=0.7",
        "cost": "fast",
        "needs": "sage",
        "commands": [
            _exp("lattice_attack_batch.py") + [
                "--problem", "mhlcp", "--nodes", "10", "--edges", "20",
                "--corrupt-ratio", "0.7", "--topologies", "10"],
        ],
    },
    "table10": {
        "paper": "Table 10 - per-topology mHSSP with Cases 2-3, eta=0.6",
        "cost": "fast",
        "needs": "sage",
        "commands": [
            _exp("lattice_attack_batch.py") + [
                "--problem", "mhssp", "--nodes", "10", "--edges", "20",
                "--corrupt-ratio", "0.6", "--topologies", "10"],
        ],
    },
    "table11": {
        "paper": "Table 11 - per-topology mHSSP with Cases 2-3, eta=0.7",
        "cost": "fast",
        "needs": "sage",
        "commands": [
            _exp("lattice_attack_batch.py") + [
                "--problem", "mhssp", "--nodes", "10", "--edges", "20",
                "--corrupt-ratio", "0.7", "--topologies", "10"],
        ],
    },
    "table5": {
        "paper": "Table 5 - directed push-sum mHLCP with mass-scalar validation",
        "cost": "medium",
        "needs": "sage",
        "commands": [
            _exp("lattice_attack_batch.py") + [
                "--graph", "pushsum", "--problem", "mhlcp", "--nodes", "10",
                "--edges", "30", "--corrupt-ratio", "0.7", "--topologies", "5"],
        ],
    },

    # ---------------- real datasets (SageMath + PyTorch + assets) ----------
    "figure3": {
        "paper": "Figure 3 - CIFAR-10 ground truth vs. spurious reconstructions",
        "cost": "slow",
        "needs": "sage+assets",
        "commands": [
            _exp("attack_cifar.py") + ["--candidates", "30"],
            _fig("make_figure3.py"),
        ],
    },
    "figure4": {
        "paper": "Figure 4 - CIFAR-10 statistical reconstruction quality",
        "cost": "slow",
        "needs": "sage+assets",
        "commands": [
            _exp("attack_cifar.py") + ["--candidates", "30"],
            _fig("make_stat_figures.py") + ["--dataset", "cifar"],
        ],
    },
    "figure9": {
        "paper": "Figure 9 - Purchase-100 statistical reconstruction quality",
        "cost": "medium",
        "needs": "sage+assets",
        "commands": [
            _exp("attack_purchase.py") + ["--candidates", "30"],
            _fig("make_stat_figures.py") + ["--dataset", "purchase"],
        ],
    },
    "figure10": {
        "paper": "Figure 10 - Sentiment140 statistical reconstruction quality",
        "cost": "slow",
        "needs": "sage+assets+vec2text",
        "commands": [
            _exp("attack_sentiment140.py") + ["--candidates", "30"],
            _exp("vec2text_eval.py"),
            _fig("make_stat_figures.py") + ["--dataset", "sentiment140"],
        ],
    },
    "figures12-17": {
        "paper": "Figures 12-14 (CIFAR) and 15-17 (Purchase) - all-candidate grids",
        "cost": "slow",
        "needs": "sage+assets",
        "commands": [
            _exp("attack_cifar.py") + ["--candidates", "30"],
            _exp("attack_purchase.py") + ["--candidates", "30"],
            _fig("make_case_grids.py"),
        ],
    },
    "tables12-14": {
        "paper": "Tables 12-14 (and the Table 2 excerpt) - Sentiment140 text reconstructions",
        "cost": "slow",
        "needs": "sage+assets+vec2text",
        "commands": [
            _exp("attack_sentiment140.py") + ["--candidates", "30"],
            _exp("vec2text_eval.py"),
            _exp("add_bertscore.py"),
            _fig("make_text_tables.py"),
        ],
    },

    # ---------------- robustness and defenses ------------------------------
    "table3": {
        "paper": "Table 3 - robustness to finite-precision truncation",
        "cost": "medium",
        "needs": "sage+assets",
        "commands": [_exp("finite_precision.py")],
    },
    "figure5": {
        "paper": "Figure 5 - test accuracy and attack success under LDP / CDP",
        "cost": "slow",
        "needs": "sage+assets(dp)",
        "commands": [
            _exp("dp_defense.py") + ["--step1", "exact",
                                     "--group-pattern", DP_MAIN_GROUPS,
                                     "--output", "results/dp_defense/dp_exact_stats.csv"],
            _fig("make_figure5.py"),
        ],
    },
    "figure11": {
        "paper": "Figure 11 - exact vs. noise-tolerant Step 1 under DP",
        "cost": "slow",
        "needs": "sage+assets(dp)",
        "commands": [
            _exp("dp_defense.py") + ["--step1", "exact",
                                     "--group-pattern", DP_MAIN_GROUPS,
                                     "--output", "results/dp_defense/dp_exact_stats.csv"],
            _exp("dp_defense.py") + ["--step1", "noisy",
                                     "--group-pattern", DP_MAIN_GROUPS,
                                     "--output", "results/dp_defense/dp_noisy_stats.csv"],
            _exp("dp_defense.py") + ["--step1", "noisy",
                                     "--group-pattern", DP_EXTENDED_GROUPS,
                                     "--output",
                                     "results/dp_defense/dp_noisy_extended_stats.csv"],
            _fig("make_figure11.py"),
        ],
    },
    "table6": {
        "paper": "Table 6 - downstream GIA transferability (Breaching, batches 1-8)",
        "cost": "heavy",
        "needs": "sage+assets+breaching",
        "commands": [
            [PY, str(ROOT / "experiments" / "gia_transfer" / "run_pipeline.py")],
        ],
    },
}

# Table 2 and Table 4 have no separate driver.
NOTES = {
    "table2": "Table 2 is a hand-picked excerpt of Table 12; run `tables12-14`.",
    "table4": "Table 4 is a literature comparison; it has no experiment.",
    "figure1": "Figure 1 is a conceptual illustration drawn for the paper.",
}

GROUPS = {
    "fast": [k for k, v in TARGETS.items() if v["cost"] == "fast"],
    "medium": [k for k, v in TARGETS.items() if v["cost"] == "medium"],
    "slow": [k for k, v in TARGETS.items() if v["cost"] == "slow"],
    "heavy": [k for k, v in TARGETS.items() if v["cost"] == "heavy"],
    "topology": ["figure2", "figure6", "figure7", "figure8"],
    "lattice": ["table1", "table5", "table7", "table8", "table9", "table10", "table11"],
    "realdata": ["figure3", "figure4", "figure9", "figure10",
                 "figures12-17", "tables12-14"],
    "defense": ["table3", "figure5", "figure11"],
    "all": list(TARGETS),
}


def list_targets():
    print(f"{'target':<14} {'cost':<7} {'needs':<24} paper result")
    print("-" * 100)
    for name, spec in TARGETS.items():
        print(f"{name:<14} {spec['cost']:<7} {spec['needs']:<24} {spec['paper']}")
    print()
    print("groups: " + ", ".join(GROUPS))
    print()
    for name, note in NOTES.items():
        print(f"note: {note}")


def expand(names):
    resolved, seen = [], set()
    for name in names:
        for target in GROUPS.get(name, [name]):
            if target not in TARGETS:
                raise SystemExit(f"unknown target: {target!r} "
                                 f"(try --list)")
            if target not in seen:
                seen.add(target)
                resolved.append(target)
    return resolved


def run(names, dry_run=False):
    failures = []
    for name in names:
        spec = TARGETS[name]
        print(f"\n{'=' * 78}")
        print(f"{name}: {spec['paper']}")
        print(f"cost={spec['cost']}  needs={spec['needs']}")
        print("=" * 78, flush=True)

        for command in spec["commands"]:
            full = [str(part) for part in command]
            print("+ " + " ".join(shlex.quote(part) for part in full), flush=True)
            if dry_run:
                continue
            started = time.time()
            result = subprocess.run(full, cwd=ROOT)
            elapsed = time.time() - started
            if result.returncode != 0:
                print(f"  FAILED after {elapsed:.1f}s (exit {result.returncode})")
                failures.append((name, " ".join(full)))
                break
            print(f"  ok ({elapsed:.1f}s)")

    print(f"\n{'=' * 78}")
    if failures:
        print(f"{len(failures)} step(s) failed:")
        for name, command in failures:
            print(f"  {name}: {command}")
        return 1
    print("all requested targets completed")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("targets", nargs="*",
                        help="target or group names (see --list)")
    parser.add_argument("--list", action="store_true",
                        help="show every target and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running them")
    args = parser.parse_args()

    if args.list or not args.targets:
        list_targets()
        return 0
    return run(expand(args.targets), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

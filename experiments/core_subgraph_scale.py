#!/usr/bin/env python3
"""Core sub-graph size distribution in large networks -- paper Figure 6.

After the topological simplification of Section 4.2, a large sparse DFL graph
decomposes into several independent core sub-graphs.  This script measures the
distribution of the *honest* core size ``|V_h|`` over ``--trials`` random
connected Erdos-Renyi ``G(n, e)`` graphs, for each corruption ratio ``eta``.

Figure 6 panels:

    --nodes 100 --edges 150     (a)
    --nodes 100 --edges 200     (b)
    --nodes 200 --edges 300     (c)
    --nodes 200 --edges 400     (d)

The concentration of the mass in ``[0, 10]`` is what justifies evaluating the
lattice attack on core sub-graphs of size ``n = 10`` in the rest of the paper.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (  # noqa: E402
    DEFAULT_CORRUPT_RATIOS,
    DEFAULT_SEED,
    resolve_output_dir,
    savefig,
    seed_everything,
    split_by_ratio,
    use_agg_backend,
    write_run_config,
)
from experiments.topology_recovery import classify_undirected  # noqa: E402
from hssp_dfl.graph import generate_connected_graph  # noqa: E402


def run(args):
    seed_everything(args.seed)

    # size_freq[eta][core_size] -> number of observed core sub-graphs
    size_freq = defaultdict(lambda: defaultdict(int))

    for trial in range(args.trials):
        graph = generate_connected_graph(args.nodes, args.edges)
        nodes = list(range(args.nodes))

        for ratio in args.corrupt_ratios:
            n_corrupt, _ = split_by_ratio(args.nodes, ratio)
            corrupt = sorted(random.sample(nodes, n_corrupt))
            honest = sorted(set(nodes) - set(corrupt))

            _, _, _, cores = classify_undirected(graph, honest, corrupt)
            for core in cores:
                size_freq[ratio][len(core["honest"])] += 1

        if args.verbose and (trial + 1) % 10 == 0:
            print(f"  trial {trial + 1}/{args.trials}", flush=True)

    rows = []
    for ratio in args.corrupt_ratios:
        total = sum(size_freq[ratio].values())
        for size in sorted(size_freq[ratio]):
            rows.append(
                {
                    "eta": ratio,
                    "core_size": size,
                    "count": size_freq[ratio][size],
                    "probability": size_freq[ratio][size] / total if total else 0.0,
                }
            )
    return rows, size_freq


def main():
    parser = argparse.ArgumentParser(
        description="Figure 6: core sub-graph size distribution in large graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nodes", type=int, default=100, help="node count n")
    parser.add_argument("--edges", type=int, default=150, help="edge count e")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--corrupt-ratios", nargs="+", type=float,
                        default=list(DEFAULT_CORRUPT_RATIOS), dest="corrupt_ratios")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=None,
                        help="default: results/core_subgraph_scale")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out = resolve_output_dir(args.output_dir, "core_subgraph_scale")
    tag = f"node{args.nodes}_edge{args.edges}"
    print(f"[core_subgraph_scale] {tag}, trials={args.trials}, seed={args.seed}")

    rows, size_freq = run(args)

    csv_path = out / f"core_size_{tag}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=["eta", "core_size", "count", "probability"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {csv_path}")

    use_agg_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    for ratio in args.corrupt_ratios:
        sizes = sorted(size_freq[ratio])
        total = sum(size_freq[ratio].values())
        if not total:
            continue
        ax.plot(sizes, [size_freq[ratio][s] / total for s in sizes],
                marker="o", linewidth=1.5, label=rf"$\eta={ratio:.1f}$")
    ax.set_xlabel("Core subgraph size")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    savefig(fig, out / f"core_size_{tag}.png")

    write_run_config(out / f"run_config_{tag}.json", vars(args))

    for ratio in args.corrupt_ratios:
        total = sum(size_freq[ratio].values())
        if not total:
            continue
        le10 = sum(c for s, c in size_freq[ratio].items() if s <= 10)
        print(f"  eta={ratio:.1f}: {total:5d} cores, "
              f"{100 * le10 / total:5.1f}% with |V_h| <= 10")


if __name__ == "__main__":
    main()

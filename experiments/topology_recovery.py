#!/usr/bin/env python3
"""Topological vulnerability analysis -- paper Figure 2 and Figure 8.

For random topologies with a fixed node count ``n`` and edge count ``e``, every
honest node is placed into exactly one of three mutually exclusive classes
(Section 4.2 / Section 7.2 of the paper):

* ``trivial``       -- directly recoverable by the trivial differencing attack
                       (7): the honest node sits alone in its corrupt/honest
                       component.
* ``core``          -- part of a core sub-graph with ``|V_c| >= |V_h| >= 2``,
                       i.e. a target for the mHSSP/mHLCP lattice attack.
* ``unrecoverable`` -- hidden within a single iteration.

The three proportions sum to one and are averaged over ``--trials`` random
graphs per corruption ratio ``eta``.

    Figure 2 (undirected):  --graph undirected --nodes 20 --edges 40
                            --graph undirected --nodes 20 --edges 60
    Figure 8 (directed)  :  --graph directed --nodes 20 --edges 40|60|80|100

The classification only uses the support of the mixing matrix, which coincides
with the adjacency structure of the graph, so no weight matrix is built here.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402

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
from hssp_dfl.graph import (  # noqa: E402
    generate_connected_graph,
    generate_strongly_connected_directed_graph,
)


# ---------------------------------------------------------------------------
# Classification of honest nodes
# ---------------------------------------------------------------------------
def classify_undirected(graph, honest, corrupt):
    """Split honest nodes into trivial / core / unrecoverable (undirected)."""
    honest_set, corrupt_set = set(honest), set(corrupt)

    bipartite = nx.Graph()
    bipartite.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        if (u in honest_set) != (v in honest_set):
            bipartite.add_edge(u, v)

    trivial, core, unrecoverable, cores = [], [], [], []
    for component in nx.connected_components(bipartite):
        h = sorted(n for n in component if n in honest_set)
        c = sorted(n for n in component if n in corrupt_set)
        if not h:
            continue
        if len(h) == 1 and len(c) >= 1:
            trivial.extend(h)
        elif len(c) >= len(h) >= 2:
            core.extend(h)
            cores.append({"honest": h, "corrupt": c})
        else:
            unrecoverable.extend(h)
    return trivial, core, unrecoverable, cores


def classify_directed(graph, honest, corrupt):
    """Split honest nodes into trivial / core / unrecoverable (directed).

    In a directed topology node ``j`` observes node ``i`` only when the edge
    ``i -> j`` exists, so the adversarial view of an honest node is determined
    by its *out*-neighbours.
    """
    honest_set, corrupt_set = set(honest), set(corrupt)

    trivial = []
    bipartite = nx.Graph()
    bipartite.add_nodes_from(graph.nodes())
    for h in honest:
        out_neighbors = list(graph.successors(h))
        out_corrupt = [j for j in out_neighbors if j in corrupt_set]
        out_honest = [j for j in out_neighbors if j in honest_set]
        if out_corrupt and not out_honest:
            trivial.append(h)
        for c in out_corrupt:
            bipartite.add_edge(h, c)

    trivial_set = set(trivial)
    core, cores, resolved = [], [], set(trivial)
    for component in nx.connected_components(bipartite):
        h = sorted(n for n in component if n in honest_set and n not in trivial_set)
        c = sorted(n for n in component if n in corrupt_set)
        if len(h) >= 2 and len(c) >= len(h):
            core.extend(h)
            cores.append({"honest": h, "corrupt": c})
            resolved.update(h)

    unrecoverable = [h for h in honest if h not in resolved]
    return trivial, core, unrecoverable, cores


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def run(args):
    seed_everything(args.seed)
    directed = args.graph == "directed"
    classify = classify_directed if directed else classify_undirected

    totals = {
        ratio: {"trivial": 0, "core": 0, "unrecoverable": 0, "honest": 0}
        for ratio in args.corrupt_ratios
    }
    core_stats = {
        ratio: {"count": 0, "honest_total": 0} for ratio in args.corrupt_ratios
    }

    for trial in range(args.trials):
        if directed:
            graph = generate_strongly_connected_directed_graph(args.nodes, args.edges)
        else:
            graph = generate_connected_graph(args.nodes, args.edges)
        nodes = list(range(args.nodes))

        for ratio in args.corrupt_ratios:
            n_corrupt, n_honest = split_by_ratio(args.nodes, ratio)
            corrupt = sorted(random.sample(nodes, n_corrupt))
            honest = sorted(set(nodes) - set(corrupt))

            trivial, core, unrecoverable, cores = classify(graph, honest, corrupt)

            totals[ratio]["trivial"] += len(trivial)
            totals[ratio]["core"] += len(core)
            totals[ratio]["unrecoverable"] += len(unrecoverable)
            totals[ratio]["honest"] += n_honest
            core_stats[ratio]["count"] += len(cores)
            core_stats[ratio]["honest_total"] += sum(len(c["honest"]) for c in cores)

        if args.verbose and (trial + 1) % 10 == 0:
            print(f"  trial {trial + 1}/{args.trials}", flush=True)

    rows = []
    for ratio in args.corrupt_ratios:
        denom = totals[ratio]["honest"] or 1
        n_cores = core_stats[ratio]["count"]
        rows.append(
            {
                "eta": ratio,
                "trivial": totals[ratio]["trivial"] / denom,
                "core": totals[ratio]["core"] / denom,
                "unrecoverable": totals[ratio]["unrecoverable"] / denom,
                "cores_per_trial": n_cores / args.trials,
                "mean_core_honest_size": (
                    core_stats[ratio]["honest_total"] / n_cores if n_cores else 0.0
                ),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Figure 2 / Figure 8: honest-node recoverability vs. eta.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph", choices=("undirected", "directed"),
                        default="undirected")
    parser.add_argument("--nodes", type=int, default=20, help="node count n")
    parser.add_argument("--edges", type=int, default=40, help="edge count e")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--corrupt-ratios", nargs="+", type=float,
                        default=list(DEFAULT_CORRUPT_RATIOS), dest="corrupt_ratios")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=None,
                        help="default: results/topology_recovery")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out = resolve_output_dir(args.output_dir, "topology_recovery")
    tag = f"{args.graph}_node{args.nodes}_edge{args.edges}"
    print(f"[topology_recovery] {tag}, trials={args.trials}, seed={args.seed}")

    rows = run(args)

    csv_path = out / f"recovery_{tag}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {csv_path}")

    use_agg_backend()
    import matplotlib.pyplot as plt

    eta = [r["eta"] for r in rows]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(eta, [r["trivial"] for r in rows], marker="o",
            label="Directly Recoverable")
    ax.plot(eta, [r["unrecoverable"] for r in rows], marker="s",
            label="Not Recoverable")
    ax.plot(eta, [r["core"] for r in rows], marker="^", label="HSSP-Eligible")
    ax.set_xlabel(r"Corrupted node ratio $\eta$")
    ax.set_ylabel("Recovery ratio")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    savefig(fig, out / f"recovery_{tag}.png")

    write_run_config(out / f"run_config_{tag}.json", vars(args))

    print("\n   eta   trivial     core   unrecoverable")
    for r in rows:
        print(f"   {r['eta']:.1f}     {r['trivial']:.3f}    {r['core']:.3f}    "
              f"{r['unrecoverable']:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Core-subgraph scale across topology families -- paper Figure 7.

Evaluate core-subgraph scale and attack feasibility across DFL topologies.

The experiment applies the same component-based topological reduction used by
``topology_recovery.py`` to four undirected topology families:

* ER: G(n, m), conditioned on connectivity;
* ring: a k-nearest-neighbour ring lattice;
* small-world: a connected Watts-Strogatz graph;
* scale-free: a Barabasi-Albert graph, augmented to the common edge budget.

All topology families use the same number of nodes and edges.  Within a trial,
they also use the same nested corrupt-node partitions.  The script reports both
the paper's cardinality condition (|V_c| >= |V_h| >= 2) and stricter full-column
rank conditions for uniform (mHSSP) and Metropolis (mHLCP) weights.  These rank
conditions are structural feasibility proxies, not substitutes for running the
lattice solver.

Example:
    conda run -n sage python experiments/topology_families.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hssp_dfl import paths


TOPOLOGIES = ("er", "ring", "small_world", "scale_free")
TOPOLOGY_LABELS = {
    "er": "ER",
    "ring": "Ring",
    "small_world": "Small-world",
    "scale_free": "Scale-free",
}
TOPOLOGY_COLORS = {
    # Match the original paper's Matplotlib color cycle.
    "er": "#1f77b4",
    "ring": "#ff7f0e",
    "small_world": "#2ca02c",
    "scale_free": "#d62728",
}

SUMMARY_METRICS = (
    "direct_recovery_rate",
    "hssp_eligible_rate",
    "structural_attackable_rate",
    "hssp_rank_attackable_rate",
    "hlcp_rank_attackable_rate",
    "hssp_tractable_attackable_rate",
    "hlcp_tractable_attackable_rate",
    "max_core_fraction",
    "hssp_feasible_core_probability",
    "hlcp_feasible_core_probability",
    "eligible_core_count",
    "max_core_honest_size",
)


def derived_seed(master_seed: int, *parts: int) -> int:
    """Derive a stable uint32 seed without depending on Python hash order."""
    sequence = np.random.SeedSequence([int(master_seed), *(int(p) for p in parts)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def validate_graph_parameters(num_nodes: int, mean_degree: int) -> int:
    """Validate a common edge budget and return its edge count."""
    if num_nodes < 6:
        raise ValueError("num_nodes must be at least 6.")
    if mean_degree < 2 or mean_degree >= num_nodes:
        raise ValueError("mean_degree must satisfy 2 <= k < num_nodes.")
    if mean_degree % 2:
        raise ValueError(
            "mean_degree must be even so ring and Watts-Strogatz graphs "
            "share exactly the same edge budget."
        )
    return num_nodes * mean_degree // 2


def _add_random_non_edges(
    graph: nx.Graph,
    target_edges: int,
    rng: np.random.Generator,
) -> None:
    """Add uniformly sampled non-edges until ``target_edges`` is reached."""
    while graph.number_of_edges() < target_edges:
        source, target = rng.choice(graph.number_of_nodes(), size=2, replace=False)
        source, target = int(source), int(target)
        if not graph.has_edge(source, target):
            graph.add_edge(source, target)


def generate_topology(
    topology: str,
    num_nodes: int,
    mean_degree: int,
    seed: int,
    ws_rewire_probability: float = 0.1,
    er_max_attempts: int = 20_000,
) -> nx.Graph:
    """Generate one connected graph with a topology-independent edge budget."""
    target_edges = validate_graph_parameters(num_nodes, mean_degree)
    if not 0.0 <= ws_rewire_probability <= 1.0:
        raise ValueError("ws_rewire_probability must lie in [0, 1].")

    rng = np.random.default_rng(seed)
    if topology == "er":
        # A genuine fixed-edge ER graph, rather than a random spanning tree
        # with extra edges.  Conditioning is necessary because DFL consensus
        # assumes connectivity.
        for _attempt in range(er_max_attempts):
            graph_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            graph = nx.gnm_random_graph(
                num_nodes,
                target_edges,
                seed=graph_seed,
            )
            if nx.is_connected(graph):
                break
        else:
            raise RuntimeError(
                "Failed to sample a connected ER graph after "
                f"{er_max_attempts} attempts (n={num_nodes}, m={target_edges})."
            )
    elif topology == "ring":
        graph = nx.watts_strogatz_graph(
            num_nodes,
            mean_degree,
            0.0,
            seed=seed,
        )
    elif topology == "small_world":
        graph = nx.connected_watts_strogatz_graph(
            num_nodes,
            mean_degree,
            ws_rewire_probability,
            tries=1_000,
            seed=seed,
        )
    elif topology == "scale_free":
        attachment_edges = mean_degree // 2
        graph = nx.barabasi_albert_graph(
            num_nodes,
            attachment_edges,
            seed=seed,
        )
        _add_random_non_edges(graph, target_edges, rng)
    else:
        raise ValueError(
            f"Unknown topology {topology!r}; choose from {TOPOLOGIES}."
        )

    if graph.number_of_edges() != target_edges:
        raise AssertionError(
            f"{topology} generated {graph.number_of_edges()} edges, "
            f"expected {target_edges}."
        )
    if not nx.is_connected(graph):
        raise AssertionError(f"{topology} generator returned a disconnected graph.")
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def _cross_partition_components(
    graph: nx.Graph,
    honest_nodes: set[int],
    corrupt_nodes: set[int],
) -> Iterable[tuple[list[int], list[int]]]:
    """Yield honest/corrupt node lists for cross-partition components."""
    bipartite = nx.Graph()
    bipartite.add_nodes_from(graph.nodes())
    for source, target in graph.edges():
        source_is_honest = source in honest_nodes
        target_is_honest = target in honest_nodes
        if source_is_honest != target_is_honest:
            bipartite.add_edge(source, target)

    for component in nx.connected_components(bipartite):
        honest = sorted(node for node in component if node in honest_nodes)
        corrupt = sorted(node for node in component if node in corrupt_nodes)
        yield honest, corrupt


def _component_ranks(
    graph: nx.Graph,
    honest: list[int],
    corrupt: list[int],
) -> tuple[int, int]:
    """Return ranks under uniform and Metropolis-Hastings edge weights."""
    binary = np.zeros((len(corrupt), len(honest)), dtype=float)
    metropolis = np.zeros_like(binary)
    degrees = dict(graph.degree())

    for row, corrupt_node in enumerate(corrupt):
        for column, honest_node in enumerate(honest):
            if graph.has_edge(corrupt_node, honest_node):
                binary[row, column] = 1.0
                metropolis[row, column] = 1.0 / (
                    max(degrees[corrupt_node], degrees[honest_node]) + 1
                )

    return (
        int(np.linalg.matrix_rank(binary)),
        int(np.linalg.matrix_rank(metropolis)),
    )


def analyze_partition(
    graph: nx.Graph,
    honest_nodes: Iterable[int],
    corrupt_nodes: Iterable[int],
    tractable_core_size: int = 10,
) -> tuple[dict[str, float | int], list[dict[str, float | int | bool]]]:
    """Apply the paper's reduction and measure structural attack feasibility."""
    honest_set = set(honest_nodes)
    corrupt_set = set(corrupt_nodes)
    if honest_set & corrupt_set:
        raise ValueError("honest_nodes and corrupt_nodes must be disjoint.")
    if honest_set | corrupt_set != set(graph.nodes()):
        raise ValueError("the partition must cover every graph node.")
    if tractable_core_size < 2:
        raise ValueError("tractable_core_size must be at least 2.")

    direct_honest = 0
    eligible_honest = 0
    hssp_rank_honest = 0
    hlcp_rank_honest = 0
    hssp_tractable_honest = 0
    hlcp_tractable_honest = 0
    core_rows: list[dict[str, float | int | bool]] = []

    for honest, corrupt in _cross_partition_components(
        graph,
        honest_set,
        corrupt_set,
    ):
        if not honest:
            continue
        if len(honest) == 1 and corrupt:
            direct_honest += 1
            continue
        if not (len(corrupt) >= len(honest) >= 2):
            continue

        binary_rank, metropolis_rank = _component_ranks(graph, honest, corrupt)
        hssp_rank_feasible = binary_rank == len(honest)
        hlcp_rank_feasible = metropolis_rank == len(honest)
        tractable = len(honest) <= tractable_core_size

        eligible_honest += len(honest)
        if hssp_rank_feasible:
            hssp_rank_honest += len(honest)
            if tractable:
                hssp_tractable_honest += len(honest)
        if hlcp_rank_feasible:
            hlcp_rank_honest += len(honest)
            if tractable:
                hlcp_tractable_honest += len(honest)

        core_rows.append(
            {
                "core_honest_size": len(honest),
                "core_corrupt_size": len(corrupt),
                "core_total_size": len(honest) + len(corrupt),
                "binary_rank": binary_rank,
                "metropolis_rank": metropolis_rank,
                "hssp_rank_feasible": hssp_rank_feasible,
                "hlcp_rank_feasible": hlcp_rank_feasible,
                "tractable": tractable,
            }
        )

    num_honest = len(honest_set)
    max_core_size = max(
        (int(row["core_honest_size"]) for row in core_rows),
        default=0,
    )
    metrics: dict[str, float | int] = {
        "n_honest": num_honest,
        "n_corrupt": len(corrupt_set),
        "direct_honest": direct_honest,
        "eligible_honest": eligible_honest,
        "unrecoverable_honest": num_honest - direct_honest - eligible_honest,
        "hssp_rank_feasible_honest": hssp_rank_honest,
        "hlcp_rank_feasible_honest": hlcp_rank_honest,
        "eligible_core_count": len(core_rows),
        "max_core_honest_size": max_core_size,
        "mean_core_honest_size": (
            float(np.mean([row["core_honest_size"] for row in core_rows]))
            if core_rows
            else 0.0
        ),
        "direct_recovery_rate": direct_honest / num_honest,
        "hssp_eligible_rate": eligible_honest / num_honest,
        "structural_attackable_rate": (
            direct_honest + eligible_honest
        )
        / num_honest,
        "hssp_rank_attackable_rate": (
            direct_honest + hssp_rank_honest
        )
        / num_honest,
        "hlcp_rank_attackable_rate": (
            direct_honest + hlcp_rank_honest
        )
        / num_honest,
        "hssp_tractable_attackable_rate": (
            direct_honest + hssp_tractable_honest
        )
        / num_honest,
        "hlcp_tractable_attackable_rate": (
            direct_honest + hlcp_tractable_honest
        )
        / num_honest,
        "max_core_fraction": max_core_size / num_honest,
        "hssp_feasible_core_probability": int(
            any(bool(row["hssp_rank_feasible"]) for row in core_rows)
        ),
        "hlcp_feasible_core_probability": int(
            any(bool(row["hlcp_rank_feasible"]) for row in core_rows)
        ),
    }
    return metrics, core_rows


def graph_descriptors(graph: nx.Graph) -> dict[str, float | int]:
    """Return topology descriptors that help interpret attack differences."""
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    return {
        "num_edges": graph.number_of_edges(),
        "mean_degree_observed": float(degrees.mean()),
        "degree_std": float(degrees.std(ddof=0)),
        "average_clustering": float(nx.average_clustering(graph)),
        "average_shortest_path": float(nx.average_shortest_path_length(graph)),
    }


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all graph and partition trials."""
    trial_rows: list[dict[str, float | int | str]] = []
    core_rows: list[dict[str, float | int | str | bool]] = []
    graph_rows: list[dict[str, float | int | str]] = []

    topology_index = {name: index for index, name in enumerate(TOPOLOGIES)}
    config_index = 0
    for num_nodes in args.nodes:
        for mean_degree in args.mean_degrees:
            validate_graph_parameters(num_nodes, mean_degree)
            for trial in range(args.trials):
                graphs: dict[str, nx.Graph] = {}
                for topology in TOPOLOGIES:
                    graph_seed = derived_seed(
                        args.seed,
                        10,
                        config_index,
                        trial,
                        topology_index[topology],
                    )
                    graph = generate_topology(
                        topology,
                        num_nodes,
                        mean_degree,
                        graph_seed,
                        ws_rewire_probability=args.ws_rewire_probability,
                    )
                    graphs[topology] = graph
                    graph_rows.append(
                        {
                            "num_nodes": num_nodes,
                            "mean_degree": mean_degree,
                            "trial": trial,
                            "topology": topology,
                            "graph_seed": graph_seed,
                            **graph_descriptors(graph),
                        }
                    )

                # The same permutation is reused by every topology and makes
                # higher corruption ratios nested within a trial.
                partition_seed = derived_seed(
                    args.seed,
                    20,
                    config_index,
                    trial,
                )
                permutation = np.random.default_rng(partition_seed).permutation(
                    num_nodes
                )
                for corrupt_ratio in args.corrupt_ratios:
                    n_corrupt = int(round(num_nodes * corrupt_ratio))
                    n_corrupt = min(max(n_corrupt, 1), num_nodes - 2)
                    corrupt = sorted(int(node) for node in permutation[:n_corrupt])
                    honest = sorted(int(node) for node in permutation[n_corrupt:])

                    for topology, graph in graphs.items():
                        metrics, component_rows = analyze_partition(
                            graph,
                            honest,
                            corrupt,
                            tractable_core_size=args.tractable_core_size,
                        )
                        common = {
                            "num_nodes": num_nodes,
                            "mean_degree": mean_degree,
                            "trial": trial,
                            "topology": topology,
                            "corrupt_ratio": corrupt_ratio,
                            "partition_seed": partition_seed,
                        }
                        trial_rows.append({**common, **metrics})
                        for core_index, component in enumerate(component_rows):
                            core_rows.append(
                                {
                                    **common,
                                    "core_index": core_index,
                                    **component,
                                }
                            )
            config_index += 1

    return (
        pd.DataFrame(trial_rows),
        pd.DataFrame(core_rows),
        pd.DataFrame(graph_rows),
    )


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Summarize trial-level metrics with normal-approximation mean CIs."""
    keys = ["num_nodes", "mean_degree", "topology", "corrupt_ratio"]
    rows: list[dict[str, float | int | str]] = []
    for key, group in trials.groupby(keys, sort=True):
        row: dict[str, float | int | str] = dict(zip(keys, key))
        for metric in SUMMARY_METRICS:
            values = group[metric].astype(float)
            mean = float(values.mean())
            standard_error = (
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = mean - 1.96 * standard_error
            row[f"{metric}_ci_high"] = mean + 1.96 * standard_error
        rows.append(row)
    return pd.DataFrame(rows)


def core_size_distribution(cores: pd.DataFrame) -> pd.DataFrame:
    """Compute the conditional PMF of eligible honest-core sizes."""
    columns = [
        "num_nodes",
        "mean_degree",
        "topology",
        "corrupt_ratio",
        "core_honest_size",
        "count",
        "probability",
    ]
    if cores.empty:
        return pd.DataFrame(columns=columns)
    keys = [
        "num_nodes",
        "mean_degree",
        "topology",
        "corrupt_ratio",
        "core_honest_size",
    ]
    result = cores.groupby(keys, sort=True).size().rename("count").reset_index()
    denominator_keys = [
        "num_nodes",
        "mean_degree",
        "topology",
        "corrupt_ratio",
    ]
    result["probability"] = result["count"] / result.groupby(
        denominator_keys
    )["count"].transform("sum")
    return result[columns]


def core_scale_overview(cores: pd.DataFrame) -> pd.DataFrame:
    """Summarize the scale and rank of pooled cardinality-eligible cores."""
    keys = ["num_nodes", "mean_degree", "topology"]
    columns = keys + [
        "eligible_cores",
        "median_core_honest_size",
        "p90_core_honest_size",
        "tractable_core_fraction",
        "hssp_rank_feasible_fraction",
        "hlcp_rank_feasible_fraction",
    ]
    if cores.empty:
        return pd.DataFrame(columns=columns)

    result = (
        cores.groupby(keys, sort=True)
        .agg(
            eligible_cores=("core_honest_size", "size"),
            median_core_honest_size=("core_honest_size", "median"),
            p90_core_honest_size=(
                "core_honest_size",
                lambda values: values.quantile(0.9),
            ),
            tractable_core_fraction=("tractable", "mean"),
            hssp_rank_feasible_fraction=("hssp_rank_feasible", "mean"),
            hlcp_rank_feasible_fraction=("hlcp_rank_feasible", "mean"),
        )
        .reset_index()
    )
    return result[columns]


def plot_core_size_distributions(
    distributions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a topology-by-configuration figure in the Figure 6 style."""
    configurations = (
        distributions[["num_nodes", "mean_degree"]]
        .drop_duplicates()
        .sort_values(["num_nodes", "mean_degree"])
        .itertuples(index=False, name=None)
    )
    configurations = list(configurations)
    if not configurations:
        return

    # Reserve a narrow caption row below each row of panels, as in the paper's
    # Figure 6 subfigure layout.
    figure = plt.figure(
        figsize=(4.0 * len(TOPOLOGIES), 3.35 * len(configurations))
    )
    height_ratios = []
    for _ in configurations:
        height_ratios.extend((1.0, 0.16))
    grid = figure.add_gridspec(
        nrows=2 * len(configurations),
        ncols=len(TOPOLOGIES),
        height_ratios=height_ratios,
        hspace=0.45,
        wspace=0.20,
    )
    axes = np.empty((len(configurations), len(TOPOLOGIES)), dtype=object)
    shared_y_axis = None
    # Figure 6 relies on Matplotlib's default categorical cycle. Index the
    # discrete colors directly: normalized sampling from ``tab10`` can skip
    # entries (notably the original purple fifth curve).
    paper_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ratios = sorted(distributions["corrupt_ratio"].unique())
    ratio_colors = {
        ratio: paper_colors[index % len(paper_colors)]
        for index, ratio in enumerate(ratios)
    }

    for row_index, (num_nodes, mean_degree) in enumerate(configurations):
        for column_index, topology in enumerate(TOPOLOGIES):
            subplot_kwargs = {}
            if shared_y_axis is not None:
                subplot_kwargs["sharey"] = shared_y_axis
            axis = figure.add_subplot(
                grid[2 * row_index, column_index],
                **subplot_kwargs,
            )
            axes[row_index, column_index] = axis
            if shared_y_axis is None:
                shared_y_axis = axis
            subset = distributions.loc[
                (distributions["num_nodes"] == num_nodes)
                & (distributions["mean_degree"] == mean_degree)
                & (distributions["topology"] == topology)
            ]
            for ratio, line in subset.groupby("corrupt_ratio", sort=True):
                # Insert zero-probability sizes so the connecting line does not
                # visually imply mass at unobserved, non-consecutive sizes.
                maximum_size = int(line["core_honest_size"].max())
                dense_line = (
                    line.set_index("core_honest_size")["probability"]
                    .reindex(range(2, maximum_size + 1), fill_value=0.0)
                    .rename_axis("core_honest_size")
                    .reset_index()
                )
                axis.plot(
                    dense_line["core_honest_size"],
                    dense_line["probability"],
                    marker="o",
                    linewidth=1.5,
                    color=ratio_colors[ratio],
                    label=rf"$\eta={ratio:.1f}$",
                )
            # Keep only the topology as the panel title. The configuration is
            # shown once below the complete row of panels.
            axis.set_title(TOPOLOGY_LABELS[topology])
            axis.set_xlabel("Core subgraph size")
            axis.set_ylabel("Probability" if column_index == 0 else "")
            axis.grid(True, alpha=0.3)
            if not subset.empty:
                axis.legend(ncol=2)

        caption_axis = figure.add_subplot(grid[2 * row_index + 1, :])
        caption_axis.axis("off")
        caption_axis.text(
            0.5,
            0.5,
            rf"({chr(ord('a') + row_index)}) "
            rf"$n={num_nodes},\ e={num_nodes * mean_degree // 2}$",
            ha="center",
            va="center",
        )

    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_attack_feasibility(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot the paper condition and the stricter mHSSP rank condition."""
    configurations = (
        summary[["num_nodes", "mean_degree"]]
        .drop_duplicates()
        .sort_values(["num_nodes", "mean_degree"])
        .itertuples(index=False, name=None)
    )
    configurations = list(configurations)
    figure, axes = plt.subplots(
        len(configurations),
        2,
        figsize=(8.2, 3.0 * len(configurations)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    plotted_metrics = (
        ("hssp_eligible_rate", "Honest nodes in cardinality-eligible cores"),
        ("hssp_rank_attackable_rate", "Direct or full-rank mHSSP proxy"),
    )

    for row_index, (num_nodes, mean_degree) in enumerate(configurations):
        for column_index, (metric, title) in enumerate(plotted_metrics):
            axis = axes[row_index, column_index]
            for topology in TOPOLOGIES:
                line = summary.loc[
                    (summary["num_nodes"] == num_nodes)
                    & (summary["mean_degree"] == mean_degree)
                    & (summary["topology"] == topology)
                ].sort_values("corrupt_ratio")
                axis.plot(
                    line["corrupt_ratio"],
                    line[f"{metric}_mean"],
                    marker="o",
                    linewidth=1.6,
                    markersize=3.5,
                    color=TOPOLOGY_COLORS[topology],
                    label=TOPOLOGY_LABELS[topology],
                )
                axis.fill_between(
                    line["corrupt_ratio"],
                    line[f"{metric}_ci_low"].clip(0, 1),
                    line[f"{metric}_ci_high"].clip(0, 1),
                    color=TOPOLOGY_COLORS[topology],
                    alpha=0.10,
                )
            axis.set_title(
                f"{title}\n"
                rf"$n={num_nodes},\ e={num_nodes * mean_degree // 2}$"
            )
            axis.set_xlabel(r"Corrupted-node ratio $\eta$")
            axis.grid(alpha=0.25)
            if column_index == 0:
                axis.set_ylabel("Fraction of honest nodes")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(TOPOLOGIES),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    graph_summary: pd.DataFrame,
    core_overview: pd.DataFrame,
) -> None:
    """Write a concise, claim-bounded Markdown result report."""
    lines = [
        "# Core-subgraph scale and attack feasibility across DFL topologies",
        "",
        "## Design",
        "",
        (
            f"- {args.trials} graph/partition trials per configuration; master "
            f"seed `{args.seed}`."
        ),
        (
            f"- Nodes: `{args.nodes}`; mean degrees: `{args.mean_degrees}`; "
            f"corruption ratios: `{args.corrupt_ratios}`."
        ),
        (
            "- ER uses fixed-edge G(n,m) conditioned on connectivity. Ring, "
            "Watts-Strogatz, and Barabasi-Albert graphs use the same n and m."
        ),
        (
            "- Corrupt-node partitions are paired across topology families and "
            "nested across corruption ratios."
        ),
    ]
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The cardinality metric reproduces the paper's component-level "
                "condition. Full-column rank adds a stricter structural check. "
                f"The tractability proxy counts full-rank cores with at most "
                f"{args.tractable_core_size} honest nodes. None of these proxies "
                "measures end-to-end lattice recall, candidate-set ambiguity, or "
                "gradient-inversion quality; those require the attack scripts."
            ),
            "",
            "## Pooled eligible-core scale",
            "",
            (
                "The following conditional summary pools cardinality-eligible "
                "cores across corruption ratios. `tractable_core_fraction` is "
                f"the fraction with at most {args.tractable_core_size} honest "
                "nodes."
            ),
            "",
            core_overview.to_markdown(index=False, floatfmt=".3f"),
            "",
            "## Mean topology descriptors",
            "",
            graph_summary.to_markdown(index=False, floatfmt=".3f"),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate core-subgraph scale and attack feasibility for ER, ring, "
            "small-world, and scale-free DFL topologies."
        )
    )
    parser.add_argument("--nodes", nargs="+", type=int, default=[100, 200])
    parser.add_argument(
        "--mean-degrees",
        nargs="+",
        type=int,
        default=[4],
        help="Even mean degrees shared by all topology families.",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--corrupt-ratios",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--ws-rewire-probability", type=float, default=0.1)
    parser.add_argument("--tractable-core-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.RESULTS / "topology_families",
    )
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive.")
    if any(not 0.0 < ratio < 1.0 for ratio in args.corrupt_ratios):
        parser.error("--corrupt-ratios values must lie strictly between 0 and 1.")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trials, cores, graphs = run_experiment(args)
    summary = summarize_trials(trials)
    distributions = core_size_distribution(cores)
    core_overview = core_scale_overview(cores)
    graph_summary = (
        graphs.groupby(["num_nodes", "mean_degree", "topology"], sort=True)[
            [
                "num_edges",
                "degree_std",
                "average_clustering",
                "average_shortest_path",
            ]
        ]
        .mean()
        .reset_index()
    )

    output_files = {
        "trial_metrics": args.output_dir / "trial_metrics.csv",
        "eligible_cores": args.output_dir / "eligible_cores.csv",
        "graph_descriptors": args.output_dir / "graph_descriptors.csv",
        "summary": args.output_dir / "summary.csv",
        "core_size_distribution": args.output_dir / "core_size_distribution.csv",
        "core_scale_overview": args.output_dir / "core_scale_overview.csv",
        "configuration": args.output_dir / "configuration.json",
        "report": args.output_dir / "report.md",
        "core_figure": args.output_dir / "core_size_distribution.png",
        "feasibility_figure": args.output_dir / "attack_feasibility.png",
    }

    trials.to_csv(output_files["trial_metrics"], index=False)
    cores.to_csv(output_files["eligible_cores"], index=False)
    graphs.to_csv(output_files["graph_descriptors"], index=False)
    summary.to_csv(output_files["summary"], index=False)
    distributions.to_csv(output_files["core_size_distribution"], index=False)
    core_overview.to_csv(output_files["core_scale_overview"], index=False)
    output_files["configuration"].write_text(
        json.dumps(
            {
                "nodes": args.nodes,
                "mean_degrees": args.mean_degrees,
                "trials": args.trials,
                "corrupt_ratios": args.corrupt_ratios,
                "ws_rewire_probability": args.ws_rewire_probability,
                "tractable_core_size": args.tractable_core_size,
                "seed": args.seed,
                "topologies": list(TOPOLOGIES),
                "er_model": "G(n,m) conditioned on connectivity",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_core_size_distributions(distributions, output_files["core_figure"])
    plot_attack_feasibility(summary, output_files["feasibility_figure"])
    write_report(
        output_files["report"],
        args,
        graph_summary,
        core_overview,
    )

    print(f"Completed {len(trials):,} topology/partition evaluations.")
    print(f"Recorded {len(cores):,} cardinality-eligible core subgraphs.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batch mHSSP / mHLCP lattice attack on synthetic core sub-graphs.

This single entry point replaces the five near-duplicate ``dfl_attack_batch*``
scripts of the original repository and reproduces every synthetic-topology
table of the paper:

    Table 1   mHLCP recall / candidate count      --problem mhlcp  --summary-only
    Table 5   directed push-sum mHLCP             --graph pushsum
    Tables 7-8   per-topology mHLCP + Cases 1-3   --problem mhlcp
    Table 9   mHSSP recall / candidate count      --problem mhssp  --summary-only
    Tables 10-11 per-topology mHSSP + Cases 2-3   --problem mhssp

Protocol variants
-----------------
``--problem mhlcp``  Metropolis-Hastings weights (non-uniform, doubly
                     stochastic).  The hidden coefficients are bounded by
                     ``B = max(W_ch_int) + 1``, which is handed to Step 2.
``--problem mhssp``  One shared edge weight ``1 / (d_max + 1)`` (uniform,
                     doubly stochastic), i.e. binary hidden coefficients.
                     Case 1 is not applicable here: with known neighbour
                     identities and uniform weights nothing stays hidden.
``--graph pushsum``  Strongly connected digraph with a column-stochastic
                     push-sum matrix.  Every node carries an auxiliary
                     mass-balance scalar initialised to 1, which is used as a
                     secondary filter on the surviving candidate matrices.

Each trial follows Algorithm 1 of the paper:

    Phase 1  reject topologies whose reduction does not yield one single core
             sub-graph containing all honest nodes
    Phase 2  Step 1 (orthogonal lattice / LLL) + Step 2 (Nguyen-Stern / BKZ)
    Phase 3  Cases 1-3 structural filtering
    Phase 4  (downstream state inversion lives in ``real_data_attack.py``)

Determinism
-----------
Topology ``k`` is generated from ``random.Random(k * graph_retries + attempt)``
and its models from ``numpy.random.seed(k * 10000 + trial + 7)``, exactly as in
the original scripts.  ``--graph-retries`` therefore participates in the seed:
changing it changes the drawn topologies.

Requires SageMath.  Run with ``sage -python`` or inside an activated Sage
environment.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from sage.all import ZZ, matrix  # noqa: E402

from experiments._common import resolve_output_dir, write_run_config  # noqa: E402
from hssp_dfl.attacks.step1 import step1_original  # noqa: E402
from hssp_dfl.dfl import (  # noqa: E402
    recover_candidates,
    sage_to_numpy,
    topology_side_information,
)
from hssp_dfl.filter import filter_info, filter_info_directed  # noqa: E402
from hssp_dfl.graph import (  # noqa: E402
    doubly_stochastic_matrix,
    doubly_stochastic_matrix_same_weight,
    scale_up_matrix,
)
from hssp_dfl.utils import gen_pseudoprime  # noqa: E402


FIELDNAMES = [
    "topology", "trial", "n_honest", "n_corrupt", "graph_attempts",
    "scale_factor", "max_B", "found_nonempty", "found_num_vectors",
    "true_weights_all_found", "true_weights_matched",
    "case1_num_solutions", "case2_num_solutions", "case3_num_solutions",
    "case1_validated_by_mass", "step1_time", "step2_time",
]


# ---------------------------------------------------------------------------
# Timeout for the (occasionally very slow) BKZ enumeration in Step 2
# ---------------------------------------------------------------------------
class AttackTimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise AttackTimeoutError("Step 2 timed out")


signal.signal(signal.SIGALRM, _alarm_handler)


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------
def fast_connected_graph(num_nodes, num_edges, rng):
    """Random connected undirected graph: spanning tree + random extra edges."""
    graph = nx.random_labeled_tree(num_nodes, seed=rng)
    needed = num_edges - graph.number_of_edges()
    non_edges = list(nx.non_edges(graph))
    if needed > 0 and len(non_edges) >= needed:
        graph.add_edges_from(rng.sample(non_edges, needed))
    return graph


def fast_strongly_connected_digraph(num_nodes, num_edges, rng):
    """Random strongly connected digraph: Hamiltonian cycle + random arcs."""
    nodes = list(range(num_nodes))
    perm = list(nodes)
    rng.shuffle(perm)

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for i in range(num_nodes):
        graph.add_edge(perm[i], perm[(i + 1) % num_nodes])

    needed = num_edges - graph.number_of_edges()
    if needed > 0:
        possible = [(u, v) for u in nodes for v in nodes
                    if u != v and not graph.has_edge(u, v)]
        if len(possible) >= needed:
            graph.add_edges_from(rng.sample(possible, needed))
    return graph


def pushsum_weight_matrix(graph):
    """Column-stochastic push-sum matrix ``w_ij = 1 / (d_out(j) + 1)``."""
    nodes = sorted(graph.nodes())
    size = len(nodes)
    weights = np.full((size, size), Fraction(0), dtype=object)
    for j in nodes:
        successors = list(graph.successors(j))
        weight = Fraction(1, len(successors) + 1)
        weights[j, j] = weight
        for i in successors:
            weights[i, j] = weight
    return weights


def build_weight_matrix(graph, problem, is_pushsum):
    if is_pushsum:
        return pushsum_weight_matrix(graph)
    if problem == "mhssp":
        return doubly_stochastic_matrix_same_weight(graph)
    return doubly_stochastic_matrix(graph)


def core_components(graph, honest_set, corrupt_set, is_pushsum):
    """Corrupt-honest bipartite components with ``|V_c| >= |V_h| >= 2``."""
    bipartite = nx.Graph()
    bipartite.add_nodes_from(graph.nodes())
    if is_pushsum:
        # A corrupted node observes an honest node only along h -> c arcs.
        for h in honest_set:
            for c in graph.successors(h):
                if c in corrupt_set:
                    bipartite.add_edge(h, c)
    else:
        for u, v in graph.edges():
            if (u in honest_set) != (v in honest_set):
                bipartite.add_edge(u, v)

    cores = []
    for component in nx.connected_components(bipartite):
        h = sorted(n for n in component if n in honest_set)
        c = sorted(n for n in component if n in corrupt_set)
        if len(c) >= len(h) >= 2:
            cores.append({"honest_nodes": h, "corrupt_nodes": c,
                          "num_honest": len(h), "num_corrupt": len(c)})
    return cores


def generate_valid_topology(topo_idx, args):
    """Draw a topology whose reduction yields exactly one all-honest core."""
    is_pushsum = args.graph == "pushsum"
    nodes = list(range(args.nodes))
    n_corrupt = int(args.corrupt_ratio * args.nodes)
    n_honest = args.nodes - n_corrupt

    for attempt in range(args.graph_retries):
        rng = random.Random(topo_idx * args.graph_retries + attempt)

        if is_pushsum:
            graph = fast_strongly_connected_digraph(args.nodes, args.edges, rng)
        else:
            graph = fast_connected_graph(args.nodes, args.edges, rng)

        perm_nodes = list(nodes)
        rng.shuffle(perm_nodes)
        honest_nodes = sorted(perm_nodes[:n_honest])
        corrupt_nodes = sorted(perm_nodes[n_honest:])

        cores = core_components(graph, set(honest_nodes), set(corrupt_nodes),
                                is_pushsum)
        if len(cores) != 1 or cores[0]["num_honest"] != n_honest:
            continue

        weights = build_weight_matrix(graph, args.problem, is_pushsum)
        scale_factor, weights_scaled = scale_up_matrix(weights)
        if scale_factor > args.max_scale_factor:
            continue
        return {
            "graph": graph,
            "W": weights,
            "W_scaled": weights_scaled,
            "scale_factor": scale_factor,
            "honest_nodes": honest_nodes,
            "corrupt_nodes": corrupt_nodes,
            "core": cores[0],
            "attempts": attempt + 1,
        }
    return None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def count_true_weights(found, w_ch_perm):
    """How many ground-truth columns of ``W_ch`` appear in the candidate set."""
    matched = 0
    for column in w_ch_perm.T:
        if any(np.array_equal(column, vec) or np.array_equal(-column, vec)
               for vec in found):
            matched += 1
    return matched == w_ch_perm.shape[1], matched


def validate_by_mass_scalar(candidates, perm, y_extended, atol=1e-6):
    """Keep candidate matrices whose recovered push-sum scalars equal 1."""
    passed = []
    inverse_perm = np.argsort(perm)
    for candidate in candidates:
        mat = candidate if isinstance(candidate, np.ndarray) else sage_to_numpy(candidate)
        mat_t = mat.T[inverse_perm, :]
        recovered = np.linalg.pinv(mat_t) @ y_extended
        if np.allclose(recovered[:, -1], 1.0, atol=atol):
            passed.append(candidate)
    return passed


# ---------------------------------------------------------------------------
# One attack trial
# ---------------------------------------------------------------------------
def run_trial(topo_idx, trial_idx, topology, args):
    is_pushsum = args.graph == "pushsum"
    core = topology["core"]
    honest_nodes = topology["honest_nodes"]
    corrupt_nodes = topology["corrupt_nodes"]
    weights = topology["W"]
    scale_factor = topology["scale_factor"]
    n_honest = len(honest_nodes)

    w_cc = weights[np.ix_(core["corrupt_nodes"], corrupt_nodes)]
    w_ch = weights[np.ix_(core["corrupt_nodes"], core["honest_nodes"])]
    w_cc_int = np.array([[int(x * scale_factor) for x in row] for row in w_cc], dtype=int)
    w_ch_int = np.array([[int(x * scale_factor) for x in row] for row in w_ch], dtype=int)

    row = {name: None for name in FIELDNAMES}
    row.update(
        topology=topo_idx + 1,
        trial=trial_idx + 1,
        n_honest=n_honest,
        n_corrupt=len(corrupt_nodes),
        graph_attempts=topology["attempts"],
        scale_factor=int(scale_factor),
        max_B=int(np.max(w_ch_int) + 1),
        found_nonempty=False,
        found_num_vectors=0,
        true_weights_all_found=False,
        true_weights_matched=0,
    )

    # ---- fresh model states -------------------------------------------------
    x0 = gen_pseudoprime(args.prime_bits)
    np.random.seed(topo_idx * 10000 + trial_idx + 7)
    model_0 = np.random.randint(0, 101, size=(args.nodes, args.model_size))

    y_extended = None
    if is_pushsum:
        # Append the auxiliary mass-balance scalar a = 1 to every node state.
        model_0_ext = np.hstack([model_0, np.ones((args.nodes, 1), dtype=int)])
        model_1_ext = topology["W_scaled"] @ model_0_ext
        model_1 = model_1_ext[:, : args.model_size]
    else:
        model_1 = topology["W_scaled"] @ model_0

    # ---- pick a full-rank row permutation of W_ch ---------------------------
    perm = None
    for _ in range(args.max_permute):
        candidate = np.random.permutation(w_ch_int.shape[0])
        block = w_ch_int[candidate][:n_honest, :n_honest]
        if np.linalg.matrix_rank(block) == n_honest:
            perm = candidate
            break
    if perm is None:
        print("    no full-rank permutation found; skipping", flush=True)
        return row

    w_ch_perm = w_ch_int[perm, :]
    side_info = topology_side_information(matrix(ZZ, w_ch_perm))

    # ---- Step 1 + Step 2, with coordinate resampling on failure -------------
    found = None
    for resample in range(args.max_resample + 1):
        sample = sorted(np.random.choice(args.model_size, size=n_honest, replace=False))
        model_0_seg = model_0[np.ix_(corrupt_nodes, sample)]
        model_1_seg = model_1[np.ix_(core["corrupt_nodes"], sample)]
        observation = (model_1_seg - w_cc_int @ model_0_seg)[perm, :]
        observation_sage = matrix(ZZ, observation)

        started = time.time()
        kernel, _ = step1_original(n_honest, int(observation_sage.nrows()), x0,
                                   observation_sage)
        row["step1_time"] = round(time.time() - started, 6)

        bound = int(np.max(w_ch_int) + 1) if args.problem == "mhlcp" else 1
        signal.alarm(args.timeout)
        try:
            started = time.time()
            candidates = recover_candidates(n_honest,
                                            int(observation_sage.nrows()),
                                            kernel, bound)
            elapsed = time.time() - started
            signal.alarm(0)
            if candidates is not None:
                found = np.array(candidates)
                row["step2_time"] = round(elapsed, 6)
                print(f"    step 2 succeeded in {elapsed:.3f}s", flush=True)
                break
            print(f"    step 2 returned no candidates, resampling "
                  f"({resample + 1}/{args.max_resample})", flush=True)
        except AttackTimeoutError:
            signal.alarm(0)
            print(f"    step 2 timed out after {args.timeout}s, resampling "
                  f"({resample + 1}/{args.max_resample})", flush=True)
        except AssertionError as exc:
            signal.alarm(0)
            if "not enough vectors" in str(exc):
                print("    step 2 could not produce enough short vectors; "
                      "dropping trial", flush=True)
                return row
            raise

    if found is None:
        print("    no candidates after all resamples", flush=True)
        return row

    row["found_nonempty"] = True
    row["found_num_vectors"] = int(found.shape[0])
    all_found, matched = count_true_weights(found, w_ch_perm)
    row["true_weights_all_found"] = all_found
    row["true_weights_matched"] = matched
    print(f"    {found.shape[0]} candidate vectors, "
          f"ground truth {matched}/{n_honest} "
          f"({'ALL FOUND' if all_found else 'partial'})", flush=True)

    if not args.cases:
        return row

    # ---- Phase 3: Cases 1-3 structural filtering ---------------------------
    target_sum, x_binary, zero_counts = side_info

    if is_pushsum:
        y_extended = np.array(
            model_1_ext[np.ix_(core["corrupt_nodes"], range(args.model_size + 1))]
            - w_cc_int @ model_0_ext[np.ix_(corrupt_nodes, range(args.model_size + 1))],
            dtype=float,
        )
        # A directed topology is not row-stochastic, so the exact row-sum
        # identity of Case 3 does not hold; Table 5 reports Case 1 before and
        # after validating the reconstructed mass-balance scalars against their
        # known initial value of 1.
        try:
            solutions = filter_info_directed(found, number=n_honest,
                                             X_binary=x_binary,
                                             scale=scale_factor)
            row["case1_num_solutions"] = len(solutions)
            validated = validate_by_mass_scalar(solutions, perm, y_extended)
            row["case1_validated_by_mass"] = len(validated)
            print(f"    Case 1 (exact pattern): {len(solutions)} -> "
                  f"{len(validated)} after mass-scalar validation", flush=True)
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            row["case1_num_solutions"] = -1
            print(f"    Case 1 (exact pattern) error: {exc}", flush=True)
        return row

    specs = [
        ("case1_num_solutions", dict(X_binary=x_binary), "Case 1 (exact pattern)"),
        ("case2_num_solutions", dict(target_zero_counts=zero_counts), "Case 2 (zero count)"),
        ("case3_num_solutions", {}, "Case 3 (no structure)"),
    ]
    if args.problem == "mhssp":
        # Uniform weights + known neighbour identities leaves nothing hidden.
        specs = specs[1:]

    for key, kwargs, label in specs:
        try:
            solutions = filter_info(found, known_info=target_sum,
                                    number=n_honest, **kwargs)
            row[key] = len(solutions)
            print(f"    {label}: {len(solutions)} solution(s)", flush=True)
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            row[key] = -1
            print(f"    {label} error: {exc}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def summarize(rows, n_honest_expected):
    total = len(rows)
    if not total:
        return {}
    solved = [r for r in rows if r["found_nonempty"]]
    all_weights = [r for r in rows if r["true_weights_all_found"]]
    summary = {
        "trials": total,
        "recall_percent": round(100.0 * len(all_weights) / total, 1),
        "found_nonempty_percent": round(100.0 * len(solved) / total, 1),
        "avg_found_vectors": (round(float(np.mean([r["found_num_vectors"]
                                                  for r in solved])), 1)
                              if solved else None),
        "avg_step1_time": (round(float(np.mean([r["step1_time"] for r in solved
                                                if r["step1_time"] is not None])), 6)
                           if solved else None),
        "avg_step2_time": (round(float(np.mean([r["step2_time"] for r in solved
                                                if r["step2_time"] is not None])), 6)
                           if solved else None),
        "n_honest": n_honest_expected,
    }
    for case in ("case1", "case2", "case3"):
        values = [r[f"{case}_num_solutions"] for r in solved
                  if r[f"{case}_num_solutions"] not in (None, -1)]
        summary[f"avg_{case}_solutions"] = (round(float(np.mean(values)), 2)
                                            if values else None)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch mHSSP/mHLCP lattice attack on synthetic core sub-graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--problem", choices=("mhlcp", "mhssp"), default="mhlcp",
                        help="mhlcp: Metropolis weights; mhssp: uniform weights")
    parser.add_argument("--graph", choices=("undirected", "pushsum"),
                        default="undirected",
                        help="pushsum: directed column-stochastic protocol")
    parser.add_argument("--nodes", type=int, default=10, help="node count n")
    parser.add_argument("--edges", type=int, default=20, help="edge count e")
    parser.add_argument("--corrupt-ratio", type=float, default=0.6, help="eta")
    parser.add_argument("--topologies", type=int, default=100)
    parser.add_argument("--trials-per-topology", type=int, default=1)
    parser.add_argument("--model-size", type=int, default=100, help="u")
    parser.add_argument("--prime-bits", type=int, default=50,
                        help="bit length of the pseudo-prime modulus Q")
    parser.add_argument("--max-scale-factor", type=int, default=50,
                        help="reject topologies whose integer scaling beta exceeds this")
    parser.add_argument("--timeout", type=int, default=60, help="Step 2 timeout (s)")
    parser.add_argument("--max-resample", type=int, default=10,
                        help="coordinate resamples per trial (paper: 10 detailed, 1 summary)")
    parser.add_argument("--max-permute", type=int, default=100)
    parser.add_argument("--graph-retries", type=int, default=5000,
                        help="also part of the topology seed; keep at 5000 to match the paper")
    parser.add_argument("--cases", dest="cases", action="store_true", default=True,
                        help="run Phase 3 Cases 1-3 filtering (default)")
    parser.add_argument("--summary-only", dest="cases", action="store_false",
                        help="skip Cases 1-3 (Tables 1 and 9 only need recall)")
    parser.add_argument("--output-dir", default=None,
                        help="default: results/lattice_attack_batch")
    parser.add_argument("--tag", default=None, help="override the output file stem")
    args = parser.parse_args()

    n_corrupt = int(args.corrupt_ratio * args.nodes)
    n_honest = args.nodes - n_corrupt
    out = resolve_output_dir(args.output_dir, "lattice_attack_batch")
    prefix = "directed_pushsum" if args.graph == "pushsum" else "undirected"
    tag = args.tag or (f"{prefix}_{args.problem}_NODE{args.nodes}"
                       f"_EDGE{args.edges}_CORRUPT{n_corrupt}"
                       f"{'' if args.cases else '_recall'}")

    print(f"[lattice_attack_batch] {tag}")
    print(f"  problem={args.problem} graph={args.graph} "
          f"n={args.nodes} e={args.edges} eta={args.corrupt_ratio} "
          f"-> n_honest={n_honest} n_corrupt={n_corrupt}")
    print(f"  topologies={args.topologies} trials/topology={args.trials_per_topology} "
          f"cases={args.cases} Q={args.prime_bits}-bit timeout={args.timeout}s")

    rows = []
    for topo_idx in range(args.topologies):
        print(f"\n{'=' * 68}\nTopology {topo_idx + 1}/{args.topologies}\n{'=' * 68}",
              flush=True)
        topology = generate_valid_topology(topo_idx, args)
        if topology is None:
            print("  no valid topology within the retry budget; skipping", flush=True)
            continue
        print(f"  found after {topology['attempts']} attempts | "
              f"honest={topology['honest_nodes']} corrupt={topology['corrupt_nodes']} | "
              f"beta={topology['scale_factor']}", flush=True)

        for trial_idx in range(args.trials_per_topology):
            print(f"  --- trial {trial_idx + 1}/{args.trials_per_topology} ---",
                  flush=True)
            rows.append(run_trial(topo_idx, trial_idx, topology, args))

    csv_path = out / f"{tag}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, n_honest)
    summary_path = out / f"{tag}_summary.csv"
    if summary:
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)

    write_run_config(out / f"{tag}_run_config.json", vars(args))

    print(f"\n{'=' * 68}")
    print(f"{len(rows)} trials -> {csv_path}")
    if summary:
        print(f"summary                -> {summary_path}")
        print(f"  Recall (all GT vectors found): {summary['recall_percent']}%")
        print(f"  Avg. Found Vec.              : {summary['avg_found_vectors']}")
        print(f"  Avg. Step 1 / Step 2 time    : "
              f"{summary['avg_step1_time']}s / {summary['avg_step2_time']}s")
        for case in ("case1", "case2", "case3"):
            value = summary.get(f"avg_{case}_solutions")
            if value is not None:
                print(f"  Avg. |F| {case}                : {value}")


if __name__ == "__main__":
    main()

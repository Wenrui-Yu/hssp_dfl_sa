"""Generate the fixed DFL topology the real-dataset attacks run on.

Draws a connected graph, partitions the nodes into honest and corrupt sets, and
keeps only topologies whose reduction yields a single attackable core sub-graph
(``|V_c| >= |V_h| >= 2``, with every honest node inside it).  The paper uses
``n=10, e=20, eta=0.6``.

Keys written to the .mat file:

    A              adjacency matrix
    AA             oriented adjacency matrix (+1 / -1)
    W              integer weight matrix, W_frac * scale_factor
    scale_factor   beta, the LCM of all weight denominators
    honest_nodes   / corrupt_nodes   the global partition
    sg_honest      / sg_corrupt      the core sub-graph the attack targets

The topology is generated at ``assets/network.mat`` or supplied with external
assets. With the same seed, parameters and library versions, generation is
deterministic. Existing files are protected unless you pass --force.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from hssp_dfl import paths as _paths

MODEL_OUT = _paths.MODEL_DIR
DATASET_OUT = _paths.DATASET_DIR
NETWORK_MAT = str(_paths.NETWORK_MAT)
MODEL_OUT.mkdir(parents=True, exist_ok=True)
DATASET_OUT.mkdir(parents=True, exist_ok=True)


import argparse


import numpy as np
import random
import math
from fractions import Fraction
from functools import reduce

import networkx as nx
import scipy.io as scio


# =====================================================================
# Configuration -- the defaults are the paper's setting
# =====================================================================
_parser = argparse.ArgumentParser(
    description="Generate the fixed DFL topology used by the real-dataset attacks.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument("--nodes", type=int, default=10, help="node count n")
_parser.add_argument("--edges", type=int, default=20, help="edge count e")
_parser.add_argument("--corrupt-ratio", type=float, default=0.6, help="eta")
_parser.add_argument("--max-scale-factor", type=int, default=50)
_parser.add_argument("--graph-retries", type=int, default=5000)
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument("--output", default=NETWORK_MAT,
                     help="where to write the topology")
_parser.add_argument("--force", action="store_true",
                     help="overwrite an existing topology file")
ARGS = _parser.parse_args()

NUM_NODES        = ARGS.nodes
NUM_EDGES        = ARGS.edges
CORRUPT_RATIO    = ARGS.corrupt_ratio
MAX_SCALE_FACTOR = ARGS.max_scale_factor
MAX_GRAPH_RETRIES = ARGS.graph_retries
SEED             = ARGS.seed
NETWORK_MAT      = ARGS.output

n_corrupt = int(CORRUPT_RATIO * NUM_NODES)
n_honest  = NUM_NODES - n_corrupt


# =====================================================================
# Helper: fast connected graph generation
# =====================================================================
def _fast_connected_graph(num_nodes, num_edges, rng):
    G = nx.random_labeled_tree(num_nodes, seed=rng)
    edges_needed = num_edges - G.number_of_edges()
    ne = list(nx.non_edges(G))
    if edges_needed > 0 and len(ne) >= edges_needed:
        G.add_edges_from(rng.sample(ne, edges_needed))
    return G


# =====================================================================
# Metropolis-Hastings doubly stochastic matrix (Fraction-based)
# =====================================================================
def doubly_stochastic_matrix_mh(G):
    """w_{ij} = 1 / (1 + max(deg_i, deg_j)), diagonal = 1 - row_sum."""
    degree_dict = dict(G.degree())
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    W_frac = np.full((n, n), Fraction(0), dtype=object)
    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        w = Fraction(1, 1 + max(degree_dict[u], degree_dict[v]))
        W_frac[i][j] = w
        W_frac[j][i] = w

    for i in range(n):
        W_frac[i][i] = Fraction(1) - sum(W_frac[i][j] for j in range(n) if j != i)

    return W_frac


def scale_up_fraction_matrix(W_frac):
    """Compute scale_factor (LCM of denominators) and integer W."""
    n = W_frac.shape[0]
    denoms = []
    for i in range(n):
        for j in range(n):
            if isinstance(W_frac[i][j], Fraction):
                denoms.append(W_frac[i][j].denominator)
    scale_factor = int(reduce(math.lcm, denoms))

    W_int = np.zeros_like(W_frac, dtype=int)
    for i in range(n):
        for j in range(n):
            W_int[i][j] = int(W_frac[i][j] * scale_factor)

    return scale_factor, W_int


# =====================================================================
# Generate valid topology
# =====================================================================
def generate_valid_topology(seed):
    nodes = list(range(NUM_NODES))

    for attempt in range(MAX_GRAPH_RETRIES):
        rng = random.Random(seed * MAX_GRAPH_RETRIES + attempt)

        G = _fast_connected_graph(NUM_NODES, NUM_EDGES, rng)

        perm_nodes = list(nodes)
        rng.shuffle(perm_nodes)
        honest_nodes = sorted(perm_nodes[:n_honest])
        corrupt_nodes = sorted(perm_nodes[n_honest:])
        honest_set = set(honest_nodes)
        corrupt_set = set(corrupt_nodes)

        # Build bipartite subgraph (cross-type edges only)
        G_bip = nx.Graph()
        G_bip.add_nodes_from(nodes)
        for u, v in G.edges():
            if (u in honest_set) != (v in honest_set):
                G_bip.add_edge(u, v)

        filtered_subgraphs = []
        for comp in nx.connected_components(G_bip):
            h = sorted(n for n in comp if n in honest_set)
            c = sorted(n for n in comp if n in corrupt_set)
            if len(c) >= len(h) >= 2:
                filtered_subgraphs.append({
                    "honest_nodes": h, "corrupt_nodes": c,
                    "num_honest": len(h), "num_corrupt": len(c),
                })

        if len(filtered_subgraphs) == 1 and filtered_subgraphs[0]["num_honest"] == n_honest:
            sg = filtered_subgraphs[0]
            W_frac = doubly_stochastic_matrix_mh(G)
            scale_factor, W_int = scale_up_fraction_matrix(W_frac)
            if scale_factor > MAX_SCALE_FACTOR:
                continue
            return G, W_int, scale_factor, honest_nodes, corrupt_nodes, sg, attempt + 1

    return None


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    if _Path(NETWORK_MAT).exists() and not ARGS.force:
        raise SystemExit(
            f"{NETWORK_MAT} already exists.\n"
            f"Use the topology associated with your checkpoints; changing the "
            f"seed or graph parameters can make those checkpoints incompatible.\n"
            f"Pass --force to overwrite it, or --output to write elsewhere."
        )

    print(f"Generating network: NUM_NODES={NUM_NODES}, NUM_EDGES={NUM_EDGES}, "
          f"n_corrupt={n_corrupt}, n_honest={n_honest}")

    result = generate_valid_topology(SEED)
    if result is None:
        print("ERROR: Could not generate a valid topology within "
              f"{MAX_GRAPH_RETRIES} attempts.")
        raise SystemExit(1)

    G, W, scale_factor, honest_nodes, corrupt_nodes, sg_info, n_attempts = result

    # Build A and AA from graph
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=int)
    AA = np.zeros((NUM_NODES, NUM_NODES), dtype=int)
    for u, v in G.edges():
        i, j = min(u, v), max(u, v)
        A[i][j] = 1
        A[j][i] = 1
        AA[i][j] = 1
        AA[j][i] = -1

    print(f"Found valid topology after {n_attempts} attempts")
    print(f"  Honest nodes:  {honest_nodes}")
    print(f"  Corrupt nodes: {corrupt_nodes}")
    print(f"  scale_factor:  {scale_factor}")
    print(f"  Subgraph honest:  {sg_info['honest_nodes']}")
    print(f"  Subgraph corrupt: {sg_info['corrupt_nodes']}")
    print(f"\nW (scaled integers):\n{W}")

    scio.savemat(NETWORK_MAT, {
        "A":             A,
        "AA":            AA,
        "W":             W,
        "scale_factor":  scale_factor,
        "honest_nodes":  np.array(honest_nodes),
        "corrupt_nodes": np.array(corrupt_nodes),
        "sg_honest":     np.array(sg_info["honest_nodes"]),
        "sg_corrupt":    np.array(sg_info["corrupt_nodes"]),
    })
    print(f"\nSaved {NETWORK_MAT}")

"""Network-topology helpers used by the DFL paper experiments."""

import random
from fractions import Fraction
from functools import reduce
from math import lcm as math_lcm

import networkx as nx
import numpy as np

try:
    from sage.all import Graph, Integer, lcm
except ImportError:  # Pure graph experiments do not require SageMath.
    Graph = None
    Integer = int
    lcm = None


def generate_connected_graph(num_nodes, num_edges, seed=None):
    """Generate a random connected undirected graph."""
    minimum_edges = num_nodes - 1
    maximum_edges = num_nodes * (num_nodes - 1) / 2
    if num_edges < minimum_edges:
        raise ValueError("num_edges must be >= num_nodes - 1 for connectivity.")
    if num_edges > maximum_edges:
        raise ValueError("num_edges exceeds maximum for undirected graph.")

    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    graph = nx.random_labeled_tree(num_nodes)
    while graph.number_of_edges() < num_edges:
        source, target = random.sample(range(num_nodes), 2)
        if not graph.has_edge(source, target):
            graph.add_edge(source, target)
    return graph


def _adjacency_and_degrees(graph):
    nodes = list(graph.nodes())
    adjacency = nx.adjacency_matrix(graph, nodelist=nodes).todense()
    return nodes, np.asarray(adjacency, dtype=float), dict(graph.degree())


def doubly_stochastic_matrix(graph):
    """Build exact Metropolis-Hastings weights for an undirected graph."""
    nodes, adjacency, degrees = _adjacency_and_degrees(graph)
    size = adjacency.shape[0]
    weights = np.zeros_like(adjacency, dtype=object)

    for i in range(size):
        for j in range(size):
            if i != j and adjacency[i, j] != 0:
                maximum_degree = max(degrees[nodes[i]], degrees[nodes[j]])
                weights[i, j] = Fraction(1, int(Integer(maximum_degree) + 1))

    for i in range(size):
        weights[i, i] = Fraction(1) - sum(
            weights[i, j] for j in range(size) if i != j
        )
    return weights


def doubly_stochastic_matrix_same_weight(graph):
    """Build doubly stochastic weights with one shared edge weight."""
    _nodes, adjacency, degrees = _adjacency_and_degrees(graph)
    size = adjacency.shape[0]
    weights = np.zeros_like(adjacency, dtype=object)
    maximum_degree = max(degrees.values())

    for i in range(size):
        for j in range(size):
            if i != j and adjacency[i, j] != 0:
                weights[i, j] = Fraction(1, int(Integer(maximum_degree) + 1))

    for i in range(size):
        weights[i, i] = Fraction(1) - sum(
            weights[i, j] for j in range(size) if i != j
        )
    return weights


def scale_up_matrix(weights):
    """Scale a matrix of fractions to an integer matrix."""
    denominators = [
        value.denominator
        for row in weights
        for value in row
        if isinstance(value, Fraction)
    ]
    if not denominators:
        return 1, np.asarray(weights, dtype=int)

    reducer = lcm if lcm is not None else math_lcm
    scale_factor = int(reduce(reducer, denominators))
    scaled = np.zeros_like(weights, dtype=int)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            scaled[i, j] = int(weights[i, j] * scale_factor)
    return scale_factor, scaled


def partition_nodes(num_nodes, n_corrupt, seed=None):
    """Randomly partition node identifiers into corrupt and honest sets."""
    if seed is not None:
        random.seed(int(seed))
    nodes = list(range(num_nodes))
    corrupt = sorted(random.sample(nodes, n_corrupt))
    honest = sorted(set(nodes) - set(corrupt))
    return corrupt, honest


def build_bipartite_subgraph(weighted_graph, corrupt_nodes, honest_nodes):
    """Keep only corrupt-to-honest edges in a Sage or NetworkX graph."""
    corrupt_set = set(corrupt_nodes)
    honest_set = set(honest_nodes)
    nodes = sorted(corrupt_set | honest_set)

    if Graph is not None and hasattr(weighted_graph, "connected_components"):
        result = Graph()
        result.add_vertices(nodes)
        for source, target, weight in weighted_graph.edges():
            if source in corrupt_set and target in corrupt_set:
                continue
            if source in honest_set and target in honest_set:
                continue
            result.add_edge(source, target, weight)
        return result

    result = nx.Graph()
    result.add_nodes_from(nodes)
    for source, target, data in weighted_graph.edges(data=True):
        if source in corrupt_set and target in corrupt_set:
            continue
        if source in honest_set and target in honest_set:
            continue
        result.add_edge(source, target, **data)
    return result


def find_attackable_subgraphs(
    bipartite_graph,
    corrupt_nodes,
    honest_nodes,
    min_honest=2,
):
    """Return components satisfying ``#corrupt >= #honest``."""
    corrupt_set = set(corrupt_nodes)
    honest_set = set(honest_nodes)
    if hasattr(bipartite_graph, "connected_components"):
        components = bipartite_graph.connected_components()
    else:
        components = nx.connected_components(bipartite_graph)

    results = []
    for index, component_nodes in enumerate(components):
        honest = sorted(node for node in component_nodes if node in honest_set)
        corrupt = sorted(node for node in component_nodes if node in corrupt_set)
        if len(corrupt) >= len(honest) and len(honest) >= min_honest:
            results.append(
                {
                    "subgraph_index": index,
                    "honest_nodes": honest,
                    "corrupt_nodes": corrupt,
                    "num_honest": len(honest),
                    "num_corrupt": len(corrupt),
                }
            )
    return results


def generate_strongly_connected_directed_graph(num_nodes, num_edges, seed=None):
    """Generate a random strongly connected directed graph."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    while True:
        graph = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
        if nx.is_strongly_connected(graph):
            return graph


def directed_doubly_stochastic_approx(graph):
    """Return the row-normalized directed adjacency matrix used in the paper."""
    adjacency = nx.to_numpy_array(graph)
    row_sums = adjacency.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return adjacency / row_sums[:, np.newaxis]


__all__ = [
    "build_bipartite_subgraph",
    "directed_doubly_stochastic_approx",
    "doubly_stochastic_matrix",
    "doubly_stochastic_matrix_same_weight",
    "find_attackable_subgraphs",
    "generate_connected_graph",
    "generate_strongly_connected_directed_graph",
    "partition_nodes",
    "scale_up_matrix",
]

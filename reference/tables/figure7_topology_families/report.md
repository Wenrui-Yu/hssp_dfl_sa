# ER representativeness across structured DFL topologies

## Design

- 100 graph/partition trials per configuration; master seed `20260727`.
- Nodes: `[100, 200]`; mean degrees: `[4]`; corruption ratios: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`.
- ER uses fixed-edge G(n,m) conditioned on connectivity. Ring, Watts-Strogatz, and Barabasi-Albert graphs use the same n and m.
- Corrupt-node partitions are paired across topology families and nested across corruption ratios.
- A comparison is practically equivalent only when its paired 95% bootstrap CI lies wholly inside ±0.050.

## Overall equivalence coverage

| Topology | Mean equivalence rate | Mean absolute ER gap | Worst ER gap |
|---|---:|---:|---:|
| Ring | 58.3% | 0.104 | 0.707 |
| Scale-free | 62.0% | 0.076 | 0.653 |
| Small-world | 64.8% | 0.059 | 0.526 |

Equivalence coverage is descriptive evidence about the ER baseline's transportability under the declared margin. It is not evidence that all topology families have the same risk.

## Largest paired deviations from ER

| Topology | n | degree | eta | Metric | Alternative | ER | Difference | 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Ring | 200 | 4 | 0.8 | `hssp_tractable_attackable_rate` | 0.947 | 0.241 | +0.707 | [+0.676, +0.736] |
| Ring | 100 | 4 | 0.8 | `hssp_tractable_attackable_rate` | 0.952 | 0.289 | +0.663 | [+0.599, +0.728] |
| Scale-free | 200 | 4 | 0.7 | `hssp_rank_attackable_rate` | 0.106 | 0.758 | -0.653 | [-0.728, -0.571] |
| Scale-free | 200 | 4 | 0.7 | `hlcp_rank_attackable_rate` | 0.106 | 0.758 | -0.653 | [-0.725, -0.572] |
| Ring | 200 | 4 | 0.7 | `max_core_fraction` | 0.190 | 0.828 | -0.638 | [-0.653, -0.621] |
| Ring | 100 | 4 | 0.7 | `hssp_tractable_attackable_rate` | 0.754 | 0.120 | +0.634 | [+0.589, +0.680] |
| Ring | 200 | 4 | 0.6 | `max_core_fraction` | 0.221 | 0.834 | -0.613 | [-0.631, -0.595] |
| Ring | 200 | 4 | 0.7 | `hssp_tractable_attackable_rate` | 0.711 | 0.118 | +0.594 | [+0.561, +0.626] |
| Ring | 200 | 4 | 0.8 | `max_core_fraction` | 0.166 | 0.729 | -0.562 | [-0.590, -0.536] |
| Ring | 100 | 4 | 0.7 | `max_core_fraction` | 0.302 | 0.837 | -0.535 | [-0.559, -0.508] |

## Interpretation boundary

The cardinality metric reproduces the paper's component-level condition. Full-column rank adds a stricter structural check. The tractability proxy counts full-rank cores with at most 10 honest nodes. None of these proxies measures end-to-end lattice recall, candidate-set ambiguity, or gradient-inversion quality; those require the attack scripts.

## Pooled eligible-core scale

The following conditional summary pools cardinality-eligible cores across corruption ratios. `tractable_core_fraction` is the fraction with at most 10 honest nodes.

|   num_nodes |   mean_degree | topology    |   eligible_cores |   median_core_honest_size |   p90_core_honest_size |   tractable_core_fraction |   hssp_rank_feasible_fraction |   hlcp_rank_feasible_fraction |
|------------:|--------------:|:------------|-----------------:|--------------------------:|-----------------------:|--------------------------:|------------------------------:|------------------------------:|
|         100 |             4 | er          |              721 |                    10.000 |                 37.000 |                     0.510 |                         0.806 |                         0.811 |
|         100 |             4 | ring        |             2571 |                     3.000 |                  9.000 |                     0.925 |                         0.806 |                         0.806 |
|         100 |             4 | scale_free  |              656 |                    15.000 |                 36.000 |                     0.473 |                         0.602 |                         0.605 |
|         100 |             4 | small_world |             1612 |                     3.000 |                 19.000 |                     0.824 |                         0.825 |                         0.828 |
|         200 |             4 | er          |             1080 |                     3.000 |                 67.000 |                     0.628 |                         0.842 |                         0.843 |
|         200 |             4 | ring        |             5028 |                     3.000 |                 10.000 |                     0.918 |                         0.806 |                         0.806 |
|         200 |             4 | scale_free  |              760 |                    16.000 |                 70.000 |                     0.430 |                         0.583 |                         0.583 |
|         200 |             4 | small_world |             2940 |                     3.000 |                 17.000 |                     0.859 |                         0.841 |                         0.845 |

## Mean topology descriptors

|   num_nodes |   mean_degree | topology    |   num_edges |   degree_std |   average_clustering |   average_shortest_path |
|------------:|--------------:|:------------|------------:|-------------:|---------------------:|------------------------:|
|         100 |             4 | er          |     200.000 |        1.871 |                0.035 |                   3.453 |
|         100 |             4 | ring        |     200.000 |        0.000 |                0.500 |                  12.879 |
|         100 |             4 | scale_free  |     200.000 |        3.705 |                0.122 |                   3.005 |
|         100 |             4 | small_world |     200.000 |        0.615 |                0.379 |                   5.061 |
|         200 |             4 | er          |     400.000 |        1.872 |                0.019 |                   3.990 |
|         200 |             4 | ring        |     400.000 |        0.000 |                0.500 |                  25.377 |
|         200 |             4 | scale_free  |     400.000 |        4.283 |                0.085 |                   3.335 |
|         200 |             4 | small_world |     400.000 |        0.622 |                0.374 |                   6.169 |

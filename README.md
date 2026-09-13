# When Topology Betrays Privacy — Artifact

Reproduction package for

> **When Topology Betrays Privacy: Lattice-Based Reconstruction Attacks on Secure Aggregation in Decentralized Federated Learning**

The paper shows that neighbourhood-based Secure Aggregation in Decentralized
Federated Learning leaks: sparse topologies give colluding semi-honest nodes
asymmetric views of the aggregate, which reduces to a **multi-dimensional Hidden
Subset Sum Problem (mHSSP)** — or its generalisation, the **Hidden Linear
Combination Problem (mHLCP)** — and is solvable by lattice reduction.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Paper result → command](#2-paper-result--command)
3. [Repository layout](#3-repository-layout)
4. [Producing the assets from scratch](#4-producing-the-assets-from-scratch)

---

## 1. Quick start

### 1.1 Environment

The lattice attack needs **SageMath** (for LLL/BKZ); the data-reconstruction
stage needs **PyTorch**. Both must live in the *same* interpreter, because the
attack scripts call them in one process.

```bash
conda env create -f environment.yml
conda activate hssp-dfl
python -m pip install -r requirements.txt
```

Verify:

```bash
python -c "import sage.all, torch, networkx, matplotlib; print('environment ok')"
python -m hssp_dfl.paths
```

Full details, including how to reuse an existing Sage installation, are in
[`INSTALL.md`](INSTALL.md).

### 1.2 Assets

The topology experiments (Figures 2, 6, 7, 8) and the synthetic lattice attacks
(Tables 1, 5, 7–11) are self-contained.

The repository includes the base assets for the real-dataset attacks: the fixed
`assets/network.mat`, nine CIFAR-10/Purchase-100/Sentiment140 checkpoints
(`t0`, `t0.5`, `t1`) in `assets/models/`, and the corresponding per-node data
and ground-truth tweets in `assets/datasets/`. No separate download or asset
import is needed for these batch-size-1 attacks.

The Sentiment140 embedding cache is included at
`assets/data/sentiment140/embeddings_ada002.pt`. It corresponds to the default
100 samples selected by the training script with Python shuffle seed 0 from
the original 1.6-million-row CSV. Retraining still needs that CSV; text inversion
still needs the embedding API. The base assets occupy approximately 325 MB.

Noise-experiment checkpoints, full raw datasets, and Table 6 snapshots are not
included. To import additional assets from an existing source repository:

```bash
bash scripts/setup_assets.sh --from /path/to/source_repo   # symlink (default)
bash scripts/setup_assets.sh --from /path/to/source_repo --copy
bash scripts/setup_assets.sh --check                       # inventory
```

The source must contain `network.mat`, `models/`, and the per-node dataset
pickles at its root; `models_dp/` and `data/` supply the optional DP checkpoints
and raw datasets. The setup script populates this checkout's `assets/` tree.
For inputs stored elsewhere, use the environment variables in
[Where things go](#where-things-go). See
[Section 4](#4-producing-the-assets-from-scratch) to generate additional assets.
Importing replaces same-name destination files. `--clean` deletes the contents
of `assets/`, including copied files, and is not a required setup step.

---

## 2. Paper result → command

All commands are run from the repository root. Outputs land under `results/`.

### Topological analysis — no SageMath required

| Paper | Command | Output |
|---|---|---|
| **Figure 2** — honest-node recoverability vs. η, undirected `(n,e)=(20,40),(20,60)` | `python reproduce.py figure2` | `results/topology_recovery/recovery_undirected_node20_edge{40,60}.{csv,png}` |
| **Figure 6** — core sub-graph size distribution, `n=100/200` | `python reproduce.py figure6` | `results/core_subgraph_scale/core_size_node{100,200}_edge*.{csv,png}` |
| **Figure 7** — core size across ER / ring / small-world / scale-free | `python reproduce.py figure7` | `results/topology_families/{core_size_distribution.png, attack_feasibility.png, report.md}` |
| **Figure 8** — recoverability vs. η, directed `e=40,60,80,100` | `python reproduce.py figure8` | `results/topology_recovery/recovery_directed_node20_edge*.{csv,png}` |

Behind `reproduce.py` these are single scripts you can drive directly:

```bash
python experiments/topology_recovery.py --graph undirected --nodes 20 --edges 40 --trials 100
python experiments/core_subgraph_scale.py --nodes 200 --edges 300 --trials 100
python experiments/topology_families.py --nodes 100 200 --trials 100
```

### Lattice attack on core sub-graphs — needs SageMath

One script, [`experiments/lattice_attack_batch.py`](experiments/lattice_attack_batch.py),
produces every synthetic table. `--problem mhlcp` uses Metropolis–Hastings
(non-uniform) mixing weights; `--problem mhssp` uses the single shared edge
weight `1/(d_max+1)`; `--graph pushsum` switches to a directed column-stochastic
push-sum protocol with the auxiliary mass-balance scalar as a secondary filter.

| Paper | Command |
|---|---|
| **Table 1** — mHLCP recall / candidate count, 5 configurations | `python reproduce.py table1` |
| **Table 5** — directed push-sum mHLCP, `n=10, e=30, η=0.7` | `python reproduce.py table5` |
| **Table 7** — per-topology mHLCP + Cases 1–3, `η=0.6` | `python reproduce.py table7` |
| **Table 8** — per-topology mHLCP + Cases 1–3, `η=0.7` | `python reproduce.py table8` |
| **Table 9** — mHSSP recall / candidate count, 12 configurations | `python reproduce.py table9` |
| **Table 10** — per-topology mHSSP + Cases 2–3, `η=0.6` | `python reproduce.py table10` |
| **Table 11** — per-topology mHSSP + Cases 2–3, `η=0.7` | `python reproduce.py table11` |

Examples of individual configurations (these do not generate an entire table):

```bash
# Table 7: ten topologies; use --corrupt-ratio 0.7 for Table 8
python experiments/lattice_attack_batch.py --problem mhlcp \
    --nodes 10 --edges 20 --corrupt-ratio 0.6 --topologies 10

# One Table 1 configuration: mHLCP, 100 topologies, no filtering
python experiments/lattice_attack_batch.py --problem mhlcp \
    --nodes 10 --edges 20 --corrupt-ratio 0.6 --topologies 100 \
    --max-resample 1 --summary-only

# One Table 9 configuration: mHSSP, 100 topologies, no filtering
python experiments/lattice_attack_batch.py --problem mhssp \
    --nodes 20 --edges 40 --corrupt-ratio 0.8 --topologies 100 \
    --max-resample 1 --summary-only
python figures/make_summary_tables.py          # -> table1_and_table9_summary.tex

# Table 5: directed push-sum
python experiments/lattice_attack_batch.py --graph pushsum --problem mhlcp \
    --nodes 10 --edges 30 --corrupt-ratio 0.7 --topologies 5
```

### Reconstruction from real DFL checkpoints — needs SageMath, PyTorch and assets

All three run on the fixed topology in `assets/network.mat`
(`n=10, e=20, η=0.6`, mHLCP, batch size 1, 100-bit modulus).

| Paper | Command |
|---|---|
| **Figure 3** — CIFAR-10 ground truth vs. spurious candidates | `python reproduce.py figure3` |
| **Figure 4** — CIFAR-10 gradient MSE, SSIM and PSNR | `python reproduce.py figure4` |
| **Figure 9** — Purchase-100 gradient and feature MSE | `python reproduce.py figure9` |
| **Figure 10** — Sentiment140 embedding MSE, cosine and ROUGE-L | `python reproduce.py figure10` |
| **Figures 12–14 / 15–17** — all 30 candidates per case, CIFAR / Purchase | `python reproduce.py figures12-17` |
| **Table 2, Tables 12–14** — Sentiment140 text reconstructions | `python reproduce.py tables12-14` |

Direct invocations:

```bash
python experiments/attack_cifar.py        --candidates 30
python experiments/attack_purchase.py     --candidates 30
python experiments/attack_sentiment140.py --candidates 30
python experiments/vec2text_eval.py       # embeddings -> text, ROUGE-L
python experiments/add_bertscore.py       # append BERTScore
python figures/make_stat_figures.py --dataset all
python figures/make_figure3.py
python figures/make_case_grids.py
python figures/make_text_tables.py
```

Table 2 in the paper is a hand-picked excerpt of Table 12 (the Case 1 table),
showing one node whose embedding is recovered with cosine similarity 1.0.

`figure10` and `tables12-14` invoke `vec2text_eval.py`, which requires
`OPENAI_API_KEY` in the environment for hosted embedding calls. Recovering gradients and embeddings does not require
this API. Initial model and dataset downloads may still require network access.

To render the published text results without rerunning online inversion:

```bash
python figures/make_stat_figures.py --dataset sentiment140 \
    --sentiment-csv reference/tables/figure10_sentiment140_vec2text_metrics.csv
python figures/make_text_tables.py \
    --input reference/tables/table12-14_sentiment140_with_bertscore.csv
```

### Robustness and defenses

| Paper | Command | Notes |
|---|---|---|
| **Table 3** — finite-precision truncation, `γ∈{4,6,8,10}`, `r∈{1,2,4,8}` | `python reproduce.py table3` | writes `results/finite_precision/table3.tex` |
| **Figure 5** — accuracy vs. attack success under LDP / CDP | `python reproduce.py figure5` | needs the DP checkpoints; the per-epsilon accuracy comes from `training/train_cifar_dp.py`, or copy `reference/tables/fl_dp_cifar10_*.csv` into `results/dp_defense/` |
| **Figure 11** — exact vs. noise-tolerant Step 1 under DP | `python reproduce.py figure11` | needs the DP checkpoints |
| **Table 6** — downstream GIA transferability, batches 1–8 | `python reproduce.py table6` | needs `breaching`, see below |

Table 6 is the only experiment with a dependency outside the main environment.
Install it first, then run the pipeline:

```bash
python -m pip install -r experiments/gia_transfer/requirements.txt
python experiments/gia_transfer/run_pipeline.py        # resumable, stage-by-stage
```

See [`experiments/gia_transfer/run_pipeline.py`](experiments/gia_transfer/run_pipeline.py)
for the experiment configuration and resumable stages.


---

## 3. Repository layout


```
.
├── reproduce.py                   one entry point for every result
├── README.md  INSTALL.md          this file and the environment guide
├── requirements.txt  environment.yml
│
├── hssp_dfl/                      the library
│   ├── lattice.py                 orthogonal, modular and noisy augmented lattices
│   ├── attacks/step1.py           Step 1: orthogonal lattice + LLL
│   ├── attacks/noisy_step1.py     noise-tolerant Step 1 (Section 4.4)
│   ├── attacks/nguyen_stern.py    Step 2: Nguyen-Stern candidate recovery
│   ├── attacks/bkz.py             BKZ reduction and bounded-candidate extraction
│   ├── attacks/multivariate.py    Step 2, Appendix E: multivariate quadratic
│   ├── attacks/statistical.py     Step 2, Appendix E: ICA / distributional
│   ├── filter.py                  Cases 1-3 pre-filtering and MITM combination search
│   ├── dfl.py                     DFL candidate orientation and topology side information
│   ├── graph.py                   graph generation, mixing matrices, node partitions
│   ├── paths.py                   every input/output location, all env-overridable
│   ├── models.py  dlg.py  fedavg_node.py     model definitions and DFL/DLG helpers
│   ├── gia/                       Breaching adapter and image metrics
│   └── experiments/
│       ├── checkpoint_truncation_replay.py   the Table 3 engine
│       └── common.py  evaluation.py          its helpers
│
├── experiments/                   one script per paper experiment
│   ├── topology_recovery.py       Figures 2 and 8
│   ├── core_subgraph_scale.py     Figure 6
│   ├── topology_families.py       Figure 7
│   ├── lattice_attack_batch.py    Tables 1, 5, 7-11
│   ├── attack_cifar.py            Figures 3, 4, 12-14
│   ├── attack_purchase.py         Figures 9, 15-17
│   ├── attack_sentiment140.py     Figure 10, Tables 2, 12-14
│   ├── vec2text_eval.py           embedding -> text inversion and ROUGE-L
│   ├── add_bertscore.py           BERTScore for the text tables
│   ├── finite_precision.py        Table 3
│   ├── dp_defense.py              Figures 5 and 11
│   └── gia_transfer/              Table 6 (optional, needs breaching)
│
├── figures/                       plotting and LaTeX rendering only
│   ├── make_stat_figures.py       Figures 4, 9, 10
│   ├── make_figure3.py            Figure 3
│   ├── make_case_grids.py         Figures 12-17
│   ├── make_figure5.py            Figure 5
│   ├── make_figure11.py           Figure 11
│   ├── make_summary_tables.py     Tables 1 and 9
│   ├── make_table3_tex.py         Table 3
│   └── make_text_tables.py        Tables 12-14
│
├── training/                      producing the checkpoints the attack consumes
│   ├── generate_network.py        a topology satisfying the attack conditions
│   ├── train_cifar.py  train_purchase.py  train_sentiment140.py
│   └── train_cifar_dp.py          the LDP / CDP variants
│
├── scripts/
│   └── setup_assets.sh            link or copy the large inputs into assets/
│
├── reference/                     the paper's own numbers, for comparison
│   ├── tables/                    CSV and LaTeX for every table and metric figure
│   └── figures/                   the published PNGs, named by figure number
│
├── assets/                        large inputs (git-ignored, populated by you)
└── results/                       everything you produce (git-ignored)
```


### Where things go

Python entry points use `hssp_dfl/paths.py` for their default input/output
locations. Explicit command-line paths take precedence. Override defaults with:

| Variable | Default | Holds |
|---|---|---|
| `HSSP_ASSETS` | `assets/` | root for all inputs |
| `HSSP_MODEL_DIR` | `assets/models/` | DFL checkpoints |
| `HSSP_MODEL_DP_DIR` | `assets/models_dp/` | DP checkpoints |
| `HSSP_DATASET_DIR` | `assets/datasets/` | per-node dataset pickles |
| `HSSP_DATA_DIR` | `assets/data/` | raw CIFAR-10 / Purchase / Sentiment140 |
| `HSSP_NETWORK_MAT` | `assets/network.mat` | the fixed 10-node topology |
| `HSSP_RESULTS` | `results/` | all outputs |

```bash
python -m hssp_dfl.paths     # print the resolved paths and what exists
```

Some run configurations use repository-relative paths, while GIA manifests
record absolute checkpoint paths. Review generated artifacts before sharing
them if local storage paths should remain private.

---

## 4. Producing the assets from scratch

Only needed if you do not have the checkpoints. Under the default paths,
training writes ordinary checkpoints into `assets/models/`, noise-experiment
checkpoints into `assets/models_dp/`, and datasets into `assets/datasets/`.
Table 6 manages its own snapshots beneath `results/gia_transfer/`.

> **The training scripts refuse to overwrite existing checkpoints.** Use the
> topology associated with your checkpoints. Network generation is deterministic
> for a fixed seed (default 0), graph parameters and library versions; changing
> those settings can make existing checkpoints incompatible. Use a fresh
> `HSSP_ASSETS` directory for a new run, or `--force` to replace files deliberately.

```bash
# 1. a topology that satisfies the attack conditions (n=10, e=20, eta=0.6)
python training/generate_network.py --seed 0             # -> assets/network.mat

# 2. two communication rounds are enough: the attack needs t0 / t0.5 / t1
python training/train_cifar.py        --batch-size 1 --num-communications 2 --graph load
python training/train_purchase.py     --batch-size 1 --num-communications 2
python training/train_sentiment140.py --batch-size 1 --num-communications 2

# 3. Table 6: the pipeline trains nested batches 1, 2, 4, 8 itself
# Install experiments/gia_transfer/requirements.txt first.
python reproduce.py table6

# 4. Figures 5/11: checkpoints plus the 300-round accuracy logs for Figure 5
python training/train_cifar_dp.py --dp-mode none --dp-epsilon 0 --num-comm 300
for eps in 10 50 100 200 500 1000 10000; do
    python training/train_cifar_dp.py --dp-mode exchange --dp-epsilon "$eps" --num-comm 300
done
for eps in 50 100 200 500 1000 10000; do
    python training/train_cifar_dp.py --dp-mode aggregate --dp-epsilon "$eps" --num-comm 300
done

# Figure 11 extension: only early checkpoints are needed (no accuracy curve)
for eps in 20000 50000 100000 1000000 100000000 10000000000; do
    python training/train_cifar_dp.py --dp-mode aggregate --dp-epsilon "$eps" --num-comm 2
done
```

CIFAR-10 downloads itself into `assets/data/cifar10`. Purchase-100 expects
`assets/data/purchase/purchase100.npz`. Sentiment140 expects the raw CSV at
`assets/data/sentiment140/training.1600000.processed.noemoticon.csv`; encoding it
needs a key for the embedding service, but the cached embeddings at
`assets/data/sentiment140/embeddings_ada002.pt` let you skip that step.

The noise-training script now writes directly to `HSSP_MODEL_DP_DIR`
(default `assets/models_dp/`), which the attack also reads; no copy is needed.
For an older run whose noise checkpoints are in `models/`, set
`HSSP_MODEL_DP_DIR` to that directory when invoking the attack.
Use `--num-comm 2` for the main sweep only if you need attack checkpoints alone;
two-round accuracy logs do not reproduce Figure 5's final accuracy.
The 300-round setting matches the reference evaluation length, not a guarantee
of identical stochastic training results across environments.

---


## Citation

```bibtex
@article{yu2026topology,
  title         = {When Topology Betrays Privacy: Lattice-Based Reconstruction Attacks on Secure Aggregation in Decentralized Federated Learning},
  author        = {Wenrui Yu and Changlong Ji and Johannes Bjerva and Qiongxiu Li},
  year          = {2026},
  journal={arXiv preprint arXiv:2609.08476}
}
```


The lattice attack builds on the Nguyen–Stern orthogonal-lattice method for the Hidden Subset Sum Problem and the [solving_hssp](https://github.com/agnesegini/solving_hssp) codebase. The downstream data-reconstruction experiments draw on gradient-inversion methods, including those implemented in the [Breaching](https://github.com/JonasGeiping/breaching) framework.

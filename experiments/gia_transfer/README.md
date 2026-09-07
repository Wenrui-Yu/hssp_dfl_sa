# Table 6 — downstream GIA transferability

Paper result: **Table 6** (Appendix J).

- Scope: CIFAR-10 image reconstruction from one mean FedSGD gradient, or from a
  one-step SGD model delta recovered by the lattice attack.
- Threat model: honest-but-curious observer; no malicious model modification.
- Batch sizes 1, 2, 4 and 8 on honest nodes 2, 3, 6 and 9.
- Upstream implementation: [JonasGeiping/breaching](https://github.com/JonasGeiping/breaching),
  pinned to `faaab88e618dfa94d77b062ad50ac4a27afb1534`.

## Research question

After the topology/lattice stage exposes an honest client's local model update,
does gradient inversion remain effective when the update was computed from a
realistic mini-batch rather than a single example?

The topology stage is independent of how many examples produced one recovered
client update. This benchmark therefore isolates the downstream GIA boundary.
The `direct` source uses an exact batch gradient; the `snapshots` source converts
a recovered or ground-truth one-step SGD model delta back to a gradient.

## Methods

The experiment calls the upstream Breaching API; it does not copy or locally
reimplement these attacks:

- `deepleakage`: DLG baseline ([NeurIPS 2019 paper](https://papers.nips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html)).
- `invertinggradients`: cosine-similarity attack with image priors
  ([NeurIPS 2020 paper](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html)).
- `seethroughgradients`: GradInversion, designed for image-batch recovery
  ([CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.html)).

The `paper` preset uses batch sizes 8, 32, 64, and 128 with three independently
sampled batches. It uses the upstream method-specific iteration budgets rather
than shortening all methods to a common, potentially unfair budget. Labels are
known by default, matching the original repository's existing DLG experiment.
Because Breaching's stock DLG preset jointly optimizes unknown labels, the
adapter selects Breaching's data-only optimization class when labels are
provided while retaining the preset's upstream Euclidean objective and L-BFGS
optimizer.

The direct CIFAR loader reproduces `training/train_cifar.py`'s existing `raw_uint8 / 128 - 1`
normalization exactly; it does not silently switch the checkpoint to a different
preprocessing convention.

The stock See Through Gradients configuration is mismatched to the repository's
BatchNorm-free `cnn_cifar`: its DeepInversion term has no batch-statistics signal,
group regularization is unavailable in the pinned upstream implementation, and
its Langevin noise dominated the small-model reconstruction. A controlled
b=1/b=2 node-2 pilot recovered DLG-level quality by keeping the public attack
implementation while setting `optim.langevin_noise=0`, `objective.scale=0.01`,
and `init=patterned-4`. The exact pilot results and reproduction option are in
`results/gia_transfer/stg_tuning_b1n2/README.md`. The selected
setting was then frozen and completed the formal all-node paired run; the
aggregate results are written under `report_tuned_stg/`.

Scale-MIA is not included in the main comparison: it modifies the global model
sent to clients and consequently studies a malicious-server threat model rather
than a drop-in GIA back end for already recovered honest updates.

## Installation

Activate the existing Sage/PyTorch environment, then install the pinned optional
dependency:

```bash
sage -python -m pip install -r experiments/gia_transfer/requirements.txt
```

## Commands

### One-command paired three-attack table

The complete paper-table path is recorded in
`experiments/gia_transfer/run_pipeline.py`. From the repository root, one
command runs `training/train_cifar.py`, exports HSSP Case 1 solution 24, evaluates the exact and
recovered gradients through Breaching's public `deepleakage`,
`invertinggradients`, and tuned `seethroughgradients` attack presets, and writes
the paired table:

```bash
sage -python experiments/gia_transfer/run_pipeline.py
```

The fixed protocol is batch sizes 1, 2, 4, and 8; honest nodes 2, 3, 6, and 9;
training seed 0; attack seed 42; known labels; one restart; and CPU execution.
To update the original 300-iteration comparison without changing its DLG/IG
columns, Deep Leakage and Inverting Gradients reuse the existing 300-iteration
`breaching/` runs. See Through Gradients uses 2,000 iterations with Langevin
noise disabled, Euclidean objective scale 0.01, and patterned-4 initialization
under `stg_tuned2000/`. The true-update and recovered-solution runs use the same
budget and paired seed within each method. Existing validated FL snapshots,
exact gradients, HSSP solution-24 files, and completed Breaching runs are reused,
so the same command also resumes an interrupted experiment. A failed subprocess
is reported and is not retried automatically.

The final paper-ready files are:

- `results/gia_transfer/report_tuned_stg/paper_table.md`
- `results/gia_transfer/report_tuned_stg/paper_table.csv`
- `results/gia_transfer/report_tuned_stg/paper_table.tex`
- `results/gia_transfer/report_tuned_stg/gia_true_vs_recovered.png`
- `results/gia_transfer/three_gia_pipeline_manifest.json`

To rebuild the same three-line `booktabs` table directly from either the wide
aggregate CSV or the long method-summary CSV, use:

```bash
sage -python experiments/gia_transfer/export_table6_tex.py \
  --input results/gia_transfer/report_tuned_stg/paper_table.csv \
  --output results/gia_transfer/report_tuned_stg/paper_table_from_raw.tex
```

The exporter also accepts `batch_summary_long.csv`; it automatically pivots
the long rows into the same attack/source column layout.

For debugging or cluster scheduling, each resumable stage can be invoked with
`--stage fl`, `--stage hssp`, `--stage gia`, or `--stage report`. For example,
the table can be regenerated without rerunning attacks:

```bash
sage -python experiments/gia_transfer/run_pipeline.py --stage report
```

Here, `true update` is the named gradient captured by `training/train_cifar.py` immediately after
`loss.backward()` and before `optimizer.step()`. `Recovered solution (24)` is
the gradient obtained from the HSSP-recovered client state as
`(theta_before - theta_recovered_after) / 0.01`. Selecting solution 24 uses the
known correct Case 1 candidate for paired evaluation; it is an oracle alignment
device and is not claimed as an attacker-side candidate-selection rule.

Verify the full data/attack/output path with one upstream attack iteration:

```bash
sage -python experiments/gia_transfer/gia_large_batch.py \
  --preset smoke \
  --dry-run \
  --device cpu
```

Run a short GPU check:

```bash
sage -python experiments/gia_transfer/gia_large_batch.py \
  --preset quick \
  --device cuda
```

Run the registered main experiment:

```bash
sage -python experiments/gia_transfer/gia_large_batch.py \
  --preset paper \
  --device cuda
```

To generate one-step DFL snapshots for a particular larger batch:

```bash
sage -python fl.py \
  --batch-size 32 \
  --local-epochs 1 \
  --num-communications 2 \
  --gradient-output-dir results/true_gradients_b32
```

`--gradient-output-dir` stores each named parameter gradient immediately after
`loss.backward()` and before `optimizer.step()`. This is the `true update` input
used in the paired HSSP validation. For batch-size ablations, add
`--nested-batch-max-size 128` to each run so that every node uses nested examples
(`B8` is a prefix of `B32`, and so on) while keeping the initial model fixed.

Run Breaching directly on that saved gradient with:

```bash
sage -python experiments/gia_transfer/gia_large_batch.py \
  --source true-gradients \
  --checkpoint models/model_avg_ni32_N10_t0_z0_e1.pkl \
  --gradient-pickle results/true_gradients_b32/true_gradients_ni32_N10_t0_z0_e1.pkl \
  --dataset-pickle dataset_ni32_N10.pkl \
  --checkpoint-node 2 \
  --labels known \
  --device cuda
```

Then evaluate the model delta (replace `32` with the generated batch size):

```bash
sage -python experiments/gia_transfer/gia_large_batch.py \
  --source snapshots \
  --preset paper \
  --before-checkpoint models/model_avg_ni32_N10_t0_z0_e1.pkl \
  --after-checkpoint models/model_avg_ni32_N10_t0_5_z0_e1.pkl \
  --dataset-pickle dataset_ni32_N10.pkl \
  --checkpoint-node 0 \
  --device cuda
```

A lattice-recovered client state can be supplied as `--after-checkpoint` in the
same state-dict/list-of-state-dicts format. This makes the GIA adapter reusable
for exact-update and end-to-end reconstruction ablations.

The existing lattice script can export sampled topology candidates in that
format without running its legacy DLG back end. For a controlled downstream
validation, the following command exports only the Case 1 candidate with the
lowest ground-truth state MSE:

```bash
DFL_BATCH_SIZE=32 \
DFL_CIFAR_ATTACK_MODE=none \
DFL_EXPORT_RECOVERED_STATES=results/gia_recovered_states_b32 \
DFL_EXPORT_SELECTION=best \
DFL_EXPORT_CASES=case1_exact_pattern \
sage -python dfl_cifar_attack_stat.py
```

Use one exported `recovered_ni32_*.pkl` file as `--after-checkpoint` in the
snapshot command above. Keep the case, selected-candidate index, client node,
and lattice matched-MSE columns when joining the GIA result table. Selection by
`best` uses the true state and is therefore an oracle evaluation device, not an
attacker-side candidate-selection algorithm. The default
`DFL_EXPORT_SELECTION=first` behavior exports only the first sampled candidate
per case to avoid writing dozens of large checkpoints; set
`DFL_EXPORT_MAX_PER_CASE=0` only when every sampled candidate is required.

For the paired four-node comparison between exact gradients and recovered
solution 24 at batch sizes 1, 2, 4, and 8, regenerate the table from completed
runs with:

```bash
sage -python experiments/gia_transfer/run_pipeline.py --stage report
```

This produces image-level metrics, batch/method summaries, paired metric
differences, gradient-equivalence and nested-batch audits, a paper-ready LaTeX
table, and a comparison figure under `report_tuned_stg/`. LPIPS is computed with the
standard LPIPS-Alex network after the same permutation alignment used for MSE,
PSNR, and SSIM.

## Outputs and analysis rules

The default output directory is `results/gia_large_batch/`:

- `manifest.json`: resolved configuration, seed, before/after checkpoints,
  upstream commit, exported lattice metadata, label assumption, gradient source,
  and threat model.
- `summary.csv`: one row per attack/batch/trial with MSE, PSNR, SSIM, label
  accuracy, runtime, upstream objective, topology case, candidate index, and
  lattice matched MSE when available.
- `aggregate.csv`: latest-status mean/standard-deviation metrics and separate
  completed/failed counts for every attack/batch combination.
- `per_example.csv`: image-level metrics after batch alignment.
- `reconstructions/*`: ground-truth/reconstruction grids, tensors, and upstream
  attack statistics.

A mean batch gradient is invariant to example ordering. Before computing image
metrics, the script therefore performs Hungarian minimum-cost assignment on
pixel MSE. Report both central tendency and variability across the three sampled
batches. Retain failed/OOM rows in `summary.csv`, but report resource failures
separately from low-quality attack outcomes rather than assigning them an
artificial reconstruction score.

The script resumes completed combinations by default and refuses to mix a new
configuration into an existing result directory. Use a new `--output-dir` for a
deliberate rerun.

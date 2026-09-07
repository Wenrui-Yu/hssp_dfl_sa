"""CIFAR-10 image reconstruction after breaking DFL secure aggregation.

Paper results: Figure 3, Figure 4, Figures 12-14

Reproduces the "Attack on Real Dataset" pipeline of Section 7.4 on a fixed
10-node topology (n=10, e=20, eta=0.6, mHLCP, non-uniform mixing weights)
stored in ``assets/network.mat``:

    1. load the real DFL checkpoints saved at t0 / t0.5 / t1,
    2. build the corrupted nodes' observation Y and solve the mHLCP,
    3. filter the candidate set under Cases 1-3,
    4. invert every sampled candidate back to the honest client updates and
       run the downstream reconstruction.

Every command-line flag below defaults to the paper's setting, so a bare
invocation reproduces the published numbers.  Requires SageMath and PyTorch.
"""


from sage.all import *

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hssp_dfl import paths

import numpy as np
import random
import time
import threading
import pickle
import csv
import os
import copy
import collections
import math

import scipy.io as scio
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt

# ─── Local project imports ───────────────────────────────────────────
from hssp_dfl.models import cnn_cifar, fc, cnn
from hssp_dfl.dlg import leakage_from_gradients, label_to_onehot

# ─── Toolkit imports ─────────────────────────────────────────────────
from hssp_dfl.utils import gen_pseudoprime
from hssp_dfl.attacks.step1 import step1_original
from hssp_dfl.dfl import recover_candidates, sage_to_numpy, topology_side_information
from hssp_dfl.filter import filter_info


# =====================================================================
# Configuration -- every default is the paper's setting
# =====================================================================
_parser = argparse.ArgumentParser(
    description="CIFAR-10 mHLCP attack + gradient inversion (Figures 3, 4, 12-14).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument("--batch-size", type=int, default=1,
                     help="local SGD batch size |B_i| of the loaded checkpoints")
_parser.add_argument("--candidates", type=int, default=30,
                     help="candidate weight matrices sampled per case (paper: 30)")
_parser.add_argument("--attack-mode", choices=("dlg", "none"), default="dlg",
                     help="downstream inversion: dlg = the GIA framework of "
                          "Geiping et al.; none = lattice stage only, used by "
                          "the Table 6 pipeline")
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--timeout", type=int, default=10,
                     help="Step 2 timeout in seconds")
_parser.add_argument("--max-resample", type=int, default=5)
_parser.add_argument("--prime-bits", type=int, default=100,
                     help="bit length of the pseudo-prime modulus Q")
_parser.add_argument("--device", default="cpu")
_parser.add_argument("--output-dir", default=None,
                     help="default: results/real_data/cifar<candidates>[_b<batch>]")
_parser.add_argument("--export-recovered-states", default="",
                     help="dump recovered client updates here (used by the "
                          "large-batch GIA pipeline)")
_parser.add_argument("--export-max-per-case", type=int, default=1)
_parser.add_argument("--export-selection", choices=("first", "best"), default="first")
_parser.add_argument("--export-cases", default="",
                     help="comma-separated case names to export")
_parser.add_argument("--export-sample-ranks", default="",
                     help="comma-separated 1-based candidate ranks to export")
ARGS = _parser.parse_args()

SEED         = ARGS.seed
NUM_NODES    = 10
BATCH_SIZE   = ARGS.batch_size
TIMEOUT      = ARGS.timeout
MAX_RESAMPLE = ARGS.max_resample
MAX_PERMUTE  = 100
PRIME_BITS   = ARGS.prime_bits
QUANTIZE     = 1e10        # 10 decimal places, as in Section 7.5
LR           = 1e-2
CLASSES      = 10
DEVICE       = ARGS.device
DATASET      = "cifar10"
MODEL_ARCH   = "cnn"       # three conv layers + two FC layers (Appendix G)
ATTACK_MODE  = ARGS.attack_mode
CASE_SAMPLE_SIZE = ARGS.candidates

EXPORT_RECOVERED_DIR = ARGS.export_recovered_states
EXPORT_MAX_PER_CASE = ARGS.export_max_per_case
EXPORT_SELECTION = ARGS.export_selection
EXPORT_CASES = {item.strip() for item in ARGS.export_cases.split(",") if item.strip()}
EXPORT_SAMPLE_RANKS = {int(item.strip())
                       for item in ARGS.export_sample_ranks.split(",") if item.strip()}
if any(rank <= 0 for rank in EXPORT_SAMPLE_RANKS):
    raise ValueError("--export-sample-ranks must contain positive integers")

BATCH_TAG = "" if BATCH_SIZE == 1 else f"_b{BATCH_SIZE}"
OUT_DIR = Path(ARGS.output_dir) if ARGS.output_dir else paths.results_dir(
    "real_data", f"cifar{CASE_SAMPLE_SIZE}{BATCH_TAG}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_STAT_DIR = str(OUT_DIR)
METRIC_TABLE_PATH = str(OUT_DIR / f"cifar_case_metrics{CASE_SAMPLE_SIZE}{BATCH_TAG}.csv")

NETWORK_MAT  = str(paths.require(paths.NETWORK_MAT, "network.mat topology"))
PKL_T0       = str(paths.require(paths.MODEL_DIR / f"model_avg_ni{BATCH_SIZE}_N10_t0_z0_e1.pkl", "t0 checkpoint"))
PKL_T0_5     = str(paths.require(paths.MODEL_DIR / f"model_avg_ni{BATCH_SIZE}_N10_t0_5_z0_e1.pkl", "t0.5 checkpoint"))
PKL_T1       = str(paths.require(paths.MODEL_DIR / f"model_avg_ni{BATCH_SIZE}_N10_t1_z0_e1.pkl", "t1 checkpoint"))
PKL_DATASET  = str(paths.require(paths.DATASET_DIR / f"dataset_ni{BATCH_SIZE}_N10.pkl", "per-node dataset"))

print(f"[cifar attack] candidates={CASE_SAMPLE_SIZE} batch={BATCH_SIZE} "
      f"mode={ATTACK_MODE} seed={SEED}")
print(f"[cifar attack] outputs -> {OUT_DIR}")

# =====================================================================
# Helper functions
# =====================================================================

def flatten_params(state_dict):
    """Flatten all parameters in a PyTorch state_dict to a 1-D numpy array."""
    parts = []
    for key, param in state_dict.items():
        parts.append(param.ravel())
    return np.concatenate(parts)


def unflatten_params(state_dict_template, flat_params):
    """Reshape a flat numpy array back into an OrderedDict matching *template*."""
    reconstructed = collections.OrderedDict()
    idx = 0
    for name, param in state_dict_template.items():
        length = param.numel()
        shape  = param.shape
        reconstructed[name] = flat_params[idx : idx + length].reshape(shape)
        idx += length
    return reconstructed


def save_single_image(img, out_path):
    """Save one image array as a standalone file."""
    if img.ndim == 2:
        plt.imsave(out_path, img, cmap="gray")
    else:
        plt.imsave(out_path, img)


def to_hwc_uint8(image):
    """Convert image arrays to HWC/2D uint8 for plotting and metrics."""
    arr = np.array(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _ssim_single_channel(x, y, data_range=255.0):
    """SSIM for a single 2D channel."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = gaussian_filter(x, sigma=1.5)
    mu_y = gaussian_filter(y, sigma=1.5)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = gaussian_filter(x * x, sigma=1.5) - mu_x_sq
    sigma_y_sq = gaussian_filter(y * y, sigma=1.5) - mu_y_sq
    sigma_xy = gaussian_filter(x * y, sigma=1.5) - mu_xy

    sigma_x_sq = np.maximum(sigma_x_sq, 0.0)
    sigma_y_sq = np.maximum(sigma_y_sq, 0.0)

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def compute_ssim(image_a, image_b):
    """Compute SSIM for gray or RGB images."""
    a = to_hwc_uint8(image_a)
    b = to_hwc_uint8(image_b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for SSIM: {a.shape} vs {b.shape}")

    if a.ndim == 2:
        return _ssim_single_channel(a, b)
    if a.ndim == 3:
        return float(np.mean([
            _ssim_single_channel(a[..., c], b[..., c])
            for c in range(a.shape[2])
        ]))
    raise ValueError(f"Unsupported image shape for SSIM: {a.shape}")


def compute_psnr(image_a, image_b, data_range=255.0):
    """Compute PSNR for gray or RGB images."""
    a = to_hwc_uint8(image_a).astype(np.float64)
    b = to_hwc_uint8(image_b).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for PSNR: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def match_rows_by_mse(recovered, true_x):
    """Match unordered recovered vectors to true vectors by minimum MSE."""
    n_recovered = recovered.shape[0]
    n_true = true_x.shape[0]
    if n_recovered != n_true:
        raise ValueError(f"Row count mismatch: recovered={n_recovered}, true={n_true}")

    cost = np.zeros((n_recovered, n_true), dtype=float)
    for i in range(n_recovered):
        diff = true_x - recovered[i][None, :]
        cost[i, :] = np.mean(diff * diff, axis=1)

    row_ind, col_ind = linear_sum_assignment(cost)

    aligned = np.zeros_like(recovered)
    per_true_mse = {}
    for r, c in zip(row_ind, col_ind):
        aligned[c] = recovered[r]
        per_true_mse[int(c)] = float(cost[r, c])

    matched_mse = float(np.mean(cost[row_ind, col_ind]))
    pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
    return aligned, matched_mse, pairs, per_true_mse


# =====================================================================
# 1. Load network from network.mat & set random seed
# =====================================================================
set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

mat = scio.loadmat(NETWORK_MAT)
A = mat["A"]
W = np.array(mat["W"], dtype=int)
scale_factor = int(mat["scale_factor"].flat[0])
honest_nodes = mat["honest_nodes"].flatten().tolist()
corrupt_nodes = mat["corrupt_nodes"].flatten().tolist()
sg_honest = mat["sg_honest"].flatten().tolist()
sg_corrupt = mat["sg_corrupt"].flatten().tolist()

print("Weight matrix W (scaled integers, loaded from network.mat):")
print(W)
print(f"scale_factor = {scale_factor}")
print(f"Honest nodes:  {honest_nodes}")
print(f"Corrupt nodes: {corrupt_nodes}")


# =====================================================================
# 2. Build attackable sub-graphs from loaded info
# =====================================================================
filtered_subgraphs = [{
    "honest_nodes":  sg_honest,
    "corrupt_nodes": sg_corrupt,
    "num_honest":    len(sg_honest),
    "num_corrupt":   len(sg_corrupt),
}]

print(f"\nAttackable sub-graphs: {len(filtered_subgraphs)}")
for idx, sg in enumerate(filtered_subgraphs):
    print(f"  Sub-graph {idx}: corrupt={sg['corrupt_nodes']}  "
          f"honest={sg['honest_nodes']}")


# =====================================================================
# 4. Load real model parameters from pickle files
# =====================================================================
print("\nLoading model pickle files ...")

with open(PKL_T0, "rb") as f:
    models_0_raw = pickle.load(f)          # t=0
with open(PKL_T0_5, "rb") as f:
    models_0_5_raw = pickle.load(f)        # t=0.5 (mid-round)
with open(PKL_T1, "rb") as f:
    models_1_raw = pickle.load(f)          # t=1   (after aggregation)

# Quantise and flatten
# model_0[i] = quantised flat parameter vector of node i at t=0.5
# model_1[i] = quantised flat parameter vector of node i at t=1
#              (pre-multiplied by scale_factor for integer lattice)
model_0_list = []
for i in range(NUM_NODES):
    fp = flatten_params(models_0_5_raw[i])
    fp = np.round(fp * QUANTIZE).astype(np.int64)
    model_0_list.append(fp)
model_0 = np.stack(model_0_list)

model_1_list = []
for i in range(NUM_NODES):
    fp = flatten_params(models_1_raw[i])
    fp = np.round(fp * QUANTIZE * scale_factor).astype(np.int64)
    model_1_list.append(fp)
model_1 = np.stack(model_1_list)

MODEL_SIZE = model_0.shape[1]
print(f"Flattened model dimension: {MODEL_SIZE}")


# =====================================================================
# 5. Load dataset (for visualisation & label info)
# =====================================================================
with open(PKL_DATASET, "rb") as f:
    split_datasets = pickle.load(f)

all_images = split_datasets[0][0]
all_labels = split_datasets[0][1]
for node_idx in range(1, NUM_NODES):
    all_images = np.concatenate((all_images, split_datasets[node_idx][0]), axis=0)
    all_labels = np.concatenate((all_labels, split_datasets[node_idx][1]), axis=0)

print(f"Dataset images shape: {all_images.shape}, labels shape: {all_labels.shape}")


# =====================================================================
# 6. Build the DNN model (for DLG attack mode)
# =====================================================================
if DATASET == "cifar10" and MODEL_ARCH == "fc":
    model_init = fc(input_dim=32 * 32 * 3, output_dim=CLASSES)
elif DATASET == "mnist" and MODEL_ARCH == "fc":
    model_init = fc(input_dim=28 * 28, output_dim=CLASSES)
elif DATASET == "mnist" and MODEL_ARCH == "cnn":
    model_init = cnn(output_dim=CLASSES)
elif DATASET == "cifar10" and MODEL_ARCH == "cnn":
    model_init = cnn_cifar(output_dim=CLASSES)
else:
    raise ValueError(f"Unsupported dataset/model combination: {DATASET}/{MODEL_ARCH}")


# =====================================================================
# 7. Attack loop — iterate over attackable sub-graphs
# =====================================================================
x0 = gen_pseudoprime(PRIME_BITS)
print(f"\nPseudo-prime x0 ({PRIME_BITS} bits): {x0}")

os.makedirs(FIGURE_STAT_DIR, exist_ok=True)

# per sub-graph: {case_name: [selected recovered item, ...]}
all_case_results = []
metrics_rows = []

for sg_idx, info in enumerate(filtered_subgraphs):
    print(f"\n{'=' * 60}")
    print(f"Sub-graph {sg_idx}: honest={info['honest_nodes']}  "
          f"corrupt={info['corrupt_nodes']}")
    print("=" * 60)

    n_honest  = info["num_honest"]
    n_corrupt = info["num_corrupt"]

    # ── Weight sub-matrices ──────────────────────────────────────
    W_cc = W[np.ix_(info["corrupt_nodes"], corrupt_nodes)]
    W_ch = W[np.ix_(info["corrupt_nodes"], info["honest_nodes"])]

    W_cc_int = np.array(W_cc, dtype=int)
    W_ch_int = np.array(W_ch, dtype=int)

    W_cc_contiguous = np.ascontiguousarray(W_cc_int)
    W_ch_contiguous = np.ascontiguousarray(W_ch_int)
    max_value = int(np.max(W_ch_int) + 1)

    # ── Permute rows → full-rank top-left block ──────────────────
    success = False
    for _ in range(MAX_PERMUTE):
        perm = np.random.permutation(W_ch_contiguous.shape[0])
        W_ch_perm = W_ch_contiguous[perm, :]
        num_honest = n_honest  # alias used in notebook
        if np.linalg.matrix_rank(W_ch_perm[:num_honest, :num_honest]) == num_honest:
            W_ch_sage = matrix(ZZ, W_ch_perm)
            success = True
            break
    if not success:
        print(f"  Could not find full-rank permutation. Skipping.")
        all_case_results.append(None)
        continue

    print(f"  W_ch (permuted):\n{W_ch_sage}")

    # ── Resample & attack ────────────────────────────────────────
    resample_count = 0
    found = None
    dimension = n_honest

    while resample_count <= MAX_RESAMPLE:
        sample = sorted(np.random.choice(MODEL_SIZE, size=dimension,
                                         replace=False))

        model_0_seg = model_0[np.ix_(sorted(corrupt_nodes), sample)]
        model_1_seg = model_1[np.ix_(info["corrupt_nodes"], sample)]

        Y = model_1_seg - W_cc_contiguous @ model_0_seg
        Y = Y[perm, :]
        X = model_0[np.ix_(info["honest_nodes"], sample)]

        Y_sage = matrix(ZZ, Y)

        side_info = topology_side_information(W_ch_sage)

        try:
            MO, tt1 = step1_original(n_honest, int(Y_sage.nrows()), x0, Y_sage)
        except ZeroDivisionError:
            print(f"  step1 singular matrix, resampling "
                  f"({resample_count + 1}/{MAX_RESAMPLE}) ...")
            resample_count += 1
            continue
        print(f"  Step1 time: {tt1:.2f}s")

        # ── Nguyen-Stern HLCP (threaded with timeout) ────────────
        class _Result:
            value = None

        result_box = _Result()

        def _target(n_arg, m_arg, MO_arg, mv_arg, box):
            box.value = recover_candidates(n_arg, m_arg, MO_arg, mv_arg)

        t = threading.Thread(target=_target,
                             args=(n_honest, int(Y_sage.nrows()), MO, max_value, result_box))
        t0 = time.time()
        t.start()
        t.join(timeout=TIMEOUT)
        elapsed = time.time() - t0

        if t.is_alive():
            print(f"  ns_hlcp timed out ({TIMEOUT}s), resampling "
                  f"({resample_count + 1}/{MAX_RESAMPLE}) ...")
            t.join(timeout=1)
            resample_count += 1
            continue

        found = result_box.value
        if found is None:
            print(f"  ns_hlcp returned None, resampling "
                  f"({resample_count + 1}/{MAX_RESAMPLE}) ...")
            resample_count += 1
            continue

        print(f"  ns_hlcp succeeded in {elapsed:.2f}s")
        break

    if found is None:
        print("  No valid result. Skipping sub-graph.")
        all_case_results.append(None)
        continue

    # ── Filter candidate vectors under 3 cases ───────────────────
    found_np = np.array(found)
    print(f"  Candidate vectors shape: {found_np.shape}")

    target_sum, X_binary, zero_counts = side_info
    case_reconstructions = {
        "case1_exact_pattern": [],
        "case2_zero_count": [],
        "case3_no_structure": [],
    }

    try:
        case_reconstructions["case1_exact_pattern"] = filter_info(
            found_np,
            known_info=target_sum,
            number=n_honest,
            X_binary=X_binary,
        )
        print("    Case 1 (exact pattern):  "
              f"{len(case_reconstructions['case1_exact_pattern'])} solution(s)")
    except Exception as e:
        print(f"    Case 1 error: {e}")

    try:
        case_reconstructions["case2_zero_count"] = filter_info(
            found_np,
            known_info=target_sum,
            number=n_honest,
            target_zero_counts=zero_counts,
        )
        print("    Case 2 (zero-count):     "
              f"{len(case_reconstructions['case2_zero_count'])} solution(s)")
    except Exception as e:
        print(f"    Case 2 error: {e}")

    try:
        case_reconstructions["case3_no_structure"] = filter_info(
            found_np,
            known_info=target_sum,
            number=n_honest,
        )
        print("    Case 3 (no structure):   "
              f"{len(case_reconstructions['case3_no_structure'])} solution(s)")
    except Exception as e:
        print(f"    Case 3 error: {e}")

    if not any(case_reconstructions.values()):
        print("  No valid reconstruction in any case. Skipping sub-graph.")
        all_case_results.append(None)
        continue

    # ── Recover full model params for sampled combinations ───────
    Y_full = (model_1[np.ix_(info["corrupt_nodes"], np.arange(MODEL_SIZE))]
              - W_cc_contiguous @ model_0[np.ix_(sorted(corrupt_nodes),
                                                  np.arange(MODEL_SIZE))])
    Y_full = np.array(Y_full, dtype=float)
    if Y_full.ndim == 1:
        Y_full = Y_full.reshape(-1, 1)

    true_params = model_0[np.ix_(info["honest_nodes"], np.arange(MODEL_SIZE))]
    true_params = np.array(true_params, dtype=float) / QUANTIZE

    sg_case_results = {}
    for case_name, recon_list in case_reconstructions.items():
        if not recon_list:
            sg_case_results[case_name] = []
            continue

        pick_count = min(CASE_SAMPLE_SIZE, len(recon_list))
        sampled_indices = sorted(random.sample(range(len(recon_list)), pick_count))
        print(f"  {case_name}: sample {pick_count}/{len(recon_list)} combinations")

        selected_items = []
        for sample_rank, recon_idx in enumerate(sampled_indices, start=1):
            mat = recon_list[recon_idx]
            mat_np = sage_to_numpy(mat)
            mat_T = mat_np.T
            mat_T = mat_T[np.argsort(perm), :]
            pinv = np.linalg.pinv(mat_T)

            recovered_quant = np.dot(pinv, Y_full)
            recovered = np.array(recovered_quant, dtype=float) / QUANTIZE

            aligned_recovered, matched_mse, match_pairs, per_true_mse = \
                match_rows_by_mse(recovered, true_params)

            selected_items.append({
                "sample_rank": sample_rank,
                "recon_index": recon_idx + 1,
                "recovered": recovered,
                "aligned_recovered": aligned_recovered,
                "matched_mse": matched_mse,
                "match_pairs": match_pairs,
                "per_true_mse": per_true_mse,
            })

            print("    "
                  f"sel#{sample_rank} (recon idx={recon_idx + 1}) "
                  f"matched MSE={matched_mse:.6e}")

        should_export_case = not EXPORT_CASES or case_name in EXPORT_CASES
        if EXPORT_RECOVERED_DIR and should_export_case:
            if EXPORT_SAMPLE_RANKS:
                export_items = [
                    item
                    for item in selected_items
                    if item["sample_rank"] in EXPORT_SAMPLE_RANKS
                ]
                found_ranks = {item["sample_rank"] for item in export_items}
                missing_ranks = EXPORT_SAMPLE_RANKS - found_ranks
                if missing_ranks:
                    raise RuntimeError(
                        f"requested sample ranks are unavailable for {case_name}: "
                        f"{sorted(missing_ranks)}"
                    )
            elif EXPORT_SELECTION == "best":
                export_items = [
                    min(selected_items, key=lambda item: item["matched_mse"])
                ]
            elif EXPORT_MAX_PER_CASE <= 0:
                export_items = selected_items
            else:
                export_items = selected_items[:EXPORT_MAX_PER_CASE]

            for export_item in export_items:
                sample_rank = export_item["sample_rank"]
                recon_idx = export_item["recon_index"] - 1
                aligned_recovered = export_item["aligned_recovered"]
                matched_mse = export_item["matched_mse"]
                per_true_mse = export_item["per_true_mse"]
                recovered_states = {}
                for honest_idx, node in enumerate(info["honest_nodes"]):
                    recovered_sd = unflatten_params(
                        models_1_raw[0],
                        aligned_recovered[honest_idx],
                    )
                    recovered_states[int(node)] = collections.OrderedDict(
                        (name, torch.from_numpy(value))
                        for name, value in recovered_sd.items()
                    )
                os.makedirs(EXPORT_RECOVERED_DIR, exist_ok=True)
                export_name = (
                    f"recovered_ni{BATCH_SIZE}_sg{sg_idx}_{case_name}_"
                    f"sel{sample_rank}_idx{recon_idx + 1}.pkl"
                )
                export_path = os.path.join(EXPORT_RECOVERED_DIR, export_name)
                with open(export_path, "wb") as export_handle:
                    pickle.dump(
                        {
                            "format": "dfl-recovered-states-v1",
                            "states_by_node": recovered_states,
                            "metadata": {
                                "batch_size": BATCH_SIZE,
                                "subgraph": sg_idx,
                                "case": case_name,
                                "export_selection": (
                                    "sample-ranks:"
                                    + ",".join(str(rank) for rank in sorted(EXPORT_SAMPLE_RANKS))
                                    if EXPORT_SAMPLE_RANKS
                                    else EXPORT_SELECTION
                                ),
                                "sample_rank": sample_rank,
                                "recon_index": recon_idx + 1,
                                "honest_nodes": [int(node) for node in info["honest_nodes"]],
                                "matched_mse": float(matched_mse),
                                "per_true_mse": {
                                    int(key): float(value)
                                    for key, value in per_true_mse.items()
                                },
                            },
                        },
                        export_handle,
                    )
                export_item["recovered_state_path"] = export_path
                print(f"    exported recovered state -> {export_path}")

        sg_case_results[case_name] = selected_items

    all_case_results.append(sg_case_results)


if ATTACK_MODE == "none":
    print("\nLattice recovery/export completed; skipping legacy image inversion.")
    raise SystemExit(0)


# =====================================================================
# 8. Gradient reconstruction & image recovery
# =====================================================================
print(f"\n{'=' * 60}")
print("Image Reconstruction Phase")
print("=" * 60)

for sg_idx, info in enumerate(filtered_subgraphs):
    case_results = all_case_results[sg_idx]
    if case_results is None:
        print(f"\nSub-graph {sg_idx}: skipped (no valid case reconstruction).")
        continue

    n_honest = info["num_honest"]

    # Ground-truth images (honest-node order)
    gt_images = []
    for j in range(n_honest):
        node = info["honest_nodes"][j]
        gt_img = all_images[node]
        gt_img = np.clip(gt_img * 0.5 + 0.5, 0, 1) * 255
        gt_images.append(to_hwc_uint8(gt_img.astype(np.uint8)))

    # Save GT overview and per-node GT images once.
    fig, axes = plt.subplots(1, n_honest, figsize=(n_honest * 3, 3))
    if n_honest == 1:
        axes = [axes]
    for j, (ax, img) in enumerate(zip(axes, gt_images)):
        node = info["honest_nodes"][j]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(f"Node {node} (GT)")
        ax.axis("off")
    plt.suptitle(f"Sub-graph {sg_idx} · Ground Truth")
    plt.tight_layout()
    out_path = f"{FIGURE_STAT_DIR}/ground_truth_sg{sg_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")

    for j, img in enumerate(gt_images):
        node = info["honest_nodes"][j]
        out_path_single = f"{FIGURE_STAT_DIR}/ground_truth_sg{sg_idx}_node{node}.png"
        save_single_image(img, out_path_single)
        print(f"  Saved → {out_path_single}")

    for case_name, selected_items in case_results.items():
        if not selected_items:
            print(f"  {case_name}: no sampled combinations.")
            continue

        for item in selected_items:
            sample_rank = item["sample_rank"]
            recon_index = item["recon_index"]
            result = item["aligned_recovered"]

            print("\n--- "
                  f"Sub-graph {sg_idx}, {case_name}, "
                  f"sample {sample_rank} (recon idx={recon_index}) ---")

            gradient_reconstructed_list = []
            reconstructed_images = []

            # Build per-node reconstructed gradients using aligned recovered vectors.
            for j in range(n_honest):
                node = info["honest_nodes"][j]

                recon_sd = unflatten_params(models_1_raw[0], result[j])
                cleaned_sd = collections.OrderedDict()
                for k, v in recon_sd.items():
                    cleaned_sd[k] = torch.from_numpy(v)

                original_sd = models_0_raw[node]
                gradient_sd = collections.OrderedDict()
                for param_name in cleaned_sd.keys():
                    gradient_sd[param_name] = (
                        (-cleaned_sd[param_name] + original_sd[param_name]) / LR
                    )
                gradient_reconstructed_list.append(gradient_sd)

            # ── Run image reconstruction ─────────────────────────
            if ATTACK_MODE == "dlg":
                if DATASET == "cifar10":
                    data_size = split_datasets[0][0][0:1].shape
                else:
                    data_size = split_datasets[0][0][0:1].shape
                data_size = torch.Size(data_size)

                for j in range(n_honest):
                    node = info["honest_nodes"][j]
                    model_init.load_state_dict(models_0_raw[node])
                    model_init.to(DEVICE)
                    model_init.eval()

                    grads = gradient_reconstructed_list[j]
                    dummy_data_in = torch.randn(data_size).to(DEVICE).requires_grad_(True)
                    dummy_label_in = label_to_onehot(
                        torch.from_numpy(all_labels[node : node + 1]).long(),
                        num_classes=CLASSES,
                    ).to(DEVICE)

                    dummy_data, dummy_label, grad_diff_list = leakage_from_gradients(
                        copy.deepcopy(model_init), grads, dummy_data_in, dummy_label_in
                    )
                    image = dummy_data.cpu().detach().numpy()
                    if image.ndim == 4:
                        image = image[0].transpose(1, 2, 0)
                    image = np.clip(image * 0.5 + 0.5, 0, 1) * 255
                    reconstructed_images.append(to_hwc_uint8(image.astype(np.uint8)))

            elif ATTACK_MODE == "lr":
                for j in range(n_honest):
                    node = info["honest_nodes"][j]
                    label_val = int(all_labels[node : node + 1][0])

                    gradient_hat = flatten_params(gradient_reconstructed_list[j])
                    img_dim = 32 * 32 * 3
                    image = (
                        gradient_hat[label_val * img_dim : (label_val + 1) * img_dim]
                        / gradient_hat[img_dim * CLASSES + label_val]
                    )
                    image = image.reshape(3, 32, 32).transpose(1, 2, 0)
                    image = (
                        np.clip(image * 0.5 + 0.5, 0, 1) * 255
                    ).round().astype(np.uint8)
                    reconstructed_images.append(to_hwc_uint8(image))
            else:
                print(f"  Unknown ATTACK_MODE: {ATTACK_MODE}")
                continue

            if not reconstructed_images:
                print("  No reconstructed images generated.")
                continue

            # ── Save reconstructed images ───────────────────────
            fig, axes = plt.subplots(1, n_honest, figsize=(n_honest * 3, 3))
            if n_honest == 1:
                axes = [axes]
            for j, (ax, img) in enumerate(zip(axes, reconstructed_images)):
                node = info["honest_nodes"][j]
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)
                ax.set_title(f"Node {node}")
                ax.axis("off")
            plt.suptitle(
                f"Sub-graph {sg_idx} · {case_name} · "
                f"sample {sample_rank} (idx {recon_index})"
            )
            plt.tight_layout()
            out_path = (
                f"{FIGURE_STAT_DIR}/{case_name}_recon_sg{sg_idx}_"
                f"sel{sample_rank}_idx{recon_index}.png"
            )
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {out_path}")

            node_ssim_vals = []
            node_psnr_vals = []

            for j, img in enumerate(reconstructed_images):
                node = info["honest_nodes"][j]
                out_path_single = (
                    f"{FIGURE_STAT_DIR}/{case_name}_recon_sg{sg_idx}_"
                    f"sel{sample_rank}_idx{recon_index}_node{node}.png"
                )
                save_single_image(img, out_path_single)

                gt_img = gt_images[j]
                ssim_val = compute_ssim(img, gt_img)
                psnr_val = compute_psnr(img, gt_img)
                node_ssim_vals.append(ssim_val)
                node_psnr_vals.append(psnr_val)

                metrics_rows.append({
                    "sg_idx": sg_idx,
                    "case": case_name,
                    "sample_rank": sample_rank,
                    "recon_index": recon_index,
                    "node": str(node),
                    "row_type": "node",
                    "matched_mse_mean": item["matched_mse"],
                    "matched_mse_node": item["per_true_mse"].get(j, item["matched_mse"]),
                    "ssim": ssim_val,
                    "psnr": psnr_val,
                })

            metrics_rows.append({
                "sg_idx": sg_idx,
                "case": case_name,
                "sample_rank": sample_rank,
                "recon_index": recon_index,
                "node": "mean",
                "row_type": "sample_mean",
                "matched_mse_mean": item["matched_mse"],
                "matched_mse_node": item["matched_mse"],
                "ssim": float(np.mean(node_ssim_vals)),
                "psnr": float(np.mean(node_psnr_vals)),
            })

            print("  Mean metrics: "
                  f"MSE={item['matched_mse']:.6e}, "
                  f"SSIM={np.mean(node_ssim_vals):.4f}, "
                  f"PSNR={np.mean(node_psnr_vals):.4f}")


# =====================================================================
# 9. Metrics table
# =====================================================================
if metrics_rows:
    # Per-case statistics over all node-level metrics.
    case_stats = collections.defaultdict(lambda: {
        "mse": [],
        "ssim": [],
        "psnr": [],
    })

    for r in metrics_rows:
        if r.get("row_type") != "node":
            continue
        case_name = r["case"]
        case_stats[case_name]["mse"].append(float(r["matched_mse_node"]))
        case_stats[case_name]["ssim"].append(float(r["ssim"]))
        case_stats[case_name]["psnr"].append(float(r["psnr"]))

    for case_name, vals in sorted(case_stats.items()):
        if not vals["mse"]:
            continue
        mse_mean = float(np.mean(vals["mse"]))
        mse_var  = float(np.var(vals["mse"]))
        ssim_mean = float(np.mean(vals["ssim"]))
        ssim_var  = float(np.var(vals["ssim"]))
        psnr_mean = float(np.mean(vals["psnr"]))
        psnr_var  = float(np.var(vals["psnr"]))

        metrics_rows.append({
            "sg_idx": "all",
            "case": case_name,
            "sample_rank": "all",
            "recon_index": "all",
            "node": "case_summary",
            "row_type": "case_summary",
            "case_mse_mean": mse_mean,
            "case_mse_var": mse_var,
            "case_ssim_mean": ssim_mean,
            "case_ssim_var": ssim_var,
            "case_psnr_mean": psnr_mean,
            "case_psnr_var": psnr_var,
        })

        print(
            f"Case summary [{case_name}] -> "
            f"MSE(mean={mse_mean:.6e}, var={mse_var:.6e}), "
            f"SSIM(mean={ssim_mean:.6f}, var={ssim_var:.6e}), "
            f"PSNR(mean={psnr_mean:.6f}, var={psnr_var:.6e})"
        )

    fieldnames = [
        "sg_idx",
        "case",
        "sample_rank",
        "recon_index",
        "node",
        "row_type",
        "matched_mse_mean",
        "matched_mse_node",
        "ssim",
        "psnr",
        "case_mse_mean",
        "case_mse_var",
        "case_ssim_mean",
        "case_ssim_var",
        "case_psnr_mean",
        "case_psnr_var",
    ]
    with open(METRIC_TABLE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"\nSaved metrics table → {METRIC_TABLE_PATH}")
else:
    print("\nNo metrics rows generated.")


# =====================================================================
# Summary
# =====================================================================
print(f"\n{'=' * 60}")
print("Finished.")
n_ok = len([
    r for r in all_case_results
    if r is not None and any(len(v) > 0 for v in r.values())
])
print(f"Successful attacks: {n_ok} / {len(filtered_subgraphs)} sub-graphs")

"""Differential-privacy defense against the lattice attack -- Figures 5 and 11.

Runs the mHLCP attack against every complete DFL checkpoint triplet produced by
``training/train_cifar_dp.py`` and records, per privacy budget:

* whether all ground-truth weight vectors were recovered,
* the MSE and cosine similarity between the reconstructed and true gradients.

Two Step-1 variants are available (Section 7.6):

    --step1 exact   the paper's exact-arithmetic Step 1  -> Figure 5
    --step1 noisy   the noise-tolerant Step 1 of Sec 4.4 -> Figure 11

Local DP (``exchange``) perturbs X before aggregation and therefore leaves the
mHSSP/mHLCP structure intact; aggregation-level DP (``aggregate``) perturbs the
observation itself and breaks the exact algebraic relation.

Requires SageMath and PyTorch.
"""


from sage.all import *

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hssp_dfl import paths

import os
import re
import numpy as np
import random
import time
import threading
import pickle
import math
import csv

import scipy.io as scio
from scipy.special import ndtri
import torch

# ─── Toolkit imports ─────────────────────────────────────────────────
from hssp_dfl.utils import gen_pseudoprime
from hssp_dfl.attacks.noisy_step1 import noisy_step1_original
from hssp_dfl.attacks.step1 import step1_original
from hssp_dfl.dfl import recover_candidates, sage_to_numpy, topology_side_information
from hssp_dfl.filter import filter_info


def _short_path(path):
    """Record checkpoint locations relative to the repository root.

    The CSVs are compared across machines, so they must not carry the absolute
    location of somebody's checkout.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(paths.ROOT))
    except ValueError:
        return path.name


# =====================================================================
# Configuration -- every default is the paper's setting
# =====================================================================
_parser = argparse.ArgumentParser(
    description="DP defense evaluation (Figures 5 and 11).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument("--step1", choices=("exact", "noisy"), default="exact",
                     help="exact = the paper's Figure 5; noisy = Figure 11")
_parser.add_argument("--model-dir", default=str(paths.MODEL_DP_DIR),
                     help="directory holding the DP checkpoint triplets")
_parser.add_argument("--dataset", default=str(paths.DATASET_DIR / "dataset_ni500_N10.pkl"))
_parser.add_argument("--network", default=str(paths.NETWORK_MAT))
_parser.add_argument("--group-pattern", default="",
                     help="regex restricting which checkpoint groups are attacked")
_parser.add_argument("--output", default=None,
                     help="default: results/dp_defense/dp_<step1>_stats.csv")
_parser.add_argument("--seed", type=int, default=42)
ARGS = _parser.parse_args()

STEP1_MODE = ARGS.step1
SEED         = ARGS.seed
NUM_NODES    = 10
TIMEOUT      = 10          # seconds before Step 2 is killed
MAX_RESAMPLE = 5           # coordinate resamples per sub-graph
MAX_PERMUTE  = 100         # attempts to find a full-rank sub-matrix
PRIME_BITS   = 100         # bit length of the pseudo-prime modulus Q
QUANTIZE     = 1e10        # 10 decimal places, as in Section 7.5
LR           = 1e-2
CLASSES      = 10
DEVICE       = "cpu"
DATASET      = "cifar10"
MODEL_ARCH   = "cnn"
ATTACK_MODE  = "dlg"

# fl_dp.py uses these constants for its Gaussian perturbation.  Gaussian noise
# is unbounded, so noisy Step 1 receives a public high-probability bound over
# the r*m entries used by one lattice call rather than an oracle residual.
DP_DELTA = 1e-5
DP_CLIPPING_NORM = 1.0
RHO_FAILURE_PROB = 1e-5
GROUP_PATTERN = ARGS.group_pattern.strip()
OUTPUT_CSV_OVERRIDE = (ARGS.output or "").strip()

# Reference (no-DP) triplet plus the per-node dataset.
PKL_T0       = str(Path(ARGS.model_dir) / "model_avg_ni500_N10_t0_z0_e10.pkl")
PKL_T0_5     = str(Path(ARGS.model_dir) / "model_avg_ni500_N10_t0_5_z0_e10.pkl")
PKL_T1       = str(Path(ARGS.model_dir) / "model_avg_ni500_N10_t1_z0_e10.pkl")
PKL_DATASET  = ARGS.dataset


# =====================================================================
# Helper functions
# =====================================================================

def flatten_params(state_dict):
    """Flatten all parameters in a PyTorch state_dict to a 1-D numpy array."""
    parts = []
    for key, param in state_dict.items():
        parts.append(param.detach().cpu().numpy().ravel())
    return np.concatenate(parts)

def check_true_weights_in_found(found_np, W_ch_perm):
    true_cols = W_ch_perm.T
    matched = 0
    for col in true_cols:
        for vec in found_np:
            if np.array_equal(col, vec) or np.array_equal(-col, vec):
                matched += 1
                break
    return (matched == len(true_cols)), matched



# =====================================================================
# 1. Load network from network.mat & set random seed
# =====================================================================
set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

mat = scio.loadmat(ARGS.network)
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


MODEL_DIR = Path(ARGS.model_dir)
NETWORK_PATH = Path(ARGS.network)
if OUTPUT_CSV_OVERRIDE:
    OUTPUT_CSV = Path(OUTPUT_CSV_OVERRIDE)
else:
    OUTPUT_CSV = paths.results_dir("dp_defense") / f"dp_{STEP1_MODE}_stats.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
print(f"[dp_defense] step1={STEP1_MODE} models={MODEL_DIR}")
print(f"[dp_defense] output -> {OUTPUT_CSV}")

MODEL_FILE_RE = re.compile(
    r"^(?P<head>model_avg_ni\d+_N\d+)_(?P<stage>t0_5|t0|t1)_(?P<tag>z\d+_e\d+(?:_(?:exchange|aggregate)_ep[\dp]+)?)\.pkl$"
)


def compute_cosine_similarity(vec_a, vec_b):
    vec_a = np.asarray(vec_a).ravel()
    vec_b = np.asarray(vec_b).ravel()
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def load_state_dict_list(path):
    with path.open("rb") as f:
        return pickle.load(f)


def parse_dp_metadata(tag):
    match = re.search(r"_(exchange|aggregate)_ep([\dp]+)$", tag)
    if match is None:
        return "none", 0.0
    dp_mode = match.group(1)
    dp_epsilon = float(match.group(2).replace("p", "."))
    return dp_mode, dp_epsilon


def scan_model_groups(model_dir):
    grouped = {}
    for path in sorted(model_dir.glob("model_avg_*.pkl")):
        match = MODEL_FILE_RE.match(path.name)
        if match is None:
            continue
        key = f"{match.group('head')}_{match.group('tag')}"
        grouped.setdefault(key, {})[match.group("stage")] = path

    complete_groups = []
    for key in sorted(grouped):
        paths = grouped[key]
        if all(stage in paths for stage in ("t0", "t0_5", "t1")):
            dp_mode, dp_epsilon = parse_dp_metadata(key)
            complete_groups.append({
                "group": key,
                "dp_mode": dp_mode,
                "dp_epsilon": dp_epsilon,
                "t0": paths["t0"],
                "t0_5": paths["t0_5"],
                "t1": paths["t1"],
            })
    return complete_groups


def quantize_state_matrix(model_list, scale):
    return np.stack([
        np.round(flatten_params(state_dict) * scale).astype(np.int64)
        for state_dict in model_list
    ])


def gaussian_dp_residual_bound(epsilon, scale_factor, num_entries):
    """Return a public high-probability integer residual bound.

    Aggregate-mode DP adds Gaussian noise after mixing.  The attack multiplies
    the stored aggregate by QUANTIZE*scale_factor, so the same factor maps the
    model-domain standard deviation to the integer HSSP/HLCP observation.  A
    two-sided union bound covers all entries passed to one Step-1 call.
    """
    if epsilon <= 0 or num_entries <= 0:
        return 0
    sigma = (
        math.sqrt(2.0 * math.log(1.25 / DP_DELTA))
        * DP_CLIPPING_NORM
        / float(epsilon)
    )
    tail_per_side = RHO_FAILURE_PROB / (2.0 * float(num_entries))
    z_value = float(ndtri(1.0 - tail_per_side))
    noise_std_integer = sigma * QUANTIZE * float(scale_factor)
    # The additive margin covers the two fixed-point conversions.  It is tiny
    # relative to the DP term but keeps the bound conservative at large epsilon.
    quantization_margin = int(scale_factor) + 1
    return int(math.ceil(z_value * noise_std_integer)) + quantization_margin


def save_rows(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "dp_mode",
        "dp_epsilon",
        "step1_mode",
        "r",
        "rho_source",
        "rho_assumed",
        "rho_true",
        "rho_bound_valid",
        "rho_failure_probability",
        "noisy_beta",
        "noisy_step1_rows",
        "last_step1_error",
        "t0_path",
        "t0_5_path",
        "t1_path",
        "n_honest",
        "n_corrupt",
        "found_num_vectors",
        "true_vectors_matched",
        "true_vectors_all_found",
        "reconstruction_count",
        "gradient_mse",
        "gradient_cosine_similarity",
        "step1_time",
        "step2_time",
        "resample_count",
        "status",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def attack_group(group_info, topology_info, x0):
    W = topology_info["W"]
    scale_factor = topology_info["scale_factor"]
    honest_nodes = topology_info["honest_nodes"]
    corrupt_nodes = topology_info["corrupt_nodes"]
    sg_info = topology_info["sg_info"]

    models_t0_raw = load_state_dict_list(group_info["t0"])
    models_t05_raw = load_state_dict_list(group_info["t0_5"])
    models_t1_raw = load_state_dict_list(group_info["t1"])

    num_nodes = len(models_t0_raw)
    if not (len(models_t05_raw) == len(models_t1_raw) == num_nodes):
        raise ValueError("Model pickle lengths do not match")
    if num_nodes != NUM_NODES:
        raise ValueError(f"Expected {NUM_NODES} nodes, got {num_nodes}")

    model_0 = quantize_state_matrix(models_t05_raw, QUANTIZE)
    model_1 = quantize_state_matrix(models_t1_raw, QUANTIZE * scale_factor)

    model_size = model_0.shape[1]
    n_honest = len(sg_info["honest_nodes"])
    n_corrupt = len(sg_info["corrupt_nodes"])

    W_cc = W[np.ix_(sg_info["corrupt_nodes"], corrupt_nodes)]
    W_ch = W[np.ix_(sg_info["corrupt_nodes"], sg_info["honest_nodes"])]
    W_cc_int = np.array(W_cc, dtype=int)
    W_ch_int = np.array(W_ch, dtype=int)

    row = {
        "group": group_info["group"],
        "dp_mode": group_info["dp_mode"],
        "dp_epsilon": group_info["dp_epsilon"],
        "step1_mode": STEP1_MODE,
        "r": n_honest,
        "rho_source": (
            "gaussian_union_bound"
            if STEP1_MODE == "noisy" and group_info["dp_mode"] == "aggregate"
            else "zero_exact_relation"
        ),
        "rho_assumed": 0,
        "rho_true": None,
        "rho_bound_valid": None,
        "rho_failure_probability": (
            RHO_FAILURE_PROB if STEP1_MODE == "noisy" else None
        ),
        "noisy_beta": None,
        "noisy_step1_rows": None,
        "last_step1_error": None,
        "t0_path": _short_path(group_info["t0"]),
        "t0_5_path": _short_path(group_info["t0_5"]),
        "t1_path": _short_path(group_info["t1"]),
        "n_honest": n_honest,
        "n_corrupt": n_corrupt,
        "found_num_vectors": 0,
        "true_vectors_matched": 0,
        "true_vectors_all_found": False,
        "reconstruction_count": 0,
        "gradient_mse": None,
        "gradient_cosine_similarity": None,
        "step1_time": None,
        "step2_time": None,
        "resample_count": None,
        "status": "init",
    }

    max_value = int(np.max(W_ch_int) + 1)
    if STEP1_MODE == "noisy" and group_info["dp_mode"] == "aggregate":
        row["rho_assumed"] = gaussian_dp_residual_bound(
            group_info["dp_epsilon"],
            scale_factor,
            W_ch_int.shape[0] * n_honest,
        )

    perm = None
    for _ in range(MAX_PERMUTE):
        candidate_perm = np.random.permutation(W_ch_int.shape[0])
        candidate = W_ch_int[candidate_perm, :]
        if np.linalg.matrix_rank(candidate[:n_honest, :n_honest]) == n_honest:
            perm = candidate_perm
            break
    if perm is None:
        row["status"] = "no_full_rank_perm"
        return row

    W_ch_perm = W_ch_int[perm, :]
    W_ch_sage = matrix(ZZ, W_ch_perm)

    resample_count = 0
    found = None
    step1_elapsed = None
    step2_elapsed = None
    while resample_count <= MAX_RESAMPLE:
        sample = sorted(np.random.choice(model_size, size=n_honest, replace=False))
        model_0_seg = model_0[np.ix_(corrupt_nodes, sample)]
        model_1_seg = model_1[np.ix_(sg_info["corrupt_nodes"], sample)]

        Y = model_1_seg - W_cc_int @ model_0_seg
        Y = Y[perm, :]
        X = model_0[np.ix_(sg_info["honest_nodes"], sample)]

        Y_clean = W_ch_perm @ X
        residual = Y - Y_clean
        rho_true = int(np.max(np.abs(residual)))
        row["rho_true"] = rho_true
        row["rho_bound_valid"] = rho_true <= int(row["rho_assumed"])

        Y_sage = matrix(ZZ, Y)
        side_info = topology_side_information(W_ch_sage)

        t0_s1 = time.time()
        if STEP1_MODE == "exact":
            try:
                MO, tt1 = step1_original(
                    n_honest, int(Y_sage.nrows()), x0, Y_sage
                )
                step1_elapsed = round(time.time() - t0_s1, 6)
            except ZeroDivisionError:
                resample_count += 1
                continue
        else:
            try:
                MO, tt1, noisy_debug = noisy_step1_original(
                    n_honest,
                    int(Y_sage.nrows()),
                    x0,
                    Y_sage,
                    rho=int(row["rho_assumed"]),
                    return_debug=True,
                )
                step1_elapsed = round(time.time() - t0_s1, 6)
                row["noisy_beta"] = int(noisy_debug["beta"])
                row["noisy_step1_rows"] = int(
                    noisy_debug["Y_ortho"].nrows()
                )
                row["last_step1_error"] = None
            except Exception as exc:
                step1_elapsed = round(time.time() - t0_s1, 6)
                row["last_step1_error"] = (
                    f"{type(exc).__name__}:{exc}"
                )
                resample_count += 1
                continue

        class _Result:
            value = None

        result_box = _Result()

        def _target(n_arg, m_arg, MO_arg, mv_arg, box):
            box.value = recover_candidates(n_arg, m_arg, MO_arg, mv_arg)

        t0 = time.time()
        thread = threading.Thread(target=_target, args=(n_honest, int(Y_sage.nrows()), MO, max_value, result_box))
        thread.start()
        thread.join(timeout=TIMEOUT)
        if thread.is_alive():
            resample_count += 1
            continue

        step2_elapsed = round(time.time() - t0, 6)
        found = result_box.value
        if found is None:
            resample_count += 1
            continue
        break

    row["step1_time"] = step1_elapsed
    row["step2_time"] = step2_elapsed
    row["resample_count"] = resample_count

    if found is None:
        row["status"] = "no_found"
        return row

    found_np = np.array(found)
    row["found_num_vectors"] = int(found_np.shape[0])

    all_matched, n_matched = check_true_weights_in_found(found_np, W_ch_perm)
    row["true_vectors_all_found"] = bool(all_matched)
    row["true_vectors_matched"] = int(n_matched)

    if not all_matched:
        row["status"] = "partial_found"
        return row

    target_sum, X_binary, zero_counts = side_info
    reconstruct = filter_info(found_np, known_info=target_sum, number=n_honest, X_binary=X_binary)
    row["reconstruction_count"] = len(reconstruct)
    if not reconstruct:
        row["status"] = "no_reconstruction"
        return row

    best_reconstruction = reconstruct[23]
    best_np = sage_to_numpy(best_reconstruction)
    best_np = np.asarray(best_np)
    if best_np.ndim == 1:
        best_np = best_np.reshape(1, -1)

    best_T = best_np.T
    best_T = best_T[np.argsort(perm), :]
    pinv = np.linalg.pinv(best_T)

    Y_full = (
        model_1[np.ix_(sg_info["corrupt_nodes"], np.arange(model_size))]
        - W_cc_int @ model_0[np.ix_(corrupt_nodes, np.arange(model_size))]
    )
    Y_full = np.array(Y_full, dtype=float)
    if Y_full.ndim == 1:
        Y_full = Y_full.reshape(-1, 1)

    recovered = np.dot(pinv, Y_full) / QUANTIZE

    true_grad_list = []
    recon_grad_list = []
    for j, node in enumerate(sg_info["honest_nodes"]):
        true_t0 = flatten_params(models_t0_raw[node])
        true_t05 = flatten_params(models_t05_raw[node])
        recon_t05 = recovered[j]
        true_grad_list.append((true_t0 - true_t05) / LR)
        recon_grad_list.append((true_t0 - recon_t05) / LR)

    true_grad = np.concatenate(true_grad_list)
    recon_grad = np.concatenate(recon_grad_list)
    row["gradient_mse"] = float(np.mean((recon_grad - true_grad) ** 2))
    row["gradient_cosine_similarity"] = compute_cosine_similarity(recon_grad, true_grad)
    row["status"] = "ok"
    return row


# =====================================================================
# Main execution
# =====================================================================
set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

network = scio.loadmat(NETWORK_PATH)
A = network["A"]
W = np.array(network["W"], dtype=int)
scale_factor = int(network["scale_factor"].flat[0])
honest_nodes = network["honest_nodes"].flatten().tolist()
corrupt_nodes = network["corrupt_nodes"].flatten().tolist()
sg_honest = network["sg_honest"].flatten().tolist()
sg_corrupt = network["sg_corrupt"].flatten().tolist()

print("Weight matrix W (scaled integers, loaded from network.mat):")
print(W)
print(f"scale_factor = {scale_factor}")
print(f"Honest nodes:  {honest_nodes}")
print(f"Corrupt nodes: {corrupt_nodes}")

filtered_subgraphs = [{
    "honest_nodes": sg_honest,
    "corrupt_nodes": sg_corrupt,
    "num_honest": len(sg_honest),
    "num_corrupt": len(sg_corrupt),
}]

print(f"\nAttackable sub-graphs: {len(filtered_subgraphs)}")

groups = scan_model_groups(MODEL_DIR)
if GROUP_PATTERN:
    group_regex = re.compile(GROUP_PATTERN)
    groups = [group for group in groups if group_regex.search(group["group"])]
print(f"Found {len(groups)} complete model triplets in {MODEL_DIR}")
print(f"Step 1 mode: {STEP1_MODE}")

x0 = gen_pseudoprime(PRIME_BITS)
print(f"\nPseudo-prime x0 ({PRIME_BITS} bits): {x0}")

topology_info = {
    "W": W,
    "scale_factor": scale_factor,
    "honest_nodes": honest_nodes,
    "corrupt_nodes": corrupt_nodes,
    "sg_info": filtered_subgraphs[0],
}

all_rows = []
for group_info in groups:
    print(f"\n{'=' * 70}")
    print(f"Processing {group_info['group']}")
    print(f"  DP mode: {group_info['dp_mode']}, epsilon: {group_info['dp_epsilon']}")
    row = attack_group(group_info, topology_info, x0)
    all_rows.append(row)
    save_rows(OUTPUT_CSV, all_rows)
    print(
        f"  status={row['status']}, found={row['true_vectors_matched']}/"
        f"{filtered_subgraphs[0]['num_honest']}, recon={row['reconstruction_count']}, "
        f"rho={row['rho_true']}/{row['rho_assumed']}"
    )
    if row["gradient_mse"] is not None:
        print(
            f"  gradient mse={row['gradient_mse']:.6e}, "
            f"cosine={row['gradient_cosine_similarity']:.6f}"
        )

print(f"\n{'=' * 70}")
print("Finished.")
print(f"Saved CSV -> {OUTPUT_CSV}")

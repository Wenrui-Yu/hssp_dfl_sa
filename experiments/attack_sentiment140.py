"""Sentiment140 embedding reconstruction after breaking DFL SA.

Paper results: Figure 10, Table 2, Tables 12-14

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
import copy
import collections
import math
import csv
import os

import scipy.io as scio
from scipy.optimize import linear_sum_assignment
import torch

# ─── Local project imports ───────────────────────────────────────────
from hssp_dfl.models import logistic_regression

# ─── Toolkit imports ─────────────────────────────────────────────────
from hssp_dfl.utils import gen_pseudoprime
from hssp_dfl.attacks.step1 import step1_original
from hssp_dfl.dfl import recover_candidates, sage_to_numpy, topology_side_information
from hssp_dfl.filter import filter_info


# =====================================================================
# Configuration -- every default is the paper's setting
# =====================================================================
_parser = argparse.ArgumentParser(
    description="Sentiment140 mHLCP attack + embedding reconstruction "
                "(Figure 10, Tables 2, 12-14).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument("--candidates", type=int, default=30,
                     help="candidate weight matrices sampled per case (paper: 30)")
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--timeout", type=int, default=10)
_parser.add_argument("--max-resample", type=int, default=5)
_parser.add_argument("--prime-bits", type=int, default=100)
_parser.add_argument("--device", default="cpu")
_parser.add_argument("--output-dir", default=None,
                     help="default: results/real_data/sentiment140")
ARGS = _parser.parse_args()

SEED         = ARGS.seed
NUM_NODES    = 10
TIMEOUT      = ARGS.timeout
MAX_RESAMPLE = ARGS.max_resample
MAX_PERMUTE  = 100
PRIME_BITS   = ARGS.prime_bits
QUANTIZE     = 1e10
LR           = 1e-2
CLASSES      = 2
FEATURE_DIM  = 1536        # text-embedding-ada-002 dimension
DEVICE       = ARGS.device
CASE_SAMPLE_SIZE = ARGS.candidates

OUT_DIR = Path(ARGS.output_dir) if ARGS.output_dir else paths.results_dir(
    "real_data", "sentiment140")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_STAT_DIR = str(OUT_DIR)
METRIC_TABLE_PATH = str(OUT_DIR / "sentiment140_case_metrics.csv")
EMBEDDING_DUMP_PATH = str(OUT_DIR / "sentiment140_reconstructed_embeddings.pkl")

NETWORK_MAT  = str(paths.require(paths.NETWORK_MAT, "network.mat topology"))
PKL_T0       = str(paths.require(paths.MODEL_DIR / "model_sentiment140_ni1_N10_t0_z0_e1.pkl", "t0 checkpoint"))
PKL_T0_5     = str(paths.require(paths.MODEL_DIR / "model_sentiment140_ni1_N10_t0_5_z0_e1.pkl", "t0.5 checkpoint"))
PKL_T1       = str(paths.require(paths.MODEL_DIR / "model_sentiment140_ni1_N10_t1_z0_e1.pkl", "t1 checkpoint"))
PKL_DATASET  = str(paths.require(paths.DATASET_DIR / "dataset_sentiment140_ni1_N10.pkl", "per-node dataset"))

print(f"[sentiment140 attack] candidates={CASE_SAMPLE_SIZE} seed={SEED}")
print(f"[sentiment140 attack] outputs -> {OUT_DIR}")

# =====================================================================
# Helper functions
# =====================================================================

def flatten_params(state_dict):
    parts = []
    for key, param in state_dict.items():
        parts.append(param.ravel())
    return np.concatenate(parts)


def unflatten_params(state_dict_template, flat_params):
    reconstructed = collections.OrderedDict()
    idx = 0
    for name, param in state_dict_template.items():
        length = param.numel()
        shape  = param.shape
        reconstructed[name] = flat_params[idx : idx + length].reshape(shape)
        idx += length
    return reconstructed


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


def compute_cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# =====================================================================
# 1. Load network
# =====================================================================
set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

mat = scio.loadmat(NETWORK_MAT)
W = np.array(mat["W"], dtype=int)
scale_factor = int(mat["scale_factor"].flat[0])
honest_nodes  = mat["honest_nodes"].flatten().tolist()
corrupt_nodes = mat["corrupt_nodes"].flatten().tolist()
sg_honest  = mat["sg_honest"].flatten().tolist()
sg_corrupt = mat["sg_corrupt"].flatten().tolist()

print("W (scaled integers):")
print(W)
print(f"scale_factor = {scale_factor}")
print(f"Honest:  {honest_nodes}")
print(f"Corrupt: {corrupt_nodes}")


# =====================================================================
# 2. Build attackable sub-graph
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
# 3. Load model parameters
# =====================================================================
print("\nLoading model pickle files ...")

with open(PKL_T0, "rb") as f:
    models_0_raw = pickle.load(f)
with open(PKL_T0_5, "rb") as f:
    models_0_5_raw = pickle.load(f)
with open(PKL_T1, "rb") as f:
    models_1_raw = pickle.load(f)

model_0_list, model_1_list = [], []
for i in range(NUM_NODES):
    fp = flatten_params(models_0_5_raw[i])
    model_0_list.append(np.round(fp * QUANTIZE).astype(np.int64))
for i in range(NUM_NODES):
    fp = flatten_params(models_1_raw[i])
    model_1_list.append(np.round(fp * QUANTIZE * scale_factor).astype(np.int64))
model_0 = np.stack(model_0_list)
model_1 = np.stack(model_1_list)

MODEL_SIZE = model_0.shape[1]
print(f"Flattened model dimension: {MODEL_SIZE}")


# =====================================================================
# 4. Load dataset & TF-IDF vectorizer
# =====================================================================
with open(PKL_DATASET, "rb") as f:
    split_datasets = pickle.load(f)

all_features = split_datasets[0][0]
all_labels   = split_datasets[0][1]
for node_idx in range(1, NUM_NODES):
    all_features = np.concatenate((all_features, split_datasets[node_idx][0]), axis=0)
    all_labels   = np.concatenate((all_labels, split_datasets[node_idx][1]),   axis=0)

print(f"Dataset features shape: {all_features.shape}, labels shape: {all_labels.shape}")


# =====================================================================
# 5. Build model
# =====================================================================
model_init = logistic_regression(input_dim=FEATURE_DIM, output_dim=CLASSES)


# =====================================================================
# 6. Attack loop
# =====================================================================
x0 = gen_pseudoprime(PRIME_BITS)
print(f"\nPseudo-prime x0 ({PRIME_BITS} bits): {x0}")

os.makedirs(FIGURE_STAT_DIR, exist_ok=True)

all_case_results = []
metrics_rows = []
embedding_rows = []

for sg_idx, info in enumerate(filtered_subgraphs):
    print(f"\n{'=' * 60}")
    print(f"Sub-graph {sg_idx}: honest={info['honest_nodes']}  "
          f"corrupt={info['corrupt_nodes']}")
    print("=" * 60)

    n_honest  = info["num_honest"]
    n_corrupt = info["num_corrupt"]

    W_cc = W[np.ix_(info["corrupt_nodes"], corrupt_nodes)]
    W_ch = W[np.ix_(info["corrupt_nodes"], info["honest_nodes"])]
    W_cc_int = np.ascontiguousarray(np.array(W_cc, dtype=int))
    W_ch_int = np.ascontiguousarray(np.array(W_ch, dtype=int))
    max_value = int(np.max(W_ch_int) + 1)

    # Permute for full-rank
    success = False
    for _ in range(MAX_PERMUTE):
        perm = np.random.permutation(W_ch_int.shape[0])
        W_ch_perm = W_ch_int[perm, :]
        if np.linalg.matrix_rank(W_ch_perm[:n_honest, :n_honest]) == n_honest:
            W_ch_sage = matrix(ZZ, W_ch_perm)
            success = True
            break
    if not success:
        print("  Could not find full-rank permutation. Skipping.")
        all_case_results.append(None)
        continue

    print(f"  W_ch (permuted):\n{W_ch_sage}")

    # Resample & attack
    resample_count = 0
    found = None

    while resample_count <= MAX_RESAMPLE:
        sample = sorted(np.random.choice(MODEL_SIZE, size=n_honest, replace=False))
        model_0_seg = model_0[np.ix_(sorted(corrupt_nodes), sample)]
        model_1_seg = model_1[np.ix_(info["corrupt_nodes"], sample)]

        Y = model_1_seg - W_cc_int @ model_0_seg
        Y = Y[perm, :]
        X = model_0[np.ix_(info["honest_nodes"], sample)]

        Y_sage = matrix(ZZ, Y)

        print(Y.shape)

        side_info = topology_side_information(W_ch_sage)
        try:
            MO, tt1 = step1_original(n_honest, int(Y_sage.nrows()), x0, Y_sage)
        except ZeroDivisionError:
            print(f"  step1 singular matrix, resampling "
                  f"({resample_count + 1}/{MAX_RESAMPLE}) ...")
            resample_count += 1
            continue
        print(f"  Step1 time: {tt1:.2f}s")

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
              - W_cc_int @ model_0[np.ix_(sorted(corrupt_nodes),
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

        sg_case_results[case_name] = selected_items

    all_case_results.append(sg_case_results)


# =====================================================================
# 7. Embedding reconstruction metrics (without text inversion)
# =====================================================================
print(f"\n{'=' * 60}")
print("Embedding Reconstruction Phase")
print("=" * 60)

for sg_idx, info in enumerate(filtered_subgraphs):
    case_results = all_case_results[sg_idx]
    if case_results is None:
        print(f"\nSub-graph {sg_idx}: skipped (no valid case reconstruction).")
        continue

    n_honest = info["num_honest"]

    # Ground-truth embeddings and texts (honest-node order)
    gt_embeddings = []
    for j in range(n_honest):
        node = info["honest_nodes"][j]
        gt_emb = np.array(all_features[node]).flatten()[:FEATURE_DIM]
        gt_embeddings.append(gt_emb)

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

            node_cosim_vals = []

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

                label_val = int(all_labels[node:node+1][0])
                grad_w = gradient_sd["fc1.weight"].numpy()  # (2, 1536)
                grad_b = gradient_sd["fc1.bias"].numpy()     # (2,)

                if abs(grad_b[label_val]) > 1e-15:
                    embedding_hat = grad_w[label_val] / grad_b[label_val]
                else:
                    embedding_hat = np.zeros(FEATURE_DIM)

                gt_embedding = gt_embeddings[j]

                # Compute cosine similarity
                cosim_val = compute_cosine_similarity(embedding_hat, gt_embedding)
                node_cosim_vals.append(cosim_val)

                print(f"\n  Node {node} (label={label_val}):")
                print(f"    Cosine similarity: {cosim_val:.6f}")
                print(f"    Matched node MSE: {item['per_true_mse'].get(j, item['matched_mse']):.6e}")

                metrics_rows.append({
                    "sg_idx": sg_idx,
                    "case": case_name,
                    "sample_rank": sample_rank,
                    "recon_index": recon_index,
                    "node": str(node),
                    "row_type": "node",
                    "matched_mse_mean": item["matched_mse"],
                    "matched_mse_node": item["per_true_mse"].get(j, item["matched_mse"]),
                    "cosine_similarity": cosim_val,
                })

                embedding_rows.append({
                    "sg_idx": sg_idx,
                    "case": case_name,
                    "sample_rank": sample_rank,
                    "recon_index": recon_index,
                    "node": int(node),
                    "label": label_val,
                    "matched_mse_mean": float(item["matched_mse"]),
                    "matched_mse_node": float(item["per_true_mse"].get(j, item["matched_mse"])),
                    "cosine_similarity": float(cosim_val),
                    "embedding_hat": np.array(embedding_hat, dtype=np.float32),
                    "gt_embedding": np.array(gt_embedding, dtype=np.float32),
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
                "cosine_similarity": float(np.mean(node_cosim_vals)),
            })

            print("  Mean metrics: "
                  f"matched_mse={item['matched_mse']:.6e}, "
                                    f"cosine_sim={np.mean(node_cosim_vals):.6f}")


# =====================================================================
# 8. Metrics table
# =====================================================================
if metrics_rows:
    # Per-case statistics over all node-level metrics.
    case_stats = collections.defaultdict(lambda: {
        "matched_mse_node": [],
        "cosine_similarity": [],
    })

    for r in metrics_rows:
        if r.get("row_type") != "node":
            continue
        case_name = r["case"]
        case_stats[case_name]["matched_mse_node"].append(float(r["matched_mse_node"]))
        case_stats[case_name]["cosine_similarity"].append(float(r["cosine_similarity"]))

    for case_name, vals in sorted(case_stats.items()):
        if not vals["cosine_similarity"]:
            continue
        mse_mean   = float(np.mean(vals["matched_mse_node"]))
        mse_var    = float(np.var(vals["matched_mse_node"]))
        cosim_mean = float(np.mean(vals["cosine_similarity"]))
        cosim_var  = float(np.var(vals["cosine_similarity"]))

        metrics_rows.append({
            "sg_idx": "all",
            "case": case_name,
            "sample_rank": "all",
            "recon_index": "all",
            "node": "case_summary",
            "row_type": "case_summary",
            "case_mse_mean": mse_mean,
            "case_mse_var": mse_var,
            "case_cosine_similarity_mean": cosim_mean,
            "case_cosine_similarity_var": cosim_var,
        })

        print(
            f"Case summary [{case_name}] -> "
            f"MSE(mean={mse_mean:.6e}, var={mse_var:.6e}), "
            f"Cosine_sim(mean={cosim_mean:.6f}, var={cosim_var:.6e}), "
            "ROUGE-L deferred to vec2text script"
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
        "cosine_similarity",
        "case_mse_mean",
        "case_mse_var",
        "case_cosine_similarity_mean",
        "case_cosine_similarity_var",
    ]
    with open(METRIC_TABLE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"\nSaved metrics table → {METRIC_TABLE_PATH}")
else:
    print("\nNo metrics rows generated.")

if embedding_rows:
    with open(EMBEDDING_DUMP_PATH, "wb") as f:
        pickle.dump(embedding_rows, f)
    print(f"Saved reconstructed embeddings → {EMBEDDING_DUMP_PATH}")
else:
    print("No reconstructed embeddings to save.")


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

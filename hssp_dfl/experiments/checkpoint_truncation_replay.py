"""Replay truncation noise on model checkpoints from a real DFL run.

The local model states saved at ``t0_5`` are the client vectors and the model
states saved at ``t1`` are the aggregates produced by the training scripts.
For attacked rows, this module constructs the same observation as the legacy
DFL attack scripts:

    Y_observed = q(F * scale_factor * model_t1)
                 - W_cc * q(F * model_t0_5)

It then compares exact Step 1 and noisy Step 1 on the same coordinates.
"""

import argparse
import csv
import itertools
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as scio
from sage.all import ZZ, Matrix, set_random_seed

from hssp_dfl import paths
from hssp_dfl.experiments.common import run_trial_with_timeout
from hssp_dfl.experiments.evaluation import (
	run_candidate_attack,
	wilson_interval,
)
from hssp_dfl.utils import gen_pseudoprime


CHECKPOINTS = {
	"sentiment140": (
		"model_sentiment140_ni1_N10_t0_5_z0_e1.pkl",
		"model_sentiment140_ni1_N10_t1_z0_e1.pkl",
	),
	"purchase": (
		"model_purchase_ni1_N10_t0_5_z0_e1.pkl",
		"model_purchase_ni1_N10_t1_z0_e1.pkl",
	),
	"cifar": (
		"model_avg_ni1_N10_t0_5_z0_e1.pkl",
		"model_avg_ni1_N10_t1_z0_e1.pkl",
	),
}

# Settings used by the paper's ``Attack on Real Dataset`` scripts.  The
# checkpoint replay keeps these values explicit so that a truncation trial is
# performed on the same DFL instance rather than on a new synthetic topology.
PAPER_NUM_NODES = 10
PAPER_NUM_EDGES = 20
PAPER_CORRUPT_RATIO = 0.6
PAPER_BATCH_SIZE = 1
PAPER_QUANTIZE = 10**10
PAPER_PRIME_BITS = 100
PAPER_MAX_RESAMPLE = 5

DATASET_SPECS = {
	"sentiment140": {
		"model_architecture": "logistic_regression",
		"expected_model_dimension": 3074,
	},
	"purchase": {
		"model_architecture": "purchase_fc(600-256-100)",
		"expected_model_dimension": 179556,
	},
	"cifar": {
		"model_architecture": "cnn_cifar(32-64-128-512-10)",
		"expected_model_dimension": 1147466,
	},
}


def _flatten_state_dict(state_dict):
	return np.concatenate([
		tensor.detach().cpu().numpy().reshape(-1)
		for tensor in state_dict.values()
	])


def _load_checkpoint_pair(dataset, model_dir):
	if dataset not in CHECKPOINTS:
		raise ValueError(f"unknown checkpoint dataset: {dataset}")
	local_name, aggregate_name = CHECKPOINTS[dataset]
	with (Path(model_dir) / local_name).open("rb") as handle:
		local_states = pickle.load(handle)
	with (Path(model_dir) / aggregate_name).open("rb") as handle:
		aggregate_states = pickle.load(handle)
	if len(local_states) != len(aggregate_states):
		raise ValueError("t0_5 and t1 contain different client counts")

	local = np.stack([
		_flatten_state_dict(state) for state in local_states
	])
	aggregate = np.stack([
		_flatten_state_dict(state) for state in aggregate_states
	])
	if local.shape != aggregate.shape:
		raise ValueError("t0_5 and t1 checkpoint dimensions differ")
	return local, aggregate


def _load_network(network_path):
	data = scio.loadmat(network_path)
	W = np.asarray(data["W"], dtype=np.int64)
	A = np.asarray(data.get("A", np.zeros_like(W)), dtype=np.int64)
	num_edges = int(np.count_nonzero(np.triu(A, 1)))
	return {
		"W": W,
		"A": A,
		"scale_factor": int(data["scale_factor"].flat[0]),
		"honest_nodes": np.asarray(
			data["sg_honest"], dtype=np.int64
		).reshape(-1),
		"corrupt_nodes": np.asarray(
			data["corrupt_nodes"], dtype=np.int64
		).reshape(-1),
		"attack_rows": np.asarray(
			data["sg_corrupt"], dtype=np.int64
		).reshape(-1),
		"num_nodes": int(W.shape[0]),
		"num_edges": num_edges,
		"corrupt_ratio": float(
			len(np.asarray(data["corrupt_nodes"]).reshape(-1)) / W.shape[0]
		),
	}


def _full_rank_row_permutation(X):
	n = X.shape[1]
	for leading in itertools.combinations(range(X.shape[0]), n):
		if np.linalg.matrix_rank(X[list(leading)]) == n:
			leading = list(leading)
			return leading + [
				row for row in range(X.shape[0]) if row not in leading
			]
	raise ValueError("the checkpoint attack matrix is not full column rank")


def _quantize(values, scale, quantizer):
	scaled = np.asarray(values) * scale
	if quantizer == "nearest":
		rounded = np.rint(scaled)
	elif quantizer == "truncate":
		rounded = np.trunc(scaled)
	else:
		raise ValueError("quantizer must be 'nearest' or 'truncate'")
	limit = np.iinfo(np.int64)
	if np.max(rounded) > limit.max or np.min(rounded) < limit.min:
		raise OverflowError("fixed-point values do not fit in int64")
	return rounded.astype(np.int64)


def _mod_matrix(values, Q):
	return Matrix(ZZ, values.tolist()).apply_map(lambda value: value % Q)


def _checkpoint_observation(local, aggregate, network, coordinates, scale,
							quantizer):
	W = network["W"]
	scale_factor = network["scale_factor"]
	honest = network["honest_nodes"]
	corrupt = network["corrupt_nodes"]
	attack_rows = network["attack_rows"]

	X_unpermuted = W[np.ix_(attack_rows, honest)]
	permutation = _full_rank_row_permutation(X_unpermuted)
	X_values = X_unpermuted[permutation]

	local_segment = local[:, coordinates]
	aggregate_segment = aggregate[:, coordinates]
	local_int = _quantize(local_segment, scale, quantizer)
	aggregate_int = _quantize(
		aggregate_segment, scale * scale_factor, quantizer
	)

	observed = (
		aggregate_int[attack_rows]
		- W[np.ix_(attack_rows, corrupt)] @ local_int[corrupt]
	)[permutation]
	clean = (
		X_unpermuted @ local_int[honest]
	)[permutation]
	error = observed - clean

	recomputed = np.stack([
		np.stack([
			local_segment[node] * W[row, node] / scale_factor
			for node in range(W.shape[1])
		]).sum(axis=0)
		for row in attack_rows
	])
	aggregation_error = (
		aggregate_segment[attack_rows] - recomputed
	)

	# Each fixed-point conversion contributes at most q_error units. The
	# float aggregation term uses a conservative gamma bound for N products,
	# divisions, and summands in the training script.
	q_error = 0.5 if quantizer == "nearest" else 1.0
	max_row_l1 = int(max(
		np.sum(np.abs(W[row])) for row in attack_rows
	))
	unit_roundoff = np.finfo(local.dtype).eps / 2
	operation_count = 3 * W.shape[1]
	gamma = (
		operation_count * unit_roundoff
		/ (1.0 - operation_count * unit_roundoff)
	)
	model_abs_bound = float(np.max(np.abs(local)))
	rho_theoretical = int(math.ceil(
		q_error * (1 + max_row_l1)
		+ scale * gamma * max_row_l1 * model_abs_bound
	))
	rho_true = int(np.max(np.abs(error)))

	return {
		"X": Matrix(ZZ, X_values.tolist()),
		"Y_clean_signed": clean,
		"Y_observed_signed": observed,
		"E": error,
		"rho_true": rho_true,
		"rho_theoretical": rho_theoretical,
		"rho_quantization": int(math.ceil(
			q_error * (1 + max_row_l1)
		)),
		"max_row_l1": max_row_l1,
		"model_abs_bound": model_abs_bound,
		"aggregation_max_abs": float(
			np.max(np.abs(aggregation_error))
		),
		"permutation": permutation,
	}


def run_checkpoint_replay_trial(dataset="sentiment140", scale=PAPER_QUANTIZE,
								quantizer="nearest", l=None,
								step1_columns=None,
								exact_step1_columns=None,
								coordinate_seed=0, nx0=100,
								model_dir=Path("models"),
								network_path=Path("network.mat"),
								rho_multiplier=1.0,
								max_resample=PAPER_MAX_RESAMPLE):
	"""Run one paired trial on real ``t0_5`` and ``t1`` checkpoints."""
	if dataset not in DATASET_SPECS:
		raise ValueError(f"unknown dataset specification: {dataset}")
	local, aggregate = _load_checkpoint_pair(dataset, model_dir)
	network = _load_network(network_path)
	if network["num_nodes"] != PAPER_NUM_NODES:
		raise ValueError(
			f"paper real-data setup expects {PAPER_NUM_NODES} nodes, "
			f"got {network['num_nodes']}"
		)
	if network["num_edges"] != PAPER_NUM_EDGES:
		raise ValueError(
			f"paper real-data setup expects {PAPER_NUM_EDGES} edges, "
			f"got {network['num_edges']}"
		)
	if not math.isclose(
		network["corrupt_ratio"], PAPER_CORRUPT_RATIO, abs_tol=1e-12
	):
		raise ValueError(
			f"paper real-data setup expects corruption ratio "
			f"{PAPER_CORRUPT_RATIO}, got {network['corrupt_ratio']}"
		)
	if local.shape[0] != network["W"].shape[0]:
		raise ValueError("checkpoint and network client counts differ")
	expected_dimension = DATASET_SPECS[dataset]["expected_model_dimension"]
	if local.shape[1] != expected_dimension:
		raise ValueError(
			f"{dataset} checkpoint has dimension {local.shape[1]}, "
			f"expected {expected_dimension} for "
			f"{DATASET_SPECS[dataset]['model_architecture']}"
		)
	n = int(network["honest_nodes"].size)
	if step1_columns is None:
		r = n
	else:
		r = int(step1_columns)
		if r <= 0:
			raise ValueError("step1_columns must be positive")
	if exact_step1_columns is None:
		exact_r = n
	else:
		exact_r = int(exact_step1_columns)
		if exact_r <= 0:
			raise ValueError("exact_step1_columns must be positive")
	if l is None:
		# The original real-data scripts use exactly n_honest coordinates for
		# the lattice stage. Ablations with r > n use at least r coordinates.
		l = max(n, r, exact_r)
	if l < max(n, r, exact_r):
		raise ValueError(
			"l must be at least max(n_honest, step1_columns, "
			"exact_step1_columns)"
		)
	if l > local.shape[1]:
		raise ValueError("l exceeds the flattened checkpoint dimension")

	set_random_seed(coordinate_seed)
	Q = gen_pseudoprime(nx0)
	observation = None
	coordinates = None
	clean = observed_exact = noisy = None
	resample_count = 0
	# Use one random stream across retries, as in the original real-data
	# scripts.  Re-seeding every attempt would define a different retry
	# experiment and can substantially change the rank-success probability.
	rng = np.random.default_rng(coordinate_seed)
	for resample_count in range(int(max_resample) + 1):
		# The paper's real-data attack samples n_honest coordinates for Step 1.
		# A new sample is drawn if its leading observation block is singular.
		coordinates = rng.choice(
			local.shape[1], size=l, replace=False
		)
		observation = _checkpoint_observation(
			local,
			aggregate,
			network,
			coordinates,
			int(scale),
			quantizer,
		)
		max_signal = max(
			int(np.max(np.abs(observation["Y_clean_signed"]))),
			int(np.max(np.abs(observation["Y_observed_signed"]))),
		)
		rho_assumed = int(math.ceil(
			rho_multiplier * observation["rho_theoretical"]
		))
		if max_signal + rho_assumed >= Q // 2:
			raise ValueError("Q is too small for no-wrap centered decoding")

		X = observation["X"]
		# Match the original real-data scripts: max(W_ch)+1 is supplied as
		# the bounded HLCP coefficient range.
		B = int(np.max(network["W"][np.ix_(
			network["attack_rows"], network["honest_nodes"]
		)]) + 1)
		Y_clean = _mod_matrix(observation["Y_clean_signed"], Q)
		Y_observed = _mod_matrix(observation["Y_observed_signed"], Q)
		exact_columns = list(range(exact_r))
		noisy_columns = list(range(r))
		clean = run_candidate_attack(
			Y_clean, Q, X, B, noisy=False,
			attack_columns=exact_columns,
		)
		# Match the original real-data attack: a singular Step 1 or an
		# unsuccessful Step 2 causes this coordinate sample to be discarded.
		if not _has_candidate_set(clean):
			continue
		observed_exact = run_candidate_attack(
			Y_observed, Q, X, B, noisy=False,
			attack_columns=exact_columns,
		)
		noisy = run_candidate_attack(
			Y_observed, Q, X, B, noisy=True, rho=rho_assumed,
			attack_columns=noisy_columns,
		)
		if not _has_candidate_set(noisy):
			continue
		break
	else:
		raise ValueError("no nonsingular coordinate sample found")

	error_abs = np.abs(observation["E"][:, :r])
	rho_true = int(np.max(error_abs))
	return {
		"dataset": dataset,
		"checkpoint_t0_5": CHECKPOINTS[dataset][0],
		"checkpoint_t1": CHECKPOINTS[dataset][1],
		"coordinate_seed": coordinate_seed,
		"attack_coordinates": ";".join(
			str(int(value))
			for value in coordinates
		),
		"model_dimension": int(local.shape[1]),
		"l": int(l),
		"r": int(r),
		"exact_step1_columns": int(exact_r),
		"noisy_step1_columns": int(r),
		"coordinate_selection": (
			"n_honest_default"
			if l == network["honest_nodes"].size else "explicit_override"
		),
		"paper_num_nodes": network["num_nodes"],
		"paper_num_edges": network["num_edges"],
		"paper_corrupt_ratio": network["corrupt_ratio"],
		"batch_size": PAPER_BATCH_SIZE,
		"model_architecture": DATASET_SPECS[dataset]["model_architecture"],
		"resample_count": resample_count,
		"scale": int(scale),
		"quantizer": quantizer,
		"Q_bits": int(Q.nbits()),
		"n": int(X.ncols()),
		"m": int(X.nrows()),
		"B": B,
		"network_scale_factor": network["scale_factor"],
		"rho_true": rho_true,
		"rho_true_all_sampled": observation["rho_true"],
		"rho_theoretical": observation["rho_theoretical"],
		"rho_quantization": observation["rho_quantization"],
		"rho_assumed": rho_assumed,
		"rho_bound_valid": (
			rho_true <= rho_assumed
		),
		"error_mean_abs": float(np.mean(error_abs)),
		"error_p95_abs": float(np.quantile(error_abs, 0.95)),
		"error_zero_fraction": float(np.mean(observation["E"] == 0)),
		"aggregation_max_abs": observation["aggregation_max_abs"],
		"model_abs_bound": observation["model_abs_bound"],
		"max_row_l1": observation["max_row_l1"],
		"clean_status": clean["status"],
		"clean_recall": clean["recall"],
		"clean_all_found": clean["all_found"],
		"clean_n_candidates": clean.get("n_candidates"),
		"observed_exact_status": observed_exact["status"],
		"observed_exact_recall": observed_exact["recall"],
		"observed_exact_all_found": observed_exact["all_found"],
		"observed_exact_n_candidates": observed_exact.get("n_candidates"),
		"noisy_status": noisy["status"],
		"noisy_recall": noisy["recall"],
		"noisy_all_found": noisy["all_found"],
		"noisy_n_candidates": noisy.get("n_candidates"),
		"noisy_step1_rows": noisy.get("step1_rows"),
		"noisy_tt_step1": noisy.get("tt_step1"),
		"error": None,
	}


def _present_values(values):
	return [value for value in values if value is not None]


def _has_candidate_set(attack_result):
	"""Return whether Step 1/Step 2 produced a usable candidate set.

	The original real-data attack resamples model coordinates after a
	singular Step 1 or when Nguyen--Stern returns no candidates.  The replay
	runner represents both situations in the result dictionary, so this helper
	keeps the retry policy in one place.
	"""
	return (
		attack_result.get("status") == "completed"
		and attack_result.get("n_candidates") not in (None, 0)
	)


def _mean_present(values):
	present = _present_values(values)
	return None if not present else sum(present) / len(present)


def _min_present(values):
	present = _present_values(values)
	return None if not present else min(present)


def _max_present(values):
	present = _present_values(values)
	return None if not present else max(present)


def summarize_checkpoint_replay(rows):
	groups = defaultdict(list)
	for row in rows:
		groups[(
			row.get("dataset"),
			row.get("scale"),
			row.get("quantizer"),
			row.get("r"),
			row.get("exact_step1_columns"),
		)].append(row)

	summary = []
	for key in sorted(groups, key=lambda value: tuple(repr(item) for item in value)):
		group = [row for row in groups[key] if not row.get("error")]
		total = len(group)
		clean_count = sum(bool(row["clean_all_found"]) for row in group)
		exact_count = sum(
			bool(row["observed_exact_all_found"]) for row in group
		)
		noisy_count = sum(bool(row["noisy_all_found"]) for row in group)
		ci_low, ci_high = wilson_interval(noisy_count, total)
		summary.append({
			"dataset": key[0],
			"scale": key[1],
			"quantizer": key[2],
			"r": key[3],
			"exact_step1_columns": key[4],
			"trials": len(groups[key]),
			"completed_trials": total,
			"failed_trials": len(groups[key]) - total,
			"clean_all_found_count": clean_count,
			"clean_all_found_rate": (
				None if not total else clean_count / total
			),
			"observed_exact_all_found_count": exact_count,
			"observed_exact_all_found_rate": (
				None if not total else exact_count / total
			),
			"noisy_all_found_count": noisy_count,
			"noisy_all_found_rate": (
				None if not total else noisy_count / total
			),
			"noisy_all_found_ci95_low": ci_low,
			"noisy_all_found_ci95_high": ci_high,
			"max_rho_true": (
				None if not total else
				max(row["rho_true"] for row in group)
			),
			"rho_theoretical": (
				None if not total else
				max(row["rho_theoretical"] for row in group)
			),
		"mean_clean_recall": (
				None if not total else
				sum(row["clean_recall"] for row in group) / total
			),
			"mean_clean_n_candidates": _mean_present(
				row.get("clean_n_candidates") for row in group
			),
			"min_clean_n_candidates": _min_present(
				row.get("clean_n_candidates") for row in group
			),
			"max_clean_n_candidates": _max_present(
				row.get("clean_n_candidates") for row in group
			),
			"mean_observed_exact_n_candidates": _mean_present(
				row.get("observed_exact_n_candidates") for row in group
			),
			"min_observed_exact_n_candidates": _min_present(
				row.get("observed_exact_n_candidates") for row in group
			),
			"max_observed_exact_n_candidates": _max_present(
				row.get("observed_exact_n_candidates") for row in group
			),
			"mean_noisy_n_candidates": _mean_present(
				row.get("noisy_n_candidates") for row in group
			),
			"min_noisy_n_candidates": _min_present(
				row.get("noisy_n_candidates") for row in group
			),
			"max_noisy_n_candidates": _max_present(
				row.get("noisy_n_candidates") for row in group
			),
			"mean_observed_exact_recall": (
				None if not total else
				sum(row["observed_exact_recall"] for row in group) / total
			),
			"mean_noisy_recall": (
				None if not total else
				sum(row["noisy_recall"] for row in group) / total
			),
		})
	return summary


def _write_csv(path, rows):
	path.parent.mkdir(parents=True, exist_ok=True)
	fields = sorted({field for row in rows for field in row})
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)


def run_checkpoint_replay(datasets, scales, quantizers, trials=1,
						  base_seed=20260727, output_dir=None,
						  trial_timeout_seconds=10, step1_columns=None,
						  exact_step1_columns=None, **trial_kwargs):
	rows = []
	r_values = (
		[None] if step1_columns is None
		else [int(value) for value in step1_columns]
	)
	exact_r_values = (
		[None] if exact_step1_columns is None
		else [int(value) for value in exact_step1_columns]
	)
	for dataset in datasets:
		for scale in scales:
			for quantizer in quantizers:
				for r in r_values:
					for exact_r in exact_r_values:
						for trial in range(trials):
							coordinate_seed = base_seed + trial
							try:
								trial_options = dict(trial_kwargs)
								trial_options["step1_columns"] = r
								trial_options["exact_step1_columns"] = exact_r
								row = run_trial_with_timeout(
									run_checkpoint_replay_trial,
									dataset,
									timeout_seconds=trial_timeout_seconds,
									scale=scale,
									quantizer=quantizer,
									coordinate_seed=coordinate_seed,
									**trial_options,
								)
							except Exception as exc:
								row = {
									"dataset": dataset,
									"scale": scale,
									"quantizer": quantizer,
									"r": r,
									"exact_step1_columns": exact_r,
									"coordinate_seed": coordinate_seed,
									"error": f"{type(exc).__name__}:{exc}",
								}
							rows.append(row)
							effective_r = row.get("r", r)
							effective_exact_r = row.get(
								"exact_step1_columns", exact_r
							)
							print(
								f"{dataset} scale={scale} {quantizer} "
								f"r={effective_r} exact-r={effective_exact_r} "
								f"seed={coordinate_seed}: "
								f"rho={row.get('rho_true')}/"
								f"{row.get('rho_assumed')} "
								f"clean={row.get('clean_recall')} "
								f"exact-observed={row.get('observed_exact_recall')} "
								f"noisy={row.get('noisy_recall')} "
								f"candidates="
								f"{row.get('clean_n_candidates')}/"
								f"{row.get('observed_exact_n_candidates')}/"
								f"{row.get('noisy_n_candidates')} "
								f"error={row.get('error')}",
								flush=True,
							)

	summary = summarize_checkpoint_replay(rows)
	if output_dir is not None:
		output_dir = Path(output_dir)
		_write_csv(output_dir / "checkpoint_replay_trials.csv", rows)
		_write_csv(output_dir / "checkpoint_replay_summary.csv", summary)
	return rows, summary


def _parse_args():
	parser = argparse.ArgumentParser(
		description="Replay fixed-point error on actual DFL checkpoints."
	)
	parser.add_argument(
		"--datasets",
		nargs="+",
		choices=sorted(CHECKPOINTS),
		default=["sentiment140"],
	)
	parser.add_argument(
		"--scales", nargs="+", type=int, default=[PAPER_QUANTIZE]
	)
	parser.add_argument(
		"--quantizers",
		nargs="+",
		choices=["nearest", "truncate"],
		default=["nearest", "truncate"],
	)
	parser.add_argument(
		"--trials", type=int, default=1,
		help="number of coordinate trials per condition (default: 1)",
	)
	parser.add_argument("--base-seed", type=int, default=20260727)
	parser.add_argument(
		"--l", type=int, default=None,
		help="number of coordinates for Step 1 (default: n_honest, as in the paper)",
	)
	parser.add_argument(
		"--step1-columns", "--r",
		dest="step1_columns",
		nargs="+",
		type=int,
		default=None,
		help="noisy Step 1 column counts; values may exceed n_honest",
	)
	parser.add_argument(
		"--exact-step1-columns", "--exact-r",
		dest="exact_step1_columns",
		nargs="+",
		type=int,
		default=None,
		help="exact/observed-exact Step 1 column counts (default: n_honest)",
	)
	parser.add_argument("--nx0", type=int, default=PAPER_PRIME_BITS)
	parser.add_argument("--rho-multiplier", type=float, default=1.0)
	parser.add_argument(
		"--max-resample", type=int, default=PAPER_MAX_RESAMPLE
	)
	parser.add_argument("--model-dir", type=Path, default=paths.MODEL_DIR)
	parser.add_argument(
		"--network-path", type=Path, default=paths.NETWORK_MAT
	)
	parser.add_argument("--trial-timeout", type=int, default=10)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=paths.RESULTS / "finite_precision",
	)
	return parser.parse_args()


def main():
	args = _parse_args()
	_rows, summary = run_checkpoint_replay(
		args.datasets,
		args.scales,
		args.quantizers,
		trials=args.trials,
		base_seed=args.base_seed,
		output_dir=args.output_dir,
		trial_timeout_seconds=args.trial_timeout,
		l=args.l,
		nx0=args.nx0,
		step1_columns=args.step1_columns,
		exact_step1_columns=args.exact_step1_columns,
		rho_multiplier=args.rho_multiplier,
		max_resample=args.max_resample,
		model_dir=args.model_dir,
		network_path=args.network_path,
	)
	print("\nSummary")
	for row in summary:
		print(row)
	print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
	main()

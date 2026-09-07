"""Shared evaluation helpers for noisy HSSP/HLCP experiments.

This module is intentionally independent of the synthetic truncation and
paper-topology replay runners.  The real-checkpoint runner uses it to execute
exact/noisy candidate attacks and to compute Wilson confidence intervals.
"""

import math

from sage.all import Matrix

from hssp_dfl.attacks.nguyen_stern import nguyen_stern
from hssp_dfl.attacks.noisy_step1 import noisy_step1_original
from hssp_dfl.attacks.step1 import step1_original
from hssp_dfl.experiments.common import select_attack_columns


def candidate_recall(X_candidates, X_true):
	"""Return matched-column count, recall, and full-column success.

	The lattice attack can return a sign-flipped representative, so both a
	ground-truth column and its negative are accepted for evaluation.
	"""
	if X_candidates is None:
		return 0, 0.0, False

	candidates = Matrix(X_candidates)
	matched = 0
	for true_column in X_true.columns():
		true_values = list(true_column)
		negative_values = [-value for value in true_values]
		if any(
			list(candidates.column(j)) in (true_values, negative_values)
			for j in range(candidates.ncols())
		):
			matched += 1

	n = X_true.ncols()
	return matched, matched / n, matched == n


def run_candidate_attack(Y, Q, X, B, noisy=False, rho=None,
						 attack_columns=None):
	"""Run exact or noisy Step 1/Step 2 and evaluate true-column recall."""
	n = X.ncols()
	m = X.nrows()
	if attack_columns is None:
		Y_attack, attack_columns = select_attack_columns(Y, Q, n)
	else:
		attack_columns = list(attack_columns)
		if not attack_columns:
			Y_attack = None
		else:
			if min(attack_columns) < 0 or max(attack_columns) >= Y.ncols():
				raise ValueError("attack column index is outside Y")
			Y_attack = Y.matrix_from_columns(attack_columns)
	if Y_attack is None:
		return {
			"status": "no_attack_columns",
			"matched": 0,
			"recall": 0.0,
			"all_found": False,
			"n_candidates": None,
		}

	try:
		if noisy:
			MO, tt1, debug = noisy_step1_original(
				n, m, Q, Y_attack, rho=rho, return_debug=True
			)
			step1_rows = debug["Y_ortho"].nrows()
		else:
			MO, tt1 = step1_original(n, m, Q, Y_attack)
			step1_rows = MO.nrows()

		# Step 2: Nguyen-Stern candidate recovery.  kappa = -1 lets the
		# routine derive the bound from B itself.
		_beta, _elapsed, X_candidates = nguyen_stern(n, m, -1, MO, B=B, X=None)
		matched, recall, all_found = candidate_recall(
			X_candidates, X
		)
		return {
			"status": "completed",
			"matched": matched,
			"recall": recall,
			"all_found": all_found,
			"n_candidates": (
				None if X_candidates is None else X_candidates.ncols()
			),
			"tt_step1": float(tt1),
			"step1_rows": step1_rows,
			"attack_columns": attack_columns,
		}
	except Exception as exc:
		return {
			"status": f"error:{type(exc).__name__}:{exc}",
			"matched": 0,
			"recall": 0.0,
			"all_found": False,
			"n_candidates": None,
		}


def wilson_interval(successes, total, z=1.96):
	"""Compute a Wilson confidence interval for a binomial proportion."""
	if total <= 0:
		return None, None

	proportion = successes / total
	z2 = z * z
	denominator = 1 + z2 / total
	center = (proportion + z2 / (2 * total)) / denominator
	margin = (
		z
		* math.sqrt(
			proportion * (1 - proportion) / total
			+ z2 / (4 * total * total)
		)
		/ denominator
	)
	return max(0.0, center - margin), min(1.0, center + margin)

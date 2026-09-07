"""
Noisy Step 1 — augmented lattice construction and reduction.

This module contains the noisy analogue of Step 1 used by the noisy
experiments. It builds the augmented noisy lattice, extracts short projected
rows, and returns an n x m kernel matrix for Step 2.
"""

from math import ceil, sqrt

from sage.all import ZZ, Matrix, vector, cputime

from hssp_dfl.lattice import kernel_lll, noisy_lattice_mat


def estimate_beta_noisy(rho, r, safety=1.25, minimum=1):
	if r <= 0:
		raise ValueError("r must be positive")
	return max(minimum, int(ceil(float(safety) * float(rho) * sqrt(float(r)))))


def _norm_sq(v):
	return sum(ZZ(x) * ZZ(x) for x in v)


def _extract_noisy_y_rows(Mred, public_rows, beta, slack_bound=None):
	"""Extract projected y rows from rows shaped as (beta*y | z)."""
	rows = [Mred.row(i) for i in range(Mred.nrows())]
	rows.sort(key=_norm_sq)

	out = []
	for row in rows:
		scaled_y = [ZZ(x) for x in row[:public_rows]]
		if any(x % beta != 0 for x in scaled_y):
			continue
		y = vector(ZZ, [x // beta for x in scaled_y])
		if y == 0:
			continue

		slack = vector(ZZ, list(row[public_rows:]))
		if slack_bound is not None and float(slack.norm()) / float(y.norm()) > float(slack_bound):
			continue
		if y not in out:
			out.append(y)
	return out


def noisy_step1_original(n, m, Q, H_tilde, beta=None, rho=None, slack_bound=None,
						 return_debug=False):
	"""Noisy replacement for Step 1 returning an n x m MO matrix."""
	H_tilde = Matrix(ZZ, H_tilde)
	if H_tilde.nrows() != m:
		raise ValueError("H_tilde must have m rows")
	if beta is None:
		if rho is None:
			raise ValueError("provide either beta or rho")
		beta = estimate_beta_noisy(rho, H_tilde.ncols())

	target_rank = m - n
	print("Building noisy augmented lattice...")
	Maug = noisy_lattice_mat(H_tilde, Q, beta)

	print("Noisy Step 1")
	t = cputime()
	Mred = Maug.LLL()

	y_rows = _extract_noisy_y_rows(Mred, m, beta, slack_bound=slack_bound)
	Y_ortho = Matrix(ZZ, [list(row) for row in y_rows[:target_rank]])

	MO = kernel_lll(Y_ortho)
	tt1 = cputime(t)

	print(f"  beta={beta}, selected y-rows={Y_ortho.nrows()}, MO={MO.dimensions()}")
	print(f"  Total noisy Step 1: {tt1:.1f}s")

	if return_debug:
		return MO, tt1, {
			"beta": beta,
			"Maug": Maug,
			"Mred": Mred,
			"Y_ortho": Y_ortho,
			"candidate_y_rows": y_rows,
		}
	return MO, tt1

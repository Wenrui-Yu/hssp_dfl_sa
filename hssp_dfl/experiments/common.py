"""Shared helpers for experiment runners."""

import math
import multiprocessing
import queue
import random
import signal
from contextlib import contextmanager
from itertools import combinations, islice

from sage.all import (
	QQ,
	ZZ,
	Matrix,
	Integers,
	MixedIntegerLinearProgram,
	mod,
	set_random_seed,
)

from hssp_dfl.utils import gen_pseudoprime


class ExperimentTimeoutError(TimeoutError):
	"""Raised when one experiment trial exceeds its hard timeout."""


@contextmanager
def trial_timeout(seconds):
	"""Apply a Unix hard timeout to one trial; non-positive disables it."""
	if seconds is None or seconds <= 0:
		yield
		return
	if not hasattr(signal, "SIGALRM"):
		yield
		return

	def _raise_timeout(_signum, _frame):
		raise ExperimentTimeoutError(
			f"experiment trial exceeded {seconds} seconds"
		)

	previous_handler = signal.getsignal(signal.SIGALRM)
	signal.signal(signal.SIGALRM, _raise_timeout)
	signal.alarm(int(math.ceil(seconds)))
	try:
		yield
	finally:
		signal.alarm(0)
		signal.signal(signal.SIGALRM, previous_handler)


def _trial_process_worker(output_queue, function, args, kwargs):
	try:
		output_queue.put(("ok", function(*args, **kwargs)))
	except BaseException as exc:
		output_queue.put((
			"error",
			f"{type(exc).__name__}:{exc}",
		))


def run_trial_with_timeout(function, *args, timeout_seconds=300, **kwargs):
	"""Run one trial in a killable child process.

	A process boundary is required because Sage/FLINT C routines do not always
	return control to Python when ``SIGALRM`` is delivered.
	"""
	if timeout_seconds is None or timeout_seconds <= 0:
		return function(*args, **kwargs)

	try:
		context = multiprocessing.get_context("fork")
	except ValueError:
		context = multiprocessing.get_context("spawn")
	output_queue = context.Queue(maxsize=1)
	process = context.Process(
		target=_trial_process_worker,
		args=(output_queue, function, args, kwargs),
	)
	process.start()
	process.join(timeout_seconds)
	if process.is_alive():
		process.terminate()
		process.join(5)
		if process.is_alive() and hasattr(process, "kill"):
			process.kill()
			process.join(5)
		output_queue.close()
		raise ExperimentTimeoutError(
			f"experiment trial exceeded {timeout_seconds} seconds"
		)

	try:
		status, payload = output_queue.get(timeout=1)
	except queue.Empty as exc:
		raise RuntimeError(
			f"trial process exited with code {process.exitcode} without a result"
		) from exc
	finally:
		output_queue.close()
	if status == "error":
		raise RuntimeError(payload)
	return payload


def generate_X(m, n, kappa, B, rng):
	X = Matrix(ZZ, m, n)
	if kappa > 0:
		for i in range(m):
			positions = rng.sample(range(n), kappa)
			for j in positions:
				X[i, j] = rng.randint(1, B)
	else:
		for i in range(m):
			for j in range(n):
				X[i, j] = rng.randint(0, B)
	return X


def select_attack_columns(Y, Q, n, r=None):
	"""Pick r public columns usable by Step 1.

	The mathematical requirement is full column rank. The current
	``ortho_lattice_mat`` implementation also pivots on the leading r rows, so
	we choose a column subset whose leading r x r block is invertible.
	"""
	l = Y.ncols()
	if r is None:
		r = min(n, l)
	if not 1 <= r <= n:
		raise ValueError("attack column count r must satisfy 1 <= r <= n")
	if r > l:
		raise ValueError("attack column count r cannot exceed l")

	for cols in combinations(range(l), r):
		Y_sel = Y.matrix_from_columns(cols)
		if Matrix(Integers(Q), Y_sel).rank() != r:
			continue
		if Matrix(Integers(Q), Y_sel[:r, :r]).rank() == r:
			return Y_sel, list(cols)
	return None, None


def generate_instance(n, l, m, nx0, kappa, B, seed=None, attack_rank=None):
	rng = random.Random(seed)
	if seed is not None:
		set_random_seed(seed)
	Q = gen_pseudoprime(nx0)

	for _ in range(500):
		A = Matrix(ZZ, n, l)
		for i in range(n):
			for j in range(l):
				A[i, j] = rng.randrange(1000)

		X = None
		for _ in range(200):
			X = generate_X(m, n, kappa, B, rng)
			if X.rank() == n:
				break
		if X is None or X.rank() != n:
			continue

		Y = X * A
		for i in range(m):
			for j in range(l):
				Y[i, j] = mod(Y[i, j], Q)

		Y_attack, attack_columns = select_attack_columns(Y, Q, n, attack_rank)
		if Y_attack is not None:
			return Q, A, X, Y, Y_attack, attack_columns

	raise ValueError("failed to sample an instance with usable attack columns")


def match_secret_columns(X_rec, X):
	if X_rec is None or X_rec.dimensions() != X.dimensions():
		return False, None

	X_rec_zz = Matrix(ZZ, X_rec)
	X_zz = Matrix(ZZ, X)
	used = set()
	perm = []
	for j in range(X_zz.ncols()):
		rec_col = list(X_rec_zz.column(j))
		match = None
		for k in range(X_zz.ncols()):
			if k not in used and rec_col == list(X_zz.column(k)):
				match = k
				break
		if match is None:
			return False, None
		used.add(match)
		perm.append(match)
	return True, perm


def align_A(A_rec, perm, Q):
	if A_rec is None or perm is None:
		return None

	A_aligned = Matrix(Integers(Q), A_rec.nrows(), A_rec.ncols())
	for recovered_col, original_col in enumerate(perm):
		A_aligned[original_col, :] = A_rec[recovered_col, :]
	return A_aligned


def is_bounded_matrix(M, B):
	if M is None:
		return False
	return all(0 <= ZZ(x) <= B for row in M for x in row)


def has_row_weight(M, kappa):
	if M is None or kappa is None or kappa <= 0:
		return True
	return all(sum(1 for x in row if ZZ(x) != 0) == kappa for row in M)


def recover_A(X_rec, Y, Q, seed=None):
	if X_rec is None:
		return None, None

	rng = random.Random(seed)
	m, n = X_rec.dimensions()
	rows = list(range(m))

	for _ in range(200):
		rng.shuffle(rows)
		pick = rows[:n]
		Xb = Matrix(Integers(Q), [X_rec[i] for i in pick])
		try:
			inv = Xb.inverse()
		except (ZeroDivisionError, ArithmeticError):
			continue
		Yb = Matrix(Integers(Q), [Y[i] for i in pick])
		return inv * Yb, pick

	return None, None


def centered_mod(x, Q):
	"""Return the representative of ``x mod Q`` in ``[-Q/2, Q/2]``."""
	x = ZZ(x) % Q
	if x > Q // 2:
		x -= Q
	return x


def centered_lift_matrix(M, Q):
	"""Lift a modular matrix entry-wise to centered integers."""
	M = Matrix(ZZ, M)
	return Matrix(
		ZZ,
		M.nrows(),
		M.ncols(),
		[centered_mod(x, Q) for x in M.list()],
	)


def centered_residual(Y_observed, X_rec, A_rec, Q):
	"""Compute the centered modular residual ``Y - X*A``."""
	if X_rec is None or A_rec is None:
		return None
	raw = Matrix(ZZ, Y_observed) - Matrix(ZZ, X_rec) * Matrix(ZZ, A_rec)
	return centered_lift_matrix(raw, Q)


def residual_statistics(residual, rho=None):
	"""Summarize a residual matrix for noisy candidate ranking."""
	if residual is None:
		return None

	absolute = [abs(ZZ(x)) for x in residual.list()]
	if not absolute:
		return {
			"max_abs": ZZ(0),
			"l1": ZZ(0),
			"frobenius_sq": ZZ(0),
			"rmse": 0.0,
			"within_bound": True,
		}

	max_abs = max(absolute)
	frobenius_sq = sum(x * x for x in absolute)
	return {
		"max_abs": max_abs,
		"l1": sum(absolute),
		"frobenius_sq": frobenius_sq,
		"rmse": float(frobenius_sq / len(absolute)) ** 0.5,
		"within_bound": rho is None or max_abs <= rho,
	}


def _round_rational_matrix(M):
	out = Matrix(ZZ, M.nrows(), M.ncols())
	for i in range(M.nrows()):
		for j in range(M.ncols()):
			value = M[i, j]
			try:
				out[i, j] = ZZ(value.round())
			except AttributeError:
				out[i, j] = ZZ(round(float(value)))
	return out


def _candidate_row_subsets(m, n, max_row_subsets, rng):
	"""Return deterministic-plus-random row subsets without materializing C(m,n)."""
	if max_row_subsets is None:
		return list(combinations(range(m), n))
	if max_row_subsets <= 0:
		return []

	head_count = min(max_row_subsets, max(1, max_row_subsets // 2))
	subsets = list(islice(combinations(range(m), n), head_count))
	seen = set(subsets)
	max_attempts = max(100, 20 * max_row_subsets)
	for _ in range(max_attempts):
		if len(subsets) >= max_row_subsets:
			break
		pick = tuple(sorted(rng.sample(range(m), n)))
		if pick not in seen:
			seen.add(pick)
			subsets.append(pick)
	return subsets


def _recover_A_minimax(X, Y_lift):
	"""Solve integer Chebyshev regression for all coordinates at once."""
	m, n = X.dimensions()
	l = Y_lift.ncols()
	program = MixedIntegerLinearProgram(maximization=False)
	a = program.new_variable(integer=True)
	t = program.new_variable(nonnegative=True)

	for i in range(m):
		for j in range(l):
			prediction = program.sum(ZZ(X[i, k]) * a[k, j] for k in range(n))
			program.add_constraint(prediction - ZZ(Y_lift[i, j]) <= t[0])
			program.add_constraint(ZZ(Y_lift[i, j]) - prediction <= t[0])
	program.set_objective(t[0])
	program.solve()
	values = program.get_values(a)
	return Matrix(
		ZZ,
		n,
		l,
		[ZZ(round(values[i, j])) for i in range(n) for j in range(l)],
	)


def recover_A_noisy(X_rec, Y_tilde, Q, rho=None, seed=None,
					max_row_subsets=200, return_debug=False):
	"""Recover an integer ``A`` from bounded-noise observations.

	The routine assumes the fixed-point encoding was chosen without modular
	wrap-around, so centered lifting maps the observations back to their signed
	integer representatives. It evaluates an all-row least-squares candidate
	and candidates obtained from several invertible row subsets, rounds them to
	integers, and keeps the candidate with the smallest centered modular
	residual.
	"""
	if X_rec is None:
		if return_debug:
			return None, None, {
				"accepted": False,
				"reason": "X_rec is None",
				"candidates_tested": 0,
			}
		return None, None

	X_zz = Matrix(ZZ, X_rec)
	Y_lift = centered_lift_matrix(Y_tilde, Q)
	m, n = X_zz.dimensions()
	if Y_lift.nrows() != m:
		raise ValueError("Y_tilde and X_rec must have the same number of rows")
	if X_zz.rank() != n:
		if return_debug:
			return None, None, {
				"accepted": False,
				"reason": "X_rec does not have full column rank",
				"candidates_tested": 0,
			}
		return None, None

	rng = random.Random(seed)
	candidates = []

	try:
		A_minimax = _recover_A_minimax(X_zz, Y_lift)
		candidates.append((A_minimax, None, "all_rows_integer_minimax"))
	except Exception:
		# Some Sage installations omit a MILP backend. The exact-rational
		# least-squares and row-subset fallbacks remain available.
		pass

	X_q = Matrix(QQ, X_zz)
	Y_q = Matrix(QQ, Y_lift)
	gram = X_q.T * X_q
	try:
		A_least_squares = _round_rational_matrix(gram.inverse() * X_q.T * Y_q)
		candidates.append((A_least_squares, None, "all_rows_least_squares"))
	except (ZeroDivisionError, ArithmeticError):
		pass

	for pick in _candidate_row_subsets(m, n, max_row_subsets, rng):
		X_block = Matrix(QQ, [X_zz[i] for i in pick])
		try:
			A_candidate = _round_rational_matrix(
				X_block.inverse() * Matrix(QQ, [Y_lift[i] for i in pick])
			)
		except (ZeroDivisionError, ArithmeticError):
			continue
		candidates.append((A_candidate, list(pick), "invertible_row_subset"))

	best = None
	seen = set()
	for A_candidate, rows, method in candidates:
		key = tuple(A_candidate.list())
		if key in seen:
			continue
		seen.add(key)
		residual = centered_residual(Y_tilde, X_zz, A_candidate, Q)
		stats = residual_statistics(residual, rho=rho)
		score = (stats["max_abs"], stats["frobenius_sq"], stats["l1"])
		if best is None or score < best["score"]:
			best = {
				"A": A_candidate,
				"rows": rows,
				"method": method,
				"residual": residual,
				"stats": stats,
				"score": score,
			}

	if best is None:
		debug = {
			"accepted": False,
			"reason": "no invertible recovery system",
			"candidates_tested": 0,
		}
		if return_debug:
			return None, None, debug
		return None, None

	debug = {
		"accepted": best["stats"]["within_bound"],
		"reason": None if best["stats"]["within_bound"] else "residual exceeds rho",
		"candidates_tested": len(seen),
		"method": best["method"],
		"rows": best["rows"],
		"residual": best["residual"],
		**best["stats"],
	}
	if return_debug:
		return best["A"], best["rows"], debug
	return best["A"], best["rows"]


def valid_solution(X_rec, A_rec, Y, Q):
	return (
		X_rec is not None
		and A_rec is not None
		and Matrix(Integers(Q), X_rec) * A_rec == Matrix(Integers(Q), Y)
	)


def select_solution_X(X_candidates, Y, Q, n, B, kappa=None, seed=None):
	if X_candidates is None:
		return None

	X_candidates = Matrix(ZZ, X_candidates)
	m, n_candidates = X_candidates.dimensions()
	if n_candidates == n:
		if not is_bounded_matrix(X_candidates, B):
			return None
		if not has_row_weight(X_candidates, kappa):
			return None
		A_rec, _rows = recover_A(X_candidates, Y, Q, seed=seed)
		if valid_solution(X_candidates, A_rec, Y, Q):
			return X_candidates
		return None
	if n_candidates < n:
		return None

	for cols in combinations(range(n_candidates), n):
		X_rec = Matrix(ZZ, m, n)
		for j, col in enumerate(cols):
			X_rec[:, j] = X_candidates[:, col]
		if not is_bounded_matrix(X_rec, B):
			continue
		if not has_row_weight(X_rec, kappa):
			continue
		A_rec, _rows = recover_A(X_rec, Y, Q, seed=seed)
		if valid_solution(X_rec, A_rec, Y, Q):
			return X_rec

	return None


def select_solution_X_noisy(X_candidates, Y_tilde, Q, n, B, kappa=None,
							rho=0, seed=None, max_row_subsets=200,
							return_details=False):
	"""Select candidate columns using only noisy observations and a noise bound.

	Every structurally valid ``n``-column combination is scored by the best
	bounded-noise recovery of ``A``. Only combinations whose maximum centered
	residual is at most ``rho`` are accepted.
	"""
	empty_debug = {
		"accepted": False,
		"reason": "no candidates",
		"column_combinations_tested": 0,
		"structural_candidates": 0,
	}
	if X_candidates is None:
		if return_details:
			return None, None, empty_debug
		return None

	X_candidates = Matrix(ZZ, X_candidates)
	m, n_candidates = X_candidates.dimensions()
	if n_candidates < n:
		debug = dict(empty_debug)
		debug["reason"] = "fewer than n candidate columns"
		if return_details:
			return None, None, debug
		return None

	column_sets = [tuple(range(n))] if n_candidates == n else combinations(
		range(n_candidates), n
	)
	best_accepted = None
	best_rejected = None
	tested = 0
	structural = 0

	for cols in column_sets:
		tested += 1
		X_rec = Matrix(ZZ, m, n)
		for j, col in enumerate(cols):
			X_rec[:, j] = X_candidates[:, col]
		if not is_bounded_matrix(X_rec, B):
			continue
		if not has_row_weight(X_rec, kappa):
			continue
		structural += 1

		A_rec, rows, recovery = recover_A_noisy(
			X_rec,
			Y_tilde,
			Q,
			rho=rho,
			seed=seed,
			max_row_subsets=max_row_subsets,
			return_debug=True,
		)
		if A_rec is None or "max_abs" not in recovery:
			continue
		item = {
			"X": X_rec,
			"A": A_rec,
			"columns": list(cols),
			"rows": rows,
			"recovery": recovery,
			"score": (
				recovery["max_abs"],
				recovery["frobenius_sq"],
				recovery["l1"],
			),
		}
		if best_rejected is None or item["score"] < best_rejected["score"]:
			best_rejected = item
		if recovery["accepted"] and (
			best_accepted is None or item["score"] < best_accepted["score"]
		):
			best_accepted = item

	chosen = best_accepted
	if chosen is None:
		debug = {
			"accepted": False,
			"reason": (
				"no structurally valid candidate satisfies the residual bound"
				if best_rejected is not None
				else "no structurally valid candidate"
			),
			"column_combinations_tested": tested,
			"structural_candidates": structural,
		}
		if best_rejected is not None:
			debug.update({
				"best_rejected_columns": best_rejected["columns"],
				"best_rejected_recovery": best_rejected["recovery"],
			})
		if return_details:
			return None, None, debug
		return None

	debug = {
		"accepted": True,
		"reason": None,
		"columns": chosen["columns"],
		"rows": chosen["rows"],
		"column_combinations_tested": tested,
		"structural_candidates": structural,
		**chosen["recovery"],
	}
	if return_details:
		return chosen["X"], chosen["A"], debug
	return chosen["X"]


def print_matrix(name, value):
	print(f"{name}:")
	if value is None:
		print("None")
	else:
		print(value)

"""
Multivariate attack (Step 2) for HSSP.

Recovers the secret vectors using eigenvalue decomposition of the
kernel of the extended quadratic system.

Two strategies:
- multivariate_eigen          : Full eigenspace decomposition (kappa = -1).
- multivariate_bit_guessing   : Eigenspace + bit-guessing (kappa > 0).
"""

from sage.all import (
    ZZ, GF, Matrix, matrix, vector, identity_matrix,
    Integers, cputime, ones_matrix, variance,
)
from itertools import combinations
from math import comb

from hssp_dfl.utils import negate_binary
from hssp_dfl.attacks.bkz import recover_binary


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_extended_quadratic_kernel(MO, n, p=3):
    """Build the kernel of the extended quadratic system.

    Constructs the matrix whose rows encode both linear and quadratic
    relations in the secret vectors, then computes its right kernel.

    Parameters
    ----------
    MO : Matrix
        Kernel matrix from Step 1 (n x m).
    n : int
        Dimension.
    p : int, optional
        Characteristic of the base field (default 3).

    Returns
    -------
    Matrix
        The full kernel matrix ke23 of shape (dim_kernel, n*n).
    """
    K = GF(p)

    # Each column of MO.T gives a vector x.
    # Extend with quadratic products: x_i * x_j for i <= j.
    xt23 = Matrix(K, [
        (-x).list() + [
            x[i] * x[j] * (1 if i == j else 2)
            for i in range(n)
            for j in range(i, n)
        ]
        for x in MO.T
    ])

    ke3 = xt23.right_kernel().matrix()
    # print(f"  dim(ker E) = {ke3.dimensions()}")

    # Expand compressed symmetric kernel to full n x n blocks
    ke23 = Matrix(K, ke3.nrows(), n * n)
    ind = n
    for i in range(n):
        for j in range(i, n):
            ke23[:, i * n + j] = ke3[:, ind]
            ke23[:, j * n + i] = ke3[:, ind]
            ind += 1

    return ke23


def _count_unique_matches(candidates, X):
    X_rows = {tuple(ZZ(x) for x in row) for row in X.T}
    found = {
        tuple(ZZ(x) for x in row)
        for row in candidates
        if tuple(ZZ(x) for x in row) in X_rows
    }
    return len(found)


def _eigenspace_decomposition(ke23, n):
    """Split the eigenspaces by iterating over coordinates.

    For kappa = -1 (no Hamming constraint), this yields the
    exact eigenvectors. For kappa > 0, use bit_guessing instead.

    Parameters
    ----------
    ke23 : Matrix
        Full kernel matrix (dim_kernel x n*n).
    n : int
        Dimension.

    Returns
    -------
    list of Matrix
        List of 1-dim eigenspaces (each a 1 x n matrix over GF(3)).
    """
    K = GF(3)
    if ke23.nrows() != n:
        raise ValueError(
            f"expected quadratic kernel dimension {n}, got {ke23.nrows()}"
        )
    li = [Matrix(K, identity_matrix(n))]

    for j in range(n):
        M = ke23[:, j * n:(j + 1) * n]
        li2 = []
        for v in li:
            if v.nrows() == 1:
                li2.append(v)
            else:
                A = v.T.solve_right((v * M).T).T
                for _eigenval, v2 in A.eigenspaces_left():
                    li2.append(v2.matrix() * v)
        li = li2

    return li


def _eigenspace_guess(ke23, MO, n, kappa, verbose=False):
    """Split eigenspaces using coordinate guessing (for kappa > 0).

    Parameters
    ----------
    ke23 : Matrix
        Full kernel matrix.
    MO : Matrix
        Step 1 kernel.
    n : int
        Dimension.
    kappa : int
        Hamming weight constraint.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    list of vectors
        Recovered eigenvectors (each of length n, over GF(3)).
    """
    K = GF(3)

    def _space_intersection(V1, V2):
        """Compute intersection of two row-spaces."""
        VI = Matrix(K, V1.nrows() + V2.nrows(), V1.ncols())
        VI[:V1.nrows(), :] = V1
        VI[V1.nrows():, :] = V2
        lk = VI.left_kernel().matrix()
        return lk[:, :V1.nrows()] * V1

    li = [identity_matrix(ke23.nrows())]

    for j in range(n):
        ma = max(v.nrows() for v in li)
        if verbose:
            print(f"  j={j}, len(li)={len(li)}, "
                  f"tot={sum(v.nrows() for v in li)}, max={ma}")
        if ma == 1:
            break

        M = Matrix(GF(3), ke23.nrows(), n)
        for i in range(n):
            M[:, i] = ke23[:, i * n:(i + 1) * n] * MO[:, j]

        l2 = [V1 for V1 in li if V1.nrows() == 1]

        for e in range(2):
            if kappa > 0 and j == 0 and e == 1:
                continue  # Constant Hamming weight optimization
            Me = M[:, :]
            Me[:n, :] -= e * identity_matrix(n)
            V = Me.left_kernel().matrix()
            for V1 in li:
                if V1.nrows() > 1:
                    inter = _space_intersection(V1, V)
                    if inter.nrows() > 0:
                        l2.append(inter)
        li = l2

    result = [v[0][:n] for v in li]
    print(f"  Depth: {j}")
    return result


def _switching_phase(NS, n, m, kappa):
    """Resolve sign ambiguity via variance-based switching.

    For kappa > 0, flips rows of the recovered matrix until
    the column variance reaches zero (correct assignment).

    Parameters
    ----------
    NS : Matrix
        Candidate secret vectors (n x m).
    n : int
        Dimension.
    m : int
        Number of samples.
    kappa : int
        Hamming weight.

    Returns
    -------
    tuple
        (Y, tsf, n_found) where Y is the (possibly corrected) matrix.
    """
    from sage.all import copy

    ones = vector([1] * m)
    li = NS.rows()
    for NSi in NS:
        if ones - NSi not in li:
            li.append(ones - NSi)
    NS = matrix(li)

    ts = cputime()
    e = vector([1] * n)
    Y = matrix(ZZ, NS[:n] if kappa <= n // 2 else NS[n:2 * n])
    Y = copy(Y)
    v0 = variance(e * Y).n()
    i = 0

    while True:
        j = i % n
        Y[j] = negate_binary(Y[j])
        v1 = variance(e * Y).n()
        if v1 == 0:
            break
        if v1 < v0:
            v0 = v1
        else:
            Y[j] = negate_binary(Y[j])
        i += 1
        if i >= 500:
            raise RuntimeError("Switching phase did not converge in 500 iterations")

    tsf = cputime(ts)
    print(f"  Switching rounds: {i}, time: {tsf:.1f}s")

    return Y.T, tsf


def _select_constant_weight(rows, n, m, kappa):
    """Select n binary rows whose sample-wise sum is kappa."""
    ones = vector([1] * m)
    candidates = []
    def add_binary(candidate):
        if all(x in (0, 1) for x in candidate) and candidate not in candidates:
            candidates.append(candidate)

    for row in rows:
        add_binary(row)
        add_binary(ones - row)

    changed = True
    while changed and len(candidates) <= 80:
        changed = False
        for left in list(candidates):
            for right in list(candidates):
                before = len(candidates)
                add_binary(left + right)
                add_binary(left - right)
                if len(candidates) != before:
                    changed = True

    if len(candidates) < n:
        return None
    if len(candidates) > 80 or comb(len(candidates), n) > 200000:
        return None

    target = vector([kappa] * m)
    zero = vector([0] * m)
    for picked in combinations(candidates, n):
        if sum(picked, zero) == target:
            Y = matrix(ZZ, picked)
            if Y.rank() == n:
                return Y.T

    for picked in combinations(candidates, n - 1):
        missing = target - sum(picked, zero)
        if all(x in (0, 1) for x in missing) and missing not in picked:
            Y = matrix(ZZ, list(picked) + [missing])
            if Y.rank() == n:
                return Y.T
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def multivariate_eigen( MO,n, X=None):
    """Multivariate attack via full eigenspace decomposition.

    Best for kappa = -1 (no Hamming weight constraint).

    Parameters
    ----------
    n : int
        Dimension.
    MO : Matrix
        Kernel matrix from Step 1 (n x m, over GF(3)).
    X : Matrix
        Secret matrix.

    Returns
    -------
    tuple
        (t_total, NS) — timing and recovered matrix.
    """
    print("Step 2 (Multivariate Eigen)")
    t2 = cputime()

    ke23 = _build_extended_quadratic_kernel(MO, n)

    li = _eigenspace_decomposition(ke23, n)

    # Reconstruct NS matrix
    NS = Matrix([v[0] for v in li]) * MO
    for i in range(n):
        if any(c == 2 for c in NS[i]):
            NS[i] = -NS[i]
    # print(f"  Recovered vectors: {NS.nrows()}")

    if X is not None:
        n_found = _count_unique_matches(NS, X)
        print(f"  NFound={n_found} out of {n}")

    # NS = NS.T
    # invNSn = matrix(Integers(x0), NS[:n]).inverse()
    # ra = invNSn * b[:n]
    # n_a_found = sum(1 for rai in ra if rai in a)
    # print(f"  Coefficients found: {n_a_found} out of {n}")

    t_total = cputime(t2)
    print(f"  Total Step 2: {t_total:.1f}s")

    return t_total, NS

def multivariate_bit_guessing(MO,n,m, kappa,  X=None):
    """Modified bit-guessing that also returns recovered matrix and beta.

    Returns
    -------
    tuple
        (MB, beta, n_recovered) where MB is the recovered binary matrix.
    """
    print("Step 2 (Bit Guessing Modified)")

    ke23 = _build_extended_quadratic_kernel(MO, n)

    li = _eigenspace_guess(ke23, MO, n, kappa, verbose=False)

    NS = Matrix(li) * MO
    for i in range(NS.nrows()):
        if any(c == 2 for c in NS[i]):
            NS[i] = -NS[i]
    n_recovered = NS.nrows()
    # print(f"  Recovered vectors: {n_recovered}")

    if kappa != -1:
        Y = _select_constant_weight(NS.rows(), n, m, kappa)
        if Y is None:
            Y, _tsf = _switching_phase(NS, n, m, kappa)
    else:
        Y = NS.T

    if X is not None:
        n_found = _count_unique_matches(NS, X)
        print(f"  NFound={n_found} out of {n}")

    return Y

"""
BKZ-based Step 2 operations: binary and bounded vector recovery.

Provides functions for recovering the secret vectors from the
kernel matrix produced by Step 1, using LLL/BKZ lattice reduction.
"""

from sage.all import ZZ, Matrix, matrix, cputime


# ---------------------------------------------------------------------------
# Vector validation helpers
# ---------------------------------------------------------------------------

def is_all_pm_ones(v):
    """Check if all entries of v are in {-1, 0, 1}."""
    return all(vj in (-1, 0, 1) for vj in v)


def is_all_bounded(v, B):
    """Check if all entries satisfy -B <= v_j <= B."""
    return all(-B <= vj <= B for vj in v)


def normalize_binary(v):
    """Return v if entries are in {0,1}, -v if in {0,-1}, else None."""
    if all(vj in (0, 1) for vj in v):
        return v
    if all(vj in (0, -1) for vj in v):
        return -v
    return None


def normalize_bounded(v, B):
    """Return v if entries in [0,B], -v if in [-B,0], else None."""
    if all(0 <= vj <= B for vj in v):
        return v
    if all(-B <= vj <= 0 for vj in v):
        return -v
    return None


def count_unique_matches(candidates, X):
    X_rows = {tuple(ZZ(x) for x in row) for row in X.T}
    found = {
        tuple(ZZ(x) for x in row)
        for row in candidates
        if tuple(ZZ(x) for x in row) in X_rows
    }
    return len(found)


# ---------------------------------------------------------------------------
# Recovery routines
# ---------------------------------------------------------------------------

def recover_binary(M5):
    """Recover binary vectors from a BKZ-reduced basis.

    Takes the reduced matrix and extracts rows that are valid
    binary (0/1) vectors, then tries to find more by pairwise
    sums/differences.

    Parameters
    ----------
    M5 : Matrix
        BKZ-reduced matrix.

    Returns
    -------
    Matrix
        Recovered binary vectors as rows.
    """
    lv = [normalize_binary(vi) for vi in M5 if normalize_binary(vi) is not None]

    n = M5.nrows()
    for v in lv:  # iterate over the growing list (chain discovery)
        for i in range(n):
            nv = normalize_binary(M5[i] - v)
            if nv is not None and nv not in lv:
                lv.append(nv)
            nv = normalize_binary(M5[i] + v)
            if nv is not None and nv not in lv:
                lv.append(nv)
    return Matrix(lv)


def recover_bounded(M5, B):
    """Recover bounded vectors from a BKZ-reduced basis.

    Parameters
    ----------
    M5 : Matrix
        BKZ-reduced matrix.
    B : int
        Entry bound.

    Returns
    -------
    Matrix
        Recovered bounded vectors as rows.
    """
    lv = [normalize_bounded(vi, B) for vi in M5 if normalize_bounded(vi, B) is not None]

    n = M5.nrows()
    for v in lv:  # iterate over the growing list (chain discovery)
        for i in range(n):
            nv = normalize_bounded(M5[i] - v, B)
            if nv is not None and nv not in lv:
                lv.append(nv)
            nv = normalize_bounded(M5[i] + v, B)
            if nv is not None and nv not in lv:
                lv.append(nv)
    return Matrix(lv)


# ---------------------------------------------------------------------------
# Step 2: BKZ reduction
# ---------------------------------------------------------------------------

def step2_bkz(ke, B, n, X=None):
    """Step 2: Recover secret vectors via LLL/BKZ reduction.

    Progressively increases BKZ block size until all rows of the
    reduced basis are bounded (entries in [-B, B]).

    Parameters
    ----------
    ke : Matrix
        Kernel matrix from Step 1 (over ZZ).
    B : int
        Entry bound.
    n : int
        Dimension (number of secret vectors).
    X : Matrix, optional
        Secret matrix (for verification).

    Returns
    -------
    tuple
        (MB, beta) where:
        - MB : Matrix — recovered vectors
        - beta : int — BKZ block size used
    """
    beta = 2
    M5 = ke.LLL()
    M5 = M5[:n]

    while beta < n:
        if beta != 2:
            M5 = M5.BKZ(block_size=beta)

        # Check if all rows are bounded
        n_bounded = sum(1 for v in M5 if is_all_bounded(v, B))
        if n_bounded == n:
            break

        beta = 10 if beta == 2 else beta + 10

    # Recover vectors
    if B == 1:
        MB = recover_binary(M5)
    else:
        MB = recover_bounded(M5, B)

    if X is not None:
        n_found = count_unique_matches(MB, X)
        print(f"  NFound={n_found}", end="")

    return MB, beta

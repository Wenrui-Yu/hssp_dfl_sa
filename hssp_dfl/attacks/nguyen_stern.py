"""
Nguyen-Stern attack and variants for HSSP/HLCP.

Implements Step 2 of the Nguyen-Stern attack, which uses BKZ
reduction on the kernel matrix to recover the secret binary/bounded
vectors, followed by sign-switching to resolve ambiguities.

Variants
--------
- nguyen_stern          : NS with Step 1.
"""

from itertools import combinations
from math import comb

from sage.all import (
    ZZ, Matrix, matrix, vector, copy,
    Integers, cputime, ones_matrix, variance,
)

from hssp_dfl.utils import negate_binary
from hssp_dfl.attacks.bkz import (
    step2_bkz,
    recover_binary,
    recover_bounded,
    count_unique_matches,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_close(a, b, atol=0.01, rtol=1):
    """Check if a ≈ b within tolerance."""
    return abs(a - b) <= (atol + rtol * abs(b))


def _unbalanced_check(n, kappa, B):
    """Check if the Hamming weight is unbalanced.

    Parameters
    ----------
    n : int
        Dimension.
    kappa : int
        Hamming weight. Number of non-zero entries.
    B : int, optional
        Entry bound (HSSP B=1).
        if entries are in [0, B], then B/(B+1) is the "bias" of non-zero entries.

    Returns
    -------
    bool
        True if the weight is unbalanced.
    """
    bb = B / (B + 1)
    return abs(n * bb - kappa) / n > 0.2


def _unique_binary_rows(rows, m):
    """Return unique binary rows, including complements."""
    ones = vector([1] * m)
    out = []
    def add_binary(candidate):
        if all(x in (0, 1) for x in candidate) and candidate not in out:
            out.append(candidate)

    for row in rows:
        add_binary(row)
        add_binary(ones - row)

    changed = True
    while changed and len(out) <= 80:
        changed = False
        for left in list(out):
            for right in list(out):
                before = len(out)
                add_binary(left + right)
                add_binary(left - right)
                if len(out) != before:
                    changed = True
    return out


def _select_constant_weight(rows, n, m, kappa):
    """Select n source rows whose sample-wise sum is kappa."""
    candidates = _unique_binary_rows(rows, m)
    if len(candidates) < n:
        return None
    if len(candidates) > 80 or comb(len(candidates), n) > 200000:
        return None

    target = vector([kappa] * m)
    zero = vector([0] * m)
    for picked in combinations(candidates, n):
        if sum(picked, zero) == target:
            NS = matrix(ZZ, picked)
            if NS.rank() == n:
                return NS.T

    for picked in combinations(candidates, n - 1):
        missing = target - sum(picked, zero)
        if all(x in (0, 1) for x in missing) and missing not in picked:
            NS = matrix(ZZ, list(picked) + [missing])
            if NS.rank() == n:
                return NS.T
    return None


def _resolve_unbalanced(NSo, n, m, kappa):
    """Handle the unbalanced case: deduce missing vector.

    Parameters
    ----------
    NSo : Matrix
        Recovered binary vectors.
    n, m : int
        Dimensions.
    kappa : int
        Hamming weight.

    Returns
    -------
    Matrix
        Completed matrix Y (m x n, transposed secret).
    """
    li2 = NSo.rows()
    ones = vector([1] * m)

    selected = _select_constant_weight(li2, n, m, kappa)
    if selected is not None:
        return selected

    if len(li2) < n:
        if kappa > n / 2:
            missing = (n - kappa) * ones - sum(li2)
            NS = matrix(ZZ, [ones - x for x in li2 + [missing]])
        else:
            missing = kappa * ones - sum(li2)
            NS = matrix(ZZ, li2 + [missing])
    else:
        NS = matrix(ZZ, li2)

    Y = NS.T
    assert Y.rank() == n, "rank < n: extra binary vector to handle"
    return Y


def _resolve_balanced(NSo, n, m, kappa):
    """Handle the balanced case with sign-switching.

    Parameters
    ----------
    NSo : Matrix
        Recovered binary vectors.
    n, m : int
        Dimensions.
    kappa : int
        Hamming weight.

    Returns
    -------
    tuple
        (Y, n_found, t_extra).
    """
    ones = vector([1] * m)
    li = NSo.rows()

    selected = _select_constant_weight(li, n, m, kappa)
    if selected is not None:
        return selected

    # Add complements
    for NSi in NSo:
        if ones - NSi not in li:
            li.append(ones - NSi)

    # Remove duplicates (keep one of v, 1-v)
    li2 = []
    for r in li:
        if ones - r not in li2:
            li2.append(r)
    if len(li2) < n and ones not in li2:
        li2 = [ones] + li2
    assert len(li2) == n, f"Expected {n} vectors, got {len(li2)}"

    NS = matrix(ZZ, li2)
    has_ones = ones in li2
    print(f"  ones in NS: {has_ones}")

    # Switching phase
    e = vector([1] * n)
    Y = matrix(ZZ, NS[:n])
    Y = copy(Y)
    v0 = variance(e * Y).n()

    if has_ones:
        var_w = kappa * (n - kappa) / n**2
        cv0 = _is_close(v0, var_w)
    else:
        var_w = 0
        cv0 = True

    ts = cputime()
    i = 0
    v1 = v0
    while i < n**2 and not cv0:
        j = i % n
        if Y[j] == ones:
            i += 1
            continue
        Y[j] = negate_binary(Y[j])
        v1 = variance(e * Y).n()
        if _is_close(v1, var_w) and has_ones:
            break
        if v1 == 0 and not has_ones:
            break
        if v1 < v0:
            v0 = v1
        else:
            Y[j] = negate_binary(Y[j])
        i += 1

    print(f"  Switching rounds: {i}, time: {cputime(ts):.1f}s")

    # Final correction
    if has_ones:
        wm = sum(xi for xi in Y if xi != ones)
        k0 = max(wm)
        wm = k0 * ones - wm
        jone = (Y.rows()).index(ones)
        Y[jone] = wm
        v1 = variance(e * Y).n()

    assert v1 == 0, f"Switching failed: variance = {v1}"

    if has_ones and k0 == n - kappa:
        for j in range(n):
            Y[j] = negate_binary(Y[j])

    return Y.T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nguyen_stern(n, m, kappa, MO, B=1, X=None):
    """Nguyen-Stern attack

    Parameters
    ----------
    n, m, kappa : int
        Problem parameters.
    MO : Matrix
        Kernel matrix from Step 1.
    X : Matrix
        Secret matrix (for verification).
    B : int, optional
        Entry bound (default 1 = binary).

    Returns
    -------
    tuple
        (beta, tt_step2) — BKZ block size, Step 2 time.
    """

    unbalanced = _unbalanced_check(n, kappa, B)
    print(f"  Unbalanced: {unbalanced}")

    t2 = cputime()
    NSo, beta = step2_bkz(matrix(ZZ, MO), B, n, X)
    # print(f"  Recovered vectors: {NSo.nrows()}", end="")
    tt_step2 = cputime(t2)

    assert NSo.nrows() >= n - 1, "Step 2 failed: not enough vectors found"

    if kappa > 0 and unbalanced:
        Y = _resolve_unbalanced(NSo, n, m, kappa)
    elif kappa > 0 and not unbalanced:
        Y= _resolve_balanced(NSo, n, m, kappa)
    elif kappa == -1:
        Y = NSo.T

    if X is not None:
        n_found = count_unique_matches(Y.T, X)
        print(f"  NFound={n_found} out of {n}")
        return beta, tt_step2, n_found, Y

    return beta, tt_step2, Y

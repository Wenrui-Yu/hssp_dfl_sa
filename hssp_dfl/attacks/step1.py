"""
Step 1 — Orthogonal lattice construction and reduction.

This module implements the first phase of HSSP/HLCP attacks:
building the orthogonal lattice and reducing it to extract
a basis related to the secret vectors.

Two variants are provided:
- step1_original : The original Nguyen-Stern approach (full LLL on the lattice).
- step1_improved : An improved approach with block-wise Babai reduction.
"""

from sage.all import (
    ZZ, GF, Matrix, Integer, RealField, RDF,
    cputime, identity_matrix,
)

from hssp_dfl.lattice import (
    kernel_lll,
    ortho_lattice_mat,
    ortho_lattice_mod,
    mat_nbits,
    count_zero_vectors,
    round_matrix,
)


def step1_original(n, m, Q, H):
    """Step 1 using the original Nguyen-Stern approach.

    Constructs the orthogonal lattice, applies full LLL reduction,
    then computes the kernel to obtain candidate secret directions.

    Parameters
    ----------
    n : int
    m : int
    Q : int
        Modulus.
    H : vector or Matrix

    Returns
    -------
    tuple
        (MO, tt_step1) where:
        - MO : Matrix — candidate secret direction matrix (n x m)
        - tt_step1 : float — total Step 1 CPU time
    """

    print("Building orthogonal lattice...")
    M = ortho_lattice_mat(H, Q)

    print("Step 1 (Original)")
    t = cputime()

    M2 = M.LLL()
    MOrtho = M2[:m - n]
    ke = kernel_lll(MOrtho)

    tt_step1 = cputime(t)
    print(f"  Total Step 1: {tt_step1:.1f}s")

    return ke, tt_step1


def step1_improved(n, m, Q, H, X=None, use_bkz=False):
    """Step 1 using the improved block-wise Babai reduction.

    A faster approach that avoids full LLL on the entire lattice
    by using block-wise size reduction (Babai nearest-plane).
    This variant follows the modular orthogonal lattice construction
    and requires m >= 3*n and m % n == 0.

    Parameters
    ----------
    n : int
    m : int
    Q : int
        Modulus.
    H : Matrix
    X : Matrix
        Secret matrix (for verification).
    use_bkz : bool, optional
        If True, compute kernel over ZZ (for BKZ-based Step 2).
        If False, compute modulo 3 (for multivariate Step 2). Default False.
    Returns
    -------
    tuple
        (MO, tt_step1) where:
        - MO : Matrix — candidate secret direction matrix (n x m)
        - tt_step1 : float — total Step 1 CPU time
    """
    k = 4
    if n > 200:
        k = 5

    print("Building M...")
    M = ortho_lattice_mod(H, n, Q)

    print("Step 1 (Improved)")
    t = cputime()

    # Block LLL on first n/k rows
    M[:n // k, :n // k] = M[:n // k, :n // k].LLL()
    M2 = M[:2 * n, :2 * n].LLL()

    RF = RealField(mat_nbits(M))
    M4i = Matrix(RF, M[:n // k, :n // k]).inverse()
    M2i = Matrix(RDF, M2).inverse()

    # Size reduction pass 1 (Babai)
    while True:
        flag = True
        for i in range((Integer(m / n) - 2) * k):
            indf = 2 * n + n // k * (i + 1)
            if i == (Integer(m / n) - 2) * k - 1:
                indf = m
            mv = round_matrix(M[2 * n + n // k * i:indf, :n // k] * M4i)
            if mv == 0:
                continue
            flag = False
            M[2 * n + n // k * i:indf, :] -= mv * M[:n // k, :]
        if flag:
            break

    M[:2 * n, :2 * n] = M2

    # Size reduction pass 2
    while True:
        mv = round_matrix(M[2 * n:, :2 * n] * M2i)
        if mv == 0:
            break
        M[2 * n:, :] -= mv * M[:2 * n, :]

    # Verify orthogonality
    if X is not None:
        northo = count_zero_vectors(M[:n, :2 * n] * X[:2 * n])
        for i in range(2, Integer(m / n)):
            northo += count_zero_vectors(
                M[i * n:(i + 1) * n, :2 * n] * X[:2 * n] + X[i * n:(i + 1) * n]
            )
        print(f"  #ortho vecs={northo} out of {m - n}", end="")

    # Compute kernel of orthogonal vectors
    KK = ZZ if use_bkz else GF(3)
    MO = Matrix(KK, n, m)

    MO[:, :2 * n] = kernel_lll(M[:n, :2 * n])

    for i in range(2, Integer(m / n)):
        MO[:, i * n:(i + 1) * n] = -(M[i * n:(i + 1) * n, :2 * n] * MO[:, :2 * n].T).T

    tt_step1 = cputime(t)
    print(f"  Total Step 1: {tt_step1:.1f}s")

    del M, mv
    return MO, tt_step1

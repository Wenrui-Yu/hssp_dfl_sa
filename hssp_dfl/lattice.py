"""
Core lattice operations for HSSP/HLCP attacks.

Provides functions for building orthogonal lattices and
computing lattice kernels via LLL reduction.
"""

from sage.all import (
    ZZ, Matrix, identity_matrix, Integer, Integers,
    block_matrix, zero_matrix,
    inverse_mod, mod, RealField, RDF,
)
from time import time as wall_time
import sage.modules.vector_integer_dense


# ---------------------------------------------------------------------------
# Kernel computation
# ---------------------------------------------------------------------------

def kernel_lll(M):
    """Compute the right kernel of integer matrix M using LLL.

    Follows the approach from https://hal.archives-ouvertes.fr/hal-01921335/document.
    Requires m >= 2*n where M is n x m.

    Parameters
    ----------
    M : Matrix
        An integer matrix of shape (n, m).

    Returns
    -------
    Matrix
        Rows spanning the right kernel of M.
    """
    n = M.nrows()
    m = M.ncols()
    if m < 2 * n:
        return M.right_kernel().matrix()

    K = 2**(m // 2) * M.height()

    MB = Matrix(ZZ, m + n, m)
    MB[:n] = K * M
    MB[n:] = identity_matrix(m)

    MB2 = MB.T.LLL().T

    assert MB2[:n, :m - n] == 0
    Ke = MB2[n:, :m - n].T
    return Ke


# ---------------------------------------------------------------------------
# Orthogonal lattice construction
# ---------------------------------------------------------------------------

def ortho_lattice(h, Q):
    """Build the lattice of vectors orthogonal to h modulo Q (single vector).

    Parameters
    ----------
    h : vector
        Sample vector of length m.
    Q : int
        Modulus.

    Returns
    -------
    Matrix
        An m x m integer matrix whose rows are orthogonal to h mod Q.
    """
    m = h.length()
    M = Matrix(ZZ, m, m)
    for i in range(1, m):
        M[i, i] = 1
    M[1:m, 0] = -h[1:m] * inverse_mod(h[0], Q)
    M[0, 0] = Q
    for i in range(1, m):
        M[i, 0] = mod(M[i, 0], Q)
    return M


def ortho_lattice_mat(H, Q):
    """Build orthogonal lattice for a matrix or vector.

    If H is a vector, delegates to :func:`ortho_lattice`.
    If H is a matrix of shape (m, l), builds an m x m lattice
    with the first l x l block scaled by Q.

    Parameters
    ----------
    H : Matrix or vector
        Sample data (matrix or vector).
    Q : int
        Modulus.

    Returns
    -------
    Matrix
        Orthogonal lattice matrix.
    """
    if isinstance(H, sage.modules.vector_integer_dense.Vector_integer_dense):
        return ortho_lattice(H, Q)

    m, l = H.dimensions()
    M = identity_matrix(ZZ, m)
    M[:l, :l] = Q * M[:l, :l]

    H0i = Matrix(Integers(Q), H[:l, :l]).inverse()
    M[l:m, 0:l] = -H[l:m, :] * H0i
    return M


def ortho_lattice_mod(H, n, Q):
    """Build the modular orthogonal lattice (block structure for large m).

    Constructs a structured m x 3n lattice used in the improved Step 1.

    Parameters
    ----------
    H : vector or Matrix
        Sample data (vector or matrix).
    n : int
        Dimension parameter.
    Q : int
        Modulus.

    Returns
    -------
    Matrix
        An m x 3n integer matrix.
    """
    m = H.length()
    assert m >= 3 * n, "m must be >= 3*n for modular orthogonal lattice"
    assert m % n == 0, "m must be divisible by n"

    M = Matrix(ZZ, m, 3 * n)
    M[:2 * n, :2 * n] = identity_matrix(2 * n)
    for i in range(2, Integer(m / n)):
        M[i * n:(i + 1) * n, 2 * n:3 * n] = identity_matrix(n)

    M[1:, 0] = -H[1:] * inverse_mod(H[0], Q)
    M[0, 0] = Q
    for i in range(1, m):
        M[i, 0] = mod(M[i, 0], Q)
    return M


def noisy_lattice_mat(H, Q, beta):
    """Row basis of the augmented orthogonal lattice.

    O_{Q,beta}(H) = { (beta*y | z) : z + y^T H = 0 mod Q }.

    With row-basis convention, a row combination (y, k) gives
        (beta*y | -y^T H + k*Q).
    """
    H = Matrix(ZZ, H)
    M = H.nrows()
    r = H.ncols()

    top = block_matrix(ZZ, 1, 2, [beta * identity_matrix(ZZ, M), -H])
    bottom = block_matrix(ZZ, 1, 2, [zero_matrix(ZZ, r, M), Q * identity_matrix(ZZ, r)])

    return block_matrix(ZZ, 2, 1, [top, bottom])


# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

def mat_nbits(M):
    """Return the maximum bit-length of any entry in M.

    Parameters
    ----------
    M : Matrix
        An integer matrix.

    Returns
    -------
    int
        Maximum number of bits.
    """
    return max(M[i, j].nbits()
               for i in range(M.nrows())
               for j in range(M.ncols()))


def count_zero_vectors(M):
    """Count the number of zero row-vectors in M.

    Parameters
    ----------
    M : Matrix or iterable of vectors
        Input to check.

    Returns
    -------
    int
        Number of zero vectors.
    """
    return sum(1 for vi in M if vi == 0)


def round_matrix(M):
    """Round a real matrix to the nearest integer matrix.

    Parameters
    ----------
    M : Matrix
        A real-valued matrix.

    Returns
    -------
    Matrix
        Integer matrix with rounded entries.
    """
    M2 = Matrix(ZZ, M.nrows(), M.ncols())
    for i in range(M.nrows()):
        for j in range(M.ncols()):
            try:
                M2[i, j] = ZZ(M[i, j].round())
            except AttributeError:
                M2[i, j] = ZZ(round(M[i, j]))
    return M2

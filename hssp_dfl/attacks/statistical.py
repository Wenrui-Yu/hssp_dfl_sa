"""
Statistical/ICA Attack (Step 2) — Independent Component Analysis.

A heuristic statistical attack that treats the rows of the kernel matrix MO
as linear mixtures of independent source signals (the rows of X^T).
After LLL reduction, ICA is applied to separate sources and recover X.

Two variants are provided:
- 'roundA' : Inverts the recovered mixing matrix A to get X.
- 'roundX' : Directly rounds the ICA output to get X.

Note: ICA computation may be performed externally (e.g., via scikit-learn's
FastICA or a custom implementation). By default, this module loads precomputed
ICA results from ``results/ica_scratch/``. When ``use_sklearn=True`` is passed,
it uses scikit-learn's FastICA directly.

References
----------
- Coron, J.-S., Notarnicola, L., & Ciampi, G. (2020).
  "Cryptanalysis of the Hidden Subset Sum Problem using ICA."
"""

import os
import numpy as np
from sage.all import ZZ, Matrix, Integers, GF, cputime, mod

from hssp_dfl import paths
from hssp_dfl.lattice import kernel_lll, ortho_lattice_mat, mat_nbits

# Scratch directory for the intermediate matrices an external ICA run would
# consume.  It lives under results/ so nothing is written into the source tree.
DEFAULT_ICA_DIR = str(paths.RESULTS / "ica_scratch")


def _ica_dir(ica_dir):
    """Resolve (and create) the ICA scratch directory."""
    path = ica_dir or DEFAULT_ICA_DIR
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# NumPy <-> Sage conversion
# ---------------------------------------------------------------------------

def sage_to_numpy_matrix(MO):
    """Convert a Sage matrix to a NumPy array.

    Parameters
    ----------
    MO : Matrix
        A Sage integer matrix.

    Returns
    -------
    numpy.ndarray
        The matrix as a float64 NumPy array.
    """
    try:
        return np.array(MO, dtype=float)
    except (TypeError, ValueError):
        return np.array(MO.numpy(), dtype=float)


# ---------------------------------------------------------------------------
# ICA wrappers
# ---------------------------------------------------------------------------

def run_ica(MOn, use_sklearn=False, ica_dir=None):
    """Run ICA to recover source signals S.

    Parameters
    ----------
    MOn : numpy.ndarray
        The kernel matrix as a NumPy array (n x m).
    use_sklearn : bool, optional
        If True, use scikit-learn's FastICA. Otherwise load precomputed
        results from ``ica_dir``.
    ica_dir : str, optional
        Directory holding precomputed ICA results.  Defaults to
        ``results/ica_scratch/``.

    Returns
    -------
    numpy.ndarray
        Recovered source matrix S as a Sage integer matrix.
    """

    if use_sklearn:
        try:
            from sklearn.decomposition import FastICA
        except ImportError:
            raise ImportError(
                "scikit-learn is required for use_sklearn=True. "
                "Install it with: pip install scikit-learn"
            )
        n_components = MOn.shape[0]
        ica = FastICA(n_components=n_components, max_iter=5000, tol=1e-4, random_state=0)
        S_ = ica.fit_transform(MOn.T).T
    else:
        A_ = np.load(os.path.join(_ica_dir(ica_dir), "A_.npy"))
        S_ = np.dot(np.linalg.inv(A_), MOn)

    S2 = Matrix(ZZ, MOn.shape[0], MOn.shape[1],
                [int(round(x)) for row in S_ for x in row])
    return S2


def run_ica_mixing(MOn, use_sklearn=False, ica_dir=None):
    """Run ICA to recover the mixing matrix A.

    Parameters
    ----------
    MOn : numpy.ndarray
        The kernel matrix as a NumPy array (n x m).
    use_sklearn : bool, optional
        If True, use scikit-learn's FastICA.
    ica_dir : str, optional
        Directory for precomputed ICA results.

    Returns
    -------
    Matrix
        Recovered mixing matrix A as a Sage integer matrix (n x n).
    """
    if use_sklearn:
        try:
            from sklearn.decomposition import FastICA
        except ImportError:
            raise ImportError(
                "scikit-learn is required for use_sklearn=True. "
                "Install it with: pip install scikit-learn"
            )
        n_components = MOn.shape[0]
        ica = FastICA(n_components=n_components, max_iter=5000, tol=1e-4, random_state=0)
        ica.fit(MOn.T)
        A_ = ica.mixing_
    else:
        A_ = np.load(os.path.join(_ica_dir(ica_dir), "A_.npy"))

    A2 = Matrix(ZZ, MOn.shape[0], MOn.shape[0],
                [int(round(x)) for row in A_ for x in row])
    return A2


# ---------------------------------------------------------------------------
# Statistical attack — Phase 1 (LLL + save for ICA)
# ---------------------------------------------------------------------------

def statistical_phase1(MO, n, m, variant=None):
    """Statistical attack Phase 1: LLL-reduce kernel matrix and prepare for ICA.

    Parameters
    ----------
    MO : Matrix
        Kernel matrix from Step 1 (n x m).
    n : int
        Dimension.
    m : int
        Number of samples.
    variant : str or None, optional
        'roundA' or 'roundX'. Auto-selects based on n if None.

    Returns
    -------
    tuple
        (MOn, MO_reduced) where:
        - MOn : numpy.ndarray — LLL-reduced kernel as NumPy array.
        - MO_reduced : Matrix — LLL-reduced kernel as Sage matrix.
    """
    if variant is None:
        variant = "roundA" if n <= 200 else "roundX"

    print(f"  Step 2-ICA: {variant}")

    tlll = cputime()
    MO = MO.LLL()
    print(f"  LLL: {cputime(tlll):.1f}s  matNbits={mat_nbits(MO)}", end="")

    MOn = sage_to_numpy_matrix(MO)

    # Save for an external ICA run if one is wanted.
    ica_dir = _ica_dir(None)
    np.save(os.path.join(ica_dir, "MOn.npy"), MOn)

    return MOn, MO


# ---------------------------------------------------------------------------
# Statistical attack — Phase 2 (ICA recovery)
# ---------------------------------------------------------------------------

def statistical_phase2(MOn, MO, n, X=None, variant=None, use_sklearn=False, ica_dir=None):
    """Statistical attack Phase 2: Apply ICA and recover secret vectors.

    Parameters
    ----------
    MOn : numpy.ndarray
        LLL-reduced kernel as NumPy array.
    MO : Matrix
        LLL-reduced kernel as Sage matrix.
    n : int
        Dimension.
    X : Matrix
        Secret matrix (for verification).
    kappa : int
        Hamming weight.
    B : int, optional
        Entry bound (default 1).
    variant : str or None, optional
        'roundA' or 'roundX'. Auto-selects based on n if None.
    use_sklearn : bool, optional
        If True, use scikit-learn. Otherwise load from ica_dir.
    ica_dir : str, optional
        ICA results directory.

    Returns
    -------
    Matrix
        Recovered source matrix S as a Sage integer matrix.
    """
    if variant is None:
        variant = "roundA" if n <= 200 else "roundX"

    if variant == "roundA":
        A2 = run_ica_mixing(MOn, use_sklearn, ica_dir)
        try:
            S2 = A2.inverse() * MO
            print(f"  matNbits(A)={mat_nbits(A2)}", end="")
        except (ZeroDivisionError, ArithmeticError):
            print("  ICA mixing matrix non-invertible")
            return 0
    elif variant == "roundX":
        S2 = run_ica(MOn, use_sklearn, ica_dir)
        print(f"  matNbits(X)={mat_nbits(S2)}", end="")
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if X is not None:
        # Verify recovered rows
        Y = X.T
        nfound = 0
        for i in range(n):
            for j in range(n):
                if S2[i, :n] == Y[j, :n] and S2[i] == Y[j]:
                    nfound += 1
        print(f"  NFound={nfound} out of {n}")

    # NS = S2.T

    # # Save recovered X
    # np.save(os.path.join(ica_dir, "resX.npy"),
    #         sage_to_numpy_matrix(S2, n, m))

    # # Recover secret coefficients a
    # try:
    #     invNSn = Matrix(Integers(Q), NS[:n]).inverse()
    #     ra = invNSn * b[:n]
    #     nrafound = len([True for rai in ra if rai in a])
    # except (ZeroDivisionError, ArithmeticError):
    #     nrafound = 0
    # print(f"  Coefs of a found={nrafound} out of {n} ")

    return S2

# ---------------------------------------------------------------------------
# Convenience: full statistical attack (Phase 1 + Phase 2)
# ---------------------------------------------------------------------------

def statistical_attack(MO, n, m, X,variant=None,
                       use_sklearn=False, ica_dir=None):
    """Run the full statistical/ICA attack (Phase 1 + Phase 2).

    Parameters
    ----------
    MO : Matrix
        Kernel matrix from Step 1 (n x m).
    n : int
        Dimension.
    m : int
        Number of samples.
    X : Matrix
        Secret matrix (for verification).
    variant : str or None, optional
        'roundA' or 'roundX'.
    use_sklearn : bool, optional
        If True, use scikit-learn FastICA.
    ica_dir : str, optional
        ICA results directory.

    Returns
    -------
    tuple
        (t_ica, t_total, n_a_found).
    """
    print("\nStatistical/ICA Attack")
    MOn, MO_reduced = statistical_phase1(
        MO, n, m, variant
    )
    S2 = statistical_phase2(
        MOn, MO_reduced, n, X, variant, use_sklearn, ica_dir
    )
    return S2

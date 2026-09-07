"""Integration helpers for the DFL reconstruction experiments.

These helpers keep DFL-specific data handling out of the generic lattice
attack modules. Ground-truth model updates are deliberately not accepted by
``recover_candidates`` because they are unnecessary for the attack itself.
"""

import numpy as np


def topology_side_information(weight_matrix):
    """Return the three topology constraints used by Cases 1-3.

    Parameters
    ----------
    weight_matrix : array-like
        Corrupt-to-honest integer weight matrix with one candidate vector per
        column.

    Returns
    -------
    tuple
        ``(target_sum, binary_pattern, zero_counts)`` with the same values as
        the legacy ``HGenerator.get_known_info()`` method.
    """
    weights = np.asarray(weight_matrix)
    target_sum = [int(row.sum()) for row in weights]
    binary_pattern = np.where(weights != 0, 1, 0)
    zero_counts = np.sum(weights == 0, axis=0).tolist()
    return target_sum, binary_pattern, zero_counts


def recover_candidates(n, m, kernel, bound=1):
    """Recover row-oriented candidate vectors with Nguyen-Stern Step 2.

    The optimized Nguyen-Stern API returns recovered vectors as columns. DFL
    filtering expects one candidate per row, so this function is the single
    orientation boundary used by every paper experiment.
    """
    from hssp_dfl.attacks.nguyen_stern import nguyen_stern

    _beta, _elapsed, recovered_columns = nguyen_stern(
        n,
        m,
        -1,
        kernel,
        B=bound,
    )
    return recovered_columns.T


def sage_to_numpy(sage_matrix):
    """Convert a Sage matrix (or compatible array-like value) to floats."""
    try:
        return np.asarray(sage_matrix, dtype=float)
    except Exception:
        pass

    try:
        return np.asarray(sage_matrix.tolist(), dtype=float)
    except Exception:
        pass

    try:
        rows = int(sage_matrix.nrows())
        columns = int(sage_matrix.ncols())
        result = np.zeros((rows, columns), dtype=float)
        for i in range(rows):
            for j in range(columns):
                result[i, j] = float(sage_matrix[i, j])
        return result
    except Exception as exc:
        raise RuntimeError(f"Cannot convert Sage matrix to NumPy: {exc}") from exc


__all__ = [
    "recover_candidates",
    "sage_to_numpy",
    "topology_side_information",
]

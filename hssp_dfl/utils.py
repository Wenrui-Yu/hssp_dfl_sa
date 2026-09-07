"""
Miscellaneous utility functions.
"""

from sage.all import ZZ, mod, random_prime


def gen_random_subset(n, size):
    """Generate a random subset of {0, ..., n-1} with exactly `size` elements.

    Parameters
    ----------
    n : int
        Universe size.
    size : int
        Desired subset size.

    Returns
    -------
    list[int]
        A list of `size` distinct random integers in [0, n).
    """
    selected = []
    while len(selected) < size:
        p = ZZ.random_element(n)
        if p not in selected:
            selected.append(p)
    return selected


def gen_pseudoprime(eta, eta_min=211):
    """Generate a pseudo-prime of approximately `eta` bits.

    For eta <= 2*eta_min, returns a single random prime in [2^(eta-1), 2^eta).
    Otherwise, recursively splits into smaller primes.

    Parameters
    ----------
    eta : int
        Target bit-length.
    eta_min : int, optional
        Minimum prime bit-length (default 211).

    Returns
    -------
    int
        A pseudo-prime of approximately `eta` bits.
    """
    if eta <= 2 * eta_min:
        return random_prime(2**eta, False, 2**(eta - 1))
    else:
        return random_prime(2**eta_min, False, 2**(eta_min - 1)) * gen_pseudoprime(eta - eta_min)


def negate_binary(v):
    """Negate a binary vector: return ones - v.

    Parameters
    ----------
    v : vector
        A binary (0/1) vector.

    Returns
    -------
    vector
        The complemented vector.
    """
    from sage.all import vector
    m = len(v)
    ones = vector([1] * m)
    return ones - v


# Alias for backward compatibility with original 'ned' function
ned = negate_binary

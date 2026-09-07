"""
Vector filtering and MITM combination search for DFL attacks.

After recovering candidate weight vectors from the HSSP/HLCP attack,
these routines filter and reassemble them into valid weight matrices
using side information.

Three filtering cases are supported:

1. **Exact pattern** (``X_binary`` provided): the attacker knows *which*
   positions are zero in each secret vector.  Pre-filter by matching
   the exact zero-position set; the MITM target is the known column
   sums (``known_info``).

2. **Zero-count only** (``target_zero_counts`` provided, ``X_binary``
   is ``None``): the attacker knows only *how many* zeros each secret
   vector has (e.g. number of in-neighbours).  Pre-filter by matching
   the zero-count; the MITM target is the known column sums.

3. **Scale bound** (``known_info`` is ``None``, ``scale`` provided):
   the attacker has no column-sum information but knows that each
   element-wise column sum of the true matrix is <= ``scale``.
   All candidate vectors are kept; the MITM checks
   ``sum <= scale`` element-wise.
"""

import numpy as np
import collections


# ---------------------------------------------------------------------------
# Pre-filter helpers
# ---------------------------------------------------------------------------

def _pre_filter_by_pattern(original_vectors, X_binary):
    """Pre-filter vectors by exact zero-position pattern.

    Parameters
    ----------
    original_vectors : ndarray, shape (N, dim)
    X_binary : ndarray
        Binary mask of X.  Will be transposed so that rows correspond
        to individual secret vectors.

    Returns
    -------
    vectors, index_map, vec_meta, target_criterion, ordered_patterns
    """
    patterns_matrix = X_binary.T  # rows = secret vectors

    ordered_patterns = []
    pattern_set = set()
    for row in patterns_matrix:
        p = frozenset(np.where(row == 0)[0])
        ordered_patterns.append(p)
        pattern_set.add(p)

    filtered, index_map, vec_meta = [], [], []
    for i, vec in enumerate(original_vectors):
        p = frozenset(np.where(vec == 0)[0])
        if p in pattern_set:
            filtered.append(vec)
            index_map.append(i)
            vec_meta.append(p)  # store the pattern itself

    target_criterion = collections.Counter(ordered_patterns)
    print(f"Pre-filter (exact pattern): {original_vectors.shape[0]} → "
          f"{len(filtered)} vectors")
    return (np.array(filtered) if filtered
            else np.empty((0, original_vectors.shape[1]))),\
           index_map, vec_meta, target_criterion, ordered_patterns


def _pre_filter_by_counts(original_vectors, target_zero_counts):
    """Pre-filter vectors whose zero-count matches any of the targets.

    Parameters
    ----------
    original_vectors : ndarray, shape (N, dim)
    target_zero_counts : list[int]
        One entry per secret vector, giving the expected number of zeros.

    Returns
    -------
    vectors, index_map, vec_meta, target_criterion
    """
    target_freq = collections.Counter(target_zero_counts)
    allowed = set(target_freq.keys())

    filtered, index_map, vec_meta = [], [], []
    for i, vec in enumerate(original_vectors):
        zc = int(np.sum(vec == 0))
        if zc in allowed:
            filtered.append(vec)
            index_map.append(i)
            vec_meta.append(zc)  # store the zero-count

    print(f"Pre-filter (zero-count): {original_vectors.shape[0]} → "
          f"{len(filtered)} vectors")
    return (np.array(filtered) if filtered
            else np.empty((0, original_vectors.shape[1]))),\
           index_map, vec_meta, target_freq


# ---------------------------------------------------------------------------
# Sorting helpers
# ---------------------------------------------------------------------------

def _sort_by_pattern(combo_vectors, ordered_patterns):
    """Sort combo rows to match the order given by *ordered_patterns*.

    Handles duplicate patterns via pop(0).
    Returns ndarray or None on failure.
    """
    buckets = collections.defaultdict(list)
    for v in combo_vectors:
        buckets[frozenset(np.where(v == 0)[0])].append(v)

    out = []
    tmp = {k: list(vs) for k, vs in buckets.items()}
    try:
        for p in ordered_patterns:
            out.append(tmp[p].pop(0))
    except (KeyError, IndexError):
        return None
    return np.array(out)


def _sort_by_zero_counts(combo_vectors, target_zero_counts):
    """Sort combo rows to match the order given by *target_zero_counts*.

    Handles duplicate counts via pop(0).
    Returns ndarray or None on failure.
    """
    buckets = collections.defaultdict(list)
    for v in combo_vectors:
        buckets[int(np.sum(v == 0))].append(v)

    out = []
    tmp = {k: list(vs) for k, vs in buckets.items()}
    try:
        for c in target_zero_counts:
            out.append(tmp[c].pop(0))
    except (KeyError, IndexError):
        return None
    return np.array(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_info(original_vectors, known_info, number,
                X_binary=None, target_zero_counts=None):
    """Unified MITM combination search with three filtering modes.

    Parameters
    ----------
    original_vectors : ndarray, shape (N, dim)
        Candidate vectors recovered by the lattice attack.
    known_info : array-like or None
        Expected element-wise column sum of the secret matrix.
        If ``None``, the *scale*-bound mode is used instead.
    number : int
        Number of vectors to select (= number of honest nodes).
    X_binary : ndarray or None
        If provided, use **Case 1** (exact zero-pattern match).
    target_zero_counts : list[int] or None
        If provided (and ``X_binary`` is None), use **Case 2**
        (zero-count match).  One entry per secret vector.
    scale : int or None

    Returns
    -------
    list[ndarray]
        Each entry is an ``(number, dim)`` matrix representing
        a valid reconstruction.
    """
    use_exact_pattern = X_binary is not None

    # ------------------------------------------------------------------ #
    # Phase 1 — Pre-filter
    # ------------------------------------------------------------------ #
    if use_exact_pattern:
        vectors, index_map, vec_meta, target_criterion, ordered_patterns = \
            _pre_filter_by_pattern(original_vectors, X_binary)
    elif target_zero_counts is not None:
        vectors, index_map, vec_meta, target_criterion = \
            _pre_filter_by_counts(original_vectors, target_zero_counts)
        ordered_patterns = None
    else:
        # No structural filter — keep everything
        vectors = original_vectors
        index_map = list(range(len(original_vectors)))
        vec_meta = [None] * len(original_vectors)
        target_criterion = None
        ordered_patterns = None

    n_vecs = vectors.shape[0]
    dim = vectors.shape[1]
    if n_vecs < number:
        print(f"Too few candidate vectors ({n_vecs} < {number}).")
        return []

    # ------------------------------------------------------------------ #
    # Phase 2 — MITM enumeration
    # ------------------------------------------------------------------ #
    k1 = number // 2
    k2 = number - k1

    def _enumerate(k):
        """Build {sum_tuple: [(indices, meta), …]} for all k-subsets."""
        result = collections.defaultdict(list)

        def _bt(start, cur_sum, cur_idx, cur_meta):
            if len(cur_idx) == k:
                result[tuple(cur_sum)].append((tuple(cur_idx),
                                               tuple(cur_meta)))
                return
            for i in range(start, n_vecs):
                if n_vecs - i < k - len(cur_idx):
                    return  # prune
                _bt(i + 1, cur_sum + vectors[i],
                    cur_idx + [i], cur_meta + [vec_meta[i]])

        _bt(0, np.zeros(dim, dtype=vectors.dtype), [], [])
        return result

    print(f"MITM enumeration (k1={k1}, k2={k2}) ...")
    sums_k1 = _enumerate(k1)
    sums_k2 = _enumerate(k2)
    print(f"  k1-subsets: {sum(len(v) for v in sums_k1.values())}, "
          f"k2-subsets: {sum(len(v) for v in sums_k2.values())}")

    # ------------------------------------------------------------------ #
    # Phase 3 — Match & validate
    # ------------------------------------------------------------------ #
    found_hashes = set()
    results = []

    def _try_accept(idx1, meta1, idx2, meta2):
        """Validate the zero-structure criterion, sort, dedup, store."""
        if not set(idx1).isdisjoint(set(idx2)):
            return

        combined_meta = meta1 + meta2
        if target_criterion is not None:
            if collections.Counter(combined_meta) != target_criterion:
                return

        combo_vecs = [vectors[i] for i in idx1] + [vectors[i] for i in idx2]

        # Sort rows into canonical order
        if use_exact_pattern:
            mat = _sort_by_pattern(combo_vecs, ordered_patterns)
        elif target_zero_counts is not None:
            mat = _sort_by_zero_counts(combo_vecs, target_zero_counts)
        else:
            mat = np.array(combo_vecs)
        if mat is None:
            return

        h = tuple(mat.flatten())
        if h in found_hashes:
            return
        found_hashes.add(h)
        results.append(mat)

    if known_info is not None:
        # exact column-sum constraint
        target_sum = np.array(known_info)
        for s1_tuple, data1 in sums_k1.items():
            needed = tuple(target_sum - np.array(s1_tuple))
            if any(x < 0 for x in needed):
                continue
            if needed in sums_k2:
                for idx1, meta1 in data1:
                    for idx2, meta2 in sums_k2[needed]:
                        _try_accept(idx1, meta1, idx2, meta2)

    else:
        print("Error: both known_info and scale are None — nothing to match.")
        return []

    print(f"Search complete: {len(results)} valid reconstruction(s) found.")
    # for i, mat in enumerate(results):
    #     print(f"  Matrix {i + 1}:\n{mat}\n")
    return results

def filter_info_directed(original_vectors, number,
                X_binary=None, target_zero_counts=None, scale=None):
    """Unified MITM combination search with three filtering modes.

    Parameters
    ----------
    original_vectors : ndarray, shape (N, dim)
        Candidate vectors recovered by the lattice attack.
    known_info : array-like or None
        Expected element-wise column sum of the secret matrix.
        If ``None``, the *scale*-bound mode is used instead.
    number : int
        Number of vectors to select (= number of honest nodes).
    X_binary : ndarray or None
        If provided, use **Case 1** (exact zero-pattern match).
    target_zero_counts : list[int] or None
        If provided (and ``X_binary`` is None), use **Case 2**
        (zero-count match).  One entry per secret vector.
    scale : int or None
        If ``known_info`` is None, use **Case 3**: accept any
        combination whose element-wise sum is <= *scale*.

    Returns
    -------
    list[ndarray]
        Each entry is an ``(number, dim)`` matrix representing
        a valid reconstruction.
    """
    use_exact_pattern = X_binary is not None

    # ------------------------------------------------------------------ #
    # Phase 1 — Pre-filter
    # ------------------------------------------------------------------ #
    if use_exact_pattern:
        vectors, index_map, vec_meta, target_criterion, ordered_patterns = \
            _pre_filter_by_pattern(original_vectors, X_binary)
    elif target_zero_counts is not None:
        vectors, index_map, vec_meta, target_criterion = \
            _pre_filter_by_counts(original_vectors, target_zero_counts)
        ordered_patterns = None
    else:
        # No structural filter — keep everything
        vectors = original_vectors
        index_map = list(range(len(original_vectors)))
        vec_meta = [None] * len(original_vectors)
        target_criterion = None
        ordered_patterns = None

    n_vecs = vectors.shape[0]
    dim = vectors.shape[1]
    if n_vecs < number:
        print(f"Too few candidate vectors ({n_vecs} < {number}).")
        return []

    # ------------------------------------------------------------------ #
    # Phase 2 — MITM enumeration
    # ------------------------------------------------------------------ #
    k1 = number // 2
    k2 = number - k1

    def _enumerate(k):
        """Build {sum_tuple: [(indices, meta), …]} for all k-subsets."""
        result = collections.defaultdict(list)

        def _bt(start, cur_sum, cur_idx, cur_meta):
            if len(cur_idx) == k:
                result[tuple(cur_sum)].append((tuple(cur_idx),
                                               tuple(cur_meta)))
                return
            for i in range(start, n_vecs):
                if n_vecs - i < k - len(cur_idx):
                    return  # prune
                _bt(i + 1, cur_sum + vectors[i],
                    cur_idx + [i], cur_meta + [vec_meta[i]])

        _bt(0, np.zeros(dim, dtype=vectors.dtype), [], [])
        return result

    print(f"MITM enumeration (k1={k1}, k2={k2}) ...")
    sums_k1 = _enumerate(k1)
    sums_k2 = _enumerate(k2)
    print(f"  k1-subsets: {sum(len(v) for v in sums_k1.values())}, "
          f"k2-subsets: {sum(len(v) for v in sums_k2.values())}")

    # ------------------------------------------------------------------ #
    # Phase 3 — Match & validate
    # ------------------------------------------------------------------ #
    found_hashes = set()
    results = []

    def _try_accept(idx1, meta1, idx2, meta2):
        """Validate the zero-structure criterion, sort, dedup, store."""
        if not set(idx1).isdisjoint(set(idx2)):
            return

        combined_meta = meta1 + meta2
        if target_criterion is not None:
            if collections.Counter(combined_meta) != target_criterion:
                return

        combo_vecs = [vectors[i] for i in idx1] + [vectors[i] for i in idx2]

        # Sort rows into canonical order
        if use_exact_pattern:
            mat = _sort_by_pattern(combo_vecs, ordered_patterns)
        elif target_zero_counts is not None:
            mat = _sort_by_zero_counts(combo_vecs, target_zero_counts)
        else:
            mat = np.array(combo_vecs)
        if mat is None:
            return

        h = tuple(mat.flatten())
        if h in found_hashes:
            return
        found_hashes.add(h)
        results.append(mat)

    if scale is not None:
        # scale-bound constraint (sum <= scale element-wise)
        for s1_tuple, data1 in sums_k1.items():
            s1 = np.array(s1_tuple)
            for s2_tuple, data2 in sums_k2.items():
                if np.all(np.array(s2_tuple) <= (scale - s1)):
                    for idx1, meta1 in data1:
                        for idx2, meta2 in data2:
                            _try_accept(idx1, meta1, idx2, meta2)
    else:
        print("scale are None")
        return []

    print(f"Search complete: {len(results)} valid reconstruction(s) found.")
    # for i, mat in enumerate(results):
    #     print(f"  Matrix {i + 1}:\n{mat}\n")
    return results

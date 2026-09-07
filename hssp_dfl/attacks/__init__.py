"""Attack stages, loaded lazily by public symbol.

Step 1 builds the orthogonal lattice and reduces it (``step1``, or
``noisy_step1`` for the noise-tolerant variant of Section 4.4); Step 2 recovers
the short candidate vectors (``nguyen_stern``, which reduces with ``bkz``).

``multivariate`` and ``statistical`` are the two alternative Step-2 approaches
discussed in Appendix E -- the multivariate quadratic reformulation and the
ICA-based distributional attack of Coron and Gini.  No result in the paper is
produced with them; they are kept as reference implementations alongside the
Nguyen-Stern Step 2 that the experiments do use.
"""

from importlib import import_module


_EXPORTS = {
    # Step 1: orthogonal lattice + LLL
    "step1_original": ("hssp_dfl.attacks.step1", "step1_original"),
    "step1_improved": ("hssp_dfl.attacks.step1", "step1_improved"),
    # Step 1, noise-tolerant (Section 4.4)
    "noisy_step1_original": (
        "hssp_dfl.attacks.noisy_step1",
        "noisy_step1_original",
    ),
    "estimate_beta_noisy": (
        "hssp_dfl.attacks.noisy_step1",
        "estimate_beta_noisy",
    ),
    # Step 2: Nguyen-Stern candidate recovery
    "nguyen_stern": ("hssp_dfl.attacks.nguyen_stern", "nguyen_stern"),
    "step2_bkz": ("hssp_dfl.attacks.bkz", "step2_bkz"),
    # Step 2: alternative approaches (Appendix E, unused by the experiments)
    "multivariate_eigen": (
        "hssp_dfl.attacks.multivariate",
        "multivariate_eigen",
    ),
    "multivariate_bit_guessing": (
        "hssp_dfl.attacks.multivariate",
        "multivariate_bit_guessing",
    ),
    "statistical_attack": (
        "hssp_dfl.attacks.statistical",
        "statistical_attack",
    ),
    "statistical_phase1": (
        "hssp_dfl.attacks.statistical",
        "statistical_phase1",
    ),
    "statistical_phase2": (
        "hssp_dfl.attacks.statistical",
        "statistical_phase2",
    ),
    "run_ica": ("hssp_dfl.attacks.statistical", "run_ica"),
    "run_ica_mixing": (
        "hssp_dfl.attacks.statistical",
        "run_ica_mixing",
    ),
    "sage_to_numpy_matrix": (
        "hssp_dfl.attacks.statistical",
        "sage_to_numpy_matrix",
    ),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)

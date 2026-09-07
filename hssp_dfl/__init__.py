"""Optimized HSSP/HLCP cryptanalysis toolkit.

The package keeps its public API lazy so graph and topology helpers can be
used without importing SageMath. Lattice and attack modules are loaded only
when their symbols are first requested.

Examples
--------
>>> from hssp_dfl.attacks import step1_original
>>> from hssp_dfl.dfl import recover_candidates
>>> from hssp_dfl.graph import generate_connected_graph
"""

from importlib import import_module


_EXPORTS = {
    # Lattice operations
    "kernel_lll": ("hssp_dfl.lattice", "kernel_lll"),
    "ortho_lattice": ("hssp_dfl.lattice", "ortho_lattice"),
    "ortho_lattice_mod": ("hssp_dfl.lattice", "ortho_lattice_mod"),
    # Step 1
    "step1_original": ("hssp_dfl.attacks.step1", "step1_original"),
    "step1_improved": ("hssp_dfl.attacks.step1", "step1_improved"),
    # Nguyen-Stern
    "nguyen_stern": ("hssp_dfl.attacks.nguyen_stern", "nguyen_stern"),
    # BKZ
    "step2_bkz": ("hssp_dfl.attacks.bkz", "step2_bkz"),
    # Alternative Step 2 approaches (Appendix E, unused by the experiments)
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
    # Filtering and DFL integration
    "filter_info": ("hssp_dfl.filter", "filter_info"),
    "filter_info_directed": (
        "hssp_dfl.filter",
        "filter_info_directed",
    ),
    "recover_candidates": (
        "hssp_dfl.dfl",
        "recover_candidates",
    ),
    "sage_to_numpy": ("hssp_dfl.dfl", "sage_to_numpy"),
    "topology_side_information": (
        "hssp_dfl.dfl",
        "topology_side_information",
    ),
}


def __getattr__(name):
    """Load public symbols on demand."""
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

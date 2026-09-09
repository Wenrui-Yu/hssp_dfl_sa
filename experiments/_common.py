"""Shared helpers for the paper's experiment entry points.

Everything in this module is deliberately dependency-light (no SageMath, no
PyTorch) so that the purely structural topology experiments can run in a plain
Python environment.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

# Repository root, so every script can be launched from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hssp_dfl import paths

DEFAULT_SEED = 20260727

# The nine corruption ratios eta used throughout the paper's topology figures.
DEFAULT_CORRUPT_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def seed_everything(seed: int) -> None:
    """Seed the two RNGs that drive graph generation and node partitioning."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))


def resolve_output_dir(override, *default_parts: str) -> Path:
    """Return (and create) an explicit directory or a path under HSSP_RESULTS."""
    path = Path(override) if override else paths.RESULTS.joinpath(*default_parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relativise(value):
    """Rewrite any path under the repository root as a relative path.

    Run configurations are meant to be shared, so they must not carry the
    absolute location of somebody's checkout.
    """
    if isinstance(value, dict):
        return {k: _relativise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_relativise(v) for v in value]
    text = str(value) if isinstance(value, Path) else value
    if isinstance(text, str) and text.startswith(str(ROOT)):
        return str(Path(text).relative_to(ROOT))
    return value


def write_run_config(path, config: dict) -> Path:
    """Persist the resolved CLI configuration next to the outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_relativise(config), indent=2, sort_keys=True, default=str)
    )
    return path


def split_by_ratio(num_nodes: int, corrupt_ratio: float) -> tuple[int, int]:
    """Map a corruption ratio to ``(n_corrupt, n_honest)``.

    At least one corrupted node and at least two honest nodes are kept so that
    the ``|V_c| >= |V_h| >= 2`` core condition can be evaluated at every eta.
    """
    n_corrupt = int(round(num_nodes * corrupt_ratio))
    n_corrupt = min(max(n_corrupt, 1), num_nodes - 2)
    return n_corrupt, num_nodes - n_corrupt


def use_agg_backend() -> None:
    """Select a headless Matplotlib backend before pyplot is imported."""
    import matplotlib

    matplotlib.use("Agg")


def savefig(fig, path) -> Path:
    """Write a figure to ``path`` (creating parents) and report the location."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  wrote {path}")
    return path

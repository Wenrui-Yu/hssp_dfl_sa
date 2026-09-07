"""Single place where the artifact resolves large input assets and outputs.

Nothing in the repository hard-codes a path to a checkpoint or a dataset.
Every location below can be overridden with an environment variable, so the
same commands work whether the assets live inside ``assets/`` (the default,
populated by ``scripts/setup_assets.sh``) or on a scratch disk.

    HSSP_ASSETS       root for all inputs            default <repo>/assets
    HSSP_MODEL_DIR    DFL checkpoints (*.pkl)        default <assets>/models
    HSSP_MODEL_DP_DIR DP checkpoints (*.pkl)         default <assets>/models_dp
    HSSP_DATASET_DIR  per-node dataset pickles       default <assets>/datasets
    HSSP_DATA_DIR     raw datasets (CIFAR-10, ...)   default <assets>/data
    HSSP_NETWORK_MAT  fixed 10-node topology         default <assets>/network.mat
    HSSP_RESULTS      output root                    default <repo>/results
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


ASSETS = _env_path("HSSP_ASSETS", ROOT / "assets")
MODEL_DIR = _env_path("HSSP_MODEL_DIR", ASSETS / "models")
MODEL_DP_DIR = _env_path("HSSP_MODEL_DP_DIR", ASSETS / "models_dp")
DATASET_DIR = _env_path("HSSP_DATASET_DIR", ASSETS / "datasets")
DATA_DIR = _env_path("HSSP_DATA_DIR", ASSETS / "data")
NETWORK_MAT = _env_path("HSSP_NETWORK_MAT", ASSETS / "network.mat")
RESULTS = _env_path("HSSP_RESULTS", ROOT / "results")
REFERENCE = ROOT / "reference"


def results_dir(*parts: str) -> Path:
    """Return (and create) ``<results>/<parts...>``."""
    path = RESULTS.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require(path, what: str) -> Path:
    """Fail early, with an actionable message, when an asset is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {what}: {path}\n"
            f"Populate the asset tree first:  bash scripts/setup_assets.sh --help"
        )
    return path


def describe() -> str:
    lines = ["resolved asset paths:"]
    for name in ("ASSETS", "MODEL_DIR", "MODEL_DP_DIR", "DATASET_DIR",
                 "DATA_DIR", "NETWORK_MAT", "RESULTS"):
        value = globals()[name]
        mark = "ok " if Path(value).exists() else "-- "
        lines.append(f"  {mark}{name:14s} {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())

"""Thin adapter around the upstream ``breaching`` attack framework.

The adapter deliberately contains no reimplementation of a gradient-inversion
attack.  It translates this repository's model/gradient objects into the public
Breaching API used by its ``minimal_example.py``.
"""

from __future__ import annotations

import copy
import importlib.metadata
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


BREACHING_REPOSITORY = "https://github.com/JonasGeiping/breaching"
BREACHING_COMMIT = "faaab88e618dfa94d77b062ad50ac4a27afb1534"
SUPPORTED_ATTACKS = (
    "deepleakage",
    "invertinggradients",
    "seethroughgradients",
    "modern",
    "wei",
)


@dataclass(frozen=True)
class DataSpec:
    """Dataset metadata required by Breaching's public attack API."""

    shape: tuple[int, ...]
    classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    modality: str = "vision"
    task: str = "classification"

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("shape must contain at least one dimension")
        if self.classes < 2:
            raise ValueError("classes must be at least 2")
        if len(self.mean) != self.shape[0] or len(self.std) != self.shape[0]:
            raise ValueError("mean/std must contain one value per input channel")
        if any(value <= 0 for value in self.std):
            raise ValueError("all standard deviations must be positive")


@dataclass
class AttackResult:
    """Reconstruction and provenance returned by an upstream attack."""

    data: torch.Tensor
    labels: torch.Tensor | None
    stats: Mapping[str, Any]
    attack: str
    elapsed_seconds: float
    breaching_version: str


def _import_breaching():
    try:
        import breaching
        from omegaconf import DictConfig, OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "The optional GIA dependency is missing. Install the pinned upstream "
            "implementation with `pip install -r requirements-gia.txt`."
        ) from exc
    return breaching, DictConfig, OmegaConf


def _apply_config_overrides(cfg_attack, overrides: Mapping[str, Any] | None, OmegaConf) -> None:
    """Apply explicitly recorded dotted-path overrides to an upstream config."""

    if not overrides:
        return
    for path, value in overrides.items():
        if not isinstance(path, str) or not path or path.startswith(".") or path.endswith("."):
            raise ValueError(f"invalid Breaching config override path: {path!r}")
        node = cfg_attack
        for component in path.split("."):
            if not hasattr(node, "keys") or component not in node:
                raise ValueError(f"unknown Breaching config override path: {path}")
            node = node[component]
        OmegaConf.update(cfg_attack, path, value, merge=False)


def _dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.float32: "float",
        torch.float64: "double",
        torch.bfloat16: "bfloat16",
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported reconstruction dtype: {dtype}") from exc


def _clone_tensors(tensors: Sequence[torch.Tensor], device: torch.device, dtype: torch.dtype):
    return [tensor.detach().clone().to(device=device, dtype=dtype) for tensor in tensors]


def gradients_from_batch(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: torch.nn.Module | None = None,
) -> list[torch.Tensor]:
    """Compute a mean FedSGD gradient for one local image batch."""

    if inputs.shape[0] != labels.shape[0]:
        raise ValueError("inputs and labels must have the same batch dimension")
    if inputs.shape[0] == 0:
        raise ValueError("cannot compute gradients for an empty batch")
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(inputs), labels.long())
    return [
        gradient.detach().clone()
        for gradient in torch.autograd.grad(loss, tuple(model.parameters()))
    ]


def gradients_from_model_delta(
    model: torch.nn.Module,
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    learning_rate: float,
) -> list[torch.Tensor]:
    """Recover one-step SGD gradients from pre/post local model snapshots."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    gradients = []
    for name, parameter in model.named_parameters():
        if name not in before or name not in after:
            raise KeyError(f"Missing parameter {name!r} in model snapshots")
        before_tensor = torch.as_tensor(before[name], device=parameter.device, dtype=parameter.dtype)
        after_tensor = torch.as_tensor(after[name], device=parameter.device, dtype=parameter.dtype)
        if before_tensor.shape != parameter.shape or after_tensor.shape != parameter.shape:
            raise ValueError(f"Snapshot shape mismatch for parameter {name!r}")
        gradients.append(((before_tensor - after_tensor) / learning_rate).detach().clone())
    return gradients


def _make_metadata(data_spec: DataSpec, DictConfig):
    return DictConfig(
        {
            "modality": data_spec.modality,
            "task": data_spec.task,
            "shape": list(data_spec.shape),
            "classes": data_spec.classes,
            "normalize": True,
            "mean": list(data_spec.mean),
            "std": list(data_spec.std),
        }
    )


def run_breaching_attack(
    model: torch.nn.Module,
    gradients: Sequence[torch.Tensor],
    num_data_points: int,
    data_spec: DataSpec,
    *,
    attack: str,
    labels: torch.Tensor | None,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    max_iterations: int | None = None,
    restarts: int | None = None,
    callback: int | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    seed: int = 0,
    dryrun: bool = False,
) -> AttackResult:
    """Run an unmodified attack implementation through Breaching's public API."""

    if attack not in SUPPORTED_ATTACKS:
        raise ValueError(
            f"Unsupported attack {attack!r}; choose one of {', '.join(SUPPORTED_ATTACKS)}"
        )
    if num_data_points <= 0:
        raise ValueError("num_data_points must be positive")

    breaching, DictConfig, OmegaConf = _import_breaching()
    device = torch.device(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    attack_model = copy.deepcopy(model).to(device=device, dtype=dtype)
    attack_model.eval()
    parameters = tuple(attack_model.parameters())
    if len(gradients) != len(parameters):
        raise ValueError(
            f"Expected {len(parameters)} gradient tensors, received {len(gradients)}"
        )
    for index, (gradient, parameter) in enumerate(zip(gradients, parameters)):
        if gradient.shape != parameter.shape:
            raise ValueError(
                f"Gradient {index} has shape {tuple(gradient.shape)}; "
                f"expected {tuple(parameter.shape)}"
            )

    cfg_attack = breaching.get_attack_config(attack=attack)
    _apply_config_overrides(cfg_attack, config_overrides, OmegaConf)
    cfg_attack.impl.dtype = _dtype_name(dtype)
    if attack == "deepleakage" and labels is not None:
        # The upstream DLG preset jointly optimizes labels and explicitly rejects
        # supplied labels.  This repository's legacy DLG comparison assumes
        # known labels, so retain the upstream DLG objective/optimizer while
        # selecting its data-only optimization class.
        cfg_attack.attack_type = "optimization"
    if max_iterations is not None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        cfg_attack.optim.max_iterations = max_iterations
    if restarts is not None:
        if restarts <= 0:
            raise ValueError("restarts must be positive")
        cfg_attack.restarts.num_trials = restarts
    if callback is not None:
        if callback <= 0:
            raise ValueError("callback must be positive")
        cfg_attack.optim.callback = callback

    setup = {"device": device, "dtype": dtype}
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    attacker = breaching.attacks.prepare_attack(
        attack_model,
        loss_fn,
        cfg_attack,
        setup,
    )

    metadata = _make_metadata(data_spec, DictConfig)
    server_payload = [
        {
            "parameters": _clone_tensors(parameters, device, dtype),
            "buffers": _clone_tensors(tuple(attack_model.buffers()), device, dtype),
            "metadata": metadata,
        }
    ]
    shared_metadata = {
        "num_data_points": num_data_points,
        "labels": None if labels is None else labels.detach().clone().to(device=device).long(),
        "local_hyperparams": None,
    }
    shared_data = [
        {
            "gradients": _clone_tensors(gradients, device, dtype),
            "buffers": None,
            "metadata": shared_metadata,
        }
    ]

    start = time.perf_counter()
    reconstructed, stats = attacker.reconstruct(
        server_payload,
        shared_data,
        {},
        dryrun=dryrun,
    )
    elapsed = time.perf_counter() - start

    try:
        version = importlib.metadata.version("breaching")
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"

    return AttackResult(
        data=reconstructed["data"].detach().cpu(),
        labels=(
            None
            if reconstructed.get("labels") is None
            else reconstructed["labels"].detach().cpu()
        ),
        stats=stats,
        attack=attack,
        elapsed_seconds=elapsed,
        breaching_version=version,
    )

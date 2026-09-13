#!/usr/bin/env python3
"""Benchmark open-source gradient inversion attacks on realistic CIFAR batches.

The registered ``paper`` preset evaluates batch sizes 8/32/64/128 with the attack
implementations shipped by the pinned Breaching repository.  The experiment is
resumable at the (attack, batch size, trial) level.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from hssp_dfl import paths as _paths
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hssp_dfl.gia import (  # noqa: E402
    BREACHING_COMMIT,
    BREACHING_REPOSITORY,
    SUPPORTED_ATTACKS,
    DataSpec,
    align_and_score_batch,
    gradients_from_batch,
    gradients_from_model_delta,
    run_breaching_attack,
)
from hssp_dfl.gia.metrics import denormalize_images  # noqa: E402
from hssp_dfl.models import cnn_cifar  # noqa: E402


_CIFAR_TRAINING_SCALE = 128.0 / 255.0
CIFAR_SPEC = DataSpec(
    shape=(3, 32, 32),
    classes=10,
    # fl.py trains on raw_uint8 / 128 - 1. Expressing that transform as
    # torchvision Normalize keeps direct gradients and snapshot deltas aligned.
    mean=(_CIFAR_TRAINING_SCALE,) * 3,
    std=(_CIFAR_TRAINING_SCALE,) * 3,
)
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "gia_transfer" / "configs" / "gia_large_batch.json"
DEFAULT_CHECKPOINT = _paths.MODEL_DIR / "model_avg_ni1_N10_t0_z0_e1.pkl"

SUMMARY_FIELDS = [
    "status",
    "source",
    "attack",
    "batch_size",
    "trial",
    "seed",
    "known_labels",
    "mean_mse",
    "median_mse",
    "mean_psnr",
    "median_psnr",
    "mean_ssim",
    "median_ssim",
    "label_accuracy",
    "elapsed_seconds",
    "final_objective",
    "breaching_version",
    "breaching_commit",
    "checkpoint",
    "after_checkpoint",
    "lattice_case",
    "lattice_sample_rank",
    "lattice_recon_index",
    "lattice_matched_mse",
    "error",
]
EXAMPLE_FIELDS = [
    "source",
    "attack",
    "batch_size",
    "trial",
    "seed",
    "target_index",
    "reconstructed_index",
    "mse",
    "psnr",
    "ssim",
]
AGGREGATE_FIELDS = [
    "source",
    "attack",
    "batch_size",
    "expected_trials",
    "completed_trials",
    "failed_trials",
    "mean_mse",
    "std_mse",
    "mean_psnr",
    "std_psnr",
    "mean_ssim",
    "std_ssim",
    "mean_elapsed_seconds",
    "std_elapsed_seconds",
]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def parse_attack_list(value: str) -> list[str]:
    attacks = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(attacks) - set(SUPPORTED_ATTACKS))
    if not attacks or invalid:
        raise argparse.ArgumentTypeError(
            f"invalid attacks {invalid}; supported: {', '.join(SUPPORTED_ATTACKS)}"
        )
    return attacks


def parse_attack_config_overrides(value: str) -> dict[str, dict[str, Any]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("attack config overrides must be a JSON object")
    invalid = sorted(set(parsed) - set(SUPPORTED_ATTACKS))
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported override attacks: {invalid}")
    if any(not isinstance(overrides, dict) for overrides in parsed.values()):
        raise argparse.ArgumentTypeError("each attack override must be a JSON object")
    return parsed


def parse_dtype(value: str) -> torch.dtype:
    aliases = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"choose one of {', '.join(aliases)}") from exc


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_presets(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        presets = json.load(handle)
    if not isinstance(presets, dict):
        raise ValueError("preset file must contain a JSON object")
    return presets


def load_snapshot_with_metadata(
    path: Path,
    node: int,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    metadata: Mapping[str, Any] = {}
    if isinstance(snapshot, Mapping) and snapshot.get("format") == "dfl-recovered-states-v1":
        metadata = snapshot.get("metadata", {})
        states_by_node = snapshot["states_by_node"]
        try:
            snapshot = states_by_node[node]
        except KeyError:
            try:
                snapshot = states_by_node[str(node)]
            except KeyError as exc:
                raise ValueError(f"recovered checkpoint contains no node {node}") from exc
    if isinstance(snapshot, (list, tuple)):
        try:
            snapshot = snapshot[node]
        except IndexError as exc:
            raise ValueError(f"checkpoint contains no node {node}") from exc
    if not isinstance(snapshot, Mapping):
        raise TypeError(f"unsupported checkpoint object in {path}")
    return snapshot, metadata


def load_snapshot(path: Path, node: int) -> Mapping[str, torch.Tensor]:
    snapshot, _metadata = load_snapshot_with_metadata(path, node)
    return snapshot


def load_true_gradients_with_metadata(
    path: Path,
    node: int,
    model: torch.nn.Module,
) -> tuple[list[torch.Tensor], Mapping[str, Any]]:
    """Load gradients captured by fl.py before the local optimizer step."""

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping) or payload.get("format") != "dfl-true-gradients-v1":
        raise TypeError(f"unsupported true-gradient object in {path}")
    gradients_by_node = payload.get("gradients_by_node", {})
    try:
        gradient_state = gradients_by_node[node]
    except KeyError:
        try:
            gradient_state = gradients_by_node[str(node)]
        except KeyError as exc:
            raise ValueError(f"true-gradient checkpoint contains no node {node}") from exc
    if not isinstance(gradient_state, Mapping):
        raise TypeError(f"true gradients for node {node} are not a parameter mapping")
    gradients = []
    for name, parameter in model.named_parameters():
        if name not in gradient_state:
            raise KeyError(f"true-gradient checkpoint is missing parameter {name!r}")
        gradient = torch.as_tensor(gradient_state[name])
        if gradient.shape != parameter.shape:
            raise ValueError(
                f"true gradient {name!r} has shape {tuple(gradient.shape)}; "
                f"expected {tuple(parameter.shape)}"
            )
        gradients.append(gradient.detach().clone())
    return gradients, payload.get("metadata", {})


def build_model(
    checkpoint: Path | None,
    node: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    model = cnn_cifar(output_dim=CIFAR_SPEC.classes)
    if checkpoint is not None:
        model.load_state_dict(load_snapshot(checkpoint, node), strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_cifar_batches(
    root: Path,
    batch_sizes: list[int],
    trials: int,
    seed: int,
    download: bool,
) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_SPEC.mean, CIFAR_SPEC.std),
        ]
    )
    dataset = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=download,
        transform=transform,
    )
    max_batch_size = max(batch_sizes)
    required = max_batch_size * trials
    if required > len(dataset):
        raise ValueError(f"requested {required} examples from a dataset of size {len(dataset)}")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(dataset), generator=generator)[:required].tolist()
    batches = {}
    for trial in range(trials):
        trial_indices = permutation[trial * max_batch_size : (trial + 1) * max_batch_size]
        for batch_size in batch_sizes:
            examples = [dataset[index] for index in trial_indices[:batch_size]]
            images = torch.stack([example[0] for example in examples])
            labels = torch.as_tensor([example[1] for example in examples], dtype=torch.long)
            batches[(batch_size, trial)] = (images, labels)
    return batches


def load_snapshot_batch(
    dataset_path: Path,
    node: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with dataset_path.open("rb") as handle:
        split_datasets = pickle.load(handle)
    try:
        images, labels = split_datasets[node]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"dataset pickle contains no node {node}") from exc

    images = torch.as_tensor(images)
    labels = torch.as_tensor(labels).long()
    if images.ndim != 4:
        raise ValueError(f"expected a four-dimensional image batch, got {tuple(images.shape)}")
    if images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
        images = images.permute(0, 3, 1, 2)
    return images, labels


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def append_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def completed_runs(path: Path) -> set[tuple[str, int, int, str]]:
    completed = set()
    if not path.exists():
        return completed
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "completed":
                completed.add(
                    (
                        row["attack"],
                        int(row["batch_size"]),
                        int(row["trial"]),
                        row["source"],
                    )
                )
    return completed


def write_aggregate(path: Path, summary_path: Path, expected_trials: int) -> None:
    """Write latest-status method/batch aggregates for partial or complete runs."""

    latest_rows: dict[tuple[str, str, int, int], dict[str, str]] = {}
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["source"],
                    row["attack"],
                    int(row["batch_size"]),
                    int(row["trial"]),
                )
                latest_rows[key] = row

    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for (source, attack, batch_size, _trial), row in latest_rows.items():
        grouped.setdefault((source, attack, batch_size), []).append(row)

    output_rows = []
    for (source, attack, batch_size), rows in sorted(grouped.items()):
        completed = [row for row in rows if row["status"] == "completed"]
        failed = [row for row in rows if row["status"] != "completed"]
        output = {
            "source": source,
            "attack": attack,
            "batch_size": batch_size,
            "expected_trials": expected_trials,
            "completed_trials": len(completed),
            "failed_trials": len(failed),
        }
        for field in ("mean_mse", "mean_psnr", "mean_ssim", "elapsed_seconds"):
            values = [float(row[field]) for row in completed if row.get(field)]
            output_name = (
                "mean_elapsed_seconds" if field == "elapsed_seconds" else field
            )
            std_name = (
                "std_elapsed_seconds"
                if field == "elapsed_seconds"
                else field.replace("mean_", "std_")
            )
            output[output_name] = float(np.mean(values)) if values else ""
            output[std_name] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else (0.0 if values else "")
            )
        output_rows.append(output)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    temp_path.replace(path)


def summarize(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.median(values))


def aligned_label_accuracy(
    reconstructed_labels: torch.Tensor | None,
    true_labels: torch.Tensor,
    example_metrics: list[dict[str, float | int]],
) -> float:
    if reconstructed_labels is None or reconstructed_labels.numel() != true_labels.numel():
        return float("nan")
    aligned = torch.empty_like(true_labels.cpu())
    reconstructed_labels = reconstructed_labels.reshape(-1).cpu().long()
    for metric in example_metrics:
        aligned[int(metric["target_index"])] = reconstructed_labels[
            int(metric["reconstructed_index"])
        ]
    return float((aligned == true_labels.cpu()).float().mean().item())


def save_image_artifacts(
    output_dir: Path,
    run_name: str,
    target: torch.Tensor,
    aligned: torch.Tensor,
    labels: torch.Tensor,
    reconstructed: torch.Tensor,
    reconstructed_labels: torch.Tensor | None,
) -> None:
    from torchvision.utils import save_image

    run_dir = output_dir / "reconstructions" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    nrow = min(16, max(1, int(math.ceil(math.sqrt(target.shape[0])))))
    target_pixels = denormalize_images(target.cpu(), CIFAR_SPEC.mean, CIFAR_SPEC.std)
    save_image(target_pixels, run_dir / "ground_truth.png", nrow=nrow)
    save_image(aligned.cpu(), run_dir / "reconstruction.png", nrow=nrow)
    torch.save(
        {
            "target": target.cpu(),
            "labels": labels.cpu(),
            "reconstructed": reconstructed.cpu(),
            "reconstructed_labels": reconstructed_labels,
        },
        run_dir / "tensors.pt",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preset", default="quick", choices=("paper", "quick", "smoke"))
    parser.add_argument("--batch-sizes", type=parse_int_list)
    parser.add_argument("--attacks", type=parse_attack_list)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--labels", choices=("known", "inferred"))
    parser.add_argument(
        "--source",
        choices=("direct", "true-gradients", "snapshots"),
        default="direct",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gradient-pickle", type=Path)
    parser.add_argument("--before-checkpoint", type=Path)
    parser.add_argument("--after-checkpoint", type=Path)
    parser.add_argument("--dataset-pickle", type=Path)
    parser.add_argument("--checkpoint-node", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--dataset-root", type=Path, default=_paths.DATA_DIR / "cifar10")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--callback", type=int, default=1000)
    parser.add_argument(
        "--attack-config-overrides",
        type=parse_attack_config_overrides,
        default={},
        help=(
            "JSON mapping from attack name to dotted Breaching config overrides; "
            "stored in the run manifest."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=_paths.RESULTS / "gia_transfer" / "large_batch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Execute one iteration; write results under OUTPUT_DIR/dry-run")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def resolve_experiment(args: argparse.Namespace) -> dict[str, Any]:
    presets = load_presets(args.config)
    if args.preset not in presets:
        raise ValueError(f"unknown preset {args.preset!r} in {args.config}")
    config = dict(presets[args.preset])
    if args.batch_sizes is not None:
        config["batch_sizes"] = args.batch_sizes
    if args.attacks is not None:
        config["attacks"] = args.attacks
    if args.trials is not None:
        config["trials"] = args.trials
    if args.max_iterations is not None:
        config["max_iterations"] = args.max_iterations
    if args.restarts is not None:
        config["restarts"] = args.restarts
    if args.labels is not None:
        config["known_labels"] = args.labels == "known"
    if args.attack_config_overrides:
        config["attack_config_overrides"] = args.attack_config_overrides
    if config["trials"] <= 0:
        raise ValueError("trials must be positive")
    invalid = sorted(set(config["attacks"]) - set(SUPPORTED_ATTACKS))
    if invalid:
        raise ValueError(f"unsupported attacks in preset: {invalid}")
    return config


def prepare_batches_and_model(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]],
    Path | None,
    Mapping[str, Any],
]:
    if args.source == "direct":
        if args.checkpoint is not None and not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        model = build_model(args.checkpoint, args.checkpoint_node, device, args.dtype)
        raw_batches = load_cifar_batches(
            args.dataset_root,
            config["batch_sizes"],
            config["trials"],
            args.seed,
            args.download,
        )
        batches = {
            key: (images, labels, None)
            for key, (images, labels) in raw_batches.items()
        }
        return model, batches, args.checkpoint, {}

    if args.source == "true-gradients":
        required = {
            "--checkpoint": args.checkpoint,
            "--gradient-pickle": args.gradient_pickle,
            "--dataset-pickle": args.dataset_pickle,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"true-gradient source requires {', '.join(missing)}")
        for path in required.values():
            if not path.exists():
                raise FileNotFoundError(path)
        model = build_model(args.checkpoint, args.checkpoint_node, device, args.dtype)
        gradients, source_metadata = load_true_gradients_with_metadata(
            args.gradient_pickle,
            args.checkpoint_node,
            model,
        )
        images, labels = load_snapshot_batch(args.dataset_pickle, args.checkpoint_node)
        batch_size = int(images.shape[0])
        metadata_batch_size = source_metadata.get("batch_size")
        if metadata_batch_size is not None and int(metadata_batch_size) != batch_size:
            raise ValueError(
                f"gradient batch size {metadata_batch_size} does not match "
                f"dataset batch size {batch_size}"
            )
        config["batch_sizes"] = [batch_size]
        config["trials"] = 1
        batches = {(batch_size, 0): (images, labels, gradients)}
        return model, batches, args.checkpoint, source_metadata

    required = {
        "--before-checkpoint": args.before_checkpoint,
        "--after-checkpoint": args.after_checkpoint,
        "--dataset-pickle": args.dataset_pickle,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"snapshot source requires {', '.join(missing)}")
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)

    model = build_model(args.before_checkpoint, args.checkpoint_node, device, args.dtype)
    after, source_metadata = load_snapshot_with_metadata(
        args.after_checkpoint,
        args.checkpoint_node,
    )
    before = load_snapshot(args.before_checkpoint, args.checkpoint_node)
    gradients = gradients_from_model_delta(model, before, after, args.learning_rate)
    images, labels = load_snapshot_batch(args.dataset_pickle, args.checkpoint_node)
    batch_size = int(images.shape[0])
    config["batch_sizes"] = [batch_size]
    config["trials"] = 1
    batches = {(batch_size, 0): (images, labels, gradients)}
    return model, batches, args.before_checkpoint, source_metadata


def main() -> int:
    args = build_parser().parse_args()
    config = resolve_experiment(args)
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir.resolve()
    if args.dry_run:
        output_dir = output_dir / "dry-run"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    examples_path = output_dir / "per_example.csv"
    aggregate_path = output_dir / "aggregate.csv"

    model, batches, checkpoint, source_metadata = prepare_batches_and_model(
        args,
        config,
        device,
    )
    manifest = {
        "command": sys.argv,
        "dry_run": args.dry_run,
        "source": args.source,
        "preset": args.preset,
        "resolved_config": config,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(args.dtype),
        "data_spec": {
            "shape": CIFAR_SPEC.shape,
            "classes": CIFAR_SPEC.classes,
            "mean": CIFAR_SPEC.mean,
            "std": CIFAR_SPEC.std,
        },
        "checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
        "after_checkpoint": (
            None
            if args.source != "snapshots"
            else str(args.after_checkpoint.resolve())
        ),
        "gradient_pickle": (
            None
            if args.source != "true-gradients"
            else str(args.gradient_pickle.resolve())
        ),
        "checkpoint_node": args.checkpoint_node,
        "source_metadata": source_metadata,
        "breaching": {
            "repository": BREACHING_REPOSITORY,
            "commit": BREACHING_COMMIT,
        },
        "threat_model": {
            "server": "honest-but-curious",
            "shared_signal": (
                "mean gradient computed from the private batch"
                if args.source == "direct"
                else (
                    "exact mean gradient captured by fl.py before optimizer.step"
                    if args.source == "true-gradients"
                    else "one-step SGD model delta converted to a mean gradient"
                )
            ),
            "labels": "known" if config["known_labels"] else "inferred by the upstream attack",
            "batch_order": "unknown; Hungarian matching is used before image metrics",
        },
    }
    manifest_path = output_dir / "manifest.json"
    identity = {
        key: manifest[key]
        for key in (
            "source",
            "dry_run",
            "resolved_config",
            "seed",
            "dtype",
            "data_spec",
            "checkpoint",
            "after_checkpoint",
            "gradient_pickle",
            "checkpoint_node",
            "source_metadata",
            "breaching",
            "threat_model",
        )
    }
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        # Older manifests recorded this flag only in the command line.
        previous_manifest.setdefault(
            "dry_run", "--dry-run" in previous_manifest.get("command", [])
        )
        previous_identity = {
            key: previous_manifest.get(key)
            for key in identity
        }
        if json_safe(identity) != previous_identity:
            raise RuntimeError(
                f"{output_dir} contains results from a different configuration; "
                "choose a new --output-dir"
            )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(manifest), handle, indent=2, sort_keys=True)
    completed = completed_runs(summary_path)

    print(
        f"Running on {device}: attacks={config['attacks']}, "
        f"batch_sizes={config['batch_sizes']}, trials={config['trials']}"
    )
    for batch_size in config["batch_sizes"]:
        for trial in range(config["trials"]):
            images_cpu, labels_cpu, provided_gradients = batches[(batch_size, trial)]
            images = images_cpu.to(device=device, dtype=args.dtype)
            labels = labels_cpu.to(device=device).long()
            gradients = (
                gradients_from_batch(model, images, labels)
                if provided_gradients is None
                else [
                    gradient.to(device=device, dtype=args.dtype)
                    for gradient in provided_gradients
                ]
            )

            for attack_index, attack in enumerate(config["attacks"]):
                key = (attack, batch_size, trial, args.source)
                if key in completed:
                    print(f"Skipping completed run {key}")
                    continue

                run_seed = args.seed + trial * 10_000 + batch_size * 100 + attack_index
                run_name = f"{args.source}_{attack}_b{batch_size}_t{trial}_s{run_seed}"
                print(f"\n[{run_name}]")
                summary_row = {
                    "status": "failed",
                    "source": args.source,
                    "attack": attack,
                    "batch_size": batch_size,
                    "trial": trial,
                    "seed": run_seed,
                    "known_labels": config["known_labels"],
                    "breaching_commit": BREACHING_COMMIT,
                    "checkpoint": "" if checkpoint is None else str(checkpoint.resolve()),
                    "after_checkpoint": (
                        ""
                        if args.source != "snapshots"
                        else str(args.after_checkpoint.resolve())
                    ),
                    "lattice_case": source_metadata.get("case", ""),
                    "lattice_sample_rank": source_metadata.get("sample_rank", ""),
                    "lattice_recon_index": source_metadata.get("recon_index", ""),
                    "lattice_matched_mse": source_metadata.get("matched_mse", ""),
                    "error": "",
                }
                try:
                    result = run_breaching_attack(
                        model,
                        gradients,
                        batch_size,
                        CIFAR_SPEC,
                        attack=attack,
                        labels=labels if config["known_labels"] else None,
                        device=device,
                        dtype=args.dtype,
                        max_iterations=config.get("max_iterations"),
                        restarts=config.get("restarts"),
                        callback=args.callback,
                        config_overrides=config.get("attack_config_overrides", {}).get(attack),
                        seed=run_seed,
                        dryrun=args.dry_run,
                    )
                    aligned, example_metrics = align_and_score_batch(
                        result.data,
                        images_cpu.to(dtype=result.data.dtype),
                        CIFAR_SPEC.mean,
                        CIFAR_SPEC.std,
                    )
                    mse_mean, mse_median = summarize(
                        [float(metric["mse"]) for metric in example_metrics]
                    )
                    psnr_mean, psnr_median = summarize(
                        [float(metric["psnr"]) for metric in example_metrics]
                    )
                    ssim_mean, ssim_median = summarize(
                        [float(metric["ssim"]) for metric in example_metrics]
                    )
                    label_accuracy = aligned_label_accuracy(
                        result.labels,
                        labels_cpu,
                        example_metrics,
                    )
                    final_objective = result.stats.get("opt_value", float("nan"))
                    summary_row.update(
                        {
                            "status": "completed",
                            "mean_mse": mse_mean,
                            "median_mse": mse_median,
                            "mean_psnr": psnr_mean,
                            "median_psnr": psnr_median,
                            "mean_ssim": ssim_mean,
                            "median_ssim": ssim_median,
                            "label_accuracy": label_accuracy,
                            "elapsed_seconds": result.elapsed_seconds,
                            "final_objective": final_objective,
                            "breaching_version": result.breaching_version,
                        }
                    )
                    per_example_rows = []
                    for metric in example_metrics:
                        per_example_rows.append(
                            {
                                "source": args.source,
                                "attack": attack,
                                "batch_size": batch_size,
                                "trial": trial,
                                "seed": run_seed,
                                **metric,
                            }
                        )
                    append_csv(examples_path, EXAMPLE_FIELDS, per_example_rows)
                    save_image_artifacts(
                        output_dir,
                        run_name,
                        images_cpu,
                        aligned,
                        labels_cpu,
                        result.data,
                        result.labels,
                    )
                    with (output_dir / "reconstructions" / run_name / "stats.json").open(
                        "w", encoding="utf-8"
                    ) as handle:
                        json.dump(json_safe(result.stats), handle, indent=2, sort_keys=True)
                    print(
                        f"completed: PSNR={psnr_mean:.3f}, SSIM={ssim_mean:.4f}, "
                        f"time={result.elapsed_seconds:.1f}s"
                    )
                except Exception as exc:
                    summary_row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"failed: {summary_row['error']}", file=sys.stderr)
                    traceback.print_exc()
                    if args.fail_fast:
                        append_csv(summary_path, SUMMARY_FIELDS, [summary_row])
                        write_aggregate(
                            aggregate_path,
                            summary_path,
                            config["trials"],
                        )
                        raise
                append_csv(summary_path, SUMMARY_FIELDS, [summary_row])
                write_aggregate(
                    aggregate_path,
                    summary_path,
                    config["trials"],
                )

    print(f"\nResults written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

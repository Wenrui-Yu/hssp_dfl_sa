#!/usr/bin/env python3
"""Summarize paired Breaching attacks on true and HSSP-recovered gradients."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hssp_dfl import paths

from experiments.gia_transfer.gia_large_batch import (
    CIFAR_SPEC,
    build_model,
    load_snapshot,
    load_snapshot_with_metadata,
    load_true_gradients_with_metadata,
)
from hssp_dfl.gia import align_and_score_batch, gradients_from_model_delta
from hssp_dfl.gia.metrics import denormalize_images


BATCH_SIZES = (1, 2, 4, 8)
NODES = (2, 3, 6, 9)
ATTACKS = ("deepleakage", "invertinggradients", "seethroughgradients")
SOURCES = ("true_update", "recovered_solution24")
ATTACK_LABELS = {
    "deepleakage": "Deep Leakage",
    "invertinggradients": "Inverting Gradients",
    "seethroughgradients": "See Through Gradients",
    "wei": "Wei et al.",
}
SOURCE_LABELS = {
    "true_update": "true update",
    "recovered_solution24": "recovered solution (24)",
}
METRICS = ("ssim", "lpips", "psnr")


def parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def parse_csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated non-empty list")
    unknown = sorted(set(parsed) - set(ATTACK_LABELS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown attacks: {unknown}")
    return parsed


def parse_attack_runs(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "attack runs must use attack=subdirectory entries"
            )
        attack, subdirectory = (part.strip() for part in item.split("=", 1))
        if attack not in ATTACK_LABELS:
            raise argparse.ArgumentTypeError(f"unknown attack in run mapping: {attack}")
        if not subdirectory or Path(subdirectory).is_absolute() or ".." in Path(subdirectory).parts:
            raise argparse.ArgumentTypeError(
                f"run subdirectory must be a safe relative path: {subdirectory!r}"
            )
        mapping[attack] = subdirectory
    if not mapping:
        raise argparse.ArgumentTypeError("attack run mapping cannot be empty")
    return mapping


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    std = 0.0 if array.size == 1 else float(array.std(ddof=1))
    return float(array.mean()), std


def locate_tensors(source_dir: Path, attack: str, batch_size: int) -> Path:
    matches = sorted(
        source_dir.glob(
            f"reconstructions/*_{attack}_b{batch_size}_t0_s*/tensors.pt"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one tensors.pt for {attack} in {source_dir}, found {matches}"
        )
    return matches[0]


def load_lpips():
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("lpips==0.1.4 is required to compute LPIPS-Alex") from exc
    model = lpips.LPIPS(net="alex", verbose=False)
    model.eval()
    return model


def collect_per_example(
    root: Path,
    batch_sizes: Iterable[int] = BATCH_SIZES,
    nodes: Iterable[int] = NODES,
    attacks: Iterable[str] = ATTACKS,
    runs_subdir: str = "breaching",
    attack_runs: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    attack_runs = {} if attack_runs is None else attack_runs
    lpips_model = load_lpips()
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        for node in nodes:
            for source in SOURCES:
                for attack in attacks:
                    source_dir = (
                        root
                        / attack_runs.get(attack, runs_subdir)
                        / f"b{batch_size}"
                        / f"node{node}"
                        / source
                    )
                    tensors_path = locate_tensors(source_dir, attack, batch_size)
                    payload = torch.load(tensors_path, map_location="cpu", weights_only=False)
                    target = payload["target"].float()
                    reconstructed = payload["reconstructed"].float()
                    aligned, metrics = align_and_score_batch(
                        reconstructed,
                        target,
                        CIFAR_SPEC.mean,
                        CIFAR_SPEC.std,
                    )
                    target_pixels = denormalize_images(
                        target, CIFAR_SPEC.mean, CIFAR_SPEC.std
                    )
                    with torch.no_grad():
                        lpips_values = (
                            lpips_model(aligned * 2 - 1, target_pixels * 2 - 1)
                            .reshape(-1)
                            .cpu()
                        )
                    if lpips_values.numel() != len(metrics):
                        raise RuntimeError(f"LPIPS returned wrong shape for {tensors_path}")
                    for metric, lpips_value in zip(metrics, lpips_values):
                        rows.append(
                            {
                                "batch_size": batch_size,
                                "node": node,
                                "source": source,
                                "attack": attack,
                                "target_index": metric["target_index"],
                                "reconstructed_index": metric["reconstructed_index"],
                                "mse": metric["mse"],
                                "ssim": metric["ssim"],
                                "lpips": float(lpips_value.item()),
                                "psnr": metric["psnr"],
                                "tensors_path": str(tensors_path.resolve()),
                            }
                        )
    return rows


def aggregate_rows(per_example: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_example:
        groups[(row["batch_size"], row["attack"], row["source"])].append(row)
    output = []
    for (batch_size, attack, source), rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "batch_size": batch_size,
            "attack": attack,
            "source": source,
            "num_images": len(rows),
            "num_nodes": len({row["node"] for row in rows}),
        }
        for metric in ("mse", *METRICS):
            mean, std = mean_std(float(row[metric]) for row in rows)
            summary[f"mean_{metric}"] = mean
            summary[f"std_{metric}"] = std
        output.append(summary)
    return output


def paired_rows(per_example: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in per_example:
        key = (row["batch_size"], row["node"], row["attack"], row["target_index"])
        pairs[key][row["source"]] = row
    output = []
    for (batch_size, node, attack, target_index), sources in sorted(pairs.items()):
        if set(sources) != set(SOURCES):
            raise RuntimeError(f"unpaired result for {(batch_size, node, attack, target_index)}")
        row = {
            "batch_size": batch_size,
            "node": node,
            "attack": attack,
            "target_index": target_index,
        }
        for metric in METRICS:
            true_value = float(sources["true_update"][metric])
            recovered_value = float(sources["recovered_solution24"][metric])
            row[f"true_{metric}"] = true_value
            row[f"recovered_{metric}"] = recovered_value
            row[f"delta_{metric}"] = recovered_value - true_value
            row[f"abs_delta_{metric}"] = abs(recovered_value - true_value)
        output.append(row)
    return output


def paired_summary_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[(row["batch_size"], row["attack"])].append(row)
    output = []
    for (batch_size, attack), rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "batch_size": batch_size,
            "attack": attack,
            "num_pairs": len(rows),
        }
        for metric in METRICS:
            deltas = np.asarray([row[f"delta_{metric}"] for row in rows], dtype=float)
            summary[f"mean_delta_{metric}"] = float(deltas.mean())
            summary[f"mean_abs_delta_{metric}"] = float(np.abs(deltas).mean())
            summary[f"max_abs_delta_{metric}"] = float(np.abs(deltas).max())
        output.append(summary)
    return output


def locate_recovered_solution(root: Path, batch_size: int, sample_rank: int = 24) -> Path:
    matches = sorted(
        (root / "hssp" / f"b{batch_size}").glob(
            f"recovered_ni{batch_size}_sg0_case1_exact_pattern_sel{sample_rank}_idx*.pkl"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Case 1 solution {sample_rank} for batch {batch_size}, "
            f"found {matches}"
        )
    return matches[0]


def gradient_audit(
    root: Path,
    batch_sizes: Iterable[int] = BATCH_SIZES,
    nodes: Iterable[int] = NODES,
    solution_rank: int = 24,
) -> list[dict[str, Any]]:
    rows = []
    for batch_size in batch_sizes:
        before_path = root / "snapshots" / f"b{batch_size}" / "models" / (
            f"model_avg_ni{batch_size}_N10_t0_z0_e1.pkl"
        )
        true_path = root / "true_gradients" / f"b{batch_size}" / (
            f"true_gradients_ni{batch_size}_N10_t0_z0_e1.pkl"
        )
        recovered_path = locate_recovered_solution(root, batch_size, solution_rank)
        for node in nodes:
            model = build_model(before_path, node, torch.device("cpu"), torch.float64)
            true, _ = load_true_gradients_with_metadata(true_path, node, model)
            before = load_snapshot(before_path, node)
            after, metadata = load_snapshot_with_metadata(recovered_path, node)
            recovered = gradients_from_model_delta(model, before, after, 0.01)
            true_vector = torch.cat([tensor.reshape(-1).double() for tensor in true])
            recovered_vector = torch.cat(
                [tensor.reshape(-1).double() for tensor in recovered]
            )
            delta = recovered_vector - true_vector
            rows.append(
                {
                    "batch_size": batch_size,
                    "node": node,
                    "case": metadata.get("case"),
                    "sample_rank": metadata.get("sample_rank"),
                    "recon_index": metadata.get("recon_index"),
                    "hssp_state_mse": metadata.get("matched_mse"),
                    "gradient_relative_l2": float(delta.norm() / true_vector.norm()),
                    "gradient_cosine": float(
                        torch.nn.functional.cosine_similarity(
                            true_vector, recovered_vector, dim=0
                        )
                    ),
                    "gradient_max_abs": float(delta.abs().max()),
                }
            )
    return rows


def controlled_data_audit(
    root: Path,
    batch_sizes: Iterable[int] = BATCH_SIZES,
) -> list[dict[str, Any]]:
    batch_sizes = tuple(batch_sizes)
    datasets = {}
    checkpoints = {}
    for batch_size in batch_sizes:
        dataset_path = root / "snapshots" / f"b{batch_size}" / "datasets" / (
            f"dataset_ni{batch_size}_N10.pkl"
        )
        checkpoint_path = root / "snapshots" / f"b{batch_size}" / "models" / (
            f"model_avg_ni{batch_size}_N10_t0_z0_e1.pkl"
        )
        with dataset_path.open("rb") as handle:
            datasets[batch_size] = pickle.load(handle)
        with checkpoint_path.open("rb") as handle:
            checkpoints[batch_size] = pickle.load(handle)

    rows = []
    reference_batch = max(batch_sizes)
    for batch_size in batch_sizes:
        for node in range(10):
            images, labels = datasets[batch_size][node]
            reference_images, reference_labels = datasets[reference_batch][node]
            state = checkpoints[batch_size][node]
            reference_state = checkpoints[reference_batch][node]
            max_abs = max(
                float((torch.as_tensor(state[name]) - torch.as_tensor(reference_state[name])).abs().max())
                for name in state
            )
            rows.append(
                {
                    "batch_size": batch_size,
                    "node": node,
                    "reference_batch_size": reference_batch,
                    "nested_images_equal": bool(
                        torch.equal(torch.as_tensor(images), torch.as_tensor(reference_images)[:batch_size])
                    ),
                    "nested_labels_equal": bool(
                        torch.equal(torch.as_tensor(labels), torch.as_tensor(reference_labels)[:batch_size])
                    ),
                    "initial_model_max_abs_vs_b8": max_abs,
                }
            )
    if any(
        not row["nested_images_equal"]
        or not row["nested_labels_equal"]
        or row["initial_model_max_abs_vs_b8"] != 0
        for row in rows
    ):
        raise RuntimeError("controlled nested-batch audit failed")
    return rows


def check_manifests(
    root: Path,
    batch_sizes: Iterable[int] = BATCH_SIZES,
    nodes: Iterable[int] = NODES,
    attacks: Iterable[str] = ATTACKS,
    runs_subdir: str = "breaching",
    attack_runs: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    batch_sizes = tuple(batch_sizes)
    nodes = tuple(nodes)
    attacks = tuple(attacks)
    attack_runs = {} if attack_runs is None else attack_runs
    rows = []
    for batch_size in batch_sizes:
        for node in nodes:
            for source in SOURCES:
                for attack in attacks:
                    run_subdir = attack_runs.get(attack, runs_subdir)
                    directory = root / run_subdir / f"b{batch_size}" / f"node{node}" / source
                    with (directory / "manifest.json").open(encoding="utf-8") as handle:
                        manifest = json.load(handle)
                    with (directory / "summary.csv").open(newline="", encoding="utf-8") as handle:
                        summary = list(csv.DictReader(handle))
                    completed = [
                        row
                        for row in summary
                        if row["status"] == "completed" and row["attack"] == attack
                    ]
                    config = manifest["resolved_config"]
                    rows.append(
                        {
                            "batch_size": batch_size,
                            "node": node,
                            "source": source,
                            "attack": attack,
                            "runs_subdir": run_subdir,
                            "completed_runs": len(completed),
                            "max_iterations": config["max_iterations"],
                            "restarts": config["restarts"],
                            "known_labels": config["known_labels"],
                            "attack_config_overrides": json.dumps(
                                config.get("attack_config_overrides", {}).get(attack, {}),
                                sort_keys=True,
                            ),
                            "breaching_commit": manifest["breaching"]["commit"],
                        }
                    )
    expected_rows = len(batch_sizes) * len(nodes) * len(SOURCES) * len(attacks)
    if len(rows) != expected_rows or any(
        row["completed_runs"] != 1
        for row in rows
    ):
        raise RuntimeError("formal Breaching results are incomplete")
    return rows


def make_wide_rows(
    aggregate: list[dict[str, Any]],
    batch_sizes: Iterable[int] = BATCH_SIZES,
    attacks: Iterable[str] = ATTACKS,
) -> list[dict[str, Any]]:
    lookup = {
        (row["batch_size"], row["attack"], row["source"]): row for row in aggregate
    }
    output = []
    for batch_size in batch_sizes:
        row: dict[str, Any] = {"batch_size": batch_size}
        for attack in attacks:
            for source in SOURCES:
                values = lookup[(batch_size, attack, source)]
                prefix = f"{attack}_{source}"
                for metric in METRICS:
                    row[f"{prefix}_{metric}"] = values[f"mean_{metric}"]
        output.append(row)
    return output


def make_markdown(
    aggregate: list[dict[str, Any]],
    batch_sizes: Iterable[int] = BATCH_SIZES,
    attacks: Iterable[str] = ATTACKS,
    nodes: Iterable[int] = NODES,
    manifest_rows: Iterable[dict[str, Any]] | None = None,
) -> str:
    batch_sizes = tuple(batch_sizes)
    attacks = tuple(attacks)
    nodes = tuple(nodes)
    lookup = {
        (row["batch_size"], row["attack"], row["source"]): row for row in aggregate
    }
    headers = ["Batch size"]
    for attack in attacks:
        for source in SOURCES:
            headers.append(f"{ATTACK_LABELS[attack]} on {SOURCE_LABELS[source]}")
    lines = [
        "# Paired Breaching GIA results",
        "",
        "Each cell reports mean SSIM / LPIPS-Alex / PSNR (dB) over the selected honest nodes and all images in the batch.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for batch_size in batch_sizes:
        cells = [str(batch_size)]
        for attack in attacks:
            for source in SOURCES:
                row = lookup[(batch_size, attack, source)]
                cells.append(
                    f"{row['mean_ssim']:.4f} / {row['mean_lpips']:.4f} / "
                    f"{row['mean_psnr']:.2f}"
                )
        lines.append("| " + " | ".join(cells) + " |")
    settings = (
        "Settings: Breaching commit `faaab88e618dfa94d77b062ad50ac4a27afb1534`; "
        "known labels; one restart; paired initialization seeds; honest nodes "
        + ", ".join(str(node) for node in nodes)
    )
    if manifest_rows is not None:
        manifest_rows = list(manifest_rows)
        iteration_parts = []
        override_parts = []
        for attack in attacks:
            attack_rows = [row for row in manifest_rows if row["attack"] == attack]
            iterations = sorted({int(row["max_iterations"]) for row in attack_rows})
            if len(iterations) != 1:
                raise RuntimeError(f"inconsistent iteration budgets for {attack}: {iterations}")
            iteration_parts.append(f"{ATTACK_LABELS[attack]} {iterations[0]}")
            overrides = sorted(
                {row["attack_config_overrides"] for row in attack_rows}
            )
            if overrides != ["{}"]:
                if len(overrides) != 1:
                    raise RuntimeError(f"inconsistent overrides for {attack}: {overrides}")
                override_parts.append(f"{ATTACK_LABELS[attack]} {overrides[0]}")
        settings += "; max iterations: " + ", ".join(iteration_parts)
        if override_parts:
            settings += "; overrides: " + ", ".join(override_parts)
    lines.extend(["", settings + ".", ""])
    return "\n".join(lines)


def make_latex(
    aggregate: list[dict[str, Any]],
    batch_sizes: Iterable[int] = BATCH_SIZES,
    attacks: Iterable[str] = ATTACKS,
    nodes: Iterable[int] = NODES,
) -> str:
    batch_sizes = tuple(batch_sizes)
    attacks = tuple(attacks)
    nodes = tuple(nodes)
    lookup = {
        (row["batch_size"], row["attack"], row["source"]): row for row in aggregate
    }
    column_spec = "r" + "cc" * len(attacks)
    attack_header = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{ATTACK_LABELS[attack]}}}" for attack in attacks
    )
    cmidrules = "".join(
        f"\\cmidrule(lr){{{2 + 2 * index}-{3 + 2 * index}}}"
        for index in range(len(attacks))
    )
    source_header = " & ".join(
        [r"$|\mathcal{B}_i^{(0)}|$"]
        + [label for _attack in attacks for label in ("True", "Recovered (24)")]
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired downstream GIA performance on the exact gradient $\boldsymbol{g}_{i,\mathrm{true}}^{(0)}$ and the gradient $\widehat{\boldsymbol{g}}_i^{(0)}$ derived from recovered Case~1 solution 24. Each entry is mean SSIM / LPIPS-Alex / PSNR (dB) over the selected honest nodes and all aligned images.}",
        r"\label{tab:gia_true_vs_recovered}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "& " + attack_header + r" \\",
        cmidrules,
        source_header + r" \\",
        r"\midrule",
    ]
    for batch_size in batch_sizes:
        cells = [str(batch_size)]
        for attack in attacks:
            for source in SOURCES:
                row = lookup[(batch_size, attack, source)]
                cells.append(
                    f"{row['mean_ssim']:.4f} / {row['mean_lpips']:.4f} / "
                    f"{row['mean_psnr']:.2f}"
                )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def make_plot(
    aggregate: list[dict[str, Any]],
    output: Path,
    batch_sizes: Iterable[int] = BATCH_SIZES,
    attacks: Iterable[str] = ATTACKS,
) -> None:
    import matplotlib.pyplot as plt

    batch_sizes = tuple(batch_sizes)
    attacks = tuple(attacks)
    lookup = {
        (row["batch_size"], row["attack"], row["source"]): row for row in aggregate
    }
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    colors = {attack: palette[index % len(palette)] for index, attack in enumerate(attacks)}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
    for axis, metric, ylabel in zip(
        axes, METRICS, ("SSIM ↑", "LPIPS-Alex ↓", "PSNR (dB) ↑")
    ):
        for attack in attacks:
            for source, linestyle, marker, alpha in (
                ("true_update", "-", "o", 1.0),
                ("recovered_solution24", "--", "x", 0.85),
            ):
                values = [
                    lookup[(batch_size, attack, source)][f"mean_{metric}"]
                    for batch_size in batch_sizes
                ]
                axis.plot(
                    batch_sizes,
                    values,
                    color=colors[attack],
                    linestyle=linestyle,
                    marker=marker,
                    alpha=alpha,
                    label=f"{ATTACK_LABELS[attack]} — {SOURCE_LABELS[source]}",
                )
        axis.set_xscale("log", base=2)
        axis.set_xticks(batch_sizes, [str(value) for value in batch_sizes])
        axis.set_xlabel("Batch size")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=max(1, min(3, len(attacks) * len(SOURCES))),
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=paths.RESULTS / "gia_transfer"
    )
    parser.add_argument("--batch-sizes", type=parse_csv_ints, default=BATCH_SIZES)
    parser.add_argument("--nodes", type=parse_csv_ints, default=NODES)
    parser.add_argument("--attacks", type=parse_csv_strings, default=ATTACKS)
    parser.add_argument("--runs-subdir", default="breaching")
    parser.add_argument(
        "--attack-runs",
        type=parse_attack_runs,
        default={},
        help="Optional per-attack result directories, e.g. attack=directory,...",
    )
    parser.add_argument("--report-subdir", default="report")
    parser.add_argument("--solution-rank", type=int, default=24)
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = root / args.report_subdir

    manifest_rows = check_manifests(
        root,
        args.batch_sizes,
        args.nodes,
        args.attacks,
        args.runs_subdir,
        args.attack_runs,
    )
    controlled_rows = controlled_data_audit(root, args.batch_sizes)
    audit_rows = gradient_audit(
        root,
        args.batch_sizes,
        args.nodes,
        args.solution_rank,
    )
    per_example = collect_per_example(
        root,
        args.batch_sizes,
        args.nodes,
        args.attacks,
        args.runs_subdir,
        args.attack_runs,
    )
    aggregate = aggregate_rows(per_example)
    paired = paired_rows(per_example)
    paired_summary = paired_summary_rows(paired)
    wide_rows = make_wide_rows(aggregate, args.batch_sizes, args.attacks)

    write_csv(
        report_dir / "per_example_metrics.csv",
        per_example,
        (
            "batch_size", "node", "source", "attack", "target_index",
            "reconstructed_index", "mse", "ssim", "lpips", "psnr", "tensors_path",
        ),
    )
    write_csv(report_dir / "batch_summary_long.csv", aggregate, aggregate[0].keys())
    write_csv(report_dir / "paper_table.csv", wide_rows, wide_rows[0].keys())
    write_csv(report_dir / "paired_differences.csv", paired, paired[0].keys())
    write_csv(
        report_dir / "paired_difference_summary.csv",
        paired_summary,
        paired_summary[0].keys(),
    )
    write_csv(report_dir / "gradient_equivalence_audit.csv", audit_rows, audit_rows[0].keys())
    write_csv(
        report_dir / "controlled_data_audit.csv",
        controlled_rows,
        controlled_rows[0].keys(),
    )
    write_csv(report_dir / "run_manifest_audit.csv", manifest_rows, manifest_rows[0].keys())
    (report_dir / "paper_table.md").write_text(
        make_markdown(
            aggregate,
            args.batch_sizes,
            args.attacks,
            args.nodes,
            manifest_rows,
        ),
        encoding="utf-8",
    )
    (report_dir / "paper_table.tex").write_text(
        make_latex(aggregate, args.batch_sizes, args.attacks, args.nodes),
        encoding="utf-8",
    )
    make_plot(
        aggregate,
        report_dir / "gia_true_vs_recovered.png",
        args.batch_sizes,
        args.attacks,
    )

    print(f"Wrote {len(per_example)} per-image rows and {len(aggregate)} summaries")
    print(f"Report: {report_dir / 'paper_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Downstream GIA transferability -- paper Table 6 (Appendix J).

Runs the full pipeline end to end and is resumable at every stage:

    1. train CIFAR-10 DFL checkpoints at batch sizes 1, 2, 4 and 8
       (training/train_cifar.py)
    2. break secure aggregation and export the Case 1, rank-24 recovered
       client updates (experiments/attack_cifar.py --attack-mode none)
    3. invert both the ground-truth update g and the recovered update g-hat
       with three Breaching attacks: Deep Leakage (Zhu et al.), Inverting
       Gradients (Geiping et al.) and See Through Gradients (Yin et al.)
    4. aggregate the paired SSIM / LPIPS / PSNR into Table 6

This is the only experiment that needs a dependency outside the main
environment; see experiments/gia_transfer/README.md and install
experiments/gia_transfer/requirements.txt first.
"""


from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from hssp_dfl import paths

DEFAULT_OUTPUT_ROOT = paths.RESULTS / "gia_transfer"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_NODES = (2, 3, 6, 9)
SOURCES = ("true_update", "recovered_solution24")
ITERATION_BUDGETS = {
    "deepleakage": 300,
    "invertinggradients": 300,
    "seethroughgradients": 2000,
}
ATTACKS = ("deepleakage", "invertinggradients", "seethroughgradients")
BASELINE_ATTACKS = ("deepleakage", "invertinggradients", "seethroughgradients")
BASELINE_RUNS_SUBDIR = "breaching"
STG_RUNS_SUBDIR = "stg_tuned2000"
REPORT_SUBDIR = "report_tuned_stg"
STG_OVERRIDES = {
    "seethroughgradients": {
        "optim.langevin_noise": 0.0,
        "objective.scale": 0.01,
        "init": "patterned-4",
    }
}


def parse_positive_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    additions = (
        PROJECT_ROOT / "tmp" / "breaching_paired" / "deps",
        PROJECT_ROOT / "tmp" / "breaching_paired" / "source" / "breaching",
        PROJECT_ROOT,
    )
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in additions] + ([previous] if previous else [])
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.output_root.resolve()
        self.env = build_environment()
        self.commands: list[list[str]] = []
        self.events: list[dict[str, Any]] = []

    def run_command(
        self,
        command: list[str],
        *,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        command = [str(item) for item in command]
        self.commands.append(command)
        print("+ " + shlex.join(command), flush=True)
        env = self.env.copy()
        if extra_env:
            env.update(extra_env)
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

    def fl_paths(self, batch_size: int) -> dict[str, Path]:
        snapshot_dir = self.root / "snapshots" / f"b{batch_size}"
        model_dir = snapshot_dir / "models"
        return {
            "snapshot_dir": snapshot_dir,
            "dataset": snapshot_dir / "datasets" / f"dataset_ni{batch_size}_N10.pkl",
            "before": model_dir / f"model_avg_ni{batch_size}_N10_t0_z0_e1.pkl",
            "local_after": model_dir / f"model_avg_ni{batch_size}_N10_t0_5_z0_e1.pkl",
            "mixed_after": model_dir / f"model_avg_ni{batch_size}_N10_t1_z0_e1.pkl",
            "gradient": self.root / "true_gradients" / f"b{batch_size}" / (
                f"true_gradients_ni{batch_size}_N10_t0_z0_e1.pkl"
            ),
        }

    def valid_fl_outputs(self, batch_size: int) -> bool:
        paths = self.fl_paths(batch_size)
        required = (
            paths["dataset"],
            paths["before"],
            paths["local_after"],
            paths["mixed_after"],
            paths["gradient"],
        )
        if not all(path.is_file() for path in required):
            return False
        try:
            with paths["gradient"].open("rb") as handle:
                payload = pickle.load(handle)
            metadata = payload["metadata"]
            gradients = payload["gradients_by_node"]
            return (
                payload.get("format") == "dfl-true-gradients-v1"
                and int(metadata["batch_size"]) == batch_size
                and int(metadata["seed"]) == self.args.training_seed
                and int(metadata["nested_batch_max_size"]) == self.args.nested_batch_max_size
                and int(metadata["local_epochs"]) == 1
                and all(node in gradients or str(node) in gradients for node in self.args.nodes)
            )
        except (KeyError, TypeError, ValueError, pickle.UnpicklingError):
            return False

    def run_fl(self) -> None:
        for batch_size in self.args.batch_sizes:
            if self.valid_fl_outputs(batch_size):
                print(f"[reuse] valid train_cifar.py outputs for batch size {batch_size}")
                self.events.append({"stage": "fl", "batch_size": batch_size, "status": "reused"})
                continue
            paths = self.fl_paths(batch_size)
            paths["snapshot_dir"].mkdir(parents=True, exist_ok=True)
            paths["gradient"].parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                PROJECT_ROOT / "training" / "train_cifar.py",
                "--batch-size", str(batch_size),
                "--num-clients", "10",
                "--num-communications", "2",
                "--local-epochs", "1",
                "--iid", "1",
                "--graph", "load",
                "--dataset", "cifar10",
                "--model", "cnn",
                "--seed", str(self.args.training_seed),
                "--nested-batch-max-size", str(self.args.nested_batch_max_size),
                "--cuda-visible-devices", "",
                "--output-dir", paths["snapshot_dir"],
                "--gradient-output-dir", paths["gradient"].parent,
            ]
            self.run_command(command)
            if not self.valid_fl_outputs(batch_size):
                raise RuntimeError(f"train_cifar.py outputs failed validation for batch {batch_size}")
            self.events.append({"stage": "fl", "batch_size": batch_size, "status": "completed"})

    def locate_solution(self, batch_size: int) -> Path | None:
        matches = sorted(
            (self.root / "hssp" / f"b{batch_size}").glob(
                f"recovered_ni{batch_size}_sg0_case1_exact_pattern_"
                f"sel{self.args.solution_rank}_idx*.pkl"
            )
        )
        valid = []
        for path in matches:
            try:
                with path.open("rb") as handle:
                    payload = pickle.load(handle)
                metadata = payload["metadata"]
                states = payload["states_by_node"]
                if (
                    payload.get("format") == "dfl-recovered-states-v1"
                    and int(metadata["batch_size"]) == batch_size
                    and metadata["case"] == "case1_exact_pattern"
                    and int(metadata["sample_rank"]) == self.args.solution_rank
                    and all(node in states or str(node) in states for node in self.args.nodes)
                ):
                    valid.append(path)
            except (KeyError, TypeError, ValueError, pickle.UnpicklingError):
                continue
        if len(valid) > 1:
            raise RuntimeError(f"ambiguous recovered solution {self.args.solution_rank}: {valid}")
        return valid[0] if valid else None

    def run_hssp(self) -> None:
        for batch_size in self.args.batch_sizes:
            paths = self.fl_paths(batch_size)
            if not self.valid_fl_outputs(batch_size):
                raise RuntimeError(f"missing valid train_cifar.py inputs for batch {batch_size}")
            solution = self.locate_solution(batch_size)
            if solution is not None:
                print(f"[reuse] {solution}")
                self.events.append({"stage": "hssp", "batch_size": batch_size, "status": "reused"})
                continue
            export_dir = self.root / "hssp" / f"b{batch_size}"
            export_dir.mkdir(parents=True, exist_ok=True)
            # The attack reads its checkpoints through hssp_dfl.paths, so the
            # per-batch snapshot directory produced above is injected there.
            extra_env = {
                "HSSP_MODEL_DIR": str(paths["snapshot_dir"] / "models"),
                "HSSP_DATASET_DIR": str(paths["snapshot_dir"] / "datasets"),
            }
            self.run_command(
                [sys.executable, PROJECT_ROOT / "experiments" / "attack_cifar.py",
                 "--batch-size", str(batch_size),
                 "--attack-mode", "none",
                 "--export-recovered-states", str(export_dir),
                 "--export-cases", "case1_exact_pattern",
                 "--export-sample-ranks", str(self.args.solution_rank),
                 "--output-dir", str(paths["snapshot_dir"] / "attack")],
                extra_env=extra_env,
            )
            solution = self.locate_solution(batch_size)
            if solution is None:
                raise RuntimeError(
                    f"HSSP did not export Case 1 solution {self.args.solution_rank} "
                    f"for batch {batch_size}"
                )
            self.events.append({"stage": "hssp", "batch_size": batch_size, "status": "completed"})

    def gia_command(
        self,
        batch_size: int,
        node: int,
        source: str,
        solution: Path,
        *,
        profile: str,
    ) -> list[str]:
        paths = self.fl_paths(batch_size)
        if profile == "baseline":
            attacks = BASELINE_ATTACKS
            iterations = 300
            runs_subdir = BASELINE_RUNS_SUBDIR
            overrides = None
        elif profile == "stg_tuned":
            attacks = ("seethroughgradients",)
            iterations = 2000
            runs_subdir = STG_RUNS_SUBDIR
            overrides = STG_OVERRIDES
        else:
            raise ValueError(f"unknown GIA profile: {profile}")
        output_dir = self.root / runs_subdir / f"b{batch_size}" / f"node{node}" / source
        common = [
            sys.executable,
            PROJECT_ROOT / "experiments" / "gia_transfer" / "gia_large_batch.py",
            "--preset", "quick",
            "--attacks", ",".join(attacks),
            "--trials", "1",
            "--max-iterations", str(iterations),
            "--restarts", "1",
            "--labels", "known",
            "--dataset-pickle", paths["dataset"],
            "--checkpoint-node", str(node),
            "--device", self.args.device,
            "--seed", str(self.args.attack_seed),
            "--callback", "1000",
            "--output-dir", output_dir,
            "--fail-fast",
        ]
        if overrides is not None:
            common.extend(
                ["--attack-config-overrides", json.dumps(overrides, separators=(",", ":"))]
            )
        if source == "true_update":
            return common + [
                "--source", "true-gradients",
                "--checkpoint", paths["before"],
                "--gradient-pickle", paths["gradient"],
            ]
        return common + [
            "--source", "snapshots",
            "--before-checkpoint", paths["before"],
            "--after-checkpoint", solution,
            "--learning-rate", "0.01",
        ]

    def run_gia(self) -> None:
        for batch_size in self.args.batch_sizes:
            if not self.valid_fl_outputs(batch_size):
                raise RuntimeError(f"missing valid train_cifar.py inputs for batch {batch_size}")
            solution = self.locate_solution(batch_size)
            if solution is None:
                raise RuntimeError(
                    f"missing HSSP solution {self.args.solution_rank} for batch {batch_size}"
                )
            for node in self.args.nodes:
                for source in SOURCES:
                    for profile in ("baseline", "stg_tuned"):
                        self.run_command(
                            self.gia_command(
                                batch_size,
                                node,
                                source,
                                solution,
                                profile=profile,
                            )
                        )
                        self.events.append(
                            {
                                "stage": "gia",
                                "profile": profile,
                                "batch_size": batch_size,
                                "node": node,
                                "source": source,
                                "status": "completed_or_reused",
                            }
                        )

    def run_report(self) -> None:
        command = [
            sys.executable,
            PROJECT_ROOT / "experiments" / "gia_transfer" / "summarize.py",
            "--root", self.root,
            "--batch-sizes", ",".join(str(value) for value in self.args.batch_sizes),
            "--nodes", ",".join(str(value) for value in self.args.nodes),
            "--attacks", ",".join(ATTACKS),
            "--runs-subdir", BASELINE_RUNS_SUBDIR,
            "--attack-runs", (
                f"deepleakage={BASELINE_RUNS_SUBDIR},"
                f"invertinggradients={BASELINE_RUNS_SUBDIR},"
                f"seethroughgradients={STG_RUNS_SUBDIR}"
            ),
            "--report-subdir", REPORT_SUBDIR,
            "--solution-rank", str(self.args.solution_rank),
        ]
        self.run_command(command)
        self.events.append({"stage": "report", "status": "completed"})

    def write_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "dfl-three-gia-paired-pipeline-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "invocation": sys.argv,
            "python": sys.executable,
            "configuration": {
                "batch_sizes": self.args.batch_sizes,
                "nodes": self.args.nodes,
                "training_seed": self.args.training_seed,
                "attack_seed": self.args.attack_seed,
                "nested_batch_max_size": self.args.nested_batch_max_size,
                "solution_rank": self.args.solution_rank,
                "attacks": ATTACKS,
                "iteration_budgets": ITERATION_BUDGETS,
                "attack_runs": {
                    "deepleakage": BASELINE_RUNS_SUBDIR,
                    "invertinggradients": BASELINE_RUNS_SUBDIR,
                    "seethroughgradients": STG_RUNS_SUBDIR,
                },
                "stg_overrides": STG_OVERRIDES,
                "restarts": 1,
                "labels": "known",
                "device": self.args.device,
            },
            "events": self.events,
            "commands": self.commands,
            "final_table": str((self.root / REPORT_SUBDIR / "paper_table.md").resolve()),
        }
        (self.root / "three_gia_pipeline_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-sizes", type=parse_positive_ints, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--nodes", type=parse_positive_ints, default=DEFAULT_NODES)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--attack-seed", type=int, default=42)
    parser.add_argument("--nested-batch-max-size", type=int, default=8)
    parser.add_argument("--solution-rank", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stage",
        choices=("all", "fl", "hssp", "gia", "report"),
        default="all",
        help="Run the full pipeline or one resumable stage.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.training_seed < 0 or args.attack_seed < 0:
        raise ValueError("seeds must be non-negative")
    if args.nested_batch_max_size < max(args.batch_sizes):
        raise ValueError("nested batch max size must cover the largest batch")
    if args.solution_rank <= 0:
        raise ValueError("solution rank must be positive")

    pipeline = Pipeline(args)
    stages = ("fl", "hssp", "gia", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        getattr(pipeline, f"run_{stage}")()
    pipeline.write_manifest()
    print(f"Final Markdown table: {pipeline.root / REPORT_SUBDIR / 'paper_table.md'}")
    print(f"Final CSV table: {pipeline.root / REPORT_SUBDIR / 'paper_table.csv'}")
    print(f"Final LaTeX table: {pipeline.root / REPORT_SUBDIR / 'paper_table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

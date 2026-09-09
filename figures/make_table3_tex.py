#!/usr/bin/env python3
"""Render paper Table 3 (finite-precision robustness) as a booktabs LaTeX table.

Input: the per-trial CSV written by ``experiments/finite_precision.py``.

The table layout is:

    Dataset | condition/scale | exact clean | r=1 | ... | r=8

For each dataset, the first row is the requested exact-clean baseline.  The
following rows contain the noisy results for the requested scales.  A cell is
formatted as ``recall / candidate-count`` and ``1.00 / 1152`` is wrapped in
``\\textbf{...}``.

The exact-clean baseline is read from a row with
``exact_step1_columns=4``, ``r=4`` and ``scale=10^10``.  The current
resampled ablation CSVs contain this baseline, so the default invocation can
export the table directly.

Example:

    python figures/make_table3_tex.py \
      --input results/finite_precision/checkpoint_replay_trials.csv \
      --output results/finite_precision/table3.tex
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hssp_dfl import paths


DEFAULT_INPUTS = (
	paths.RESULTS / "finite_precision/checkpoint_replay_trials.csv",
)
DEFAULT_SCALES = (10**4, 10**6, 10**8, 10**10)
DEFAULT_R_VALUES = (1, 2, 4, 8)
DATASET_ORDER = ("sentiment140", "purchase", "cifar")
DATASET_LABELS = {
	"sentiment140": "Sentiment140",
	"purchase": "Purchase",
	"cifar": "CIFAR",
}


def _read_csv(paths: list[Path]) -> list[dict[str, str]]:
	rows: list[dict[str, str]] = []
	for path in paths:
		if not path.exists():
			raise FileNotFoundError(f"CSV file does not exist: {path}")
		with path.open(newline="", encoding="utf-8") as handle:
			rows.extend(csv.DictReader(handle))
	return rows


def _int(row: dict[str, str], field: str) -> int | None:
	value = row.get(field, "")
	return None if value in (None, "") else int(float(value))


def _float(row: dict[str, str], field: str) -> float | None:
	value = row.get(field, "")
	return None if value in (None, "") else float(value)


def _is_completed(row: dict[str, str], prefix: str) -> bool:
	return row.get(f"{prefix}_status") == "completed"


def _metric(row: dict[str, str], prefix: str) -> str:
	"""Return ``recall / candidates`` with the requested bolding rule."""
	recall = _float(row, f"{prefix}_recall")
	candidates = _int(row, f"{prefix}_n_candidates")
	if recall is None or candidates is None:
		return r"\text{--}"

	text = f"{recall:.2f} / {candidates}"
	if abs(recall - 1.0) < 1e-12 and candidates == 1152:
		return rf"\textbf{{{text}}}"
	return text


def _find_one(
	rows: list[dict[str, str]], *, dataset: str, scale: int,
	exact_r: int, baseline_r: int,
) -> dict[str, str]:
	matches = [
		row for row in rows
		if row.get("dataset") == dataset
		and _int(row, "scale") == scale
		and _int(row, "exact_step1_columns") == exact_r
		and _int(row, "r") == baseline_r
		and _is_completed(row, "clean")
	]
	if not matches:
		raise ValueError(
			"missing exact-clean baseline for "
			f"dataset={dataset}, scale={scale}, exact-r={exact_r}, "
			f"r={baseline_r}"
		)
	if len(matches) > 1:
		raise ValueError(
			"expected one baseline row, found "
			f"{len(matches)} for dataset={dataset}, scale={scale}, "
			f"exact-r={exact_r}, r={baseline_r}; use a single-trial CSV"
		)
	return matches[0]


def _find_noisy(
	rows: list[dict[str, str]], *, dataset: str, scale: int, r: int,
) -> dict[str, str] | None:
	matches = [
		row for row in rows
		if row.get("dataset") == dataset
		and _int(row, "scale") == scale
		and _int(row, "r") == r
		and _is_completed(row, "noisy")
	]
	if not matches:
		return None
	if len(matches) > 1:
		raise ValueError(
			"expected one noisy row, found "
			f"{len(matches)} for dataset={dataset}, scale={scale}, r={r}; "
			"use a single-trial CSV"
		)
	return matches[0]


def _latex_escape(value: str) -> str:
	return (
		value.replace("&", r"\&")
		.replace("%", r"\%")
		.replace("_", r"\_")
	)


def build_table(
	rows: list[dict[str, str]], baseline_rows: list[dict[str, str]],
	*, scales: tuple[int, ...] = DEFAULT_SCALES,
	r_values: tuple[int, ...] = DEFAULT_R_VALUES,
	baseline_scale: int = 10**10,
	baseline_exact_r: int = 4,
	baseline_r: int = 4,
) -> str:
	"""Build the complete booktabs table body and wrapper."""
	all_rows = rows + baseline_rows
	observed_datasets = {row["dataset"] for row in rows}
	datasets = [
		dataset for dataset in DATASET_ORDER if dataset in observed_datasets
	]
	datasets.extend(sorted(observed_datasets - set(datasets)))
	if not datasets:
		raise ValueError("no noisy rows found")

	lines = [
		r"\begin{table*}[t]",
		r"\centering",
		r"\label{tab:truncation-single-trial}",
		"\\begin{tabular}{ll" + "c" * (1 + len(r_values)) + "}",
		r"\toprule",
		"Dataset & Condition & Exact clean & "
		+ " & ".join(f"$r={r}$" for r in r_values)
		+ r" \\",
		r"\midrule",
	]

	for dataset_index, dataset in enumerate(datasets):
		baseline = _find_one(
			all_rows,
			dataset=dataset,
			scale=baseline_scale,
			exact_r=baseline_exact_r,
			baseline_r=baseline_r,
		)
		baseline_label = "Baseline"
		baseline_noisy_cells = " & ".join(
			[r"\text{--}"] * len(r_values)
		)
		lines.append(
			f"\\multirow{{{1 + len(scales)}}}{{*}}{{"
			f"{_latex_escape(DATASET_LABELS.get(dataset, dataset))}}} & "
			f"{baseline_label} & "
			f"{_metric(baseline, 'clean')} & "
			+ baseline_noisy_cells
			+ r" \\",
		)

		for scale in scales:
			cells = []
			for r in r_values:
				row = _find_noisy(
					rows,
					dataset=dataset,
					scale=scale,
					r=r,
				)
				cells.append(r"\text{--}" if row is None else _metric(row, "noisy"))
			lines.append(
				f"& $S=10^{{{len(str(scale)) - 1}}}$ & "
				+ " & ".join([r"\text{--}"] + cells)
				+ r" \\",
			)

		if dataset_index != len(datasets) - 1:
			lines.append(r"\addlinespace")

	lines.extend([
		r"\bottomrule",
		r"\end{tabular}",
		r"\caption{Robustness to finite-precision errors via noisy HSSP/HLCP results.}",
		r"\end{table*}",
	])
	return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input", dest="inputs", action="append", type=Path,
		help="CSV containing noisy rows; repeat for r=1,2,4 and r=8 outputs",
	)
	parser.add_argument(
		"--baseline-input", action="append", type=Path, default=[],
		help="CSV containing exact-clean baseline rows (repeat if needed)",
	)
	parser.add_argument(
		"--output", type=Path,
		default=paths.RESULTS / "finite_precision/table3.tex",
	)
	parser.add_argument("--baseline-scale", type=int, default=10**10)
	parser.add_argument("--baseline-exact-r", type=int, default=4)
	parser.add_argument(
		"--baseline-r", type=int, default=4,
		help="noisy-r row used to select the duplicated clean baseline (default: 4)",
	)
	parser.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_SCALES))
	parser.add_argument("--r-values", nargs="+", type=int, default=list(DEFAULT_R_VALUES))
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	input_paths = args.inputs or list(DEFAULT_INPUTS)
	rows = _read_csv(input_paths)
	baseline_rows = _read_csv(args.baseline_input) if args.baseline_input else []
	tex = build_table(
		rows,
		baseline_rows,
		scales=tuple(args.scales),
		r_values=tuple(args.r_values),
		baseline_scale=args.baseline_scale,
		baseline_exact_r=args.baseline_exact_r,
		baseline_r=args.baseline_r,
	)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(tex, encoding="utf-8")
	print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
	main()

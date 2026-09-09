"""Recall / candidate-count summary tables -- paper Tables 1 and 9.

Aggregates every per-trial CSV written by experiments/lattice_attack_batch.py
into the two LaTeX summary tables of Sections 7.3:

    Table 1  mHLCP (non-uniform Metropolis weights)
    Table 9  mHSSP (uniform weights)
"""

import csv
import math
import re
from pathlib import Path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hssp_dfl import paths



import argparse

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--input-dir",
                     default=str(paths.RESULTS / "lattice_attack_batch"),
                     help="directory of per-trial attack CSVs")
_parser.add_argument("--output",
                     default=str(paths.RESULTS / "figures" /
                                 "table1_and_table9_summary.tex"))
_ARGS = _parser.parse_args()
ROOT = paths.ROOT
TABLE_DIR = Path(_ARGS.input_dir)
OUT_TEX = Path(_ARGS.output)

CONFIG_RE = re.compile(r"NODE(\d+)_EDGE(\d+)_CORRUPT(\d+)")


def _safe_float(value):
	if value is None:
		return None
	text = str(value).strip()
	if text == "" or text.lower() == "nan":
		return None
	try:
		return float(text)
	except ValueError:
		return None


def _safe_int(value):
	if value is None:
		return None
	text = str(value).strip()
	if text == "" or text.lower() == "nan":
		return None
	try:
		return int(float(text))
	except ValueError:
		return None


def _parse_config(stem):
	match = CONFIG_RE.search(stem)
	if not match:
		return None
	nodes = int(match.group(1))
	edges = int(match.group(2))
	corrupt = int(match.group(3))
	return nodes, edges, corrupt


def _task_from_name(stem):
	text = stem.lower()
	if "hlcp" in text:
		return "HLCP"
	if "hssp" in text:
		return "HSSP"
	return "UNKNOWN"


def _read_rows(path):
	with path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		return list(reader)


def _mean(values):
	if not values:
		return None
	return sum(values) / len(values)


def _format_value(value, fmt):
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return "NA"
	return fmt.format(value)


def _summarize_file(path):
	rows = _read_rows(path)
	total = len(rows)
	if total == 0:
		return None

	config = _parse_config(path.stem)
	n_honest = None
	if rows and "n_honest" in rows[0]:
		n_honest = _safe_int(rows[0].get("n_honest"))
	if n_honest is None and config is not None:
		n_honest = config[0] - config[2]
	if n_honest is None:
		n_honest = max((_safe_int(r.get("true_weights_matched")) or 0) for r in rows)

	all_found_flags = []
	for r in rows:
		matched = _safe_int(r.get("true_weights_matched"))
		all_found_flags.append(matched is not None and matched == n_honest)

	all_found_count = sum(1 for v in all_found_flags if v)
	all_found_ratio = all_found_count / total if total else 0.0

	found_vecs = []
	for r, ok in zip(rows, all_found_flags):
		if ok:
			found_num = _safe_int(r.get("found_num_vectors"))
			if found_num is not None:
				found_vecs.append(found_num)

	step1_times = [_safe_float(r.get("step1_time")) for r in rows]
	step1_times = [v for v in step1_times if v is not None]
	step2_times = [_safe_float(r.get("step2_time")) for r in rows]
	step2_times = [v for v in step2_times if v is not None]

	label = path.stem
	if config is not None:
		nodes, edges, corrupt = config
		label = f"N{nodes}-E{edges}-C{corrupt}"

	return {
		"task": _task_from_name(path.stem),
		"label": label,
		"all_found_ratio": all_found_ratio,
		"avg_found_vecs": _mean(found_vecs),
		"avg_step1": _mean(step1_times),
		"avg_step2": _mean(step2_times),
		"total": total,
	}


def _build_table_lines(rows, task):
	lines = []
	lines.append(r"\begin{table*}[ht]")
	lines.append(r"\centering")
	lines.append(r"\begin{tabular}{l c c c c}")
	lines.append(r"\hline")
	lines.append(
		r"Config & All Weights Found (\%) & Found Vec. (avg) & Step1 Time (s) & Step2 Time (s)\\"
	)
	lines.append(r"\hline")

	for item in rows:
		ratio = _format_value(item["all_found_ratio"] * 100.0, "{:.1f}")
		avg_vecs = _format_value(item["avg_found_vecs"], "{:.1f}")
		avg_step1 = _format_value(item["avg_step1"], "{:.6f}")
		avg_step2 = _format_value(item["avg_step2"], "{:.6f}")

		lines.append(
			f"{item['label']} & {ratio}\\% & {avg_vecs} & {avg_step1} & {avg_step2} \\\\"  # noqa: E501
		)

	lines.append(r"\hline")
	lines.append(r"\end{tabular}")
	lines.append(
		f"\\caption{{Recall and candidate-set size of the Nguyen-Stern {task} "
		f"attack, averaged over 100 random core sub-graphs per configuration.}}"
	)
	lines.append(f"\\label{{tab.large_{task.lower()}_summary}}")
	lines.append(r"\end{table*}")
	return lines


def build_table():
	rows = []
	# Only the per-trial CSVs of the undirected runs: the directory also holds
	# the *_summary.csv companions and the directed push-sum runs (Table 5).
	for path in sorted(TABLE_DIR.glob("undirected_*.csv")):
		if path.stem.endswith("_summary"):
			continue
		summary = _summarize_file(path)
		if summary is not None:
			rows.append(summary)

	groups = {}
	for item in rows:
		groups.setdefault(item["task"], []).append(item)

	ordered_tasks = ["HLCP", "HSSP", "UNKNOWN"]
	lines = []
	for task in ordered_tasks:
		task_rows = groups.get(task, [])
		if not task_rows:
			continue
		lines.extend(_build_table_lines(task_rows, task))

	text = "\n".join(lines)
	OUT_TEX.write_text(text)
	print(text)


if __name__ == "__main__":
	if not TABLE_DIR.exists():
		raise FileNotFoundError(f"Missing directory: {TABLE_DIR}")
	build_table()

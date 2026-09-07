#!/usr/bin/env python3
"""
Append BERTScore (F1) to sentiment140_vec2text_metrics.csv.

Reads the existing vec2text metrics CSV and writes a new CSV
with an extra final column: bertscore_f1.

Usage:
	python addBertscore.py

Requires:
	- bert-score
	- torch
"""

import csv
from pathlib import Path

from bert_score import score


import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hssp_dfl import paths

_SENTIMENT_DIR = paths.RESULTS / "real_data" / "sentiment140"
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--input", default=str(_SENTIMENT_DIR / "sentiment140_vec2text_metrics.csv"))
_parser.add_argument(
    "--output",
    default=str(_SENTIMENT_DIR / "sentiment140_vec2text_metrics_with_bertscore.csv"))
_ARGS = _parser.parse_args()

INPUT_CSV_PATH = Path(_ARGS.input)
OUTPUT_CSV_PATH = Path(_ARGS.output)

LANG = "en"
MODEL_TYPE = None  # e.g., "bert-base-uncased"; None uses bert-score default for LANG
BATCH_SIZE = 32
RESCALE_WITH_BASELINE = True


def _read_rows(path):
	with path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)
		return reader.fieldnames or [], rows


def _write_rows(path, fieldnames, rows):
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main():
	if not INPUT_CSV_PATH.exists():
		raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV_PATH}")

	fieldnames, rows = _read_rows(INPUT_CSV_PATH)
	if not rows:
		print("No rows found. Exiting.")
		return

	indices = []
	candidates = []
	references = []

	for idx, row in enumerate(rows):
		cand = (row.get("recovered_text") or "").strip()
		ref = (row.get("ground_truth_text") or "").strip()
		if cand and ref and row.get("row_type") == "node":
			indices.append(idx)
			candidates.append(cand)
			references.append(ref)

	if indices:
		print(f"Computing BERTScore for {len(indices)} rows...")
		P, R, F1 = score(
			candidates,
			references,
			lang=LANG,
			model_type=MODEL_TYPE,
			batch_size=BATCH_SIZE,
			rescale_with_baseline=RESCALE_WITH_BASELINE,
		)
		f1_vals = [float(v) for v in F1]
		for idx, val in zip(indices, f1_vals):
			rows[idx]["bertscore_f1"] = f"{val:.6f}"
	else:
		print("No eligible rows found for BERTScore.")

	if "bertscore_f1" not in fieldnames:
		fieldnames = fieldnames + ["bertscore_f1"]

	_write_rows(OUTPUT_CSV_PATH, fieldnames, rows)
	print(f"Saved with BERTScore -> {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
	main()

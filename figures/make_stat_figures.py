"""Statistical reconstruction quality -- paper Figures 4, 9 and 10.

Turns the per-candidate metric CSVs written by the three real-dataset attacks
into the two-panel figures of Section 7.4:

    Figure 4   CIFAR-10       MSE of recovered gradients | SSIM and PSNR
    Figure 9   Purchase-100   MSE of recovered gradients | feature MSE
    Figure 10  Sentiment140   MSE of recovered embeddings | cosine and ROUGE-L

Inputs (produced by experiments/attack_*.py and experiments/vec2text_eval.py)
    results/real_data/cifar30/cifar_case_metrics30.csv
    results/real_data/purchase/purchase_case_metrics.csv
    results/real_data/sentiment140/sentiment140_vec2text_metrics.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hssp_dfl import paths

_REAL = paths.RESULTS / "real_data"
_parser = argparse.ArgumentParser(
    description="Build Figures 4, 9 and 10 from the saved per-candidate metrics.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument("--dataset", choices=("cifar", "purchase", "sentiment140", "all"),
                     default="all", help="which figure(s) to build")
_parser.add_argument("--cifar-csv",
                     default=str(_REAL / "cifar30" / "cifar_case_metrics30.csv"))
_parser.add_argument("--purchase-csv",
                     default=str(_REAL / "purchase" / "purchase_case_metrics.csv"))
_parser.add_argument("--sentiment-csv",
                     default=str(_REAL / "sentiment140" /
                                 "sentiment140_vec2text_metrics.csv"))
_parser.add_argument("--true-solution-rank", type=int, default=24,
                     help="candidate rank of the true solution in Case 1 "
                          "(the paper's runs put it at 24)")
_parser.add_argument("--output-dir", default=str(paths.RESULTS / "figures"))
_ARGS = _parser.parse_args()

CIFAR_CSV = Path(_ARGS.cifar_csv)
PURCHASE_CSV = Path(_ARGS.purchase_csv)
SENTIMENT_CSV = Path(_ARGS.sentiment_csv)
FIGURE_DIR = Path(_ARGS.output_dir)


def _true_solution_row(rows, mse_field="matched_mse_mean"):
	"""Locate the Case 1 candidate that is the ground-truth weight matrix.

	Preference order: the rank named on the command line, otherwise the Case 1
	candidate whose recovered gradients are closest to the true ones (the true
	solution reconstructs them up to machine epsilon).
	"""
	case1 = [r for r in rows
	         if r.get("case") == "case1_exact_pattern"
	         and r.get("row_type") == "sample_mean"]
	if not case1:
		raise SystemExit("no Case 1 sample_mean rows in the metrics CSV")
	wanted = str(_ARGS.true_solution_rank)
	exact = [r for r in case1 if r.get("sample_rank") == wanted]
	if exact:
		return exact[0]
	return min(case1, key=lambda r: float(r[mse_field]))


def _true_solution_nodes(rows, case="case1_exact_pattern"):
	"""Per-node rows of the true solution (used by the Sentiment140 panel).

	This CSV has no per-sample mean rows, so the candidate is picked directly
	from the ``node`` rows: the requested rank when present, otherwise the
	candidate with the smallest mean per-node MSE.
	"""
	by_rank = {}
	for row in rows:
		if row.get("case") != case or row.get("row_type") != "node":
			continue
		by_rank.setdefault(row.get("sample_rank"), []).append(row)
	if not by_rank:
		raise SystemExit("no Case 1 node rows in the metrics CSV")
	wanted = str(_ARGS.true_solution_rank)
	if wanted in by_rank:
		return by_rank[wanted]
	best = min(by_rank.values(),
	           key=lambda group: sum(float(r["matched_mse_node"]) for r in group))
	return best



def main() -> None:
	root = Path(__file__).resolve().parent
	csv_path = CIFAR_CSV

	with csv_path.open("r", newline="") as f:
		rows = list(csv.DictReader(f))

	# File line 121 corresponds to data row index 119 (header is line 1).
	gt_row = _true_solution_row(rows)

	case_summary = {
		r["case"]: r
		for r in rows
		if r.get("row_type") == "case_summary"
	}

	case_order = [
		"case1_exact_pattern",
		"case2_zero_count",
		"case3_no_structure",
	]
	labels = ["True solution", "Case 1", "Case 2", "Case 3"]
	x = list(range(len(labels)))

	mse_means = [
		float(gt_row["matched_mse_mean"]),
		*[float(case_summary[c]["case_mse_mean"]) for c in case_order],
	]
	mse_err = [
		0.0,
		*[
			math.sqrt(float(case_summary[c]["case_mse_var"]))
			for c in case_order
		],
	]

	ssim_means = [
		float(gt_row["ssim"]),
		*[float(case_summary[c]["case_ssim_mean"]) for c in case_order],
	]
	ssim_err = [
		0.0,
		*[
			math.sqrt(float(case_summary[c]["case_ssim_var"]))
			for c in case_order
		],
	]

	psnr_means = [
		float(gt_row["psnr"]),
		*[float(case_summary[c]["case_psnr_mean"]) for c in case_order],
	]
	psnr_err = [
		0.0,
		*[
			math.sqrt(float(case_summary[c]["case_psnr_var"]))
			for c in case_order
		],
	]

	fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True)

	# Top subplot: MSE line with variance-derived error bars for non-GT entries.
	axes[0].errorbar(
		x,
		mse_means,
		# yerr=mse_err,
		fmt="o-",
		capsize=5,
		linewidth=1.8,
		color="#4C72B0",
		label="MSE",
	)
	axes[0].set_ylabel("MSE")
	axes[0].legend(loc="best")
	axes[0].set_yscale("log")
	# axes[0].set_title("CIFAR Cases: MSE, SSIM and PSNR")
	axes[0].grid(axis="y", linestyle="--", alpha=0.35)

	# Bottom subplot: SSIM (left y-axis) and PSNR (right y-axis).
	ax_ssim = axes[1]
	ax_psnr = ax_ssim.twinx()

	ax_ssim.errorbar(
		x,
		ssim_means,
		yerr=ssim_err,
		fmt="o-",
		capsize=4,
		linewidth=1.8,
		label="SSIM",
		color="#1F77B4",
	)
	ax_psnr.errorbar(
		x,
		psnr_means,
		yerr=psnr_err,
		fmt="s-",
		capsize=4,
		linewidth=1.8,
		label="PSNR",
		color="#D62728",
	)

	ax_ssim.set_ylabel("SSIM")
	ax_psnr.set_ylabel("PSNR")
	ax_ssim.set_xticks(x, labels)
	ax_ssim.grid(axis="y", linestyle="--", alpha=0.35)

	lines1, labels1 = ax_ssim.get_legend_handles_labels()
	lines2, labels2 = ax_psnr.get_legend_handles_labels()
	ax_ssim.legend(lines1 + lines2, labels1 + labels2, loc="best")

	# Add (a) and (b) labels at the bottom left of each subplot
	axes[0].text(0.5, -0.12, '(a)', transform=axes[0].transAxes, fontsize=12, va='top', ha='center')
	axes[1].text(0.5, -0.22, '(b)', transform=axes[1].transAxes, fontsize=12, va='top', ha='center')

	plt.tight_layout()
	# Save figure
	out_path = FIGURE_DIR / 'figure4_cifar.png'
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=300, bbox_inches='tight')
	plt.close()

import math
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main_purchase() -> None:
	root = Path(__file__).resolve().parent
	csv_path = PURCHASE_CSV

	with csv_path.open("r", newline="") as f:
		rows = list(csv.DictReader(f))

	# File line 121 corresponds to data row index 119 (header is line 1).
	gt_row = _true_solution_row(rows)

	case_summary = {
		r["case"]: r
		for r in rows
		if r.get("row_type") == "case_summary"
	}

	case_order = [
		"case1_exact_pattern",
		"case2_zero_count",
		"case3_no_structure",
	]
	labels = ["True solution", "Case 1", "Case 2", "Case 3"]
	x = list(range(len(labels)))

	recover_mse_means = [
		float(gt_row["matched_mse_mean"]),
		*[float(case_summary[c]["case_recover_mse_mean"]) for c in case_order],
	]
	recover_mse_err = [
		0.0,
		*[
			math.sqrt(float(case_summary[c]["case_recover_mse_var"]))
			for c in case_order
		],
	]

	feat_bin_mse_means = [
		float(gt_row["feat_bin_mse"]),
		*[float(case_summary[c]["case_feat_bin_mse_mean"]) for c in case_order],
	]
	feat_bin_mse_err = [
		0.0,
		*[
			math.sqrt(float(case_summary[c]["case_feat_bin_mse_var"]))
			for c in case_order
		],
	]

	fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

	# Top subplot: MSE line with variance-derived error bars for non-GT entries.
	axes[0].errorbar(
		x,
		recover_mse_means,
		# yerr=recover_mse_err,
		fmt="o-",
		capsize=5,
		linewidth=1.8,
		color="#4C72B0",
		label="MSE",
	)
	axes[0].set_ylabel("MSE")
	axes[0].legend(loc="best")
	axes[0].set_yscale("log")
	# axes[0].set_title("CIFAR Cases: MSE, SSIM and PSNR")
	axes[0].grid(axis="y", linestyle="--", alpha=0.35)

	# Bottom subplot: Feat-bin MSE.
	ax_mse = axes[1]
	ax_mse.errorbar(
		x,
		feat_bin_mse_means,
		yerr=feat_bin_mse_err,
		fmt="s-",
		capsize=4,
		linewidth=1.8,
		label="MSE",
		color="#D62728",
	)

	ax_mse.set_ylabel("MSE")
	ax_mse.set_xticks(x, labels)
	ax_mse.grid(axis="y", linestyle="--", alpha=0.35)
	ax_mse.legend(loc="best")

	# Add (a) and (b) labels at the bottom left of each subplot
	axes[0].text(0.5, -0.12, '(a)', transform=axes[0].transAxes, fontsize=12, va='top', ha='center')
	axes[1].text(0.5, -0.22, '(b)', transform=axes[1].transAxes, fontsize=12, va='top', ha='center')

	plt.tight_layout()
	# Save figure
	out_path = FIGURE_DIR / 'figure9_purchase.png'
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=300, bbox_inches='tight')
	plt.close()

def main_sentiment140() -> None:
	root = Path(__file__).resolve().parent
	csv_path = SENTIMENT_CSV

	with csv_path.open("r", newline="") as f:
		rows = list(csv.DictReader(f))

	# Compute GT baseline from rows 92-95 (indices 91:95).
	gt_rows = _true_solution_nodes(rows)
	gt_mse_vals = [float(r["matched_mse_node"]) for r in gt_rows]
	gt_cosine_vals = [float(r["cosine_similarity"]) for r in gt_rows]
	gt_rouge_vals = [float(r["rouge_l"]) for r in gt_rows]

	case_summary = {
		r["case"]: r
		for r in rows
		if r.get("row_type") == "case_summary"
	}

	case_order = [
		"case1_exact_pattern",
		"case2_zero_count",
		"case3_no_structure",
	]
	labels = ["True solution", "Case 1", "Case 2", "Case 3"]
	x = list(range(len(labels)))

	mse_means = [
		np.mean(gt_mse_vals),
		*[float(case_summary[c]["case_mse_mean"]) for c in case_order],
	]
	mse_err = [
		np.std(gt_mse_vals),
		*[
			math.sqrt(float(case_summary[c]["case_mse_var"]))
			for c in case_order
		],
	]

	cosine_means = [
		np.mean(gt_cosine_vals),
		*[float(case_summary[c]["case_cosine_similarity_mean"]) for c in case_order],
	]
	cosine_err = [
		np.std(gt_cosine_vals),
		*[
			math.sqrt(float(case_summary[c]["case_cosine_similarity_var"]))
			for c in case_order
		],
	]

	rouge_means = [
		np.mean(gt_rouge_vals),
		*[float(case_summary[c]["case_rouge_l_mean"]) for c in case_order],
	]
	rouge_err = [
		np.std(gt_rouge_vals),
		*[
			math.sqrt(float(case_summary[c]["case_rouge_l_var"]))
			for c in case_order
		],
	]

	fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

	# Top subplot: MSE.
	axes[0].errorbar(
		x,
		mse_means,
		# yerr=mse_err,
		fmt="o-",
		capsize=5,
		linewidth=1.8,
		color="#4C72B0",
		label="MSE",
	)
	axes[0].set_ylabel("MSE")
	axes[0].legend(loc="best")
	axes[0].set_yscale("log")
	axes[0].grid(axis="y", linestyle="--", alpha=0.35)

	# Bottom subplot: Cosine Similarity (left) + ROUGE-L (right).
	ax_cos = axes[1]
	ax_rouge = ax_cos.twinx()

	ax_cos.errorbar(
		x,
		cosine_means,
		yerr=cosine_err,
		fmt="o-",
		capsize=4,
		linewidth=1.8,
		label="Cosine similarity",
		color="#1F77B4",
	)
	ax_rouge.errorbar(
		x,
		rouge_means,
		yerr=rouge_err,
		fmt="s-",
		capsize=4,
		linewidth=1.8,
		label="ROUGE-L",
		color="#D62728",
	)

	ax_cos.set_ylabel("Cosine similarity")
	ax_rouge.set_ylabel("ROUGE-L")
	ax_cos.set_xticks(x, labels)
	ax_cos.grid(axis="y", linestyle="--", alpha=0.35)

	lines1, labels1 = ax_cos.get_legend_handles_labels()
	lines2, labels2 = ax_rouge.get_legend_handles_labels()
	ax_cos.legend(lines1 + lines2, labels1 + labels2, loc="best")

	# Add (a) and (b) labels under each subplot.
	axes[0].text(0.5, -0.12, '(a)', transform=axes[0].transAxes, fontsize=12, va='top', ha='center')
	axes[1].text(0.5, -0.22, '(b)', transform=axes[1].transAxes, fontsize=12, va='top', ha='center')

	plt.tight_layout()
	out_path = FIGURE_DIR / 'figure10_sentiment140.png'
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=300, bbox_inches='tight')
	plt.close()

if __name__ == "__main__":
	FIGURE_DIR.mkdir(parents=True, exist_ok=True)
	if _ARGS.dataset in ("cifar", "all") and CIFAR_CSV.exists():
		main()
	if _ARGS.dataset in ("purchase", "all") and PURCHASE_CSV.exists():
		main_purchase()
	if _ARGS.dataset in ("sentiment140", "all") and SENTIMENT_CSV.exists():
		main_sentiment140()
	for label, path in (("cifar", CIFAR_CSV), ("purchase", PURCHASE_CSV),
	                    ("sentiment140", SENTIMENT_CSV)):
		if _ARGS.dataset in (label, "all") and not path.exists():
			print(f"skipped {label}: missing {path}")

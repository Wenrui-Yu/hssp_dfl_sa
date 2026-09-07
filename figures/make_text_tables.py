"""Reconstructed-text tables -- paper Tables 12, 13 and 14.

Renders the per-candidate Sentiment140 reconstructions (Cases 1-3) as three
LaTeX longtables.  Table 2 of the main paper is a hand-picked excerpt of the
Case 1 table.

Input : results/real_data/sentiment140/sentiment140_vec2text_metrics_with_bertscore.csv
Output: results/figures/table{12,13,14}_sentiment140_case{1,2,3}.tex
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hssp_dfl import paths

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--input",
    default=str(paths.RESULTS / "real_data" / "sentiment140" /
                "sentiment140_vec2text_metrics_with_bertscore.csv"),
)
_parser.add_argument("--output-dir", default=str(paths.RESULTS / "figures"))
_ARGS = _parser.parse_args()

INPUT_CSV = _ARGS.input
OUTPUT_DIR = Path(_ARGS.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_NUMBER = {
    "case1_exact_pattern": 12,
    "case2_zero_count": 13,
    "case3_no_structure": 14,
}

import csv




def is_missing(value):
    return value is None or str(value).strip() == ""


def latex_escape(text):
    if is_missing(text):
        return "-"
    s = str(text)
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def fmt_metric(v):
    if is_missing(v):
        return "-"
    try:
        return f"{float(v):.3f}"
    except ValueError:
        return "-"


def parse_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def write_case_table(case_rows, case_name):
    number = TABLE_NUMBER.get(case_name, 0)
    out_file = str(OUTPUT_DIR /
                   f"table{number}_sentiment140_{case_name}.tex")

    # Keep only per-node rows and sort by sample rank then node.
    node_rows = []
    for row in case_rows:
        if row.get("row_type") != "node":
            continue
        sample_rank = parse_int(row.get("sample_rank"))
        node = parse_int(row.get("node"))
        if sample_rank is None or node is None:
            continue
        normalized = dict(row)
        normalized["sample_rank"] = sample_rank
        normalized["node"] = node
        node_rows.append(normalized)

    node_rows.sort(key=lambda x: (x["sample_rank"], x["node"]))

    # Ground-truth row block: one entry per node.
    gt_map = {}
    for row in node_rows:
        node = row["node"]
        if node not in gt_map and not is_missing(row.get("ground_truth_text")):
            gt_map[node] = row.get("ground_truth_text")
    gt_rows = [(node, gt_map[node]) for node in sorted(gt_map.keys())]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\\begin{longtable}{cccp{8cm}cc}\n")
        f.write("\\hline\n")
        f.write(" & Node & Cosine Similarity & Recovered Text & Rouge-L & BERTScore \\\\\n")
        f.write("\\hline\n")

        if gt_rows:
            n_gt = len(gt_rows)
            for i, (node, gt_text) in enumerate(gt_rows):
                first_col = f"\\multirow{{{n_gt}}}{{*}}{{Ground Truth}}" if i == 0 else ""
                f.write(
                    f"{first_col} & {node} & - & {latex_escape(gt_text)} & - & - \\\\\n"
                )
            f.write("\\hline\n")

        by_rank = {}
        for row in node_rows:
            by_rank.setdefault(row["sample_rank"], []).append(row)

        for rank in sorted(by_rank.keys()):
            rows = by_rank[rank]
            n_rows = len(rows)
            for i, row in enumerate(rows):
                rank_cell = f"\\multirow{{{n_rows}}}{{*}}{{{rank}}}" if i == 0 else ""
                f.write(
                    f"{rank_cell} & {row['node']} & {fmt_metric(row.get('cosine_similarity'))} "
                    f"& {latex_escape(row.get('recovered_text'))} & {fmt_metric(row.get('rouge_l'))} "
                    f"& {fmt_metric(row.get('bertscore_f1'))} \\\\\n"
                )
            f.write("\\hline\n")

        f.write("\\end{longtable}\n")

    return out_file


def main():
    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    case_map = {}
    for row in rows:
        case_name = row.get("case")
        if is_missing(case_name):
            continue
        case_map.setdefault(case_name, []).append(row)

    outputs = []
    for case_name in sorted(case_map.keys()):
        outputs.append(write_case_table(case_map[case_name], case_name))

    print("LaTeX tables written:")
    for out in outputs:
        print(f"- {out}")


if __name__ == "__main__":
    main()
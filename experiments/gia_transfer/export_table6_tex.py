#!/usr/bin/env python3
"""Convert aggregated GIA CSV data into a booktabs three-line LaTeX table.

The input may be either:

* the wide ``paper_table.csv`` produced by
  ``summarize_gia_true_vs_recovered.py``; or
* the long ``batch_summary_long.csv`` produced by the same report.

Each table cell is formatted as ``SSIM / LPIPS-Alex / PSNR``.  The output uses
``booktabs`` rules (``\\toprule``, ``\\midrule``, ``\\bottomrule``) and no
vertical lines.

Example::

    sage -python experiments/export_gia_three_line_tex.py \
      --input results/gia_true_vs_recovered_paired/report_tuned_stg/paper_table.csv \
      --output results/gia_true_vs_recovered_paired/report_tuned_stg/paper_table_from_raw.tex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


DEFAULT_ATTACKS = (
    "deepleakage",
    "invertinggradients",
    "seethroughgradients",
)
ATTACK_LABELS = {
    "deepleakage": "Deep Leakage",
    "invertinggradients": "Inverting Gradients",
    "seethroughgradients": "See Through Gradients",
}
SOURCES = ("true_update", "recovered_solution24")
SOURCE_LABELS = {
    "true_update": "True",
    "recovered_solution24": "Recovered (24)",
}
METRICS = ("ssim", "lpips", "psnr")


def parse_attacks(value: str) -> tuple[str, ...]:
    attacks = tuple(item.strip() for item in value.split(",") if item.strip())
    if not attacks:
        raise argparse.ArgumentTypeError("attack list cannot be empty")
    unknown = sorted(set(attacks) - set(ATTACK_LABELS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown attacks: {unknown}")
    if len(set(attacks)) != len(attacks):
        raise argparse.ArgumentTypeError("attack list contains duplicates")
    return attacks


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input CSV is empty: {path}")
    return rows


def _as_float(value: str, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value in {field}: {value!r}") from exc


def _as_batch_size(value: str) -> int:
    try:
        batch_size = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid batch size: {value!r}") from exc
    if batch_size <= 0:
        raise ValueError(f"batch size must be positive: {batch_size}")
    return batch_size


def _pivot_long_rows(
    rows: Iterable[dict[str, str]], attacks: tuple[str, ...]
) -> list[dict[str, str]]:
    """Pivot ``batch_summary_long.csv`` into the wide schema used by the table."""
    output: dict[int, dict[str, str]] = {}
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        batch_size = _as_batch_size(row.get("batch_size", ""))
        attack = row.get("attack", "")
        source = row.get("source", "")
        if attack not in attacks:
            continue
        if source not in SOURCES:
            raise ValueError(f"unknown source in long CSV: {source!r}")
        key = (batch_size, attack, source)
        if key in seen:
            raise ValueError(f"duplicate long-CSV row: {key}")
        seen.add(key)
        target = output.setdefault(batch_size, {"batch_size": str(batch_size)})
        for metric in METRICS:
            field = f"mean_{metric}"
            if field not in row or row[field] == "":
                raise ValueError(f"missing {field} for long-CSV row: {key}")
            target[f"{attack}_{source}_{metric}"] = row[field]

    expected = {
        (batch_size, attack, source)
        for batch_size in output
        for attack in attacks
        for source in SOURCES
    }
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"long CSV is missing table rows: {missing}")
    return [output[batch_size] for batch_size in sorted(output)]


def _wide_rows(
    rows: list[dict[str, str]], attacks: tuple[str, ...]
) -> list[dict[str, str]]:
    """Validate and normalize the wide ``paper_table.csv`` schema."""
    normalized = []
    seen_batches: set[int] = set()
    for row in rows:
        batch_size = _as_batch_size(row.get("batch_size", ""))
        if batch_size in seen_batches:
            raise ValueError(f"duplicate wide-CSV row for batch size {batch_size}")
        seen_batches.add(batch_size)
        for attack in attacks:
            for source in SOURCES:
                for metric in METRICS:
                    field = f"{attack}_{source}_{metric}"
                    if field not in row or row[field] == "":
                        raise ValueError(f"missing wide-CSV field: {field}")
                    _as_float(row[field], field=field)
        normalized.append({**row, "batch_size": str(batch_size)})
    return sorted(normalized, key=lambda row: int(row["batch_size"]))


def load_table_rows(path: Path, attacks: tuple[str, ...]) -> list[dict[str, str]]:
    rows = read_csv(path)
    if {"attack", "source", "mean_ssim", "mean_lpips", "mean_psnr"}.issubset(rows[0]):
        return _pivot_long_rows(rows, attacks)
    return _wide_rows(rows, attacks)


def _cell(row: dict[str, str], attack: str, source: str) -> str:
    values = [
        _as_float(row[f"{attack}_{source}_{metric}"], field=f"{attack}_{source}_{metric}")
        for metric in METRICS
    ]
    return f"{values[0]:.4f} / {values[1]:.4f} / {values[2]:.2f}"


def build_table(
    rows: list[dict[str, str]],
    attacks: tuple[str, ...] = DEFAULT_ATTACKS,
    *,
    caption: str = (
        "Paired downstream GIA performance on the exact gradient "
        r"$\boldsymbol{g}_{i,\mathrm{true}}^{(0)}$ and the gradient "
        r"$\widehat{\boldsymbol{g}}_i^{(0)}$ derived from recovered Case~1 "
        "solution 24. Each entry is mean SSIM / LPIPS-Alex / PSNR (dB) over "
        "the selected honest nodes and all aligned images."
    ),
    label: str = "tab:gia_true_vs_recovered",
) -> str:
    if not rows:
        raise ValueError("no table rows")
    columns = "r" + "cc" * len(attacks)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        "& "
        + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{ATTACK_LABELS[attack]}}}" for attack in attacks
        )
        + " \\\\",
        "".join(
            rf"\cmidrule(lr){{{2 + 2 * index}-{3 + 2 * index}}}"
            for index in range(len(attacks))
        ),
        r"$|\mathcal{B}_i^{(0)}|$ & "
        + " & ".join(
            SOURCE_LABELS[source] for _attack in attacks for source in SOURCES
        )
        + " \\\\",
        r"\midrule",
    ]
    for row in rows:
        cells = [row["batch_size"]]
        cells.extend(
            _cell(row, attack, source)
            for attack in attacks
            for source in SOURCES
        )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/gia_true_vs_recovered_paired/report_tuned_stg/paper_table.csv"
        ),
        help="wide paper_table.csv or long batch_summary_long.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/gia_true_vs_recovered_paired/report_tuned_stg/paper_table_from_raw.tex"
        ),
    )
    parser.add_argument("--attacks", type=parse_attacks, default=DEFAULT_ATTACKS)
    parser.add_argument(
        "--caption",
        default=(
            "Paired downstream GIA performance on the exact gradient "
            r"$\boldsymbol{g}_{i,\mathrm{true}}^{(0)}$ and the gradient "
            r"$\widehat{\boldsymbol{g}}_i^{(0)}$ derived from recovered Case~1 "
            "solution 24. Each entry is mean SSIM / LPIPS-Alex / PSNR (dB) over "
            "the selected honest nodes and all aligned images."
        ),
    )
    parser.add_argument("--label", default="tab:gia_true_vs_recovered")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_table_rows(args.input, args.attacks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_table(rows, args.attacks, caption=args.caption, label=args.label),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare freshly produced results against the numbers published in the paper.

    python scripts/check_reference.py              # check whatever exists
    python scripts/check_reference.py table7       # check one result

Timing columns are ignored: they are wall-clock measurements and are expected
to differ between machines.  Everything else -- recall, candidate counts,
Cases 1-3 solution counts, per-candidate metrics -- must match exactly for the
deterministic experiments, and within a tolerance for the ones the paper itself
averages over random trials.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hssp_dfl import paths  # noqa: E402  (stdlib only; honours HSSP_RESULTS)

REFERENCE = ROOT / "reference" / "tables"
RESULTS = paths.RESULTS

# Wall-clock and machine-dependent columns never have to agree.
IGNORED = {"step1_time", "step2_time", "noisy_tt_step1", "graph_attempts",
           "avg_step1_time", "avg_step2_time",
           "t0_path", "t0_5_path", "t1_path"}

# The reference CSVs keep the column names of the scripts that produced the
# paper's numbers; the reorganised producers renamed a few.  reference -> produced.
ALIASES = {
    "case1_solutions_before": "case1_num_solutions",
    "case1_solutions_after": "case1_validated_by_mass",
}

# Columns produced by a stochastic downstream optimiser rather than by the
# lattice attack.  The DLG image inversion in the CIFAR pipeline re-randomises
# its dummy input every run, so per-image SSIM/PSNR are not reproducible even
# though the recovered gradients they are computed from are exact.  Their
# aggregate statistics are checked with a loose tolerance instead.
LOOSE = {
    "ssim": None, "psnr": None,          # per-image: not compared at all
    "case_ssim_mean": 0.05, "case_ssim_var": 0.25,
    "case_psnr_mean": 0.05, "case_psnr_var": 0.25,
}

# target -> (reference csv, produced csv, key columns, tolerance)
CHECKS = {
    "table7": (
        REFERENCE / "table7_mhlcp_n10_e20_eta0.6.csv",
        RESULTS / "lattice_attack_batch" / "undirected_mhlcp_NODE10_EDGE20_CORRUPT6.csv",
        ["topology"], 0.0,
    ),
    "table8": (
        REFERENCE / "table8_mhlcp_n10_e20_eta0.7.csv",
        RESULTS / "lattice_attack_batch" / "undirected_mhlcp_NODE10_EDGE20_CORRUPT7.csv",
        ["topology"], 0.0,
    ),
    "table10": (
        REFERENCE / "table10_mhssp_n10_e20_eta0.6.csv",
        RESULTS / "lattice_attack_batch" / "undirected_mhssp_NODE10_EDGE20_CORRUPT6.csv",
        ["topology"], 0.0,
    ),
    "table11": (
        REFERENCE / "table11_mhssp_n10_e20_eta0.7.csv",
        RESULTS / "lattice_attack_batch" / "undirected_mhssp_NODE10_EDGE20_CORRUPT7.csv",
        ["topology"], 0.0,
    ),
    "table5": (
        REFERENCE / "table5_pushsum_mhlcp_n10_e30_eta0.7.csv",
        RESULTS / "lattice_attack_batch" / "directed_pushsum_mhlcp_NODE10_EDGE30_CORRUPT7.csv",
        ["topology"], 0.0,
    ),
    "table3": (
        REFERENCE / "table3_finite_precision_trials_r1-4.csv",
        RESULTS / "finite_precision" / "checkpoint_replay_trials.csv",
        ["dataset", "scale", "quantizer", "r"], 0.0,
    ),
    "table3-r8": (
        REFERENCE / "table3_finite_precision_trials_r8.csv",
        RESULTS / "finite_precision" / "checkpoint_replay_trials.csv",
        ["dataset", "scale", "quantizer", "r"], 0.0,
    ),
    "figure4": (
        REFERENCE / "figure4_cifar_case_metrics30.csv",
        RESULTS / "real_data" / "cifar30" / "cifar_case_metrics30.csv",
        ["case", "sample_rank", "node", "row_type"], 1e-6,
    ),
    "figure9": (
        REFERENCE / "figure9_purchase_case_metrics.csv",
        RESULTS / "real_data" / "purchase" / "purchase_case_metrics.csv",
        ["case", "sample_rank", "node", "row_type"], 1e-6,
    ),
    "figure5": (
        REFERENCE / "figure5_dp_exact_stats.csv",
        RESULTS / "dp_defense" / "dp_exact_stats.csv",
        ["dp_mode", "dp_epsilon"], 1e-6,
    ),
    "figure11": (
        REFERENCE / "figure11_dp_noisy_stats.csv",
        RESULTS / "dp_defense" / "dp_noisy_stats.csv",
        ["dp_mode", "dp_epsilon"], 1e-6,
    ),
    "figure11-extended": (
        REFERENCE / "figure11_dp_noisy_extended_stats.csv",
        RESULTS / "dp_defense" / "dp_noisy_extended_stats.csv",
        ["dp_mode", "dp_epsilon"], 1e-6,
    ),
    # The embedding-recovery stage: no OpenAI access needed.
    "figure10-embeddings": (
        REFERENCE / "figure10_sentiment140_case_metrics.csv",
        RESULTS / "real_data" / "sentiment140" / "sentiment140_case_metrics.csv",
        ["case", "sample_rank", "node", "row_type"], 1e-6,
    ),
    # The vec2text stage: only produced when OPENAI_API_KEY is set.
    "figure10-text": (
        REFERENCE / "figure10_sentiment140_vec2text_metrics.csv",
        RESULTS / "real_data" / "sentiment140" / "sentiment140_vec2text_metrics.csv",
        ["case", "sample_rank", "node", "row_type"], 1e-6,
    ),
}


def _read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def _index(rows, keys):
    """Index rows by key; a repeated key would otherwise be silently overwritten."""
    index, duplicates = {}, []
    for row in rows:
        k = _key(row, keys)
        if k in index:
            duplicates.append(k)
        index[k] = row
    return index, duplicates


def _key(row, columns):
    return tuple(row.get(name, "") for name in columns)


def _same(reference, produced, tolerance):
    if reference == produced:
        return True
    try:
        a, b = float(reference), float(produced)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b  # inf equals only inf; a finite value is never "within tolerance" of it
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= tolerance * scale


def check(name):
    ref_path, out_path, keys, tolerance = CHECKS[name]
    if not ref_path.exists():
        return "skip", f"no reference file ({ref_path.name})"
    if not out_path.exists():
        return "skip", f"not produced yet ({_rel(out_path)})"

    reference, ref_dupes = _index(_read(ref_path), keys)
    produced, out_dupes = _index(_read(out_path), keys)
    if ref_dupes or out_dupes:
        return "fail", (f"duplicate keys: {len(ref_dupes)} in the reference, "
                        f"{len(out_dupes)} in the produced file")

    shared = set(reference) & set(produced)
    if not shared:
        return "fail", "no rows with matching keys"
    missing_rows = len(reference) - len(shared)

    mismatches, loosened = [], 0
    columns = [c for c in reference[next(iter(shared))] if c not in IGNORED]
    for key in sorted(shared):
        for column in columns:
            expected = reference[key][column]
            target = ALIASES.get(column, column)
            if target not in produced[key]:
                if expected.strip() == "":
                    continue  # the paper's run recorded nothing here either
                mismatches.append(f"{keys}={key} {column}: paper={expected!r} "
                                  f"got=<column {target!r} missing>")
                continue
            actual = produced[key][target]
            if column in LOOSE:
                limit = LOOSE[column]
                if limit is None:
                    continue
                if _same(expected, actual, limit):
                    loosened += 1
                    continue
            elif _same(expected, actual, tolerance):
                continue
            mismatches.append(f"{keys}={key} {column}: paper={expected!r} got={actual!r}")

    detail = f"{len(shared)} of {len(reference)} reference rows compared"
    if loosened:
        detail += f", {loosened} stochastic-GIA values within tolerance"
    if len(produced) > len(reference):
        detail += f" (+{len(produced) - len(reference)} extra produced rows)"
    if mismatches:
        return "fail", detail + "\n      " + "\n      ".join(mismatches[:12]) + (
            f"\n      ... and {len(mismatches) - 12} more" if len(mismatches) > 12 else "")
    if missing_rows:
        return "partial", detail + f" -- {missing_rows} reference rows were not produced"
    return "pass", detail


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", metavar="TARGET",
                        help=f"results to check (default: all). "
                             f"one of: {', '.join(sorted(CHECKS))}")
    args = parser.parse_args()

    unknown = [name for name in args.targets if name not in CHECKS]
    if unknown:
        parser.error(f"unknown target(s): {', '.join(unknown)}; "
                     f"choose from {', '.join(sorted(CHECKS))}")
    names = args.targets or sorted(CHECKS)
    verdicts = {}
    for name in names:
        status, detail = check(name)
        verdicts[name] = status
        mark = {"pass": "PASS", "partial": "PART", "fail": "FAIL", "skip": "skip"}[status]
        print(f"[{mark}] {name:<10} {detail}")

    failed = [n for n, v in verdicts.items() if v == "fail"]
    partial = [n for n, v in verdicts.items() if v == "partial"]
    passed = [n for n, v in verdicts.items() if v == "pass"]
    checked = len(passed) + len(partial) + len(failed)
    print()
    print(f"{len(passed)}/{checked} checked results match the paper in full"
          + (f", {len(partial)} only on a subset of rows" if partial else "")
          + f" ({len(names) - checked} skipped)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

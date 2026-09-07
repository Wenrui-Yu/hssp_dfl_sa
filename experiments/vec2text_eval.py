"""vec2text inversion and text metrics for Sentiment140 -- Figure 10, Tables 2 and 12-14.

Reads the reconstructed ada-002 embeddings dumped by
experiments/attack_sentiment140.py, inverts them back to natural language with
vec2text, and scores ROUGE-L against the ground-truth tweets.

Does not need SageMath; it does need `vec2text` and `rouge-score`.

Requires an OpenAI API key: the ada-002 corrector re-embeds each hypothesis
through the OpenAI embedding API at every correction step.

    export OPENAI_API_KEY=<your key>

The first run also downloads the corrector checkpoint (~1 GB).  Inverting all
360 embeddings with the paper's 20 correction steps takes a few hours.  Only
this final embedding-to-text step needs the API -- the lattice attack, the
embedding MSE and the cosine similarity are already in
sentiment140_case_metrics.csv and need nothing external.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hssp_dfl import paths

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--embeddings",
    default=str(paths.RESULTS / "real_data" / "sentiment140" /
                "sentiment140_reconstructed_embeddings.pkl"),
)
_parser.add_argument("--texts",
                     default=str(paths.DATASET_DIR / "texts_sentiment140_ni1_N10.pkl"))
_parser.add_argument("--num-steps", type=int, default=20,
                     help="vec2text correction steps; 0 skips correction "
                          "(much faster, lower ROUGE-L)")
_parser.add_argument(
    "--output",
    default=str(paths.RESULTS / "real_data" / "sentiment140" /
                "sentiment140_vec2text_metrics.csv"),
)
_ARGS = _parser.parse_args()

import collections
import csv
import os
import pickle

import numpy as np
from rouge_score import rouge_scorer
import torch
import vec2text


EMBEDDING_DUMP_PATH = _ARGS.embeddings
PKL_TEXTS = _ARGS.texts
OUT_CSV_PATH = _ARGS.output

CORRECTOR_NAME = "text-embedding-ada-002"
NUM_STEPS = _ARGS.num_steps
SEQUENCE_BEAM_WIDTH = 4


def compute_rouge_l(reference, hypothesis):
    if not reference or not hypothesis:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return float(scores["rougeL"].fmeasure)


def load_all_texts(path):
    with open(path, "rb") as f:
        split_texts = pickle.load(f)

    all_texts = []
    for node_idx in range(len(split_texts)):
        all_texts.extend(split_texts[node_idx])
    return all_texts


def invert_embedding(corrector, device, embedding, num_steps):
    emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    emb_tensor = emb_tensor / emb_tensor.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    emb_tensor = emb_tensor.to(device)

    return vec2text.invert_embeddings(
        embeddings=emb_tensor,
        corrector=corrector,
        num_steps=num_steps,
        sequence_beam_width=SEQUENCE_BEAM_WIDTH,
    )[0]


def main():
    if not os.path.exists(EMBEDDING_DUMP_PATH):
        raise FileNotFoundError(
            f"Missing embedding dump: {EMBEDDING_DUMP_PATH}. "
            "Run dfl_sentiment140_attack_stat.py first."
        )

    if not os.path.exists(PKL_TEXTS):
        raise FileNotFoundError(f"Missing text pickle: {PKL_TEXTS}")

    with open(EMBEDDING_DUMP_PATH, "rb") as f:
        embedding_rows = pickle.load(f)

    if not embedding_rows:
        print("No embedding rows found. Exiting.")
        return

    all_texts = load_all_texts(PKL_TEXTS)

    print(f"Loaded {len(embedding_rows)} reconstructed embeddings")
    print(f"Loaded {len(all_texts)} ground-truth texts")
    print(f"Loading vec2text corrector: {CORRECTOR_NAME}")

    corrector = vec2text.load_pretrained_corrector(CORRECTOR_NAME)
    device = next(corrector.model.parameters()).device

    # The ada-002 corrector re-embeds each hypothesis through the OpenAI
    # embedding API at every correction step, so inversion needs a key.  The
    # zero-step path is not a usable fallback: it depends on a vec2text
    # internal ("hypothesis_input_ids") that recent releases no longer
    # populate, so say so up front instead of failing per sample.
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n\n"
            "vec2text inverts ada-002 embeddings by re-embedding its hypotheses\n"
            "through the OpenAI embedding API, so this step needs a key:\n"
            "    export OPENAI_API_KEY=<your key>\n\n"
            "To build Figure 10 and Tables 12-14 without re-running the\n"
            "inversion, use the published metrics instead:\n"
            "    python figures/make_stat_figures.py --dataset sentiment140 \\\n"
            "        --sentiment-csv reference/tables/"
            "figure10_sentiment140_vec2text_metrics.csv\n"
            "    python figures/make_text_tables.py \\\n"
            "        --input reference/tables/"
            "table12-14_sentiment140_with_bertscore.csv\n\n"
            "Everything up to the recovered embeddings -- the lattice attack,\n"
            "the cosine similarity and the embedding MSE in\n"
            "results/real_data/sentiment140/sentiment140_case_metrics.csv --\n"
            "needs no API access and has already been computed."
        )
    effective_num_steps = NUM_STEPS

    metric_rows = []

    for idx, row in enumerate(embedding_rows, start=1):
        node = int(row["node"])
        embedding_hat = np.array(row["embedding_hat"], dtype=np.float32)

        try:
            recovered_text = invert_embedding(
                corrector,
                device,
                embedding_hat,
                num_steps=effective_num_steps,
            )
        except Exception as e:
            err_text = str(e)
            needs_openai_retry = (
                effective_num_steps > 0
                and ("OPENAI_API_KEY" in err_text or "api_key" in err_text)
            )
            if needs_openai_retry:
                print(
                    "vec2text requested OpenAI API during correction. "
                    "Retrying this sample with num_steps=0 (offline mode)."
                )
                recovered_text = invert_embedding(
                    corrector,
                    device,
                    embedding_hat,
                    num_steps=0,
                )
            else:
                raise
        gt_text = all_texts[node] if 0 <= node < len(all_texts) else ""
        rouge_l = compute_rouge_l(gt_text, recovered_text)

        out_row = {
            "sg_idx": row["sg_idx"],
            "case": row["case"],
            "sample_rank": row["sample_rank"],
            "recon_index": row["recon_index"],
            "node": node,
            "label": row.get("label", ""),
            "row_type": "node",
            "matched_mse_mean": row.get("matched_mse_mean", ""),
            "matched_mse_node": row.get("matched_mse_node", ""),
            "cosine_similarity": row.get("cosine_similarity", ""),
            "rouge_l": rouge_l,
            "recovered_text": recovered_text,
            "ground_truth_text": gt_text,
        }
        metric_rows.append(out_row)

        print(
            f"[{idx}/{len(embedding_rows)}] "
            f"sg={row['sg_idx']} case={row['case']} node={node} "
            f"rouge_l={rouge_l:.6f}"
        )

    case_stats = collections.defaultdict(lambda: {
        "matched_mse_node": [],
        "cosine_similarity": [],
        "rouge_l": [],
    })

    for r in metric_rows:
        case_name = r["case"]
        if r["matched_mse_node"] != "":
            case_stats[case_name]["matched_mse_node"].append(float(r["matched_mse_node"]))
        if r["cosine_similarity"] != "":
            case_stats[case_name]["cosine_similarity"].append(float(r["cosine_similarity"]))
        case_stats[case_name]["rouge_l"].append(float(r["rouge_l"]))

    for case_name, vals in sorted(case_stats.items()):
        if not vals["rouge_l"]:
            continue

        metric_rows.append({
            "sg_idx": "all",
            "case": case_name,
            "sample_rank": "all",
            "recon_index": "all",
            "node": "case_summary",
            "label": "",
            "row_type": "case_summary",
            "matched_mse_mean": "",
            "matched_mse_node": "",
            "cosine_similarity": "",
            "rouge_l": "",
            "recovered_text": "",
            "ground_truth_text": "",
            "case_mse_mean": float(np.mean(vals["matched_mse_node"])) if vals["matched_mse_node"] else "",
            "case_mse_var": float(np.var(vals["matched_mse_node"])) if vals["matched_mse_node"] else "",
            "case_cosine_similarity_mean": float(np.mean(vals["cosine_similarity"])) if vals["cosine_similarity"] else "",
            "case_cosine_similarity_var": float(np.var(vals["cosine_similarity"])) if vals["cosine_similarity"] else "",
            "case_rouge_l_mean": float(np.mean(vals["rouge_l"])),
            "case_rouge_l_var": float(np.var(vals["rouge_l"])),
        })

    os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)
    fieldnames = [
        "sg_idx",
        "case",
        "sample_rank",
        "recon_index",
        "node",
        "label",
        "row_type",
        "matched_mse_mean",
        "matched_mse_node",
        "cosine_similarity",
        "rouge_l",
        "recovered_text",
        "ground_truth_text",
        "case_mse_mean",
        "case_mse_var",
        "case_cosine_similarity_mean",
        "case_cosine_similarity_var",
        "case_rouge_l_mean",
        "case_rouge_l_var",
    ]

    with open(OUT_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"Saved vec2text + ROUGE metrics -> {OUT_CSV_PATH}")


if __name__ == "__main__":
    main()

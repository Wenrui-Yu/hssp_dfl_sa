"""All-candidate reconstruction grids -- paper Figures 12-17.

For each of Cases 1-3, arranges the ground truth plus every sampled candidate
reconstruction into one figure:

    Figures 12-14  CIFAR-10      (results/real_data/cifar30)
    Figures 15-17  Purchase-100  (results/real_data/purchase)
"""

#!/usr/bin/env python3
"""
Build case-wise summary figures from figure_stat folders.

For each input directory (figure_stat30, figure_stat_purchase), this script:
1) reads ground-truth node images and case{1,2,3} reconstruction node images,
2) arranges rows as: Ground Truth + Recon 1..N in a 3-panel layout,
3) writes three outputs (one per case).

Expected filename patterns:
- ground_truth_sg0_node{node}.png
- case1_exact_pattern_recon_sg0_sel{sel}_idx{idx}_node{node}.png
- case2_zero_count_recon_sg0_sel{sel}_idx{idx}_node{node}.png
- case3_no_structure_recon_sg0_sel{sel}_idx{idx}_node{node}.png
"""

import glob
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hssp_dfl import paths

import numpy as np


import argparse

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--input-dirs", nargs="+", default=[
    str(paths.RESULTS / "real_data" / "cifar30"),
    str(paths.RESULTS / "real_data" / "purchase"),
])
_parser.add_argument("--output-dir",
                     default=str(paths.RESULTS / "figures"))
_ARGS = _parser.parse_args()
INPUT_DIRS = _ARGS.input_dirs

FIGURE_NUMBER = {
    "cifar": {"case1": 12, "case2": 13, "case3": 14},
    "purchase": {"case1": 15, "case2": 16, "case3": 17},
}
CASE_SPECS = [
    ("case1", "case1_exact_pattern", "Case 1"),
    ("case2", "case2_zero_count", "Case 2"),
    ("case3", "case3_no_structure", "Case 3"),
]


def node_key(path):
    m = re.search(r"_node(\d+)\.png$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def sel_key(path):
    m = re.search(r"_sel(\d+)_idx\d+_node\d+\.png$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def load_img(path):
    img = mpimg.imread(path)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def build_three_panel_figure(rows_data, node_ids,
                             left_pad=0.9, col_width=2.0, row_height=2.0,
                             col_gap=0.1, row_gap=0.1, panel_gap=0.4,
                             n_panels=3, label_fontsize=20,
                             first_row_only_gt=True,
                             gt_row_scale=1.35):
    n_rows = len(rows_data)
    n_cols = len(node_ids)
    panel_w = left_pad + n_cols * col_width + (n_cols - 1) * col_gap
    fig_w = n_panels * panel_w + (n_panels - 1) * panel_gap

    if not first_row_only_gt or n_rows == 0:
        rpp = max(1, math.ceil(max(1, n_rows) / n_panels))
        panel_h = rpp * row_height + (rpp - 1) * row_gap
        fig_h = panel_h
        fig = plt.figure(figsize=(fig_w, fig_h))

        for r_idx, (row_label, node_imgs) in enumerate(rows_data):
            p = r_idx // rpp
            local_r = r_idx % rpp
            panel_x0 = p * (panel_w + panel_gap)
            node_map = {nid: img for nid, img in node_imgs}

            for c_idx, nid in enumerate(node_ids):
                left = (panel_x0 + left_pad + c_idx * (col_width + col_gap)) / fig_w
                bottom = 1.0 - (local_r + 1) * (row_height + row_gap) / fig_h
                width = col_width / fig_w
                height = row_height / fig_h

                ax = fig.add_axes([left, bottom, width, height])
                img = node_map.get(nid)
                if img is not None:
                    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                ax.axis("off")

                if r_idx == 0:
                    ax.set_title(f"Node {nid}", fontsize=20, pad=4)

            label_x = (panel_x0 + left_pad * 0.5) / fig_w
            label_y = 1.0 - (local_r + 0.5) * (row_height + row_gap) / fig_h
            fig.text(label_x, label_y, row_label,
                     ha="center", va="center", fontsize=label_fontsize,
                     rotation=90, fontweight="bold")

        return fig

    gt_row = rows_data[0]
    recon_rows = rows_data[1:]
    n_recon = len(recon_rows)

    rpp = max(1, math.ceil(max(1, n_recon) / n_panels))
    recon_h = rpp * row_height + (rpp - 1) * row_gap
    gt_h = row_height * gt_row_scale
    fig_h = gt_h + row_gap + recon_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    gt_label, gt_imgs = gt_row
    gt_map = {nid: img for nid, img in gt_imgs}
    gt_block_w = n_cols * col_width + (n_cols - 1) * col_gap
    gt_x0 = (fig_w - gt_block_w) / 2.0
    gt_bottom_in = fig_h - gt_h

    for c_idx, nid in enumerate(node_ids):
        left = (gt_x0 + c_idx * (col_width + col_gap)) / fig_w
        bottom = gt_bottom_in / fig_h
        width = col_width / fig_w
        height = gt_h / fig_h

        ax = fig.add_axes([left, bottom, width, height])
        img = gt_map.get(nid)
        if img is not None:
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.axis("off")
        ax.set_title(f"Node {nid}", fontsize=20, pad=4)

    gt_label_y = min(0.995, (gt_bottom_in + gt_h) / fig_h + 0.004)
    fig.text(0.5, gt_label_y, gt_label,
             ha="center", va="bottom", fontsize=20, fontweight="bold")

    for r_idx, (row_label, node_imgs) in enumerate(recon_rows):
        p = r_idx // rpp
        local_r = r_idx % rpp

        panel_x0 = p * (panel_w + panel_gap)
        node_map = {nid: img for nid, img in node_imgs}

        for c_idx, nid in enumerate(node_ids):
            left = (panel_x0 + left_pad + c_idx * (col_width + col_gap)) / fig_w
            bottom_in = recon_h - (local_r + 1) * row_height - local_r * row_gap
            bottom = bottom_in / fig_h
            width = col_width / fig_w
            height = row_height / fig_h

            ax = fig.add_axes([left, bottom, width, height])
            img = node_map.get(nid)
            if img is not None:
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.axis("off")

        label_x = (panel_x0 + left_pad * 0.5) / fig_w
        label_y = (recon_h - local_r * (row_height + row_gap) - row_height / 2.0) / fig_h
        fig.text(label_x, label_y, row_label,
                 ha="center", va="center", fontsize=label_fontsize,
                 rotation=90, fontweight="bold")

    return fig


def collect_gt_rows(input_dir):
    gt_paths = sorted(
        glob.glob(os.path.join(input_dir, "ground_truth_sg0_node*.png")),
        key=node_key,
    )
    return [(node_key(p), load_img(p)) for p in gt_paths if node_key(p) >= 0]


def collect_case_rows(input_dir, case_prefix):
    pattern = os.path.join(input_dir, f"{case_prefix}_recon_sg0_sel*_idx*_node*.png")
    paths = sorted(glob.glob(pattern), key=lambda p: (sel_key(p), node_key(p)))

    rows = {}
    for p in paths:
        s = sel_key(p)
        n = node_key(p)
        if s == 10**9 or n < 0:
            continue
        rows.setdefault(s, []).append((n, load_img(p)))

    ordered_rows = []
    for s in sorted(rows):
        ordered_rows.append((f"Recon {s}", sorted(rows[s], key=lambda x: x[0])))
    return ordered_rows


def build_for_dir(input_dir):
    if not os.path.isdir(input_dir):
        print(f"Skip: directory not found -> {input_dir}")
        return

    gt_imgs = collect_gt_rows(input_dir)
    if not gt_imgs:
        print(f"Skip: no GT node images in {input_dir}")
        return

    node_ids = [nid for nid, _ in gt_imgs]

    for case_short, case_prefix, case_title in CASE_SPECS:
        recon_rows = collect_case_rows(input_dir, case_prefix)
        if not recon_rows:
            print(f"Skip: no {case_short} recon node images in {input_dir}")
            continue

        rows_data = [("Ground Truth", gt_imgs)] + recon_rows
        fig = build_three_panel_figure(rows_data, node_ids, first_row_only_gt=True)

        # Figures 12-14 are the CIFAR grids, 15-17 the Purchase ones.
        is_purchase = "purchase" in os.path.basename(input_dir)
        number = FIGURE_NUMBER["purchase" if is_purchase else "cifar"][case_short]
        dataset = "purchase" if is_purchase else "cifar"
        os.makedirs(_ARGS.output_dir, exist_ok=True)
        out_path = os.path.join(
            _ARGS.output_dir,
            f"figure{number}_{dataset}_{case_short}.png",
        )
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(
            f"Saved -> {out_path} "
            f"({case_title}, rows={len(rows_data)}, nodes={len(node_ids)})"
        )


def main():
    for d in INPUT_DIRS:
        build_for_dir(d)


if __name__ == "__main__":
    main()

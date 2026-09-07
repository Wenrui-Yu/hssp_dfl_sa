"""Visual comparison of CIFAR-10 reconstructions -- paper Figure 3.

Stitches the per-node PNGs written by experiments/attack_cifar.py into the
Ground Truth / true solution / Case 1 / Case 2 / Case 3 panel of Section 7.4.

Input : results/real_data/cifar30/            (--input-dir)
Output: results/figures/figure3_cifar_recon_summary.png
"""


import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from hssp_dfl import paths

_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--input-dir",
                     default=str(paths.RESULTS / "real_data" / "cifar30"),
                     help="directory of per-node reconstruction PNGs")
_parser.add_argument("--output",
                     default=str(paths.RESULTS / "figures" /
                                 "figure3_cifar_recon_summary.png"))
for _case, _default in (("case1", "1:1"), ("case2", "24:371"), ("case3", "9:513406")):
    _parser.add_argument(
        f"--{_case}-candidate", default=_default, metavar="SEL:IDX",
        help=f"which {_case} candidate to show, as 'sample_rank:recon_index'. "
             f"'auto' picks the first available one. The default is the "
             f"candidate the paper shows; a fresh run whose candidate indices "
             f"differ falls back to 'auto' automatically.")
_ARGS = _parser.parse_args()
FIG_STAT30_DIR = _ARGS.input_dir


def _parse_candidate(text):
    if text is None or text.strip().lower() == "auto":
        return None
    sel, _, idx = text.partition(":")
    return int(sel), int(idx)


# Rows 3-5 of Figure 3: one representative candidate per case.  The paper's
# choices are the defaults; they are only honoured if the run actually produced
# them, so re-running with a different candidate sample still yields a figure.
MANUAL_CASE_SELECTION = {
    "case1": _parse_candidate(_ARGS.case1_candidate),
    "case2": _parse_candidate(_ARGS.case2_candidate),
    "case3": _parse_candidate(_ARGS.case3_candidate),
}

def node_key(path):
    m = re.search(r"node(\d+)", path)
    return int(m.group(1)) if m else 0


def load_img(path):
    img = mpimg.imread(path)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _collect_case_groups(case_prefix):
    pattern = f"{FIG_STAT30_DIR}/{case_prefix}_recon_sg0_sel*_idx*_node*.png"
    files = glob.glob(pattern)
    rx = re.compile(r"_sel(\d+)_idx(\d+)_node\d+\.png$")
    groups = set()
    for p in files:
        m = rx.search(p)
        if m:
            groups.add((int(m.group(1)), int(m.group(2))))
    return sorted(groups)


def _group_pattern(case_prefix, sel_idx_pair):
    sel, idx = sel_idx_pair
    return f"{FIG_STAT30_DIR}/{case_prefix}_recon_sg0_sel{sel}_idx{idx}_node*.png"


def _pick_one_group(case_prefix, excluded=None):
    excluded = excluded or set()
    for pair in _collect_case_groups(case_prefix):
        if pair not in excluded:
            return pair
    raise FileNotFoundError(f"No selectable group found for {case_prefix}")


def _resolve_group(case_prefix, manual_pair=None, excluded=None):
    excluded = excluded or set()
    available = _collect_case_groups(case_prefix)
    if manual_pair is not None:
        if manual_pair in available and manual_pair not in excluded:
            return manual_pair
        print(f"note: candidate {manual_pair[0]}:{manual_pair[1]} is not available "
              f"for {case_prefix} in this run; falling back to the first one. "
              f"Available: {available[:5]}")

    for pair in available:
        if pair not in excluded:
            return pair
    raise FileNotFoundError(f"No selectable group found for {case_prefix}")


def build_three_panel_figure(rows_data, node_ids,
                             left_pad=0.9, col_width=2.0, row_height=2.0,
                             col_gap=0.1, row_gap=0.1, panel_gap=0.4,
                             n_panels=3, label_fontsize=20,
                             first_row_only_gt=False,
                             gt_row_scale=1.35):
    """Lay out rows_data into n_panels side-by-side panels to reduce height."""
    import math
    n_rows  = len(rows_data)
    n_cols  = len(node_ids)
    panel_w = left_pad + n_cols * col_width + (n_cols - 1) * col_gap
    fig_w = n_panels * panel_w + (n_panels - 1) * panel_gap

    if not first_row_only_gt or n_rows == 0:
        rpp = max(1, math.ceil(max(1, n_rows) / n_panels))
        panel_rows = rpp
        panel_h = panel_rows * row_height + (panel_rows - 1) * row_gap
        fig_h = panel_h
        fig = plt.figure(figsize=(fig_w, fig_h))

        for r_idx, (row_label, node_imgs) in enumerate(rows_data):
            p = r_idx // rpp
            local_r = r_idx % rpp

            panel_x0 = p * (panel_w + panel_gap)
            node_map = {nid: img for nid, img in node_imgs}

            for c_idx, nid in enumerate(node_ids):
                left   = (panel_x0 + left_pad + c_idx * (col_width + col_gap)) / fig_w
                bottom = 1.0 - (local_r + 1) * (row_height + row_gap) / fig_h
                width  = col_width / fig_w
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
                     ha="center", va="center", fontsize=label_fontsize, rotation=90,
                     fontweight="bold")

        return fig

    # Special layout: top GT row centered across full figure width,
    # remaining rows split into n_panels below.
    gt_row = rows_data[0]
    recon_rows = rows_data[1:]
    n_recon = len(recon_rows)

    rpp = max(1, math.ceil(max(1, n_recon) / n_panels))
    recon_h = rpp * row_height + (rpp - 1) * row_gap
    gt_h = row_height * gt_row_scale
    fig_h = gt_h + row_gap + recon_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    # 1) Centered GT row on top
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

    # 2) Recon rows in panels below GT
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
                 ha="center", va="center", fontsize=label_fontsize, rotation=90,
                 fontweight="bold")

    return fig


# ── File patterns ────────────────────────────────────────────────────
case1_true = (24, 24)
case1_alt = _resolve_group(
    "case1_exact_pattern",
    manual_pair=MANUAL_CASE_SELECTION["case1"],
    excluded={case1_true},
)
case2_alt = _resolve_group("case2_zero_count", manual_pair=MANUAL_CASE_SELECTION["case2"])
case3_alt = _resolve_group("case3_no_structure", manual_pair=MANUAL_CASE_SELECTION["case3"])

print(f"Selected Case 1 group: sel={case1_alt[0]}, idx={case1_alt[1]}")
print(f"Selected Case 2 group: sel={case2_alt[0]}, idx={case2_alt[1]}")
print(f"Selected Case 3 group: sel={case3_alt[0]}, idx={case3_alt[1]}")

PATTERNS = [
    ("Ground Truth", f"{FIG_STAT30_DIR}/ground_truth_sg0_node*.png"),
    ("True ", _group_pattern("case1_exact_pattern", case1_true)),
    ("Case 1  ", _group_pattern("case1_exact_pattern", case1_alt)),
    ("Case 2   ", _group_pattern("case2_zero_count", case2_alt)),
    ("Case 3    ", _group_pattern("case3_no_structure", case3_alt)),
]

# ── Collect and sort files ───────────────────────────────────────────
rows_data = []   # list of (label, [(node_id, img), ...])
for label, pattern in PATTERNS:
    files = sorted(glob.glob(pattern), key=node_key)
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    rows_data.append((label, [(node_key(p), load_img(p)) for p in files]))

# Derive ordered node list from the GT row.
node_ids = [nid for nid, _ in rows_data[0][1]]

# Layout: GT as a single top row; remaining 4 rows arranged as 2x2 panels.
fig = build_three_panel_figure(
    rows_data,
    node_ids,
    n_panels=2,
    first_row_only_gt=True,
)

out_path = _ARGS.output
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")

# # ########################################################################
# # ── File patterns ────────────────────────────────────────────────────
# PATTERNS = [
#     ("Ground Truth",               "figure/purchase_gt_sg0_node*.png"),
#     ("5th", "figure/purchase_recon_sg0_r5_node*.png"),
#     ("24th (true)","figure/purchase_recon_sg0_r24_node*.png"),
# ]

# # ── Collect and sort files ───────────────────────────────────────────
# rows_data = []   # list of (label, [(node_id, img), ...])
# for label, pattern in PATTERNS:
#     files = sorted(glob.glob(pattern), key=node_key)
#     if not files:
#         raise FileNotFoundError(f"No files matched: {pattern}")
#     rows_data.append((label, [(node_key(p), load_img(p)) for p in files]))

# # Derive ordered node list from the GT row
# node_ids = [nid for nid, _ in rows_data[0][1]]
# n_rows   = len(rows_data)
# n_cols   = len(node_ids)

# # ── Build figure ─────────────────────────────────────────────────────
# LEFT_PAD   = 0.6   # inches reserved for row label
# COL_WIDTH  = 2.0   # inches per column
# ROW_HEIGHT = 2.0   # inches per row
# COL_GAP = 0.1
# ROW_GAP = 0.1

# fig_w = LEFT_PAD + n_cols * COL_WIDTH + (n_cols - 1) * COL_GAP
# fig_h = n_rows * ROW_HEIGHT + (n_rows - 1) * ROW_GAP

# fig = plt.figure(figsize=(fig_w, fig_h))

# for r_idx, (row_label, node_imgs) in enumerate(rows_data):
#     # Build a lookup so missing nodes get a blank slot
#     node_map = {nid: img for nid, img in node_imgs}

#     for c_idx, nid in enumerate(node_ids):
#         # axes position: leave left margin for row labels
#         left = (LEFT_PAD + c_idx * (COL_WIDTH + COL_GAP)) / fig_w
#         bottom = 1.0 - (r_idx + 1) * (ROW_HEIGHT + ROW_GAP) / fig_h
#         width  = COL_WIDTH / fig_w
#         height = ROW_HEIGHT / fig_h

#         ax = fig.add_axes([left, bottom, width, height])

#         img = node_map.get(nid)
#         if img is not None:
#             if img.ndim == 2:
#                 ax.imshow(img, cmap="gray")
#             else:
#                 ax.imshow(img)
#         ax.axis("off")

#         # Column header: node id (top row only)
#         if r_idx == 0:
#             ax.set_title(f"Node {nid}", fontsize=20, pad=4)

#     # Row label on the left
#     label_x = (LEFT_PAD * 0.5) / fig_w
#     label_y = 1.0 - (r_idx + 0.5) * ROW_HEIGHT / fig_h
#     fig.text(label_x, label_y, row_label,
#              ha="center", va="center", fontsize=20, rotation=90,
#              fontweight="bold")

# out_path = "figure/purchase_recon_summary.png"
# plt.savefig(out_path, dpi=180, bbox_inches="tight")
# plt.close(fig)
# print(f"Saved → {out_path}")


# # ########################################################################
# # ── CIFAR: GT + Recon 1-30 (31 rows, 3-panel layout) ───────────────
# PATTERNS_CIFAR_ALL = (
#     [("Ground Truth", "figure/ground_truth_sg0_node*.png")] +
#     [(f"Recon {i}", f"figure/recon_sg0_r{i}_node*.png") for i in range(1, 31)]
# )

# rows_data = []
# for label, pattern in PATTERNS_CIFAR_ALL:
#     files = sorted(glob.glob(pattern), key=node_key)
#     if files:
#         rows_data.append((label, [(node_key(p), load_img(p)) for p in files]))

# if rows_data:
#     node_ids = [nid for nid, _ in rows_data[0][1]]
#     fig = build_three_panel_figure(rows_data, node_ids, first_row_only_gt=True)
#     out_path = "figure/cifar_recon_summary_all.png"
#     plt.savefig(out_path, dpi=180, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved → {out_path}")


# # ── Purchase: GT + Recon 1-30 (31 rows, 3-panel layout) ────────────
# PATTERNS_PURCHASE_ALL = (
#     [("Ground Truth", "figure/purchase_gt_sg0_node*.png")] +
#     [(f"Recon {i}", f"figure/purchase_recon_sg0_r{i}_node*.png") for i in range(1, 31)]
# )

# rows_data = []
# for label, pattern in PATTERNS_PURCHASE_ALL:
#     files = sorted(glob.glob(pattern), key=node_key)
#     if files:
#         rows_data.append((label, [(node_key(p), load_img(p)) for p in files]))

# if rows_data:
#     node_ids = [nid for nid, _ in rows_data[0][1]]
#     fig = build_three_panel_figure(rows_data, node_ids, first_row_only_gt=True)
#     out_path = "figure/purchase_recon_summary_all.png"
#     plt.savefig(out_path, dpi=180, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved → {out_path}")
#!/usr/bin/env bash
# Populate assets/ with the large inputs the real-dataset experiments need.
#
# Base batch-size-1 checkpoints and per-node datasets are included in assets/.
# This script imports additional checkpoints and datasets from another tree.
#
# Usage
#   bash scripts/setup_assets.sh --from /path/to/source_repo          # symlink
#   bash scripts/setup_assets.sh --from /path/to/source_repo --copy   # copy
#   bash scripts/setup_assets.sh --check                              # report only
#   bash scripts/setup_assets.sh --clean                              # empty assets/
#
# Use --clean before packaging the artifact: symlinks record the absolute path
# of wherever your checkpoints live, which you probably do not want to ship.
#
# Everything can also be pointed at directly with environment variables; see
# hssp_dfl/paths.py.  If you have no checkpoints at all, train them from
# scratch with the scripts in training/ (see README, "Producing the assets").

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$ROOT/assets"
SOURCE=""
MODE="link"
CHECK_ONLY=0
CLEAN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) SOURCE="$2"; shift 2 ;;
        --copy) MODE="copy"; shift ;;
        --link) MODE="link"; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        --clean) CLEAN=1; shift ;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ "$CLEAN" == 1 ]]; then
    echo "emptying $ASSETS"
    find "$ASSETS" -mindepth 1 -not -name '.gitkeep' -maxdepth 1 -exec rm -rf {} +
    touch "$ASSETS/.gitkeep"
    echo "done -- assets/ now holds only its placeholder"
    exit 0
fi

mkdir -p "$ASSETS"/{models,models_dp,datasets,data}

# --- what the experiments look for -----------------------------------------
NETWORK="network.mat"

MODELS_CORE=(
    "model_avg_ni1_N10_t0_z0_e1.pkl"
    "model_avg_ni1_N10_t0_5_z0_e1.pkl"
    "model_avg_ni1_N10_t1_z0_e1.pkl"
    "model_purchase_ni1_N10_t0_z0_e1.pkl"
    "model_purchase_ni1_N10_t0_5_z0_e1.pkl"
    "model_purchase_ni1_N10_t1_z0_e1.pkl"
    "model_sentiment140_ni1_N10_t0_z0_e1.pkl"
    "model_sentiment140_ni1_N10_t0_5_z0_e1.pkl"
    "model_sentiment140_ni1_N10_t1_z0_e1.pkl"
)

DATASETS_CORE=(
    "dataset_ni1_N10.pkl"
    "dataset_purchase_ni1_N10.pkl"
    "dataset_sentiment140_ni1_N10.pkl"
    "texts_sentiment140_ni1_N10.pkl"
)

# Optional: only needed for the large-batch GIA transferability table (Table 6).
MODELS_LARGE_BATCH=(
    "model_avg_ni8_N10_t0_z0_e1.pkl"    "model_avg_ni8_N10_t0_5_z0_e1.pkl"
    "model_avg_ni8_N10_t1_z0_e1.pkl"
    "model_avg_ni32_N10_t0_z0_e1.pkl"   "model_avg_ni32_N10_t0_5_z0_e1.pkl"
    "model_avg_ni32_N10_t1_z0_e1.pkl"
)
DATASETS_LARGE_BATCH=("dataset_ni8_N10.pkl" "dataset_ni32_N10.pkl")

# Optional: only needed for the DP defense figures (Figure 5, Figure 11).
DATASETS_DP=("dataset_ni500_N10.pkl")

place() {  # place <src-file> <dst-file>
    local src="$1" dst="$2"
    [[ -e "$src" ]] || return 1
    rm -f "$dst"
    if [[ "$MODE" == "copy" ]]; then
        cp "$src" "$dst"
    else
        ln -s "$src" "$dst"
    fi
    return 0
}

report() {  # report <label> <path>
    if [[ -e "$2" ]]; then
        printf '  ok   %s\n' "$1"
    else
        printf '  --   %s   (missing: %s)\n' "$1" "$2"
    fi
}

if [[ "$CHECK_ONLY" == 1 ]]; then
    echo "asset inventory under $ASSETS"
    report "$NETWORK" "$ASSETS/$NETWORK"
    for f in "${MODELS_CORE[@]}";   do report "models/$f"   "$ASSETS/models/$f";   done
    for f in "${DATASETS_CORE[@]}"; do report "datasets/$f" "$ASSETS/datasets/$f"; done
    echo "  -- optional (Table 6, large-batch GIA):"
    for f in "${MODELS_LARGE_BATCH[@]}";   do report "models/$f"   "$ASSETS/models/$f";   done
    for f in "${DATASETS_LARGE_BATCH[@]}"; do report "datasets/$f" "$ASSETS/datasets/$f"; done
    echo "  -- optional (Figures 5 and 11, DP defense):"
    for f in "${DATASETS_DP[@]}"; do report "datasets/$f" "$ASSETS/datasets/$f"; done
    report "models_dp/" "$ASSETS/models_dp"
    report "data/"      "$ASSETS/data"
    exit 0
fi

if [[ -z "$SOURCE" ]]; then
    echo "error: --from <source repository> is required (or use --check)" >&2
    exit 2
fi
SOURCE="$(cd -- "$SOURCE" && pwd)"
echo "populating $ASSETS from $SOURCE  (mode: $MODE)"

placed=0; missing=0
place "$SOURCE/$NETWORK" "$ASSETS/$NETWORK" && placed=$((placed + 1)) || { missing=$((missing + 1)); echo "  missing $NETWORK"; }

for f in "${MODELS_CORE[@]}" "${MODELS_LARGE_BATCH[@]}"; do
    place "$SOURCE/models/$f" "$ASSETS/models/$f" && placed=$((placed + 1)) || { missing=$((missing + 1)); echo "  missing models/$f"; }
done

for f in "${DATASETS_CORE[@]}" "${DATASETS_LARGE_BATCH[@]}" "${DATASETS_DP[@]}"; do
    place "$SOURCE/$f" "$ASSETS/datasets/$f" && placed=$((placed + 1)) || { missing=$((missing + 1)); echo "  missing $f"; }
done

# Whole directories: DP checkpoints and raw datasets.
for d in models_dp data; do
    if [[ -d "$SOURCE/$d" ]]; then
        rmdir "$ASSETS/$d" 2>/dev/null || true
        rm -f "$ASSETS/$d"
        if [[ "$MODE" == "copy" ]]; then
            cp -R "$SOURCE/$d" "$ASSETS/$d"
        else
            ln -s "$SOURCE/$d" "$ASSETS/$d"
        fi
        placed=$((placed + 1))
    else
        missing=$((missing + 1)); echo "  missing directory $d/"
    fi
done

echo "done: $placed placed, $missing missing"
echo
bash "${BASH_SOURCE[0]}" --check

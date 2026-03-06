#!/bin/bash
set -euo pipefail

# Reference: benchmarks/basic.sh
# Purpose:
# 1) Train each scene.
# 2) Save checkpoints in <result_dir>/<scene>/ckpts.
# 3) Trigger dual evaluation at checkpoint steps so sorting decision files
#    (sorting_decision_step<iter>.json) are saved in the same ckpts folder.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXAMPLES_DIR"

# ==== User-configurable defaults ====
SCENE_DIR="${SCENE_DIR:-data/360_v2}"
RESULT_ROOT="${RESULT_ROOT:-results/Benchmark_Training}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CONFIG_NAME="${CONFIG_NAME:-default}"

# Training schedule (override from env if needed)
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-7000 30000}"
EVAL_STEPS="${EVAL_STEPS:-7000 30000}"

# If SCENE_LIST is not provided, discover all scene folders in SCENE_DIR.
if [[ -z "${SCENE_LIST:-}" ]]; then
    mapfile -t _SCENE_ARRAY < <(find "$SCENE_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
    SCENE_LIST="${_SCENE_ARRAY[*]}"
fi

echo "Scene dir    : $SCENE_DIR"
echo "Result root  : $RESULT_ROOT"
echo "Scenes       : $SCENE_LIST"
echo "Max steps    : $MAX_STEPS"
echo "Save steps   : $SAVE_STEPS"
echo "Eval steps   : $EVAL_STEPS"

auto_data_factor() {
    local scene="$1"
    case "$scene" in
        bonsai|counter|kitchen|room) echo 2 ;;
        *) echo 4 ;;
    esac
}

for SCENE in $SCENE_LIST; do
    DATA_PATH="$SCENE_DIR/$SCENE"
    if [[ ! -d "$DATA_PATH" ]]; then
        echo "[Skip] Missing scene directory: $DATA_PATH"
        continue
    fi

    DATA_FACTOR="$(auto_data_factor "$SCENE")"
    RESULT_DIR="$RESULT_ROOT/$SCENE"

    echo "========================================"
    echo "Training scene: $SCENE"
    echo "Data path     : $DATA_PATH"
    echo "Data factor   : $DATA_FACTOR"
    echo "Result dir    : $RESULT_DIR"
    echo "========================================"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python simple_trainer.py "$CONFIG_NAME" \
        --disable_viewer \
        --data_dir "$DATA_PATH" \
        --data_factor "$DATA_FACTOR" \
        --result_dir "$RESULT_DIR" \
        --max_steps "$MAX_STEPS" \
        --save_steps $SAVE_STEPS \
        --eval_steps $EVAL_STEPS

    echo "[Done] $SCENE"
    echo "Checkpoints and sorting decisions should be in: $RESULT_DIR/ckpts"

done

echo "All requested scenes finished."

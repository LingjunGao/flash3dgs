#!/bin/bash
set -euo pipefail

# Purpose:
# For each scene and checkpoint step, run evaluation 10 times with optimized
# rasterization and 10 times with unoptimized (original) rasterization.
#
# Output structure per scene:
#   results/Evaluation/<scene>/stats/step<step>/
#     ├── optimized.jsonl      (10 lines: per-run metrics & timings)
#     ├── unoptimized.jsonl    (10 lines: per-run metrics & timings)
#     └── summary.json         (averaged comparison)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXAMPLES_DIR"

SCENE_DIR="${SCENE_DIR:-data/360_v2}"
CKPT_ROOT="${CKPT_ROOT:-results/Benchmark_Training}"
RESULT_DIR="${RESULT_DIR:-results/Evaluation}"
SCENE_LIST="${SCENE_LIST:-bicycle bonsai counter garden kitchen room stump}"
RENDER_TRAJ_PATH="${RENDER_TRAJ_PATH:-ellipse}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-6999 29999}"
EVAL_NUM_RUNS="${EVAL_NUM_RUNS:-10}"

pick_data_factor() {
    local scene="$1"
    case "$scene" in
        bonsai|counter|kitchen|room) echo 2 ;;
        *) echo 4 ;;
    esac
}

write_decision_false() {
    local file="$1"
    local step="$2"
    local scene="$3"
    python3 - "$file" "$step" "$scene" <<'PY'
import json, sys
p, step, scene = sys.argv[1], int(sys.argv[2]), sys.argv[3]
payload = {
    "step": step,
    "scene": scene,
    "use_early_sorting": False,
    "forced_by": "Evaluation.sh",
}
with open(p, "w") as f:
    json.dump(payload, f, indent=2)
PY
}

generate_summary() {
    local step_dir="$1"
    local scene="$2"
    local step="$3"
    local n_runs="$4"
    python3 - "$step_dir" "$scene" "$step" "$n_runs" <<'PY'
import json, sys, os
import numpy as np

step_dir, scene, step, n_runs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

opt = read_jsonl(os.path.join(step_dir, "optimized.jsonl"))
unopt = read_jsonl(os.path.join(step_dir, "unoptimized.jsonl"))

avg_tt_opt = float(np.mean([r["TT"] for r in opt]))
avg_tt_unopt = float(np.mean([r["TT"] for r in unopt]))
ratio = avg_tt_unopt / avg_tt_opt if avg_tt_opt != 0 else float("inf")

# Max per-run TT ratio
max_ratio = max(
    (u["TT"] / o["TT"] if o["TT"] != 0 else float("inf"))
    for o, u in zip(opt, unopt)
)

summary = {
    "scene": scene,
    "step": step,
    "num_runs_each": n_runs,
    "average_TT_optimized": avg_tt_opt,
    "average_TT_unoptimized_no_early_sorting": avg_tt_unopt,
    "ratio_unoptimized_over_optimized": float(ratio),
    "max_ratio_unoptimized_over_optimized": float(max_ratio),
}

out = os.path.join(step_dir, "summary.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to {out}")
PY
}

echo "Scene dir        : $SCENE_DIR"
echo "Checkpoint root  : $CKPT_ROOT"
echo "Result dir       : $RESULT_DIR"
echo "Scenes           : $SCENE_LIST"
echo "Checkpoint steps : $CHECKPOINT_STEPS"
echo "Eval runs        : $EVAL_NUM_RUNS"

for SCENE in $SCENE_LIST; do
    DATA_FACTOR="$(pick_data_factor "$SCENE")"
    CKPT_DIR="$CKPT_ROOT/$SCENE/ckpts"

    echo "========================================"
    echo "Scene: $SCENE  (data_factor=$DATA_FACTOR)"
    echo "Checkpoint dir: $CKPT_DIR"
    echo "========================================"

    for STEP in $CHECKPOINT_STEPS; do
        CKPT_FILE="$CKPT_DIR/ckpt_${STEP}_rank0.pt"
        if [[ ! -f "$CKPT_FILE" ]]; then
            echo "[Skip] Missing checkpoint: $CKPT_FILE"
            continue
        fi

        SCENE_RESULT_DIR="$RESULT_DIR/$SCENE"
        STEP_STATS_DIR="$SCENE_RESULT_DIR/stats/step${STEP}"
        mkdir -p "$STEP_STATS_DIR"

        OPT_JSONL="$STEP_STATS_DIR/optimized.jsonl"
        UNOPT_JSONL="$STEP_STATS_DIR/unoptimized.jsonl"

        # ── 1) Optimized rasterization (uses sorting decision + optimized raster) ──
        echo "[Run][Optimized] scene=$SCENE step=$STEP runs=$EVAL_NUM_RUNS"
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python simple_trainer.py default \
            --disable_viewer \
            --data_factor "$DATA_FACTOR" \
            --eval_num_runs "$EVAL_NUM_RUNS" \
            --eval_use_optimized_raster \
            --eval_output_jsonl "$OPT_JSONL" \
            --ckpt "$CKPT_FILE" \
            --render_traj_path "$RENDER_TRAJ_PATH" \
            --data_dir "$SCENE_DIR/$SCENE/" \
            --result_dir "$SCENE_RESULT_DIR"

        # ── 2) Unoptimized rasterization (force no early sorting) ──
        # Temporarily override sorting decision to force early_sorting=false
        FORCED_DEC="$CKPT_DIR/sorting_decision_step${STEP}.json"
        BACKUP=""
        CREATED_NEW="0"
        if [[ -f "$FORCED_DEC" ]]; then
            BACKUP="${FORCED_DEC}.bak_eval"
            cp "$FORCED_DEC" "$BACKUP"
        else
            CREATED_NEW="1"
        fi
        write_decision_false "$FORCED_DEC" "$STEP" "$SCENE"

        echo "[Run][Unoptimized] scene=$SCENE step=$STEP runs=$EVAL_NUM_RUNS"
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python simple_trainer.py default \
            --disable_viewer \
            --data_factor "$DATA_FACTOR" \
            --eval_num_runs "$EVAL_NUM_RUNS" \
            --eval_output_jsonl "$UNOPT_JSONL" \
            --ckpt "$CKPT_FILE" \
            --render_traj_path "$RENDER_TRAJ_PATH" \
            --data_dir "$SCENE_DIR/$SCENE/" \
            --result_dir "$SCENE_RESULT_DIR"

        # Restore sorting decision file.
        if [[ -n "$BACKUP" && -f "$BACKUP" ]]; then
            mv "$BACKUP" "$FORCED_DEC"
        elif [[ "$CREATED_NEW" == "1" ]]; then
            rm -f "$FORCED_DEC"
        fi

        # ── 3) Generate summary.json from both JSONL files ──
        generate_summary "$STEP_STATS_DIR" "$SCENE" "$STEP" "$EVAL_NUM_RUNS"

        echo "[Done] scene=$SCENE step=$STEP"
        echo "       $OPT_JSONL"
        echo "       $UNOPT_JSONL"
        echo "       $STEP_STATS_DIR/summary.json"
    done
done

echo "All evaluation finished."

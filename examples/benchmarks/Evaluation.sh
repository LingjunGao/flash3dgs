#!/bin/bash
set -euo pipefail

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
    if [[ "$scene" == "bonsai" || "$scene" == "counter" || "$scene" == "kitchen" || "$scene" == "room" ]]; then
        echo 2
    else
        echo 4
    fi
}

read_decision() {
    local ckpt_dir="$1"
    local step="$2"
    local f1="$ckpt_dir/sorting_decision_step${step}.json"
    local f2="$ckpt_dir/sorting_decision_${step}.json"

    local file=""
    if [[ -f "$f1" ]]; then
        file="$f1"
    elif [[ -f "$f2" ]]; then
        file="$f2"
    else
        echo "MISSING"
        return 0
    fi

    python - "$file" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, "r") as f:
    d = json.load(f)
print("true" if bool(d.get("use_early_sorting", False)) else "false")
PY
}

write_decision_false() {
    local file="$1"
    local step="$2"
    local scene="$3"
    python - "$file" "$step" "$scene" <<'PY'
import json, sys
p, step, scene = sys.argv[1], int(sys.argv[2]), sys.argv[3]
payload = {
    "step": step,
    "scene": scene,
    "use_early_sorting": False,
    "forced_by": "eval_rendering_by_sorting_decision.sh",
}
with open(p, "w") as f:
    json.dump(payload, f, indent=2)
PY
}

read_last_total_time() {
    local stats_file="$1"
    python - "$stats_file" <<'PY'
import json, sys
p = sys.argv[1]
last = None
with open(p, "r") as f:
    for line in f:
        s = line.strip()
        if s:
            last = s
if last is None:
    raise RuntimeError(f"No JSON line found in {p}")
obj = json.loads(last)
if "total_time" in obj:
    print(float(obj["total_time"]))
elif "TT" in obj:
    print(float(obj["TT"]))
else:
    raise RuntimeError(f"total_time missing in last JSON record of {p}")
PY
}

sum_float() {
    local a="$1"
    local b="$2"
    python - "$a" "$b" <<'PY'
import sys
print(float(sys.argv[1]) + float(sys.argv[2]))
PY
}

avg_float() {
    local s="$1"
    local n="$2"
    python - "$s" "$n" <<'PY'
import sys
print(float(sys.argv[1]) / float(sys.argv[2]))
PY
}

ratio_float() {
    local num="$1"
    local den="$2"
    python - "$num" "$den" <<'PY'
import sys
u, o = float(sys.argv[1]), float(sys.argv[2])
print(u / o if o != 0 else float('inf'))
PY
}

read_max_image_ratio_from_last_two() {
    local stats_file="$1"
    python - "$stats_file" <<'PY'
import json, sys
p = sys.argv[1]
rows = []
with open(p, "r") as f:
    for line in f:
        s = line.strip()
        if s:
            rows.append(json.loads(s))

if len(rows) < 2:
    raise RuntimeError(f"Need at least 2 JSON lines in {p} to compare optimized/unoptimized runs")

opt = rows[-2]
unopt = rows[-1]
opt_list = opt.get("per_image_total_time", opt.get("per_image_TT", [])) or []
unopt_list = unopt.get("per_image_total_time", unopt.get("per_image_TT", [])) or []

if len(opt_list) == 0 or len(unopt_list) == 0:
    print("inf")
    raise SystemExit(0)

n = min(len(opt_list), len(unopt_list))
best_ratio = -1.0
for i in range(n):
    o = float(opt_list[i])
    u = float(unopt_list[i])
    r = (u / o) if o != 0 else float("inf")
    if r > best_ratio:
        best_ratio = r

print(f"{best_ratio}")
PY
}

write_summary_json() {
    local out_file="$1"
    local scene="$2"
    local step="$3"
    local n_runs="$4"
    local avg_opt="$5"
    local avg_unopt="$6"
    local ratio="$7"
    local max_img_ratio="$8"
    python - "$out_file" "$scene" "$step" "$n_runs" "$avg_opt" "$avg_unopt" "$ratio" "$max_img_ratio" <<'PY'
import json, sys
out_file, scene, step, n_runs, avg_opt, avg_unopt, ratio, max_ratio = sys.argv[1:9]
payload = {
    "scene": scene,
    "step": int(step),
    "num_runs_each": int(n_runs),
    "average_total_time_optimized": float(avg_opt),
    "average_total_time_unoptimized": float(avg_unopt),
    "ratio_unoptimized_over_optimized": float(ratio),
    "max_ratio_unoptimized_over_optimized": float(max_ratio),
}
with open(out_file, "w") as f:
    json.dump(payload, f, indent=2)
print(out_file)
PY
}

for SCENE in $SCENE_LIST; do
    DATA_FACTOR="$(pick_data_factor "$SCENE")"
    CKPT_DIR="$CKPT_ROOT/$SCENE/ckpts"

    echo "========================================"
    echo "Scene: $SCENE"
    echo "Data factor: $DATA_FACTOR"
    echo "Checkpoint dir: $CKPT_DIR"
    echo "========================================"

    for STEP in $CHECKPOINT_STEPS; do
        STEP_PADDED=$(printf "%04d" "$STEP")
        CKPT_FILE="$CKPT_DIR/ckpt_${STEP}_rank0.pt"
        if [[ ! -f "$CKPT_FILE" ]]; then
            echo "[Skip] Missing checkpoint: $CKPT_FILE"
            continue
        fi

        DECISION="$(read_decision "$CKPT_DIR" "$STEP")"
        if [[ "$DECISION" == "MISSING" ]]; then
            echo "[Warn] No sorting decision JSON for scene=$SCENE step=$STEP"
        else
            echo "[Info] scene=$SCENE step=$STEP use_early_sorting=$DECISION"
        fi

        SCENE_RESULT_DIR="$RESULT_DIR/$SCENE"
        STATS_DIR="$SCENE_RESULT_DIR/stats"
        mkdir -p "$STATS_DIR"
        STATS_FILE="$STATS_DIR/val_step${STEP_PADDED}.json"
        SUMMARY_FILE="$STATS_DIR/eval_total_time_comparison_step${STEP}.json"

        # 1) Optimized rasterization (uses decision JSON as-is for early sorting)
        echo "[Run][Optimized] scene=$SCENE step=$STEP eval_num_runs=$EVAL_NUM_RUNS"
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python simple_trainer.py default \
            --disable_viewer \
            --data_factor "$DATA_FACTOR" \
            --eval_num_runs "$EVAL_NUM_RUNS" \
            --eval_use_optimized_raster \
            --ckpt "$CKPT_FILE" \
            --render_traj_path "$RENDER_TRAJ_PATH" \
            --data_dir "$SCENE_DIR/$SCENE/" \
            --result_dir "$SCENE_RESULT_DIR"

        avg_opt="$(read_last_total_time "$STATS_FILE")"

        # 2) Original rasterization + force no early sorting
        forced_dec_file="$CKPT_DIR/sorting_decision_step${STEP}.json"
        backup_file=""
        created_new="0"
        if [[ -f "$forced_dec_file" ]]; then
            backup_file="${forced_dec_file}.bak_eval"
            cp "$forced_dec_file" "$backup_file"
        else
            created_new="1"
        fi
        write_decision_false "$forced_dec_file" "$STEP" "$SCENE"

        echo "[Run][Unoptimized,no-early-sorting] scene=$SCENE step=$STEP eval_num_runs=$EVAL_NUM_RUNS"
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python simple_trainer.py default \
            --disable_viewer \
            --data_factor "$DATA_FACTOR" \
            --eval_num_runs "$EVAL_NUM_RUNS" \
            --ckpt "$CKPT_FILE" \
            --render_traj_path "$RENDER_TRAJ_PATH" \
            --data_dir "$SCENE_DIR/$SCENE/" \
            --result_dir "$SCENE_RESULT_DIR"

        avg_unopt="$(read_last_total_time "$STATS_FILE")"

        # Restore sorting decision file state.
        if [[ -n "$backup_file" && -f "$backup_file" ]]; then
            mv "$backup_file" "$forced_dec_file"
        elif [[ "$created_new" == "1" ]]; then
            rm -f "$forced_dec_file"
        fi

        ratio="$(ratio_float "$avg_unopt" "$avg_opt")"
        max_img_ratio="$(read_max_image_ratio_from_last_two "$STATS_FILE")"
        summary_path="$(write_summary_json "$SUMMARY_FILE" "$SCENE" "$STEP" "$EVAL_NUM_RUNS" "$avg_opt" "$avg_unopt" "$ratio" "$max_img_ratio")"

        echo "[Summary] scene=$SCENE step=$STEP"
        echo "          avg Total Time optimized   = $avg_opt"
        echo "          avg Total Time Unoptimized = $avg_unopt"
        echo "          ratio (unoptimized / optimized)     = $ratio"
        echo "          max per-image ratio                 = $max_img_ratio"
        echo "          saved to                            = $summary_path"
    done
done

echo "Done."

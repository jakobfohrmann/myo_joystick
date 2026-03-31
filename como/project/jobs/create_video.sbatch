#!/bin/bash
#SBATCH --job-name=thumb_video
#SBATCH --partition=paula
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# -----------------------------
# Environment
# -----------------------------

module load Mesa/24.1.3-GCCcore-13.3.0
source /software/easybuild/el9/amd_zen3/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
export MUJOCO_GL=osmesa

conda activate thumb-rl || { echo "ERROR: Failed to activate conda environment thumb-rl"; exit 1; }
export PATH="$HOME/.conda/envs/thumb-rl/bin:$PATH"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# -----------------------------
# Paths / defaults
# -----------------------------

PROJECT_DIR="$HOME/como/project"
cd "$PROJECT_DIR" || { echo "ERROR: Project directory not found: $PROJECT_DIR"; exit 1; }

mkdir -p logs

# Auto-select latest model if MODEL_PATH not provided
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH=$(ls -1t logs/thumb_reach_*/final_model.zip 2>/dev/null | head -n 1 || true)
  if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH=$(find logs -maxdepth 2 -name final_model.zip -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -n 1 | awk '{print $2}' || true)
  fi
  if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: Kein final_model.zip gefunden. Erwartet z.B.: logs/thumb_reach_*/final_model.zip"
    find logs -name final_model.zip -print 2>/dev/null || true
    exit 2
  fi
fi

OUTPUT_PATH="${OUTPUT_PATH:-video.mp4}"
N_EPISODES="${N_EPISODES:-1}"
FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
CAMERA="${CAMERA:-cam_thumb_close}"

# Overlay-Test-Defaults
OVERLAY_TRANSFORM="${OVERLAY_TRANSFORM:-rot-90}"      # rot-90|none
OVERLAY_DEBUG_EVERY="${OVERLAY_DEBUG_EVERY:-0}"      # z.B. 50, 0 = aus
RUN_ALL_TRANSFORMS="${RUN_ALL_TRANSFORMS:-0}"        # 1 = alle Modi rendern

echo "=========================================="
echo "[$(date)] Starting Video Generation"
echo "Selected MODEL_PATH: ${MODEL_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Episodes: ${N_EPISODES}"
echo "FPS: ${FPS} | Size: ${WIDTH}x${HEIGHT} | Camera: ${CAMERA}"
echo "Overlay: transform=${OVERLAY_TRANSFORM} | debug_every=${OVERLAY_DEBUG_EVERY} | run_all=${RUN_ALL_TRANSFORMS}"
echo "=========================================="

python - << 'PY'
import myosuite, gymnasium, stable_baselines3, cv2
import envs
print("All imports OK")
print(f"OpenCV version: {cv2.__version__}")
PY

make_outpath_for_mode () {
  local base="$1"
  local mode="$2"
  if [[ "$base" == *.* ]]; then
    local stem="${base%.*}"
    local ext="${base##*.}"
    echo "${stem}_${mode}.${ext}"
  else
    echo "${base}_${mode}"
  fi
}

echo "[$(date)] Starting video generation..."
echo "=========================================="

if [[ "$RUN_ALL_TRANSFORMS" == "1" ]]; then
  MODES=("rot-90" "none")
  for mode in "${MODES[@]}"; do
    OUT="$(make_outpath_for_mode "$OUTPUT_PATH" "$mode")"
    echo "[$(date)] Rendering mode=$mode -> $OUT"

    python models/create_video.py \
      --model "$MODEL_PATH" \
      --output "$OUT" \
      --episodes "$N_EPISODES" \
      --fps "$FPS" \
      --width "$WIDTH" \
      --height "$HEIGHT" \
      --camera "$CAMERA" \
      --overlay-transform "$mode" \
      --overlay-debug-every "$OVERLAY_DEBUG_EVERY"
  done
else
  python models/create_video.py \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    --episodes "$N_EPISODES" \
    --fps "$FPS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --camera "$CAMERA" \
    --overlay-transform "$OVERLAY_TRANSFORM" \
    --overlay-debug-every "$OVERLAY_DEBUG_EVERY"
fi

echo "[$(date)] Video generation finished!"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=fitts_eval
#SBATCH --partition=paula
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Mesa-Modul für headless Rendering laden
module load Mesa/24.1.3-GCCcore-13.3.0

# Conda-Umgebung laden
source /software/easybuild/el9/amd_zen3/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
export MUJOCO_GL=osmesa

# Environment aktivieren
conda activate thumb-rl || {
    echo "ERROR: Failed to activate conda environment thumb-rl"
    exit 1
}

export PATH="$HOME/.conda/envs/thumb-rl/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$HOME/como/project"
mkdir -p logs

echo "=========================================="
echo "[$(date)] Fitts Law Evaluation"
echo "Working Directory: $(pwd)"
echo "Python: $(which python)"
echo "=========================================="

echo "[$(date)] Checking environment..."
python - << 'PY'
import gymnasium
import stable_baselines3
import envs
print("All imports OK")
PY

# Konfiguration: nur MODEL_PATH ist Pflicht, Rest optional (--export=ALL,VAR=value)
#   MODEL_PATH   = Pfad zum Checkpoint (.zip)
#   OUTPUT_DIR   = optional; Default: <Modell-Ordner>/eval_fitts
#   LOG_DIR      = optional; Default: Modell-Ordner (für final_env.pkl)
#   N_EPISODES   = Default: 300 (bei 100 Paaren ≈ 3 Ep/Paar; mind. 200 für 2 Ep/Paar, 500 für 5 Ep/Paar)
#   N_TARGETS    = optional; Anzahl (Target,Radius)-Paare (Default: alle = 20 Targets × 5 Radien = 100)
#   RADII        = optional; z.B. RADII="0.2 0.25 0.3 0.35 0.4" (Default im Skript)
#   SEED         = optional
#   ID_FORMULA   = shannon (default) oder mackenzie (wie UITB)
#   NO_TRAJ      = "1" = keine Trajektorien speichern
#   NO_WANDB     = "1" = kein WandB
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
LOG_DIR="${LOG_DIR:-}"
N_EPISODES="${N_EPISODES:-300}"
N_TARGETS="${N_TARGETS:-}"
# Default-Radien hier setzen, damit auf dem Cluster immer 5 Radien laufen (nicht abhängig von Script-Standard)
RADII="${RADII:-0.02 0.06 0.2 0.3 0.4}"
SEED="${SEED:-}"
ID_FORMULA="${ID_FORMULA:-shannon}"
NO_TRAJ="${NO_TRAJ:-}"
NO_WANDB="${NO_WANDB:-}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH must be set (path to checkpoint .zip)"
    echo ""
    echo "Beispiele:"
    echo "  sbatch --export=ALL,MODEL_PATH=logs/thumb_reach_XXX/best_model.zip jobs/evaluate_fitts.sbatch"
    echo "  sbatch --export=ALL,MODEL_PATH=logs/run1/best_model.zip,N_EPISODES=200 jobs/evaluate_fitts.sbatch   # 20 Paare, 10 Ep/Paar"
    echo "  sbatch --export=ALL,MODEL_PATH=logs/run1/best_model.zip,N_EPISODES=200,NO_TRAJ=1 jobs/evaluate_fitts.sbatch"
    echo "  sbatch --export=ALL,MODEL_PATH=logs/run1/best_model.zip,ID_FORMULA=mackenzie jobs/evaluate_fitts.sbatch"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    model_dir="$(dirname "$MODEL_PATH")"
    OUTPUT_DIR="${model_dir}/eval_fitts"
fi

echo "[$(date)] Running evaluation..."
echo "  Checkpoint: $MODEL_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "  Episodes:   $N_EPISODES"
echo "  (Target,Radius) pairs: ${N_TARGETS:-all (20×radii)}"
echo "  Radii: ${RADII:-0.02 0.06 0.2 0.3 0.4 (default)}"
echo "  ID formula: $ID_FORMULA"
echo "  Log-dir:    ${LOG_DIR:-<auto>}"
echo "  Seed:       ${SEED:-<none>}"
echo "  Trajectories: $([ -n "$NO_TRAJ" ] && echo "no" || echo "yes")"
echo "  WandB:      $([ -n "$NO_WANDB" ] && echo "no" || echo "yes")"
echo "=========================================="

ARGS=(--model "$MODEL_PATH" --n-episodes "$N_EPISODES" --id-formula "$ID_FORMULA" --radii $RADII)
[[ -n "$N_TARGETS" ]]  && ARGS+=(--n-targets "$N_TARGETS")
[[ -n "$OUTPUT_DIR" ]] && ARGS+=(--output "$OUTPUT_DIR")
[[ -n "$LOG_DIR" ]]    && ARGS+=(--log-dir "$LOG_DIR")
[[ -n "$SEED" ]]       && ARGS+=(--seed "$SEED")
[[ -n "$NO_TRAJ" ]]    && ARGS+=(--no-trajectories)
[[ -n "$NO_WANDB" ]]   && ARGS+=(--no-wandb)

python models/evaluate_fitts.py "${ARGS[@]}"

echo "[$(date)] Evaluation finished!"
echo "Ergebnisse: trials.csv, trajectories/, fitts_summary.json (siehe Skript-Ausgabe für Pfad)"
echo "=========================================="

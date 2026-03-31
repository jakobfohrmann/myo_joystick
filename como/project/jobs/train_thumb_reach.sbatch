#!/bin/bash
#SBATCH --job-name=thumb_reach
#SBATCH --partition=paula
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --export=ALL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Mesa-Modul für headless Rendering laden
module load Mesa/24.1.3-GCCcore-13.3.0

# Conda-Umgebung laden
source /software/easybuild/el9/amd_zen3/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
export MUJOCO_GL=osmesa  # CPU-Rendering

# Environment aktivieren (explizit PATH setzen für Sicherheit)
conda activate thumb-rl || {
    echo "ERROR: Failed to activate conda environment thumb-rl"
    exit 1
}

# Sicherstellen, dass Python aus dem Environment verwendet wird
export PATH="$HOME/.conda/envs/thumb-rl/bin:$PATH"

# PyTorch auf CPU zwingen
export CUDA_VISIBLE_DEVICES=""

# Threading
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Im project Ordner bleiben
cd "$HOME/como/project"
mkdir -p logs

# Optional: gleicher Seed für mehrere Trainings-Jobs (Slurm):
#   sbatch --export=ALL,SEED=42 jobs/train_thumb_reach.sbatch
#   sbatch --export=ALL,SEED=42,WANDB_RUN_NAME=run_b jobs/train_thumb_reach.sbatch
SEED="${SEED:-}"
if [[ -n "$SEED" ]]; then
  export SEED
fi

# KEIN PYTHONPATH für myosuite_dev nötig (ist als Paket installiert)
# project ist schon im Pfad durch sys.path.insert in train_thumb_reach.py

# Debug-Info
echo "=========================================="
echo "[$(date)] Starting Thumb Reach Training (CPU)"
echo "Working Directory: $(pwd)"
echo "Python: $(which python)"
echo "Python Path: $(python -c 'import sys; print(sys.executable)')"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-unset}"
echo "SEED: ${SEED:-<none>}"
echo "=========================================="

# Quick Import-Check
echo "[$(date)] Checking environment..."
python - << 'PY'
import myosuite
import gymnasium
import stable_baselines3
import torch
import envs  # registriert ThumbReach-v0
print("All imports OK")
print(f"PyTorch device: CPU")
print(f"CUDA available: {torch.cuda.is_available()}")
PY

# Checkpoint loading (only if explicitly set via LOAD_MODEL_PATH)
if [[ -n "${LOAD_MODEL_PATH:-}" ]]; then
    echo "[$(date)] Resuming training from checkpoint: $LOAD_MODEL_PATH"
else
    echo "[$(date)] Starting training from scratch (no checkpoint specified)"
fi

# Training starten
echo "[$(date)] Launching trainer..."
python models/train_thumb_reach.py

echo "[$(date)] Training finished!"
echo "Check logs in: logs/"
echo "=========================================="


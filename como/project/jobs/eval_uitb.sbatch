#!/bin/bash -l
#SBATCH --job-name=uitb_eval
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=uitb_eval_%j.out
#SBATCH --error=uitb_eval_%j.err

set -euo pipefail

module purge
module load Mesa/24.1.3-GCCcore-13.3.0

source /software/easybuild/el9/amd_zen3/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate thumb-rl-eval

# Ganz wichtig: keine fremden Python-Pfade "reinbluten" lassen
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
# PYTHONPATH wird weiter unten auf $PROJECT gesetzt, damit envs + uitb_evaluate gefunden werden

echo "=== SANITY ==="
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "PATH=$PATH"
echo "PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"

# Conda-Python explizit nutzen (auf Compute-Nodes steht oft Modul-Python zuerst im PATH)
SANITY_PYTHON="${CONDA_PREFIX}/bin/python"
which "$SANITY_PYTHON" || true
"$SANITY_PYTHON" -V || true

"$SANITY_PYTHON" - <<'PY'
import sys
print("sys.executable:", sys.executable)
print("sys.path[:5]:", sys.path[:5])

try:
    import numpy as np
    print("numpy version:", np.__version__)
    print("numpy file   :", np.__file__)
    # numpy._core existiert nur ab NumPy 2.x; stable_baselines3/gymnasium brauchen es
    if int(np.__version__.split(".")[0]) < 2:
        print("WARN: NumPy < 2. Bitte im Env ausfuehren: pip install 'numpy>=2'")
except Exception as e:
    print("numpy import FAILED:", repr(e))
PY
echo "=== /SANITY ==="

export MUJOCO_GL=osmesa
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""

PROJECT=/home/sc.uni-leipzig.de/vt89zidu/como/project
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

# LOGDIR muss auf den Trainingsrun zeigen
if [[ -z "${LOGDIR:-}" ]]; then
  echo "ERROR: LOGDIR nicht gesetzt. Beispiel:"
  echo "  sbatch --export=ALL,LOGDIR=logs/thumb_reach_YYYY-MM-DD_HH-MM-SS,EPISODES=50,MODEL_CHOICE=best eval_uitb.sbatch"
  exit 1
fi

# Modellwahl: best oder final (wenn nur best_model.zip existiert, automatisch best nutzen)
MODEL_CHOICE="${MODEL_CHOICE:-final}"   # best|final

if [[ "$MODEL_CHOICE" == "best" ]]; then
  MODEL_BASENAME="best_model"
else
  MODEL_BASENAME="final_model"
fi

# Checks
if [[ ! -f "$LOGDIR/env_config.json" ]]; then
  echo "ERROR: env_config.json fehlt in $LOGDIR"
  ls -la "$LOGDIR" | head -n 50
  exit 1
fi

# Wenn gewähltes Modell fehlt, aber best_model.zip existiert: automatisch best nutzen
if [[ ! -f "$LOGDIR/${MODEL_BASENAME}.zip" ]]; then
  if [[ -f "$LOGDIR/best_model.zip" ]]; then
    echo "Hinweis: ${MODEL_BASENAME}.zip fehlt, nutze best_model.zip"
    MODEL_CHOICE=best
    MODEL_BASENAME="best_model"
  else
    echo "ERROR: Weder final_model.zip noch best_model.zip in $LOGDIR"
    ls -la "$LOGDIR" | egrep 'model\.zip|_model\.zip' || true
    exit 1
  fi
fi

# --- final_model.zip nur als Alias setzen, wenn wir "best" nutzen ---
if [[ "$MODEL_CHOICE" == "best" ]]; then
  rm -f "$LOGDIR/final_model.zip"
  ln -s "best_model.zip" "$LOGDIR/final_model.zip"
else
  # Wenn final gewählt ist, NIEMALS final_model.zip auf final_model.zip linken.
  # Optional: kaputten Self-Link aufräumen
  if [[ -L "$LOGDIR/final_model.zip" ]] && [[ "$(readlink "$LOGDIR/final_model.zip")" == "final_model.zip" ]]; then
    rm -f "$LOGDIR/final_model.zip"
  fi
fi


PYTHON="$CONDA_PREFIX/bin/python"

echo "=== Runtime sanity ==="
echo "python: $($PYTHON -V)  ($PYTHON)"
$PYTHON -c "
import numpy as np
print('numpy', np.__version__, np.__file__)
if int(np.__version__.split('.')[0]) < 2:
    raise SystemExit('NumPy 2.x erforderlich (pip install numpy>=2)')
import importlib
importlib.import_module('numpy._core.numeric')
print('numpy._core.numeric OK')
"
$PYTHON -c "import stable_baselines3 as sb3; print('sb3', sb3.__version__)"
$PYTHON -c "import envs; print('envs OK')"
$PYTHON -c "from uitb_evaluate.trajectory_data import TrajectoryData_RL; from uitb_evaluate.evaluate_main import trajectoryplot; print('uitb_evaluate OK')"

EPISODES="${EPISODES:-50}"

echo "=== Start eval ==="
echo "LOGDIR=$LOGDIR"
echo "MODEL_CHOICE=$MODEL_CHOICE (using $MODEL_BASENAME.zip)"
echo "EPISODES=$EPISODES"

$PYTHON -u scripts/uitb_rollout_and_plot.py --logdir "$LOGDIR" --episodes "$EPISODES"

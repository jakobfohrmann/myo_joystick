# HPC Cluster Setup - Thumb Reach Training

## Schritt 1: Conda initialisieren

```bash
# Auf dem Login-Node einloggen
ssh <dein-uni-kürzel>@login01.sc.uni-leipzig.de

# Conda initialisieren (Anaconda3 ist bereits auf dem Cluster installiert)
source /software/easybuild/el9/amd_zen3/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
```

## Schritt 2: Conda-Environment erstellen

```bash
# Neues Environment für ThumbReach erstellen
conda create -n thumb-rl python=3.11 -y
conda activate thumb-rl

# Basis-Dependencies installieren
conda install ffmpeg -y

# myosuite_dev als Paket installieren (damit myosuite importierbar ist)
cd ~/myosuite_dev
pip install -e .

# Stable-Baselines3 + PyTorch (CPU-Version) + Dependencies
pip install stable-baselines3[extra] wandb gymnasium torch --index-url https://download.pytorch.org/whl/cpu
```

## Schritt 3: Job abschicken

```bash
# In den project Ordner wechseln
cd ~/como/project

# Job abschicken
sbatch jobs/train_thumb_reach.sbatch

# Status prüfen
squeue -u $USER
```

## Schritt 4: Logs überwachen

```bash
# Logs anschauen (sobald Job läuft)
tail -f logs/thumb_reach_*.out
tail -f logs/thumb_reach_*.err

# Oder alle Logs anzeigen
ls -lh logs/
```

## Nützliche Befehle

```bash
# Job abbrechen
scancel <JOBID>

# Job-Status detailliert anzeigen
scontrol show job <JOBID>

# Alle deine Jobs anzeigen
squeue -u $USER

# Logs nach Fehlern durchsuchen
grep -i error logs/thumb_reach_*.err
```

## Troubleshooting

### Falls Imports fehlschlagen:
```bash
# Environment prüfen
module load Mesa/24.1.3-GCCcore-13.3.0
export MUJOCO_GL=osmesa
source /software/easybuild/el9/amd_zen3/all/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate thumb-rl
python -c "import myosuite; print('OK')"
python -c "import stable_baselines3; print('OK')"
python -c "import gymnasium; print('OK')"
```

### Falls Job nicht startet:
```bash
# Prüfe ob genug Ressourcen verfügbar sind
sinfo -p paula

# Prüfe Job-Details
scontrol show job <JOBID>
```


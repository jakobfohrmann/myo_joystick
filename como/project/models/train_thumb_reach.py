"""Main Training Script für Thumb Reach Task

Einfaches Training: Daumen bewegt sich zu Zielkoordinaten
"""
import os
import shutil
import sys
from datetime import datetime
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

# Importiere Custom Components
# Füge project Ordner zum Python-Pfad hinzu
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.trainer import SimpleTrainer
from models.callbacks import WandBEvalCallback, CheckpointCallback, CurriculumCallback, RewardComponentsCallback
from models.joystick_tracker_callback import JoystickTrackerCallback
import models.callbacks as callbacks
callbacks_src = callbacks.__file__


# Environment Registration (muss vor gym.make() importiert werden)
import envs  # Registriert ThumbReach-v0

# Konfiguration
ENV_NAME = "ThumbReach-v0"
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# WandB Run-Name: Kann über Umgebungsvariable gesetzt werden
_env_run_name = os.environ.get("WANDB_RUN_NAME") or os.environ.get("JOB_NAME")
WANDB_RUN_NAME = _env_run_name or f"thumb_reach_{now}"


def _safe_log_subdir(name: str) -> str:
    """Ordnernamen für logs/ bereinigen (Slurm-Export darf keine Pfadzeichen enthalten)."""
    for c in '<>:"/\\|?\n\r\x00':
        name = name.replace(c, "_")
    name = name.strip().strip(".")
    return name if name else f"thumb_reach_{now}"


# Log-Ordner: wie WandB-Run, wenn NAME per Env gesetzt; sonst wie bisher mit Zeitstempel
_log_subdir = _safe_log_subdir(_env_run_name) if _env_run_name else f"thumb_reach_{now}"
LOG_DIR = os.path.join("logs", _log_subdir)
CALLBACKS_FILE = os.path.join(os.path.dirname(__file__), "callbacks.py")
THUMB_REACH_ENV_FILE = os.path.join(project_root, "envs", "thumb_reach.py")
# Wird in envs/__init__.py gesetzt (Pfad zur aktuell verwendeten XML)
ENV_XML_PATH = getattr(envs, "model_path", None)

# Reproduzierbarkeit: gleicher SEED für mehrere Jobs (Slurm: --export=ALL,SEED=42)
_SEED_ENV = os.environ.get("SEED", "").strip()
TRAIN_SEED = int(_SEED_ENV) if _SEED_ENV else None
N_TRAIN_ENVS = 4
# Eval-Env bekommt eigenen Offset, damit er sich nicht mit Train-Rängen 0..N-1 überschneidet
EVAL_ENV_SEED_RANK = 10_000

# Environment Config - Targets (normalisiert [-1, 1])
# Default Werte -> wird in CurriculumCallback überschrieben
env_config = {
    "target_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
    },
    "early_term_steps": 250,  # Early termination nach 250 Steps (5 Sekunden bei 50 Hz) wenn Ziel nicht erreicht
    "weighted_reward_keys": {
        "reach": 1.0,      # negative Distanz
        "bonus": 4.0,      # Hit-Bonus: +1.0 * 4.0 = +4.0 (wenn Treffer)
        "penalty": 50.0,   # Early Term Strafe: -1.0 * 4.0 = -4.0 (bei Timeout ohne Treffer)
    },
}

# Model Config (Hyperparameter)
model_config = {
    "learning_rate": 3e-4,
    "n_steps": 2048,      # Steps pro Rollout
    "batch_size": 64,     # Batch Size
    "n_epochs": 10,       # PPO Epochs
    "gamma": 0.99,        # Discount Factor
    "gae_lambda": 0.95,   # GAE Lambda
    "clip_range": 0.2,    # PPO Clip Range
    "ent_coef": 0.01,     # Entropy Coefficient
    "vf_coef": 0.5,       # Value Function Coefficient
    "max_grad_norm": 0.5, # Gradient Clipping
    "policy_kwargs": {
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],  # Netzwerk-Architektur
    },
}

if TRAIN_SEED is not None:
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(TRAIN_SEED)
    model_config["seed"] = TRAIN_SEED


def make_env_factory(rank: int):
    """Factory für DummyVecEnv: rank unterscheidet parallele Env-Seeds (Basis = TRAIN_SEED)."""

    def _init():
        kwargs = dict(env_config)
        if TRAIN_SEED is not None:
            kwargs["seed"] = TRAIN_SEED + rank
        env = gym.make(ENV_NAME, **kwargs)
        # info_keywords: Monitor nimmt diese aus info (Top-Level) und schreibt sie in info["episode"] → ep_info_buffer
        env = Monitor(
            env,
            LOG_DIR,
            info_keywords=("solved", "success", "target_idx", "reward_reach", "reward_bonus", "reward_dense", "reward_sparse", "reach_dist"),
        )
        return env

    return _init


# Vectorized Environment (4 parallele Environments)
envs = DummyVecEnv([make_env_factory(i) for i in range(N_TRAIN_ENVS)])
envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Evaluation Environment (separat, nicht normalisiert für Rewards)
eval_env = DummyVecEnv([make_env_factory(EVAL_ENV_SEED_RANK)])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
eval_env.training = False  # Wichtig für Evaluation!

# callback
callbacks = []

# 0. Curriculum Learning (muss zuerst kommen, damit Phase-Wechsel funktioniert)
curriculum_callback = CurriculumCallback(
    verbose=1,
)
callbacks.append(curriculum_callback)


# 2. Evaluation + WandB + Best Model (kombiniert)
eval_callback = WandBEvalCallback(
    eval_env=eval_env,
    wandb_project="thumb_reach",
    wandb_run_name=WANDB_RUN_NAME,  # Verwendet Umgebungsvariable oder automatischen Namen
    log_freq=10_000,  # Alle 10k Steps evaluieren
    n_eval_episodes=10,
    best_model_save_path=LOG_DIR,
    curriculum_callback=curriculum_callback,  # Curriculum-Logs (Target/Hold) gehen über WandB-Callback
    verbose=1,
)
callbacks.append(eval_callback)

# 3. Joystick Position Tracker (für Heatmap-Visualisierung)
joystick_tracker = JoystickTrackerCallback(
    save_path=LOG_DIR,
    verbose=1,
    save_freq=50_000,  # Speichere alle 50k Steps (um Speicher zu sparen)
)
callbacks.append(joystick_tracker)

# 4. Checkpoints (optional, für einfaches Setup)
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,  # Alle 50k Steps Checkpoint
    save_path=LOG_DIR,
    verbose=1,
    generate_heatmap=True,
)
callbacks.append(checkpoint_callback)

# Trainer
# Load from checkpoint if LOAD_MODEL_PATH environment variable is set
load_model_path = os.environ.get("LOAD_MODEL_PATH")
trainer = SimpleTrainer(
    envs=envs,
    env_config=env_config,
    load_model_path=load_model_path, 
    log_dir=LOG_DIR,
    model_config=model_config,
    callbacks=callbacks,
    timesteps=5_000_000,
)

#  Training
if __name__ == "__main__":
    # Log-Directory erstellen und Skript kopieren (für Reproduzierbarkeit)
    os.makedirs(LOG_DIR, exist_ok=True)
    if TRAIN_SEED is not None:
        with open(os.path.join(LOG_DIR, "train_seed.txt"), "w", encoding="utf-8") as f:
            f.write(f"{TRAIN_SEED}\n")
    shutil.copy(__file__, LOG_DIR)
    # Copy XML file to LOG_DIR
    xml_src = getattr(envs, "model_path", None)
    if xml_src and os.path.isfile(xml_src):
        print(f"Copying XML file to {LOG_DIR}")
        shutil.copy(xml_src, LOG_DIR)
    # Copy checkpoint_callback.py to LOG_DIR
    checkpoint_callback_src = getattr(checkpoint_callback, "save_path", None)
    if checkpoint_callback_src and os.path.isfile(checkpoint_callback_src):
        print(f"Copying checkpoint_callback.py to {LOG_DIR}")
        shutil.copy(checkpoint_callback_src, LOG_DIR)
    
    # Auch Callbacks und verwendetes XML sichern
    if os.path.isfile(CALLBACKS_FILE):
        shutil.copy(CALLBACKS_FILE, LOG_DIR)
    else:
        print(f"Warning: Callbacks file not found at {CALLBACKS_FILE}")

    if os.path.isfile(THUMB_REACH_ENV_FILE):
        shutil.copy(THUMB_REACH_ENV_FILE, LOG_DIR)
    else:
        print(f"Warning: env thumb_reach.py not found at {THUMB_REACH_ENV_FILE}")

    if ENV_XML_PATH and os.path.isfile(ENV_XML_PATH):
        shutil.copy(ENV_XML_PATH, LOG_DIR)
    else:
        print(f"Warning: XML model file not found (expected at {ENV_XML_PATH})")
    
    print("=" * 60)
    print("Starting Thumb Reach Training")
    print("=" * 60)
    print(f"WandB Run Name: {WANDB_RUN_NAME}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"Environment: {ENV_NAME}")
    print(f"Total Timesteps: {trainer.timesteps:,}")
    print(f"Parallel Environments: {N_TRAIN_ENVS}")
    print(f"Random seed: {TRAIN_SEED if TRAIN_SEED is not None else '<none (set SEED env)>'}")
    print(f"Hyperparameters:")
    print(f"   - Learning Rate: {model_config['learning_rate']}")
    print(f"   - Batch Size: {model_config['batch_size']}")
    print(f"   - Network: {model_config['policy_kwargs']['net_arch']}")
    print("=" * 60)
    
    # Training starten
    trainer.train()
    
    # Finales Model speichern
    trainer.save()

    from scripts.uitb_rollout_and_plot import rollout_to_uitb_logs, make_uitb_plots

    sub = rollout_to_uitb_logs(LOG_DIR, out_subdir="uitb_eval", n_episodes=20)
    plots_dir = make_uitb_plots(LOG_DIR, sub)
    print("uitb plots:", plots_dir)

    
    print("=" * 60)
    print("Training completed!")
    print(f"Results saved to: {LOG_DIR}")
    print("=" * 60)


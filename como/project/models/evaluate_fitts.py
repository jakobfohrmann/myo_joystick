"""
Fitts Law Evaluation Script für Thumb Reach Checkpoint

Lädt ein trainiertes Modell, führt Episoden mit festen Targets durch und speichert:
- trials.csv: pro Episode pair_idx, target_idx, target_radius, D, W, ID, MT, success, Start-/Zielposition
- trajectories/ep_XXX.npz: Zeitreihe t, Position (x,y), reach_dist, hit pro Step
- fitts_summary.json: MT = a + b·ID Regression (a, b, R²)

ID-Formeln (wie in uitb-tools/uitb_evaluate/evaluate_summarystatistics.py und trajectory_data.py):

D = Abstand Start–Ziel, W = 2*Target-Radius. Nur erfolgreiche Trials für Regression; Outlier |z(MT)|>3 werden ausgeschlossen.

Verwendung:
  python models/evaluate_fitts.py --model logs/thumb_reach_XXX/best_model.zip --output eval_fitts/run1
  python models/evaluate_fitts.py --model logs/.../best_model.zip --id-formula mackenzie  # wie UITB
"""
import os
import sys
import argparse
import json
import importlib.util
import re
import numpy as np
import csv
from datetime import datetime
from scipy import stats

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Projekt-Root zum Pfad hinzufügen
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import envs  # Registriert ThumbReach-v0
from models.callbacks import CurriculumCallback

# Schritt-Dauer in Sekunden (frame_skip=10, dt=0.002)
STEP_DT = 0.02


def compute_ID(D, W, formula="shannon"):
    """Index of Difficulty."""
    if W <= 0 or D < 0:
        return 0.0
    if formula == "mackenzie":
        return float(np.log2((D / W) + 1.0))
    return float(np.log2(2.0 * D / W))


def fit_fitts_law(IDs, MTs, outlier_std=None):
    """
    Lineare Regression MT = a + b·ID 
    """
    IDs = np.asarray(IDs, dtype=np.float64).ravel()
    MTs = np.asarray(MTs, dtype=np.float64).ravel()
    if len(IDs) < 2 or len(MTs) < 2 or len(IDs) != len(MTs):
        return None, None, np.nan, None
    if outlier_std is not None and len(MTs) > 2:
        z = np.abs(stats.zscore(MTs, nan_policy="omit"))
        z = np.nan_to_num(z, nan=0.0)
        mask = z <= outlier_std
        if np.sum(mask) < 2:
            mask = np.ones(len(MTs), dtype=bool)
        IDs, MTs = IDs[mask], MTs[mask]
    # stats.linregress benötigt mindestens zwei unterschiedliche x-Werte
    if np.allclose(IDs, IDs[0]):
        return None, None, np.nan, None
    slope, intercept, r_value, p_value, std_err = stats.linregress(IDs, MTs)
    a, b = float(intercept), float(slope)
    y_pred = a + b * IDs
    r2 = r_value ** 2
    return a, b, r2, y_pred


# (obs_dim, action_dim) → (env_id, use_full_action_wrapper, config_id)
# ThumbReach-v0: 13 = Daumen+Handgelenk+Unterarm (Standard), 10/5 = ältere Modelle (mit Pad), 39 = alle Muskeln (mit Wrapper)
SPACES_TO_ENV = {
    (101, 13): ("ThumbReach-v0", False, "ThumbReach-v0"),
    (101, 10): ("ThumbReach-v0", False, "ThumbReach-v0"),
    (51, 13): ("ThumbReach-v0", False, "ThumbReach-v1"),
    (51, 10): ("ThumbReach-v0", False, "ThumbReach-v1"),
    (51, 5): ("ThumbReach-v0", False, "ThumbReach-v1"),
    (51, 39): ("ThumbReach-v0", True, "ThumbReach-v1"),
}


def get_env_config(env_id="ThumbReach-v0"):
    """Environment-Config wie im Training (v0 vs v1).TODO: sobald copy funktion (xml, env, config et) bereit, übernehme aus kopiertem Ordner"""
    base = {
        "target_range": {"x": (0.0, 1.0), "y": (0.0, 1.0)},
        "weighted_reward_keys": {"reach": 1.0, "bonus": 4.0},
    }
    if env_id == "ThumbReach-v1":
        base["early_term_steps"] = 100
        base["early_term_threshold"] = 0.15
        base["far_th"] = 0.2
    else:
        base["early_term_steps"] = 250
    return base


def _resolve_snapshot_xml(log_dir):
    """Findet ein XML im Log-Ordner; bevorzugt controller_with_hand.xml."""
    if not log_dir:
        return None
    preferred = os.path.join(log_dir, "controller_with_hand.xml")
    if os.path.isfile(preferred):
        return preferred
    try:
        xmls = [f for f in os.listdir(log_dir) if f.lower().endswith(".xml")]
    except Exception:
        return None
    if len(xmls) == 1:
        return os.path.join(log_dir, xmls[0])
    return None


def _materialize_eval_xml(xml_path, log_dir):
    """Erzeugt ein eval-taugliches XML mit aufgelösten Include-Pfaden.
    Snapshot-XMLs wurden aus project/xml nach logs/... kopiert; relative Includes
    (z.B. ../../myo_sim_repo/...) brechen dort. Wir mappen fehlende relative
    Include-Dateien auf den ursprünglichen Bezugspfad project/xml.
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    xml_dir = os.path.dirname(xml_path)
    xml_ref_dir = os.path.join(project_root, "xml")
    include_pattern = re.compile(r'(<include\s+[^>]*file=")([^"]+)(")')
    changed = False

    def _replace(match):
        nonlocal changed
        prefix, rel, suffix = match.groups()
        # absolute include bleibt unverändert
        if os.path.isabs(rel):
            return match.group(0)
        candidate_from_snapshot = os.path.normpath(os.path.join(xml_dir, rel))
        if os.path.isfile(candidate_from_snapshot):
            return match.group(0)
        candidate_from_project_xml = os.path.normpath(os.path.join(xml_ref_dir, rel))
        if os.path.isfile(candidate_from_project_xml):
            changed = True
            return f'{prefix}{candidate_from_project_xml}{suffix}'
        return match.group(0)

    patched = include_pattern.sub(_replace, xml_text)

    # Manche Snapshot-Setups enthalten einen kaputten Mesh-Pfad mit doppeltem
    # ".../hand/myo_sim/meshes/...". Korrigiere diesen robust auf
    # ".../myo_sim/meshes/..." falls vorhanden.
    broken_token = "/myo_sim/hand/myo_sim/meshes/"
    fixed_token = "/myo_sim/meshes/"
    if broken_token in patched:
        patched = patched.replace(broken_token, fixed_token)
        changed = True
    if not changed:
        return xml_path

    xml_out_dir = os.path.join(project_root, "xml")
    os.makedirs(xml_out_dir, exist_ok=True)
    run_tag = os.path.basename(os.path.normpath(log_dir)) if log_dir else "run"
    out_xml = os.path.join(xml_out_dir, f"_eval_resolved_{run_tag}_controller_with_hand.xml")
    with open(out_xml, "w", encoding="utf-8") as f:
        f.write(patched)
    return out_xml


def _load_snapshot_env_class(log_dir):
    """Lädt ThumbReachEnvV0 aus <log_dir>/thumb_reach.py, falls vorhanden."""
    if not log_dir:
        return None
    env_py = os.path.join(log_dir, "thumb_reach.py")
    if not os.path.isfile(env_py):
        return None
    try:
        mod_name = f"thumb_reach_snapshot_{abs(hash(env_py))}"
        spec = importlib.util.spec_from_file_location(mod_name, env_py)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Konnte Import-Spec für {env_py} nicht erstellen.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env_cls = getattr(module, "ThumbReachEnvV0", None)
        if env_cls is None:
            raise RuntimeError(f"Snapshot-Datei {env_py} enthält keine Klasse ThumbReachEnvV0.")
        return env_cls
    except Exception as e:
        raise RuntimeError(f"Snapshot-Env aus {env_py} konnte nicht geladen werden: {e}") from e


def make_env(env_id, env_config, seed=None, log_dir=None):
    """Einzelnes Environment ohne Monitor (für Eval).
    Verwendet strikt Snapshot-Dateien aus dem Log-Ordner (thumb_reach.py + XML).
    """
    env_kwargs = dict(env_config)
    if seed is not None:
        env_kwargs["seed"] = seed

    if not log_dir:
        raise ValueError("Für Evaluation ist ein Log-Ordner erforderlich (mit Snapshot-Dateien).")

    env_py = os.path.join(log_dir, "thumb_reach.py")
    if not os.path.isfile(env_py):
        raise FileNotFoundError(
            f"Snapshot-Env fehlt: {env_py}. "
            "Bitte trainierten Run-Ordner mit gesichertem thumb_reach.py verwenden."
        )
    snapshot_env_cls = _load_snapshot_env_class(log_dir)
    snapshot_xml = _resolve_snapshot_xml(log_dir)
    if snapshot_xml is None:
        raise FileNotFoundError(
            f"Snapshot-XML im Log-Ordner nicht gefunden: {log_dir}. "
            "Erwartet z.B. controller_with_hand.xml oder genau eine .xml-Datei."
        )

    resolved_xml = _materialize_eval_xml(snapshot_xml, log_dir)
    env_kwargs["model_path"] = resolved_xml
    print(f"Hinweis: Verwende Snapshot-Env aus {log_dir} (XML: {os.path.basename(resolved_xml)})")
    return snapshot_env_cls(**env_kwargs)


class FullActionSpaceWrapper(gym.Wrapper):
    """Wrapper, der nur action_space auf (39,) setzt; step(action) wird durchgereicht.
    ThumbReach-v1.step() akzeptiert bereits 39-dim Actions (ruft super().step(action) auf).
    """
    def __init__(self, env, action_dim=39):
        super().__init__(env)
        self._action_dim = action_dim
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )


class ActionPadWrapper(gym.Wrapper):
    """Für Modelle mit weniger Aktionen als das Env: Env erwartet mehr (z.B. 13).
    Meldet action_space (model_action_dim,); in step() wird auf env_action_dim gepaddet (Rest = 0).
    """
    def __init__(self, env, model_action_dim=5, env_action_dim=10):
        super().__init__(env)
        self._pad_to = env_action_dim
        self._model_dim = model_action_dim
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(model_action_dim,), dtype=np.float32
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.shape[0] != self._model_dim:
            action = np.resize(action, self._model_dim)
        full = np.zeros(self._pad_to, dtype=np.float32)
        full[: self._model_dim] = np.clip(action, 0.0, 1.0)
        return self.env.step(full)


class ActionCropWrapper(gym.Wrapper):
    """Für Modelle mit mehr Aktionen als das Env erwartet.
    Meldet action_space (model_action_dim,); in step() werden nur die ersten env_action_dim genutzt.
    """
    def __init__(self, env, model_action_dim=13, env_action_dim=5):
        super().__init__(env)
        self._env_dim = env_action_dim
        self._model_dim = model_action_dim
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(model_action_dim,), dtype=np.float32
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.shape[0] != self._model_dim:
            action = np.resize(action, self._model_dim)
        cropped = np.clip(action[: self._env_dim], 0.0, 1.0)
        return self.env.step(cropped)


class ObsSliceWrapper(gym.Wrapper):
    """Schneidet Observation auf die ersten obs_dim Dimensionen (Modell erwartet weniger als Env liefert, z.B. 51 vs 61)."""
    def __init__(self, env, obs_dim):
        super().__init__(env)
        self._obs_dim = int(obs_dim)
        # Gleicher Typ wie typische normierte Obs (VecNormalize)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32
        )

    def _slice(self, obs):
        if getattr(obs, "shape", None) is not None and obs.shape[-1] > self._obs_dim:
            return np.asarray(obs[..., :self._obs_dim], dtype=np.float32)
        return obs

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        info = (out[1] if isinstance(out, tuple) and len(out) > 1 else {})
        return (self._slice(obs), info) if isinstance(out, tuple) else self._slice(obs)

    def step(self, action):
        obs, *rest = self.env.step(action)
        return (self._slice(obs), *rest)


class ActionSpaceAlignWrapper(gym.Wrapper):
    """Meldet action_space wie das Modell (z.B. [-1,1]); transformiert Aktionen in Env-Bereich (z.B. [0,1]) vor step()."""
    def __init__(self, env, target_action_space):
        super().__init__(env)
        self._target = target_action_space
        self.action_space = target_action_space
        # Env erwartet typisch [0, 1]
        esp = env.action_space
        self._env_low = np.broadcast_to(np.asarray(esp.low, dtype=np.float32), esp.shape)
        self._env_high = np.broadcast_to(np.asarray(esp.high, dtype=np.float32), esp.shape)
        self._model_low = np.broadcast_to(np.asarray(target_action_space.low, dtype=np.float32), target_action_space.shape)
        self._model_high = np.broadcast_to(np.asarray(target_action_space.high, dtype=np.float32), target_action_space.shape)

    def step(self, action):
        # action ist im Modell-Bereich; skaliere in Env-Bereich
        action = np.asarray(action, dtype=np.float32)
        t = (action - self._model_low) / (self._model_high - self._model_low + 1e-8)
        t = np.clip(t, 0.0, 1.0)
        action_env = self._env_low + t * (self._env_high - self._env_low)
        return self.env.step(action_env)


def load_env_and_model(model_path, log_dir=None, seed=None):
    """
    Lädt Modell und erstellt Eval-Env (1 Env; Training kann mit n_envs=4 gelaufen sein).
    Liest zuerst Obs-/Action-Dimensionen aus dem gespeicherten Modell und wählt
    passendes Env (v0/v1, ggf. mit Full-Action-Wrapper für (51, 39)).
    PPO.load(path, env=env) erlaubt anderes n_envs als beim Training (z.B. 1 für Eval).
    """
    # Modell ohne Env laden, nur um Spaces zu lesen
    model_temp = PPO.load(model_path)
    obs_dim = model_temp.observation_space.shape[0]
    act_dim = model_temp.action_space.shape[0]
    model_action_space = model_temp.action_space
    del model_temp

    key = (obs_dim, act_dim)
    entry = SPACES_TO_ENV.get(key)
    if entry is None:
        raise ValueError(
            f"Modell hat obs_dim={obs_dim}, act_dim={act_dim}. "
            f"Unterstützt: (101, 13), (51, 13), (101, 10), (51, 10), (51, 5), (51, 39) – ThumbReach-v0 mit passenden Wrappers."
        )
    env_id, use_full_action_wrapper, config_id = entry
    env_config = get_env_config(config_id)

    def _make():
        e = make_env(env_id, env_config, seed=seed, log_dir=(log_dir or os.path.dirname(os.path.abspath(model_path))))
        if use_full_action_wrapper:
            e = FullActionSpaceWrapper(e, action_dim=act_dim)
        else:
            env_act_dim = int(e.action_space.shape[0])
            if act_dim < env_act_dim:
                e = ActionPadWrapper(e, model_action_dim=act_dim, env_action_dim=env_act_dim)
            elif act_dim > env_act_dim:
                e = ActionCropWrapper(e, model_action_dim=act_dim, env_action_dim=env_act_dim)
        # Env kann mehr Obs liefern als Modell (z.B. 61 vs 51 bei anderer Env-Version/XML)
        if e.observation_space.shape[0] != obs_dim:
            e = ObsSliceWrapper(e, obs_dim=obs_dim)
        # Action-Space-Bereich anpassen (Modell z.B. [-1,1], Env [0,1])
        esp = e.action_space
        same_shape = esp.shape == model_action_space.shape
        if (not same_shape or np.any(esp.low != model_action_space.low) or np.any(esp.high != model_action_space.high)):
            e = ActionSpaceAlignWrapper(e, model_action_space)
        return e

    env = DummyVecEnv([_make])

    env_dir = log_dir or os.path.dirname(os.path.abspath(model_path))
    env_pkl = os.path.join(env_dir, "final_env.pkl")
    if os.path.isfile(env_pkl):
        try:
            env = VecNormalize.load(env_pkl, env)
            env.training = False
            env.norm_reward = False
        except Exception as e:
            print(f"Hinweis: VecNormalize aus {env_pkl} konnte nicht geladen werden: {e}")
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            env.training = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        env.training = False
        print("Hinweis: final_env.pkl nicht gefunden – Eval-Env nutzt neue Normalisierung.")

    # Mit env laden, damit anderes n_envs (1 statt 4) erlaubt ist; set_env(env) würde 1==4 verlangen
    model = PPO.load(model_path, env=env)
    return env, model, env_config


def get_unwrapped(env):
    """Einzelnes unwrapped Env aus DummyVecEnv."""
    e = env.envs[0]
    while hasattr(e, "env"):
        e = e.env
    return e


def set_target_on_unwrapped(unwrapped, target_pos):
    """Setzt Target auf unwrapped Env (VecNormalize leitet options in reset() nicht weiter)."""
    unwrapped.target_pos = np.asarray(target_pos, dtype=np.float64)
    tid = getattr(unwrapped, "target_site_id", None)
    if tid is not None:
        target_3d = np.array([unwrapped.target_pos[0], unwrapped.target_pos[1], 0.0])
        unwrapped.sim.model.site_pos[tid] = target_3d
        if hasattr(unwrapped, "sim_obsd") and unwrapped.sim_obsd is not None:
            unwrapped.sim_obsd.model.site_pos[tid] = target_3d
    if hasattr(unwrapped, "robot") and hasattr(unwrapped.robot, "sync_sims"):
        unwrapped.robot.sync_sims(unwrapped.sim, unwrapped.sim_obsd)


def _trim_vecenv_autoreset_tail(pos_list, t_list, reach_dist_list, hit_list, start_pos):
    """
    VecEnv/Gymnasium: Nach terminated kann step() schon den Reset-Zustand liefern.
    Der letzte gespeicherte Joystick-Punkt ist dann oft wieder nahe Start (0,0),
    obwohl die Episode am Ziel endete — erzeugt geschlossene „Blüten“ im Plot.
    Entfernt genau einen solchen Schwanz-Endpunkt per Heuristik.
    """
    if len(pos_list) < 2:
        return
    last = np.asarray(pos_list[-1], dtype=np.float64).ravel()[:2]
    prev = np.asarray(pos_list[-2], dtype=np.float64).ravel()[:2]
    start = np.asarray(start_pos, dtype=np.float64).ravel()[:2]
    d_ls = float(np.linalg.norm(last - start))
    d_ps = float(np.linalg.norm(prev - start))
    jump = float(np.linalg.norm(last - prev))
    # Großer Sprung zurück Richtung Start, letzter Punkt nahe Start, vorheriger weiter weg
    if jump >= 0.06 and d_ls <= 0.2 and d_ps > d_ls + 0.04:
        pos_list.pop()
        t_list.pop()
        reach_dist_list.pop()
        hit_list.pop()


def _build_obs_after_set_target(env, unwrapped, obs_dim):
    """
    Baut die Observation neu, nachdem Target und Radius auf dem Env gesetzt wurden.
    So sieht die Policy beim ersten Step dasselbe Ziel wie im Env (wie im Training).
    Verwendet dieselbe obs_keys-Reihenfolge und VecNormalize wie beim Training.
    """
    obs_keys = getattr(unwrapped, "obs_keys", getattr(unwrapped, "DEFAULT_OBS_KEYS"))
    obs_dict = unwrapped.get_obs_dict(unwrapped.sim)
    raw = np.concatenate([np.ravel(obs_dict[k]) for k in obs_keys], axis=0).astype(np.float32)
    if raw.size > obs_dim:
        raw = raw[:obs_dim]
    # Gleiche Normalisierung wie VecNormalize (training=False): (obs - mean) / sqrt(var + eps), clip
    if hasattr(env, "obs_rms") and env.obs_rms is not None:
        eps = getattr(env, "epsilon", 1e-8)
        clip = getattr(env, "clip_obs", 10.0)
        normalized = (raw - env.obs_rms.mean) / np.sqrt(env.obs_rms.var + eps)
        raw = np.clip(normalized, -clip, clip).astype(np.float32)
    return raw.ravel()


def run_fitts_evaluation(
    model_path,
    output_dir,
    n_episodes=100,
    n_targets=None,
    radii=None,
    seed=None,
    log_dir=None,
    save_trajectories=True,
    use_wandb=True,
    id_formula="shannon",
    trim_autoreset_tail=True,
):
    """
    Fitts-Evaluation mit gleicher Logik wie Training:

    - Observation Space: Gleiche obs_keys (qpos, qvel, act, joystick_2d, joystick_angles, target_pos, reach_err),
      gleiche Reihenfolge, VecNormalize aus final_env.pkl (gleiche Normalisierung). Nach set_target wird die
      erste Obs neu gebaut, damit die Policy sofort das richtige Ziel sieht (wie beim Training via Callback).

    - Action Space: ThumbReach-v0 hat 13 Muskeln (Daumen + Handgelenk + Unterarm) [0,1]; (51,10)/(51,5)/(51,39) mit Pad/Wrapper; bei abweichendem Modell-Bereich (z.B. [-1,1])
      wird ActionSpaceAlignWrapper verwendet.

    - Hit/Success: Env nutzt reach_dist < self.target_radius (get_reward_dict); wir setzen
      unwrapped.target_radius nach reset(). Am Episode-Ende wird success aus info["success"]
      übernommen (Env-Entscheidung), damit CSV und [DONE-CHECK] has_hit immer übereinstimmen.

    Pro Episode: (Target, Radius)-Paar zyklisch; Paare = len(fixed_targets) × len(radii).
    radii: Liste der Target-Radien (Default z.B. [0.2, 0.25, 0.3, 0.35, 0.4] oder per --radii).
    id_formula: "shannon" oder "mackenzie".
    trim_autoreset_tail: Wenn True, wird der letzte Trajektorienpunkt entfernt, wenn er durch VecEnv-Auto-Reset
        nahe dem Start liegt (nur für sauberere Plots). Beeinflusst NICHT trials.csv, MT, success, Fitts-Regression —
        nur die Arrays in trajectories/ep_*.npz (und ggf. len(position) < n_steps).
    """
    os.makedirs(output_dir, exist_ok=True)
    trajectories_dir = os.path.join(output_dir, "trajectories")
    if save_trajectories:
        os.makedirs(trajectories_dir, exist_ok=True)

    env, model, env_config = load_env_and_model(model_path, log_dir=log_dir, seed=seed)
    unwrapped = get_unwrapped(env)
    # Gleiche Target-Liste wie im Training (CurriculumCallback: 20 Targets)
    fixed_targets = list(CurriculumCallback.FIXED_TARGETS)
    if radii is None:
        radii = [0.02, 0.06, 0.1, 0.2, 0.3, 0.4]  # mehrere Radien für Fitts-Varianz
    radii = [float(r) for r in radii]
    # Paare = (Target, Radius): len(fixed_targets) × len(radii)
    pairs = [(ti, r) for ti in range(len(fixed_targets)) for r in radii]
    n_pairs = len(pairs)
    n_targets = min(n_targets, n_pairs) if n_targets is not None else n_pairs

    trials = []

    for episode in range(n_episodes):
        pair_idx = episode % n_targets
        target_idx, target_radius = pairs[pair_idx]
        target_pos = np.array(fixed_targets[target_idx], dtype=np.float64)
        target_radius = float(target_radius)

        # Reset (VecNormalize leitet options nicht weiter, daher Target danach setzen)
        reset_kw = {}
        if episode == 0 and seed is not None:
            reset_kw["seed"] = seed
        out = env.reset(**reset_kw) if reset_kw else env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        info = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
        set_target_on_unwrapped(unwrapped, target_pos)
        # Env-Schwelle für Hit setzen – NACH reset(), da reset()/generate_target() target_radius überschreibt.
        # thumb_reach.get_reward_dict() liest self.target_radius für in_radius = reach_dist < near_th.
        unwrapped.target_radius = target_radius
        if hasattr(unwrapped, "current_target_radius"):
            unwrapped.current_target_radius = target_radius

        # Observation mit korrektem Ziel neu bauen (reset() hatte noch altes Ziel aus generate_target())
        obs = _build_obs_after_set_target(env, unwrapped, env.observation_space.shape[0])

        # Startposition aus erster Observation nach Set-Target
        obs_dict = unwrapped.get_obs_dict(unwrapped.sim)
        start_pos = np.array(obs_dict["joystick_2d"], dtype=np.float64)
        D = float(np.linalg.norm(target_pos - start_pos))
        # W = effektive Zielbreite (Durchmesser = 2*Radius im gleichen normierten Raum)
        W = 2.0 * target_radius
        ID = compute_ID(D, W, formula=id_formula)

        # Trajektorie sammeln
        t_list = []
        pos_list = []
        reach_dist_list = []
        hit_list = []

        done = False
        step = 0
        MT = np.nan
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # VecEnv erwartet Batch (n_envs, action_dim); bei 1 Env sonst actions[0] = Skalar
            if np.ndim(action) == 1:
                action = np.expand_dims(action, axis=0)
            step_out = env.step(action)
            if len(step_out) == 4:
                obs, reward, done, info = step_out
                terminated, truncated = done, False
            else:
                obs, reward, terminated, truncated, info = step_out
            if isinstance(obs, tuple):
                obs = obs[0]
            done = terminated or truncated
            step += 1
            t = step * STEP_DT

            obs_dict = unwrapped.get_obs_dict(unwrapped.sim)
            joystick_2d = np.array(obs_dict["joystick_2d"])
            reach_err = obs_dict["reach_err"]
            reach_dist = float(np.linalg.norm(reach_err))
            hit = reach_dist < target_radius

            t_list.append(t)
            pos_list.append(joystick_2d.copy())
            reach_dist_list.append(reach_dist)
            hit_list.append(hit)

            if hit and not success:
                MT = t
                success = True

        if trim_autoreset_tail:
            _trim_vecenv_autoreset_tail(pos_list, t_list, reach_dist_list, hit_list, start_pos)

        # Success nur bei terminated (Treffer); truncated = Timeout/Early-Term ohne Treffer → success bleibt False
        if terminated:
            info_dict = info[0] if (isinstance(info, (list, tuple)) and len(info) > 0) else info
            if isinstance(info_dict, dict) and "success" in info_dict:
                success = bool(info_dict["success"])
                if success and (MT is None or (isinstance(MT, (float, np.floating)) and np.isnan(MT))):
                    MT = step * STEP_DT  # Movement Time = Zeitpunkt des Terminalschritts

        if not success:
            MT = step * STEP_DT

        trials.append({
            "episode_id": episode,
            "pair_idx": pair_idx,
            "target_idx": target_idx,
            "target_radius": target_radius,
            "target_x": target_pos[0],
            "target_y": target_pos[1],
            "start_x": start_pos[0],
            "start_y": start_pos[1],
            "D": D,
            "W": W,
            "ID": ID,
            "MT": MT,
            "success": success,
            "n_steps": step,
        })

        if save_trajectories:
            np.savez(
                os.path.join(trajectories_dir, f"ep_{episode:04d}.npz"),
                t=np.array(t_list, dtype=np.float64),
                position=np.array(pos_list, dtype=np.float64),
                reach_dist=np.array(reach_dist_list, dtype=np.float64),
                hit=np.array(hit_list, dtype=bool),
                target_pos=target_pos,
                start_pos=start_pos,
                D=D,
                W=W,
                ID=ID,
                MT=MT,
                success=success,
            )

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}  pair={pair_idx} (target={target_idx}, r={target_radius})  MT={MT:.2f}s  success={success}")

    env.close()

    # trials.csv schreiben
    csv_path = os.path.join(output_dir, "trials.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_id", "pair_idx", "target_idx", "target_radius", "target_x", "target_y",
                "start_x", "start_y", "D", "W", "ID", "MT", "success", "n_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(trials)

    print(f"Gespeichert: {csv_path}")
    if save_trajectories:
        print(f"Trajektorien: {trajectories_dir}/")

    # Fitts Law Regression (wie steering_law_calculations: MT = a + b·ID, R²)
    success_trials = [t for t in trials if t["success"]]
    if len(success_trials) >= 2:
        ids_ok = np.array([t["ID"] for t in success_trials])
        mts_ok = np.array([t["MT"] for t in success_trials])
        a, b, r2, y_pred = fit_fitts_law(ids_ok, mts_ok, outlier_std=3.0)
        if a is None or b is None or r2 is None or np.isnan(r2):
            print("Fitts-Regression übersprungen: alle IDs identisch oder unzureichende Varianz.")
        else:
            fitts_summary = {
                "n_success": len(success_trials),
                "n_trials": len(trials),
                "a": a,
                "b": b,
                "R2": float(r2) if not np.isnan(r2) else None,
                "formula": "MT = a + b * ID",
                "ID_formula": id_formula,
                "ID_expression": "log2(2*D/W)" if id_formula == "shannon" else "log2(D/W+1)",
            }
            summary_path = os.path.join(output_dir, "fitts_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(fitts_summary, f, indent=2)
            print(f"Fitts: MT = {a:.4f} + {b:.4f}·ID  (R² = {r2:.4f})  -> {summary_path}")
    else:
        print("Fitts-Regression: zu wenig erfolgreiche Trials (mind. 2).")

    # Success-Rate gesamt und pro Quartal für WandB
    success_overall = sum(1 for t in trials if t["success"]) / len(trials) if trials else 0.0
    success_by_quartal = {}
    for q in range(n_targets):
        subset = [t for t in trials if t["pair_idx"] == q]
        success_by_quartal[q] = (sum(1 for t in subset if t["success"]) / len(subset)) if subset else 0.0

    if use_wandb and WANDB_AVAILABLE:
        run_name = f"fitts_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        wandb.init(project="thumb_reach", name=run_name, config={
            "model_path": model_path,
            "output_dir": output_dir,
            "n_episodes": n_episodes,
            "n_targets": n_targets,
            "seed": seed,
        })
        log_dict = {"eval_fitts/success_rate": success_overall}
        for q in range(n_targets):
            log_dict[f"eval_fitts/success_rate_quartal_{q}"] = success_by_quartal[q]
        wandb.log(log_dict)
        wandb.finish()
        print(f"WandB: success_rate={success_overall:.2%}, quartal_0={success_by_quartal.get(0, 0):.2%}, quartal_1={success_by_quartal.get(1, 0):.2%}, quartal_2={success_by_quartal.get(2, 0):.2%}, quartal_3={success_by_quartal.get(3, 0):.2%}")

    return trials


def main():
    parser = argparse.ArgumentParser(
        description="Fitts Law Evaluation für Thumb Reach Checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python models/evaluate_fitts.py --model logs/thumb_reach_XXX/best_model.zip
  python models/evaluate_fitts.py --model logs/.../best_model.zip --n-episodes 100 --no-trajectories
  python models/evaluate_fitts.py --model .../best_model.zip --output mein_ordner --id-formula mackenzie
        """,
    )
    parser.add_argument("--model", type=str, required=True, help="Pfad zum Checkpoint (.zip); daraus werden Env und Ausgabe abgeleitet")
    parser.add_argument("--output", type=str, default=None,
                        help="Ausgabe-Ordner (trials.csv, trajectories/, fitts_summary.json). Default: <Modell-Ordner>/eval_fitts")
    parser.add_argument("--log-dir", type=str, default=None, help="Ordner mit final_env.pkl (Default: Modell-Ordner)")
    parser.add_argument("--n-episodes", type=int, default=100, help="Anzahl Episoden (zyklisch über Ziel-Radius-Paare)")
    parser.add_argument("--radii", type=float, nargs="+", default=None,
                        help="Target-Radien für (Target,Radius)-Paare. Default: 0.2 0.25 0.3 0.35 0.4")
    parser.add_argument("--n-targets", type=int, default=None, help="Anzahl (Target,Radius)-Paare (Default: alle = Targets × Radien)")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed für Reproduzierbarkeit")
    parser.add_argument("--no-trajectories", action="store_true", help="Keine Trajektorien speichern (nur trials.csv)")
    parser.add_argument("--no-wandb", action="store_true", help="Kein WandB-Logging (Success-Rate pro Quartal)")
    parser.add_argument("--id-formula", type=str, default="shannon", choices=["shannon", "mackenzie"],
                        help="ID-Formel: shannon=log2(2*D/W), mackenzie=log2(D/W+1) wie uitb_evaluate")
    parser.add_argument(
        "--no-trim-traj",
        action="store_true",
        help="Kein Trimmen des letzten Trajektorienpunkts (rohe VecEnv-Schritte in ep_*.npz; trials.csv unverändert)",
    )
    args = parser.parse_args()

    model_path = args.model
    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(model_path)), "eval_fitts")

    print("=" * 60)
    print("Fitts Law Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {model_path}")
    print(f"Output:     {output_dir}")
    print(f"Episoden:   {args.n_episodes}")
    print(f"Radien:     {args.radii if args.radii is not None else '0.2 0.25 0.3 0.35 0.4 (Default)'}")
    print("=" * 60)
    run_fitts_evaluation(
        model_path=model_path,
        output_dir=output_dir,
        n_episodes=args.n_episodes,
        n_targets=args.n_targets,
        radii=args.radii,
        seed=args.seed,
        log_dir=args.log_dir,
        save_trajectories=not args.no_trajectories,
        use_wandb=not args.no_wandb,
        id_formula=args.id_formula,
        trim_autoreset_tail=not args.no_trim_traj,
    )

    print("=" * 60)
    print("Evaluation beendet.")
    print("=" * 60)


if __name__ == "__main__":
    main()

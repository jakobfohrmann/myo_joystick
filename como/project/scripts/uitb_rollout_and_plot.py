import os
import json
import pickle
import argparse
import numpy as np

# Headless plotting (wichtig auf Servern)
import matplotlib
matplotlib.use("Agg")

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uitb_evaluate.trajectory_data import TrajectoryData_RL
from uitb_evaluate.evaluate_main import trajectoryplot


ENV_NAME = "ThumbReach-v0"


def build_eval_env(log_dir: str):
    """Erstellt Eval-Env passend zur gespeicherten env_config.json + lädt VecNormalize Stats."""
    env_config_path = os.path.join(log_dir, "env_config.json")
    with open(env_config_path, "r", encoding="utf8") as f:
        env_config = json.load(f)

    # envs importieren/registrieren (so wie in deinem Training)
    import envs  # noqa: F401

    def _make():
        e = gym.make(ENV_NAME, **env_config)
        e = Monitor(e)
        return e

    venv = DummyVecEnv([_make])

    # VecNormalize laden (falls vorhanden)
    vn_path = os.path.join(log_dir, "final_env.pkl")
    if os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = False
        venv.norm_reward = False

    return venv


def unwrap_base_env(vec_env):
    """VecNormalize/DummyVecEnv/Monitor runterwrappen -> echtes MyoSuite Env."""
    env = vec_env
    while hasattr(env, "venv"):
        env = env.venv
    env = env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def rollout_to_uitb_logs(log_dir: str, out_subdir: str, n_episodes: int = 20):
    """Rollt die Policy aus und speichert state_log.pickle + action_log.pickle im uitb-Format."""
    venv = build_eval_env(log_dir)
    base_env = unwrap_base_env(venv)

    model_path = os.path.join(log_dir, "final_model")
    model = PPO.load(model_path, env=venv)

    state_log = {}
    action_log = {}

    for ep in range(n_episodes):
        obs = venv.reset()

        # Listen: wir loggen auch den Startzustand (nach reset)
        t_list, pos_list, targ_list = [], [], []
        qpos_list, qvel_list = [], []
        act_list, rew_list = [], []

        # --- initial obs (Startpunkt)
        od = base_env.get_obs_dict(base_env.sim)
        t_list.append(float(np.asarray(od["time"]).ravel()[0]))
        pos_list.append(np.asarray(od["joystick_2d"], dtype=np.float32).ravel()[:2])
        targ_list.append(np.asarray(od["target_pos"], dtype=np.float32).ravel()[:2])
        qpos_list.append(np.asarray(od["qpos"], dtype=np.float32).ravel())
        qvel_list.append(np.asarray(od["qvel"], dtype=np.float32).ravel())

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)

            od = base_env.get_obs_dict(base_env.sim)
            t_list.append(float(np.asarray(od["time"]).ravel()[0]))
            pos_list.append(np.asarray(od["joystick_2d"], dtype=np.float32).ravel()[:2])
            targ_list.append(np.asarray(od["target_pos"], dtype=np.float32).ravel()[:2])
            qpos_list.append(np.asarray(od["qpos"], dtype=np.float32).ravel())
            qvel_list.append(np.asarray(od["qvel"], dtype=np.float32).ravel())

            act_list.append(np.asarray(action[0], dtype=np.float32))
            rew_list.append(float(reward[0]))

        t_arr = np.asarray(t_list, dtype=np.float32)
        pos_arr = np.stack(pos_list, axis=0)      # (T,2)
        targ_arr = np.stack(targ_list, axis=0)    # (T,2)

        # Velocity aus Position ableiten (robust, falls time mal komisch ist)
        if len(t_arr) >= 2 and np.all(np.diff(t_arr) > 0):
            vel_arr = np.gradient(pos_arr, t_arr, axis=0)
        else:
            dt = float(getattr(base_env, "dt", 1 / 50))
            vel_arr = np.gradient(pos_arr, dt, axis=0)

        qpos_arr = np.stack(qpos_list, axis=0)
        qvel_arr = np.stack(qvel_list, axis=0)
        qacc_arr = np.gradient(qvel_arr, axis=0)

        ep_key = f"episode_{ep:03d}"
        state_log[ep_key] = {
            "timestep": t_arr,
            "fingertip_xpos": pos_arr,          # wir mappen joystick_2d -> fingtertipp (endeffector)
            "fingertip_xvelp": vel_arr,
            "target_position": targ_arr,        # target_name="target" nutzt target_position
            "target_radius": np.ones((len(t_arr),), dtype=np.float32) * float(getattr(base_env, "target_radius", 0.5)),
            "target_idx": np.ones((len(t_arr),), dtype=np.float32) * float(
                                                                                getattr(base_env, "curriculum_quadrant_idx", -1)
                                                                                if getattr(base_env, "curriculum_quadrant_idx", -1) is not None
                                                                                else -1
                                                                            ),
            "qpos": qpos_arr,
            "qvel": qvel_arr,
            "qacc": qacc_arr,
        }
        action_log[ep_key] = {
            "action": np.stack(act_list, axis=0) if len(act_list) else np.zeros((0,), dtype=np.float32),
            "ctrl": np.stack(act_list, axis=0) if len(act_list) else np.zeros((0,), dtype=np.float32),
            "reward": np.asarray(rew_list, dtype=np.float32),
        }

    out_dir = os.path.join(log_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "state_log.pickle"), "wb") as f:
        pickle.dump(state_log, f)
    with open(os.path.join(out_dir, "action_log.pickle"), "wb") as f:
        pickle.dump(action_log, f)

    return out_subdir


def make_uitb_plots(log_dir: str, subdir: str):
    plots_dir = os.path.join(log_dir, "plots_uitb")
    os.makedirs(plots_dir, exist_ok=True)

    traj = TrajectoryData_RL(
        DIRNAME_SIMULATION=log_dir,
        filename=subdir,
        REPEATED_MOVEMENTS=False,
    )

    # Trick: filename enthält "tracking" -> split_trials=False im evaluate_main
    fname = "tracking_thumbreach"

    # 1) Tracking-Distance Plot (sehr aussagekräftig)
    trajectoryplot(
        PLOTTING_ENV="RL-UIB",
        USER_ID="PPO",
        TASK_CONDITION="ThumbReach",
        common_simulation_subdir="",
        filename=fname,
        trajectories_SIMULATION=traj,
        PLOT_TRACKING_DISTANCE=True,
        PLOT_ENDEFFECTOR=True,
        PLOT_TIME_SERIES=True,
        STORE_PLOT=True,
        PLOTS_DIR=plots_dir,
    )

    # 2) Endeffector Zeitreihen (x/y) + optional Vel/Acc
    trajectoryplot(
        PLOTTING_ENV="RL-UIB",
        USER_ID="PPO",
        TASK_CONDITION="ThumbReach",
        common_simulation_subdir="",
        filename=fname,
        trajectories_SIMULATION=traj,
        PLOT_TRACKING_DISTANCE=False,
        PLOT_ENDEFFECTOR=True,
        PLOT_TIME_SERIES=True,
        PLOT_VEL_ACC=True,
        STORE_PLOT=True,
        PLOTS_DIR=plots_dir,
    )

    return plots_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="z.B. logs/thumb_reach_YYYY-MM-DD_HH-MM-SS")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    sub = rollout_to_uitb_logs(args.logdir, out_subdir="uitb_eval", n_episodes=args.episodes)
    plots_dir = make_uitb_plots(args.logdir, sub)

    print("uitb logs:", os.path.join(args.logdir, sub))
    print("uitb plots:", plots_dir)


if __name__ == "__main__":
    main()

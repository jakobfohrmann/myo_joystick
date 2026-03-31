"""Simple Thumb Reach Environment - Daumen bewegt sich zu Zielkoordinaten"""
import collections
import os
import sys
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils import gym
import gymnasium as gym_space


class ThumbReachEnvV0(BaseV0):
"""
Joystick Reach Environment: The joystick is moved toward target coordinates.

Observations:
    - qpos: Joint positions
    - qvel: Joint velocities  
    - joystick_2d: Normalized joystick position in [-1, 1] 
    - joystick_angles: Joystick angles in radians
    - target_pos: Normalized target position in [-1, 1] 
    - reach_err: Error vector (target - joystick_2d)

Actions:
    - 13 controllable muscles: thumb (5) + wrist (5) + forearm (PL, PT, PQ)
    - Finger muscles (FDS*, FDP*, EDC*, EDM, EIP, RI*, LU_*, UI_*) are fixed (baseline = 0.0)

Reward:
    - Negative distance to the target (normalized)
    - Bonus when close to the target
    - Penalty when far from the target
"""
    
    DEFAULT_OBS_KEYS = ["qpos", "qvel", "act", "joystick_2d", "joystick_angles", "target_pos", "reach_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50.0,
    }


    


    
    INFO_KEYWORDS = (
        "solved", "target_idx", "reward_reach", "reward_bonus",
        "reward_dense", "reward_sparse", "reach_dist",
    )

    def _finalize_info(self, info, obs_dict=None, rwd_dict=None):
        """Ensures that all info_keywords and reward_components are set before every return from step()."""
        if info is None:
            info = {}
        if obs_dict is None:
            obs_dict = self.get_obs_dict(self.sim)
        if rwd_dict is None:
            rwd_dict = self.get_reward_dict(obs_dict)
        reach_err = obs_dict.get("reach_err")
        reach_dist = float(np.linalg.norm(reach_err)) if reach_err is not None else 0.0
        # Top-level keys for Monitor (info_keywords)
        info["solved"] = bool(rwd_dict.get("solved", False))
        info["target_idx"] = int(getattr(self, "current_target_idx", -1))
        info["reward_reach"] = float(rwd_dict.get("reach", 0.0))
        info["reward_bonus"] = float(rwd_dict.get("bonus", 0.0))
        info["reward_dense"] = float(rwd_dict.get("dense", 0.0))
        info["reward_sparse"] = float(rwd_dict.get("sparse", 0.0))
        info["reach_dist"] = reach_dist
        # reward_components for callback buffer
        info["reward_components"] = {
            "reach": float(rwd_dict.get("reach", 0.0)),
            "bonus": float(rwd_dict.get("bonus", 0.0)),
            "penalty": float(rwd_dict.get("penalty", 0.0)),
            "dense": float(rwd_dict.get("dense", 0.0)),
            "sparse": float(rwd_dict.get("sparse", 0.0)),
            "reach_dist": reach_dist,
        }
        return info, obs_dict, rwd_dict

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
        )
        self._setup(**kwargs)
    
    def _setup(
        self,
        target_range: dict = None, 
        early_term_steps=250,  
        obs_keys: list = None,
        weighted_reward_keys: dict = None,
        **kwargs,
    ):
        self.early_term_steps = early_term_steps
        self.steps_since_reset = 0
        self.has_hit = False  # Track ob ein Treffer erreicht wurde
        self.target_radius = self.TARGET_RADIUS
        self.curriculum_target_idx = None
        # make target range two dimensional
        self.target_range = target_range or {
            "x": (-0.8, 0.8),  
            "y": (-0.8, 0.8),
        }
        
    
        self.curriculum_callback = None
        
    
        try:
            self.thumb_site_id = self.sim.model.site_name2id("thumbstick_marker")
        except:
            # first site
            self.thumb_site_id = 0
            print("no thumbstick marker")
        
        try:
            self.joystick_rx_joint_id = self.sim.model.joint_name2id("thumbstick_rx")
            self.joystick_ry_joint_id = self.sim.model.joint_name2id("thumbstick_ry")
        except:
            print("no joystick joints")
            self.joystick_rx_joint_id = None
            self.joystick_ry_joint_id = None
        self.thumb_joint_ids = []
        model = self.sim.model
        name_fn = getattr(model, "joint_id2name", None)
        for joint_id in range(model.njnt):
            joint_name = name_fn(joint_id) if name_fn else None
            if joint_name is None and hasattr(model, "joint"):
                try:
                    joint_name = model.joint(joint_id).name
                except Exception:
                    joint_name = None
            if not joint_name:
                continue
            if isinstance(joint_name, bytes):
                joint_name = joint_name.decode("utf-8")
            if "thumb" in str(joint_name).lower():
                self.thumb_joint_ids.append(joint_id)
        if len(self.thumb_joint_ids) == 0:
            print("fallback: full qvel/qpos")
        
        # center + span
        self.joystick_rx_center = 0.0
        self.joystick_ry_center = 0.0
        self.joystick_rx_span = 0.35  
        self.joystick_ry_span = 0.35  
        
        # create target site if not found
        try:
            self.target_site_id = self.sim.model.site_name2id("target_site")
        except:
            
            self.target_site_id = None
        
        # 5 thumb, 5 hand, 3 forearm muscles
        self.thumb_muscle_names = ["EPL", "EPB", "FPL", "APL", "OP"]
        self.wrist_muscle_names = ["FCU", "FCR", "ECRL", "ECRB", "ECU"]
        self.forearm_muscle_names = ["PL", "PT", "PQ"] 
        self.controllable_muscle_names = (
            self.thumb_muscle_names + self.wrist_muscle_names + self.forearm_muscle_names
        )
        self.thumb_muscle_indices = []
        self.wrist_muscle_indices = []
        self.forearm_muscle_indices = []
        for name in self.thumb_muscle_names:
            try:
                idx = self.sim.model.actuator_name2id(name)
                self.thumb_muscle_indices.append(idx)
            except Exception:
                print(f" Muscle {name} not found")
        for name in self.wrist_muscle_names:
            try:
                idx = self.sim.model.actuator_name2id(name)
                self.wrist_muscle_indices.append(idx)
            except Exception:
                print(f" Muscle {name} not found")
        for name in self.forearm_muscle_names:
            try:
                idx = self.sim.model.actuator_name2id(name)
                self.forearm_muscle_indices.append(idx)
            except Exception:
                print(f" Muscle {name} not found")
        self.controllable_muscle_indices = (
            self.thumb_muscle_indices + self.wrist_muscle_indices + self.forearm_muscle_indices
        )



# Optional: Restrict the action space to the 5 thumb muscles (instead of 13).
# To activate, uncomment the next line.
#
# - The action_space (below) will automatically have shape (5,),
#   since len(self.controllable_muscle_indices) is used.
# - In step(), only these indices are set; wrist, forearm, and all
#   remaining actuators stay at fixed_muscle_values (here 0.0).
# - Update the docstring and argument description in step() if needed
#   (it currently still refers to "13 values").
#
# self.controllable_muscle_indices = list(self.thumb_muscle_indices)



        if len(self.thumb_muscle_indices) != 5:
            print(f" Expected 5 thumb muscles, found {len(self.thumb_muscle_indices)}")
        if len(self.wrist_muscle_indices) != 5:
            print(f" Expected 5 wrist muscles, found {len(self.wrist_muscle_indices)}")
        if len(self.forearm_muscle_indices) != 3:
            print(f" Expected 3 forearm muscles, found {len(self.forearm_muscle_indices)}")

        # baseline activations for fixed muscles
        self.fixed_muscle_values = self._get_baseline_activations()

        # default target (robust for _setup/step before erstem reset())
        self.current_target_idx = 0
        self.target_pos = self.FIXED_TARGETS[0].copy()
        if self.target_site_id is not None:
            target_3d = np.array([self.target_pos[0], self.target_pos[1], 0.0])
            self.sim.model.site_pos[self.target_site_id] = target_3d
            if getattr(self, "sim_obsd", None) is not None:
                self.sim_obsd.model.site_pos[self.target_site_id] = target_3d

        # Call super()._setup() – BaseV0 overrides action_space
        obs_keys = obs_keys or self.DEFAULT_OBS_KEYS
        weighted_reward_keys = weighted_reward_keys or self.DEFAULT_RWD_KEYS_AND_WEIGHTS

        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )

        # action space
        self.action_space = gym_space.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.controllable_muscle_indices),),
            dtype=np.float32
        )
        
        # initiale state snapshot (for hard resets)
        self._qpos0 = self.sim.data.qpos.copy()
        self._qvel0 = self.sim.data.qvel.copy()

    
    def _get_baseline_activations(self):
        """Returns baseline activations for all non-controlled muscles (finger muscles fixed)."""
        baseline = np.zeros(self.sim.model.nu)
        
        # Baseline-Values
        # no baseline activations
        muscle_map = {}
        
        return baseline
    
    def get_obs_dict(self, sim):
        """observations: joystick position, target position, error"""
        obs_dict = {}
        obs_dict["time"] = np.array([sim.data.time])
        
        if hasattr(self, 'thumb_joint_ids') and len(self.thumb_joint_ids) > 0:
            obs_dict["qpos"] = sim.data.qpos[self.thumb_joint_ids].copy()
            obs_dict["qvel"] = sim.data.qvel[self.thumb_joint_ids].copy() * self.dt
        else:
            obs_dict["qpos"] = sim.data.qpos.copy()
            obs_dict["qvel"] = sim.data.qvel.copy() * self.dt
        
        obs_dict["act"] = sim.data.act[:].copy() if hasattr(sim.data, 'act') and sim.data.act is not None else np.zeros(self.sim.model.na)
        
          
        if self.joystick_rx_joint_id is not None and self.joystick_ry_joint_id is not None:
            
            joystick_angles = np.array([
                sim.data.qpos[self.joystick_rx_joint_id],
                sim.data.qpos[self.joystick_ry_joint_id]
            ])
            obs_dict["joystick_angles"] = joystick_angles  
            
        
            rx_norm = (joystick_angles[0] - self.joystick_rx_center) / self.joystick_rx_span if self.joystick_rx_span > 0 else 0
            ry_norm = (joystick_angles[1] - self.joystick_ry_center) / self.joystick_ry_span if self.joystick_ry_span > 0 else 0
            joystick_2d = np.array([rx_norm, ry_norm])
            obs_dict["joystick_2d"] = joystick_2d
        
        
        obs_dict["target_pos"] = self.target_pos.copy()
        obs_dict["reach_err"] = self.target_pos - obs_dict["joystick_2d"]
        
        return obs_dict
    
    def get_reward_dict(self, obs_dict):
        """reward: negative distance + hit bonus + penalty (early termination). episode ends on hit."""
        reach_dist = np.linalg.norm(obs_dict["reach_err"], axis=-1)
        reach_dist = float(np.atleast_1d(reach_dist)[0])

        near_th = getattr(self, "target_radius", None)
        if near_th is None:
            near_th = self.TARGET_RADIUS

        in_radius = reach_dist < near_th
        hit_bonus = 0.0
        penalty = 0.0
        episode_done = False

        if in_radius:
            self.has_hit = True
            hit_bonus = 1.0
            dist_reward = 0.0 
            if getattr(self, "steps_since_reset", 0) > 0:
                episode_done = True 
        else:
            dist_reward = (-1.0 * reach_dist)

       
        if hasattr(self, "steps_since_reset") and hasattr(self, "early_term_steps"):
            if self.steps_since_reset >= self.early_term_steps and not episode_done:
                penalty = -1.0
                episode_done = True

    
        solved = self.has_hit
        sparse_reward = dist_reward
        rwd_dict = collections.OrderedDict(
            (
                ("reach", dist_reward),
                ("bonus", hit_bonus),
                ("penalty", penalty),
                ("sparse", sparse_reward),
                ("solved", solved),
                ("done", episode_done),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict
        
    def generate_target(self):
        """generates a random target within target_range (2D for joystick)"""
        if self.curriculum_callback is not None:
            idx = int(self.curriculum_callback.sample_target_idx())
            fixed = self.curriculum_callback.FIXED_TARGETS
            self.target_pos = fixed[idx].copy()
            self.curriculum_target_idx = idx
            self.current_target_idx = idx 
            self.target_radius = self.curriculum_callback.get_target_radius(idx)
        else:
            
            x_val = self.np_random.uniform(low=self.target_range["x"][0], high=self.target_range["x"][1])
            y_val = self.np_random.uniform(low=self.target_range["y"][0], high=self.target_range["y"][1])
            self.target_pos = np.array([x_val, y_val])
            self.curriculum_target_idx = None
            self.current_target_idx = -1  
            self.target_radius = self.TARGET_RADIUS
        
        ´
        if self.target_site_id is not None:
            target_3d = np.array([self.target_pos[0], self.target_pos[1], 0.0])
            self.sim.model.site_pos[self.target_site_id] = target_3d
            self.sim_obsd.model.site_pos[self.target_site_id] = target_3d
    
    def step(self, action):
"""
Override step: thumb + wrist + forearm controlled via action; finger muscles fixed.

Args:
    action: array with 13 values [EPL, EPB, FPL, APL, OP, FCU, FCR, ECRL, ECRB, ECU, PL, PT, PQ]

Returns:
    standard (obs, reward, terminated, truncated, info) tuple
"""
        action = np.atleast_1d(np.asarray(action, dtype=np.float64))
        if len(action) == self.sim.model.nu:
            obs, reward, terminated, truncated, info = super().step(action)
            info, _, _ = self._finalize_info(info)
            return obs, reward, terminated, truncated, info

        )
        if len(action) != len(self.controllable_muscle_indices):
            raise ValueError(
                f"Action must have {len(self.controllable_muscle_indices)} values (thumb + wrist + forearm), "
                f"got {len(action)}"
            )

        
        full_action = self.fixed_muscle_values.copy()
        for i, idx in enumerate(self.controllable_muscle_indices):
            full_action[idx] = np.clip(action[i], 0.0, 1.0)
        
       
        obs, reward, terminated, truncated, info = super().step(full_action)
        
        
        self.steps_since_reset += 1
        
        
        obs_dict = self.get_obs_dict(self.sim)
        rwd_dict = self.get_reward_dict(obs_dict)

        episode_done = bool(rwd_dict["done"])
        
        if episode_done:
            if self.has_hit:
                terminated = True   
            else:
                truncated = True    

        
        if terminated or truncated:
            if info is None:
                info = {}
            info["TimeLimit.truncated"] = truncated  
        
        if episode_done or terminated or truncated:
            msg = (
                f"[DONE-CHECK] t={self.steps_since_reset} "
                f"has_hit={self.has_hit} "
                f"reach_dist={float(np.linalg.norm(obs_dict['reach_err'])):.4f} "
                f"rwd_done={episode_done} "
                f"terminated={terminated} truncated={truncated}"
            )
            print(msg, file=sys.stderr, flush=True)
        elif os.environ.get("DEBUG_DONE") == "1" and self.steps_since_reset % 50 == 0:
            print(
                f"[DONE-DEBUG] t={self.steps_since_reset} reach_dist={float(np.linalg.norm(obs_dict['reach_err'])):.4f} "
                f"episode_done={episode_done} early_term={self.steps_since_reset >= getattr(self, 'early_term_steps', 250)}",
                file=sys.stderr,
                flush=True,
            )

        
       
        if info is None:
            info = {}
        if os.environ.get("DEBUG_REWARD_COMPONENTS") == "1":
            info["debug_thumb_step"] = True
            info["debug_action_len"] = len(action)

        
        info, obs_dict, rwd_dict = self._finalize_info(info, obs_dict=obs_dict, rwd_dict=rwd_dict)

        if terminated or truncated:
            joystick_terminal = obs_dict.get("joystick_2d", None)
            if joystick_terminal is not None:
                info["terminal_joystick_2d"] = np.asarray(joystick_terminal).ravel()[:2].tolist()
            target_terminal = obs_dict.get("target_pos", None)
            if target_terminal is not None:
                info["terminal_target_pos"] = np.asarray(target_terminal).ravel()[:2].tolist()

            info["success"] = float(self.has_hit)  
            if self.curriculum_callback is not None:
                self.curriculum_callback.record_episode_result(
                    success=bool(self.has_hit),
                    target_idx=self.curriculum_target_idx,
                )

        return obs, reward, terminated, truncated, info
    
    def _hard_reset_qpos_qvel(self):
       
        qpos = getattr(self, "init_qpos", self._qpos0).copy()
        qvel = getattr(self, "init_qvel", self._qvel0).copy()

       
        if hasattr(self, "set_state"):
            self.set_state(qpos, qvel)
        else:
        
            self.sim.data.qpos[:] = qpos
            self.sim.data.qvel[:] = qvel
            if hasattr(self.sim, "forward"):
                self.sim.forward()

    def reset(self, **kwargs):
        self.steps_since_reset = 0
        self.best_reach_dist = None
        self.steps_since_improvement = 0
        self.has_hit = False
        self.curriculum_target_idx = None

        base_out = super().reset(**kwargs)
        info = base_out[1] if isinstance(base_out, tuple) else {}

       
        self._hard_reset_qpos_qvel()

        
        options = kwargs.get("options") or {}
        if "target_idx" in options:
            idx = int(options["target_idx"])
            if 0 <= idx < len(self.FIXED_TARGETS):
                self.target_pos = self.FIXED_TARGETS[idx].copy()
                self.current_target_idx = idx
                if self.target_site_id is not None:
                    target_3d = np.array([self.target_pos[0], self.target_pos[1], 0.0])
                    self.sim.model.site_pos[self.target_site_id] = target_3d
                    self.sim_obsd.model.site_pos[self.target_site_id] = target_3d
            else:
                self.generate_target()
        else:
            self.generate_target()

        
        self.robot.sync_sims(self.sim, self.sim_obsd)

       
        obs_dict = self.get_obs_dict(self.sim)
        obs_keys = getattr(self, "obs_keys", self.DEFAULT_OBS_KEYS)
        obs = np.concatenate([np.ravel(obs_dict[k]) for k in obs_keys], axis=0)

        return (obs, info) if isinstance(base_out, tuple) else obs

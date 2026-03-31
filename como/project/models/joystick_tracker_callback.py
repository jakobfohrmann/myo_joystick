"""Callback um Joystick-Positionen während des Trainings für die Heatmap zu tracken"""
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict


class JoystickTrackerCallback(BaseCallback):
    """
    Verfolgt Joystick-Positionen und Zielpositionen während der Trainings-Episoden.

    Speichert:
    - Alle während des Trainings besuchten Joystick-Positionen (für eine Heatmap)
    - Die Endposition jeder Episode (für Splash-Punkte)
    - Alle generierten Zielpositionen (für eine Ziel-Heatmap)
    - Die Zielposition pro Episode (für die Zielvisualisierung)
    """
    
    def __init__(self, save_path="logs", verbose=0, save_freq=10000):
        super().__init__(verbose)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.save_freq = save_freq
        
        # Trackt alle besuchten joystick-positionen (für heatmap)
        self.all_positions = []
        
        # Trackt finale joystick-position pro episode (for splash points)
        self.episode_final_positions = []
        self.current_episode_positions = []
        
        # Trackt alle target positions (for target heatmap)
        self.all_target_positions = []
        
        # Trackt target positionen per episode (one per episode, generated at reset)
        self.episode_target_positions = []
        self.current_episode_target = None
        
        # Track episode info
        self.episode_rewards = []
        self.episode_lengths = []
        
        # For vectorized envs - track positions per environment
        from collections import defaultdict
        self.episode_positions_dict = defaultdict(list)
        self.episode_targets_dict = defaultdict(lambda: None)
        self.last_episode_count = {}

        # If the env provides terminal coordinates in info["terminal_*"], we can log true final positions
        # from infos directly (VecEnv can reset obs immediately on done).
        self._terminal_positions_from_info = False
        
    def _on_step(self) -> bool:
        """Called at each step - track joystick and target positions"""
        # Periodically save data to avoid memory issues
        if self.n_calls % self.save_freq == 0:
            self._save_data()

        infos = []
        if hasattr(self, "locals") and self.locals is not None:
            infos = self.locals.get("infos", []) or []
        
        # Get the environment (unwrap if needed)
        env = self.training_env
        
        # For vectorized environments, we need to check each env
        if hasattr(env, 'envs'):
            # Vectorized environment - track all envs
            for env_idx, single_env in enumerate(env.envs):
                # Unwrap to get the actual environment
                unwrapped = single_env
                while hasattr(unwrapped, 'env'):
                    unwrapped = unwrapped.env

                # If an episode ended, VecEnv may already have reset obs. Prefer terminal coordinates from info.
                info = infos[env_idx] if env_idx < len(infos) else {}
                if isinstance(info, dict):
                    terminal_pos = info.get("terminal_joystick_2d", None)
                    if terminal_pos is not None:
                        self._terminal_positions_from_info = True
                        terminal_pos = np.asarray(terminal_pos).ravel()[:2]
                        self.all_positions.append(terminal_pos.copy())
                        self.episode_final_positions.append(terminal_pos.copy())

                        terminal_target = info.get("terminal_target_pos", None)
                        if terminal_target is not None:
                            terminal_target = np.asarray(terminal_target).ravel()[:2]
                            self.all_target_positions.append(terminal_target.copy())
                            self.episode_target_positions.append(terminal_target.copy())

                        # Clear so older rollout-end logic can't mis-attribute positions to episodes.
                        self.episode_positions_dict[env_idx] = []
                        self.episode_targets_dict[env_idx] = None
                        continue

                # Try to get joystick position from obs (non-terminal step)
                joystick_pos = self._get_joystick_position(unwrapped)
                if joystick_pos is not None:
                    self.all_positions.append(joystick_pos.copy())
                    self.episode_positions_dict[env_idx].append(joystick_pos.copy())

                # Try to get target position from obs (non-terminal step)
                target_pos = self._get_target_position(unwrapped)
                if target_pos is not None:
                    self.all_target_positions.append(target_pos.copy())
                    self.episode_targets_dict[env_idx] = target_pos.copy()
        else:
            # Single environment
            unwrapped = env
            while hasattr(unwrapped, 'env'):
                unwrapped = unwrapped.env

            info = infos[0] if infos else {}
            if isinstance(info, dict):
                terminal_pos = info.get("terminal_joystick_2d", None)
                if terminal_pos is not None:
                    self._terminal_positions_from_info = True
                    terminal_pos = np.asarray(terminal_pos).ravel()[:2]
                    self.all_positions.append(terminal_pos.copy())
                    self.episode_final_positions.append(terminal_pos.copy())

                    terminal_target = info.get("terminal_target_pos", None)
                    if terminal_target is not None:
                        terminal_target = np.asarray(terminal_target).ravel()[:2]
                        self.all_target_positions.append(terminal_target.copy())
                        self.episode_target_positions.append(terminal_target.copy())

                    self.current_episode_positions = []
                    self.current_episode_target = None
                else:
                    # Track joystick position
                    joystick_pos = self._get_joystick_position(unwrapped)
                    if joystick_pos is not None:
                        self.all_positions.append(joystick_pos.copy())
                        self.current_episode_positions.append(joystick_pos.copy())

                    # Track target position
                    target_pos = self._get_target_position(unwrapped)
                    if target_pos is not None:
                        self.all_target_positions.append(target_pos.copy())
                        self.current_episode_target = target_pos.copy()
            else:
                # Track joystick position (no info dict available)
                joystick_pos = self._get_joystick_position(unwrapped)
                if joystick_pos is not None:
                    self.all_positions.append(joystick_pos.copy())
                    self.current_episode_positions.append(joystick_pos.copy())

                # Track target position
                target_pos = self._get_target_position(unwrapped)
                if target_pos is not None:
                    self.all_target_positions.append(target_pos.copy())
                    self.current_episode_target = target_pos.copy()
        
        return True
    
    def _get_joystick_position(self, env):
        """Extract joystick position from environment"""
        try:
            # Try to get from observation dictionary
            if hasattr(env, 'get_obs_dict') and hasattr(env, 'sim'):
                obs_dict = env.get_obs_dict(env.sim)
                joystick_pos = obs_dict.get('joystick_2d', None)
                if joystick_pos is not None:
                    # Ensure it's a 2D array
                    joystick_pos = np.array(joystick_pos)
                    if len(joystick_pos) > 2:
                        joystick_pos = joystick_pos[:2]
                    return joystick_pos
            
            # Fallback: calculate from joint positions
            if hasattr(env, 'joystick_rx_joint_id') and hasattr(env, 'joystick_ry_joint_id'):
                if (env.joystick_rx_joint_id is not None and 
                    env.joystick_ry_joint_id is not None and 
                    hasattr(env, 'sim')):
                    sim = env.sim
                    joystick_angles = np.array([
                        sim.data.qpos[env.joystick_rx_joint_id],
                        sim.data.qpos[env.joystick_ry_joint_id]
                    ])
                    # Normalize
                    rx_norm = ((joystick_angles[0] - env.joystick_rx_center) / env.joystick_rx_span 
                              if env.joystick_rx_span > 0 else 0)
                    ry_norm = ((joystick_angles[1] - env.joystick_ry_center) / env.joystick_ry_span 
                              if env.joystick_ry_span > 0 else 0)
                    return np.array([rx_norm, ry_norm])
        except Exception as e:
            # Silently fail - environment might not be ready yet
            pass
        
        return None
    
    def _get_target_position(self, env):
        """Extract target position from environment"""
        try:
            # Try to get from observation dictionary
            if hasattr(env, 'get_obs_dict') and hasattr(env, 'sim'):
                obs_dict = env.get_obs_dict(env.sim)
                target_pos = obs_dict.get('target_pos', None)
                if target_pos is not None:
                    # Ensure it's a 2D array
                    target_pos = np.array(target_pos)
                    if len(target_pos) > 2:
                        target_pos = target_pos[:2]
                    return target_pos
            
            # Fallback: get directly from environment attribute
            if hasattr(env, 'target_pos'):
                target_pos = env.target_pos
                if target_pos is not None:
                    target_pos = np.array(target_pos)
                    if len(target_pos) > 2:
                        target_pos = target_pos[:2]
                    return target_pos
        except Exception as e:
            # Silently fail - environment might not be ready yet
            pass
        
        return None
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout - extract episode final positions"""
        # If we already log terminal positions via info["terminal_*"], skip the old "last obs" logic.
        if getattr(self, "_terminal_positions_from_info", False):
            return
        # Check for completed episodes using Monitor stats
        if hasattr(self.training_env, 'envs'):
            # Vectorized environment - check Monitor wrappers
            for env_idx, single_env in enumerate(self.training_env.envs):
                unwrapped = single_env
                # Unwrap to find Monitor
                monitor = None
                while hasattr(unwrapped, 'env'):
                    if hasattr(unwrapped, 'get_episode_rewards'):
                        monitor = unwrapped
                        break
                    unwrapped = unwrapped.env
                
                # Check if new episodes completed
                if monitor is not None:
                    episode_rewards = monitor.get_episode_rewards()
                    current_count = len(episode_rewards)
                    last_count = self.last_episode_count.get(env_idx, 0)
                    
                    # If new episodes completed, save final positions and targets
                    if current_count > last_count and hasattr(self, 'episode_positions_dict'):
                        if env_idx in self.episode_positions_dict:
                            positions = self.episode_positions_dict[env_idx]
                            if len(positions) > 0:
                                # Save final position for each completed episode
                                num_new_episodes = current_count - last_count
                                # For simplicity, save the last position
                                final_pos = positions[-1]
                                self.episode_final_positions.append(final_pos.copy())
                                # Clear for next episode
                                self.episode_positions_dict[env_idx] = []
                        
                        # Save target position for completed episode
                        if env_idx in self.episode_targets_dict:
                            target_pos = self.episode_targets_dict[env_idx]
                            if target_pos is not None:
                                self.episode_target_positions.append(target_pos.copy())
                                # Clear for next episode
                                self.episode_targets_dict[env_idx] = None
                    
                    self.last_episode_count[env_idx] = current_count
        else:
            # Single env - check Monitor
            unwrapped = self.training_env
            monitor = None
            while hasattr(unwrapped, 'env'):
                if hasattr(unwrapped, 'get_episode_rewards'):
                    monitor = unwrapped
                    break
                unwrapped = unwrapped.env
            
            # Check if new episode completed
            if monitor is not None:
                episode_rewards = monitor.get_episode_rewards()
                current_count = len(episode_rewards)
                last_count = getattr(self, 'last_episode_count_single', 0)
                
                if current_count > last_count and len(self.current_episode_positions) > 0:
                    final_pos = self.current_episode_positions[-1]
                    self.episode_final_positions.append(final_pos.copy())
                    self.current_episode_positions = []
                    
                    # Save target position for completed episode
                    if self.current_episode_target is not None:
                        self.episode_target_positions.append(self.current_episode_target.copy())
                        self.current_episode_target = None
                
                self.last_episode_count_single = current_count
    
    def _on_training_end(self) -> None:
        """Called at the end of training - save all data"""
        self._save_data()
    
    def on_episode_end(self, episode_num: int, episode_info: dict) -> None:
        """Called when an episode ends - save final position and target"""
        # Always track episode metadata so _save_data() can write lengths/rewards.
        self.episode_rewards.append(episode_info.get('r', 0))
        self.episode_lengths.append(episode_info.get('l', 0))

        # If terminal positions are logged via info, avoid double-saving finals.
        if getattr(self, "_terminal_positions_from_info", False):
            self.current_episode_positions = []
            self.current_episode_target = None
            return

        # Fallback: use last obs-based position.
        if len(self.current_episode_positions) > 0:
            final_pos = self.current_episode_positions[-1]
            self.episode_final_positions.append(final_pos.copy())

            if self.current_episode_target is not None:
                self.episode_target_positions.append(self.current_episode_target.copy())

            self.current_episode_positions = []
            self.current_episode_target = None
    
    def _save_data(self):
        """Save all tracked data to numpy files"""
        # Save all joystick positions (for heatmap)
        if len(self.all_positions) > 0:
            all_positions_array = np.array(self.all_positions)
            positions_file = os.path.join(self.save_path, "joystick_all_positions.npy")
            np.save(positions_file, all_positions_array)
            if self.verbose > 0:
                print(f"Saved {len(self.all_positions)} joystick positions to {positions_file}")
        
        # Save final joystick positions per episode (for splash points)
        if len(self.episode_final_positions) > 0:
            final_positions_array = np.array(self.episode_final_positions)
            final_positions_file = os.path.join(self.save_path, "joystick_episode_final_positions.npy")
            np.save(final_positions_file, final_positions_array)
            if self.verbose > 0:
                print(f"Saved {len(self.episode_final_positions)} episode final positions to {final_positions_file}")
        
        # Save all target positions (for target heatmap)
        if len(self.all_target_positions) > 0:
            all_targets_array = np.array(self.all_target_positions)
            targets_file = os.path.join(self.save_path, "target_all_positions.npy")
            np.save(targets_file, all_targets_array)
            if self.verbose > 0:
                print(f"Saved {len(self.all_target_positions)} target positions to {targets_file}")
        
        # Save target positions per episode
        if len(self.episode_target_positions) > 0:
            episode_targets_array = np.array(self.episode_target_positions)
            episode_targets_file = os.path.join(self.save_path, "target_episode_positions.npy")
            np.save(episode_targets_file, episode_targets_array)
            if self.verbose > 0:
                print(f"Saved {len(self.episode_target_positions)} episode target positions to {episode_targets_file}")
        
        # Save episode metadata
        if len(self.episode_rewards) > 0:
            metadata = {
                'rewards': np.array(self.episode_rewards),
                'lengths': np.array(self.episode_lengths),
            }
            metadata_file = os.path.join(self.save_path, "joystick_episode_metadata.npy")
            np.savez(metadata_file, **metadata)
            if self.verbose > 0:
                print(f"Saved episode metadata to {metadata_file}")

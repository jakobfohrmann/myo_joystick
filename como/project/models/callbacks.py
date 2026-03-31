
import os
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class WandBEvalCallback(EvalCallback):
"""
Evaluation callback with WandB logging and checkpoint saving.

Combines:
    - Evaluation (from EvalCallback)
    - WandB logging
    - Best model saving
    - Environment saving
    - Reward component logging
"""
    
    def __init__(
        self,
        eval_env,
        wandb_project="thumb_reach",
        wandb_run_name=None,
        log_freq=10_000,
        n_eval_episodes=10,
        best_model_save_path=None,
        curriculum_callback=None,
        verbose=1,
        **kwargs
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            best_model_save_path=best_model_save_path,
            log_path=best_model_save_path or "logs",
            eval_freq=log_freq,
            deterministic=True,
            verbose=verbose,
            **kwargs
        )
        
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.curriculum_callback = curriculum_callback
        self.wandb_initialized = False
        self.debug_reward_components = os.environ.get("DEBUG_REWARD_COMPONENTS") == "1"
        self.debug_counts = {
            "steps_total": 0,
            "steps_with_components": 0,
            "steps_with_debug_flag": 0,
            "infos_empty": 0,
        }
        
        
        self.reward_components_buffer = {
            "reach": [],
            "bonus": [],
            "penalty": [],
            "dense": [],
            "sparse": [],
            "reach_dist": [],
        }
        
        
        if WANDB_AVAILABLE and not self.wandb_initialized:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                sync_tensorboard=True,  # Auto-sync TensorBoard
                monitor_gym=True,  # Auto-upload videos (falls aktiviert)
                save_code=True,
            )
            self.wandb_initialized = True
    
    def _on_step(self) -> bool:
        """called at every step — collects reward components"""
        result = super()._on_step()
        
       
        if hasattr(self, 'locals') and self.locals is not None:
            infos = self.locals.get('infos', [])
            if self.debug_reward_components and not infos:
                self.debug_counts["infos_empty"] += 1
                if self.debug_counts["infos_empty"] == 1:
                    
            
            )
            for info in infos:
                if self.debug_reward_components:
                    self.debug_counts["steps_total"] += 1
                    if isinstance(info, dict) and info.get("debug_thumb_step"):
                        self.debug_counts["steps_with_debug_flag"] += 1
                        # Nur beim ersten Mal printen
                        if self.debug_counts["steps_with_debug_flag"] == 1:
                            print(f"[DEBUG]  ThumbReachEnvV0.step() wird verwendet! (debug_thumb_step gefunden)")
                if isinstance(info, dict) and "reward_components" in info:
                    components = info["reward_components"]
                    if self.debug_reward_components:
                        self.debug_counts["steps_with_components"] += 1
                        # Nur beim ersten Mal printen
                        if self.debug_counts["steps_with_components"] == 1:
                            print(f"[DEBUG] reward_components gefunden! Keys: {list(components.keys())}")
                    
                   
                    for key in self.reward_components_buffer:
                        if key in components:
                            self.reward_components_buffer[key].append(components[key])
        elif self.debug_reward_components:
            
            if not hasattr(self, '_warned_no_locals'):
                print(f"locals not availabe")
                self._warned_no_locals = True
        
       
        if WANDB_AVAILABLE and self.n_calls % self.eval_freq == 0:
            pass

        # Curriculum-Logs nach WandB: immer aktuelle Metriken + ggf. neue Events (pending)
        if self.curriculum_callback is not None:
            # Jeden Step: aktuelle Stufe, Target, rollierende Raten (damit WandB-Charts erscheinen)
            to_log = self.curriculum_callback.get_curriculum_metrics_for_wandb()
            pending = getattr(self.curriculum_callback, "pending_log_dict", None)
            if pending:
                to_log.update(pending)
                self.curriculum_callback.pending_log_dict = {}
            if to_log:
                # TensorBoard logging first, damit sync_tensorboard/WandB-Bridge die Werte sieht
                for key, value in to_log.items():
                    try:
                        self.logger.record(key, float(value))
                    except (TypeError, ValueError):
                        # Fallback: direkt loggen, falls es sich nicht in float casten lässt
                        self.logger.record(key, value)
                if WANDB_AVAILABLE:
                    wandb.log(to_log, step=self.num_timesteps)
        
        # Speichere Environment wenn Best Model gefunden
        if self.best_model_save_path and hasattr(self, 'best_mean_reward'):
            env_path = os.path.join(self.best_model_save_path, "best_env.pkl")
            if hasattr(self.training_env, 'save'):
                self.training_env.save(env_path)
        
        return result
    
    def _on_rollout_end(self) -> None:
        """Wird am Ende jedes Rollouts aufgerufen - loggt Reward-Komponenten"""
        log_dict = {}
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            log_dict = {
                "rollout/ep_reward_mean": ep_info.get("r", 0),
                "rollout/ep_length_mean": ep_info.get("l", 0),
            }
           # ---- Train Success Rate -----------------------------------------------
            # Pro-Target rollende Metrik: Curriculum nutzt pro Target die letzten N Episoden,
            # in denen dieses Target ausgewählt war. Gleiche Logik für Logging → Radius-Reduktion
            # bei >90% basiert auf denselben Metriken wie in WandB sichtbar.
            if self.curriculum_callback is not None and hasattr(
                self.curriculum_callback, "success_buffers"
            ):
                buffers = self.curriculum_callback.success_buffers
                rates = []
                for t, buf in enumerate(buffers):
                    if len(buf) > 0:
                        rate = sum(buf) / len(buf)
                        log_dict[f"train/sr_T{t + 1}"] = float(rate)
                        log_dict[f"train/nbuf_T{t + 1}"] = int(len(buf))
                        rates.append(rate)
                if rates:
                    log_dict["train/success_rate"] = float(np.mean(rates))
                    log_dict["train/nbuf_total"] = int(sum(len(buf) for buf in buffers if len(buf) > 0))

    #------------------------------------------------------------------------------------------------
    # Ab hier nutzloser Code, der nichts gescheites ausgibt / Rewards-Komponenten werden nicht geloggt
    #------------------------------------------------------------------------------------------------
        # Logge Reward-Komponenten (Summen über gesammelte Steps)
        buffer_has_data = any(len(v) > 0 for v in self.reward_components_buffer.values())
        if buffer_has_data:
            for key, values in self.reward_components_buffer.items():
                if len(values) > 0:
                    log_dict[f"reward_components/{key}_sum"] = np.sum(values)
            # Print für Debugging
            if self.verbose > 0:
                print(f"[Reward Components] Rollout {self.num_timesteps}: Buffer gefüllt, logge {len([k for k, v in self.reward_components_buffer.items() if len(v) > 0])} Komponenten")
        else:
            # Print wenn Buffer leer
            if self.verbose > 0:
                buffer_sizes = {k: len(v) for k, v in self.reward_components_buffer.items()}
                print(f"[Reward Components] Rollout {self.num_timesteps}: ⚠️  Buffer LEER! Sizes: {buffer_sizes}")

        if self.debug_reward_components:
            log_dict["debug/steps_total"] = self.debug_counts["steps_total"]
            log_dict["debug/steps_with_components"] = self.debug_counts["steps_with_components"]
            log_dict["debug/steps_with_debug_flag"] = self.debug_counts["steps_with_debug_flag"]
            log_dict["debug/infos_empty"] = self.debug_counts["infos_empty"]
            # Print Debug-Info
            if self.verbose > 0:
                print(f"[DEBUG] Steps: total={self.debug_counts['steps_total']}, "
                      f"with_components={self.debug_counts['steps_with_components']}, "
                      f"with_debug_flag={self.debug_counts['steps_with_debug_flag']}, "
                      f"infos_empty={self.debug_counts['infos_empty']}")

        # Schreibe in SB3 Logger (TensorBoard)
        if len(log_dict) > 0:
            for key, value in log_dict.items():
                self.logger.record(key, value)

        if WANDB_AVAILABLE and len(log_dict) > 0:
            reward_keys = [k for k in log_dict.keys() if k.startswith("reward_components/")]
            if reward_keys:
                if self.verbose > 0:
                    print(f"[WandB] Logge {len(reward_keys)} reward_components Keys zu WandB")
            wandb.log(log_dict, step=self.num_timesteps)
        elif not WANDB_AVAILABLE:
            if self.verbose > 0 and buffer_has_data:
                print(f"[WandB] WandB nicht verfügbar, kann reward_components nicht loggen")

        # Buffer nach Rollout leeren
        for key in self.reward_components_buffer:
            self.reward_components_buffer[key] = []
        if self.debug_reward_components:
            for key in self.debug_counts:
                self.debug_counts[key] = 0
        
        return super()._on_rollout_end()


class RewardComponentsCallback(BaseCallback):
    """
    Sammelt Reward-Komponenten aus info["reward_components"] pro Step und loggt
    die Summen pro Rollout zu TensorBoard (und optional WandB).
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.reward_components_buffer = {
            "reach": [],
            "bonus": [],
            "penalty": [],
            "dense": [],
            "sparse": [],
            "reach_dist": [],
        }

    def _on_step(self) -> bool:
        if not hasattr(self, "locals") or self.locals is None:
            return True
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "reward_components" in info:
                components = info["reward_components"]
                for key in self.reward_components_buffer:
                    if key in components:
                        self.reward_components_buffer[key].append(components[key])
        return True

    def _on_rollout_end(self) -> None:
        buffer_has_data = any(len(v) > 0 for v in self.reward_components_buffer.values())
        if buffer_has_data:
            for key, values in self.reward_components_buffer.items():
                if len(values) > 0:
                    self.logger.record(f"reward_components/{key}_sum", np.sum(values))
            if self.verbose > 0:
                n = len([k for k, v in self.reward_components_buffer.items() if len(v) > 0])
                print(f"[Reward Components] Rollout {self.num_timesteps}: logged {n} components")
        if WANDB_AVAILABLE and buffer_has_data:
            log_dict = {}
            for key, values in self.reward_components_buffer.items():
                if len(values) > 0:
                    log_dict[f"reward_components/{key}_sum"] = float(np.sum(values))
            if log_dict:
                wandb.log(log_dict, step=self.num_timesteps)
        for key in self.reward_components_buffer:
            self.reward_components_buffer[key] = []
        return super()._on_rollout_end()


class CurriculumCallback(BaseCallback):
    """
    Curriculum Learning: feste Targets (FIXED_TARGETS), Soft-Worst-First-Sampling.

    Pro Target rollierender Success-Buffer; Gewichte (1-r)^alpha mit Under-Sampling-Boost,
    epsilon-Uniform-Mischung. Pro Env-Reset wird ein Target nach diesen Gewichten gezogen
    (kein globales Argmin — vermeidet Aushungern bei r=1).
    """

    # Option A: 4 feste Targets (z.B. für einfache Tests)
    TARGETS_4 = [
        np.array([0.5, 0.5]),    # Target 1
        np.array([-0.5, -0.5]),  # Target 2
        np.array([0.5, -0.5]),   # Target 3
        np.array([-0.5, 0.5]),   # Target 4
    ]

    # Option B: 20 verschiedene Targets im vollen Raum [-1, 1]
    TARGETS_20 = [
        np.array([0.82, 0.71]), np.array([-0.91, 0.58]), np.array([0.14, -0.87]), np.array([-0.76, -0.43]),
        np.array([0.61, 0.94]), np.array([-0.38, -0.72]), np.array([0.93, -0.29]), np.array([-0.55, 0.81]),
        np.array([0.27, 0.19]), np.array([-0.84, -0.16]), np.array([0.68, -0.63]), np.array([-0.22, 0.47]),
        np.array([0.45, -0.51]), np.array([-0.67, 0.33]), np.array([0.08, 0.78]), np.array([-0.96, -0.89]),
        np.array([0.53, 0.41]), np.array([-0.49, -0.56]), np.array([0.79, -0.12]), np.array([-0.11, 0.65]),
    ]

    # —— Welche Target-Liste benutzen? Eine Zeile aktiv lassen, die andere auskommentieren ——
    # FIXED_TARGETS = TARGETS_4
    FIXED_TARGETS = TARGETS_20
    
    def __init__(
        self,
        verbose=1,
        initial_target_radius=0.2,
        min_target_radius=0.02,
        radius_step=0.02,
        success_buffer_size=50,
        success_rate_threshold=0.9,
        cooldown=5000,
        sampling_alpha=1.0,
        sampling_eps=0.05,
        sampling_n_min=5,
        sampling_under_boost=0.5,
        use_adaptive_button_sampling=True,
    ):
        super().__init__(verbose)
        self.initialized = False
        self.rng = np.random.default_rng()
        self.initial_target_radius = float(initial_target_radius)
        self.min_target_radius = float(min_target_radius)
        self.radius_step = float(radius_step)
        self.success_buffer_size = int(success_buffer_size)
        self.success_rate_threshold = float(success_rate_threshold)
        self.cooldown = int(cooldown)
        self.sampling_alpha = float(sampling_alpha)
        self.sampling_eps = float(sampling_eps)
        self.sampling_n_min = int(sampling_n_min)
        self.sampling_under_boost = float(sampling_under_boost)
        self._use_adaptive_button_sampling = bool(use_adaptive_button_sampling)
        self._sampling_weights = None  # list[float], length len(FIXED_TARGETS)
        # Pro Target: FIFO-Buffer (1=gelöst, 0=nicht gelöst), max 50 Einträge
        self.success_buffers = [
            deque(maxlen=self.success_buffer_size) for _ in self.FIXED_TARGETS
        ]
        self.target_radius_by_target = [
            self.initial_target_radius for _ in self.FIXED_TARGETS
        ]
        # Buffer vollständig 0 → Target exkludiert bis Step disabled_until_step[i]
        self.disabled_until_step = {}
        # Für WandB: zu loggende Werte (werden vom WandBEvalCallback ausgelesen)
        self.pending_log_dict = {}

    def _append_pending_log(self, d: dict):
        """Hängt Werte an pending_log_dict an (WandB-Logging passiert im WandBEvalCallback)."""
        if d:
            self.pending_log_dict.update(d)

    def _compute_sampling_weights(self):
        """
        Soft worst-first weights from success_buffers: w_i ∝ (1-r_i)^alpha, under_boost if n < n_min,
        then mix with uniform (eps).
        """
        n_buttons = max(1, len(self.FIXED_TARGETS))
        alpha = float(self.sampling_alpha)
        eps = float(self.sampling_eps)
        n_min = int(self.sampling_n_min)
        under_boost = float(self.sampling_under_boost)

        w = np.zeros(n_buttons, dtype=np.float64)
        for i in range(n_buttons):
            buf = self.success_buffers[i]
            n = len(buf)
            s = int(sum(buf))
            r = (s / n) if n > 0 else 0.0
            wi = (1.0 - r) ** alpha
            if n < n_min:
                wi += under_boost
            w[i] = max(wi, 1e-12)

        if w.sum() <= 1e-12:
            w[:] = 1.0
        p = w / w.sum()
        p = (1.0 - eps) * p + eps * (1.0 / n_buttons)
        p = p / p.sum()
        return p.tolist()

    def _refresh_sampling_weights(self):
        self._sampling_weights = self._compute_sampling_weights()

    def sample_target_idx(self, visible_indices=None):
        """
        Pick one target index. Uses _sampling_weights (soft worst-first) if enabled and valid;
        otherwise uniform. Cooldown targets excluded from default visible set (fallback: all).
        """
        n_buttons = len(self.FIXED_TARGETS)
        if not self._use_adaptive_button_sampling:
            return int(self.rng.integers(0, n_buttons))

        if visible_indices is None:
            now = self.num_timesteps
            visible_indices = [
                i for i in range(n_buttons)
                if now >= self.disabled_until_step.get(i, -1)
            ]
            if len(visible_indices) == 0:
                if self.verbose > 0:
                    print("Curriculum: Alle Targets in Cooldown – Sampling uniform über alle.")
                visible_indices = list(range(n_buttons))

        if (
            self._sampling_weights is None
            or len(self._sampling_weights) < max(visible_indices) + 1
        ):
            return int(self.rng.choice(visible_indices))

        w = np.asarray(
            [self._sampling_weights[i] for i in visible_indices],
            dtype=np.float64,
        )
        if not np.isfinite(w).all() or w.sum() <= 1e-12:
            return int(self.rng.choice(visible_indices))
        p = w / w.sum()
        return int(self.rng.choice(visible_indices, p=p))

    def _on_training_start(self) -> None:
        """Initialisiert Sampling-Gewichte und Env-Referenzen."""
        if not self.initialized:
            self._refresh_sampling_weights()
            self._update_environments()
            self.initialized = True
        if self.verbose > 0:
            print(
                f"Curriculum initialized: {len(self.FIXED_TARGETS)} feste Targets, "
                f"soft worst-first (alpha={self.sampling_alpha}, eps={self.sampling_eps}, "
                f"n_min={self.sampling_n_min}, under_boost={self.sampling_under_boost})"
            )

    def _on_step(self) -> bool:
        """Gewichte werden in record_episode_result aktualisiert; kein globales Target-Wechseln hier."""
        return True

    def get_target_radius(self, target_idx: int = None):
        """Radius für target_idx; ohne Index: initialer Radius (kein globales „aktuelles“ Target mehr)."""
        if target_idx is None:
            return self.initial_target_radius
        return float(self.target_radius_by_target[target_idx])

    def get_curriculum_metrics_for_wandb(self):
        """
        Liefert Curriculum-Metriken für WandB (kein einzelnes globales Target bei VecEnv).
        """
        out = {}
        if self._sampling_weights is not None and len(self._sampling_weights) == len(
            self.FIXED_TARGETS
        ):
            p = np.asarray(self._sampling_weights, dtype=np.float64)
            p = p / p.sum()
            ent = float(-np.sum(p * np.log(p + 1e-20)))
            out["sampling/entropy"] = ent
            for k in range(min(4, len(p))):
                out[f"curriculum/w_{k + 1}"] = float(p[k])

        rates_all = [
            sum(buf) / len(buf) for buf in self.success_buffers
            if len(buf) > 0
        ]
        if rates_all:
            out["curriculum/sr_mean"] = sum(rates_all) / len(rates_all)
        for i, buf in enumerate(self.success_buffers):
            if len(buf) > 0:
                out[f"curriculum/sr_T{i + 1}"] = sum(buf) / len(buf)
        return out

    def record_episode_result(self, success: bool, target_idx: int = None):
        """
        Speichert pro Target einen Eintrag (1=gelöst, 0=nicht gelöst) in einem FIFO-Buffer (Größe 50).
        Nach 50 Einträgen: Success-Rate = Mittelwert. Bei Rate > 70%: Radius verringern, Buffer leeren.
        """
        if target_idx is not None:
            buf = self.success_buffers[target_idx]
            buf.append(1 if success else 0)

            if len(buf) >= self.success_buffer_size:
                rate = sum(buf) / self.success_buffer_size

                if rate == 0.0:
                    until = self.num_timesteps + self.cooldown
                    self.disabled_until_step[target_idx] = until
                    self.success_buffers[target_idx] = deque(maxlen=self.success_buffer_size)  # reset
                    t = self.FIXED_TARGETS[target_idx]
                    print(
                        f"Curriculum: Target {target_idx + 1} (Position [{t[0]:.2f}, {t[1]:.2f}]): "
                        f"Buffer vollständig mit 0 gefüllt – exkludiert bis Step {until} (Cooldown {self.cooldown})."
                    )

                if rate > self.success_rate_threshold:
                    old_radius = self.target_radius_by_target[target_idx]
                    new_radius = max(
                        self.min_target_radius,
                        old_radius - self.radius_step,
                    )
                    self.target_radius_by_target[target_idx] = new_radius
                    if new_radius < old_radius:
                        self.success_buffers[target_idx] = deque(maxlen=self.success_buffer_size)
                    if self.verbose > 0:
                        buf_note = ", Buffer geleert" if new_radius < old_radius else ""
                        print(
                            f"Target {target_idx + 1}: Success-Rate {rate:.0%} > {self.success_rate_threshold:.0%} → "
                            f"Radius {old_radius:.2f} → {new_radius:.2f}{buf_note}"
                        )
                    self._append_pending_log({
                        f"curriculum/sr_T{target_idx + 1}": rate,
                        f"curriculum/rad_T{target_idx + 1}": new_radius,
                    })
                else:
                    self._append_pending_log({
                        f"curriculum/sr_T{target_idx + 1}": rate,
                    })

            # Allgemeine Reach-Success-Rate: Mittelwert über alle Targets (nur wo Buffer Einträge hat)
            rates_all = [
                sum(buf) / len(buf) for buf in self.success_buffers
                if len(buf) > 0
            ]
            if rates_all:
                self._append_pending_log({
                    "curriculum/sr_mean": sum(rates_all) / len(rates_all),
                })
            self._refresh_sampling_weights()
        else:
            print("[CURRICULUM] target_idx is None -> not recording success")

    def _update_environments(self):
        """Setzt Referenz zum Curriculum Callback in allen Environments"""
        base_env = self.training_env
        if hasattr(base_env, 'venv'):
            base_env = base_env.venv
        
        if hasattr(base_env, 'envs'):
            for env in base_env.envs:
                unwrapped = env
                while hasattr(unwrapped, 'env'):
                    unwrapped = unwrapped.env
                
                # Setze Referenz zum Curriculum Callback
                unwrapped.curriculum_callback = self

        if self._sampling_weights is not None and len(self._sampling_weights) == len(
            self.FIXED_TARGETS
        ):
            p = np.asarray(self._sampling_weights, dtype=np.float64)
            p = p / p.sum()
            log_sampling = {"sampling/entropy": float(-np.sum(p * np.log(p + 1e-20)))}
            for k in range(min(4, len(p))):
                log_sampling[f"curriculum/w_{k + 1}"] = float(p[k])
            self._append_pending_log(log_sampling)



class CheckpointCallback(BaseCallback):
    """simple checkpoint callback, saves periodically"""
    
    def __init__(
        self,
        save_freq=50_000,
        save_path="logs",
        verbose=1,
        generate_heatmap=False,
        heatmap_kde=False,
        heatmap_bins=50,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.generate_heatmap = generate_heatmap
        self.heatmap_kde = heatmap_kde
        self.heatmap_bins = heatmap_bins
        os.makedirs(save_path, exist_ok=True)

    def _generate_heatmap(self, num_timesteps: int) -> None:
        """Generate a joystick heatmap snapshot for this checkpoint."""
        try:
            from visualize_joystick_heatmap import (
                load_joystick_data,
                create_heatmap,
                create_kde_heatmap,
                log_dir_title,
            )
        except Exception as exc:
            if self.verbose > 0:
                print(f"Heatmap generation skipped (import failed): {exc}")
            return

        all_positions, final_positions, all_targets, episode_targets, _ = load_joystick_data(self.save_path)
        if all_positions is None or len(all_positions) == 0:
            if self.verbose > 0:
                print("Heatmap generation skipped (no joystick data yet).")
            return

        output_suffix = "kde" if self.heatmap_kde else "hist"
        output_name = f"joystick_heatmap_checkpoint_{num_timesteps}_{output_suffix}.png"
        output_path = os.path.join(self.save_path, output_name)
        fig_title = log_dir_title(self.save_path, kde=self.heatmap_kde)

        if self.heatmap_kde:
            create_kde_heatmap(
                all_positions,
                final_positions=final_positions,
                all_targets=all_targets,
                episode_targets=episode_targets,
                save_path=output_path,
                title=fig_title,
                show_plot=False,
            )
        else:
            create_heatmap(
                all_positions,
                final_positions=final_positions,
                all_targets=all_targets,
                episode_targets=episode_targets,
                bins=self.heatmap_bins,
                save_path=output_path,
                title=fig_title,
                show_plot=False,
            )
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}")
            self.model.save(checkpoint_path)
            if hasattr(self.training_env, 'save'):
                self.training_env.save(os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}_env.pkl"))
            if self.verbose > 0:
                print(f"Checkpoint saved at {checkpoint_path}")
            if self.generate_heatmap:
                self._generate_heatmap(self.num_timesteps)
        return True


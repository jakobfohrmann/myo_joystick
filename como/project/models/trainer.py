"""Vereinfachte Trainer-Klasse für einfaches Training"""
import json
import os
from dataclasses import dataclass, field
from typing import List
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


@dataclass
class SimpleTrainer:
    
    envs: VecNormalize
    env_config: dict
    load_model_path: str = None
    log_dir: str = "logs"
    model_config: dict = field(default_factory=dict)
    callbacks: List[BaseCallback] = field(default_factory=list)
    timesteps: int = 1_000_000
    
    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.dump_env_config()
        self.agent = self._init_agent()
    
    def dump_env_config(self):
        """Speichert Config als JSON für Reproduzierbarkeit"""
        with open(os.path.join(self.log_dir, "env_config.json"), "w", encoding="utf8") as f:
            json.dump(self.env_config, f, indent=2)
    
    def _init_agent(self) -> PPO:
        """Initialisiert oder lädt PPO Agent"""
        if self.load_model_path and os.path.exists(self.load_model_path):
            print(f"Loading model from {self.load_model_path}")
            load_kw = dict(
                env=self.envs,
                tensorboard_log=self.log_dir,
            )
            seed = self.model_config.get("seed")
            if seed is not None:
                load_kw["seed"] = seed
            return PPO.load(self.load_model_path, **load_kw)
        
        print("Creating new PPO model")
        return PPO(
            "MlpPolicy", 
            self.envs,
            verbose=1,
            tensorboard_log=self.log_dir,
            **self.model_config,
        )
    
    def train(self, total_timesteps: int = None):
        """Training starten"""
        timesteps = total_timesteps or self.timesteps
        self.agent.learn(
            total_timesteps=timesteps,
            callback=self.callbacks,
            reset_num_timesteps=True,
        )
    
    def save(self):
        """Model und Environment speichern"""
        self.agent.save(os.path.join(self.log_dir, "final_model"))
        self.envs.save(os.path.join(self.log_dir, "final_env.pkl"))
        print(f" Model saved to {self.log_dir}")



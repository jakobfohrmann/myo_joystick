"""Environment Registration für Thumb Reach Task"""
import gymnasium as gym
import os

# Pfad zum XML Model (relativ zu diesem File)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(ROOT_DIR, "xml", "controller_with_hand.xml")

# Registriere ThumbReach Environment
gym.register(
    id="ThumbReach-v0",
    entry_point="envs.thumb_reach:ThumbReachEnvV0",
    max_episode_steps=250,  # 5 Sekunden bei frame_skip=10, dt=0.002
    kwargs={
        "model_path": model_path,
        "normalize_act": True,
        "frame_skip": 10,
    },
)
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from src.observation import MissionToArrayWrapper
from model.VLA_feature_extractor import VLAFeatureExtractor
from transformers import PreTrainedTokenizerFast
import minigrid

from src.hyperparam import *

env = gym.make(env_ids, render_mode='human')  # human 모드
env = MissionToArrayWrapper(env, tokenizer, mission_max_length)

model = PPO.load(f"model/save_model/8x8_model_{test_learning_steps}.zip", env=env, device='cuda')  # 또는 'cpu'

while True:
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
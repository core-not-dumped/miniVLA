import torch
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from src.observation import MissionToArrayWrapper
from model.feature_extractor import VLAFeatureExtractor
from transformers import PreTrainedTokenizerFast
import minigrid

from src.hyperparam import *
from src.env import *

env = RandomMiniGridEnv(env_ids=env_ids, render_human=True)
env = MissionToArrayWrapper(env, tokenizer, mission_max_length)

model = DQN.load(f"model/save_model/8x8_model_{test_learning_steps}.zip", env=env, device='cuda')  # 또는 'cpu'

total_reward = 0
episode = 0
while True:
    episode += 1
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f'epi_rew_mean = {total_reward / episode}')
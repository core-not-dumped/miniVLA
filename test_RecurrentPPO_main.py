import torch
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from src.observation import MissionToArrayWrapper
from model.feature_extractor import VLAFeatureExtractor
from transformers import PreTrainedTokenizerFast
import minigrid

from src.hyperparam_RecurrentPPO import *
from src.env import *

env = RandomMiniGridEnv(env_ids=env_ids, max_len=max_len, frame_num=recurrent_frame_num, render_human=True)
env = MissionToArrayWrapper(env, tokenizer, mission_max_length, recurrent_frame_num*3)

model = RecurrentPPO.load(f"model/save_model/8x8_model_RecurrentPPO_{test_learning_steps}_0.zip", env=env, device='cuda')  # 또는 'cpu'

total_reward = 0
episode = 0
while True:
    episode += 1
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        done = terminated or truncated
        total_reward += reward
    print(f'epi_rew_mean = {total_reward / episode}')
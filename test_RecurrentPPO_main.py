import torch
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from src.observation import MissionToArrayWrapper
from model.feature_extractor import VLAFeatureExtractor
from transformers import PreTrainedTokenizerFast
import minigrid
import time
import glfw

from src.hyperparam_RecurrentPPO import *
from src.env import *

env = RandomMiniGridEnv(env_ids=env_ids, max_len=max_len, frame_num=recurrent_frame_num, scale=scale, render_human=True)
env = MissionToArrayWrapper(env, tokenizer, mission_max_length, recurrent_frame_num*3)

model = RecurrentPPO.load(f"model/save_model/8x8_model_RecurrentPPO_{test_learning_steps}.zip", env=env, device='cuda')  # 또는 'cpu'

episode = 0
total_reward = 0
episode_starts = np.ones((1,), dtype=bool)
while True:
    episode += 1
    time.sleep(1)
    obs, _ = env.reset()
    states, episode_starts[:] = None, True
    done = False
    epi_reward = 0
    while not done:
        action, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action[0])
        done = terminated or truncated
        
        total_reward += reward
        epi_reward += reward
        
        episode_starts[:] = done
    print(f'{epi_reward = }')
    print(f'epi_rew_mean = {total_reward / episode}')
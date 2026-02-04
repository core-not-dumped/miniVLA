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
import os

from src.hyperparam_RecurrentPPO import *
from src.env import *

env = RandomMiniGridEnv(env_ids=test_spe_env_id, max_len=200, frame_num=recurrent_frame_num, scale=scale, render_human=False)
env = MissionToArrayWrapper(env, tokenizer, mission_max_length, recurrent_frame_num*3)

with open('Recurrent_PPO_spe_test.txt', 'a') as f:
    f.write(f'{test_spe_env_id = }\n')

model_names = [
    f"model/save_model/8x8_model_RecurrentPPO_{train_learning_steps*(i+1)}_9.zip" for i in 10
]
for i, model_name in enumerate(range(model_names)):
    if not os.path.exists(model_name):
        print(f"{model_name} 모델이 존재하지 않습니다.")
        exit(0)
    model = RecurrentPPO.load(model_name, env=env, device='cuda')  # 또는 'cpu'

    episode = 0
    total_reward = 0
    episode_starts = np.ones((1,), dtype=bool)
    while True:
        episode += 1
        time.sleep(1)
        obs, _ = env.reset()
        states, episode_starts[:] = None, True
        done = False
        while not done:
            action, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            total_reward += reward
            episode_starts[:] = done
        if episode == 500:
            with open('Recurrent_PPO_spe_test.txt', 'a') as f:
                f.write(f'{i+1}M epi_rew_mean = {total_reward / episode}\n')
                print(f'epi_rew_mean = {total_reward / episode}')
            break
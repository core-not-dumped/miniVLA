import torch
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from src.observation import MissionToArrayWrapper
from model.feature_extractor import VLAFeatureExtractor
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import minigrid
import time
import glfw
import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from src.hyperparam_RecurrentPPO import *
from src.env import *

# Sampling rejected 메시지만 무시
import builtins
original_print = builtins.print
def print_override(*args, **kwargs):
    if args and "Sampling rejected" in str(args[0]):
        return
    original_print(*args, **kwargs)
builtins.print = print_override

def make_custom_env():
    env = RandomMiniGridEnv(env_ids=test_spe_env_id, max_len=200, frame_num=recurrent_frame_num, scale=0, render_human=False)
    env = MissionToArrayWrapper(env, tokenizer, mission_max_length, recurrent_frame_num*3)
    return env
env = make_vec_env(make_custom_env, n_envs=num_cpu)

with open('Recurrent_PPO_spe_test.txt', 'a') as f:
    f.write(f'{test_spe_env_id = }\n')

model_names = [
    f"model/save_model/8x8_model_RecurrentPPO_{train_learning_steps*(i+1)}.zip" for i in range(70)
]

total_rewards = []
total_episodes = []
total_successes = []
for i, model_name in enumerate(model_names):
    if not os.path.exists(model_name):
        print(f"{model_name} 모델이 존재하지 않습니다.")
        break
    model = RecurrentPPO.load(model_name, env=env, device='cuda')  # 또는 'cpu'

    print(f'{i+1}M test')

    total_reward = 0
    total_episode = 0
    total_success = 0
    states, episode_starts = None, np.ones((num_cpu,), dtype=bool)
    obs = env.reset()
    with tqdm(total=spe_test_episodes) as pbar:
        while True:
            action, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            episode_starts[:] = dones
            total_episode += np.sum(dones)
            total_reward += reward.sum()
            total_success += np.sum((reward > 0) & (dones > 0))
            pbar.set_postfix(value=f"{total_episode:.4f}")
            pbar.update(np.sum(dones))
            if total_episode >= spe_test_episodes:    break
    with open('Recurrent_PPO_spe_test.txt', 'a') as f:
        f.write(f'{i+1}M success_rate = {total_success / total_episode}\n')
        print(f'epi_rew_mean = {total_reward / total_episode:.2}')
        print(f'success_rate = {total_success / total_episode:.2%}')
    total_episodes.append(total_episode)
    total_rewards.append(total_reward)
    total_successes.append(total_success)

values = np.round(np.array(total_successes)/np.array(total_episodes), 2)

# x축: 1M, 2M, 3M, ...
x = np.arange(1, len(values) + 1)

plt.figure(figsize=(8, 4))
plt.plot(x, values)

# x축을 5M 단위로 표시
xticks = np.arange(5, len(values) + 1, 5)
plt.xticks(xticks, [f"{i}M" for i in xticks])

plt.xlabel("Steps")
plt.ylabel("Success Rate")
plt.title("Training Curve")

plt.grid(True)
plt.tight_layout()

plt.savefig("./test/training_curve.png", dpi=150)
plt.close()
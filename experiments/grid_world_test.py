import gymnasium as gym
import minigrid
from src.env import *
from src.hyperparam import *

env = RandomMiniGridEnv(env_ids=env_ids, render=True)#,"MiniGrid-DoorKey-8x8-v0","MiniGrid-RedBlueDoors-8x8-v0"], render=True)
obs, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs['image'].shape)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
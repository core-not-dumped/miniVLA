import gymnasium as gym
from minigrid.wrappers import PositionBonus
import time
from src.env import ScaledPositionBonus

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode='human')
env_bonus = ScaledPositionBonus(env, scale=0.01)
obs, _ = env_bonus.reset(seed=0)
obs, reward, terminated, truncated, info = env_bonus.step(2)
print(reward)
obs, reward, terminated, truncated, info = env_bonus.step(2)
print(reward)
obs, reward, terminated, truncated, info = env_bonus.step(1)
print(reward)
obs, reward, terminated, truncated, info = env_bonus.step(2)
print(reward)
obs, reward, terminated, truncated, info = env_bonus.step(1)
print(reward)
obs, reward, terminated, truncated, info = env_bonus.step(1)
print(reward)
obs, reward, terminated, truncated, info = env_bonus.step(2)
print(reward)
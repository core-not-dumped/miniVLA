import gymnasium as gym
import minigrid
import minigrid.envs  # 이 줄이 핵심

for env_id in gym.envs.registry.keys():
    if env_id.startswith("MiniGrid-") and "8x8" in env_id:
        print(env_id)

print()

for env_id in gym.envs.registry.keys():
    if env_id.startswith("MiniGrid-"):
        print(env_id)
    if env_id.startswith("BabyAI-"):
        print(env_id)
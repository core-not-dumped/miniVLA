import torch
import gymnasium as gym
import numpy as np

class MissionToArrayWrapper(gym.ObservationWrapper):
    def __init__(self, env, tokenizer, max_length, channels):
        super().__init__(env)
        obs_space = env.observation_space
        self.tokenizer = tokenizer
        self.observation_space = gym.spaces.Dict(obs_space.spaces.copy())
        self.observation_space.spaces['image'] = gym.spaces.Box(
            low=0, high=8, shape=(channels, 7, 7), dtype=np.uint8
        )
        self.observation_space.spaces['mission'] = gym.spaces.Box(
            low=0, high=self.tokenizer.vocab_size-1, shape=(max_length,), dtype=np.int32
        )
        self.observation_space.spaces['direction'] = gym.spaces.Box(
            low=0, high=3, shape=(channels//3,), dtype=np.int32
        )
        self.max_length = max_length

    def observation(self, obs):
        obs = obs.copy()
        mission_str = obs['mission']
        token_ids = self.tokenizer.encode(mission_str, truncation=True, max_length=self.max_length, padding='max_length')
        obs['mission'] = torch.tensor(token_ids, dtype=torch.long)
        obs['direction'] = np.array([obs['direction']])
        obs['image'] = np.transpose(obs['image'], (2, 0, 1))
        return obs

import torch
import gymnasium as gym
import numpy as np

class MissionToArrayWrapper(gym.ObservationWrapper):
    def __init__(self, env, tokenizer, max_length):
        super().__init__(env)
        obs_space = env.observation_space
        self.tokenizer = tokenizer
        self.observation_space = gym.spaces.Dict(obs_space.spaces.copy())
        self.observation_space.spaces['mission'] = gym.spaces.Box(
            low=0, high=self.tokenizer.vocab_size-1, shape=(max_length,), dtype=np.int32
        )
        self.observation_space.spaces['direction'] = gym.spaces.Box(
            low=0, high=3, shape=(), dtype=np.int32
        )
        self.max_length = max_length

    def observation(self, obs):
        obs = obs.copy()
        mission_str = obs['mission']
        token_ids = self.tokenizer.encode(mission_str, truncation=True, max_length=self.max_length, padding='max_length')
        obs['mission'] = torch.tensor(token_ids, dtype=torch.long)
        return obs
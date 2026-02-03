import gymnasium as gym
import random
import numpy as np
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
from gymnasium.core import Wrapper
import math

from collections import deque


class RandomCurriculumMiniGridEnv(gym.Env):
    def __init__(self, env_ids, max_len=100, frame_num=4, rho=0.3, beta=0.1, scale=0.005, random_epi_num=1000, render_human=True):
        super().__init__()
        self.env_ids = env_ids
        self.n_envs = len(env_ids)
        self.render_human = render_human
        self.max_len = max_len
        self.frame_num=frame_num
        self.beta = beta
        self.scale = scale
        self.random_epi_num = random_epi_num

        # PLR tracking
        self.rho = rho              # PLR replay weight
        self.L_seen = []            # visited level
        self.S = []                 # score
        self.C = []                 # timestamp
        self.global_episode = 0     # 전체 에피소드 수

        self._make_new_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _sample_new_level(self):
        unseen = list(set(self.env_ids) - set(self.L_seen))
        if len(unseen) == 0:    return None
        li = np.random.choice(unseen)
        self.L_seen.append(li)
        self.S.append(deque([0.0], maxlen=200))
        self.C.append(0)
        return li

    def _sample_replay_level(self):
        if self.global_episode < self.random_epi_num:
            return np.random.choice(self.L_seen)

        # PS(l|S): score 기반 낮은 점수
        s_arr = np.array([np.mean(s) for s in self.S])
        Goldilocks = 1.0 - np.abs(2 * s_arr - 1.0)
        sorted_indices = np.argsort(-Goldilocks)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(Goldilocks)+1)
        h = 1.0 / ranks
        Ps = h ** self.beta # beta
        Ps /= Ps.sum()

        # 98퍼 이상 성공한 경우 확률 더 낮춤
        high_score_mask = s_arr >= 0.7
        linear_scale = (1.0 - s_arr) / 0.3
        linear_scale = np.clip(linear_scale, 0.0, 1.0)
        Ps[high_score_mask] *= linear_scale[high_score_mask]
        Ps /= Ps.sum()  # 다시 정규화

        # PC(l|C, c): 최근 방문한 레벨은 적게 replay
        C_arr = np.array(self.C)
        recency = self.global_episode - C_arr
        Pc = recency / recency.sum()

        probs = (1 - self.rho) * Ps + self.rho * Pc
        probs /= probs.sum()
        li = np.random.choice(self.L_seen, p=probs)

        if self.global_episode % 300 == 0:
            for s, env_name, prob in zip(s_arr, self.L_seen, probs):
                print(f'env: {env_name}, {s = }, {prob = }')
        return li

    def _make_new_env(self, li=None):
        if li is None:
            d = np.random.choice([0, 1], p=[0.3, 0.7])  # 30% 새 레벨, 70% replay
            if d == 0 or not self.L_seen:
                li = self._sample_new_level()
                if li is None:  li = self._sample_replay_level()
            else:
                li = self._sample_replay_level()

        self.env_idx = self.L_seen.index(li)
        self.env_id = li
        if self.render_human:   env = gym.make(self.env_id, render_mode='human', max_episode_steps=self.max_len)
        else:                   env = gym.make(self.env_id, max_episode_steps=self.max_len)
        self.env = env

    def _get_frame_obs(self, obs):
        obs['image'] = np.concatenate(self.image, axis=2)
        obs['direction'] = np.array(self.direction)
        obs['carry'] = np.array(self.carry)
        return obs

    def reset(self, **kwargs):
        self.global_episode += 1
        self._make_new_env()
        obs, info = self.env.reset(**kwargs)
        self.image = deque([obs['image']] * self.frame_num, maxlen=self.frame_num)
        self.direction = deque([obs['direction']] * self.frame_num, maxlen=self.frame_num)
        self.carry = [0, 0]
        self.counts = {}
        return self._get_frame_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.image.append(obs['image'])
        self.direction.append(obs['direction'])
        carrying = self.env.unwrapped.carrying
        self.carry = [OBJECT_TO_IDX[carrying.type], COLOR_TO_IDX[carrying.color]] if carrying else [0, 0]
        if terminated or truncated:
            self.S[self.env_idx].append(1 if reward > 0.3 else 0)
            self.C[self.env_idx] = self.global_episode
        if reward < 0:  reward = 0

        # new position reward
        tup = tuple(self.env.unwrapped.agent_pos)
        pre_count = 0
        if tup in self.counts:  pre_count = self.counts[tup]
        new_count = pre_count + 1
        self.counts[tup] = new_count
        bonus = 1 / math.sqrt(new_count)
        reward += bonus * self.scale

        return self._get_frame_obs(obs), reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)
    

class RandomMiniGridEnv(gym.Env):
    def __init__(self, env_ids, max_len=100, frame_num=4, scale=0.002, render_human=True):
        super().__init__()
        self.env_ids = env_ids
        self.render_human = render_human
        self.max_len = max_len
        self._make_new_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.frame_num = frame_num
        self.scale = scale

    def _make_new_env(self):
        self.env_id = random.choice(self.env_ids)
        if self.render_human:   env = gym.make(self.env_id, render_mode='human', max_episode_steps=self.max_len)
        else:                   env = gym.make(self.env_id, max_episode_steps=self.max_len)
        self.env = env

    def _get_frame_obs(self, obs):
        obs['image'] = np.concatenate(self.image, axis=2)
        obs['direction'] = np.array(self.direction)
        obs['carry'] = np.array(self.carry)
        return obs
    
    def reset(self, **kwargs):
        self._make_new_env()
        obs, info = self.env.reset(**kwargs)
        self.image = deque([obs['image']] * self.frame_num, maxlen=self.frame_num)
        self.direction = deque([obs['direction']] * self.frame_num, maxlen=self.frame_num)
        self.carry = [0, 0]
        self.counts = {}
        return self._get_frame_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.image.append(obs['image'])
        self.direction.append(obs['direction'])
        carrying = self.env.unwrapped.carrying
        self.carry = [OBJECT_TO_IDX[carrying.type], COLOR_TO_IDX[carrying.color]] if carrying else [0, 0]
        
        # new position reward
        tup = tuple(self.env.unwrapped.agent_pos)
        pre_count = 0
        if tup in self.counts:  pre_count = self.counts[tup]
        new_count = pre_count + 1
        self.counts[tup] = new_count
        bonus = 1 / math.sqrt(new_count)
        reward += bonus * self.scale

        return self._get_frame_obs(obs), reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)
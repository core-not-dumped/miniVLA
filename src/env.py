import gymnasium as gym
import random
import numpy as np
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
from gymnasium.core import Wrapper
import math

from collections import deque

# mokey patch for bug func in minigrid
import types
from minigrid.minigrid_env import MiniGridEnv
def safe_place_agent(self, i=None, j=None, rand_dir=True, max_tries=1000):
    if i is None:   i = self._rand_int(0, self.num_cols)
    if j is None:   j = self._rand_int(0, self.num_rows)
    room = self.room_grid[j][i]
    for _ in range(max_tries):
        MiniGridEnv.place_agent(self, room.top, room.size, rand_dir, max_tries=1000)
        front_cell = self.grid.get(*self.front_pos)
        if front_cell is None or front_cell.type == "wall":
            return self.agent_pos

    raise RecursionError("safe_place_agent: failed to place agent")


class RandomCurriculumMiniGridEnv(gym.Env):
    def __init__(self,
            env_ids,
            max_len=100,
            frame_num=4,
            rho=0.3,
            beta=0.1,
            scale=0.003,
            random_epi_num=1000,
            score_len=100,
            pickup_toggle_minus_reward=-0.005,
            step_minus_reward=-0.002,
            render_human=True
        ):
        super().__init__()
        self.env_ids = env_ids
        self.n_envs = len(env_ids)
        self.render_human = render_human
        self.max_len = max_len
        self.frame_num=frame_num
        self.beta = beta
        self.scale = scale
        self.random_epi_num = random_epi_num
        self.score_len = score_len
        self.pickup_toggle_minus_reward = pickup_toggle_minus_reward
        self.step_minus_reward = step_minus_reward

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
        self.S.append(deque([0.0] * self.score_len, maxlen=self.score_len))
        self.C.append(0)
        return li

    def _sample_replay_level(self):
        if self.global_episode < self.random_epi_num:
            return np.random.choice(self.L_seen)

        # PS(l|S): score 기반 낮은 점수, 가우시안 커널(간략화해서 distance, rank) 이용해서 score 계산
        s_arr = np.array([np.mean(s) for s in self.S])
        low_09 = s_arr[s_arr <= 0.9]
        s_mean = min(low_09.mean() if len(low_09) > 0 else 0.5, 0.5) # 최대 0.5로 잡음
        distance = (s_arr - s_mean) ** 2
        ranks = np.argsort(np.argsort(distance)) + 1 # 자기 자신의 랭크 값
        Ps = (1.0 / ranks) ** self.beta
        Ps /= Ps.sum()

        # 70퍼 이상 성공한 경우 확률 더 낮춤
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
                print(f'env: {env_name}, s = {float(s):.1%}, p = {float(prob):.2%}')
            print(f'success_rate = {float(s_arr.mean()):.2%}')
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
        self.env.unwrapped.place_agent = types.MethodType(safe_place_agent, self.env.unwrapped)
    
    def _get_frame_obs(self, obs):
        obs['image'] = np.concatenate(self.image, axis=2)
        obs['direction'] = np.array(self.direction)
        obs['carry'] = np.array(self.carry)
        return obs

    def reset(self, **kwargs):
        self.global_episode += 1
        # map을 만들지 못할경우, recursionerror나옴
        for _ in range(20):
            try:
                self._make_new_env()
                obs, info = self.env.reset(**kwargs)
                break
            except RecursionError:
                continue
        else:
            raise RuntimeError("Env reset failed repeatedly")

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

        # minus reward, every step
        if (not terminated):
            if  action == 3 or action == 5:
                # little minus loss for toggle, pickup
                reward += self.pickup_toggle_minus_reward
            else:
                # step minus reward
                reward += self.step_minus_reward

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
        self.env.unwrapped.place_agent = types.MethodType(safe_place_agent, self.env.unwrapped)

    def _get_frame_obs(self, obs):
        obs['image'] = np.concatenate(self.image, axis=2)
        obs['direction'] = np.array(self.direction)
        obs['carry'] = np.array(self.carry)
        return obs
    
    def reset(self, **kwargs):
        # map을 만들지 못할경우, recursionerror나옴
        for _ in range(20):
            try:
                self._make_new_env()
                obs, info = self.env.reset(**kwargs)
                break
            except RecursionError:
                continue
        else:
            raise RuntimeError("Env reset failed repeatedly")
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
    
# sencing data를 이용해서 어떻게 사람 대신 수행하는 ai를 만들 수 있을지 이 프로젝트와 연결하여 고민
# VLA같은 경우 V가 sencing data가 되고 L이 사람이 명령을 내리고 A가 AI가 action을 자동으로 수행하게 된다.
# 
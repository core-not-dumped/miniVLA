import gymnasium as gym
import random
import numpy as np

from collections import deque


# nstep_DQN과 반드시 같이 써야함!!
class NStepReturnWrapper(gym.Wrapper):
    def __init__(self, env, n_step=3):
        super().__init__(env)
        self.n_step = n_step
        self.nstep_buffer = deque(maxlen=n_step+1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.nstep_buffer.clear()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.nstep_buffer.append((self.last_obs, action, reward, terminated or truncated))
        info['nstep_buffer'] = list(self.nstep_buffer)
        return obs, reward, terminated, truncated, info

    
class RandomCurriculumMiniGridEnv(gym.Env):
    def __init__(self, env_ids, max_len=100, frame_num=4, rho=0.2, render_human=True):
        super().__init__()
        self.env_ids = env_ids
        self.n_envs = len(env_ids)
        self.render_human = render_human
        self.max_len = max_len
        self.frame_num=frame_num

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
        self.S.append(deque([0.0], maxlen=1000))
        self.C.append(0)
        return li

    def _sample_replay_level(self):
        # PS(l|S): score 기반 낮은 점수
        S_arr = np.array([sum(s) for s in self.S])
        difficulty = 1.0 - (S_arr / (S_arr.max() + 1e-8))  # 점수 낮으면 어렵다고 판단
        Ps = difficulty / difficulty.sum()

        # PC(l|C, c): 최근 방문한 레벨은 적게 replay
        C_arr = np.array(self.C)
        recency = self.global_episode - C_arr
        Pc = recency / recency.sum()

        probs = (1 - self.rho) * Ps + self.rho * Pc
        probs /= probs.sum()
        li = np.random.choice(self.L_seen, p=probs)
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
        return obs

    def reset(self, **kwargs):
        self.global_episode += 1
        self._make_new_env()
        obs, info = self.env.reset(**kwargs)
        self.image = deque([obs['image']] * self.frame_num, maxlen=self.frame_num)
        self.direction = deque([obs['direction']] * self.frame_num, maxlen=self.frame_num)
        return self._get_frame_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.image.append(obs['image'])
        self.direction.append(obs['direction'])
        if terminated or truncated:
            self.S[self.env_idx].append(reward)
            self.C[self.env_idx] = self.global_episode
        return self._get_frame_obs(obs), reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)
    

class RandomMiniGridEnv(gym.Env):
    def __init__(self, env_ids, max_len=100, frame_num=4, render_human=True):
        super().__init__()
        self.env_ids = env_ids
        self.render_human = render_human
        self._make_new_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_len = max_len
        self.frame_num = frame_num

    def _make_new_env(self):
        self.env_id = random.choice(self.env_ids)
        if self.render_human:   env = gym.make(self.env_id, render_mode='human', max_episode_steps=self.max_len)
        else:                   env = gym.make(self.env_id, max_episode_steps=self.max_len)
        self.env = env

    def _get_frame_obs(self):
        return np.concatenate(list(self.image), axis=2)

    def reset(self, **kwargs):
        self._make_new_env()
        obs, info = self.env.reset(**kwargs)
        self.image = deque([obs['image']] * self.frame_num, maxlen=self.frame_num)
        return self._get_frame_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        print(self.env.unwrapped.carrying)
        self.image.append(obs['image'])
        return self._get_frame_obs(obs), reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)
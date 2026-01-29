import gymnasium as gym
import random
import minigrid.core.world_object as obj

def find_goal(env):
    for x in range(env.width):
        for y in range(env.height):
            if isinstance(env.unwrapped.grid.get(x, y), obj.Goal):
                return (x, y)
    return None

class RandomMiniGridEnv(gym.Env):
    def __init__(self, env_ids, render=True):
        super().__init__()
        self.env_ids = env_ids
        self._make_new_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render = render

    def _make_new_env(self):
        self.env_id = random.choice(self.env_ids)
        if self.render: env = gym.make(self.env_id, render_mode='human', max_episode_steps=150)
        else:           env = gym.make(self.env_id, max_episode_steps=150)
        self.env = env

    def reset(self, **kwargs):
        self._make_new_env()
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)
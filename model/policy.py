from typing import Any, Union
from stable_baselines3.dqn.policies import MultiInputPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import DQN
from copy import deepcopy
import torch.nn as nn
import numpy as np

class DuelingQNetwork(nn.Module):
    def __init__(self, features_dim, action_dim):
        super().__init__()
        self.value_net = nn.Linear(features_dim, 1)
        self.adv_net   = nn.Linear(features_dim, action_dim)

    def forward(self, features):
        v = self.value_net(features)
        a = self.adv_net(features)
        return v + a - a.mean(dim=1, keepdim=True)
    
class MultiInputDuelingPolicy(MultiInputPolicy):
    def make_q_net(self):
        q_net = super().make_q_net()
        q_net.q_net = DuelingQNetwork(
            q_net.features_dim,
            q_net.action_space.n
        )
        return q_net
    
# nstep env wrapper와 같이 써야함 !!
class nstep_DQN(DQN):
    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])  # type: ignore[assignment]

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
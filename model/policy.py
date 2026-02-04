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
    
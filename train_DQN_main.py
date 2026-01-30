import wandb
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
import minigrid

from model.VLA_feature_extractor import *
from src.observation import *
from src.hyperparam import *
from src.callback import *
from src.env import *

def make_custom_env():
    env = RandomMiniGridEnv(env_ids=env_ids, render=False)
    env = MissionToArrayWrapper(env, tokenizer, mission_max_length)
    return env
env = make_vec_env(make_custom_env, n_envs=num_cpu)

features_extractor_class = VLAFeatureExtractor
features_extractor_kwargs = dict(
    features_dim=features_dim,
    vocab_size = tokenizer.vocab_size-1,
    device=device
)

policy_kwargs = dict(
    features_extractor_class = features_extractor_class,
    features_extractor_kwargs = features_extractor_kwargs,
    optimizer_class = torch.optim.Adam,
    normalize_images=False,
    net_arch = [128, 128],
)

model = DQN(
    env=env,
    policy="MultiInputPolicy",
    policy_kwargs=policy_kwargs,
    learning_rate=lr,
    buffer_size=buffer_size,
    learning_starts=learning_starts,
    batch_size=batch_size,
    gamma=gamma,
    device=device,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
)

name = 'grid_world'
run = wandb.init(project='grid_world', name=name)

for epoch in range(epochs):
    if epoch < exploration_mid_iter:
        def linear_decay_exploration_scheduler(progress_remaining):
            # mid_iter동안 1->0으로 감
            progress = (exploration_mid_iter - epoch - (1 - progress_remaining)) / exploration_mid_iter
            eps = exploration_mid_eps + (exploration_initial_eps - exploration_mid_eps) * progress
            return eps
    else:
        def linear_decay_exploration_scheduler(progress_remaining):
            # i가 mid_iter보다 크거나 같을 경우 1->0으로 감
            progress = (exploration_final_iter - (epoch - exploration_mid_iter) - (1 - progress_remaining)) / exploration_final_iter
            eps = exploration_final_eps + (exploration_mid_eps - exploration_final_eps) * progress
            eps = max(eps, exploration_final_eps)
            return eps
    model.exploration_schedule = linear_decay_exploration_scheduler
    model.learn(
        total_timesteps=train_learning_steps,
        callback=WandbCallbackcustom(num_cpu=num_cpu)
    )
    if epoch >= 1:  model.learning_starts = 0
    model.save(f"model/save_model/8x8_model_{(epoch+1)*train_learning_steps}")

wandb.finish()
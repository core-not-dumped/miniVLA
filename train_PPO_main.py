import wandb
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
import minigrid

from model.feature_extractor import *
from src.observation import *
from src.hyperparam import *
from src.callback import *
from src.env import *

def make_custom_env():
    env = RandomCurriculumMiniGridEnv(env_ids=env_ids, max_len=max_len, frame_num=frame_num, render_human=False)
    env = MissionToArrayWrapper(env, tokenizer, mission_max_length, frame_num*3)
    return env
env = make_vec_env(make_custom_env, n_envs=num_cpu)

features_extractor_class = VLAFeatureExtractor
features_extractor_kwargs = dict(
    features_dim=features_dim,
    vocab_size = tokenizer.vocab_size-1,
    start_channels=frame_num*3,
    device=device
)

policy_class = MultiInputActorCriticPolicy
policy_kwargs = dict(
    features_extractor_class = features_extractor_class,
    features_extractor_kwargs = features_extractor_kwargs,
    optimizer_class = torch.optim.Adam,
    net_arch = [256, 256],
    normalize_images=False,
)

model = PPO(
    env=env,
    policy=policy_class,
    policy_kwargs=policy_kwargs,
    learning_rate=lr,
    n_steps=n_steps,
    batch_size=batch_size,
    n_epochs=n_epochs,
    gamma=gamma,
    gae_lambda=gae_lambda,
    device=device,
    verbose=1,
)

name = f'PPO_{lr}_{batch_size}_{gamma}_{features_dim}'
run = wandb.init(project='grid_world', name=name)

for epoch in range(epochs):
    model.learn(
        total_timesteps=train_learning_steps,
        callback=WandbCallbackcustom(num_cpu=num_cpu, use_PPO=True)
    )
    model.save(f"model/save_model/8x8_model_{train_learning_steps}_{(epoch+1)*train_learning_steps}")

wandb.finish()
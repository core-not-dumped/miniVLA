import wandb
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
import minigrid

from model.feature_extractor import *
from src.observation import *
from src.hyperparam_RecurrentPPO import *
from src.callback import *
from src.env import *

def make_custom_env():
    env = RandomCurriculumMiniGridEnv(env_ids=env_ids, max_len=max_len, frame_num=recurrent_frame_num, beta=beta, scale=scale, random_epi_num=random_epi_num, render_human=False)
    env = MissionToArrayWrapper(env, tokenizer, mission_max_length, recurrent_frame_num*3)
    return env

env = make_vec_env(make_custom_env, n_envs=num_cpu)

features_extractor_class = VLAFeatureExtractor
features_extractor_kwargs = dict(
    features_dim=features_dim,
    vocab_size = tokenizer.vocab_size-1,
    start_channels=recurrent_frame_num*3,
    device=device
)

if retrain:
    model = RecurrentPPO.load(f"model/save_model/8x8_model_RecurrentPPO_{retrain_learning_steps}_{level-1}.zip", env=env, device='cuda')  # 또는 'cpu'
else:
    policy_class = RecurrentMultiInputActorCriticPolicy
    policy_kwargs = dict(
        features_extractor_class = features_extractor_class,
        features_extractor_kwargs = features_extractor_kwargs,
        optimizer_class = torch.optim.Adam,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        lstm_hidden_size=features_dim,
        normalize_images=False,
    )

    model = RecurrentPPO(
        env=env,
        policy=policy_class,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        device=device,
        verbose=1,
    )

name = f'RecurrentPPO_{lr}_{batch_size}_{gamma}_{features_dim}_{level}'
run = wandb.init(project='grid_world', name=name)

for epoch in range(epochs):
    if linear_decay_lr:
        def linear_decay_scheduler(progress_remaining):
            pres_lr = lr * (epochs - epoch - (1 - progress_remaining)) / epochs
            return pres_lr
        model.lr_schedule = linear_decay_scheduler
    model.learn(
        total_timesteps=train_learning_steps,
        callback=WandbCallbackcustom(num_cpu=num_cpu, use_PPO=True)
    )
    if retrain:
        model.save(f"model/save_model/8x8_model_RecurrentPPO_{(epoch+1)*train_learning_steps+retrain_learning_steps}_{level}")
    else:
        model.save(f"model/save_model/8x8_model_RecurrentPPO_{(epoch+1)*train_learning_steps}_{level}")

wandb.finish()
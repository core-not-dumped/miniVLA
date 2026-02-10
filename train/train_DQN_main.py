import wandb
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import minigrid

from model.feature_extractor import *
from model.policy import *
from src.observation import *
from src.hyperparam_DQN import *
from src.callback import *
from src.env import *

# Sampling rejected 메시지만 무시
import builtins
original_print = builtins.print
def print_override(*args, **kwargs):
    if args and "Sampling rejected" in str(args[0]):
        return
    original_print(*args, **kwargs)
builtins.print = print_override

def make_custom_env():
    env = RandomCurriculumMiniGridEnv(
        env_ids=env_ids,
        max_len=max_len,
        frame_num=recurrent_frame_num,
        beta=beta,
        scale=scale,
        random_epi_num=random_epi_num,
        score_len=score_len,
        pickup_toggle_minus_reward=pickup_toggle_minus_reward,
        step_minus_reward=step_minus_reward,
        render_human=False)
    env = MissionToArrayWrapper(env, tokenizer, mission_max_length, recurrent_frame_num*3)
    return env

env = make_vec_env(make_custom_env, n_envs=num_cpu)

features_extractor_class = VLAFeatureExtractor
features_extractor_kwargs = dict(
    features_dim=features_dim,
    vocab_size=tokenizer.vocab_size-1,
    start_channels=DQN_frame_num*3,
    device=device
)

if retrain:
    model = DQN.load(f"model/save_model/8x8_model_DQN_{retrain_learning_steps}.zip", env=env, device='cuda')  # 또는 'cpu'
    model.replay_buffer.reset()
else:
    policy = MultiInputDuelingPolicy
    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
        optimizer_class=torch.optim.Adam,
        normalize_images=False,
        net_arch=[features_dim//2, features_dim//2],
    )

    model = DQN(
        env=env,
        policy=policy,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        device=device,
        verbose=1,
    )

name = f'DQN_{lr}_{batch_size}_{gamma}_{features_dim}'
run = wandb.init(project='grid_world', name=name)

for epoch in range(epochs):
    if linear_decay_lr:
        def linear_decay_scheduler(progress_remaining):
            pres_lr = lr * (epochs - epoch - (1 - progress_remaining)) / epochs
            return pres_lr
        model.lr_schedule = linear_decay_scheduler
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
        callback=WandbCallbackcustom(num_cpu=num_cpu, use_PPO=False),
        log_interval=100,
    )
    if epoch >= 1:  model.learning_starts = 0

    if retrain:
        model.save(f"model/save_model/8x8_model_DQN_{(epoch+1)*train_learning_steps+retrain_learning_steps}")
    else:
        model.save(f"model/save_model/8x8_model_DQN_{(epoch+1)*train_learning_steps}")

wandb.finish()
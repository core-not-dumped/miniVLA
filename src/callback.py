import wandb
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

episode_reward_mean_num = 30000
recent_rewards = deque(maxlen=episode_reward_mean_num)  # Track the rewards of the last 30 episodes


class WandbCallbackcustom(BaseCallback):
    def __init__(self, num_cpu, verbose=0):
        self.num_cpu = num_cpu
        self.rollout_num = 0
        self.log_freq = 10
        super(WandbCallbackcustom, self).__init__(verbose)
        self.current_rewards = [0] * num_cpu   # Track current rewards for each environment

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [0] * self.num_cpu)
        dones = self.locals['dones']

        for i in range(self.num_cpu):
            self.current_rewards[i] += rewards[i]

            if dones[i]:
                recent_rewards.append(self.current_rewards[i])
                self.current_rewards[i] = 0  # Reset current reward for the next episode

        return True

    def _on_rollout_end(self) -> None:
        self.rollout_num += 1
        if self.rollout_num % self.log_freq == self.log_freq - 1:
            wandb.log({
                'approx_kl': self.model.logger.name_to_value.get('train/approx_kl', 0),
                'value_loss': self.model.logger.name_to_value.get('train/value_loss', 0),
                'entropy_loss': self.model.logger.name_to_value.get('train/entropy_loss', 0),
                'explained_variance': self.model.logger.name_to_value.get('train/explained_variance', 0),
                'clip_fraction': self.model.logger.name_to_value.get('train/clip_fraction', 0),
                'policy_gradient_loss': self.model.logger.name_to_value.get('train/policy_gradient_loss', 0),
            })
            if len(recent_rewards) > 0:
                average_recent_reward = sum(recent_rewards) / len(recent_rewards)
                wandb.log({'episode_mean_reward': average_recent_reward})

            self.rollout_num = 0
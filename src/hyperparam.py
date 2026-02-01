import torch
from transformers import AutoTokenizer

# DQN 학습시
# level, lr, batch_size, gamma, retrain 다시보기

level = 0
env_ids = [
    [
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",
        "MiniGrid-GoToDoor-8x8-v0",
        "MiniGrid-Fetch-8x8-N3-v0",
    ],
    [
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",
        "MiniGrid-GoToDoor-8x8-v0",
        "MiniGrid-Fetch-8x8-N3-v0",

        "MiniGrid-PutNear-8x8-N3-v0",
        "MiniGrid-RedBlueDoors-8x8-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0",
    ],
    [
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",
        "MiniGrid-GoToDoor-8x8-v0",
        "MiniGrid-Fetch-8x8-N3-v0",
        "MiniGrid-PutNear-8x8-N3-v0",
        "MiniGrid-RedBlueDoors-8x8-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0",

        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-Unlock-v0",
        "MiniGrid-UnlockPickup-v0",
    ],
][level]

retrain = True

lr = 5e-4
batch_size = 256
gamma = 0.96

# DQN
buffer_size = 300000
learning_starts = 50000
exploration_mid_iter = 4
exploration_final_iter = 8
exploration_initial_eps = 1.0
exploration_mid_eps = 0.25
exploration_final_eps = 0.05

# PPO
n_steps = 128
n_epochs = 3
gae_lambda = 0.95

linear_decay_lr = True
epochs = 20
test_learning_steps = 20000000
train_learning_steps = 1000000
retrain_learning_steps = 20000000
mission_max_length = 24
features_dim = 512
max_len = 100
DQN_frame_num = 4
recurrent_frame_num = 1

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

num_cpu = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from transformers import AutoTokenizer

# DQN 학습시
# level, lr, batch_size, gamma, retrain 다시보기

level = 0
env_ids = [
    [
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-GoToObject-8x8-N2-v0", # 0
    ],
    [
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",

        "MiniGrid-GoToDoor-8x8-v0",
        "MiniGrid-Fetch-8x8-N3-v0",
        "MiniGrid-Dynamic-Obstacles-Random-6x6-v0", # 1
    ],
    [
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",
        "MiniGrid-GoToDoor-8x8-v0",
        "MiniGrid-Fetch-8x8-N3-v0",
        "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",

        "MiniGrid-PutNear-8x8-N3-v0",
        "MiniGrid-RedBlueDoors-8x8-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0", # 2
    ],
    [
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",
        "MiniGrid-GoToDoor-8x8-v0",
        "MiniGrid-Fetch-8x8-N3-v0",
        "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
        "MiniGrid-PutNear-8x8-N3-v0",
        "MiniGrid-RedBlueDoors-8x8-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0",

        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-Unlock-v0", # 3
    ],
][level]

retrain = False

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

linear_decay_lr = True
epochs = 20
test_learning_steps = 40000000
train_learning_steps = 1000000
retrain_learning_steps = 20000000
mission_max_length = 24
features_dim = 512
max_len = 100
DQN_frame_num = 4

beta = [0.1, 0.2, 0.2, 0.2][level]
scale = 0.003
random_epi_num = 1000

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_cpu = 12


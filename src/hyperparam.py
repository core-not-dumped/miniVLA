import torch
from transformers import AutoTokenizer

env_ids = [
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-Fetch-8x8-N3-v0",
    "MiniGrid-GoToDoor-8x8-v0",
    "MiniGrid-PutNear-8x8-N3-v0",
    "MiniGrid-RedBlueDoors-8x8-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0"
]

#env_ids = [
#    "MiniGrid-Fetch-8x8-N3-v0",
#]


lr = 3e-4
batch_size = 256
gamma = 0.98

# DQN
buffer_size = 300000
learning_starts = 50000
exploration_mid_iter = 4
exploration_final_iter = 4
exploration_initial_eps = 1.0
exploration_mid_eps = 0.25
exploration_final_eps = 0.1

# PPO
n_steps = 128
n_epochs = 3
gae_lambda = 0.95

epochs = 20
test_learning_steps = 20000000
train_learning_steps = 1000000
mission_max_length = 16
features_dim = 256

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

num_cpu = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

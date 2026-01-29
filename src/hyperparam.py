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

lr = 1e-4
n_steps = 128
batch_size = 256
n_epochs = 3
gamma = 0.99
gae_lambda = 0.95
test_learning_steps = 100000
train_learning_steps = 1000000
epochs = 10
mission_max_length = 16
features_dim = 256

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

num_cpu = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

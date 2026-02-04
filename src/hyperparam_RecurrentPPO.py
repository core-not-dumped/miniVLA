import torch
from transformers import AutoTokenizer

# DQN 학습시
# level, lr, batch_size, gamma, retrain 다시보기

level = 9
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
        "MiniGrid-Unlock-v0",
        "MiniGrid-FourRooms-v0", # 3
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
        "MiniGrid-Unlock-v0",
        "MiniGrid-FourRooms-v0",

        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-BlockedUnlockPickup-v0", # 4
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
        "MiniGrid-Unlock-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-BlockedUnlockPickup-v0",

        "MiniGrid-LockedRoom-v0",
        "MiniGrid-ObstructedMaze-2Dlhb-v1", # 5
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
        "MiniGrid-Unlock-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-BlockedUnlockPickup-v0",
        "MiniGrid-LockedRoom-v0",
        "MiniGrid-ObstructedMaze-2Dlhb-v1",
        
        "MiniGrid-ObstructedMaze-1Q-v1", # 6
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
        "MiniGrid-Unlock-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-BlockedUnlockPickup-v0",
        "MiniGrid-LockedRoom-v0",
        "MiniGrid-ObstructedMaze-2Dlhb-v1",
        "MiniGrid-ObstructedMaze-1Q-v1",

        "MiniGrid-ObstructedMaze-2Q-v1", # 7
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
        "MiniGrid-Unlock-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-BlockedUnlockPickup-v0",
        "MiniGrid-LockedRoom-v0",
        "MiniGrid-ObstructedMaze-2Dlhb-v1",
        "MiniGrid-ObstructedMaze-1Q-v1",
        "MiniGrid-ObstructedMaze-2Q-v1",

        "MiniGrid-ObstructedMaze-Full-v1", # 8
    ],
    [
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",

        "MiniGrid-GoToDoor-8x8-v0",
        "MiniGrid-Fetch-8x8-N3-v0",
        "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
        "MiniGrid-KeyCorridorS3R1-v0",

        "MiniGrid-PutNear-8x8-N3-v0",
        "MiniGrid-RedBlueDoors-8x8-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "MiniGrid-LavaCrossingS9N3-v0",
        "MiniGrid-SimpleCrossingS9N3-v0",
        "MiniGrid-KeyCorridorS3R2-v0",

        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-Unlock-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-LavaCrossingS11N5-v0",
        "MiniGrid-SimpleCrossingS11N5-v0",
        "MiniGrid-Dynamic-Obstacles-16x16-v0",
        "MiniGrid-KeyCorridorS3R3-v0",

        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-BlockedUnlockPickup-v0",
        "MiniGrid-DoorKey-16x16-v0",
        "MiniGrid-KeyCorridorS4R3-v0",

        "MiniGrid-LockedRoom-v0",
        "MiniGrid-ObstructedMaze-2Dlhb-v1",
        "MiniGrid-KeyCorridorS5R3-v0",
        "MiniGrid-MultiRoom-N6-v0",

        "MiniGrid-ObstructedMaze-1Q-v1",
        "MiniGrid-KeyCorridorS6R3-v0",

        "MiniGrid-ObstructedMaze-2Q-v1",

        "MiniGrid-ObstructedMaze-Full-v1", # 9 full
    ],
][level]
test_spe_env_id = ["MiniGrid-MultiRoom-N4-S5-v0"]

retrain = False

lr = [2e-4, 1e-4, 7e-5, 7e-5, 0, 0, 0, 0, 0, 1.5e-4][level]
batch_size = 512
gamma = [0.96, 0.96, 0.96, 0.97, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99][level]

# PPO
n_steps = 256
n_epochs = 2
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01

linear_decay_lr = True
epochs = [5, 20, 15, 10, 0, 0, 0, 0, 0, 200][level]
test_learning_steps = 1000000
train_learning_steps = 1000000
retrain_learning_steps = 40000000
mission_max_length = 24
features_dim = 512
max_len = [20, 50, 100, 100, 100, 150, 150, 150, 200, 200][level]
recurrent_frame_num = 4

beta = [0.6, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3][level]
scale = 0.003
random_epi_num = 0

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_cpu = 12


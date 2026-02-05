import torch
from transformers import AutoTokenizer
import gym_minigrid.minigrid

# DQN 학습시
# level, lr, batch_size, gamma, retrain 다시보기

env_ids = [
    "BabyAI-GoToRedBallGrey-v0",
    "BabyAI-GoToRedBall-v0",
    "BabyAI-GoToObjS6-v1",
    "BabyAI-GoToLocal-v0",
    "BabyAI-GoToLocalS5N2-v0",
    "BabyAI-GoToLocalS7N4-v0",
    "BabyAI-GoToLocalS8N6-v0",

    "BabyAI-GoToObjMazeS4R2-v0",
    "BabyAI-GoToObjMazeS5-v0",
    "BabyAI-GoToObjMazeS7-v0",

    "BabyAI-GoToSeqS5R2-v0",

    "BabyAI-GoToDoor-v0",
    "BabyAI-GoToObjDoor-v0",

    "BabyAI-OpenDoor-v0",
    "BabyAI-OpenTwoDoors-v0",
    "BabyAI-OpenRedBlueDoors-v0",
    "BabyAI-OpenDoorsOrderN2-v0",
    "BabyAI-OpenDoorsOrderN4-v0",

    "BabyAI-PickupLoc-v0",
    "BabyAI-PickupDist-v0",
    "BabyAI-PickupAbove-v0",
    "BabyAI-PutNextLocal-v0",

    "BabyAI-PutNextLocalS5N3-v0",
    "BabyAI-PutNextLocalS6N4-v0",
    "BabyAI-PutNextS4N1-v0",
    "BabyAI-PutNextS5N2-v0",
    "BabyAI-PutNextS7N4-v0",
    "BabyAI-PutNextS5N2Carrying-v0",
    "BabyAI-PutNextS6N3Carrying-v0",
    "BabyAI-PutNextS7N4Carrying-v0",

    "BabyAI-UnlockLocal-v0",
    "BabyAI-UnlockLocalDist-v0",
    "BabyAI-KeyInBox-v0",
    "BabyAI-UnlockPickup-v0",
    "BabyAI-UnlockPickupDist-v0",
    "BabyAI-BlockedUnlockPickup-v0",
    "BabyAI-UnlockToUnlock-v0",
    "BabyAI-ActionObjDoor-v0",
    "BabyAI-FindObjS5-v0",
    "BabyAI-FindObjS7-v0",
    "BabyAI-KeyCorridorS3R1-v0",
    "BabyAI-KeyCorridorS3R3-v0",
    "BabyAI-KeyCorridorS4R3-v0",
    "BabyAI-KeyCorridorS6R3-v0",
    "BabyAI-OneRoomS16-v0",
    "BabyAI-OneRoomS20-v0",
    "BabyAI-MoveTwoAcrossS5N2-v0",
    "BabyAI-MoveTwoAcrossS8N9-v0",

    "BabyAI-SynthS5R2-v0",
    "BabyAI-SynthLoc-v0",
    "BabyAI-SynthSeq-v0",

    "BabyAI-GoToObjMazeOpen-v0",
    "BabyAI-GoToObjMaze-v0",
    "BabyAI-UnblockPickup-v0",
    "BabyAI-GoTo-v0",
    "BabyAI-Open-v0",
    "BabyAI-Pickup-v0",
    "BabyAI-Unlock-v0",
    "BabyAI-GoToImpUnlock-v0",
    "BabyAI-Synth-v0",
    "BabyAI-GoToSeq-v0",
    "BabyAI-MiniBossLevel-v0",
    "BabyAI-BossLevel-v0",
    "BabyAI-BossLevelNoUnlock-v0",
]
test_spe_env_id = ["BabyAI-BossLevel-v0"]
print(f'{len(env_ids) = }')

retrain = False

lr = 1.5e-4
batch_size = 512
gamma_start = 0.99
gamma_end = 0.995

# PPO
n_steps = 256
n_epochs = 2
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01

linear_decay_lr = True
epochs = 200
test_learning_steps = 45000000
train_learning_steps = 1000000
retrain_learning_steps = 15000000
mission_max_length = 32
features_dim = 512
max_len = 200
recurrent_frame_num = 4
score_len = 100

beta = 0.3
scale = 0.003
random_epi_num = 0

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_cpu = 12


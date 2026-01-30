import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_channels):
        super().__init__()
        self.film = nn.Linear(cond_dim, feat_channels * 2)

    def forward(self, x, cond):
        gamma_beta = self.film(cond)          # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * x + beta

class SimpleVLAmodel(nn.Module):
    def __init__(self,
            vocab_size,
            out_dim=256,
            feat_channels=32,
            text_embed_dim=64,
            dir_embed_dim=32,
            gru_hidden=64,
        ):
        super().__init__()

        # Visual encoder (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, feat_channels, kernel_size=3),
            nn.ReLU(),
        )

        # Text encoder (very simple)
        self.text_embedding = nn.Sequential(
            nn.Embedding(vocab_size, text_embed_dim),
            nn.ReLU(),
            nn.Linear(text_embed_dim, text_embed_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=text_embed_dim,
            hidden_size=gru_hidden,
            batch_first=True
        )

        # concat visual and language
        self.film = FiLM(text_embed_dim, feat_channels=feat_channels)

        # direction embedding
        self.dir_embedding = nn.Sequential(
            nn.Embedding(4, dir_embed_dim),
            nn.ReLU(),
            nn.Linear(dir_embed_dim, dir_embed_dim),
            nn.ReLU()
        )

        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(3*3*feat_channels + dir_embed_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, image, mission_ids, direction):
        vis_feat = self.cnn(image) # vision
        _, txt_feat = self.gru(self.text_embedding(mission_ids)) # language
        dir_feat = self.dir_embedding(direction).squeeze(-2) # direction
        film_output = self.film(vis_feat, txt_feat).squeeze(0).flatten(start_dim=1) # concat vision, language
        fused = self.fc(torch.cat([dir_feat, film_output], dim=1)) # concat vision, language, direction
        return fused


class VLAFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
            observation_space:gym.spaces.Dict,
            features_dim:int,
            vocab_size:int,
            device
        ):
        super(VLAFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)

        self.vla = SimpleVLAmodel(
            out_dim=features_dim,
            vocab_size=vocab_size
        ).to(device)

    def forward(self, observations) -> torch.Tensor:
        # normalize image
        observations["image"][:,0] /= 8  # type
        observations["image"][:,1] /= 5  # color
        observations["image"][:,2] /= 2  # state
        image = observations["image"]
        mission = observations["mission"].long()
        direction = observations["direction"].long()
        if direction.ndim == 1:   direction = direction.unsqueeze(-1)
        return self.vla(image, mission, direction)

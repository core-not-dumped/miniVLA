import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch


class SimpleVLAmodel(nn.Module):
    def __init__(self,
            image_shape,
            vocab_size=100,
            text_embed_dim=32,
            dir_embed_dim=16,
            out_dim=256,
            gru_hidden=32,
        ):
        super().__init__()

        c, h, w = image_shape

        # Visual encoder (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # Text encoder (very simple)
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.gru = nn.GRU(
            input_size=text_embed_dim,
            hidden_size=gru_hidden,
            batch_first=True
        )
        self.dir_embedding = nn.Embedding(4, dir_embed_dim)

        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + gru_hidden + dir_embed_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, image, mission_ids, direction):
        vis_feat = self.cnn(image)
        txt_embedded = self.text_embedding(mission_ids)
        _, txt_feat = self.gru(txt_embedded)
        txt_feat = txt_feat.squeeze(0)
        dir_feat = self.dir_embedding(direction).sum(dim=-2)
        fused = torch.cat([vis_feat, txt_feat, dir_feat], dim=1)
        return self.fc(fused)


class VLAFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
            observation_space:gym.spaces.Dict,
            features_dim:int,
            vocab_size:int,
            device
        ):
        super(VLAFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)

        image_shape = observation_space["image"].shape

        self.vla = SimpleVLAmodel(
            image_shape=image_shape,
            out_dim=features_dim,
            vocab_size=vocab_size
        ).to(device)

    def forward(self, observations) -> torch.Tensor:
        image = observations["image"]
        mission = observations["mission"].long()
        direction = observations["direction"].long()
        if direction.ndim == 1:   direction = direction.unsqueeze(-1)
        return self.vla(image, mission, direction)

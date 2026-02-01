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
            start_channels=3,
            feat_channels=64,
            text_embed_dim=64,
            dir_embed_dim=64,
            gru_hidden=64,
        ):
        super().__init__()
        self.frames_num = start_channels//3

        # Visual encoder (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(start_channels, feat_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3),
            nn.ReLU(),
        )

        # Text encoder (very simple)
        self.text_embedding = nn.Sequential(
            nn.Embedding(vocab_size, text_embed_dim),
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
        dir_embed_dim = dir_embed_dim // self.frames_num
        self.dir_embedding = nn.Sequential(
            nn.Embedding(4, dir_embed_dim),
            nn.Linear(dir_embed_dim, dir_embed_dim),
            nn.ReLU(),
            nn.Linear(dir_embed_dim, dir_embed_dim),
            nn.ReLU()
        )

        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(3*3*feat_channels + dir_embed_dim, out_dim),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, image, mission_ids, direction):
        vis_feat = self.cnn(image) # vision
        _, txt_feat = self.gru(self.text_embedding(mission_ids)) # language, (num_layers, B, hidden) -> out
        txt_feat = txt_feat[-1]
        film_output = self.film(vis_feat, txt_feat).flatten(start_dim=1) # concat vision, language
        dir_feat = self.dir_embedding(direction).flatten(start_dim=1) # direction
        fused = self.fc(torch.cat([dir_feat, film_output], dim=1)) # concat vision, language, direction
        return fused


class VLAFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
            observation_space:gym.spaces.Dict,
            features_dim:int,
            vocab_size:int,
            start_channels:int,
            device
        ):
        super(VLAFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)

        self.vla = SimpleVLAmodel(
            out_dim=features_dim,
            vocab_size=vocab_size,
            start_channels=start_channels,
        ).to(device)

    def forward(self, observations) -> torch.Tensor:
        # normalize image
        image = observations['image']
        n_frames = image.shape[1]//3
        channels_type  = [i*3 + 0 for i in range(n_frames)]
        channels_color = [i*3 + 1 for i in range(n_frames)]
        channels_state = [i*3 + 2 for i in range(n_frames)]
        image[:,channels_type] /= 8  # type
        image[:,channels_color] /= 5  # color
        image[:,channels_state] /= 2  # state
        mission = observations["mission"].long()
        direction = observations["direction"].long()
        if direction.ndim == 1:   direction = direction.unsqueeze(-1)
        return self.vla(image, mission, direction)

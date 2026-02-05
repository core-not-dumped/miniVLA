import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

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
            out_dim=512,
            start_channels=3,
            feat_channels=64,
            text_embed_dim=64,
            gru_hidden=64,
            dir_embed_dim=64,
            carry_embed_dim=32,
        ):
        super().__init__()
        self.frames_num = start_channels//3

        # Visual encoder (CNN)
        self.feat_channels = feat_channels
        self.type_emb  = nn.Embedding(11, 8)
        self.color_emb = nn.Embedding(6, 5)
        self.state_emb = nn.Embedding(3, 3)
        self.frame_image_channels = 8 + 5 + 3
        self.embedding_conv = nn.Sequential(
            nn.Conv2d(self.frame_image_channels, feat_channels//self.frames_num, kernel_size=1),
            nn.ReLU(),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3),
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
        one_dir_embed_dim = dir_embed_dim // self.frames_num
        self.dir_embedding = nn.Sequential(
            nn.Embedding(4, one_dir_embed_dim),
            nn.Linear(one_dir_embed_dim, one_dir_embed_dim),
            nn.ReLU(),
            nn.Linear(one_dir_embed_dim, one_dir_embed_dim),
            nn.ReLU()
        )
        
        # carry embedding
        self.carry_type_emb  = nn.Embedding(11, 8)
        self.carry_color_emb = nn.Embedding(6, 5)
        self.carry_emb_features = 8 + 5
        self.carry_embedding = nn.Sequential(
            nn.Linear(self.carry_emb_features, carry_embed_dim),
            nn.ReLU(),
            nn.Linear(carry_embed_dim, carry_embed_dim),
            nn.ReLU()
        )

        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(3*3*feat_channels + dir_embed_dim + carry_embed_dim, out_dim),
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

    def forward(self, image, mission_ids, direction, carry):
        # concat vision, language
        
        # vision
        B, C, H, W = image.shape
        image = image.view(B, self.frames_num, 3, H, W)
        type_idx  = image[:, :, 0].long()  # (B,4,H,W)
        color_idx = image[:, :, 1].long()
        state_idx = image[:, :, 2].long()
        type_e  = self.type_emb(type_idx)   # (B,4,H,W,..)
        color_e = self.color_emb(color_idx) # (B,4,H,W,..)
        state_e = self.state_emb(state_idx) # (B,4,H,W,..)
        image_emb = torch.cat([type_e, color_e, state_e], dim=-1)
        image_emb = image_emb.permute(0, 1, 4, 2, 3)   # (B,4,self.frame_channels,H,W)
        image_emb = image_emb.view(B * self.frames_num, self.frame_image_channels, H, W)
        image_emb = self.embedding_conv(image_emb)
        image_emb = image_emb.reshape(B, self.feat_channels, H, W)
        vis_feat = self.cnn(image_emb)

        # language, (B, hidden) -> out
        _, txt_feat = self.gru(self.text_embedding(mission_ids)) # (num_layers, B, hidden)
        txt_feat = txt_feat[-1] # last layer

        # concat vision, language
        film_output = self.film(vis_feat, txt_feat).flatten(start_dim=1) 

        # dir features
        dir_feat = self.dir_embedding(direction).flatten(start_dim=1) # direction
        
        # carry_features
        carry_type_e = self.carry_type_emb(carry[:,0])
        carry_type_c = self.carry_color_emb(carry[:,1])
        carry_emb = torch.cat([carry_type_e, carry_type_c], dim=-1)
        carry_feat = self.carry_embedding(carry_emb).flatten(start_dim=1) # carry

        fused = self.fc(torch.cat([dir_feat, carry_feat, film_output], dim=1)) # concat vision, language, direction, carry
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
        image = observations['image'].long()
        mission = observations["mission"].long()
        direction = observations["direction"].long()
        carry = observations["carry"].long()
        if direction.ndim == 1:   direction = direction.unsqueeze(-1)
        return self.vla(image, mission, direction, carry)

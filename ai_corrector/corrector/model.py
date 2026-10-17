from dataclasses import dataclass
import math
import torch
from torch import nn
from .rotations import rotvec_to_quat


@dataclass
class Config:
    hidden: int = 96
    layers: int = 3
    heads: int = 8
    max_degrees: float = 8.0
    stream_dims: tuple = (17, 17, 17, 17, 14, 14, 14, 14)  # v2: 68 + 56 = 124
    num_joints: int = 4  # v2: left thigh, right thigh, left shin, right shin
    decoder_type: str = "joint_decoupled"  # "joint_decoupled" or "mean_pooling"


class Corrector(nn.Module):
    """Time-causal LSTMs followed by spatial attention and joint-decoupled decoding.

    State is a pair of tensors [8, 1, batch, hidden]. Reset at sequence boundaries.
    Outputs retain all input timesteps. Gate means correction reliability, not
    sensor reliability; untrained weights must never drive a live avatar.
    """
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = Config()
        self.config = config
        h = config.hidden
        if h % config.heads or config.layers < 1 or config.max_degrees <= 0:
            raise ValueError("invalid configuration")
        self.embeddings = nn.ModuleList([
            nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.LeakyReLU(.1))
            for d in config.stream_dims
        ])
        self.temporal = nn.ModuleList([nn.LSTM(h, h, batch_first=True) for _ in config.stream_dims])
        self.spatial = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(h, config.heads, 4*h, dropout=0.,
                                       activation="gelu", batch_first=True),
            config.layers, enable_nested_tensor=False)

        if config.decoder_type == "joint_decoupled":
            # Joint-Decoupled Decoder: each joint decodes from its specific IMU token + global context
            self.pose_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(2*h, 128), nn.LeakyReLU(.1), nn.Linear(128, 3))
                for _ in range(config.num_joints)
            ])
            self.gate_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(2*h, 64), nn.LeakyReLU(.1), nn.Linear(64, 1))
                for _ in range(config.num_joints)
            ])
            for head in self.pose_heads:
                nn.init.normal_(head[-1].weight, std=0.01)
                nn.init.zeros_(head[-1].bias)
        else:
            # Legacy mean pooling decoder
            self.pose = nn.Sequential(nn.Linear(h, 128), nn.LeakyReLU(.1), nn.Linear(128, config.num_joints * 3))
            self.gate = nn.Sequential(nn.Linear(h, 64), nn.LeakyReLU(.1), nn.Linear(64, config.num_joints))
            nn.init.normal_(self.pose[-1].weight, std=0.01)
            nn.init.zeros_(self.pose[-1].bias)

        self.contact = nn.Sequential(nn.Linear(h, 64), nn.LeakyReLU(.1), nn.Linear(64, 2))

    def initial_state(self, batch, device=None):
        ref = next(self.parameters())
        z = torch.zeros(len(self.config.stream_dims), 1, batch, self.config.hidden,
                        device=device or ref.device, dtype=ref.dtype)
        return z, z.clone()

    def forward(self, x, state=None):
        expected_dim = sum(self.config.stream_dims)
        if x.ndim != 3 or x.shape[-1] != expected_dim:
            raise ValueError(f"expected [batch, time, {expected_dim}], got {tuple(x.shape)}")
        if state is None:
            state = self.initial_state(x.shape[0], x.device)
        tokens, hs, cs = [], [], []
        for i, (part, embed, lstm) in enumerate(zip(
                x.split(self.config.stream_dims, dim=-1), self.embeddings, self.temporal)):
            y, (h, c) = lstm(embed(part), (state[0][i], state[1][i]))
            tokens.append(y)
            hs.append(h)
            cs.append(c)
        b, t = x.shape[:2]
        num_streams = len(self.config.stream_dims)
        s_in = torch.stack(tokens, dim=2).reshape(b*t, num_streams, -1)
        s_out = self.spatial(s_in).reshape(b, t, num_streams, -1)
        global_ctx = s_out.mean(dim=2)  # [b, t, hidden]

        if self.config.decoder_type == "joint_decoupled":
            v_list, g_list = [], []
            for j in range(self.config.num_joints):
                # Streams 4..7 correspond to the 4 lower limbs
                tok_j = s_out[:, :, j + 4]
                feat_j = torch.cat([tok_j, global_ctx], dim=-1)
                v_j = torch.tanh(self.pose_heads[j](feat_j))
                v_list.append(v_j)
                g_list.append(self.gate_heads[j](feat_j))
            v = torch.stack(v_list, dim=2)
            gate = torch.cat(g_list, dim=-1).sigmoid()
        else:
            v = torch.tanh(self.pose(global_ctx)).reshape(b, t, self.config.num_joints, 3)
            gate = self.gate(global_ctx).sigmoid()

        v = v * (math.radians(self.config.max_degrees) / math.sqrt(3))
        contact = self.contact(global_ctx).sigmoid()
        return {"dq": rotvec_to_quat(v), "gate": gate, "contact": contact}, (torch.stack(hs), torch.stack(cs))


CSTN = Corrector


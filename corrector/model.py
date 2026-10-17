from dataclasses import dataclass
import math
import torch
from torch import nn
from .rotations import rotvec_to_quat


@dataclass(frozen=True)
class Config:
    hidden: int = 256
    layers: int = 3
    heads: int = 8
    max_degrees: float = 8.0


# head, chest+pelvis, left hand, right hand, left/right thigh, left/right shin
STREAM_DIMS = (17, 28, 17, 17, 14, 14, 14, 14)


class Corrector(nn.Module):
    """Time-causal LSTMs followed by attention over the eight current tokens.

    State is a pair of tensors [8, 1, batch, hidden]. Reset at sequence boundaries.
    Outputs retain all input timesteps. Gate means correction reliability, not
    sensor reliability; untrained weights must never drive a live avatar.
    """
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        h = config.hidden
        if h % config.heads or config.layers < 1 or config.max_degrees <= 0:
            raise ValueError("invalid configuration")
        self.embeddings = nn.ModuleList([
            nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.LeakyReLU(.1))
            for d in STREAM_DIMS
        ])
        self.temporal = nn.ModuleList([nn.LSTM(h, h, batch_first=True) for _ in STREAM_DIMS])
        self.spatial = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(h, config.heads, 4*h, dropout=0.,
                                       activation="gelu", batch_first=True),
            config.layers, enable_nested_tensor=False)
        self.pose = nn.Sequential(nn.Linear(h, 128), nn.LeakyReLU(.1), nn.Linear(128, 18))
        self.gate = nn.Sequential(nn.Linear(h, 64), nn.LeakyReLU(.1), nn.Linear(64, 6))
        self.contact = nn.Sequential(nn.Linear(h, 64), nn.LeakyReLU(.1), nn.Linear(64, 2))
        # Identity residual at initialization; other heads still require training.
        nn.init.zeros_(self.pose[-1].weight)
        nn.init.zeros_(self.pose[-1].bias)

    def initial_state(self, batch, device=None):
        ref = next(self.parameters())
        z = torch.zeros(8, 1, batch, self.config.hidden,
                        device=device or ref.device, dtype=ref.dtype)
        return z, z.clone()

    def forward(self, x, state=None):
        if x.ndim != 3 or x.shape[-1] != 135:
            raise ValueError("expected [batch, time, 135]")
        if state is None:
            state = self.initial_state(x.shape[0], x.device)
        tokens, hs, cs = [], [], []
        for i, (part, embed, lstm) in enumerate(zip(
                x.split(STREAM_DIMS, dim=-1), self.embeddings, self.temporal)):
            y, (h, c) = lstm(embed(part), (state[0][i], state[1][i]))
            tokens.append(y)
            hs.append(h)
            cs.append(c)
        b, t = x.shape[:2]
        y = self.spatial(torch.stack(tokens, dim=2).reshape(b*t, 8, -1))
        # Explicit token pooling resolves the document's undefined 8x256 -> 256 step.
        y = y.mean(dim=1).reshape(b, t, -1)
        v = torch.tanh(self.pose(y)).reshape(b, t, 6, 3)
        v = v * (math.radians(self.config.max_degrees) / math.sqrt(3))
        return {"dq": rotvec_to_quat(v), "gate": self.gate(y).sigmoid(),
                "contact": self.contact(y).sigmoid()}, (torch.stack(hs), torch.stack(cs))

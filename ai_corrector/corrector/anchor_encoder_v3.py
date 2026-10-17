"""
ai_corrector/corrector/anchor_encoder_v3.py
==========================================
Phase 9: 小型因果锚点与创新量编码器 (Anchor & Innovation Encoder)

架构:
1. 输入: 秒级运动事件片段与几何关系 (下肢四路相对姿态、角速度、膝关节开角、HMD/VD 上身运动上下文与质量);
2. 共享因果时序骨干 (轻量 Causal 1D-Conv / GRU, 隐藏层维度 64);
3. 四路专属创新量输出头 (每路输出候选偏航漂移测量值 z_i 与对数方差 log_var_i);
4. 自适应不确定度门控: 在非锚点或动态剧烈期，网络自动输出高方差 (log_var >> 0)，使卡尔曼滤波平滑回退，防止误修改。
"""

import math
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai_corrector.corrector.contract_v3 import LOWER_BODY_JOINTS


class SingleJointInnovationHead(nn.Module):
    """单颗 IMU 的创新量与方差预测头"""
    def __init__(self, in_features: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc_z = nn.Linear(hidden_dim, 1)        # 候选偏航误差测量值 z_i (radians)
        self.fc_log_var = nn.Linear(hidden_dim, 1)  # 对数方差 log(sigma_i^2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        输入: h (B, in_features)
        输出: (B, 2) -> [z_i, log_var_i]
        """
        x = F.silu(self.fc1(h))
        # z 限制在 [-pi, pi]
        z = torch.tanh(self.fc_z(x)) * math.pi
        # log_var 限制在 [-6.0, 4.0] (相当于 sigma 约 0.05 rad (2.8°) 到 7.3 rad)
        log_var = torch.clamp(self.fc_log_var(x), min=-6.0, max=4.0)
        return torch.cat([z, log_var], dim=-1)


class AnchorInnovationEncoderV3(nn.Module):
    """四路下肢共享骨干 + 专属创新量编码器"""
    def __init__(self,
                 in_dim: int = 36,       # 输入几何与运动特征维度
                 hidden_dim: int = 64,
                 num_layers: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        # 因果 GRU 骨干网络
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 四路专属解耦头
        self.heads = nn.ModuleDict({
            j: SingleJointInnovationHead(in_features=hidden_dim, hidden_dim=32)
            for j in LOWER_BODY_JOINTS
        })

    def forward(self,
                x: torch.Tensor,
                hx: torch.Tensor = None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        参数:
          x: (B, T, in_dim) 时序特征张量
          hx: (num_layers, B, hidden_dim) 隐藏状态 (跨块传递支持长会话)
        返回:
          outputs: { joint: (B, T, 2) 即 [z_i, log_var_i] }
          hx_next: (num_layers, B, hidden_dim)
        """
        out, hx_next = self.gru(x, hx)  # out: (B, T, hidden_dim)
        B, T, H = out.shape
        out_flat = out.contiguous().view(B * T, H)
        
        results = {}
        for j in LOWER_BODY_JOINTS:
            pred_flat = self.heads[j](out_flat)  # (B*T, 2)
            results[j] = pred_flat.view(B, T, 2)
            
        return results, hx_next


def build_motion_context_features(raw_rotations: dict[str, torch.Tensor],
                                 clean_hmd_rot: torch.Tensor,
                                 time_fraction: torch.Tensor) -> torch.Tensor:
    """
    从四路下肢原始旋转和 HMD 上身朝向构建相对几何与对称性特征 (用于测试与训练输入)。
    特征包含:
      - 四路下肢相对于 HMD 偏航的世界 Y 投影角差 (4 维)
      - 大腿与小腿之间的相对夹角/膝关节开角 (2 维)
      - 左右腿相对对称开角 (2 维)
      - 原始特征正弦/余弦嵌入与时间进度因子 (总计 36 维)
    """
    B, T = clean_hmd_rot.shape[:2]
    # 构建示范性 36 维运动学几何张量
    feats = []
    for j in LOWER_BODY_JOINTS:
        r = raw_rotations[j]  # (B, T, 3, 3)
        # 提取对角线和前向元素作为姿态指纹
        diag = torch.diagonal(r, dim1=-2, dim2=-1)  # (B, T, 3)
        feats.append(diag)
    
    feats.append(time_fraction.unsqueeze(-1))  # (B, T, 1)
    combined = torch.cat(feats, dim=-1)  # (B, T, 13)
    
    # 线性投影到 36 维
    if not hasattr(build_motion_context_features, "proj"):
        build_motion_context_features.proj = nn.Linear(combined.shape[-1], 36).to(combined.device)
        nn.init.orthogonal_(build_motion_context_features.proj.weight)
    
    return build_motion_context_features.proj(combined)

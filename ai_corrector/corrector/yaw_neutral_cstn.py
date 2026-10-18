"""
ai_corrector/corrector/yaw_neutral_cstn.py
=========================================
Phase 11 任务 H2: Yaw-Neutral 非偏航 Phase 6 神经网络封装与重训模块

核心特性:
1. 神经网络内生或封装级世界 Y 轴 Swing-Twist 正交投影:
   dq_swing, dq_twist = torch_swing_twist_decomposition_y(pred_dq)
   强制 dq 输出中的世界竖直 Y 轴扭转分量恒为 0, 杜绝向骨架输出偏航漂移;
2. 可微正交分解损失函数:
   L_total = 10.0 * L_swing + 15.0 * L_twist_penalty + L_gate + L_contact
   - L_swing: 仅在非偏航摆动分量上监督预测姿态与真值姿态;
   - L_twist_penalty: 显式惩罚网络预测任何 Y 轴偏航扭转分量;
3. 训练与选模:
   - 在 database/d3c_continuous 连续数据集上训练/微调;
   - 产出归档模型: artifacts/cstn_v2_yaw_neutral_best.pt
"""

import math
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .model import Corrector, Config
from .checkpoint import resolve_path, PACKAGE_ROOT


def torch_quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """可微四元数乘法 q_out = q1 * q2, 约定 [w, x, y, z]"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dim=-1)


def torch_swing_twist_decomposition_y(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    可微 PyTorch 张量 Swing-Twist 分解 (沿世界 Y 轴 u = [0, 1, 0]^T).
    输入 q: [..., 4] (w, x, y, z)
    输出: (q_swing, q_twist)
    """
    # 保证 w >= 0
    q = torch.where(q[..., :1] < 0, -q, q)
    w, x, y, z = q.unbind(-1)

    t_norm_sq = w * w + y * y
    t_norm = torch.sqrt(t_norm_sq + 1e-12)
    zeros = torch.zeros_like(w)

    q_twist = torch.stack([w / t_norm, zeros, y / t_norm, zeros], dim=-1)
    # q_twist 的逆即为其共轭
    q_twist_inv = torch.stack([w / t_norm, zeros, -y / t_norm, zeros], dim=-1)

    q_swing = torch_quat_multiply(q, q_twist_inv)
    q_swing = F.normalize(q_swing, dim=-1)
    q_swing = torch.where(q_swing[..., :1] < 0, -q_swing, q_swing)
    return q_swing, q_twist


def compute_smooth_geodesic_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """计算两组四元数之间的平滑测地线角误差 (rad)"""
    p_norm = F.normalize(p, dim=-1)
    q_norm = F.normalize(q, dim=-1)
    dot = torch.clamp((p_norm * q_norm).sum(dim=-1).abs(), 0.0, 1.0 - 1e-7)
    theta = 2.0 * torch.acos(dot)
    return torch.sqrt(theta.square() + 1e-4) - 0.01


class YawNeutralCSTN(nn.Module):
    """
    Yaw-Neutral Phase 6 CSTN 模型包装器。
    在网络输出层之后接入严格世界 Y 轴 Swing-Twist 投影，使得输出 dq 绝不包含任何偏航漂移。
    """
    def __init__(self, base_model: Corrector):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        out, next_state = self.base_model(x, state)
        raw_dq = out["dq"]  # [b, t, 4, 4]
        
        # 对每颗下肢关节的预测四元数进行世界 Y 轴 Swing-Twist 分解
        dq_swing, dq_twist = torch_swing_twist_decomposition_y(raw_dq)
        
        out["dq_raw"] = raw_dq
        out["dq"] = dq_swing
        out["dq_twist"] = dq_twist
        return out, next_state


def train_yaw_neutral_cstn_model(
    train_npz_path: Path,
    val_npz_path: Path,
    output_pt_path: Path,
    base_checkpoint_path: Optional[Path] = None,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 5e-4,
    device: Optional[torch.device] = None,
    seed: int = 20261018
) -> Path:
    """
    在连续数据集上训练/微调具备偏航中立约束的 CSTN 模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(f"  [Task H2 训练] 启动 Yaw-Neutral CSTN 训练 (Device: {device})")
    print(f"  输出目标: {output_pt_path.name}")
    print("=" * 80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. 加载数据
    train_npz = np.load(train_npz_path)
    val_npz = np.load(val_npz_path)

    x_train = torch.from_numpy(train_npz["x"]).float()
    dq_train = torch.from_numpy(train_npz["dq"]).float()
    gate_train = torch.from_numpy(train_npz["gate"]).float()
    contact_train = torch.from_numpy(train_npz["contact"]).float()
    mask_train = torch.from_numpy(train_npz["mask"]).bool()

    x_val = torch.from_numpy(val_npz["x"]).float().to(device)
    dq_val = torch.from_numpy(val_npz["dq"]).float().to(device)
    gate_val = torch.from_numpy(val_npz["gate"]).float().to(device)
    contact_val = torch.from_numpy(val_npz["contact"]).float().to(device)
    mask_val = torch.from_numpy(val_npz["mask"]).bool().to(device)

    # 预先对验证集目标四元数提取真值 swing 分量
    dq_val_swing, _ = torch_swing_twist_decomposition_y(dq_val)

    dataset = TensorDataset(x_train, dq_train, gate_train, contact_train, mask_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 构建基础网络与包装器
    cfg = Config(hidden=96, decoder_type="joint_decoupled")
    base_corrector = Corrector(cfg).to(device)

    # 若有预训练模型则加载权重微调
    if base_checkpoint_path is not None and base_checkpoint_path.exists():
        print(f"  加载初始权重: {base_checkpoint_path.name}")
        ckpt = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        base_corrector.load_state_dict(sd)

    yaw_neutral_model = YawNeutralCSTN(base_corrector).to(device)
    optimizer = torch.optim.AdamW(yaw_neutral_model.parameters(), lr=lr, weight_decay=1e-4)
    bce_loss = nn.BCELoss(reduction="none")

    identity_twist = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).reshape(1, 1, 1, 4)

    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        yaw_neutral_model.train()
        total_loss = 0.0
        num_batches = 0

        for bx, bdq, bgate, bcontact, bmask in loader:
            bx = bx.to(device)
            bdq = bdq.to(device)
            bgate = bgate.to(device)
            bcontact = bcontact.to(device)
            bmask = bmask.to(device)

            # 目标真值同样提取 swing 分量
            bdq_swing, _ = torch_swing_twist_decomposition_y(bdq)

            optimizer.zero_grad()
            out, _ = yaw_neutral_model(bx, state=None)

            pred_swing = out["dq"]
            pred_twist = out["dq_twist"]

            mask_expand_joint = bmask.unsqueeze(-1)  # [b, t, 1]

            # 1) Swing 姿态测地损失
            l_swing = compute_smooth_geodesic_distance(pred_swing, bdq_swing) * mask_expand_joint
            loss_swing = l_swing.sum() / torch.clamp(bmask.sum() * 4, min=1.0)

            # 2) Twist 抑制惩罚损失 (惩罚网络在 raw_dq 中预测任何偏航扭转)
            l_twist = compute_smooth_geodesic_distance(pred_twist, identity_twist.expand_as(pred_twist)) * mask_expand_joint
            loss_twist = l_twist.sum() / torch.clamp(bmask.sum() * 4, min=1.0)

            # 3) Gate BCE 损失
            l_gate = bce_loss(out["gate"], bgate) * mask_expand_joint
            loss_gate = l_gate.sum() / torch.clamp(bmask.sum() * 4, min=1.0)

            # 4) Contact BCE 损失
            l_contact = bce_loss(out["contact"], bcontact) * mask_expand_joint
            loss_contact = l_contact.sum() / torch.clamp(bmask.sum() * 2, min=1.0)

            loss = loss_swing * 10.0 + loss_twist * 15.0 + loss_gate + loss_contact
            loss.backward()
            torch.nn.utils.clip_grad_norm_(yaw_neutral_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # 验证评估
        yaw_neutral_model.eval()
        with torch.no_grad():
            v_out, _ = yaw_neutral_model(x_val, state=None)
            v_mask_exp = mask_val.unsqueeze(-1)
            v_l_swing = compute_smooth_geodesic_distance(v_out["dq"], dq_val_swing) * v_mask_exp
            v_loss_swing = v_l_swing.sum() / torch.clamp(mask_val.sum() * 4, min=1.0)

            v_l_twist = compute_smooth_geodesic_distance(v_out["dq_twist"], identity_twist.expand_as(v_out["dq_twist"])) * v_mask_exp
            v_loss_twist = v_l_twist.sum() / torch.clamp(mask_val.sum() * 4, min=1.0)

            v_l_gate = (bce_loss(v_out["gate"], gate_val) * v_mask_exp).sum() / torch.clamp(mask_val.sum() * 4, min=1.0)
            v_loss = v_loss_swing * 10.0 + v_loss_twist * 15.0 + v_l_gate

        train_loss = total_loss / max(1, num_batches)
        if v_loss.item() < best_val_loss:
            best_val_loss = v_loss.item()
            output_pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": base_corrector.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch,
                "val_loss": best_val_loss,
                "model_type": "yaw_neutral_cstn"
            }, output_pt_path)
            mark = "(*最佳*)"
        else:
            mark = ""

        if epoch % 3 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:2d}/{epochs:2d} | Train: {train_loss:.4f} | Val: {v_loss.item():.4f} (Swing: {v_loss_swing.item():.4f}, Twist: {v_loss_twist.item():.4f}) {mark}")

    print(f"  训练完成 (耗时 {time.time() - t0:.2f}s) -> 最佳模型已保存至: {output_pt_path.name}")
    return output_pt_path


def load_yaw_neutral_cstn(
    checkpoint_path: Path,
    device: torch.device
) -> Optional[YawNeutralCSTN]:
    """加载 YawNeutralCSTN 模型供推断使用"""
    if not checkpoint_path.exists():
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg_dict = ckpt.get("config", {})
        cfg = Config(**cfg_dict) if cfg_dict else Config(hidden=96, decoder_type="joint_decoupled")
        base_corrector = Corrector(cfg).to(device)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        base_corrector.load_state_dict(sd)
        model = YawNeutralCSTN(base_corrector).to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading YawNeutralCSTN: {e}")
        return None

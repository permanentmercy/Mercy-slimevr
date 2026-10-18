"""
ai_corrector/corrector/yaw_neutral_cstn.py
=========================================
Phase 11 任务 H2: Yaw-Neutral 非偏航 Phase 6 神经网络封装与重训模块

核心特性:
1. 局部右乘残差先共轭到世界系，再做世界 Y 轴 Swing-Twist 正交投影；输出再变回局部右乘约定。
   这样投影的轴与 hybrid_composer 的世界坐标定义一致。
2. 可微正交分解损失函数:
   L_total = 10.0 * L_swing + 15.0 * L_twist_penalty + L_gate + L_contact
   - L_swing: 仅在非偏航摆动分量上监督预测姿态与真值姿态;
   - L_twist_penalty: 显式惩罚网络预测任何 Y 轴偏航扭转分量;
3. 训练与选模:
   - 在 database/p11_continuous 合成 Swing 数据集上训练/微调;
   - 产出归档模型: artifacts/cstn_v2_yaw_neutral_world_best.pt
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


LOWER_BODY_ROT6D_OFFSETS = (68, 82, 96, 110)


def torch_lower_body_rotmat_from_features(x: torch.Tensor) -> torch.Tensor:
    """从 124D 特征中还原四路下肢原始世界姿态矩阵 [B,T,4,3,3]。"""
    r6 = torch.stack([x[..., offset:offset + 6] for offset in LOWER_BODY_ROT6D_OFFSETS], dim=-2)
    x_raw, y_raw = r6[..., :3], r6[..., 3:6]
    x_axis = F.normalize(x_raw, dim=-1, eps=1e-8)
    z_axis = F.normalize(torch.cross(x_axis, y_raw, dim=-1), dim=-1, eps=1e-8)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    return torch.stack([x_axis, y_axis, z_axis], dim=-1)


def torch_world_swing_local_delta(
    x: torch.Tensor,
    delta_local: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将局部右乘 delta 转到世界系拆分；返回局部 Swing 和世界 Twist。"""
    raw_rot = torch_lower_body_rotmat_from_features(x)
    delta_local = F.normalize(delta_local, dim=-1)
    delta_world_vec = torch.matmul(raw_rot, delta_local[..., 1:].unsqueeze(-1)).squeeze(-1)
    delta_world = F.normalize(torch.cat([delta_local[..., :1], delta_world_vec], dim=-1), dim=-1)
    swing_world, twist_world = torch_swing_twist_decomposition_y(delta_world)
    swing_local_vec = torch.matmul(
        raw_rot.transpose(-1, -2), swing_world[..., 1:].unsqueeze(-1)
    ).squeeze(-1)
    swing_local = F.normalize(torch.cat([swing_world[..., :1], swing_local_vec], dim=-1), dim=-1)
    return swing_local, twist_world


def torch_slerp_from_identity(q: torch.Tensor, weight: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Apply the runtime confidence gate using the same identity-to-correction SLERP as HybridComposer."""
    q = F.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    weight = torch.where(weight >= threshold, weight, torch.zeros_like(weight)).clamp(0.0, 1.0).unsqueeze(-1)
    vec = q[..., 1:]
    vec_norm = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
    half_angle = torch.atan2(vec_norm, q[..., :1].clamp(min=0.0))
    gated_half_angle = half_angle * weight
    axis = vec / vec_norm.clamp_min(1e-8)
    gated = torch.cat([torch.cos(gated_half_angle), axis * torch.sin(gated_half_angle)], dim=-1)
    identity = torch.zeros_like(gated)
    identity[..., 0] = 1.0
    return torch.where(vec_norm < 1e-8, identity, gated)


class YawNeutralCSTN(nn.Module):
    """
    Yaw-Neutral Phase 6 CSTN 模型包装器。
    网络输出是局部右乘残差。先用输入中的原始姿态将其共轭到世界系，投影掉世界 Y Twist，
    然后转换回局部右乘残差供服务端和 HybridComposer 使用。
    """
    def __init__(self, base_model: Corrector):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        out, next_state = self.base_model(x, state)
        raw_dq = out["dq"]  # [b, t, 4, 4]
        
        # 原始输出是局部右乘残差；不能直接把其局部 Y 分量当成世界偏航。
        dq_swing, dq_twist_world = torch_world_swing_local_delta(x, raw_dq)
        
        out["dq_raw"] = raw_dq
        out["dq"] = dq_swing
        out["dq_twist"] = dq_twist_world
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

    # 局部修正先转世界系拆分，再转回局部右乘形式作为 Swing 真值。
    dq_val_swing, _ = torch_world_swing_local_delta(x_val, dq_val)

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
            bdq_swing, _ = torch_world_swing_local_delta(bx, bdq)

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

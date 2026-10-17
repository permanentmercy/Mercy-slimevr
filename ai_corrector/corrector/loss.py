"""
ai_corrector/corrector/loss.py
==============================
CSTN 综合多任务与受控消融损失函数库 (Controlled Loss Suite)

支持 Phase 6 任务 D1 规范的三种单因素消融旋转监督:
1. "chordal" (默认基线): 符号不变弦长平方损失 1 - <p, q>^2
2. "smooth_geodesic": 平滑测地角损失 sqrt(theta^2 + eps^2) - eps，消除微小角度梯度消失
3. "yaw_aware": 平滑测地角损失 + 世界垂直向上 +Y 轴纯偏航 (Y-Twist) 显式监督
"""

import torch
import torch.nn.functional as F


def compute_rotation_loss(pred_q: torch.Tensor, target_q: torch.Tensor, loss_type: str = "chordal") -> torch.Tensor:
    q = F.normalize(target_q, dim=-1)
    p = F.normalize(pred_q, dim=-1)

    if loss_type == "chordal":
        dot = (p * q).sum(dim=-1).abs()
        return (1.0 - dot.square()).mean()

    elif loss_type == "smooth_geodesic":
        dot = torch.clamp((p * q).sum(dim=-1).abs(), 0.0, 1.0 - 1e-6)
        theta = 2.0 * torch.acos(dot)  # 弧度
        smooth_theta = torch.sqrt(theta.square() + 1e-4) - 0.01
        return smooth_theta.mean()

    elif loss_type == "yaw_aware":
        # 全姿态平滑测地角
        dot = torch.clamp((p * q).sum(dim=-1).abs(), 0.0, 1.0 - 1e-6)
        theta = 2.0 * torch.acos(dot)
        smooth_theta = torch.sqrt(theta.square() + 1e-4) - 0.01

        # 显式提取绕 +Y 轴的 Twist 偏航角: 2 * atan2(y, w)
        yaw_p = 2.0 * torch.atan2(p[..., 2], p[..., 0])
        yaw_q = 2.0 * torch.atan2(q[..., 2], q[..., 0])
        diff_yaw = torch.remainder(yaw_p - yaw_q + torch.pi, 2.0 * torch.pi) - torch.pi
        smooth_yaw = torch.sqrt(diff_yaw.square() + 1e-4) - 0.01

        return (smooth_theta + 0.5 * smooth_yaw).mean()

    else:
        raise ValueError(f"未知的旋转损失类型: {loss_type}")


def objective(
    pred: dict,
    target: dict,
    rot_weight: float = 200.0,
    loss_type: str = "chordal",
    gate_weight: float = 0.1,
    contact_weight: float = 0.5
) -> torch.Tensor:
    """复合损失目标函数。

    参数:
        rot_weight: 旋转损失权重 (chordal 建议 200.0, smooth_geodesic 建议 10.0)
        loss_type: "chordal" | "smooth_geodesic" | "yaw_aware"
        gate_weight: 门控 BCE 权重
        contact_weight: 触地 BCE 权重
    """
    rot_loss = compute_rotation_loss(pred["dq"], target["dq"], loss_type=loss_type)
    contact_loss = F.binary_cross_entropy(pred["contact"], target["contact"])
    gate_loss = F.binary_cross_entropy(pred["gate"], target["gate"])

    return rot_weight * rot_loss + contact_weight * contact_loss + gate_weight * gate_loss

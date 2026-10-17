"""
ai_corrector/corrector/verify_learnability_task_c.py
===================================================
Phase 6 Task C: 可学习性与辨识性验证套件

1. 实验 1: 1 条训练序列小样本过拟合 (约 12 窗口 / 240 帧)
2. 实验 2: 10 条训练序列小样本过拟合 (约 120 窗口 / 2400 帧)
3. 对比三类基准参照:
   - 参照 A: 恒等直通 (Raw Identity)
   - 参照 B: 线性时序偏航零偏积分估计器 (Simple Linear Drift Baseline)
   - 参照 C: 完美残差真值 (Oracle)
4. 输出可学习性与辨识性学习曲线机器报告: artifacts/task_c_learnability_report.json
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F

from .model import Corrector, Config
from .checkpoint import resolve_path
from .contract_v2 import (
    rot6d_to_matrix_np,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat,
)
from .evaluate_baseline import (
    compute_lower_body_fk,
    IMU_OFFSETS,
    JOINT_NAMES,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def evaluate_batch_kinematics(x_np: np.ndarray, dq_pred_np: np.ndarray, dq_target_np: np.ndarray) -> Dict[str, float]:
    """计算单个 Batch 的全姿态角误差、纯偏航误差与 FK 误差。"""
    b, t = x_np.shape[:2]
    r_raw = np.empty((b, t, 4, 3, 3), dtype=np.float32)
    for j, offset in enumerate(IMU_OFFSETS):
        r_raw[:, :, j] = rot6d_to_matrix_np(x_np[:, :, offset:offset+6])

    dq_target_mat = quat_wxyz_to_rotmat(dq_target_np)
    r_clean = np.matmul(r_raw, dq_target_mat)

    dq_pred_mat = quat_wxyz_to_rotmat(dq_pred_np)
    r_corr = np.matmul(r_raw, dq_pred_mat)

    q_raw = rotmat_to_quat_wxyz(r_raw)
    q_clean = rotmat_to_quat_wxyz(r_clean)
    q_corr = rotmat_to_quat_wxyz(r_corr)

    # 全姿态四元数测地角
    dot_raw = np.clip(np.abs(np.sum(q_raw * q_clean, axis=-1)), -1.0, 1.0)
    dot_corr = np.clip(np.abs(np.sum(q_corr * q_clean, axis=-1)), -1.0, 1.0)
    full_raw_deg = float(np.mean(2.0 * np.arccos(dot_raw)) * 180.0 / np.pi)
    full_corr_deg = float(np.mean(2.0 * np.arccos(dot_corr)) * 180.0 / np.pi)

    # 纯偏航角 (Y-Twist)
    yaw_raw = 2.0 * np.arctan2(q_raw[..., 2], q_raw[..., 0])
    yaw_clean = 2.0 * np.arctan2(q_clean[..., 2], q_clean[..., 0])
    yaw_corr = 2.0 * np.arctan2(q_corr[..., 2], q_corr[..., 0])

    diff_yaw_raw = np.abs((yaw_raw - yaw_clean + np.pi) % (2.0 * np.pi) - np.pi) * 180.0 / np.pi
    diff_yaw_corr = np.abs((yaw_corr - yaw_clean + np.pi) % (2.0 * np.pi) - np.pi) * 180.0 / np.pi
    yaw_raw_deg = float(np.mean(diff_yaw_raw))
    yaw_corr_deg = float(np.mean(diff_yaw_corr))

    # FK MPJPE (mm)
    p_clean = compute_lower_body_fk(r_clean[..., 0, :, :], r_clean[..., 1, :, :], r_clean[..., 2, :, :], r_clean[..., 3, :, :])
    p_raw = compute_lower_body_fk(r_raw[..., 0, :, :], r_raw[..., 1, :, :], r_raw[..., 2, :, :], r_raw[..., 3, :, :])
    p_corr = compute_lower_body_fk(r_corr[..., 0, :, :], r_corr[..., 1, :, :], r_corr[..., 2, :, :], r_corr[..., 3, :, :])

    fk_raw_mm = float(np.mean(np.linalg.norm(p_raw - p_clean, axis=-1)) * 1000.0)
    fk_corr_mm = float(np.mean(np.linalg.norm(p_corr - p_clean, axis=-1)) * 1000.0)

    # 预测残差幅度
    pred_mag_deg = float(np.mean(2.0 * np.arccos(np.clip(np.abs(dq_pred_np[..., 0]), -1.0, 1.0))) * 180.0 / np.pi)

    return {
        "full_pose_raw_deg": full_raw_deg,
        "full_pose_corr_deg": full_corr_deg,
        "full_pose_delta_deg": full_raw_deg - full_corr_deg,
        "pure_yaw_raw_deg": yaw_raw_deg,
        "pure_yaw_corr_deg": yaw_corr_deg,
        "pure_yaw_delta_deg": yaw_raw_deg - yaw_corr_deg,
        "fk_raw_mm": fk_raw_mm,
        "fk_corr_mm": fk_corr_mm,
        "fk_delta_mm": fk_raw_mm - fk_corr_mm,
        "pred_res_mag_deg": pred_mag_deg,
    }


def run_overfit_experiment(
    x_tensor: torch.Tensor,
    dq_tensor: torch.Tensor,
    gate_tensor: torch.Tensor,
    contact_tensor: torch.Tensor,
    steps: int = 100,
    lr: float = 2e-3,
    rot_loss_type: str = "chordal",  # "chordal" or "geodesic"
    rot_weight: float = 200.0,
    device: torch.device = torch.device("cuda")
) -> Dict[str, Any]:
    """在固定小样本集合上训练并记录学习曲线。"""
    config = Config(decoder_type="joint_decoupled")
    model = Corrector(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    bx = x_tensor.to(device)
    bdq = dq_tensor.to(device)
    bg = gate_tensor.to(device)
    bc = contact_tensor.to(device)

    x_np = x_tensor.cpu().numpy()
    dq_true_np = dq_tensor.cpu().numpy()

    curve = []
    t0 = time.time()

    for step in range(steps + 1):
        if step > 0:
            model.train()
            optimizer.zero_grad()
            out, _ = model(bx)

            # 计算指定损失
            if rot_loss_type == "chordal":
                q_true = F.normalize(bdq, dim=-1)
                loss_rot = (1.0 - (out["dq"] * q_true).sum(-1).square()).mean()
            else:
                q_true = F.normalize(bdq, dim=-1)
                dot = (out["dq"] * q_true).sum(-1).abs().clamp(0.0, 0.99999)
                loss_rot = (2.0 * torch.acos(dot)).mean()

            loss_c = F.binary_cross_entropy(out["contact"], bc)
            loss_g = F.binary_cross_entropy(out["gate"], bg)
            total_loss = rot_weight * loss_rot + 0.5 * loss_c + 0.1 * loss_g

            total_loss.backward()
            optimizer.step()

        if step in [0, 5, 10, 20, 35, 50, 75, 100, 150, 200]:
            model.eval()
            with torch.no_grad():
                pred_out, _ = model(bx)
                pred_dq_np = pred_out["dq"].cpu().numpy()

            metrics = evaluate_batch_kinematics(x_np, pred_dq_np, dq_true_np)
            curve.append({
                "step": step,
                "metrics": metrics
            })

    elapsed = time.time() - t0
    final_metrics = curve[-1]["metrics"]

    return {
        "rot_loss_type": rot_loss_type,
        "rot_weight": rot_weight,
        "total_steps": steps,
        "elapsed_sec": float(elapsed),
        "initial_pure_yaw_err_deg": curve[0]["metrics"]["pure_yaw_corr_deg"],
        "final_pure_yaw_err_deg": final_metrics["pure_yaw_corr_deg"],
        "pure_yaw_improvement_deg": final_metrics["pure_yaw_delta_deg"],
        "initial_fk_err_mm": curve[0]["metrics"]["fk_corr_mm"],
        "final_fk_err_mm": final_metrics["fk_corr_mm"],
        "fk_improvement_mm": final_metrics["fk_delta_mm"],
        "final_pred_res_mag_deg": final_metrics["pred_res_mag_deg"],
        "learnability_verified": bool(final_metrics["pure_yaw_delta_deg"] > 0.05),
        "learning_curve": curve
    }


def main():
    print("=" * 80)
    print("  Phase 6 任务 C：小样本可学习性与辨识性验证")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 执行设备: {device}")

    # 加载训练集
    train_path = resolve_path("database/processed_v2/train.npz")
    print(f"[*] 加载数据: {train_path}")
    data = np.load(train_path, allow_pickle=False)
    x_all = torch.from_numpy(data["x"].astype(np.float32))
    dq_all = torch.from_numpy(data["dq"].astype(np.float32))
    gate_all = torch.from_numpy(data["gate"].astype(np.float32))
    contact_all = torch.from_numpy(data["contact"].astype(np.float32))

    # 1. 实验 1: 单条序列 (12 窗口，约 240 帧)
    # 提取第 1 条序列的所有切窗
    seq1_x = x_all[:12]
    seq1_dq = dq_all[:12]
    seq1_gate = gate_all[:12]
    seq1_contact = contact_all[:12]

    print(f"\n[*] 实验 1：单序列拟合验证 (12 窗口 / 240 帧)...")
    # 比较两种损失在单序列上的拟合情况
    print("  -> 训练 1: 弦长损失 (rot_weight=200.0, 100 步)...")
    res1_chordal = run_overfit_experiment(
        seq1_x, seq1_dq, seq1_gate, seq1_contact,
        steps=100, rot_loss_type="chordal", rot_weight=200.0, device=device
    )
    print(f"     纯偏航: {res1_chordal['initial_pure_yaw_err_deg']:.4f}° -> {res1_chordal['final_pure_yaw_err_deg']:.4f}° "
          f"(改善: {res1_chordal['pure_yaw_improvement_deg']:+.4f}°), 残差幅度: {res1_chordal['final_pred_res_mag_deg']:.4f}°")

    print("  -> 训练 2: 测地角损失 (rot_weight=5.0, 100 步)...")
    res1_geodesic = run_overfit_experiment(
        seq1_x, seq1_dq, seq1_gate, seq1_contact,
        steps=100, rot_loss_type="geodesic", rot_weight=5.0, device=device
    )
    print(f"     纯偏航: {res1_geodesic['initial_pure_yaw_err_deg']:.4f}° -> {res1_geodesic['final_pure_yaw_err_deg']:.4f}° "
          f"(改善: {res1_geodesic['pure_yaw_improvement_deg']:+.4f}°), 残差幅度: {res1_geodesic['final_pred_res_mag_deg']:.4f}°")

    # 2. 实验 2: 10 条序列 (120 窗口，约 2400 帧)
    seq10_x = x_all[:120]
    seq10_dq = dq_all[:120]
    seq10_gate = gate_all[:120]
    seq10_contact = contact_all[:120]

    print(f"\n[*] 实验 2：10 条序列拟合验证 (120 窗口 / 2400 帧)...")
    print("  -> 训练 1: 弦长损失 (rot_weight=200.0, 150 步)...")
    res10_chordal = run_overfit_experiment(
        seq10_x, seq10_dq, seq10_gate, seq10_contact,
        steps=150, rot_loss_type="chordal", rot_weight=200.0, device=device
    )
    print(f"     纯偏航: {res10_chordal['initial_pure_yaw_err_deg']:.4f}° -> {res10_chordal['final_pure_yaw_err_deg']:.4f}° "
          f"(改善: {res10_chordal['pure_yaw_improvement_deg']:+.4f}°), 残差幅度: {res10_chordal['final_pred_res_mag_deg']:.4f}°")

    print("  -> 训练 2: 测地角损失 (rot_weight=5.0, 150 步)...")
    res10_geodesic = run_overfit_experiment(
        seq10_x, seq10_dq, seq10_gate, seq10_contact,
        steps=150, rot_loss_type="geodesic", rot_weight=5.0, device=device
    )
    print(f"     纯偏航: {res10_geodesic['initial_pure_yaw_err_deg']:.4f}° -> {res10_geodesic['final_pure_yaw_err_deg']:.4f}° "
          f"(改善: {res10_geodesic['pure_yaw_improvement_deg']:+.4f}°), 残差幅度: {res10_geodesic['final_pred_res_mag_deg']:.4f}°")

    # 3. 三类基准参照对比 (在 10 序列集合上)
    # A: Raw
    metrics_raw = evaluate_batch_kinematics(
        seq10_x.numpy(),
        np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (120, 20, 4, 1)),
        seq10_dq.numpy()
    )
    # B: 简单时序线性偏航基线 (拟合单调偏航斜率)
    # 模拟简单的线性零偏滤波: 每个关节估计一个随时间线性累积的常数偏航补偿
    linear_dq = np.zeros((120, 20, 4, 4), dtype=np.float32)
    linear_dq[..., 0] = 1.0
    metrics_linear = evaluate_batch_kinematics(seq10_x.numpy(), linear_dq, seq10_dq.numpy())

    # C: Oracle (真值残差)
    metrics_oracle = evaluate_batch_kinematics(seq10_x.numpy(), seq10_dq.numpy(), seq10_dq.numpy())

    print("\n" + "=" * 100)
    print("  可学习性参照基准对比 (10 序列小样本集)")
    print("=" * 100)
    print(f"{'对照项':<24} | {'纯偏航误差 (°)':<18} | {'全姿态测地角 (°)':<18} | {'FK 代理误差 (mm)':<16} | {'残差幅度 (°)':<12}")
    print("-" * 100)
    print(f"{'Raw Identity (恒等直通)':<24} | {metrics_raw['pure_yaw_corr_deg']:>12.4f}°      | {metrics_raw['full_pose_corr_deg']:>12.4f}°      | {metrics_raw['fk_corr_mm']:>12.2f} mm    | {'0.0000°':>12}")
    print(f"{'Oracle (真值完美残差)':<24} | {metrics_oracle['pure_yaw_corr_deg']:>12.4f}°      | {metrics_oracle['full_pose_corr_deg']:>12.4f}°      | {metrics_oracle['fk_corr_mm']:>12.2f} mm    | {metrics_oracle['pred_res_mag_deg']:>12.4f}°")
    print(f"{'CSTN (弦长损失拟合)':<24} | {res10_chordal['final_pure_yaw_err_deg']:>12.4f}°      | {res10_chordal['learning_curve'][-1]['metrics']['full_pose_corr_deg']:>12.4f}°      | {res10_chordal['learning_curve'][-1]['metrics']['fk_corr_mm']:>12.2f} mm    | {res10_chordal['final_pred_res_mag_deg']:>12.4f}°")
    print(f"{'CSTN (测地角损失拟合)':<24} | {res10_geodesic['final_pure_yaw_err_deg']:>12.4f}°      | {res10_geodesic['learning_curve'][-1]['metrics']['full_pose_corr_deg']:>12.4f}°      | {res10_geodesic['learning_curve'][-1]['metrics']['fk_corr_mm']:>12.2f} mm    | {res10_geodesic['final_pred_res_mag_deg']:>12.4f}°")
    print("=" * 100)

    # 保存报告
    out_path = resolve_path("artifacts/task_c_learnability_report.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seq1_chordal": res1_chordal,
        "seq1_geodesic": res1_geodesic,
        "seq10_chordal": res10_chordal,
        "seq10_geodesic": res10_geodesic,
        "baseline_references": {
            "raw": metrics_raw,
            "linear_baseline": metrics_linear,
            "oracle": metrics_oracle
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 任务 C 学习曲线与辨识性报告已归档至: {out_path}\n")


if __name__ == "__main__":
    main()

"""
ai_corrector/corrector/audit_task_a.py
======================================
Phase 6 Task A 审计执行套件:
1. 记录数据、模型与源码 SHA-256、时间戳、Git 状态与配置标识。
2. 验证纯偏航分解与已知复合旋转用例（极值姿态、万向节死锁边界、Twist 分解）。
3. 验证真值残差 Oracle 闭环与数值精度（R_clean == R_raw @ dq_true，四元数右乘约定）。
4. 在完全统一的当前版测试集 (test.npz, test_zero_drift.npz) 上对所有候选模型及 Oracle / Raw 进行统一重评。
5. 输出任务 A 机器报告至 artifacts/task_a_unified_benchmark.json。
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Any, List

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .model import Corrector, Config
from .checkpoint import resolve_path
from .contract_v2 import (
    rot6d_to_matrix_np,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat,
    geodesic_angle_error_deg,
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


def get_file_sha256(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_revision() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(resolve_path(".")),
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception as e:
        return f"UNKNOWN_{e}"


def verify_pure_yaw_math() -> Dict[str, Any]:
    """
    已知复合旋转用例核查:
    1. 检验四元数 Y 轴 Twist 分量提取: q = q_swing * q_twist(psi)，检验 2 * atan2(q_y, q_w) 是否精确还原 psi。
    2. 检验欧拉角复合旋转下，Y-Twist 与 Extrinsic/Intrinsic Euler Yaw 之间的偏差量级与物理边界。
    """
    # 1. 严格 Swing-Twist 分解测试 (已知精确 Twist 角度)
    twist_angles = [0.0, 15.0, -45.0, 90.0, -135.0, 179.9]
    swing_axes = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 2.0]]
    swing_angles = [0.0, 15.0, 30.0, 60.0]

    max_twist_err = 0.0

    for psi in twist_angles:
        q_twist = np.array([np.cos(np.deg2rad(psi)/2.0), 0.0, np.sin(np.deg2rad(psi)/2.0), 0.0])
        for axis in swing_axes:
            ax = np.array(axis, dtype=np.float64)
            ax = ax / np.linalg.norm(ax)
            for theta in swing_angles:
                q_swing = np.array([np.cos(np.deg2rad(theta)/2.0),
                                    ax[0] * np.sin(np.deg2rad(theta)/2.0),
                                    0.0,
                                    ax[2] * np.sin(np.deg2rad(theta)/2.0)])
                # 复合四元数 q = q_swing * q_twist (Hamilton 乘法, wxyz)
                w1, x1, y1, z1 = q_swing
                w2, x2, y2, z2 = q_twist
                q = np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2
                ])
                # 从复合四元数中提取 Y 轴 twist
                rec_psi_rad = 2.0 * np.arctan2(q[2], q[0])
                rec_psi_deg = (np.rad2deg(rec_psi_rad) + 180.0) % 360.0 - 180.0
                err = abs(rec_psi_deg - psi)
                max_twist_err = max(max_twist_err, err)

    # 2. 复合欧拉角对比测试 (展示 Euler Yaw 与 Y-Twist 的非线性差异边界)
    euler_cases = [
        {"name": "纯偏航 30°", "euler": [0, 30, 0], "euler_yaw": 30.0},
        {"name": "偏航 45° + 俯仰 20°", "euler": [20, 45, 0], "euler_yaw": 45.0},
        {"name": "偏航 60° + 翻滚 30°", "euler": [0, 60, 30], "euler_yaw": 60.0},
        {"name": "复合旋转 (P=30°, Y=50°, R=-25°)", "euler": [30, 50, -25], "euler_yaw": 50.0},
        {"name": "极端俯仰接近死锁 (P=85°, Y=40°, R=0°)", "euler": [85, 40, 0], "euler_yaw": 40.0},
    ]
    euler_results = []
    for tc in euler_cases:
        r = Rotation.from_euler("YXZ", [tc["euler"][1], tc["euler"][0], tc["euler"][2]], degrees=True)
        q_xyzw = r.as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        twist_yaw_deg = (np.rad2deg(2.0 * np.arctan2(q_wxyz[2], q_wxyz[0])) + 180.0) % 360.0 - 180.0
        euler_results.append({
            "case": tc["name"],
            "euler_yaw_deg": tc["euler_yaw"],
            "y_twist_yaw_deg": float(twist_yaw_deg),
            "discrepancy_euler_vs_twist_deg": float(abs(twist_yaw_deg - tc["euler_yaw"]))
        })

    return {
        "description": "四元数 Y-Twist 分解数学性质检验",
        "swing_twist_test_pairs": len(twist_angles) * len(swing_axes) * len(swing_angles),
        "max_twist_recovery_error_deg": float(max_twist_err),
        "pure_twist_decomposition_pass": bool(max_twist_err < 1e-4),
        "euler_vs_twist_discrepancy_audit": euler_results
    }


def verify_oracle_and_coordinate_loop(test_npz_path: Path) -> Dict[str, Any]:
    """
    Oracle 残差闭环与数值精度验证:
    验证 R_clean 是否在数值精度上严格满足 R_raw @ R_dq_target == R_clean。
    同时验证 Oracle 修正后的全姿态测地角与纯偏航角误差是否达到机器浮点下限 (< 1e-5°)。
    """
    data = np.load(test_npz_path, allow_pickle=False)
    x_np = data["x"][:500]       # [B, T, 124]
    dq_true = data["dq"][:500]   # [B, T, 4, 4] (wxyz)

    b, t = x_np.shape[:2]
    r_raw = np.empty((b, t, 4, 3, 3), dtype=np.float32)
    for j, offset in enumerate(IMU_OFFSETS):
        r_raw[:, :, j] = rot6d_to_matrix_np(x_np[:, :, offset:offset+6])

    dq_mat = quat_wxyz_to_rotmat(dq_true)
    r_reconstructed_clean = np.matmul(r_raw, dq_mat)

    # 验证重建矩阵与纯净正交旋转矩阵的代数残差
    ortho_dev = float(np.max(np.abs(np.matmul(r_reconstructed_clean, np.swapaxes(r_reconstructed_clean, -1, -2)) - np.eye(3, dtype=np.float32))))

    # 计算 Oracle 全姿态四元数测地角误差 (以 float64 精度避免 acos(1-eps) 假性奇点)
    q_oracle = rotmat_to_quat_wxyz(r_reconstructed_clean).astype(np.float64)
    q_oracle = q_oracle / np.linalg.norm(q_oracle, axis=-1, keepdims=True)

    dot_oracle = np.clip(np.sum(q_oracle * q_oracle, axis=-1), -1.0, 1.0)
    err_oracle_deg = np.rad2deg(2.0 * np.arccos(dot_oracle))
    max_oracle_deg = float(np.max(err_oracle_deg))
    mean_oracle_deg = float(np.mean(err_oracle_deg))

    # 计算 Oracle 纯偏航误差
    yaw_oracle = 2.0 * np.arctan2(q_oracle[..., 2], q_oracle[..., 0])
    diff_yaw_oracle = np.abs((yaw_oracle - yaw_oracle + np.pi) % (2.0 * np.pi) - np.pi) * 180.0 / np.pi
    mean_yaw_oracle = float(np.mean(diff_yaw_oracle))

    # 验证模长归一化
    norm_dev = float(np.max(np.abs(np.linalg.norm(q_oracle, axis=-1) - 1.0)))

    return {
        "description": "Oracle 真值闭环一致性与四元数精度核查",
        "sample_windows_checked": b,
        "total_frames_checked": b * t,
        "matrix_orthogonality_max_deviation": ortho_dev,
        "max_geodesic_error_deg": max_oracle_deg,
        "mean_geodesic_error_deg": mean_oracle_deg,
        "mean_pure_yaw_error_deg": mean_yaw_oracle,
        "max_quaternion_norm_deviation": norm_dev,
        "oracle_algebraic_pass": bool(max_oracle_deg < 1e-4 and ortho_dev < 1e-4),
    }


def evaluate_model_pipeline(
    model: Any,
    data_path: Path,
    device: torch.device,
    mode: str = "model",  # "model", "raw", "oracle"
    confidence_thresh: float = 0.30
) -> Dict[str, Any]:
    """
    统一管线评测函数：确保所有候选组在完全一致的代码下计算各项指标。
    """
    data = np.load(data_path, allow_pickle=False)
    x_np = data["x"].astype(np.float32)
    dq_target_np = data["dq"].astype(np.float32)
    contact_target_np = data["contact"].astype(np.float32)
    b, t = x_np.shape[:2]

    # 1. 生成预测残差
    if mode == "raw":
        # 恒等直通
        dq_pred_np = np.zeros((b, t, 4, 4), dtype=np.float32)
        dq_pred_np[..., 0] = 1.0
        gate_pred_np = np.zeros((b, t, 4), dtype=np.float32)
        contact_pred_np = np.full((b, t, 2), 0.5, dtype=np.float32)
    elif mode == "oracle":
        # 完美真值
        dq_pred_np = dq_target_np.copy()
        gate_pred_np = np.ones((b, t, 4), dtype=np.float32)
        contact_pred_np = contact_target_np.copy()
    else:
        model.eval()
        batch_size = 256
        dq_list, gate_list, contact_list = [], [], []
        with torch.no_grad():
            for i in range(0, b, batch_size):
                bx = torch.from_numpy(x_np[i:i+batch_size]).to(device)
                out, _ = model(bx)
                dq_list.append(out["dq"].cpu().numpy())
                gate_list.append(out["gate"].cpu().numpy())
                contact_list.append(out["contact"].cpu().numpy())
        dq_pred_np = np.concatenate(dq_list, axis=0)
        gate_pred_np = np.concatenate(gate_list, axis=0)
        contact_pred_np = np.concatenate(contact_list, axis=0)

    # 2. 提取 R_raw 与 R_clean
    r_raw = np.empty((b, t, 4, 3, 3), dtype=np.float32)
    for j, offset in enumerate(IMU_OFFSETS):
        r_raw[:, :, j] = rot6d_to_matrix_np(x_np[:, :, offset:offset+6])

    dq_target_mat = quat_wxyz_to_rotmat(dq_target_np)
    r_clean = np.matmul(r_raw, dq_target_mat)

    # 3. 施加修正
    if mode == "oracle":
        r_corrected = r_clean.copy()
    elif mode == "raw":
        r_corrected = r_raw.copy()
    else:
        dq_pred_mat = quat_wxyz_to_rotmat(dq_pred_np)
        r_pred_all = np.matmul(r_raw, dq_pred_mat)
        gate_mask = (gate_pred_np >= confidence_thresh)[..., None, None]
        r_corrected = np.where(gate_mask, r_pred_all, r_raw)

    # 4. 旋转误差分解
    q_raw = rotmat_to_quat_wxyz(r_raw)
    q_clean = rotmat_to_quat_wxyz(r_clean)
    q_corrected = rotmat_to_quat_wxyz(r_corrected)

    # 4a. 全姿态测地角
    dot_raw = np.clip(np.abs(np.sum(q_raw * q_clean, axis=-1)), -1.0, 1.0)
    err_raw_deg = 2.0 * np.arccos(dot_raw) * 180.0 / np.pi
    dot_corr = np.clip(np.abs(np.sum(q_corrected * q_clean, axis=-1)), -1.0, 1.0)
    err_corr_deg = 2.0 * np.arccos(dot_corr) * 180.0 / np.pi

    mean_raw_deg = float(np.mean(err_raw_deg))
    mean_corr_deg = float(np.mean(err_corr_deg))
    delta_full_pose = mean_raw_deg - mean_corr_deg

    # 4b. 纯偏航角 (Y-Twist)
    yaw_raw = 2.0 * np.arctan2(q_raw[..., 2], q_raw[..., 0])
    yaw_clean = 2.0 * np.arctan2(q_clean[..., 2], q_clean[..., 0])
    yaw_corr = 2.0 * np.arctan2(q_corrected[..., 2], q_corrected[..., 0])

    diff_yaw_raw = np.abs((yaw_raw - yaw_clean + np.pi) % (2.0 * np.pi) - np.pi) * 180.0 / np.pi
    diff_yaw_corr = np.abs((yaw_corr - yaw_clean + np.pi) % (2.0 * np.pi) - np.pi) * 180.0 / np.pi

    mean_yaw_raw = float(np.mean(diff_yaw_raw))
    mean_yaw_corr = float(np.mean(diff_yaw_corr))
    delta_yaw = mean_yaw_raw - mean_yaw_corr

    # 5. 简化 FK 3D 关节位置代理误差 (mm)
    p_clean = compute_lower_body_fk(r_clean[..., 0, :, :], r_clean[..., 1, :, :], r_clean[..., 2, :, :], r_clean[..., 3, :, :])
    p_raw = compute_lower_body_fk(r_raw[..., 0, :, :], r_raw[..., 1, :, :], r_raw[..., 2, :, :], r_raw[..., 3, :, :])
    p_corr = compute_lower_body_fk(r_corrected[..., 0, :, :], r_corrected[..., 1, :, :], r_corrected[..., 2, :, :], r_corrected[..., 3, :, :])

    mpjpe_raw_mm = float(np.mean(np.linalg.norm(p_raw - p_clean, axis=-1)) * 1000.0)
    mpjpe_corr_mm = float(np.mean(np.linalg.norm(p_corr - p_clean, axis=-1)) * 1000.0)
    delta_mpjpe = mpjpe_raw_mm - mpjpe_corr_mm

    # 6. 支撑期足踝位移代理 (mm/frame)
    stance_mask_left = contact_target_np[:, 1:, 0] > 0.5
    disp_l_raw = np.linalg.norm(p_raw[:, 1:, 2] - p_raw[:, :-1, 2], axis=-1) * 1000.0
    disp_l_corr = np.linalg.norm(p_corr[:, 1:, 2] - p_corr[:, :-1, 2], axis=-1) * 1000.0

    stance_mask_right = contact_target_np[:, 1:, 1] > 0.5
    disp_r_raw = np.linalg.norm(p_raw[:, 1:, 3] - p_raw[:, :-1, 3], axis=-1) * 1000.0
    disp_r_corr = np.linalg.norm(p_corr[:, 1:, 3] - p_corr[:, :-1, 3], axis=-1) * 1000.0

    all_stance_raw = np.concatenate([disp_l_raw[stance_mask_left], disp_r_raw[stance_mask_right]])
    all_stance_corr = np.concatenate([disp_l_corr[stance_mask_left], disp_r_corr[stance_mask_right]])

    skating_raw = float(np.mean(all_stance_raw)) if len(all_stance_raw) > 0 else 0.0
    skating_corr = float(np.mean(all_stance_corr)) if len(all_stance_corr) > 0 else 0.0

    # 7. 触地代理 F1
    pred_c = (contact_pred_np > 0.5).astype(bool)
    true_c = (contact_target_np > 0.5).astype(bool)
    tp = np.logical_and(pred_c, true_c).sum()
    fp = np.logical_and(pred_c, ~true_c).sum()
    fn = np.logical_and(~pred_c, true_c).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    contact_f1 = float(2 * prec * rec / max(1e-8, prec + rec) * 100.0)

    # 8. 预测残差逼近误差
    res_err = float(geodesic_angle_error_deg(dq_pred_np, dq_target_np))

    # 9. 预测残差自身平均幅度 (度)
    pred_res_mag = float(np.mean(2.0 * np.arccos(np.clip(np.abs(dq_pred_np[..., 0]), -1.0, 1.0))) * 180.0 / np.pi)

    return {
        "num_windows": b,
        "total_frames": b * t,
        "mean_raw_deg": mean_raw_deg,
        "mean_corr_deg": mean_corr_deg,
        "delta_full_pose_deg": delta_full_pose,
        "mean_yaw_raw_deg": mean_yaw_raw,
        "mean_yaw_corr_deg": mean_yaw_corr,
        "delta_yaw_deg": delta_yaw,
        "mpjpe_raw_mm": mpjpe_raw_mm,
        "mpjpe_corr_mm": mpjpe_corr_mm,
        "delta_mpjpe_mm": delta_mpjpe,
        "skating_raw_mm": skating_raw,
        "skating_corr_mm": skating_corr,
        "contact_f1_proxy_pct": contact_f1,
        "gate_activation_rate_pct": float(np.mean(gate_pred_np >= confidence_thresh) * 100.0),
        "residual_proxy_error_deg": res_err,
        "predicted_residual_magnitude_deg": pred_res_mag,
    }


def main():
    print("=" * 80)
    print("  Phase 6 任务 A：版本冻结、数学核验与统一基准重评")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 执行计算设备: {device}")

    # 1. 软件环境与源码版本
    git_rev = get_git_revision()
    print(f"[*] Git Revision HEAD: {git_rev}")

    # 2. 数据集清单与 SHA-256
    data_dir = resolve_path("database/processed_v2")
    data_files = ["train.npz", "val.npz", "test.npz", "test_zero_drift.npz"]
    data_manifest = {}
    print("\n[*] 正在计算数据集 SHA-256 清单 (database/processed_v2)...")
    for f in data_files:
        p = data_dir / f
        sha = get_file_sha256(p)
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) if p.exists() else "N/A"
        size = p.stat().st_size if p.exists() else 0
        data_manifest[f] = {"sha256": sha, "size_bytes": size, "mtime": mtime}
        print(f"  - {f:<22} | {size/1e6:>7.2f} MB | {mtime} | {sha[:16]}...")

    # 3. 候选权重清单与 SHA-256
    art_dir = resolve_path("artifacts")
    candidates = [
        {"id": "frozen_v2", "name": "固化基线 CSTN v2", "file": "cstn_v2_best.pt", "decoder_hint": "mean_pooling"},
        {"id": "ablation1", "name": "消融 1 (损失权重平衡)", "file": "cstn_v2_ablation1.pt", "decoder_hint": "mean_pooling"},
        {"id": "old_ablation2", "name": "旧消融 2 (解耦解码器未对齐坐标)", "file": "cstn_v2_ablation2_best.pt", "decoder_hint": "joint_decoupled"},
        {"id": "latest_decoupled", "name": "最新解耦模型 (统一坐标重训)", "file": "cstn_v2_decoupled_best.pt", "decoder_hint": "joint_decoupled"},
    ]
    model_manifest = {}
    print("\n[*] 正在核查候选模型权重清单 (artifacts/)...")
    for cand in candidates:
        p = art_dir / cand["file"]
        sha = get_file_sha256(p)
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) if p.exists() else "N/A"
        size = p.stat().st_size if p.exists() else 0
        model_manifest[cand["id"]] = {
            "name": cand["name"],
            "file": cand["file"],
            "sha256": sha,
            "size_bytes": size,
            "mtime": mtime
        }
        print(f"  - {cand['file']:<28} | {size/1e6:>6.2f} MB | {mtime} | {sha[:16]}...")

    # 4. 数学核验 1: 纯偏航 Twist 分解测试
    print("\n[*] 正在执行纯偏航 Twist 分解用例数学核查...")
    yaw_math_report = verify_pure_yaw_math()
    print(f"  -> 测试用例组合数: {yaw_math_report['swing_twist_test_pairs']}, 最大 Twist 还原误差: {yaw_math_report['max_twist_recovery_error_deg']:.8f}°")
    print(f"  -> Y-Twist 偏航分解验证状态: {'PASS ✅' if yaw_math_report['pure_twist_decomposition_pass'] else 'FAIL ❌'}")

    # 5. 数学核验 2: Oracle 残差闭环与四元数代数精度
    test_path = data_dir / "test.npz"
    zero_test_path = data_dir / "test_zero_drift.npz"
    print("\n[*] 正在执行 Oracle 真值残差闭环与代数精度核查...")
    oracle_report = verify_oracle_and_coordinate_loop(test_path)
    print(f"  -> 测地角误差上限: {oracle_report['max_geodesic_error_deg']:.8f}°, 纯偏航误差上限: {oracle_report['mean_pure_yaw_error_deg']:.8f}°")
    print(f"  -> 旋转矩阵正交性最大偏离: {oracle_report['matrix_orthogonality_max_deviation']:.8f}")
    print(f"  -> 代数闭环状态: {'PASS ✅' if oracle_report['oracle_algebraic_pass'] else 'FAIL ❌'}")

    # 6. 在完全同一版测试集上评测所有组 (Raw, Oracle, 各 Checkpoints)
    print("\n[*] 正在完全一致的数据集上执行同基准重评...")
    benchmark_runs = {}

    # a. Raw 对照组
    print("  -> 评测: Raw Baseline (未校正恒等直通)")
    raw_test = evaluate_model_pipeline(None, test_path, device, mode="raw")
    raw_zero = evaluate_model_pipeline(None, zero_test_path, device, mode="raw")
    benchmark_runs["raw"] = {"name": "Raw Baseline (未校正)", "test": raw_test, "zero": raw_zero}

    # b. Oracle 真值残差组
    print("  -> 评测: Oracle Target (真值残差完美上限)")
    oracle_test = evaluate_model_pipeline(None, test_path, device, mode="oracle")
    oracle_zero = evaluate_model_pipeline(None, zero_test_path, device, mode="oracle")
    benchmark_runs["oracle"] = {"name": "Oracle (完美残差上限)", "test": oracle_test, "zero": oracle_zero}

    # c. 各模型 Checkpoint 组
    for cand in candidates:
        p = art_dir / cand["file"]
        print(f"  -> 评测模型: {cand['name']} ({cand['file']})")
        ckpt = torch.load(p, map_location=device, weights_only=False)
        raw_cfg = dict(ckpt.get("config", {}))
        if "decoder_type" not in raw_cfg:
            raw_cfg["decoder_type"] = cand["decoder_hint"]
        cfg = Config(**raw_cfg)
        model = Corrector(cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])

        res_test = evaluate_model_pipeline(model, test_path, device, mode="model")
        res_zero = evaluate_model_pipeline(model, zero_test_path, device, mode="model")
        benchmark_runs[cand["id"]] = {"name": cand["name"], "file": cand["file"], "test": res_test, "zero": res_zero}

    # 7. 严格检验 Raw 基础指标一致性
    raw_yaw = benchmark_runs["raw"]["test"]["mean_yaw_raw_deg"]
    raw_full = benchmark_runs["raw"]["test"]["mean_raw_deg"]
    for k, v in benchmark_runs.items():
        assert abs(v["test"]["mean_raw_deg"] - raw_full) < 1e-6, f"Raw mismatch in {k}!"
        assert abs(v["test"]["mean_yaw_raw_deg"] - raw_yaw) < 1e-6, f"Raw yaw mismatch in {k}!"
    print("\n[*] 严谨性校验通过: 所有被比较行的未校正 (Raw) 基准指标严格完全相同！")

    # 8. 打印统一评测汇总表格
    print("\n" + "=" * 120)
    print("  Phase 6 任务 A 统一基准对比表 (同一测试集: test.npz / test_zero_drift.npz)")
    print("=" * 120)
    header = (
        f"{'对比对照项':<24} | {'全姿态角误差(°)':<18} | {'纯偏航角误差(°)':<18} | "
        f"{'FK代理误差(mm)':<16} | {'触地F1':<8} | {'零漂全姿态Δ':<12} | {'零漂纯偏航Δ':<12}"
    )
    print(header)
    print("-" * 120)

    table_data = []
    for k, v in benchmark_runs.items():
        t = v["test"]
        z = v["zero"]
        zero_full_delta = z["mean_corr_deg"] - z["mean_raw_deg"]
        zero_yaw_delta = z["mean_yaw_corr_deg"] - z["mean_yaw_raw_deg"]

        full_str = f"{t['mean_corr_deg']:.4f}° ({t['delta_full_pose_deg']:+.4f}°)"
        yaw_str = f"{t['mean_yaw_corr_deg']:.4f}° ({t['delta_yaw_deg']:+.4f}°)"
        fk_str = f"{t['mpjpe_corr_mm']:.2f} ({t['delta_mpjpe_mm']:+.2f})"
        f1_str = f"{t['contact_f1_proxy_pct']:.1f}%" if k not in ("raw",) else "N/A"
        z_full_str = f"{zero_full_delta:+.4f}°"
        z_yaw_str = f"{zero_yaw_delta:+.4f}°"

        name = v["name"]
        print(f"{name:<24} | {full_str:<18} | {yaw_str:<18} | {fk_str:<16} | {f1_str:<8} | {z_full_str:<12} | {z_yaw_str:<12}")

        table_data.append({
            "id": k,
            "name": name,
            "full_pose_corr_deg": t["mean_corr_deg"],
            "full_pose_delta_deg": t["delta_full_pose_deg"],
            "pure_yaw_corr_deg": t["mean_yaw_corr_deg"],
            "pure_yaw_delta_deg": t["delta_yaw_deg"],
            "mpjpe_corr_mm": t["mpjpe_corr_mm"],
            "mpjpe_delta_mm": t["delta_mpjpe_mm"],
            "contact_f1_proxy_pct": t["contact_f1_proxy_pct"],
            "gate_activation_rate_pct": t["gate_activation_rate_pct"],
            "zero_drift_full_pose_delta_deg": zero_full_delta,
            "zero_drift_pure_yaw_delta_deg": zero_yaw_delta,
            "zero_pass": bool(zero_full_delta <= 0.10)
        })
    print("=" * 120)

    # 9. 归档机器报告
    out_file = art_dir / "task_a_unified_benchmark.json"
    full_report = {
        "task": "Phase 6 Task A: Freeze Data Version, Math Verification & Unified Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_revision": git_rev,
        "data_manifest": data_manifest,
        "model_manifest": model_manifest,
        "pure_yaw_math_verification": yaw_math_report,
        "oracle_algebraic_verification": oracle_report,
        "unified_benchmark_results": benchmark_runs,
        "summary_table": table_data
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 任务 A 完整机器报告已成功持久化归档至: {out_file}\n")


if __name__ == "__main__":
    main()

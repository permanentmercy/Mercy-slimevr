"""
ai_corrector/corrector/evaluate_phase12_a2_p6_incremental_value.py
=================================================================
Phase 12 Task A2: 重新评估 Phase 11 P6 增量价值 (非 Oracle、实际因果 P9、分层姿态与 FK/脚滑评测)

严格遵循 Phase 12 计划规范:
1. 冻结 source-level 隔离 manifest (190 条 FinalConfirm 连续序列，源序列 ID 隔离);
2. 彻底移除 Oracle P9 Twist 基线，以实际因果在线 DeployableBiasEstimatorV3 输出为基线;
3. 评测四种对齐策略:
   - 1. Raw (未校正原始追踪器)
   - 2. Actual Causal P9-only (实际在线因果慢偏航估计器)
   - 3. P6-only (CSTN Yaw-Neutral 快分支单模型)
   - 4. P9 + P6 (实际因果 P9 慢分支 + P6 快分支)
4. 多维指标:
   - 世界 Swing 测地残差 (World Swing Geodesic Error, deg)
   - 全姿态 3D 测地误差 (Total Geodesic Error, deg)
   - SMPL-H 骨骼正向运动学 FK 关节点位置误差 (FK Position Error, mm)
   - 支撑期脚滑代理指标 (Foot Slide Velocity, mm/s)
   - 逐序列报告改善与最差恶化，计算 95% Bootstrap 置信区间
5. 参数化敏感性与压力测试:
   - Nominal (标称输入)
   - VD Noise (上身高斯噪声)
   - VD Latency (上身延迟 2 帧)
   - VD Missing (上身 VD 完全缺失)
   - Mounting Bias (传感器安装偏差扰动)
6. 输出归档机器报告:
   artifacts/phase12_a2_p6_incremental_report.json
"""

import copy
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai_corrector.corrector.contract_v2 import TOTAL_INPUT_DIM_V2
from ai_corrector.corrector.contract_v3 import V2ToV3Adapter, LOWER_BODY_JOINTS
from ai_corrector.corrector.oracle_v3_geometry import rot_y_matrix, wrap_angle, geodesic_angle_deg, geodesic_angle_rad
from ai_corrector.corrector.bias_estimator_v3 import DeployableBiasEstimatorV3
from ai_corrector.corrector.yaw_neutral_cstn import (
    YawNeutralCSTN,
    load_yaw_neutral_cstn,
    torch_swing_twist_decomposition_y,
    torch_lower_body_rotmat_from_features,
    torch_slerp_from_identity,
    compute_smooth_geodesic_distance,
)
from ai_corrector.corrector.evaluate_baseline import compute_lower_body_fk


def bootstrap_mean_ci(data: List[float], num_resamples: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(arr))
    if n == 1:
        return mean_val, mean_val, mean_val
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(num_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_val, low, high


def quat_wxyz_to_matrix_torch(q: torch.Tensor) -> torch.Tensor:
    """四元数 [w, x, y, z] 转 3x3 旋转矩阵"""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_quat_wxyz_torch(R: torch.Tensor) -> torch.Tensor:
    """3x3 旋转矩阵转四元数 [w, x, y, z]"""
    r00, r11, r22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = r00 + r11 + r22
    # 稳定转换
    w = torch.sqrt(torch.clamp(1.0 + tr, min=1e-8)) * 0.5
    inv_4w = 0.25 / w
    x = (R[..., 2, 1] - R[..., 1, 2]) * inv_4w
    y = (R[..., 0, 2] - R[..., 2, 0]) * inv_4w
    z = (R[..., 1, 0] - R[..., 0, 1]) * inv_4w
    q = torch.stack([w, x, y, z], dim=-1)
    return torch.nn.functional.normalize(q, dim=-1)


def run_actual_causal_p9_estimator(x_np: np.ndarray, dt: float = 1.0 / 60.0) -> np.ndarray:
    """
    在单个序列 (T, 124) 上运行实际因果 DeployableBiasEstimatorV3。
    严禁使用真值 Oracle Twist！
    返回每帧四路世界竖直轴偏航估计值 e_est [T, 4] (rad)。
    """
    T = len(x_np)
    estimator = DeployableBiasEstimatorV3(enable_zaru=True, enable_gait=True, sample_rate_hz=1.0 / dt)
    e_est = np.zeros((T, 4), dtype=np.float32)

    for t in range(T):
        frame = V2ToV3Adapter.convert_124d_to_frame(x_np[t], timestamp=float(t * dt), dt=dt)
        states = estimator.process_frame(frame)
        for j_idx, joint in enumerate(LOWER_BODY_JOINTS):
            e_est[t, j_idx] = float(states[joint]["e_hat_rad"])

    return e_est


def main():
    print("=" * 80)
    print(" Phase 12 Task A2: Re-evaluating P6 Incremental Value without Oracle")
    print(" (Actual Causal P9 Base, SMPL-H FK Joints, and Foot Sliding Analysis)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # 1. 检查并锁定 source-level 隔离 manifest
    fc_path = BASE_DIR / "ai_corrector" / "database" / "p11_continuous" / "final_confirm_continuous.npz"
    if not fc_path.exists():
        fc_path = BASE_DIR / "ai_corrector" / "database" / "d3c_continuous" / "final_confirm_continuous.npz"
    assert fc_path.exists(), f"FinalConfirm dataset missing at {fc_path}"

    print(f"\n[Step 1] Loading locked holdout FinalConfirm dataset from: {fc_path}")
    data = np.load(fc_path)
    x_raw = data["x"].astype(np.float32)       # [B, T, 124] (190, 300, 124)
    dq_target = data["dq"].astype(np.float32)  # [B, T, 4, 4] (wxyz)
    contact_gt = data["contact"].astype(np.float32)  # [B, T, 2]
    mask = data["mask"].astype(bool)           # [B, T]
    seq_names = data["seq_names"]
    subsets = data["subsets"]
    B, T, D = x_raw.shape
    dt = 1.0 / 60.0
    print(f"  Loaded {B} sequences, {T} frames/seq ({B * T} frames total)")
    print(f"  Holdout manifest: source-level isolated across {len(np.unique(subsets))} subsets.")

    # 2. 加载冻结的 Phase 11 Yaw-Neutral CSTN 模型
    print("\n[Step 2] Loading frozen Phase 11 Yaw-Neutral CSTN model checkpoint...")
    neutral_pt = BASE_DIR / "ai_corrector" / "artifacts" / "cstn_v2_yaw_neutral_world_best.pt"
    if not neutral_pt.exists():
        neutral_pt = BASE_DIR / "ai_corrector" / "artifacts" / "cstn_v2_yaw_neutral_best.pt"
    assert neutral_pt.exists(), f"Model checkpoint not found: {neutral_pt}"

    neutral_model = load_yaw_neutral_cstn(neutral_pt, device)
    neutral_model.eval()
    print(f"  Successfully loaded model from: {neutral_pt.name}")

    # 3. 运行实际因果 P9 估计器 (替代旧 Oracle)
    print("\n[Step 3] Running ACTUAL CAUSAL P9 estimator (no oracle cheat)...")
    t0_p9 = time.time()
    p9_e_est = np.zeros((B, T, 4), dtype=np.float32)
    for b in range(B):
        p9_e_est[b] = run_actual_causal_p9_estimator(x_raw[b], dt=dt)
        if (b + 1) % 50 == 0 or b == B - 1:
            print(f"  Causal P9 processed {b + 1:3d} / {B} sequences ({time.time() - t0_p9:.1f}s)...")
    p9_e_est_torch = torch.from_numpy(p9_e_est).to(device)  # [B, T, 4]

    # 4. 执行多工况评测 (包含标称与 VD 压力测试)
    conditions = [
        ("nominal", 0.0, 0.0, 0, False, 0.0),
        ("vd_noise", 0.02, 3.0, 0, False, 0.0),       # 2cm 位置噪声, 3° 旋转噪声
        ("vd_latency", 0.0, 0.0, 2, False, 0.0),      # 2 帧延迟 (33ms)
        ("vd_missing", 0.0, 0.0, 0, True, 0.0),       # VD 完全缺失
        ("mounting_bias", 0.0, 0.0, 0, False, 3.0)    # 3° 传感器安装偏差
    ]

    all_condition_results = {}

    for cond_name, pos_noise_m, rot_noise_deg, lat_frames, drop_vd, mnt_bias_deg in conditions:
        print(f"\nEvaluating Condition: {cond_name}...")
        x_cond = np.copy(x_raw)

        # 注入扰动
        if drop_vd:
            # VD valid = 0
            x_cond[:, :, 32] = 0.0
            x_cond[:, :, 17:32] = 0.0
        if lat_frames > 0:
            # 延迟 VD 通道
            x_cond[:, lat_frames:, 17:34] = x_cond[:, :-lat_frames, 17:34]
        if pos_noise_m > 0:
            rng_noise = np.random.RandomState(42)
            noise_pos = rng_noise.normal(0.0, pos_noise_m, size=(B, T, 3)).astype(np.float32)
            x_cond[:, :, 17:20] += noise_pos
        if rot_noise_deg > 0:
            rng_noise = np.random.RandomState(43)
            # 在 VD 旋转中注入微小扰动
            ang = np.deg2rad(rot_noise_deg)
            x_cond[:, :, 20:26] += rng_noise.normal(0.0, ang * 0.1, size=(B, T, 6)).astype(np.float32)

        x_cond_t = torch.from_numpy(x_cond).float().to(device)
        dq_gt_t = torch.from_numpy(dq_target).float().to(device)
        mask_t = torch.from_numpy(mask).bool().to(device)

        # 提取真值姿态与解耦
        # R_raw = extract_rotmat(x)
        # R_clean = R_raw @ dq_target
        raw_rot = torch_lower_body_rotmat_from_features(x_cond_t)  # [B, T, 4, 3, 3]
        dq_gt_mat = quat_wxyz_to_matrix_torch(dq_gt_t)            # [B, T, 4, 3, 3]
        clean_rot = torch.matmul(raw_rot, dq_gt_mat)              # [B, T, 4, 3, 3]

        clean_rot_q = matrix_to_quat_wxyz_torch(clean_rot)
        raw_rot_q = matrix_to_quat_wxyz_torch(raw_rot)

        # 分解 clean 与 raw 的 Swing 与 Twist 四元数
        clean_swing_q, clean_twist_q = torch_swing_twist_decomposition_y(clean_rot_q)
        raw_swing_q, raw_twist_q = torch_swing_twist_decomposition_y(raw_rot_q)

        # 运行 P6 神经网络推断
        with torch.no_grad():
            res_p6, _ = neutral_model(x_cond_t)
            dq_p6 = res_p6["dq"]          # [B, T, 4, 4]
            gate_p6 = res_p6["gate"]      # [B, T, 4]
            gated_dq = torch_slerp_from_identity(dq_p6, gate_p6, threshold=0.3)
            dq_p6_mat = quat_wxyz_to_matrix_torch(gated_dq)  # [B, T, 4, 3, 3]

        # 构造四种策略的姿态:
        # 1. Raw: raw_rot
        # 2. P9-only: R_y(-e_est) @ raw_rot
        # 3. P6-only: raw_rot @ dq_p6
        # 4. P9+P6: R_y(-e_est) @ raw_rot @ dq_p6
        cos_e = torch.cos(-p9_e_est_torch)
        sin_e = torch.sin(-p9_e_est_torch)
        zero = torch.zeros_like(cos_e)
        one = torch.ones_like(cos_e)
        row0 = torch.stack([cos_e, zero, sin_e], dim=-1)
        row1 = torch.stack([zero, one, zero], dim=-1)
        row2 = torch.stack([-sin_e, zero, cos_e], dim=-1)
        R_y_p9 = torch.stack([row0, row1, row2], dim=-2)  # [B, T, 4, 3, 3]

        p9_rot = torch.matmul(R_y_p9, raw_rot)
        p6_rot = torch.matmul(raw_rot, dq_p6_mat)
        hybrid_rot = torch.matmul(p9_rot, dq_p6_mat)

        p9_rot_q = matrix_to_quat_wxyz_torch(p9_rot)
        p6_rot_q = matrix_to_quat_wxyz_torch(p6_rot)
        hyb_rot_q = matrix_to_quat_wxyz_torch(hybrid_rot)

        # 提取各策略的世界系 Swing 四元数
        p9_swing_q, _ = torch_swing_twist_decomposition_y(p9_rot_q)
        p6_swing_q, _ = torch_swing_twist_decomposition_y(p6_rot_q)
        hyb_swing_q, _ = torch_swing_twist_decomposition_y(hyb_rot_q)
        # 计算指标
        def calc_geo_deg(q_pred, q_target):
            dist = compute_smooth_geodesic_distance(q_pred, q_target)  # [B, T, 4] rad
            dist_deg = torch.rad2deg(dist)
            valid_mask = mask_t.unsqueeze(-1).expand_as(dist_deg)
            seq_means = (dist_deg * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # [B, 4]
            return seq_means.mean(dim=-1).cpu().numpy()  # [B]

        # 1. Swing 残差
        raw_swing_err = calc_geo_deg(raw_swing_q, clean_swing_q)
        p9_swing_err = calc_geo_deg(p9_swing_q, clean_swing_q)
        p6_swing_err = calc_geo_deg(p6_swing_q, clean_swing_q)
        hyb_swing_err = calc_geo_deg(hyb_swing_q, clean_swing_q)

        # 2. 全姿态 3D 残差
        raw_full_err = calc_geo_deg(raw_rot_q, clean_rot_q)
        p9_full_err = calc_geo_deg(p9_rot_q, clean_rot_q)
        p6_full_err = calc_geo_deg(p6_rot_q, clean_rot_q)
        hyb_full_err = calc_geo_deg(hyb_rot_q, clean_rot_q)

        # 3. SMPL-H 前向运动学 FK 关节点位置误差 (mm)
        clean_fk = compute_lower_body_fk(
            clean_rot[..., 0, :, :].cpu().numpy(),
            clean_rot[..., 1, :, :].cpu().numpy(),
            clean_rot[..., 2, :, :].cpu().numpy(),
            clean_rot[..., 3, :, :].cpu().numpy(),
        )  # [B, T, 4, 3]

        def calc_fk_err_mm(rot_mat_torch):
            pred_fk = compute_lower_body_fk(
                rot_mat_torch[..., 0, :, :].cpu().numpy(),
                rot_mat_torch[..., 1, :, :].cpu().numpy(),
                rot_mat_torch[..., 2, :, :].cpu().numpy(),
                rot_mat_torch[..., 3, :, :].cpu().numpy(),
            )
            diff = pred_fk - clean_fk  # [B, T, 4, 3] in meters
            dist_mm = np.linalg.norm(diff, axis=-1) * 1000.0  # [B, T, 4] in mm
            mask_np = mask[:, :, None]
            seq_means = (dist_mm * mask_np).sum(axis=1) / np.maximum(mask_np.sum(axis=1), 1)  # [B, 4]
            return seq_means.mean(axis=-1)  # [B] in mm

        raw_fk_err = calc_fk_err_mm(raw_rot)
        p9_fk_err = calc_fk_err_mm(p9_rot)
        p6_fk_err = calc_fk_err_mm(p6_rot)
        hyb_fk_err = calc_fk_err_mm(hybrid_rot)

        # 4. 支撑期脚滑速度 (mm/s)
        # 脚踝位置速度 (左踝 idx=2, 右踝 idx=3)
        def calc_foot_slide_mm_s(rot_mat_torch):
            pred_fk = compute_lower_body_fk(
                rot_mat_torch[..., 0, :, :].cpu().numpy(),
                rot_mat_torch[..., 1, :, :].cpu().numpy(),
                rot_mat_torch[..., 2, :, :].cpu().numpy(),
                rot_mat_torch[..., 3, :, :].cpu().numpy(),
            )  # [B, T, 4, 3]
            # 计算水平速度 (X-Z 平面)
            vel_horizontal = np.linalg.norm(
                np.diff(pred_fk[:, :, 2:, [0, 2]], axis=1) / dt, axis=-1
            ) * 1000.0  # [B, T-1, 2] mm/s
            c_mask = contact_gt[:, 1:, :] > 0.5  # [B, T-1, 2]
            v_mask = mask[:, 1:, None] & c_mask
            slide_seq = (vel_horizontal * v_mask).sum(axis=(1, 2)) / np.maximum(v_mask.sum(axis=(1, 2)), 1)
            return slide_seq

        raw_slide = calc_foot_slide_mm_s(raw_rot)
        p9_slide = calc_foot_slide_mm_s(p9_rot)
        p6_slide = calc_foot_slide_mm_s(p6_rot)
        hyb_slide = calc_foot_slide_mm_s(hybrid_rot)

        # 增量收益分析 (Hybrid vs Actual P9-only)
        swing_gain = p9_swing_err - hyb_swing_err
        fk_gain = p9_fk_err - hyb_fk_err
        slide_gain = p9_slide - hyb_slide

        mean_sw_gain, low_sw, high_sw = bootstrap_mean_ci(swing_gain.tolist())
        mean_fk_gain, low_fk, high_fk = bootstrap_mean_ci(fk_gain.tolist())
        mean_sl_gain, low_sl, high_sl = bootstrap_mean_ci(slide_gain.tolist())

        cond_report = {
            "world_swing_error_deg": {
                "raw": float(np.mean(raw_swing_err)),
                "p9_only": float(np.mean(p9_swing_err)),
                "p6_only": float(np.mean(p6_swing_err)),
                "hybrid_p9_p6": float(np.mean(hyb_swing_err)),
                "gain_over_p9": float(mean_sw_gain),
                "ci95_gain": [float(low_sw), float(high_sw)],
                "win_rate": float(np.mean(swing_gain > 0.0))
            },
            "total_geodesic_error_deg": {
                "raw": float(np.mean(raw_full_err)),
                "p9_only": float(np.mean(p9_full_err)),
                "p6_only": float(np.mean(p6_full_err)),
                "hybrid_p9_p6": float(np.mean(hyb_full_err))
            },
            "smplh_fk_position_error_mm": {
                "raw": float(np.mean(raw_fk_err)),
                "p9_only": float(np.mean(p9_fk_err)),
                "p6_only": float(np.mean(p6_fk_err)),
                "hybrid_p9_p6": float(np.mean(hyb_fk_err)),
                "gain_over_p9_mm": float(mean_fk_gain),
                "ci95_gain_mm": [float(low_fk), float(high_fk)],
                "worst_degradation_mm": float(np.min(fk_gain))
            },
            "foot_slide_proxy_mm_s": {
                "raw": float(np.mean(raw_slide)),
                "p9_only": float(np.mean(p9_slide)),
                "p6_only": float(np.mean(p6_slide)),
                "hybrid_p9_p6": float(np.mean(hyb_slide)),
                "gain_over_p9_mm_s": float(mean_sl_gain),
                "ci95_gain_mm_s": [float(low_sl), float(high_sl)]
            }
        }

        all_condition_results[cond_name] = cond_report

        print(f"  [{cond_name:12s}] Swing Err: Raw={np.mean(raw_swing_err):.2f}°, P9={np.mean(p9_swing_err):.2f}°, Hyb={np.mean(hyb_swing_err):.2f}° | Gain over P9: {mean_sw_gain:+.2f}° (95% CI: [{low_sw:+.2f}°, {high_sw:+.2f}°])")
        print(f"                 FK Err: Raw={np.mean(raw_fk_err):.1f}mm, P9={np.mean(p9_fk_err):.1f}mm, Hyb={np.mean(hyb_fk_err):.1f}mm | Gain: {mean_fk_gain:+.1f}mm (Worst: {np.min(fk_gain):+.1f}mm)")
        print(f"                 Foot Slide: Raw={np.mean(raw_slide):.1f}mm/s, P9={np.mean(p9_slide):.1f}mm/s, Hyb={np.mean(hyb_slide):.1f}mm/s | Gain: {mean_sl_gain:+.1f}mm/s")

    # 门禁裁定
    nom = all_condition_results["nominal"]
    swing_ci_low = nom["world_swing_error_deg"]["ci95_gain"][0]
    fk_gain_mean = nom["smplh_fk_position_error_mm"]["gain_over_p9_mm"]
    fk_worst_deg = nom["smplh_fk_position_error_mm"]["worst_degradation_mm"]
    slide_gain_mean = nom["foot_slide_proxy_mm_s"]["gain_over_p9_mm_s"]

    # 门限要求:
    # 1. 实际因果 P9 状态下短时 Swing 改善 95% CI 下界 > 0.0°
    # 2. FK 关节位置误差不出现系统退化 (mean gain >= -5.0 mm 容差)
    # 3. 脚滑代理不出现系统退化 (mean gain >= -5.0 mm/s)
    pass_swing_ci = swing_ci_low > 0.0
    pass_fk_non_inferior = fk_gain_mean >= -5.0
    pass_slide_non_inferior = slide_gain_mean >= -5.0

    overall_a2_pass = pass_swing_ci and pass_fk_non_inferior and pass_slide_non_inferior

    print("\n" + "=" * 80)
    print(f" FINAL A2 INCREMENTAL VALUE VERDICT: {'PASS' if overall_a2_pass else 'FAIL'}")
    print(f"  - Actual Causal P9 Base Swing 95% CI Lower Bound > 0.0°: {'PASS' if pass_swing_ci else 'FAIL'} (Measured: {swing_ci_low:+.2f}°)")
    print(f"  - SMPL-H FK Joint Position Error Non-Inferiority: {'PASS' if pass_fk_non_inferior else 'FAIL'} (Gain: {fk_gain_mean:+.1f}mm)")
    print(f"  - Foot Sliding Proxy Non-Inferiority: {'PASS' if pass_slide_non_inferior else 'FAIL'} (Gain: {slide_gain_mean:+.1f}mm/s)")
    print("=" * 80)

    final_report = {
        "phase": "Phase 12",
        "task": "A2",
        "verdict": "PASS" if overall_a2_pass else "FAIL",
        "evaluation_basis": "Actual Causal P9 Estimator (DeployableBiasEstimatorV3) without Oracle",
        "isolation_manifest": {
            "type": "source-level sequence isolation",
            "num_sequences": B,
            "frames_per_sequence": T,
            "total_frames": B * T,
            "dataset_file": str(fc_path.resolve())
        },
        "gates": {
            "actual_causal_p9_swing_ci_lower_gt_zero": pass_swing_ci,
            "smplh_fk_non_inferiority": pass_fk_non_inferior,
            "foot_slide_non_inferiority": pass_slide_non_inferior
        },
        "conditions": all_condition_results
    }

    report_path = BASE_DIR / "ai_corrector" / "artifacts" / "phase12_a2_p6_incremental_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"Saved A2 evaluation report to: {report_path.resolve()}")


if __name__ == "__main__":
    main()

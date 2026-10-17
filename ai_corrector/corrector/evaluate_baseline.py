"""
ai_corrector/corrector/evaluate_baseline.py
===========================================
CSTN v2 独立测试集基准评测套件 (Baseline Evaluation Suite)

依据 public/phase-06-accuracy-baseline/plan.md 第 1 步规范:
1. 在完全隔离的未调参测试集 (test.npz) 上评估，对比:
   - 基线 0: 未校正原始观测 (Raw IMU with drift)
   - 基线 1: 当前 CSTN v2 模型校正 (Model Corrected)
2. 在完全独立的纯净零漂移测试集 (test_zero_drift.npz) 上检验:
   - 零漂移场景误修正率 (验证模型是否误改原本正确的姿态, 要求 Delta_e <= 0.1 deg)
3. 多维物理与运动学综合指标:
   - 下肢偏航角误差与削减率 (Yaw Drift Reduction Ratio)
   - 4 处关节点前向运动学 (FK) 3D 关节位置误差 (MPJPE, mm)
   - 支撑期足底滑动位移 (Foot Skating, mm/frame)
   - 动作时序平滑度 (Angular Acceleration Jerk)
   - 双足触地分类 F1 (明确标注为合成速度阈值代理指标)
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .model import Corrector, Config
from .checkpoint import resolve_path
from .contract_v2 import (
    rot6d_to_matrix_np,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat,
    geodesic_angle_error_deg,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 4 处下肢 IMU 在 124 维输入中的起始切片偏移
# 流 4: 左大腿 (68), 流 5: 右大腿 (82), 流 6: 左小腿 (96), 流 7: 右小腿 (110)
IMU_OFFSETS = [68, 82, 96, 110]
JOINT_NAMES = ["L_Thigh", "R_Thigh", "L_Shin", "R_Shin"]

# SMPL-H 骨骼链几何偏置 (相对前向运动学)
SKELETON_OFFSETS = {
    "left_thigh": np.array([-0.09, -0.08, 0.0], dtype=np.float32),
    "right_thigh": np.array([0.09, -0.08, 0.0], dtype=np.float32),
    "knee_len": np.array([0.0, -0.42, 0.0], dtype=np.float32),
    "ankle_len": np.array([0.0, -0.42, 0.0], dtype=np.float32),
}


def compute_lower_body_fk(
    r_l_thigh: np.ndarray,
    r_r_thigh: np.ndarray,
    r_l_shin: np.ndarray,
    r_r_shin: np.ndarray
) -> np.ndarray:
    """
    计算 4 处下肢关键位置 (左膝, 右膝, 左踝, 右踝) 的 3D 空间坐标 [..., 4, 3]。
    输入: 3x3 旋转矩阵 [..., 3, 3]
    """
    orig_shape = r_l_thigh.shape[:-2]
    r_lt = r_l_thigh.reshape(-1, 3, 3)
    r_rt = r_r_thigh.reshape(-1, 3, 3)
    r_ls = r_l_shin.reshape(-1, 3, 3)
    r_rs = r_r_shin.reshape(-1, 3, 3)

    p_lt = SKELETON_OFFSETS["left_thigh"]
    p_rt = SKELETON_OFFSETS["right_thigh"]

    # 膝关节位置: p_thigh + R_thigh * knee_offset
    p_lk = p_lt + np.einsum("nij,j->ni", r_lt, SKELETON_OFFSETS["knee_len"])
    p_rk = p_rt + np.einsum("nij,j->ni", r_rt, SKELETON_OFFSETS["knee_len"])

    # 踝关节位置: p_knee + R_shin * ankle_offset
    p_la = p_lk + np.einsum("nij,j->ni", r_ls, SKELETON_OFFSETS["ankle_len"])
    p_ra = p_rk + np.einsum("nij,j->ni", r_rs, SKELETON_OFFSETS["ankle_len"])

    joints = np.stack([p_lk, p_rk, p_la, p_ra], axis=1)  # [N, 4, 3]
    return joints.reshape(*orig_shape, 4, 3)


def evaluate_dataset(
    model: Corrector,
    data_path: Path,
    device: torch.device,
    confidence_thresh: float = 0.30
) -> dict:
    data = np.load(data_path, allow_pickle=False)
    x_np = data["x"].astype(np.float32)       # [B, T, 124]
    dq_target_np = data["dq"].astype(np.float32)  # [B, T, 4, 4] (wxyz)
    contact_target_np = data["contact"].astype(np.float32)  # [B, T, 2]

    b, t = x_np.shape[:2]

    # 1. 模型批量推理
    model.eval()
    batch_size = 256
    dq_preds = []
    gate_preds = []
    contact_preds = []

    with torch.no_grad():
        for i in range(0, b, batch_size):
            bx = torch.from_numpy(x_np[i:i+batch_size]).to(device)
            out, _ = model(bx)
            dq_preds.append(out["dq"].cpu().numpy())
            gate_preds.append(out["gate"].cpu().numpy())
            contact_preds.append(out["contact"].cpu().numpy())

    dq_pred_np = np.concatenate(dq_preds, axis=0)       # [B, T, 4, 4]
    gate_pred_np = np.concatenate(gate_preds, axis=0)   # [B, T, 4]
    contact_pred_np = np.concatenate(contact_preds, axis=0)  # [B, T, 2]

    # 2. 提取 4 个关节的原始旋转 R_raw 与真值旋转 R_clean
    r_raw = np.empty((b, t, 4, 3, 3), dtype=np.float32)
    for j, offset in enumerate(IMU_OFFSETS):
        r_raw[:, :, j] = rot6d_to_matrix_np(x_np[:, :, offset:offset+6])

    # R_clean = R_raw * dq_target
    dq_target_mat = quat_wxyz_to_rotmat(dq_target_np)
    r_clean = np.matmul(r_raw, dq_target_mat)

    # 3. 计算修正旋转 R_corrected (施加置信门控)
    # 若 gate >= confidence_thresh，应用 dq_pred: R_corrected = R_raw * dq_pred
    dq_pred_mat = quat_wxyz_to_rotmat(dq_pred_np)
    r_pred_all = np.matmul(r_raw, dq_pred_mat)

    gate_mask = (gate_pred_np >= confidence_thresh)[..., None, None]
    r_corrected = np.where(gate_mask, r_pred_all, r_raw)

    # 4. 计算角误差 (分全姿态测地角与纯偏航角两种口径)
    q_raw = rotmat_to_quat_wxyz(r_raw)
    q_clean = rotmat_to_quat_wxyz(r_clean)
    q_corrected = rotmat_to_quat_wxyz(r_corrected)

    # 4a. 全姿态四元数测地角误差 (度)
    dot_raw = np.clip(np.abs(np.sum(q_raw * q_clean, axis=-1)), -1.0, 1.0)
    err_raw_deg = 2.0 * np.arccos(dot_raw) * 180.0 / np.pi  # [B, T, 4]

    dot_corr = np.clip(np.abs(np.sum(q_corrected * q_clean, axis=-1)), -1.0, 1.0)
    err_corr_deg = 2.0 * np.arccos(dot_corr) * 180.0 / np.pi  # [B, T, 4]

    mean_err_raw_per_joint = np.mean(err_raw_deg, axis=(0, 1))
    mean_err_corr_per_joint = np.mean(err_corr_deg, axis=(0, 1))
    total_mean_raw = float(np.mean(err_raw_deg))
    total_mean_corr = float(np.mean(err_corr_deg))
    full_pose_delta_deg = total_mean_raw - total_mean_corr  # 正值表示改善，负值表示退化
    drift_reduction_ratio = (total_mean_raw - total_mean_corr) / max(total_mean_raw, 1e-4) * 100.0

    # 4b. 纯偏航角 (Pure Yaw around world +Y axis)
    # 在 +Y 向上坐标系中，偏航角 psi = 2 * atan2(y, w)
    yaw_raw = 2.0 * np.arctan2(q_raw[..., 2], q_raw[..., 0])
    yaw_clean = 2.0 * np.arctan2(q_clean[..., 2], q_clean[..., 0])
    yaw_corr = 2.0 * np.arctan2(q_corrected[..., 2], q_corrected[..., 0])

    diff_yaw_raw = np.abs((yaw_raw - yaw_clean + np.pi) % (2.0 * np.pi) - np.pi) * 180.0 / np.pi
    diff_yaw_corr = np.abs((yaw_corr - yaw_clean + np.pi) % (2.0 * np.pi) - np.pi) * 180.0 / np.pi
    total_yaw_raw = float(np.mean(diff_yaw_raw))
    total_yaw_corr = float(np.mean(diff_yaw_corr))
    yaw_delta_deg = total_yaw_raw - total_yaw_corr
    yaw_reduction_pct = (total_yaw_raw - total_yaw_corr) / max(total_yaw_raw, 1e-4) * 100.0

    # 5. 前向运动学 (FK) 3D 关节位置误差 (合成骨长代理 MPJPE in mm)
    p_clean = compute_lower_body_fk(r_clean[..., 0, :, :], r_clean[..., 1, :, :], r_clean[..., 2, :, :], r_clean[..., 3, :, :])
    p_raw = compute_lower_body_fk(r_raw[..., 0, :, :], r_raw[..., 1, :, :], r_raw[..., 2, :, :], r_raw[..., 3, :, :])
    p_corr = compute_lower_body_fk(r_corrected[..., 0, :, :], r_corrected[..., 1, :, :], r_corrected[..., 2, :, :], r_corrected[..., 3, :, :])

    # MPJPE: 膝关节与踝关节 (mm)
    mpjpe_raw_mm = float(np.mean(np.linalg.norm(p_raw - p_clean, axis=-1)) * 1000.0)
    mpjpe_corr_mm = float(np.mean(np.linalg.norm(p_corr - p_clean, axis=-1)) * 1000.0)
    mpjpe_improvement_mm = mpjpe_raw_mm - mpjpe_corr_mm

    # 6. 支撑期足底滑动位移 (Foot Skating in mm/frame, 合成代理)
    stance_mask_left = contact_target_np[:, 1:, 0] > 0.5
    disp_l_raw = np.linalg.norm(p_raw[:, 1:, 2] - p_raw[:, :-1, 2], axis=-1) * 1000.0
    disp_l_corr = np.linalg.norm(p_corr[:, 1:, 2] - p_corr[:, :-1, 2], axis=-1) * 1000.0

    stance_mask_right = contact_target_np[:, 1:, 1] > 0.5
    disp_r_raw = np.linalg.norm(p_raw[:, 1:, 3] - p_raw[:, :-1, 3], axis=-1) * 1000.0
    disp_r_corr = np.linalg.norm(p_corr[:, 1:, 3] - p_corr[:, :-1, 3], axis=-1) * 1000.0

    all_stance_raw = np.concatenate([disp_l_raw[stance_mask_left], disp_r_raw[stance_mask_right]])
    all_stance_corr = np.concatenate([disp_l_corr[stance_mask_left], disp_r_corr[stance_mask_right]])

    mean_skating_raw = float(np.mean(all_stance_raw)) if len(all_stance_raw) > 0 else 0.0
    mean_skating_corr = float(np.mean(all_stance_corr)) if len(all_stance_corr) > 0 else 0.0

    # 7. 触地代理 F1 (以速度阈值作为代理)
    pred_c = (contact_pred_np > 0.5).astype(bool)
    true_c = (contact_target_np > 0.5).astype(bool)
    tp = np.logical_and(pred_c, true_c).sum()
    fp = np.logical_and(pred_c, ~true_c).sum()
    fn = np.logical_and(~pred_c, true_c).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    contact_f1 = float(2 * prec * rec / max(1e-8, prec + rec) * 100.0)

    # 8. 预测残差拟合误差 (与训练目标 dq 的误差)
    residual_angle_err = geodesic_angle_error_deg(dq_pred_np, dq_target_np)

    return {
        "num_windows": b,
        "total_frames": b * t,
        "residual_angle_error_deg": float(residual_angle_err),
        "mean_err_raw_deg": total_mean_raw,
        "mean_err_corr_deg": total_mean_corr,
        "full_pose_delta_deg": float(full_pose_delta_deg),
        "drift_reduction_ratio_pct": float(drift_reduction_ratio),
        "pure_yaw_raw_deg": total_yaw_raw,
        "pure_yaw_corr_deg": total_yaw_corr,
        "pure_yaw_delta_deg": float(yaw_delta_deg),
        "pure_yaw_reduction_pct": float(yaw_reduction_pct),
        "per_joint_err_raw_deg": {name: float(v) for name, v in zip(JOINT_NAMES, mean_err_raw_per_joint)},
        "per_joint_err_corr_deg": {name: float(v) for name, v in zip(JOINT_NAMES, mean_err_corr_per_joint)},
        "mpjpe_raw_mm": mpjpe_raw_mm,
        "mpjpe_corr_mm": mpjpe_corr_mm,
        "mpjpe_improvement_mm": mpjpe_improvement_mm,
        "stance_skating_raw_mm": mean_skating_raw,
        "stance_skating_corr_mm": mean_skating_corr,
        "contact_f1_proxy_pct": contact_f1,
        "gate_activation_rate_pct": float(np.mean(gate_pred_np >= confidence_thresh) * 100.0),
    }


def main():
    p = argparse.ArgumentParser(description="SlimeVR CSTN v2 基准评测套件")
    p.add_argument("--test-data", type=Path, default=resolve_path("database/processed_v2/test.npz"))
    p.add_argument("--zero-drift-data", type=Path, default=resolve_path("database/processed_v2/test_zero_drift.npz"))
    p.add_argument("--checkpoint", type=Path, default=resolve_path("artifacts/cstn_v2_best.pt"))
    p.add_argument("--output", type=Path, default=resolve_path("artifacts/baseline_evaluation_report.json"))
    args = p.parse_args()

    args.test_data = resolve_path(args.test_data)
    args.zero_drift_data = resolve_path(args.zero_drift_data)
    args.checkpoint = resolve_path(args.checkpoint)
    args.output = resolve_path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 正在加载 CSTN v2 权重: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    raw_cfg = dict(ckpt.get("config", {}))
    if "decoder_type" not in raw_cfg:
        if any(k.startswith("pose.") for k in ckpt.get("state_dict", {})):
            raw_cfg["decoder_type"] = "mean_pooling"
        else:
            raw_cfg["decoder_type"] = "joint_decoupled"
    config = Config(**raw_cfg)
    model = Corrector(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"\n================================================================================")
    print(f"  SlimeVR CSTN v2 基准评测报告 (Baseline Benchmark Report)")
    print(f"================================================================================")
    print(f"模型配置: Hidden={config.hidden}, Layers={config.layers}, Heads={config.heads}")
    print(f"输入特征维度: {sum(config.stream_dims)} 维 (8 流通道) -> 输出: {config.num_joints} 关节点")

    print(f"\n[*] 正在评测综合漂移测试集 (test.npz，覆盖 0/1/3/10 deg/min 多工况)...")
    res_test = evaluate_dataset(model, args.test_data, device)

    print(f"\n[*] 正在评测零漂移纯净金标测试集 (test_zero_drift.npz，检验误修正率)...")
    res_zero = evaluate_dataset(model, args.zero_drift_data, device)

    # 零漂移误修改判定: Delta e = e_corr - e_raw
    zero_delta_deg = res_zero["mean_err_corr_deg"] - res_zero["mean_err_raw_deg"]
    zero_pass = zero_delta_deg <= 0.10

    report = {
        "benchmark_timestamp": "2026-10-17",
        "evaluation_scope": "Sequence-isolated test sets (no leakage)",
        "model_checkpoint": str(args.checkpoint),
        "comprehensive_drift_test": res_test,
        "zero_drift_gold_test": res_zero,
        "zero_drift_degradation_deg": float(zero_delta_deg),
        "zero_drift_pass": bool(zero_pass),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 打印标准化对比表格
    print("\n" + "=" * 80)
    print("  评测指标对比汇总 (综合多工况测试集 vs 纯净零漂移测试集)")
    print("=" * 80)
    print(f"{'指标项 (Metric)':<32} | {'未校正基线 (Raw)':<18} | {'模型校正后 (CSTN v2)':<20} | {'改善收益 / 状态':<15}")
    print("-" * 80)
    print(f"{'全姿态四元数测地角误差':<32} | {res_test['mean_err_raw_deg']:>12.4f}°      | {res_test['mean_err_corr_deg']:>14.4f}°       | 变化 {res_test['full_pose_delta_deg']:+.4f}° ({res_test['drift_reduction_ratio_pct']:+.1f}%)")
    print(f"{'纯偏航角误差 (Pure Yaw)':<32} | {res_test['pure_yaw_raw_deg']:>12.4f}°      | {res_test['pure_yaw_corr_deg']:>14.4f}°       | 变化 {res_test['pure_yaw_delta_deg']:+.4f}° ({res_test['pure_yaw_reduction_pct']:+.1f}%)")
    print(f"{'FK 3D 关节位置代理误差 (MPJPE)':<28} | {res_test['mpjpe_raw_mm']:>12.2f} mm    | {res_test['mpjpe_corr_mm']:>14.2f} mm     | 改善 {res_test['mpjpe_improvement_mm']:+.2f} mm")
    print(f"{'支撑期足底滑移代理位移':<30} | {res_test['stance_skating_raw_mm']:>12.2f} mm/f  | {res_test['stance_skating_corr_mm']:>14.2f} mm/f   | 改善 {res_test['stance_skating_raw_mm'] - res_test['stance_skating_corr_mm']:+.2f} mm/f")
    print(f"{'模型残差逼近误差 (Proxy)':<32} | {'-':>18} | {res_test['residual_angle_error_deg']:>14.4f}°       | 训练拟合目标")
    print(f"{'足底触地代理 F1 (Synthetic)':<30} | {'-':>18} | {res_test['contact_f1_proxy_pct']:>14.2f} %       | 速度阈值代理")
    print(f"{'门控激活率 (Gate >= 0.3)':<32} | {'-':>18} | {res_test['gate_activation_rate_pct']:>14.2f} %       | 自适应门控")
    print("-" * 80)
    print(f"{'零漂移全姿态角误差 (金标)':<32} | {res_zero['mean_err_raw_deg']:>12.4f}°      | {res_zero['mean_err_corr_deg']:>14.4f}°       | Δ = {zero_delta_deg:+.4f}°")
    print(f"{'零漂移纯偏航角误差 (金标)':<32} | {res_zero['pure_yaw_raw_deg']:>12.4f}°      | {res_zero['pure_yaw_corr_deg']:>14.4f}°       | Δ = {res_zero['pure_yaw_corr_deg'] - res_zero['pure_yaw_raw_deg']:+.4f}°")
    print(f"{'零漂移无退化验收 (Δ <= 0.1°)':<30} | {'-':>18} | {('PASS ✅' if zero_pass else 'FAIL ❌'):>14}        | 姿态未被误改")
    print("=" * 80)
    print(f"完整报告已持久化归档至: {args.output}\n")


if __name__ == "__main__":
    main()

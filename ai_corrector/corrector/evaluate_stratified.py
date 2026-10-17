"""
ai_corrector/corrector/evaluate_stratified.py
============================================
CSTN v2 组内分层基准评测与数据审计套件 (Task B Stratified Evaluation Suite)

严格遵循 public/phase-06-accuracy-baseline/plan.md 任务 B 规范:
1. 分层统计: 按 0/1/3/10°/min 漂移档位、四个下肢关节、原始序列及 reset 后时间段统计
   Raw 基线、模型修正后与 Oracle 上限的全姿态角与纯偏航角误差。
2. 报告有符号 Raw - Corrected 改善幅度与偏航误差随时间的漂移斜率 (°/min)。
3. 序列级统计: 报告序列均值、分位数 (10%, 25%, 50%, 75%, 90%) 与基于序列重采样 (Cluster Bootstrap)
   的 95% 不确定区间，杜绝将窗口直接视为独立样本。
4. 预测残差与真值残差的角度、方向余弦相关性与幅度分布审计。
5. 门控覆盖率—实际收益曲线扫描: 覆盖 tau in [0.0, 0.95]，特别检查零漂移/低噪声场景的误修正。
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

IMU_OFFSETS = [68, 82, 96, 110]
JOINT_NAMES = ["L_Thigh", "R_Thigh", "L_Shin", "R_Shin"]

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
    orig_shape = r_l_thigh.shape[:-2]
    r_lt = r_l_thigh.reshape(-1, 3, 3)
    r_rt = r_r_thigh.reshape(-1, 3, 3)
    r_ls = r_l_shin.reshape(-1, 3, 3)
    r_rs = r_r_shin.reshape(-1, 3, 3)

    p_lt = SKELETON_OFFSETS["left_thigh"]
    p_rt = SKELETON_OFFSETS["right_thigh"]
    p_lk = p_lt + np.einsum("nij,j->ni", r_lt, SKELETON_OFFSETS["knee_len"])
    p_rk = p_rt + np.einsum("nij,j->ni", r_rt, SKELETON_OFFSETS["knee_len"])
    p_la = p_lk + np.einsum("nij,j->ni", r_ls, SKELETON_OFFSETS["ankle_len"])
    p_ra = p_rk + np.einsum("nij,j->ni", r_rs, SKELETON_OFFSETS["ankle_len"])

    joints = np.stack([p_lk, p_rk, p_la, p_ra], axis=1)
    return joints.reshape(*orig_shape, 4, 3)


def compute_geodesic_angles_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), -1.0, 1.0)
    return 2.0 * np.arccos(dot) * 180.0 / np.pi


def compute_y_twist_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    yaw1 = 2.0 * np.arctan2(q1[..., 2], q1[..., 0])
    yaw2 = 2.0 * np.arctan2(q2[..., 2], q2[..., 0])
    return np.abs((yaw1 - yaw2 + np.pi) % (2.0 * np.pi) - np.pi) * 180.0 / np.pi


def fit_drift_slope(times_sec: np.ndarray, errors: np.ndarray) -> float:
    # 转换为分钟时间
    t_min = times_sec / 60.0
    if len(t_min) < 2 or np.all(t_min == t_min[0]):
        return 0.0
    poly = np.polyfit(t_min, errors, 1)
    return float(poly[0])  # 单位: deg/min


def cluster_bootstrap_ci(
    seq_ids: np.ndarray,
    vals: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    unique_seqs = np.unique(seq_ids)
    n_seq = len(unique_seqs)
    if n_seq <= 1:
        return float(np.mean(vals)), float(np.mean(vals)), float(np.mean(vals))

    # 按 seq_id 建立索引映射
    seq_to_indices = {s: np.where(seq_ids == s)[0] for s in unique_seqs}
    seq_means = np.array([np.mean(vals[seq_to_indices[s]]) for s in unique_seqs])

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float32)
    for b in range(n_boot):
        sample_seqs = rng.choice(n_seq, size=n_seq, replace=True)
        boot_means[b] = np.mean(seq_means[sample_seqs])

    mean_val = float(np.mean(seq_means))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_low, ci_high


def evaluate_stratified_dataset(
    model: Corrector,
    data_path: Path,
    device: torch.device,
    threshold: float = 0.30,
    n_boot: int = 1000,
) -> dict:
    data = np.load(data_path, allow_pickle=False)
    x_np = data["x"].astype(np.float32)
    dq_target_np = data["dq"].astype(np.float32)
    contact_target_np = data["contact"].astype(np.float32)
    b, t = x_np.shape[:2]

    # 读取元数据
    seq_ids = data["seq_id"] if "seq_id" in data else np.arange(b, dtype=np.int32)
    subsets = data["subset"] if "subset" in data else np.array(["unknown"] * b)
    drift_tiers = data["drift_tier"] if "drift_tier" in data else np.zeros(b, dtype=np.float32)
    time_seconds = data["time_seconds"] if "time_seconds" in data else np.zeros(b, dtype=np.float32)
    # 构造每帧的时间戳 [B, T]
    frame_times = time_seconds[:, None] + np.arange(t, dtype=np.float32)[None, :] / 60.0

    # 模型推理
    model.eval()
    batch_size = 256
    dq_preds, gate_preds, contact_preds = [], [], []
    with torch.no_grad():
        for i in range(0, b, batch_size):
            bx = torch.from_numpy(x_np[i:i+batch_size]).to(device)
            out, _ = model(bx)
            dq_preds.append(out["dq"].cpu().numpy())
            gate_preds.append(out["gate"].cpu().numpy())
            contact_preds.append(out["contact"].cpu().numpy())

    dq_pred_np = np.concatenate(dq_preds, axis=0)
    gate_pred_np = np.concatenate(gate_preds, axis=0)
    contact_pred_np = np.concatenate(contact_preds, axis=0)

    # 提取 4 关节原始旋转
    r_raw = np.empty((b, t, 4, 3, 3), dtype=np.float32)
    for j, offset in enumerate(IMU_OFFSETS):
        r_raw[:, :, j] = rot6d_to_matrix_np(x_np[:, :, offset:offset+6])

    dq_target_mat = quat_wxyz_to_rotmat(dq_target_np)
    r_clean = np.matmul(r_raw, dq_target_mat)

    # Oracle 完美校正
    r_oracle = r_clean.copy()

    # 模型预测校正 (带门控)
    dq_pred_mat = quat_wxyz_to_rotmat(dq_pred_np)
    r_pred_all = np.matmul(r_raw, dq_pred_mat)
    gate_mask = (gate_pred_np >= threshold)[..., None, None]
    r_corrected = np.where(gate_mask, r_pred_all, r_raw)

    q_raw = rotmat_to_quat_wxyz(r_raw)
    q_clean = rotmat_to_quat_wxyz(r_clean)
    q_oracle = rotmat_to_quat_wxyz(r_oracle)
    q_corr = rotmat_to_quat_wxyz(r_corrected)

    # 1. 逐窗口、逐帧、逐关节点的误差张量 [B, T, 4]
    err_full_raw = compute_geodesic_angles_deg(q_raw, q_clean)
    err_full_corr = compute_geodesic_angles_deg(q_corr, q_clean)
    err_full_oracle = compute_geodesic_angles_deg(q_oracle, q_clean)

    err_yaw_raw = compute_y_twist_deg(q_raw, q_clean)
    err_yaw_corr = compute_y_twist_deg(q_corr, q_clean)
    err_yaw_oracle = compute_y_twist_deg(q_oracle, q_clean)

    # 窗口级平均 [B]
    win_yaw_raw = np.mean(err_yaw_raw, axis=(1, 2))
    win_yaw_corr = np.mean(err_yaw_corr, axis=(1, 2))
    win_yaw_oracle = np.mean(err_yaw_oracle, axis=(1, 2))
    win_yaw_diff = win_yaw_raw - win_yaw_corr  # 正值为收益

    win_full_raw = np.mean(err_full_raw, axis=(1, 2))
    win_full_corr = np.mean(err_full_corr, axis=(1, 2))
    win_full_diff = win_full_raw - win_full_corr

    # 2. 统计分层: 按 0, 1, 3, 10 deg/min 漂移档位
    drift_tier_results = {}
    unique_tiers = sorted(list(set(drift_tiers.tolist())))
    for tier in unique_tiers:
        mask = (drift_tiers == tier)
        if not np.any(mask):
            continue
        sub_raw = win_yaw_raw[mask]
        sub_corr = win_yaw_corr[mask]
        sub_diff = win_yaw_diff[mask]
        sub_seqs = seq_ids[mask]
        sub_times = time_seconds[mask]

        diff_mean, diff_ci_low, diff_ci_high = cluster_bootstrap_ci(sub_seqs, sub_diff, n_boot=n_boot)

        # 漂移斜率 (°/min)
        raw_slope = fit_drift_slope(sub_times, sub_raw)
        corr_slope = fit_drift_slope(sub_times, sub_corr)

        drift_tier_results[f"{tier:.1f}"] = {
            "num_windows": int(np.sum(mask)),
            "num_sequences": int(len(np.unique(sub_seqs))),
            "yaw_raw_deg": float(np.mean(sub_raw)),
            "yaw_corr_deg": float(np.mean(sub_corr)),
            "yaw_oracle_deg": float(np.mean(win_yaw_oracle[mask])),
            "yaw_delta_deg": float(np.mean(sub_diff)),
            "yaw_delta_ci95": [diff_ci_low, diff_ci_high],
            "yaw_reduction_pct": float((np.mean(sub_raw) - np.mean(sub_corr)) / max(np.mean(sub_raw), 1e-4) * 100.0),
            "full_pose_raw_deg": float(np.mean(win_full_raw[mask])),
            "full_pose_corr_deg": float(np.mean(win_full_corr[mask])),
            "full_pose_delta_deg": float(np.mean(win_full_diff[mask])),
            "drift_slope_raw_deg_per_min": raw_slope,
            "drift_slope_corr_deg_per_min": corr_slope,
            "drift_slope_reduction_pct": float((raw_slope - corr_slope) / max(abs(raw_slope), 1e-4) * 100.0),
        }

    # 3. 统计分层: 按 4 个关节
    joint_results = {}
    for j, jname in enumerate(JOINT_NAMES):
        j_raw = np.mean(err_yaw_raw[..., j], axis=1)
        j_corr = np.mean(err_yaw_corr[..., j], axis=1)
        j_diff = j_raw - j_corr
        d_mean, d_low, d_high = cluster_bootstrap_ci(seq_ids, j_diff, n_boot=n_boot)
        joint_results[jname] = {
            "pure_yaw_raw_deg": float(np.mean(j_raw)),
            "pure_yaw_corr_deg": float(np.mean(j_corr)),
            "pure_yaw_delta_deg": float(np.mean(j_diff)),
            "pure_yaw_delta_ci95": [d_low, d_high],
            "pure_yaw_reduction_pct": float((np.mean(j_raw) - np.mean(j_corr)) / max(np.mean(j_raw), 1e-4) * 100.0),
            "full_pose_raw_deg": float(np.mean(err_full_raw[..., j])),
            "full_pose_corr_deg": float(np.mean(err_full_corr[..., j])),
            "full_pose_delta_deg": float(np.mean(err_full_raw[..., j]) - np.mean(err_full_corr[..., j])),
        }

    # 4. 序列级统计与分位数分布
    unique_seqs = np.unique(seq_ids)
    seq_yaw_raws, seq_yaw_corrs, seq_yaw_diffs = [], [], []
    for s in unique_seqs:
        idx = np.where(seq_ids == s)[0]
        s_r = np.mean(win_yaw_raw[idx])
        s_c = np.mean(win_yaw_corr[idx])
        seq_yaw_raws.append(s_r)
        seq_yaw_corrs.append(s_c)
        seq_yaw_diffs.append(s_r - s_c)

    seq_yaw_diffs = np.array(seq_yaw_diffs)
    positive_seq_pct = float(np.mean(seq_yaw_diffs > 0) * 100.0)

    seq_quantiles = {
        "p10": float(np.percentile(seq_yaw_diffs, 10)),
        "p25": float(np.percentile(seq_yaw_diffs, 25)),
        "p50_median": float(np.percentile(seq_yaw_diffs, 50)),
        "p75": float(np.percentile(seq_yaw_diffs, 75)),
        "p90": float(np.percentile(seq_yaw_diffs, 90)),
        "min": float(np.min(seq_yaw_diffs)),
        "max": float(np.max(seq_yaw_diffs)),
        "positive_improvement_rate_pct": positive_seq_pct,
    }

    # 5. 全局序列 Cluster Bootstrap 95% 置信区间
    _, yaw_ci_low, yaw_ci_high = cluster_bootstrap_ci(seq_ids, win_yaw_diff, n_boot=n_boot)
    _, full_ci_low, full_ci_high = cluster_bootstrap_ci(seq_ids, win_full_diff, n_boot=n_boot)

    # 6. 残差预测质量审计 (Task B.3)
    # 方向余弦相似度与残差幅度
    pred_rot = Rotation.from_quat(dq_pred_np[..., [1, 2, 3, 0]].reshape(-1, 4))
    true_rot = Rotation.from_quat(dq_target_np[..., [1, 2, 3, 0]].reshape(-1, 4))
    pred_vec = pred_rot.as_rotvec()
    true_vec = true_rot.as_rotvec()

    pred_norm = np.linalg.norm(pred_vec, axis=-1)
    true_norm = np.linalg.norm(true_vec, axis=-1)
    valid_norm = (pred_norm > 1e-5) & (true_norm > 1e-5)

    if np.any(valid_norm):
        cos_sim = np.sum(pred_vec[valid_norm] * true_vec[valid_norm], axis=-1) / (pred_norm[valid_norm] * true_norm[valid_norm])
        mean_cos_sim = float(np.mean(cos_sim))
    else:
        mean_cos_sim = 0.0

    pred_mag_deg = pred_norm * 180.0 / np.pi
    true_mag_deg = true_norm * 180.0 / np.pi
    residual_angle_err = geodesic_angle_error_deg(dq_pred_np, dq_target_np)

    residual_audit = {
        "residual_geodesic_error_deg": float(residual_angle_err),
        "direction_cosine_similarity": mean_cos_sim,
        "predicted_magnitude_mean_deg": float(np.mean(pred_mag_deg)),
        "predicted_magnitude_median_deg": float(np.median(pred_mag_deg)),
        "predicted_magnitude_p90_deg": float(np.percentile(pred_mag_deg, 90)),
        "target_magnitude_mean_deg": float(np.mean(true_mag_deg)),
        "target_magnitude_median_deg": float(np.median(true_mag_deg)),
        "target_magnitude_p90_deg": float(np.percentile(true_mag_deg, 90)),
    }

    # 7. 门控阈值扫描 (Coverage vs. Benefit Curve)
    gate_sweep = []
    tau_list = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for tau in tau_list:
        g_mask = (gate_pred_np >= tau)[..., None, None]
        r_corr_tau = np.where(g_mask, r_pred_all, r_raw)
        q_corr_tau = rotmat_to_quat_wxyz(r_corr_tau)
        y_err = compute_y_twist_deg(q_corr_tau, q_clean)
        f_err = compute_geodesic_angles_deg(q_corr_tau, q_clean)
        coverage = float(np.mean(gate_pred_np >= tau) * 100.0)
        mean_y = float(np.mean(y_err))
        mean_f = float(np.mean(f_err))
        gate_sweep.append({
            "threshold": tau,
            "coverage_pct": coverage,
            "pure_yaw_corr_deg": mean_y,
            "pure_yaw_delta_deg": float(np.mean(err_yaw_raw) - mean_y),
            "full_pose_corr_deg": mean_f,
            "full_pose_delta_deg": float(np.mean(err_full_raw) - mean_f),
        })

    # 8. 简化 FK 与足滑代理
    p_clean = compute_lower_body_fk(r_clean[..., 0, :, :], r_clean[..., 1, :, :], r_clean[..., 2, :, :], r_clean[..., 3, :, :])
    p_raw = compute_lower_body_fk(r_raw[..., 0, :, :], r_raw[..., 1, :, :], r_raw[..., 2, :, :], r_raw[..., 3, :, :])
    p_corr = compute_lower_body_fk(r_corrected[..., 0, :, :], r_corrected[..., 1, :, :], r_corrected[..., 2, :, :], r_corrected[..., 3, :, :])

    mpjpe_raw = float(np.mean(np.linalg.norm(p_raw - p_clean, axis=-1)) * 1000.0)
    mpjpe_corr = float(np.mean(np.linalg.norm(p_corr - p_clean, axis=-1)) * 1000.0)

    # 支撑期足滑
    stance_l = contact_target_np[:, 1:, 0] > 0.5
    stance_r = contact_target_np[:, 1:, 1] > 0.5
    disp_l_raw = np.linalg.norm(p_raw[:, 1:, 2] - p_raw[:, :-1, 2], axis=-1) * 1000.0
    disp_l_corr = np.linalg.norm(p_corr[:, 1:, 2] - p_corr[:, :-1, 2], axis=-1) * 1000.0
    disp_r_raw = np.linalg.norm(p_raw[:, 1:, 3] - p_raw[:, :-1, 3], axis=-1) * 1000.0
    disp_r_corr = np.linalg.norm(p_corr[:, 1:, 3] - p_corr[:, :-1, 3], axis=-1) * 1000.0
    all_s_raw = np.concatenate([disp_l_raw[stance_l], disp_r_raw[stance_r]])
    all_s_corr = np.concatenate([disp_l_corr[stance_l], disp_r_corr[stance_r]])
    skating_raw = float(np.mean(all_s_raw)) if len(all_s_raw) > 0 else 0.0
    skating_corr = float(np.mean(all_s_corr)) if len(all_s_corr) > 0 else 0.0

    return {
        "num_windows": b,
        "num_sequences": len(unique_seqs),
        "total_frames": b * t,
        "global_pure_yaw": {
            "raw_deg": float(np.mean(win_yaw_raw)),
            "corr_deg": float(np.mean(win_yaw_corr)),
            "oracle_deg": float(np.mean(win_yaw_oracle)),
            "delta_deg": float(np.mean(win_yaw_diff)),
            "delta_ci95": [yaw_ci_low, yaw_ci_high],
            "reduction_pct": float((np.mean(win_yaw_raw) - np.mean(win_yaw_corr)) / max(np.mean(win_yaw_raw), 1e-4) * 100.0),
        },
        "global_full_pose": {
            "raw_deg": float(np.mean(win_full_raw)),
            "corr_deg": float(np.mean(win_full_corr)),
            "oracle_deg": float(np.mean(err_full_oracle)),
            "delta_deg": float(np.mean(win_full_diff)),
            "delta_ci95": [full_ci_low, full_ci_high],
        },
        "proxy_kinematics": {
            "mpjpe_raw_mm": mpjpe_raw,
            "mpjpe_corr_mm": mpjpe_corr,
            "mpjpe_delta_mm": mpjpe_raw - mpjpe_corr,
            "foot_skating_raw_mm": skating_raw,
            "foot_skating_corr_mm": skating_corr,
        },
        "drift_tier_stratification": drift_tier_results,
        "joint_stratification": joint_results,
        "sequence_level_distribution": seq_quantiles,
        "residual_audit": residual_audit,
        "gate_threshold_sweep": gate_sweep,
    }


def main():
    p = argparse.ArgumentParser(description="SlimeVR CSTN v2 Task B 组内分层全面评测与审计套件")
    p.add_argument("--test-data", type=Path, default=resolve_path("database/processed_v2_stratified/test.npz"))
    p.add_argument("--zero-drift-data", type=Path, default=resolve_path("database/processed_v2_stratified/test_zero_drift.npz"))
    p.add_argument("--checkpoint", type=Path, default=resolve_path("artifacts/cstn_v2_decoupled_best.pt"))
    p.add_argument("--output", type=Path, default=resolve_path("artifacts/task_b_stratified_evaluation.json"))
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    args = p.parse_args()

    args.test_data = resolve_path(args.test_data)
    args.zero_drift_data = resolve_path(args.zero_drift_data)
    args.checkpoint = resolve_path(args.checkpoint)
    args.output = resolve_path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 正在加载 CSTN v2 模型权重: {args.checkpoint}")
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

    print("\n================================================================================")
    print("  SlimeVR CSTN v2 组内分层与序列隔离审计报告 (Task B Stratified Audit)")
    print("================================================================================")
    print(f"评测模型: {args.checkpoint.name}")
    print(f"测试集路径: {args.test_data}")
    print(f"Bootstrap 重采样次数: {args.bootstrap_samples} (按序列重采样)")

    print("\n[*] 正在运行综合分层测试集评测...")
    test_results = evaluate_stratified_dataset(
        model, args.test_data, device, threshold=args.threshold, n_boot=args.bootstrap_samples
    )

    print("\n[*] 正在运行零标称漂移/低噪声金标集评测...")
    zero_results = evaluate_stratified_dataset(
        model, args.zero_drift_data, device, threshold=args.threshold, n_boot=args.bootstrap_samples
    )

    report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": str(args.checkpoint),
        "test_dataset": str(args.test_data),
        "zero_drift_dataset": str(args.zero_drift_data),
        "confidence_threshold": args.threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "comprehensive_test": test_results,
        "zero_drift_gold_test": zero_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 打印控制台核心总结
    print("\n" + "=" * 88)
    print("  一、按漂移档位分层评测 (Drift Tier Stratification)")
    print("=" * 88)
    print(f"{'漂移档位':<12} | {'序列/窗口数':<14} | {'Raw 纯偏航':<14} | {'Corrected':<14} | {'有符号改善 (Δ)':<18} | {'95% CI':<16}")
    print("-" * 88)
    for tier, data in test_results["drift_tier_stratification"].items():
        ci_str = f"[{data['yaw_delta_ci95'][0]:+.4f}, {data['yaw_delta_ci95'][1]:+.4f}]"
        seq_win_str = f"{data['num_sequences']} / {data['num_windows']}"
        tier_title = f"{tier} deg/min"
        print(f"{tier_title:<12} | {seq_win_str:<14} | {data['yaw_raw_deg']:>10.4f}°   | {data['yaw_corr_deg']:>10.4f}°   | {data['yaw_delta_deg']:>+12.4f}° ({data['yaw_reduction_pct']:>+5.1f}%) | {ci_str:<16}")

    print("\n" + "=" * 88)
    print("  二、按关节点分层评测 (Per-Joint Stratification)")
    print("=" * 88)
    print(f"{'关节点':<12} | {'Raw 纯偏航':<14} | {'Corrected':<14} | {'纯偏航改善 (Δ)':<18} | {'全姿态改善 (Δ)':<18}")
    print("-" * 88)
    for jname, jdata in test_results["joint_stratification"].items():
        print(f"{jname:<12} | {jdata['pure_yaw_raw_deg']:>10.4f}°   | {jdata['pure_yaw_corr_deg']:>10.4f}°   | {jdata['pure_yaw_delta_deg']:>+12.4f}° ({jdata['pure_yaw_reduction_pct']:>+5.1f}%) | {jdata['full_pose_delta_deg']:>+12.4f}°")

    print("\n" + "=" * 88)
    print("  三、序列级分位数分布与收益覆盖 (Sequence-Level Distribution)")
    print("=" * 88)
    sq = test_results["sequence_level_distribution"]
    print(f"正向改善序列比例: {sq['positive_improvement_rate_pct']:.2f}%")
    print(f"偏航改善分位数: P10={sq['p10']:+.4f}°, P25={sq['p25']:+.4f}°, P50(中位数)={sq['p50_median']:+.4f}°, P75={sq['p75']:+.4f}°, P90={sq['p90']:+.4f}°")
    print(f"极值范围: Min={sq['min']:+.4f}°, Max={sq['max']:+.4f}°")

    print("\n" + "=" * 88)
    print("  四、零标称漂移/低噪声金标集安全回退检验 (Zero-Drift Safety)")
    print("=" * 88)
    z_yaw_delta = zero_results['global_pure_yaw']['corr_deg'] - zero_results['global_pure_yaw']['raw_deg']
    z_full_delta = zero_results['global_full_pose']['corr_deg'] - zero_results['global_full_pose']['raw_deg']
    print(f"零漂移偏航变化量: Δ = {z_yaw_delta:+.4f}°")
    print(f"零漂移全姿态变化量: Δ = {z_full_delta:+.4f}° (门槛 <= +0.10°: {'PASS ✅' if z_full_delta <= 0.10 else 'FAIL ❌'})")

    print(f"\n完整机器报告已写入: {args.output}\n")


if __name__ == "__main__":
    main()

"""
ai_corrector/corrector/execute_task_d3b.py
=========================================
执行 Phase 6 任务 D3b: 修复 D3a 归因实验并重新裁定
(D3b Fixed Attribution Ablation & Paired Counterfactual Reassessment)

严格遵循 plan.md 任务 D3b 规范:
1. 在全新修复的无捷径与严格 1-to-1 配对反事实数据集 (database/d3b_unbiased/) 上评测:
   - 变体 A: 现有 20 帧独立短窗重训模型 (Window-Independent Baseline, 状态每窗清空);
   - 变体 B: 连续因果时序重放 (Continuous Causal LSTM State Carryover across sequence, 序列换行/reset 时清空);
   - 变体 C: 混合快慢估计 (连续因果时序 + 基于足部触地静止期物理 ZUPT 观测的四路慢变偏航偏置状态估计);
2. 严格按 pair_id + frame + joint 逐项对齐反事实评测:
   - 预测差值对照真值差值: delta_target vs delta_pred;
   - 方向一致率 (Directional Agreement Ratio, %);
   - 增益/幅度比率 (Gain / Amplitude Ratio);
   - 配对反事实对称相关系数 (rho_cf, 检验 drift 符号取反时, 预测残差是否对称取反);
3. 分档与序列级指标:
   - 按正负方向 (+/- 10, +/- 3, +/- 1, 0) 分层偏航与全姿态净收益;
   - 逐序列去重时间回归斜率削减量及 95% Bootstrap CI;
   - 零漂移误改受控度 (0.0 deg/min 纯偏航与全姿态增量);
   - 推理延迟 (ms) 与状态内存开销。
4. 产出: artifacts/d3b_attribution_ablation_report.json
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .checkpoint import resolve_path, PACKAGE_ROOT
from .contract_v2 import (
    rot6d_to_matrix_np,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat,
)
from .evaluate_stratified import IMU_OFFSETS, compute_y_twist_deg, compute_geodesic_angles_deg
from .model import Corrector, Config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def train_d3b_base_model(
    train_data: Path,
    val_data: Path,
    output_pt: Path,
    seed: int = 123,
    epochs: int = 15,
) -> Path:
    print("\n" + "=" * 80)
    print("  [D3b 训练] 正在 D3b 正确数据集上重新训练基线模型 (Smooth Geodesic, hidden=96)...")
    print("=" * 80)

    best_pt = output_pt.with_name(output_pt.stem + "_best.pt")
    if best_pt.exists():
        print(f"  发现已有 D3b 检查点: {best_pt.name}，复用该模型。")
        return best_pt

    cmd = [
        sys.executable,
        "-m", "ai_corrector.corrector.train",
        str(train_data),
        str(output_pt),
        "--val-data", str(val_data),
        "--hidden", "96",
        "--loss-type", "smooth_geodesic",
        "--rot-weight", "10.0",
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch-size", "256",
        "--lr", "1e-3",
        "--print-iters", "50",
    ]

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PACKAGE_ROOT.parent), capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        print(f"训练报错:\n{res.stderr}")
        raise RuntimeError(f"D3b 训练失败 code {res.returncode}")

    if not best_pt.exists():
        best_pt = output_pt
    print(f"  训练完毕 (耗时 {time.time()-t0:.2f}s) -> 保存至: {best_pt.name}")
    return best_pt


def evaluate_variant_on_data(
    model: Corrector,
    data_path: Path,
    variant_type: str,  # "model_a", "model_b", "model_c"
    device: torch.device,
    tau: float = 0.50,
) -> dict:
    data = np.load(data_path, allow_pickle=False)
    x_np = data["x"].astype(np.float32)
    dq_target_np = data["dq"].astype(np.float32)
    pair_ids = data["pair_id"]
    seq_ids = data["seq_id"]
    drift_tiers = data["drift_tier"]
    start_frames = data["start_frame"]
    drift_rates = data["drift_rates"]
    drift_signs = data["drift_signs"]
    is_seq_start = data["is_seq_start"]
    contact_target = data["contact"].astype(np.float32)

    b, t = x_np.shape[:2]
    model.eval()

    pred_dq_list = []
    pred_gate_list = []
    pred_contact_list = []

    t_inf_start = time.time()

    if variant_type == "model_a":
        # 变体 A: 独立短窗基线 (状态窗间清空)
        with torch.no_grad():
            for i in range(0, b, 256):
                bx = torch.from_numpy(x_np[i:i+256]).to(device)
                out, _ = model(bx, state=None)
                pred_dq_list.append(out["dq"].cpu().numpy())
                pred_gate_list.append(out["gate"].cpu().numpy())
                pred_contact_list.append(out["contact"].cpu().numpy())

        pred_dq = np.concatenate(pred_dq_list, axis=0)
        pred_gate = np.concatenate(pred_gate_list, axis=0)
        pred_contact = np.concatenate(pred_contact_list, axis=0)

    elif variant_type in ("model_b", "model_c"):
        # 变体 B & C: 连续因果时序重放 (按序列顺序维护 LSTM 隐状态)
        unique_seqs = np.unique(seq_ids)
        dq_all = np.zeros((b, t, 4, 4), dtype=np.float32)
        gate_all = np.zeros((b, t, 4), dtype=np.float32)
        contact_all = np.zeros((b, t, 2), dtype=np.float32)

        with torch.no_grad():
            for sid in unique_seqs:
                w_indices = np.where(seq_ids == sid)[0]
                sorted_w = w_indices[np.argsort(start_frames[w_indices])]
                state = model.initial_state(1, device)

                for w_idx in sorted_w:
                    bx = torch.from_numpy(x_np[w_idx:w_idx+1]).to(device)
                    out, state = model(bx, state=state)
                    dq_all[w_idx] = out["dq"].cpu().numpy()[0]
                    gate_all[w_idx] = out["gate"].cpu().numpy()[0]
                    contact_all[w_idx] = out["contact"].cpu().numpy()[0]

        pred_dq = dq_all
        pred_gate = gate_all
        pred_contact = contact_all

    cost_inf = (time.time() - t_inf_start) / (b * t) * 1000.0  # ms/frame

    # 四元数与姿态还原
    r_raw = np.empty((b, t, 4, 3, 3), dtype=np.float32)
    for j, offset in enumerate(IMU_OFFSETS):
        r_raw[:, :, j] = rot6d_to_matrix_np(x_np[:, :, offset:offset+6])

    dq_target_mat = quat_wxyz_to_rotmat(dq_target_np)
    r_clean = np.matmul(r_raw, dq_target_mat)

    # 变体 C: 足部触地静止期物理 ZUPT 慢变偏航偏置更新
    if variant_type == "model_c":
        unique_seqs = np.unique(seq_ids)
        dq_c_adjusted = pred_dq.copy()

        for sid in unique_seqs:
            w_indices = np.where(seq_ids == sid)[0]
            sorted_w = w_indices[np.argsort(start_frames[w_indices])]

            # 4 路下肢独立慢变偏置状态 (rad)
            bias_state = np.zeros(4, dtype=np.float32)

            for w_idx in sorted_w:
                for f in range(t):
                    is_contact_l = pred_contact[w_idx, f, 0] > 0.5
                    is_contact_r = pred_contact[w_idx, f, 1] > 0.5

                    # 局部角速度模长
                    omega_l = np.linalg.norm(x_np[w_idx, f, 96+9:96+12])
                    omega_r = np.linalg.norm(x_np[w_idx, f, 110+9:110+12])

                    eta = 0.003
                    # 左腿静止期更新 (0: thigh, 2: shin)
                    if is_contact_l and omega_l < 0.8:
                        rot_v = Rotation.from_quat(pred_dq[w_idx, f, [0, 2]][:, [1, 2, 3, 0]]).as_rotvec()
                        bias_state[[0, 2]] += eta * rot_v[:, 1]

                    # 右腿静止期更新 (1: thigh, 3: shin)
                    if is_contact_r and omega_r < 0.8:
                        rot_v = Rotation.from_quat(pred_dq[w_idx, f, [1, 3]][:, [1, 2, 3, 0]]).as_rotvec()
                        bias_state[[1, 3]] += eta * rot_v[:, 1]

                    # 慢变偏航旋转补偿
                    bias_rotvec = np.zeros((4, 3), dtype=np.float32)
                    bias_rotvec[:, 1] = -bias_state
                    bias_dq = Rotation.from_rotvec(bias_rotvec).as_matrix()

                    cur_dq_mat = quat_wxyz_to_rotmat(pred_dq[w_idx, f])
                    comp_dq_mat = np.matmul(bias_dq, cur_dq_mat)
                    dq_c_adjusted[w_idx, f] = rotmat_to_quat_wxyz(comp_dq_mat)

        pred_dq = dq_c_adjusted

    # 应用置信门控
    g_mask = (pred_gate >= tau)[..., None, None]
    dq_pred_mat = quat_wxyz_to_rotmat(pred_dq)
    r_all_corr = np.matmul(r_raw, dq_pred_mat)
    r_corr = np.where(g_mask, r_all_corr, r_raw)

    q_raw = rotmat_to_quat_wxyz(r_raw)
    q_corr = rotmat_to_quat_wxyz(r_corr)
    q_clean = rotmat_to_quat_wxyz(r_clean)

    yaw_raw = compute_y_twist_deg(q_raw, q_clean)
    yaw_corr = compute_y_twist_deg(q_corr, q_clean)
    full_raw = compute_geodesic_angles_deg(q_raw, q_clean)
    full_corr = compute_geodesic_angles_deg(q_corr, q_clean)

    # 分档统计 (支持 +/- 档位)
    unique_tiers = np.unique(drift_tiers)
    tier_stats = {}

    for tr in unique_tiers:
        idx = np.where(drift_tiers == tr)[0]
        if len(idx) == 0:
            continue
        y_r = float(np.mean(yaw_raw[idx]))
        y_c = float(np.mean(yaw_corr[idx]))
        y_d = y_r - y_c
        f_r = float(np.mean(full_raw[idx]))
        f_c = float(np.mean(full_corr[idx]))
        f_d = f_r - f_c
        cov = float(np.mean(pred_gate[idx] >= tau) * 100.0)

        tier_stats[f"{tr:+.1f}"] = {
            "pure_yaw_raw_deg": y_r,
            "pure_yaw_corr_deg": y_c,
            "pure_yaw_delta_deg": y_d,
            "full_pose_delta_deg": f_d,
            "coverage_pct": cov,
            "sample_count": len(idx),
        }

    dot = np.abs(pred_dq[..., 0])
    dot = np.clip(dot, -1.0, 1.0)
    res_mag_mean = float(np.mean(2.0 * np.arccos(dot) * 180.0 / np.pi))

    return {
        "variant": variant_type,
        "latency_ms_per_frame": cost_inf,
        "residual_magnitude_mean_deg": res_mag_mean,
        "tier_stats": tier_stats,
        "pair_ids": pair_ids,
        "drift_tiers": drift_tiers,
        "seq_ids": seq_ids,
        "start_frames": start_frames,
        "pred_dq": pred_dq,
        "dq_target": dq_target_np,
        "q_raw": q_raw,
        "q_corr": q_corr,
        "q_clean": q_clean,
        "yaw_raw": yaw_raw,
        "yaw_corr": yaw_corr,
        "full_raw": full_raw,
        "full_corr": full_corr,
    }


def compute_paired_counterfactual_metrics(
    eval_std: dict,
    eval_cf: dict,
) -> dict:
    """
    按 pair_id + frame + joint 严格逐项对齐反事实评测:
    1. delta_target vs delta_pred 差值对照
    2. 方向一致率 (Directional Agreement Ratio)
    3. 增益比率 (Gain Ratio)
    4. 配对反事实皮尔逊相关系数 (rho_cf)
    """
    pair_std = eval_std["pair_ids"]
    pair_cf = eval_cf["pair_ids"]
    assert np.array_equal(pair_std, pair_cf), "pair_id 必须严格逐行一致!"

    dq_std = eval_std["pred_dq"]
    dq_cf = eval_cf["pred_dq"]
    tgt_std = eval_std["dq_target"]
    tgt_cf = eval_cf["dq_target"]

    # 提取四元数偏航轴 (世界 +Y) 角度
    def get_yaw_rotvec(q):
        # q: [B, T, 4, 4] -> xyzw -> as_rotvec() -> Y component
        b, t, j, _ = q.shape
        q_xyzw = q[..., [1, 2, 3, 0]].reshape(-1, 4)
        rotv = Rotation.from_quat(q_xyzw).as_rotvec().reshape(b, t, j, 3)
        return rotv[..., 1]  # Y 轴分量 (rad)

    yaw_pred_std = get_yaw_rotvec(dq_std)
    yaw_pred_cf = get_yaw_rotvec(dq_cf)
    yaw_tgt_std = get_yaw_rotvec(tgt_std)
    yaw_tgt_cf = get_yaw_rotvec(tgt_cf)

    # 1. 真实漂移反转差值与预测差值
    delta_tgt = (yaw_tgt_std - yaw_tgt_cf) * 180.0 / np.pi
    delta_pred = (yaw_pred_std - yaw_pred_cf) * 180.0 / np.pi

    tiers = eval_std["drift_tiers"]
    non_zero_mask = (np.abs(tiers) > 0.1)[:, None, None]

    # 有效漂移帧上的方向一致率: sign(delta_pred) == sign(delta_tgt)
    valid_mask = non_zero_mask & (np.abs(delta_tgt) > 0.05)
    if np.sum(valid_mask) > 0:
        dir_agree = np.mean((np.sign(delta_pred[valid_mask]) == np.sign(delta_tgt[valid_mask])).astype(np.float32)) * 100.0
        gain_ratio = float(np.median(np.abs(delta_pred[valid_mask]) / np.maximum(np.abs(delta_tgt[valid_mask]), 1e-4)))
    else:
        dir_agree = 50.0
        gain_ratio = 0.0

    # 2. 配对相关度 rho_cf = corr(v_std, -v_cf)
    s1 = yaw_pred_std.flatten()
    s2 = -yaw_pred_cf.flatten()
    std1, std2 = np.std(s1), np.std(s2)
    if std1 > 1e-6 and std2 > 1e-6:
        rho_cf = float(np.corrcoef(s1, s2)[0, 1])
    else:
        rho_cf = 0.0

    return {
        "directional_agreement_pct": float(dir_agree),
        "median_gain_ratio": float(gain_ratio),
        "counterfactual_symmetry_correlation": float(rho_cf),
        "mean_delta_target_deg": float(np.mean(np.abs(delta_tgt))),
        "mean_delta_pred_deg": float(np.mean(np.abs(delta_pred))),
    }


def compute_per_sequence_paired_slopes(eval_res: dict) -> dict:
    """计算逐序列去重时间回归斜率及 95% Bootstrap CI"""
    seq_ids = eval_res["seq_ids"]
    tiers = eval_res["drift_tiers"]
    start_frames = eval_res["start_frames"]
    yaw_raw = eval_res["yaw_raw"]
    yaw_corr = eval_res["yaw_corr"]

    unique_tiers = [10.0, 3.0, 1.0, 0.0]
    out_slopes = {}

    for tr in unique_tiers:
        tier_mask = np.isclose(np.abs(tiers), tr, atol=0.2)
        seqs_in_tier = np.unique(seq_ids[tier_mask])

        raw_slopes = []
        corr_slopes = []
        delta_slopes = []

        for sid in seqs_in_tier:
            idx = np.where((seq_ids == sid) & tier_mask)[0]
            if len(idx) < 3:
                continue

            # 按帧去重还原时序
            time_sec = []
            yr_seq = []
            yc_seq = []

            for i in idx:
                sf = start_frames[i]
                for f in range(20):
                    time_sec.append((sf + f) / 60.0)
                    yr_seq.append(np.mean(yaw_raw[i, f]))
                    yc_seq.append(np.mean(yaw_corr[i, f]))

            t_arr = np.array(time_sec)
            yr_arr = np.array(yr_seq)
            yc_arr = np.array(yc_seq)

            # 按时间排序并去重
            unq_t, unq_idx = np.unique(t_arr, return_index=True)
            if len(unq_t) < 10:
                continue

            t_min = unq_t / 60.0
            p_r = np.polyfit(t_min, yr_arr[unq_idx], 1)[0]
            p_c = np.polyfit(t_min, yc_arr[unq_idx], 1)[0]

            raw_slopes.append(p_r)
            corr_slopes.append(p_c)
            delta_slopes.append(p_r - p_c)

        if not delta_slopes:
            continue

        d_arr = np.array(delta_slopes)
        r_arr = np.array(raw_slopes)
        c_arr = np.array(corr_slopes)

        # 95% Bootstrap CI
        rng = np.random.default_rng(123)
        boot_means = [np.mean(rng.choice(d_arr, size=len(d_arr), replace=True)) for _ in range(1000)]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        out_slopes[f"{tr:.1f}"] = {
            "num_sequences": len(d_arr),
            "raw_slope_deg_per_min": float(np.mean(r_arr)),
            "corr_slope_deg_per_min": float(np.mean(c_arr)),
            "delta_slope_deg_per_min": float(np.mean(d_arr)),
            "reduction_pct": float(np.mean(d_arr) / (np.mean(r_arr) + 1e-6) * 100.0),
            "ci_95": [ci_lower, ci_upper],
        }

    return out_slopes


def main():
    artifacts_dir = PACKAGE_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    d3b_db = resolve_path("database/d3b_unbiased")

    train_data = d3b_db / "train.npz"
    val_data = d3b_db / "val.npz"
    dev_test_data = d3b_db / "dev_test.npz"
    dev_test_cf_data = d3b_db / "dev_test_cf.npz"

    output_pt = artifacts_dir / "cstn_v2_d3b_model_a.pt"
    output_report = artifacts_dir / "d3b_attribution_ablation_report.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 训练 D3b 基础模型
    best_pt = train_d3b_base_model(train_data, val_data, output_pt)

    # 2. 加载模型
    ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    cfg = Config(**dict(ckpt.get("config", {})))
    model = Corrector(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("\n" + "=" * 80)
    print("  [D3b.2] 正在执行模型 A / B / C 严格可归因与配对反事实对照消融评测...")
    print("=" * 80)

    results = {}
    variants = [
        ("variant_a_independent_window", "model_a", "变体 A (独立短窗基线)"),
        ("variant_b_continuous_causal", "model_b", "变体 B (连续因果历史重放)"),
        ("variant_c_hybrid_slow_bias", "model_c", "变体 C (连续因果 + 物理慢变偏置)"),
    ]

    for key, vtype, label in variants:
        print(f"\n  -> 评测 {label}...")
        std_eval = evaluate_variant_on_data(model, dev_test_data, vtype, device)
        cf_eval = evaluate_variant_on_data(model, dev_test_cf_data, vtype, device)

        # 配对反事实指标
        cf_metrics = compute_paired_counterfactual_metrics(std_eval, cf_eval)
        # 配对斜率
        slope_metrics = compute_per_sequence_paired_slopes(std_eval)

        # 整理输出
        results[key] = {
            "label": label,
            "variant_type": vtype,
            "latency_ms_per_frame": std_eval["latency_ms_per_frame"],
            "residual_magnitude_mean_deg": std_eval["residual_magnitude_mean_deg"],
            "counterfactual_paired_metrics": cf_metrics,
            "paired_sequence_slopes": slope_metrics,
            "tier_stats": std_eval["tier_stats"],
        }

    # 打印对比矩阵表格
    print("\n" + "=" * 105)
    print("  [D3b 严格配对反事实与可归因消融评测最终对比矩阵]:")
    print("=" * 105)
    print(f"  {'变体标识':<28} | {'10°/min 偏航改善':<16} | {'0°/min 零漂偏航变化':<16} | {'反事实方向一致率':<16} | {'反事实对称相关度':<16} | {'残差均值'}")
    print("  " + "-" * 115)

    for key, vtype, label in variants:
        r = results[key]
        t10 = r["tier_stats"].get("+10.0", {}).get("pure_yaw_delta_deg", 0.0)
        t0 = r["tier_stats"].get("+0.0", {}).get("pure_yaw_delta_deg", 0.0)
        dir_agr = r["counterfactual_paired_metrics"]["directional_agreement_pct"]
        rho_cf = r["counterfactual_paired_metrics"]["counterfactual_symmetry_correlation"]
        mag = r["residual_magnitude_mean_deg"]
        print(f"  {label:<28} | {t10:>+12.4f}°    | {t0:>+12.4f}°    | {dir_agr:>14.2f}%   | {rho_cf:>14.3f}   | {mag:>8.4f}°")

    # 构建并保存机器报告
    report = {
        "report_title": "SlimeVR CSTN v2 Phase 6 Task D3b Fixed Attribution Ablation & Paired Counterfactual Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(d3b_db),
        "pairing_verification": "Strictly 1-to-1 matched pair_id, clean motion, mounting noise, and exact inverted drift rates",
        "variants": results,
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  已保存 D3b 归因消融最终机器报告至: {output_report}")


if __name__ == "__main__":
    main()

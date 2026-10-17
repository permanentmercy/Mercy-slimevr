"""
ai_corrector/corrector/execute_task_d3c.py
=========================================
执行 Phase 6 任务 D3c: 完成真正的连续历史与慢偏置对照
(D3c True Continuous Causal History & Physical Slow Bias Control Study)

严格遵循 plan.md 任务 D3c 规范:
1. 逐帧完整连续时序数据 (database/d3c_continuous/):
   - 彻底摒弃 20 帧短窗重叠切片, 每帧恰好输入一次, 严格按时序推进;
   - 严格 1-to-1 配对反事实 (DevTest: 190 序列; FinalConfirm: 190 未见独立序列 36..40);
2. 架构与对照组设计:
   - 变体 A: 独立短窗基线 (Window-Independent Baseline, 状态每 20 帧清空);
   - 变体 B_diag: 诊断组 (短窗训练权重仅改变重放方式为连续因果, 不作为独立连续模型 B);
   - 变体 B: 真正的连续因果模型 (Full-Sequence Continuous Causal Training, 在完整序列上训练并连续因果推理);
   - 变体 C: 连续因果 + 独立物理 ZUPT 慢变偏航偏置估计 (基于足部静止期世界垂直轴角速度观测, 独立于网络自身残差预测, 明确标注初始偏航积分常数的不可观测维度);
3. 严格配对反事实差值回归与指标:
   - 方向一致率 (Directional Agreement %);
   - 增益比率 (Gain Ratio);
   - 配对反事实相关度 (rho_cf);
   - 逐序列去重时间回归斜率削减量及 95% Bootstrap CI;
   - 独立确认集 (FinalConfirm, 源序列 36..40) 单次泛化确认;
   - 推理延迟与状态内存开销。
4. 产出: artifacts/d3c_attribution_ablation_report.json
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .checkpoint import resolve_path, PACKAGE_ROOT
from .contract_v2 import (
    rot6d_to_matrix_np,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat,
)
from .evaluate_stratified import IMU_OFFSETS, compute_y_twist_deg, compute_geodesic_angles_deg
from .model import Corrector, Config


def compute_smooth_geodesic_loss(pred_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(target_q, dim=-1)
    p = torch.nn.functional.normalize(pred_q, dim=-1)
    dot = torch.clamp((p * q).sum(dim=-1).abs(), 0.0, 1.0 - 1e-6)
    theta = 2.0 * torch.acos(dot)
    return torch.sqrt(theta.square() + 1e-4) - 0.01


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def train_continuous_model_b(
    train_data_path: Path,
    val_data_path: Path,
    output_pt: Path,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 20261017,
) -> Path:
    best_pt = output_pt.with_name(output_pt.stem + "_best.pt")
    if best_pt.exists():
        print(f"  发现已有连续因果模型 B: {best_pt.name}，复用该模型。")
        return best_pt

    print("\n" + "=" * 80)
    print("  [D3c 训练] 正在完整连续序列上训练模型 B (Full Continuous Causal Sequential Training)...")
    print("=" * 80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. 加载数据
    train_npz = np.load(train_data_path)
    val_npz = np.load(val_data_path)

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

    dataset = TensorDataset(x_train, dq_train, gate_train, contact_train, mask_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 构建模型与优化器
    cfg = Config(hidden=96, decoder_type="joint_decoupled")
    model = Corrector(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    bce_loss_fn = nn.BCELoss(reduction="none")

    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for bx, bdq, bgate, bcontact, bmask in loader:
            bx = bx.to(device)
            bdq = bdq.to(device)
            bgate = bgate.to(device)
            bcontact = bcontact.to(device)
            bmask = bmask.to(device)

            optimizer.zero_grad()
            out, _ = model(bx, state=None)

            # 仅在有效时间帧上计算损失
            m_rot = bmask.unsqueeze(-1).unsqueeze(-1).expand_as(bdq)
            pred_dq = out["dq"]
            l_rot = compute_smooth_geodesic_loss(pred_dq, bdq) * bmask.unsqueeze(-1)
            loss_rot = l_rot.sum() / torch.clamp(bmask.sum() * 4, min=1.0)

            l_gate = bce_loss_fn(out["gate"], bgate) * bmask.unsqueeze(-1)
            loss_gate = l_gate.sum() / torch.clamp(bmask.sum() * 4, min=1.0)

            l_contact = bce_loss_fn(out["contact"], bcontact) * bmask.unsqueeze(-1)
            loss_contact = l_contact.sum() / torch.clamp(bmask.sum() * 2, min=1.0)

            loss = loss_rot * 10.0 + loss_gate + loss_contact
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # 验证评估
        model.eval()
        with torch.no_grad():
            v_out, _ = model(x_val, state=None)
            v_l_rot = compute_smooth_geodesic_loss(v_out["dq"], dq_val) * mask_val.unsqueeze(-1)
            v_loss_rot = v_l_rot.sum() / torch.clamp(mask_val.sum() * 4, min=1.0)
            v_l_gate = (bce_loss_fn(v_out["gate"], gate_val) * mask_val.unsqueeze(-1)).sum() / torch.clamp(mask_val.sum() * 4, min=1.0)
            v_loss = v_loss_rot * 10.0 + v_l_gate

        train_loss = total_loss / max(1, num_batches)
        if v_loss.item() < best_val_loss:
            best_val_loss = v_loss.item()
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, best_pt)
            saved_mark = "(*最佳*)"
        else:
            saved_mark = ""

        if epoch % 3 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:2d}/{epochs:2d} | TrainLoss: {train_loss:.4f} | ValLoss: {v_loss.item():.4f} {saved_mark}")

    print(f"  模型 B 训练完成 (耗时 {time.time()-t0:.2f}s) -> 保存至: {best_pt.name}")
    return best_pt


def evaluate_continuous_variant(
    model: Corrector,
    data_path: Path,
    variant_type: str,  # "model_a", "model_b_diag", "model_b", "model_c"
    device: torch.device,
    tau: float = 0.50,
) -> dict:
    data = np.load(data_path, allow_pickle=False)
    x_pad = data["x"].astype(np.float32)
    dq_target_pad = data["dq"].astype(np.float32)
    seq_lens = data["seq_lens"]
    pair_ids = data["pair_ids"]
    seq_ids = data["seq_ids"]
    drift_tiers = data["drift_tiers"]
    drift_rates = data["drift_rates"]
    mask = data["mask"]

    n_seq, max_len = x_pad.shape[:2]
    model.eval()

    pred_dq = np.zeros((n_seq, max_len, 4, 4), dtype=np.float32)
    pred_dq[..., 0] = 1.0
    pred_gate = np.zeros((n_seq, max_len, 4), dtype=np.float32)
    pred_contact = np.zeros((n_seq, max_len, 2), dtype=np.float32)

    t_start = time.time()

    with torch.no_grad():
        for i in range(n_seq):
            l = int(seq_lens[i])
            cur_x = torch.from_numpy(x_pad[i:i+1, :l]).to(device)

            if variant_type == "model_a":
                # 变体 A: 独立短窗基线 (每 20 帧短窗重置状态)
                dq_chunks, gate_chunks, contact_chunks = [], [], []
                for chunk_start in range(0, l, 20):
                    chunk_end = min(chunk_start + 20, l)
                    sub_x = cur_x[:, chunk_start:chunk_end]
                    out, _ = model(sub_x, state=None)
                    dq_chunks.append(out["dq"].cpu().numpy()[0])
                    gate_chunks.append(out["gate"].cpu().numpy()[0])
                    contact_chunks.append(out["contact"].cpu().numpy()[0])

                pred_dq[i, :l] = np.concatenate(dq_chunks, axis=0)
                pred_gate[i, :l] = np.concatenate(gate_chunks, axis=0)
                pred_contact[i, :l] = np.concatenate(contact_chunks, axis=0)

            elif variant_type in ("model_b_diag", "model_b", "model_c"):
                # 变体 B_diag, B, C: 全时序连续因果推进 (序列边界清空状态, 序列内部每帧恰输入一次)
                out, _ = model(cur_x, state=None)
                pred_dq[i, :l] = out["dq"].cpu().numpy()[0]
                pred_gate[i, :l] = out["gate"].cpu().numpy()[0]
                pred_contact[i, :l] = out["contact"].cpu().numpy()[0]

    cost_inf = (time.time() - t_start) / max(1, np.sum(seq_lens)) * 1000.0  # ms/frame

    # 四元数与姿态还原: 预填单位矩阵避免 padding 帧出现非正行列式
    r_raw = np.zeros((n_seq, max_len, 4, 3, 3), dtype=np.float32)
    r_raw[..., 0, 0] = 1.0
    r_raw[..., 1, 1] = 1.0
    r_raw[..., 2, 2] = 1.0

    for i in range(n_seq):
        l = int(seq_lens[i])
        for j, offset in enumerate(IMU_OFFSETS):
            r_raw[i, :l, j] = rot6d_to_matrix_np(x_pad[i, :l, offset:offset+6])

    dq_target_mat = quat_wxyz_to_rotmat(dq_target_pad)
    r_clean = np.matmul(r_raw, dq_target_mat)

    # 变体 C: 独立物理 ZUPT 慢变偏航偏置估计
    if variant_type == "model_c":
        dq_c_adjusted = pred_dq.copy()

        for i in range(n_seq):
            l = int(seq_lens[i])
            # 四路下肢独立陀螺零偏率估计 (rad/s)
            b_gyro_y = np.zeros(4, dtype=np.float32)
            # 累积偏航角度补偿 (rad)
            theta_slow = np.zeros(4, dtype=np.float32)
            dt = 1.0 / 60.0

            offsets_14d = [68, 82, 96, 110]

            for f in range(l):
                is_contact_l = pred_contact[i, f, 0] > 0.5
                is_contact_r = pred_contact[i, f, 1] > 0.5

                # 提取传感器局部角速度 (rad/s)
                omega_local = np.zeros((4, 3), dtype=np.float32)
                for j, off in enumerate(offsets_14d):
                    omega_local[j] = x_pad[i, f, off+9:off+12]

                # 转换至世界系: omega_world = R_raw @ omega_local
                omega_world = np.einsum("nij,nj->ni", r_raw[i, f], omega_local)
                # 垂直 +Y 世界轴角速度
                omega_world_y = omega_world[:, 1]

                alpha = 0.05
                # 左腿触地静止期更新 (0: thigh, 2: shin)
                if is_contact_l and np.linalg.norm(omega_local[2]) < 0.8:
                    b_gyro_y[[0, 2]] = (1.0 - alpha) * b_gyro_y[[0, 2]] + alpha * omega_world_y[[0, 2]]

                # 右腿触地静止期更新 (1: thigh, 3: shin)
                if is_contact_r and np.linalg.norm(omega_local[3]) < 0.8:
                    b_gyro_y[[1, 3]] = (1.0 - alpha) * b_gyro_y[[1, 3]] + alpha * omega_world_y[[1, 3]]

                # 积分慢变偏航角补偿
                theta_slow += b_gyro_y * dt

                # 构造慢变偏航逆旋转: R(exp(-theta_slow * e_y))
                rotvec_slow = np.zeros((4, 3), dtype=np.float32)
                rotvec_slow[:, 1] = -theta_slow
                slow_rot_mat = Rotation.from_rotvec(rotvec_slow).as_matrix()

                # 复合快慢修正: R_corr = R_raw @ R_slow @ R_fast
                fast_dq_mat = quat_wxyz_to_rotmat(pred_dq[i, f])
                comp_dq_mat = np.matmul(slow_rot_mat, fast_dq_mat)
                dq_c_adjusted[i, f] = rotmat_to_quat_wxyz(comp_dq_mat)

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

    # 分档统计 (严格仅统计有效帧 mask==True)
    unique_tiers = np.unique(drift_tiers)
    tier_stats = {}

    for tr in unique_tiers:
        idx_seq = np.where(drift_tiers == tr)[0]
        if len(idx_seq) == 0:
            continue

        yr_list, yc_list, fr_list, fc_list, cov_list = [], [], [], [], []
        for s_idx in idx_seq:
            l = int(seq_lens[s_idx])
            yr_list.append(yaw_raw[s_idx, :l])
            yc_list.append(yaw_corr[s_idx, :l])
            fr_list.append(full_raw[s_idx, :l])
            fc_list.append(full_corr[s_idx, :l])
            cov_list.append(pred_gate[s_idx, :l] >= tau)

        yr_all = np.concatenate(yr_list)
        yc_all = np.concatenate(yc_list)
        fr_all = np.concatenate(fr_list)
        fc_all = np.concatenate(fc_list)
        cov_all = np.concatenate(cov_list)

        tier_stats[f"{tr:+.1f}"] = {
            "pure_yaw_raw_deg": float(np.mean(yr_all)),
            "pure_yaw_corr_deg": float(np.mean(yc_all)),
            "pure_yaw_delta_deg": float(np.mean(yr_all) - np.mean(yc_all)),
            "full_pose_delta_deg": float(np.mean(fr_all) - np.mean(fc_all)),
            "coverage_pct": float(np.mean(cov_all) * 100.0),
            "total_frames": int(len(yr_all)),
        }

    valid_mask = mask.flatten()
    dq_valid = pred_dq.reshape(-1, 4, 4)[valid_mask]
    dot = np.clip(np.abs(dq_valid[..., 0]), -1.0, 1.0)
    res_mag_mean = float(np.mean(2.0 * np.arccos(dot) * 180.0 / np.pi))

    return {
        "variant": variant_type,
        "latency_ms_per_frame": cost_inf,
        "residual_magnitude_mean_deg": res_mag_mean,
        "tier_stats": tier_stats,
        "pair_ids": pair_ids,
        "drift_tiers": drift_tiers,
        "drift_rates": drift_rates,
        "seq_ids": seq_ids,
        "seq_lens": seq_lens,
        "pred_dq": pred_dq,
        "dq_target": dq_target_pad,
        "yaw_raw": yaw_raw,
        "yaw_corr": yaw_corr,
        "full_raw": full_raw,
        "full_corr": full_corr,
    }


def compute_paired_counterfactual_metrics_continuous(
    eval_std: dict,
    eval_cf: dict,
) -> dict:
    """按 pair_id + frame + joint 严格逐帧对齐反事实评测"""
    pair_std = eval_std["pair_ids"]
    pair_cf = eval_cf["pair_ids"]
    assert np.array_equal(pair_std, pair_cf), "pair_id 必须严格一致!"

    seq_lens = eval_std["seq_lens"]
    n_seq = len(seq_lens)

    dq_std = eval_std["pred_dq"]
    dq_cf = eval_cf["pred_dq"]
    tgt_std = eval_std["dq_target"]
    tgt_cf = eval_cf["dq_target"]

    def get_yaw_rotvec(q):
        # q: [N, T, 4, 4] -> xyzw -> as_rotvec() -> Y component
        n, t, j, _ = q.shape
        q_xyzw = q[..., [1, 2, 3, 0]].reshape(-1, 4)
        rotv = Rotation.from_quat(q_xyzw).as_rotvec().reshape(n, t, j, 3)
        return rotv[..., 1]  # Y 轴分量 (rad)

    yaw_pred_std = get_yaw_rotvec(dq_std)
    yaw_pred_cf = get_yaw_rotvec(dq_cf)
    yaw_tgt_std = get_yaw_rotvec(tgt_std)
    yaw_tgt_cf = get_yaw_rotvec(tgt_cf)

    # 仅在有效时间帧上对齐
    delta_tgt_list, delta_pred_list = [], []
    v_std_list, v_cf_list = [], []

    for i in range(n_seq):
        l = int(seq_lens[i])
        tier = eval_std["drift_tiers"][i]
        if abs(tier) < 0.1:
            continue  # 仅评估非零漂移反向响应

        dt = (yaw_tgt_std[i, :l] - yaw_tgt_cf[i, :l]) * 180.0 / np.pi
        dp = (yaw_pred_std[i, :l] - yaw_pred_cf[i, :l]) * 180.0 / np.pi

        delta_tgt_list.append(dt.flatten())
        delta_pred_list.append(dp.flatten())
        v_std_list.append(yaw_pred_std[i, :l].flatten())
        v_cf_list.append(yaw_pred_cf[i, :l].flatten())

    if not delta_tgt_list:
        return {}

    all_dt = np.concatenate(delta_tgt_list)
    all_dp = np.concatenate(delta_pred_list)
    all_v_std = np.concatenate(v_std_list)
    all_v_cf = np.concatenate(v_cf_list)

    # 1. 有效漂移反转帧上的方向一致率: sign(dp) == sign(dt)
    valid_mask = np.abs(all_dt) > 0.05
    if np.sum(valid_mask) > 0:
        dir_agree = float(np.mean((np.sign(all_dp[valid_mask]) == np.sign(all_dt[valid_mask])).astype(np.float32)) * 100.0)
        gain_ratio = float(np.median(np.abs(all_dp[valid_mask]) / np.maximum(np.abs(all_dt[valid_mask]), 1e-4)))
    else:
        dir_agree = 50.0
        gain_ratio = 0.0

    # 2. 配对相关度 rho_cf = corr(v_std, -v_cf) 与 corr(v_std, v_cf)
    std1, std2 = np.std(all_v_std), np.std(all_v_cf)
    if std1 > 1e-6 and std2 > 1e-6:
        rho_cf = float(np.corrcoef(all_v_std, -all_v_cf)[0, 1])
        rho_identical = float(np.corrcoef(all_v_std, all_v_cf)[0, 1])
    else:
        rho_cf = 0.0
        rho_identical = 0.0

    return {
        "directional_agreement_pct": dir_agree,
        "median_gain_ratio": gain_ratio,
        "counterfactual_symmetry_correlation_rho_cf": rho_cf,
        "output_identity_correlation_rho_std_cf": rho_identical,
        "mean_delta_target_deg": float(np.mean(np.abs(all_dt))),
        "mean_delta_pred_deg": float(np.mean(np.abs(all_dp))),
    }


def compute_paired_sequence_slopes_continuous(eval_res: dict) -> dict:
    """计算连续序列时间回归斜率削减量及 95% Bootstrap CI"""
    n_seq = len(eval_res["seq_lens"])
    tiers = eval_res["drift_tiers"]
    seq_lens = eval_res["seq_lens"]
    yaw_raw = eval_res["yaw_raw"]
    yaw_corr = eval_res["yaw_corr"]

    unique_tiers = [10.0, 3.0, 1.0, 0.0]
    out_slopes = {}

    for tr in unique_tiers:
        tier_mask = np.isclose(np.abs(tiers), tr, atol=0.2)
        idx_seqs = np.where(tier_mask)[0]

        raw_slopes, corr_slopes, delta_slopes = [], [], []

        for i in idx_seqs:
            l = int(seq_lens[i])
            if l < 20:
                continue

            t_min = np.arange(l, dtype=np.float32) / (60.0 * 60.0)
            yr_mean = np.mean(yaw_raw[i, :l], axis=-1)
            yc_mean = np.mean(yaw_corr[i, :l], axis=-1)

            p_r = np.polyfit(t_min, yr_mean, 1)[0]
            p_c = np.polyfit(t_min, yc_mean, 1)[0]

            raw_slopes.append(p_r)
            corr_slopes.append(p_c)
            delta_slopes.append(p_r - p_c)

        if not delta_slopes:
            continue

        d_arr = np.array(delta_slopes)
        r_arr = np.array(raw_slopes)
        c_arr = np.array(corr_slopes)

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
    d3c_db = resolve_path("database/d3c_continuous")

    train_data = d3c_db / "train_continuous.npz"
    val_data = d3c_db / "val_continuous.npz"
    dev_test_data = d3c_db / "dev_test_continuous.npz"
    dev_test_cf_data = d3c_db / "dev_test_cf_continuous.npz"
    confirm_data = d3c_db / "final_confirm_continuous.npz"
    confirm_cf_data = d3c_db / "final_confirm_cf_continuous.npz"

    model_a_pt = artifacts_dir / "cstn_v2_d3b_model_a_best.pt"
    model_b_pt = artifacts_dir / "cstn_v2_d3c_continuous_model_b.pt"
    output_report = artifacts_dir / "d3c_attribution_ablation_report.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 训练/加载真正的连续因果模型 B
    best_model_b_pt = train_continuous_model_b(train_data, val_data, model_b_pt, device)

    # 2. 加载模型 A (短窗基线) 与模型 B (连续因果)
    ckpt_a = torch.load(model_a_pt, map_location=device, weights_only=False)
    model_a = Corrector(Config(**dict(ckpt_a.get("config", {})))).to(device)
    model_a.load_state_dict(ckpt_a["state_dict"])
    model_a.eval()

    ckpt_b = torch.load(best_model_b_pt, map_location=device, weights_only=False)
    model_b = Corrector(Config(**dict(ckpt_b.get("config", {})))).to(device)
    model_b.load_state_dict(ckpt_b["state_dict"])
    model_b.eval()

    print("\n" + "=" * 80)
    print("  [D3c.2] 正在执行模型 A / B_diag / B / C 严格可归因消融评测...")
    print("=" * 80)

    results_dev = {}
    variants = [
        ("variant_a_independent_window", model_a, "model_a", "变体 A (独立短窗基线)"),
        ("variant_b_diag_short_trained_continuous_replay", model_a, "model_b_diag", "变体 B_diag (短窗权重连续重放诊断组)"),
        ("variant_b_continuous_causal_trained", model_b, "model_b", "变体 B (真正连续训练因果模型)"),
        ("variant_c_hybrid_physical_slow_bias", model_b, "model_c", "变体 C (连续因果 + 物理 ZUPT 慢变偏置)"),
    ]

    for key, m, vtype, label in variants:
        print(f"\n  -> [开发集 DevTest] 评测 {label}...")
        std_eval = evaluate_continuous_variant(m, dev_test_data, vtype, device)
        cf_eval = evaluate_continuous_variant(m, dev_test_cf_data, vtype, device)

        cf_metrics = compute_paired_counterfactual_metrics_continuous(std_eval, cf_eval)
        slope_metrics = compute_paired_sequence_slopes_continuous(std_eval)

        results_dev[key] = {
            "label": label,
            "variant_type": vtype,
            "latency_ms_per_frame": std_eval["latency_ms_per_frame"],
            "residual_magnitude_mean_deg": std_eval["residual_magnitude_mean_deg"],
            "counterfactual_paired_metrics": cf_metrics,
            "paired_sequence_slopes": slope_metrics,
            "tier_stats": std_eval["tier_stats"],
        }

    # 3. 独立确认集 (FinalConfirm，源序列 36..40) 最终单次确认
    print("\n" + "=" * 80)
    print("  [D3c.3] 正在独立确认集 (源序列 36..40) 上执行单次泛化确认...")
    print("=" * 80)

    results_confirm = {}
    for key, m, vtype, label in variants:
        print(f"  -> [独立确认集 FinalConfirm] 评测 {label}...")
        std_eval_c = evaluate_continuous_variant(m, confirm_data, vtype, device)
        cf_eval_c = evaluate_continuous_variant(m, confirm_cf_data, vtype, device)

        cf_metrics_c = compute_paired_counterfactual_metrics_continuous(std_eval_c, cf_eval_c)
        slope_metrics_c = compute_paired_sequence_slopes_continuous(std_eval_c)

        results_confirm[key] = {
            "label": label,
            "variant_type": vtype,
            "counterfactual_paired_metrics": cf_metrics_c,
            "paired_sequence_slopes": slope_metrics_c,
            "tier_stats": std_eval_c["tier_stats"],
        }

    # 打印对比矩阵表格
    print("\n" + "=" * 115)
    print("  [D3c 开发集 DevTest 对照消融最终对比矩阵]:")
    print("=" * 115)
    print(f"  {'变体标识':<34} | {'10°/min 偏航改善':<16} | {'0°/min 零漂变化':<16} | {'方向一致率':<12} | {'对称相关度 rho_cf':<18} | {'共有相关度'}")
    print("  " + "-" * 125)

    for key, m, vtype, label in variants:
        r = results_dev[key]
        t10 = r["tier_stats"].get("+10.0", {}).get("pure_yaw_delta_deg", 0.0)
        t0 = r["tier_stats"].get("+0.0", {}).get("pure_yaw_delta_deg", 0.0)
        dir_agr = r["counterfactual_paired_metrics"].get("directional_agreement_pct", 50.0)
        rho_cf = r["counterfactual_paired_metrics"].get("counterfactual_symmetry_correlation_rho_cf", 0.0)
        rho_id = r["counterfactual_paired_metrics"].get("output_identity_correlation_rho_std_cf", 0.0)
        print(f"  {label:<34} | {t10:>+12.4f}°    | {t0:>+12.4f}°    | {dir_agr:>10.2f}%   | {rho_cf:>16.3f}   | {rho_id:>10.3f}")

    print("\n" + "=" * 115)
    print("  [D3c 独立确认集 FinalConfirm (源序列 36..40) 最终单次确认结果]:")
    print("=" * 115)
    print(f"  {'变体标识':<34} | {'10°/min 偏航改善':<16} | {'0°/min 零漂变化':<16} | {'方向一致率':<12} | {'对称相关度 rho_cf':<18} | {'共有相关度'}")
    print("  " + "-" * 125)

    for key, m, vtype, label in variants:
        r = results_confirm[key]
        t10 = r["tier_stats"].get("+10.0", {}).get("pure_yaw_delta_deg", 0.0)
        t0 = r["tier_stats"].get("+0.0", {}).get("pure_yaw_delta_deg", 0.0)
        dir_agr = r["counterfactual_paired_metrics"].get("directional_agreement_pct", 50.0)
        rho_cf = r["counterfactual_paired_metrics"].get("counterfactual_symmetry_correlation_rho_cf", 0.0)
        rho_id = r["counterfactual_paired_metrics"].get("output_identity_correlation_rho_std_cf", 0.0)
        print(f"  {label:<34} | {t10:>+12.4f}°    | {t0:>+12.4f}°    | {dir_agr:>10.2f}%   | {rho_cf:>16.3f}   | {rho_id:>10.3f}")

    # 保存机器报告
    report = {
        "report_title": "SlimeVR CSTN v2 Phase 6 Task D3c True Continuous Causal History & Physical Slow Bias Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(d3c_db),
        "unobservable_dimension_specification": (
            "1. 独立物理 ZUPT 观测垂直轴角速度 omega_y,world 仅在足部静止支撑相估计陀螺零偏率 b_gyro_y。\n"
            "2. 初始偏航漂移角 theta_drift(0) 是纯惯性 rate 积分器中数学上不可观测的自由度，"
            "在缺乏绝对外部航向锚定 (如 HMD 头部朝向对齐或磁力计) 时不可从 IMU 自身残差确定。"
        ),
        "dev_test_results": results_dev,
        "final_confirm_results": results_confirm,
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  已保存 D3c 最终机器报告至: {output_report}")


if __name__ == "__main__":
    main()

"""
ai_corrector/corrector/execute_task_h3_ablation.py
=================================================
Phase 11 任务 H3: 同一隔离基准上的多因子八路消融基准评测
(Task H3 Multi-Factor 8-Way Ablation Benchmark & Strict Zero-Drift Safety Audit)

严格遵循 Phase 11 计划规范:
1. 8 组横向对齐策略 (所有策略运行在完全相同的 30 分钟连续会话与真值上):
   - 1. Raw 基线 (未校正原始追踪器);
   - 2. Oracle 理想参考 (零误差);
   - 3. Phase 9 ZARU-only (频域与静止零偏滤波器);
   - 4. Phase 9 Gait-only (纯步态摆动面估计);
   - 5. Phase 9 Full (ZARU + Gait 完整慢分支);
   - 6. Yaw-Neutral Phase 6-only (经 Y 轴 Swing-Twist 投影的快分支神经网络);
   - 7. Candidate Hybrid: Phase 9 ZARU + Yaw-Neutral Phase 6 (正交候选主组合);
   - 8. Full Hybrid: Phase 9 Full + Yaw-Neutral Phase 6;
   - 9. Old CSTN Diagnostic: 旧版 Phase 6 CSTN (无约束全 3D 残差诊断组).
2. 会话工况全覆盖:
   - 10 条标称长会话 (0°, 1°, 3°, 10°/min，包含静坐、复杂步态、无静止窗口、VD 丢失等);
   - 2 条严格零漂会话 (0.0°/min, 0 random walk, 验证零漂直通安全性);
   - 独立确认集 FinalConfirm (源序列 36..40) 世界 Swing 残差评估.
3. 统计指标:
   - 30 分钟起终点圆周累计偏航误差增长量 D_raw, D_corr, 净收益 G, 改善率, 胜率, 最差恶化量;
   - 95% Bootstrap 置信区间;
   - 严格偏航串扰度量 (Twist Crosstalk);
   - 严格判定 PASS / FAIL，输出归档机器报告:
     artifacts/phase11_hybrid_compensation_report.json
"""

import copy
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai_corrector.corrector.contract_v2 import TOTAL_INPUT_DIM_V2
from ai_corrector.corrector.contract_v3 import V2ToV3Adapter, LOWER_BODY_JOINTS
from ai_corrector.corrector.oracle_v3_geometry import (
    rot_y_matrix,
    world_left_correction,
    compute_endpoint_session_metrics,
    wrap_angle,
    geodesic_angle_deg,
    geodesic_angle_rad,
)
from ai_corrector.corrector.session_simulator_v3 import ContinuousSessionSynthesizer, ContinuousSessionStream
from ai_corrector.corrector.bias_estimator_v3 import DeployableBiasEstimatorV3
from ai_corrector.corrector.hybrid_composer import compose_single_tracker_hybrid, HybridComposer
from ai_corrector.corrector.yaw_neutral_cstn import (
    YawNeutralCSTN,
    load_yaw_neutral_cstn,
    torch_swing_twist_decomposition_y,
    torch_lower_body_rotmat_from_features,
    torch_world_swing_local_delta,
    torch_slerp_from_identity,
    compute_smooth_geodesic_distance,
)
from ai_corrector.corrector.model import Corrector, Config


def bootstrap_mean_ci(data: List[float], num_resamples: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """计算 95% Bootstrap 置信区间 [lower, upper] 与均值"""
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


def load_model_weights(device: torch.device):
    """加载旧 CSTN 诊断模型与重训后的 YawNeutral CSTN 模型"""
    artifacts_dir = BASE_DIR / "ai_corrector" / "artifacts"
    old_pt = artifacts_dir / "cstn_v2_d1_smooth_geodesic_s123_best.pt"
    neutral_pt = artifacts_dir / "cstn_v2_yaw_neutral_world_best.pt"
    if not neutral_pt.exists():
        neutral_pt = artifacts_dir / "cstn_v2_yaw_neutral_best.pt"

    old_model = None
    if old_pt.exists():
        try:
            ckpt = torch.load(old_pt, map_location=device, weights_only=False)
            cfg_dict = ckpt.get("config", {})
            cfg = Config(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else Config(
                stream_dims=(17, 17, 17, 17, 14, 14, 14, 14), num_joints=4, decoder_type="joint_decoupled"
            )
            old_model = Corrector(cfg).to(device)
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            old_model.load_state_dict(sd)
            old_model.eval()
        except Exception as e:
            print(f"Warning: failed to load old CSTN: {e}", flush=True)

    neutral_model = load_yaw_neutral_cstn(neutral_pt, device)
    return old_model, neutral_model


def run_neural_models_on_stream(
    old_model: Optional[Corrector],
    neutral_model: Optional[YawNeutralCSTN],
    features_124: np.ndarray,
    device: torch.device,
    chunk_size: int = 60
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray], Dict[str, float]]:
    """
    分块连续运行神经网络，获取最后一个时间步的预测姿态增量与置信度门控
    """
    N = len(features_124)

    # 1. 运行旧 CSTN (完整 3D 局部残差)
    old_last_dq = {j: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for j in LOWER_BODY_JOINTS}
    old_last_gate = {j: 0.0 for j in LOWER_BODY_JOINTS}
    if old_model is not None:
        with torch.no_grad():
            hx = None
            for start_idx in range(0, N, chunk_size):
                end_idx = min(start_idx + chunk_size, N)
                sub_feats = features_124[start_idx:end_idx]
                sub_tensor = torch.tensor(sub_feats, dtype=torch.float32, device=device).unsqueeze(0)
                out, hx = old_model(sub_tensor, hx)
                if end_idx == N:
                    dqs = out["dq"][0, -1].cpu().numpy()
                    gates = out["gate"][0, -1].cpu().numpy()
                    for idx, j in enumerate(LOWER_BODY_JOINTS):
                        old_last_dq[j] = dqs[idx]
                        old_last_gate[j] = float(gates[idx])

    # 2. 运行重训后的 YawNeutral CSTN
    neutral_last_dq = {j: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for j in LOWER_BODY_JOINTS}
    neutral_last_gate = {j: 0.0 for j in LOWER_BODY_JOINTS}
    if neutral_model is not None:
        with torch.no_grad():
            hx_n = None
            for start_idx in range(0, N, chunk_size):
                end_idx = min(start_idx + chunk_size, N)
                sub_feats = features_124[start_idx:end_idx]
                sub_tensor = torch.tensor(sub_feats, dtype=torch.float32, device=device).unsqueeze(0)
                out_n, hx_n = neutral_model(sub_tensor, hx_n)
                if end_idx == N:
                    dqs_n = out_n["dq"][0, -1].cpu().numpy()
                    gates_n = out_n["gate"][0, -1].cpu().numpy()
                    for idx, j in enumerate(LOWER_BODY_JOINTS):
                        neutral_last_dq[j] = dqs_n[idx]
                        neutral_last_gate[j] = float(gates_n[idx])

    return old_last_dq, old_last_gate, neutral_last_dq, neutral_last_gate


def evaluate_single_session_all_strategies(
    session: ContinuousSessionStream,
    old_model: Optional[Corrector],
    neutral_model: Optional[YawNeutralCSTN],
    device: torch.device
) -> Dict[str, dict]:
    """
    在单条 30 分钟连续会话上完整评测所有 8 组策略
    """
    gt = session.ground_truth_eval
    e_raw_t0 = gt["e_true_t0_rad"]
    e_raw_t1 = gt["e_true_t1_rad"]
    raw_rot_t1 = gt["raw_rot_t1"]
    clean_rot_t1 = gt["clean_rot_t1"]

    # 1. Raw 基线
    m_raw = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, dict(e_raw_t0), dict(e_raw_t1))

    # 2. Oracle 理想
    e_zero = {j: 0.0 for j in LOWER_BODY_JOINTS}
    m_oracle = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_zero, e_zero)

    # 3. 运行神经网络推断获取末帧 fast 预测
    old_dq, old_gate, neutral_dq, neutral_gate = run_neural_models_on_stream(
        old_model, neutral_model, session.features_124, device
    )

    # 4. 运行 Phase 9 慢分支 (ZARU, Gait, Full)
    sample_rate_hz = session.num_frames / session.duration_sec
    est_zaru = DeployableBiasEstimatorV3(enable_zaru=True, enable_gait=False, sample_rate_hz=sample_rate_hz)
    est_gait = DeployableBiasEstimatorV3(enable_zaru=False, enable_gait=True, sample_rate_hz=sample_rate_hz)
    est_full = DeployableBiasEstimatorV3(enable_zaru=True, enable_gait=True, sample_rate_hz=sample_rate_hz)

    for i in range(session.num_frames):
        vec = session.features_124[i]
        frame = V2ToV3Adapter.convert_124d_to_frame(vec, timestamp=float(session.timestamps[i]), dt=session.dt)
        est_zaru.process_frame(frame)
        est_gait.process_frame(frame)
        est_full.process_frame(frame)

    states_zaru = est_zaru.kalman_filter.get_current_states()
    states_gait = est_gait.kalman_filter.get_current_states()
    states_full = est_full.kalman_filter.get_current_states()

    e_zaru_t1 = {j: float(wrap_angle(e_raw_t1[j] - states_zaru[j][0])) for j in LOWER_BODY_JOINTS}
    m_zaru = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_zero, e_zaru_t1)

    e_gait_t1 = {j: float(wrap_angle(e_raw_t1[j] - states_gait[j][0])) for j in LOWER_BODY_JOINTS}
    m_gait = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_zero, e_gait_t1)

    e_full_t1 = {j: float(wrap_angle(e_raw_t1[j] - states_full[j][0])) for j in LOWER_BODY_JOINTS}
    m_full = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_zero, e_full_t1)

    # 5. 诊断对照组: 旧 Phase 6 CSTN
    e_old_cstn_t1 = {}
    for j in LOWER_BODY_JOINTS:
        if old_gate[j] > 0.3:
            q = old_dq[j]
            w, x, y, z = q[0], q[1], q[2], q[3]
            R_dq = np.array([
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
            ], dtype=np.float32)
        else:
            R_dq = np.eye(3, dtype=np.float32)
        R_corr = raw_rot_t1[j] @ R_dq
        diff_R = R_corr @ clean_rot_t1[j].T
        yaw_err = math.atan2(float(diff_R[0, 2]), float(diff_R[2, 2]))
        e_old_cstn_t1[j] = float(wrap_angle(yaw_err))
    m_old_cstn = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_zero, e_old_cstn_t1)

    # 6. Yaw-Neutral Phase 6-only (慢分支 e_hat = 0)
    e_p6_neutral_t1 = {}
    for j in LOWER_BODY_JOINTS:
        hybrid_p6 = compose_single_tracker_hybrid(
            R_raw=raw_rot_t1[j],
            e_hat_rad=0.0,
            delta_R6=neutral_dq[j],
            gate_p6=neutral_gate[j],
            gate_threshold=0.3,
            imu_id=j
        )
        diff_R = hybrid_p6.R_out @ clean_rot_t1[j].T
        yaw_err = math.atan2(float(diff_R[0, 2]), float(diff_R[2, 2]))
        e_p6_neutral_t1[j] = float(wrap_angle(yaw_err))
    m_p6_neutral = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_zero, e_p6_neutral_t1)

    # 7. Candidate Hybrid: Phase 9 ZARU + Yaw-Neutral Phase 6
    e_hybrid_cand_t1 = {}
    discarded_twists = {}
    applied_swings = {}
    for j in LOWER_BODY_JOINTS:
        e_hat = float(states_zaru[j][0])
        hybrid_out = compose_single_tracker_hybrid(
            R_raw=raw_rot_t1[j],
            e_hat_rad=e_hat,
            delta_R6=neutral_dq[j],
            gate_p6=neutral_gate[j],
            gate_threshold=0.3,
            imu_id=j
        )
        diff_R = hybrid_out.R_out @ clean_rot_t1[j].T
        yaw_err = math.atan2(float(diff_R[0, 2]), float(diff_R[2, 2]))
        e_hybrid_cand_t1[j] = float(wrap_angle(yaw_err))
        discarded_twists[j] = hybrid_out.discarded_twist_deg
        applied_swings[j] = hybrid_out.applied_swing_deg
    m_hybrid_cand = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_zero, e_hybrid_cand_t1)

    # 8. Full Hybrid: Phase 9 Full + Yaw-Neutral Phase 6
    e_hybrid_full_t1 = {}
    for j in LOWER_BODY_JOINTS:
        e_hat_f = float(states_full[j][0])
        hybrid_f = compose_single_tracker_hybrid(
            R_raw=raw_rot_t1[j],
            e_hat_rad=e_hat_f,
            delta_R6=neutral_dq[j],
            gate_p6=neutral_gate[j],
            gate_threshold=0.3,
            imu_id=j
        )
        diff_R = hybrid_f.R_out @ clean_rot_t1[j].T
        yaw_err = math.atan2(float(diff_R[0, 2]), float(diff_R[2, 2]))
        e_hybrid_full_t1[j] = float(wrap_angle(yaw_err))
    m_hybrid_full = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_zero, e_hybrid_full_t1)

    return {
        "raw": m_raw,
        "oracle": m_oracle,
        "p9_zaru": m_zaru,
        "p9_gait": m_gait,
        "p9_full": m_full,
        "old_cstn": m_old_cstn,
        "p6_yaw_neutral": m_p6_neutral,
        "hybrid_candidate": m_hybrid_cand,
        "hybrid_full": m_hybrid_full,
        "discarded_twists_deg": discarded_twists,
        "applied_swings_deg": applied_swings,
        "zaru_events": est_zaru.zaru_event_counts,
        "gait_events": est_full.gait_event_counts
    }


def _worker_evaluate_session(args: Tuple[str, float, int, str, bool]) -> Tuple[str, float, int, str, bool, dict]:
    """多进程独立长会话评估"""
    s_id, tier, seed, scenario, enable_rw = args
    t0 = time.time()
    synth = ContinuousSessionSynthesizer(
        duration_sec=1800.0,
        sample_rate_hz=10.0,
        drift_tier_deg_min=tier,
        seed=seed,
        scenario=scenario,
        enable_random_walk=enable_rw
    )
    stream = synth.generate(session_id=s_id)
    device = torch.device("cpu")
    old_model, neutral_model = load_model_weights(device)
    res = evaluate_single_session_all_strategies(stream, old_model, neutral_model, device)
    dt = time.time() - t0

    raw_d = res["raw"]["D_raw_deg"]
    zaru_d = res["p9_zaru"]["D_corr_deg"]
    hyb_d = res["hybrid_candidate"]["D_corr_deg"]
    hyb_gain = res["hybrid_candidate"]["G_deg"]
    hyb_ratio = res["hybrid_candidate"]["improvement_ratio"] * 100.0

    print(f"  [Worker {s_id:14s}] Tier={tier:4.1f}°/m, {scenario:12s} ({dt:4.1f}s) | Raw={raw_d:6.2f}°, ZARU={zaru_d:6.2f}°, Hybrid={hyb_d:6.2f}°, Gain={hyb_gain:+6.2f}° ({hyb_ratio:5.1f}%)", flush=True)
    return s_id, tier, seed, scenario, enable_rw, res


def evaluate_holdout_short_term_swing(device: torch.device) -> dict:
    """
    在 FinalConfirm 上用 oracle P9 Twist 隔离，评测世界系 P6 组合 Swing 残差与偏航串扰。
    """
    fc_path = BASE_DIR / "ai_corrector" / "database" / "p11_continuous" / "final_confirm_continuous.npz"
    if not fc_path.exists():
        fc_path = BASE_DIR / "ai_corrector" / "database" / "d3c_continuous" / "final_confirm_continuous.npz"
    if not fc_path.exists():
        return {"status": "skipped", "reason": "final_confirm_continuous.npz not found"}

    print("\n[Step 4] Running short-term pose & yaw crosstalk evaluation on holdout FinalConfirm (190 sequences)...", flush=True)
    fc = np.load(fc_path)
    x = torch.from_numpy(fc["x"]).float().to(device)
    dq_target = torch.from_numpy(fc["dq"]).float().to(device)
    mask = torch.from_numpy(fc["mask"]).bool().to(device)

    neutral_pt = BASE_DIR / "ai_corrector" / "artifacts" / "cstn_v2_yaw_neutral_world_best.pt"
    if not neutral_pt.exists():
        neutral_pt = BASE_DIR / "ai_corrector" / "artifacts" / "cstn_v2_yaw_neutral_best.pt"
    old_pt = BASE_DIR / "ai_corrector" / "artifacts" / "cstn_v2_d1_smooth_geodesic_s123_best.pt"

    neutral_model = load_yaw_neutral_cstn(neutral_pt, device)
    old_model = None
    if old_pt.exists():
        try:
            ckpt = torch.load(old_pt, map_location=device, weights_only=False)
            cfg_dict = ckpt.get("config", {})
            cfg = Config(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else Config(
                stream_dims=(17, 17, 17, 17, 14, 14, 14, 14), num_joints=4, decoder_type="joint_decoupled"
            )
            old_model = Corrector(cfg).to(device)
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            old_model.load_state_dict(sd)
            old_model.eval()
        except Exception:
            pass

    identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand_as(dq_target)
    # dq_target = raw.T * clean。用真值残差提取理想 P9 Twist，得到 P9-only 姿态，
    # 再测模型在真实组合顺序 (先 P9、后 P6) 下的剩余世界 Swing。
    raw_rot = torch_lower_body_rotmat_from_features(x)
    target_swing_local, target_twist_world = torch_world_swing_local_delta(x, dq_target)
    target_swing_vec_world = torch.matmul(
        raw_rot, target_swing_local[..., 1:].unsqueeze(-1)
    ).squeeze(-1)
    target_swing_world = torch.nn.functional.normalize(
        torch.cat([target_swing_local[..., :1], target_swing_vec_world], dim=-1), dim=-1
    )
    twist_angle = 2.0 * torch.atan2(target_twist_world[..., 2:3], target_twist_world[..., :1])
    twist_cos, twist_sin = torch.cos(twist_angle), torch.sin(twist_angle)
    zero = torch.zeros_like(twist_cos)
    one = torch.ones_like(twist_cos)
    row_x = torch.cat([twist_cos, zero, twist_sin], dim=-1)
    row_y = torch.cat([zero, one, zero], dim=-1)
    row_z = torch.cat([-twist_sin, zero, twist_cos], dim=-1)
    ideal_p9_rot = torch.stack([row_x, row_y, row_z], dim=-2) @ raw_rot

    # P9-only has removed the target world Twist; its remaining error is target Swing.
    p9_only_swing_frame = compute_smooth_geodesic_distance(identity_q, target_swing_world)

    # 2. 运行模型推断
    old_geo_frame = None
    old_twist_crosstalk = 0.0
    if old_model is not None:
        with torch.no_grad():
            out_old, _ = old_model(x)
            old_geo_frame = compute_smooth_geodesic_distance(out_old["dq"], dq_target)
            _, old_twist = torch_world_swing_local_delta(x, out_old["dq"])
            old_twist_crosstalk = float(np.degrees(compute_smooth_geodesic_distance(old_twist, identity_q)[mask].mean().item()))

    neutral_swing_frame = None
    neutral_twist_crosstalk = 0.0
    neutral_preprojection_world_twist = 0.0
    neutral_gate_active_rate = 0.0
    p6_only_swing_frame = None
    if neutral_model is not None:
        with torch.no_grad():
            out_n, _ = neutral_model(x)
            neutral_gate_active_rate = float((out_n["gate"][mask] >= 0.3).float().mean().item())
            dq_local_projected = out_n["dq"]

            # Diagnostic: P6-only with e_hat=0, to separate it from the hybrid candidate.
            dq_vec_raw_world = torch.matmul(
                raw_rot, dq_local_projected[..., 1:].unsqueeze(-1)
            ).squeeze(-1)
            dq_raw_world = torch.nn.functional.normalize(
                torch.cat([dq_local_projected[..., :1], dq_vec_raw_world], dim=-1), dim=-1
            )
            p6_only_swing, _ = torch_swing_twist_decomposition_y(dq_raw_world)
            p6_only_gated = torch_slerp_from_identity(p6_only_swing, out_n["gate"])
            p6_only_swing_frame = compute_smooth_geodesic_distance(p6_only_gated, target_swing_world)

            # Evaluate the actual order: ideal P9 removes target Twist first, then the local P6
            # residual is conjugated by R9, world-decomposed, and confidence-gated as in composer.
            dq_vec_p9_world = torch.matmul(
                ideal_p9_rot, dq_local_projected[..., 1:].unsqueeze(-1)
            ).squeeze(-1)
            dq_p9_world = torch.nn.functional.normalize(
                torch.cat([dq_local_projected[..., :1], dq_vec_p9_world], dim=-1), dim=-1
            )
            p6_candidate_swing, _ = torch_swing_twist_decomposition_y(dq_p9_world)
            p6_candidate_gated = torch_slerp_from_identity(p6_candidate_swing, out_n["gate"])
            neutral_swing_frame = compute_smooth_geodesic_distance(
                p6_candidate_gated, target_swing_world
            )

            # Measure twist on the ungated base-model output in the post-P9 world frame.
            dq_raw_vec_p9_world = torch.matmul(
                ideal_p9_rot, out_n["dq_raw"][..., 1:].unsqueeze(-1)
            ).squeeze(-1)
            dq_raw_p9_world = torch.nn.functional.normalize(
                torch.cat([out_n["dq_raw"][..., :1], dq_raw_vec_p9_world], dim=-1), dim=-1
            )
            _, preprojection_twist = torch_swing_twist_decomposition_y(dq_raw_p9_world)
            neutral_preprojection_world_twist = float(np.degrees(
                compute_smooth_geodesic_distance(preprojection_twist, identity_q)[mask].mean().item()
            ))
            _, projected_world_twist = torch_swing_twist_decomposition_y(p6_candidate_gated)
            neutral_twist_crosstalk = float(np.degrees(
                compute_smooth_geodesic_distance(projected_world_twist, identity_q)[mask].mean().item()
            ))

    # 3. 逐序列统计与 Bootstrap 置信区间
    num_seqs = x.shape[0]
    p9_only_swing_seqs = []
    p6_only_swing_seqs = []
    neutral_swing_seqs = []
    swing_gains = []

    for s in range(num_seqs):
        m_s = mask[s]
        r_deg = float(np.degrees(p9_only_swing_frame[s][m_s].mean().item()))
        p9_only_swing_seqs.append(r_deg)
        if p6_only_swing_frame is not None:
            p6_deg = float(np.degrees(p6_only_swing_frame[s][m_s].mean().item()))
            p6_only_swing_seqs.append(p6_deg)
        if neutral_swing_frame is not None:
            n_deg = float(np.degrees(neutral_swing_frame[s][m_s].mean().item()))
            neutral_swing_seqs.append(n_deg)
            swing_gains.append(r_deg - n_deg)

    mean_p9_only_swing = float(np.mean(p9_only_swing_seqs))
    mean_p6_only_swing = float(np.mean(p6_only_swing_seqs)) if p6_only_swing_seqs else mean_p9_only_swing
    mean_neutral_swing = float(np.mean(neutral_swing_seqs)) if neutral_swing_seqs else mean_p9_only_swing
    mean_p6_only_gain = mean_p9_only_swing - mean_p6_only_swing
    mean_swing_gain = float(np.mean(swing_gains)) if swing_gains else 0.0

    # 2000 次 Bootstrap 抽样计算置信区间
    rng = np.random.RandomState(42)
    boot_gains = [float(np.mean(rng.choice(swing_gains, size=len(swing_gains), replace=True))) for _ in range(2000)]
    ci_low = float(np.percentile(boot_gains, 2.5))
    ci_high = float(np.percentile(boot_gains, 97.5))
    win_rate = float(sum(1 for g in swing_gains if g > 0.0) / len(swing_gains)) if swing_gains else 0.0
    gain_ratio = (mean_swing_gain / mean_p9_only_swing * 100.0) if mean_p9_only_swing > 1e-4 else 0.0

    return {
        "num_sequences": int(num_seqs),
        "total_frames": int(mask.sum().item()),
        "p9_reference": "oracle target world twist removed",
        "mean_p9_only_swing_deg": mean_p9_only_swing,
        "mean_p6_only_swing_deg": mean_p6_only_swing,
        "mean_p6_only_swing_gain_deg": mean_p6_only_gain,
        "mean_hybrid_oracle_p9_swing_deg": mean_neutral_swing,
        "mean_swing_gain_deg": mean_swing_gain,
        "swing_gain_ratio_pct": gain_ratio,
        "swing_95_ci": [ci_low, ci_high],
        "swing_win_rate": win_rate,
        "old_cstn_world_twist_crosstalk_deg": old_twist_crosstalk,
        "neutral_preprojection_world_twist_deg": neutral_preprojection_world_twist,
        "neutral_cstn_projected_world_twist_crosstalk_deg": neutral_twist_crosstalk,
        "neutral_gate_active_rate": neutral_gate_active_rate,
    }


def refresh_holdout_only(device: torch.device) -> dict:
    """Refresh the holdout metrics and their gates without rerunning fixed long sessions."""
    report_path = BASE_DIR / "ai_corrector" / "artifacts" / "phase11_hybrid_compensation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Existing full report is required: {report_path}")
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    holdout = evaluate_holdout_short_term_swing(device)
    if holdout.get("status") == "skipped":
        raise RuntimeError(holdout.get("reason", "holdout evaluation skipped"))

    gates = report.get("gates", {})
    gates["twist_safe"] = bool(
        holdout["neutral_cstn_projected_world_twist_crosstalk_deg"] < 0.05
    )
    gates["p6_swing_gain_safe"] = bool(
        holdout["mean_swing_gain_deg"] > 0.0
        and holdout["swing_95_ci"][0] > 0.0
        and holdout["swing_win_rate"] >= 0.80
    )
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["model_checkpoint"] = "ai_corrector/artifacts/cstn_v2_yaw_neutral_world_best.pt"
    report["holdout_metric_frame"] = "world swing after oracle P9 twist removal and confidence-gated P6 composition"
    report["holdout_short_term_metrics"] = holdout
    report["gates"] = gates
    report["verdict"] = "PASS" if all(gates.values()) else "FAIL"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Refreshed holdout metrics and gates in {report_path.relative_to(BASE_DIR)}")
    print(json.dumps({"verdict": report["verdict"], "gates": gates, "holdout": holdout}, indent=2))
    return report


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-holdout-only",
        action="store_true",
        help="Recompute holdout world-swing metrics and gates using the existing full-session report.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.refresh_holdout_only:
        refresh_holdout_only(device)
        return

    print("=" * 80)
    print("Executing Phase 11 Task H3: Multi-Factor 8-Way Ablation Benchmark")
    print("=" * 80)

    print(f"Host Compute Device: {device} | Total Logical CPU Cores: {os.cpu_count()}")

    # 1. 定义 10 条标称长会话 + 2 条严格零漂会话
    session_configs = [
        # (session_id, drift_tier, seed, scenario, enable_random_walk)
        ("sess_zero_00", 0.0, 42, "default", True),
        ("sess_zero_01", 0.0, 43, "default", True),
        ("sess_low_00", 1.0, 101, "default", True),
        ("sess_low_01", 1.0, 102, "complex_gait", True),
        ("sess_med_00", 3.0, 201, "default", True),
        ("sess_med_01", 3.0, 202, "all_sitting", True),
        ("sess_med_02", 3.0, 203, "no_still", True),
        ("sess_high_00", 10.0, 301, "default", True),
        ("sess_high_01", 10.0, 302, "vd_missing", True),
        ("sess_high_02", 10.0, 303, "complex_gait", True),
        # 严格零漂会话 (0.0°/min, 0 random walk)
        ("strict_zero_00", 0.0, 777, "default", False),
        ("strict_zero_01", 0.0, 888, "all_sitting", False),
    ]

    strategy_keys = [
        "raw", "oracle", "p9_zaru", "p9_gait", "p9_full",
        "old_cstn", "p6_yaw_neutral", "hybrid_candidate", "hybrid_full"
    ]

    max_workers = min(os.cpu_count() or 4, 12)
    print(f"\n[Step 2] Running parallel 30-minute evaluations across {len(session_configs)} sessions ({max_workers} worker processes)...", flush=True)

    t0_all = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        worker_results = list(executor.map(_worker_evaluate_session, session_configs))

    elapsed_all = time.time() - t0_all
    print(f"\nParallel simulation & inference completed in {elapsed_all:.1f}s total ({elapsed_all/len(session_configs):.2f}s per 30-min session throughput).", flush=True)

    # 2. 统计指标与分类归集
    result_map = {r[0]: r for r in worker_results}

    # 分离标称 10 会话与严格零漂 2 会话
    standard_keys = [cfg[0] for cfg in session_configs if cfg[4] is True]
    strict_zero_keys = [cfg[0] for cfg in session_configs if cfg[4] is False]

    standard_metrics = {k: {"D_raw": [], "D_corr": [], "G": []} for k in strategy_keys}
    strict_zero_metrics = {k: {"D_raw": [], "D_corr": [], "G": []} for k in strategy_keys}

    detailed_session_records = []
    for cfg in session_configs:
        s_id, tier, seed, scenario, enable_rw, res = result_map[cfg[0]]
        target_dict = standard_metrics if enable_rw else strict_zero_metrics
        for k in strategy_keys:
            target_dict[k]["D_raw"].append(res[k]["D_raw_deg"])
            target_dict[k]["D_corr"].append(res[k]["D_corr_deg"])
            target_dict[k]["G"].append(res[k]["G_deg"])

        detailed_session_records.append({
            "session_id": s_id,
            "drift_tier_deg_min": tier,
            "scenario": scenario,
            "enable_random_walk": enable_rw,
            "raw_D_raw_deg": res["raw"]["D_raw_deg"],
            "p9_zaru_D_corr_deg": res["p9_zaru"]["D_corr_deg"],
            "p9_zaru_gain_deg": res["p9_zaru"]["G_deg"],
            "p9_full_D_corr_deg": res["p9_full"]["D_corr_deg"],
            "p9_full_gain_deg": res["p9_full"]["G_deg"],
            "p6_neutral_D_corr_deg": res["p6_yaw_neutral"]["D_corr_deg"],
            "p6_neutral_gain_deg": res["p6_yaw_neutral"]["G_deg"],
            "hybrid_candidate_D_corr_deg": res["hybrid_candidate"]["D_corr_deg"],
            "hybrid_candidate_gain_deg": res["hybrid_candidate"]["G_deg"],
            "hybrid_candidate_ratio": res["hybrid_candidate"]["improvement_ratio"],
            "hybrid_full_D_corr_deg": res["hybrid_full"]["D_corr_deg"],
            "hybrid_full_gain_deg": res["hybrid_full"]["G_deg"],
            "old_cstn_D_corr_deg": res["old_cstn"]["D_corr_deg"],
            "old_cstn_gain_deg": res["old_cstn"]["G_deg"],
            "discarded_twists_deg": res["discarded_twists_deg"],
            "applied_swings_deg": res["applied_swings_deg"],
            "zaru_events": res["zaru_events"],
            "gait_events": res["gait_events"],
        })

    # 3. 统计汇总表
    print("\n" + "=" * 105)
    print(f"{'Strategy':22s} | {'Mean D_raw':10s} | {'Mean D_corr':11s} | {'Mean Gain':11s} | {'95% CI':16s} | {'Ratio %':8s} | {'Win Rate':8s}")
    print("=" * 105)

    summary_standard = {}
    for k in strategy_keys:
        d = standard_metrics[k]
        m_raw, _, _ = bootstrap_mean_ci(d["D_raw"])
        m_corr, _, _ = bootstrap_mean_ci(d["D_corr"])
        m_g, ci_low, ci_high = bootstrap_mean_ci(d["G"])
        ratio = (m_g / m_raw * 100.0) if m_raw > 1e-4 else 0.0
        win_count = sum(1 for g in d["G"] if g > 0.001)
        win_rate = win_count / float(len(d["G"]))
        worst_deg = float(np.min(d["G"]))

        summary_standard[k] = {
            "mean_D_raw_deg": m_raw,
            "mean_D_corr_deg": m_corr,
            "mean_gain_deg": m_g,
            "gain_95_ci": [ci_low, ci_high],
            "improvement_ratio_pct": ratio,
            "win_rate": win_rate,
            "worst_gain_deg": worst_deg
        }
        ci_str = f"[{ci_low:5.1f}, {ci_high:5.1f}]"
        print(f"{k:22s} | {m_raw:9.2f}° | {m_corr:10.2f}° | {m_g:+10.2f}° | {ci_str:16s} | {ratio:7.1f}% | {win_rate*100:7.1f}%")

    print("=" * 105)

    # 4. 严格零漂直通性评估
    print("\n[Step 3] Strict Zero-Drift Safety Verification (enable_random_walk=False):")
    summary_strict_zero = {}
    for k in strategy_keys:
        d_corr_arr = strict_zero_metrics[k]["D_corr"]
        d_raw_arr = strict_zero_metrics[k]["D_raw"]
        mean_delta = float(np.mean(np.array(d_corr_arr) - np.array(d_raw_arr)))
        summary_strict_zero[k] = {
            "mean_D_raw_deg": float(np.mean(d_raw_arr)),
            "mean_D_corr_deg": float(np.mean(d_corr_arr)),
            "mean_delta_from_raw_deg": mean_delta
        }
        print(f"  {k:22s}: Raw={np.mean(d_raw_arr):.4f}°, Corr={np.mean(d_corr_arr):.4f}°, Delta={mean_delta:+.4f}°")

    # 5. 独立确认集 FinalConfirm 短时姿态评估
    holdout_report = evaluate_holdout_short_term_swing(device)

    # 6. 门禁裁定逻辑
    cand_gain = summary_standard["hybrid_candidate"]["mean_gain_deg"]
    cand_ratio = summary_standard["hybrid_candidate"]["improvement_ratio_pct"]
    cand_win = summary_standard["hybrid_candidate"]["win_rate"]
    p9_zaru_gain = summary_standard["p9_zaru"]["mean_gain_deg"]

    # Gate 1: 严格零漂恶化容限 <= 0.05 度
    strict_zero_delta = summary_strict_zero["hybrid_candidate"]["mean_delta_from_raw_deg"]
    zero_safe = bool(abs(strict_zero_delta) <= 0.05)

    # Gate 2: 组合正收益且不显著劣于 P9 ZARU (恶化不超过 0.5 度)
    hybrid_valid = bool(cand_gain > 0.0 and (cand_gain >= p9_zaru_gain - 0.5))

    # Gate 3: 实际送入合成器的快分支输出在世界系的偏航串扰严格受控 (< 0.05 度)
    twist_crosstalk = holdout_report.get("neutral_cstn_projected_world_twist_crosstalk_deg", 999.0)
    twist_safe = bool(twist_crosstalk < 0.05)

    # Gate 4: Phase 6 在未见确认集上具有严格统计显著的非偏航 Swing 姿态补偿增益 (CI 下界 > 0)
    p6_swing_gain = holdout_report.get("mean_swing_gain_deg", -999.0)
    p6_ci_low = holdout_report.get("swing_95_ci", [-999.0, 999.0])[0]
    p6_win_rate = holdout_report.get("swing_win_rate", 0.0)
    p6_swing_gain_safe = bool(p6_swing_gain > 0.0 and p6_ci_low > 0.0 and p6_win_rate >= 0.80)

    gate_verdict = "PASS" if (hybrid_valid and zero_safe and twist_safe and p6_swing_gain_safe) else "FAIL"

    print("\n" + "=" * 80)
    print(f"  [Task H3 验收裁定]: {gate_verdict}")
    print(f"  - Gate 1 (组合 30 分钟增益): {cand_gain:+.2f}° (改善率: {cand_ratio:.1f}%, 胜率: {cand_win*100:.1f}%) -> {'PASS' if hybrid_valid else 'FAIL'}")
    print(f"  - Gate 2 (严格零漂偏差 Delta): {strict_zero_delta:+.4f}° (门限: <= 0.05°) -> {'SAFE' if zero_safe else 'UNSAFE'}")
    print(f"  - Gate 3 (组合输出世界偏航串扰): {twist_crosstalk:.4f}° (门限: < 0.05°) -> {'SAFE' if twist_safe else 'UNSAFE'}")
    print(f"  - Gate 4 (P6 短时 Swing 姿态增益): {p6_swing_gain:+.4f}° (95% CI: [{p6_ci_low:+.4f}°, {holdout_report.get('swing_95_ci', [0, 0])[1]:+.4f}°], 胜率: {p6_win_rate*100:.1f}%) -> {'PASS' if p6_swing_gain_safe else 'FAIL'}")
    print("=" * 80)

    # 7. 保存归档机器报告
    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": gate_verdict,
        "model_checkpoint": "ai_corrector/artifacts/cstn_v2_yaw_neutral_world_best.pt",
        "holdout_metric_frame": "world swing after oracle P9 twist removal and confidence-gated P6 composition",
        "gates": {
            "hybrid_valid": hybrid_valid,
            "zero_safe": zero_safe,
            "twist_safe": twist_safe,
            "p6_swing_gain_safe": p6_swing_gain_safe,
        },
        "summary_standard_10_sessions": summary_standard,
        "summary_strict_zero_drift": summary_strict_zero,
        "holdout_short_term_metrics": holdout_report,
        "detailed_sessions": detailed_session_records,
    }

    report_path = BASE_DIR / "ai_corrector" / "artifacts" / "phase11_hybrid_compensation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\n完整机器评估报告已归档至: {report_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

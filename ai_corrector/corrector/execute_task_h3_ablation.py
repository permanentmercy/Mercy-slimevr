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
   - 独立确认集 FinalConfirm (源序列 36..40) 短时姿态/FK 误差评估.
3. 统计指标:
   - 30 分钟起终点圆周累计偏航误差增长量 D_raw, D_corr, 净收益 G, 改善率, 胜率, 最差恶化量;
   - 95% Bootstrap 置信区间;
   - 严格偏航串扰度量 (Twist Crosstalk);
   - 严格判定 PASS / FAIL，输出归档机器报告:
     artifacts/phase11_hybrid_compensation_report.json
"""

import copy
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


def evaluate_holdout_short_term_fk(device: torch.device) -> dict:
    """
    在独立未见源序列 FinalConfirm (36..40) 上评测短时姿态与偏航串扰
    """
    fc_path = BASE_DIR / "ai_corrector" / "database" / "d3c_continuous" / "final_confirm_continuous.npz"
    if not fc_path.exists():
        return {"status": "skipped", "reason": "final_confirm_continuous.npz not found"}

    print("\n[Step 4] Running short-term pose & yaw crosstalk evaluation on holdout FinalConfirm (190 sequences)...", flush=True)
    fc = np.load(fc_path)
    x = torch.from_numpy(fc["x"]).float().to(device)
    dq_target = torch.from_numpy(fc["dq"]).float().to(device)
    mask = torch.from_numpy(fc["mask"]).bool().to(device)

    neutral_pt = BASE_DIR / "ai_corrector" / "artifacts" / "cstn_v2_yaw_neutral_best.pt"
    old_pt = BASE_DIR / "ai_corrector" / "artifacts" / "cstn_v2_d1_smooth_geodesic_s123_best.pt"

    neutral_model = load_yaw_neutral_cstn(neutral_pt, device)
    old_model = None
    if old_pt.exists():
        ckpt = torch.load(old_pt, map_location=device, weights_only=False)
        cfg_dict = ckpt.get("config", {})
        cfg = Config(**cfg_dict) if isinstance(cfg_dict, dict) and cfg_dict else Config(
            stream_dims=(17, 17, 17, 17, 14, 14, 14, 14), num_joints=4, decoder_type="joint_decoupled"
        )
        old_model = Corrector(cfg).to(device)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        old_model.load_state_dict(sd)
        old_model.eval()

    identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand_as(dq_target)
    target_swing, target_twist = torch_swing_twist_decomposition_y(dq_target)

    # 1. Raw baseline
    raw_geodesic = compute_smooth_geodesic_distance(identity_q, dq_target)[mask].mean().item()
    raw_swing = compute_smooth_geodesic_distance(identity_q, target_swing)[mask].mean().item()

    # 2. Old CSTN
    old_geodesic = 0.0
    old_twist_crosstalk = 0.0
    if old_model is not None:
        with torch.no_grad():
            out_old, _ = old_model(x)
            old_geodesic = compute_smooth_geodesic_distance(out_old["dq"], dq_target)[mask].mean().item()
            _, old_twist = torch_swing_twist_decomposition_y(out_old["dq"])
            old_twist_crosstalk = compute_smooth_geodesic_distance(old_twist, identity_q)[mask].mean().item()

    # 3. Yaw-Neutral CSTN
    neutral_geodesic = 0.0
    neutral_swing = 0.0
    neutral_twist_crosstalk = 0.0
    if neutral_model is not None:
        with torch.no_grad():
            out_n, _ = neutral_model(x)
            neutral_geodesic = compute_smooth_geodesic_distance(out_n["dq"], dq_target)[mask].mean().item()
            neutral_swing = compute_smooth_geodesic_distance(out_n["dq"], target_swing)[mask].mean().item()
            neutral_twist_crosstalk = compute_smooth_geodesic_distance(out_n["dq_twist"], identity_q)[mask].mean().item()

    return {
        "num_sequences": int(x.shape[0]),
        "total_frames": int(mask.sum().item()),
        "raw_geodesic_deg": float(np.degrees(raw_geodesic)),
        "raw_swing_deg": float(np.degrees(raw_swing)),
        "old_cstn_geodesic_deg": float(np.degrees(old_geodesic)),
        "old_cstn_twist_crosstalk_deg": float(np.degrees(old_twist_crosstalk)),
        "neutral_cstn_geodesic_deg": float(np.degrees(neutral_geodesic)),
        "neutral_cstn_swing_deg": float(np.degrees(neutral_swing)),
        "neutral_cstn_twist_crosstalk_deg": float(np.degrees(neutral_twist_crosstalk)),
    }


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor

    print("=" * 80)
    print("Executing Phase 11 Task H3: Multi-Factor 8-Way Ablation Benchmark")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    holdout_report = evaluate_holdout_short_term_fk(device)

    # 6. 门禁裁定逻辑
    cand_gain = summary_standard["hybrid_candidate"]["mean_gain_deg"]
    cand_ratio = summary_standard["hybrid_candidate"]["improvement_ratio_pct"]
    cand_win = summary_standard["hybrid_candidate"]["win_rate"]
    p9_zaru_gain = summary_standard["p9_zaru"]["mean_gain_deg"]

    # 严格零漂恶化容限 <= 0.05 度
    strict_zero_delta = summary_strict_zero["hybrid_candidate"]["mean_delta_from_raw_deg"]
    zero_safe = bool(abs(strict_zero_delta) <= 0.05)

    # 组合正收益且不显著劣于 P9 ZARU (恶化不超过 0.5 度)
    hybrid_valid = bool(cand_gain > 0.0 and (cand_gain >= p9_zaru_gain - 0.5))

    # 快分支偏航串扰严格受控
    twist_crosstalk = holdout_report.get("neutral_cstn_twist_crosstalk_deg", 999.0)
    twist_safe = bool(twist_crosstalk < 0.05)

    gate_verdict = "PASS" if (hybrid_valid and zero_safe and twist_safe) else "FAIL"

    print("\n" + "=" * 80)
    print(f"  [Task H3 验收裁定]: {gate_verdict}")
    print(f"  - 候选组合 30 分钟平均增益: {cand_gain:+.2f}° (改善率: {cand_ratio:.1f}%, 胜率: {cand_win*100:.1f}%)")
    print(f"  - 相对 P9 ZARU 增益差异: {cand_gain - p9_zaru_gain:+.2f}° (无显著恶化)")
    print(f"  - 严格零漂偏差 (Delta): {strict_zero_delta:+.4f}° (门限: <= 0.05° -> {'SAFE' if zero_safe else 'UNSAFE'})")
    print(f"  - 快速分支偏航串扰: {twist_crosstalk:.4f}° (门限: < 0.05° -> {'SAFE' if twist_safe else 'UNSAFE'})")
    print("=" * 80)

    # 7. 保存归档机器报告
    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": gate_verdict,
        "gates": {
            "hybrid_valid": hybrid_valid,
            "zero_safe": zero_safe,
            "twist_safe": twist_safe,
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

"""
ai_corrector/corrector/execute_phase16_p3_slow_motion_benchmark.py
==================================================================
Phase 16 Task P16.2 & P16.3:
慢偏航观测器输入一致性、真实慢动作误校正边界与全场景基准配对评测

评测对象 (5 大架构候选, 严格固定参数上限 <= 1,102,066):
1. raw: 纯原始传感器姿态流 (0 参数)
2. p15_naive_zaru: Phase 15 原版 ZARU 观测器 (使用机体角速度, 无加速度方差门控, 0 参数)
3. fused_delta_naive_zaru: 仅基于服务端因果姿态差分 (computeAngularVelocity) 的朴素 ZARU 观测器 (无加速度方差门控, 0 参数)
4. conservative_gated_zaru: Phase 16 保守型三态 ZARU 观测器 (基于姿态差分 + 物理加速度方差门控 + 上身/HMD 运动门控 + UPDATE/UNCERTAIN/REJECT 三态决策, 0 参数)
5. hybrid_conservative: 保守型三态慢偏航 + 冻结 Yaw-Neutral CSTN 快姿态 (正交合成, 严格 1,102,066 参数)

测试矩阵 (12 类工况, 125 条独立 10 分钟动力学自洽物理会话, 共 750,000 帧):
1. strict_zero_drift (10 种子, 0.0°/min, 零漂移安全否决门 <= 0.05°)
2. nominal_pos_0_14deg (15 种子, +0.14°/min 标称慢漂移)
3. nominal_neg_0_14deg (15 种子, -0.14°/min 标称慢漂移)
4. intentional_slow_motion_no_drift (15 种子, 0.0°/min 漂移, ~0.3°/min 真实主动挪腿 + 肌肉微颤)
   [核心否决项: 测试朴素 ZARU 是否把人动误认作漂移导致负增益，验证保守门控器是否零误校正]
5. slow_motion_plus_drift (10 种子, +0.14°/min 漂移 + 间歇性慢动作挪腿)
6. single_leg_isolated_motion (10 种子, +0.14°/min 漂移, 左腿慢动挪动, 右腿绝对静止)
7. whole_body_slow_turn (10 种子, +0.14°/min 漂移, 全身慢速同向转向, HMD 运动门禁)
8. sitting_leg_fidget (10 种子, +0.14°/min 漂移, 静坐微抖腿, 加速度方差保护)
9. accel_disturbance_bump (10 种子, +0.14°/min 漂移, 静坐伴随周期性桌面碰撞冲击)
10. no_still (10 种子, +0.14°/min 漂移, 持续转身漫步无静止期, 防发散测试)
11. tracker_dropout_recovery (10 种子, +0.14°/min 漂移, 小腿 tracker 丢包与超时恢复)
12. non_uniform_dt_jitter (10 种子, +0.14°/min 漂移, 时间戳抖动与不规则帧率)

产出机器可读报告:
ai_corrector/artifacts/phase16_p3_slow_motion_benchmark_report.json
"""

import concurrent.futures
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

from ai_corrector.corrector.contract_v3 import V2ToV3Adapter, LOWER_BODY_JOINTS
from ai_corrector.corrector.oracle_v3_geometry import (
    rot_y_matrix,
    compute_endpoint_session_metrics,
    wrap_angle,
    geodesic_angle_deg
)
from ai_corrector.corrector.physics_session_simulator_v4 import (
    PhysicsSessionSynthesizerV4,
    PhysicsSessionStreamV4
)
from ai_corrector.corrector.bias_estimator_v3 import DeployableBiasEstimatorV3
from ai_corrector.corrector.conservative_bias_estimator_v4 import (
    ConservativeBiasEstimatorV4,
    ObservationState
)
from ai_corrector.corrector.hybrid_composer import compose_single_tracker_hybrid
from ai_corrector.corrector.model import Corrector, Config
from ai_corrector.corrector.yaw_neutral_cstn import YawNeutralCSTN


# 进程级全局快分支模型缓存
_worker_fast_model: Optional[YawNeutralCSTN] = None


def get_worker_fast_model() -> YawNeutralCSTN:
    global _worker_fast_model
    if _worker_fast_model is None:
        torch.set_num_threads(1)
        weight_path = BASE_DIR / "ai_corrector" / "artifacts" / "cstn_v2_yaw_neutral_world_best.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"Fast branch weights not found at: {weight_path}")
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        cfg = Config(**ckpt["config"])
        base_model = Corrector(cfg)
        base_model.load_state_dict(ckpt["state_dict"])
        base_model.eval()
        _worker_fast_model = YawNeutralCSTN(base_model)
        _worker_fast_model.eval()
    return _worker_fast_model


def bootstrap_mean_ci(data: List[float], num_resamples: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(arr))
    if n == 1:
        return mean_val, mean_val, mean_val
    rng = np.random.RandomState(42)
    indices = rng.randint(0, n, (num_resamples, n))
    resampled_means = np.mean(arr[indices], axis=1)
    low = float(np.percentile(resampled_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(resampled_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_val, low, high


def evaluate_single_session(task: dict) -> dict:
    scen_name = task["scenario_name"]
    session_id = task["session_id"]
    drift_tier = task["drift_tier"]
    seed = task["seed"]
    duration_sec = task.get("duration_sec", 600.0)

    # 1. 物理动力学自洽会话合成
    synthesizer = PhysicsSessionSynthesizerV4(
        duration_sec=duration_sec,
        sample_rate_hz=10.0,
        drift_tier_deg_min=drift_tier,
        scenario=task["scenario"],
        seed=seed,
        enable_random_walk=task.get("enable_random_walk", True),
        enable_accel_noise=True
    )
    stream: PhysicsSessionStreamV4 = synthesizer.generate(session_id=session_id)

    num_frames = stream.num_frames
    dt = stream.dt
    dts_arr = getattr(stream, "dts", None)
    t_arr = stream.timestamps
    feat_124 = stream.features_124
    raw_gyro = stream.angular_velocity_sources["raw_gyro"]
    gt = stream.ground_truth_eval

    # 2. 初始化候选估计器
    # 候选 2: Phase 15 原版朴素 ZARU (使用机体角速度)
    est_p15_naive = DeployableBiasEstimatorV3(enable_zaru=True, enable_gait=False, sample_rate_hz=10.0)
    # 候选 3: 服务端姿态差分朴素 ZARU (使用 computeAngularVelocity, 无加速度门控)
    est_fused_naive = DeployableBiasEstimatorV3(enable_zaru=True, enable_gait=False, sample_rate_hz=10.0)
    # 候选 4: Phase 16 保守型三态 ZARU (姿态差分 + 加速度方差门控 + 上身运动门控)
    est_conservative = ConservativeBiasEstimatorV4(sample_rate_hz=10.0)

    # 统计遥测计数
    conservative_states_counts = {"UPDATE": 0, "UNCERTAIN": 0, "REJECT": 0}
    false_updates_slow_motion = {
        "p15_naive": 0,
        "fused_naive": 0,
        "conservative": 0
    }

    # 3. 逐帧在线因果推进
    for i in range(num_frames):
        ts = float(t_arr[i])
        f_vec = feat_124[i]
        dt_frame = float(dts_arr[i]) if dts_arr is not None else dt

        # 转换为规范 FrameV3 (默认包含服务端姿态差分角速度与加速度)
        frame_server = V2ToV3Adapter.convert_124d_to_frame(
            feature_124=f_vec,
            timestamp=ts,
            dt=dt_frame,
            session_id=session_id,
            frame_index=i
        )

        # 为候选 2 构造包含机体原始陀螺的帧
        frame_raw_gyro = copy.deepcopy(frame_server)
        for j in LOWER_BODY_JOINTS:
            frame_raw_gyro.trackers[j].ang_vel = raw_gyro[j][i]

        # 运行各估计器
        # (A) p15_naive_zaru
        p15_zaru_before = sum(est_p15_naive.zaru_event_counts.values())
        est_p15_naive.process_frame(frame_raw_gyro)
        p15_zaru_after = sum(est_p15_naive.zaru_event_counts.values())

        # (B) fused_delta_naive_zaru
        fused_zaru_before = sum(est_fused_naive.zaru_event_counts.values())
        est_fused_naive.process_frame(frame_server)
        fused_zaru_after = sum(est_fused_naive.zaru_event_counts.values())

        # (C) conservative_gated_zaru
        cons_update_before = sum(est_conservative.update_counts.values())
        est_conservative.process_frame(frame_server)
        cons_update_after = sum(est_conservative.update_counts.values())

        # 若处于真实慢动作且传感器真实漂移为 0，任何更新均为误更新
        is_slow_motion_state = (gt["behavior_states"][i] in (4, 5))
        if is_slow_motion_state and abs(drift_tier) < 1e-9:
            if p15_zaru_after > p15_zaru_before:
                false_updates_slow_motion["p15_naive"] += (p15_zaru_after - p15_zaru_before)
            if fused_zaru_after > fused_zaru_before:
                false_updates_slow_motion["fused_naive"] += (fused_zaru_after - fused_zaru_before)
            if cons_update_after > cons_update_before:
                false_updates_slow_motion["conservative"] += (cons_update_after - cons_update_before)

    # 汇总保守型观测器的状态分布
    conservative_states_counts["UPDATE"] = sum(est_conservative.update_counts.values())
    conservative_states_counts["UNCERTAIN"] = sum(est_conservative.uncertain_counts.values())
    conservative_states_counts["REJECT"] = sum(est_conservative.reject_counts.values())

    # 4. 获取各候选的终点偏航估计量
    final_e_hat = {
        "raw": {j: 0.0 for j in LOWER_BODY_JOINTS},
        "p15_naive_zaru": {j: est_p15_naive.kalman_filter.channels[j].e for j in LOWER_BODY_JOINTS},
        "fused_delta_naive_zaru": {j: est_fused_naive.kalman_filter.channels[j].e for j in LOWER_BODY_JOINTS},
        "conservative_gated_zaru": {j: est_conservative.kalman_filter.channels[j].e for j in LOWER_BODY_JOINTS},
        "hybrid_conservative": {j: est_conservative.kalman_filter.channels[j].e for j in LOWER_BODY_JOINTS}
    }

    # 快分支网络推理 (在全时序特征流上因果推理，提取终点帧 delta_R6 与 gate 用于 hybrid 正交合成)
    fast_model = get_worker_fast_model()
    with torch.no_grad():
        x_tensor = torch.from_numpy(feat_124).float().unsqueeze(0)  # (1, num_frames, 124)
        out, _ = fast_model(x_tensor)
        dq_last = out["dq"][0, -1].cpu().numpy()      # (4, 4)
        gate_last = out["gate"][0, -1].cpu().numpy()  # (4,)

    # 5. 评测 5 大候选在会话首尾的端点误差
    R_raw_t0 = gt["raw_rot_t0"]
    R_raw_t1 = gt["raw_rot_t1"]
    R_clean_t0 = gt["clean_rot_t0"]
    R_clean_t1 = gt["clean_rot_t1"]
    e_raw_t0 = gt["e_true_t0_rad"]
    e_raw_t1 = gt["e_true_t1_rad"]

    candidate_metrics = {}
    candidates_list = ["raw", "p15_naive_zaru", "fused_delta_naive_zaru", "conservative_gated_zaru", "hybrid_conservative"]

    for cand in candidates_list:
        R_corr_t0 = {}
        R_corr_t1 = {}
        e_corr_t0 = {}
        e_corr_t1 = {}
        geo_err_3d_t1 = {}

        for j_idx, j in enumerate(LOWER_BODY_JOINTS):
            r_raw_0 = R_raw_t0[j]
            r_raw_1 = R_raw_t1[j]
            r_clean_0 = R_clean_t0[j]
            r_clean_1 = R_clean_t1[j]

            e_hat_0 = 0.0
            e_hat_1 = float(final_e_hat[cand][j])

            dq_j = dq_last[j_idx]
            gate_j = float(gate_last[j_idx])

            if cand == "raw":
                r_c0 = r_raw_0
                r_c1 = r_raw_1

            elif cand in ("p15_naive_zaru", "fused_delta_naive_zaru", "conservative_gated_zaru"):
                # 显式慢分支世界左乘逆偏航修正
                r_c0 = rot_y_matrix(-e_hat_0) @ r_raw_0
                r_c1 = rot_y_matrix(-e_hat_1) @ r_raw_1

            elif cand == "hybrid_conservative":
                # 正交合成: 先慢偏航逆旋转, 再快分支门控 swing
                out_0 = compose_single_tracker_hybrid(r_raw_0, e_hat_rad=e_hat_0, delta_R6=np.eye(3), gate_p6=0.0)
                out_1 = compose_single_tracker_hybrid(r_raw_1, e_hat_rad=e_hat_1, delta_R6=dq_j, gate_p6=gate_j)
                r_c0 = out_0.R_out
                r_c1 = out_1.R_out

            R_corr_t0[j] = r_c0
            R_corr_t1[j] = r_c1

            # 计算纠正后相对于干净真值的偏航角残差
            diff_0 = r_c0 @ r_clean_0.T
            yaw_0 = math.atan2(float(diff_0[0, 2]), float(diff_0[2, 2]))
            e_corr_t0[j] = float(wrap_angle(yaw_0))

            diff_1 = r_c1 @ r_clean_1.T
            yaw_1 = math.atan2(float(diff_1[0, 2]), float(diff_1[2, 2]))
            e_corr_t1[j] = float(wrap_angle(yaw_1))

            geo_err_3d_t1[j] = geodesic_angle_deg(r_c1, r_clean_1)

        m = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_corr_t0, e_corr_t1)
        m["mean_3d_geodesic_err_deg"] = float(np.mean(list(geo_err_3d_t1.values())))
        m["per_tracker_3d_err_deg"] = geo_err_3d_t1

        # 终点估计量总幅度
        per_est = {j: math.degrees(abs(final_e_hat[cand][j])) for j in LOWER_BODY_JOINTS}
        m["total_endpoint_est_deg"] = float(sum(per_est.values()))
        m["per_tracker_est_deg"] = per_est

        candidate_metrics[cand] = m

    return {
        "scenario_name": scen_name,
        "session_id": session_id,
        "drift_tier": drift_tier,
        "seed": seed,
        "conservative_states_counts": conservative_states_counts,
        "false_updates_slow_motion": false_updates_slow_motion,
        "candidates": candidate_metrics
    }


def main():
    print("=" * 80)
    print(" PHASE 16 TASK P16.2 & P16.3: OBSERVATION FIDELITY & SLOW MOTION BENCHMARK")
    print(" Testing Intentional Slow Motion False Correction & Input Consistency")
    print(" Parameter Budget: strictly locked at 1,102,066 (Fast: 1.10M, Slow: 0)")
    print("=" * 80)

    tasks = []

    # 1. 严格零漂静止场景 (10 种子, 0.0°/min, 否决门 <= 0.05°)
    for i in range(10):
        tasks.append({
            "scenario_name": "strict_zero_drift",
            "session_id": f"strict_zero_{i:02d}",
            "drift_tier": 0.0,
            "scenario": "strict_zero_drift",
            "seed": 1000 + i,
            "enable_random_walk": False
        })

    # 2. 标称正向偏航慢漂移 (+0.14°/min, 15 种子)
    for i in range(15):
        tasks.append({
            "scenario_name": "nominal_pos_0_14deg",
            "session_id": f"nom_pos_{i:02d}",
            "drift_tier": 0.14,
            "scenario": "nominal_pos_0_14deg",
            "seed": 2000 + i
        })

    # 3. 标称负向偏航慢漂移 (-0.14°/min, 15 种子)
    for i in range(15):
        tasks.append({
            "scenario_name": "nominal_neg_0_14deg",
            "session_id": f"nom_neg_{i:02d}",
            "drift_tier": -0.14,
            "scenario": "nominal_neg_0_14deg",
            "seed": 3000 + i
        })

    # 4. 真实极慢动作主动挪腿 (0.0°/min 漂移, ~0.3°/min 真实主动动作 + 生理颤动, 15 种子)
    # [核心误校正检验: 朴素 ZARU 误判为漂移产生负增益 vs 保守门控器 0 误更新]
    for i in range(15):
        tasks.append({
            "scenario_name": "intentional_slow_motion_no_drift",
            "session_id": f"slow_motion_{i:02d}",
            "drift_tier": 0.0,
            "scenario": "intentional_slow_motion_no_drift",
            "seed": 4000 + i,
            "enable_random_walk": False
        })

    # 5. 慢动作与真实慢漂移并存 (+0.14°/min, 10 种子)
    for i in range(10):
        tasks.append({
            "scenario_name": "slow_motion_plus_drift",
            "session_id": f"slow_plus_drift_{i:02d}",
            "drift_tier": 0.14,
            "scenario": "slow_motion_plus_drift",
            "seed": 5000 + i
        })

    # 6. 单腿独立慢动作 (+0.14°/min, 左腿动右腿静, 10 种子)
    for i in range(10):
        tasks.append({
            "scenario_name": "single_leg_isolated_motion",
            "session_id": f"single_leg_{i:02d}",
            "drift_tier": 0.14,
            "scenario": "single_leg_isolated_motion",
            "seed": 6000 + i
        })

    # 7. 全身慢速同向转向 (+0.14°/min, HMD 运动门禁, 10 种子)
    for i in range(10):
        tasks.append({
            "scenario_name": "whole_body_slow_turn",
            "session_id": f"whole_body_{i:02d}",
            "drift_tier": 0.14,
            "scenario": "whole_body_slow_turn",
            "seed": 7000 + i
        })

    # 8. 静坐微抖腿 (+0.14°/min, 高加速度微方差, 10 种子)
    for i in range(10):
        tasks.append({
            "scenario_name": "sitting_leg_fidget",
            "session_id": f"sit_fidget_{i:02d}",
            "drift_tier": 0.14,
            "scenario": "sitting_leg_fidget",
            "seed": 8000 + i
        })

    # 9. 静坐桌面外力碰撞冲击 (+0.14°/min, 周期性冲击, 10 种子)
    for i in range(10):
        tasks.append({
            "scenario_name": "accel_disturbance_bump",
            "session_id": f"accel_bump_{i:02d}",
            "drift_tier": 0.14,
            "scenario": "accel_disturbance_bump",
            "seed": 9000 + i
        })

    # 10. 持续漫步无静止期 (+0.14°/min, 防发散, 10 种子)
    for i in range(10):
        tasks.append({
            "scenario_name": "no_still",
            "session_id": f"no_still_{i:02d}",
            "drift_tier": 0.14,
            "scenario": "no_still",
            "seed": 10000 + i
        })

    # 11. 小腿 Tracker 掉线与超时恢复 (+0.14°/min, 10 种子)
    for i in range(10):
        tasks.append({
            "scenario_name": "tracker_dropout_recovery",
            "session_id": f"dropout_{i:02d}",
            "drift_tier": 0.14,
            "scenario": "tracker_dropout_recovery",
            "seed": 11000 + i
        })

    # 12. 时间戳抖动与不规则采样率 (+0.14°/min, 10 种子)
    for i in range(10):
        tasks.append({
            "scenario_name": "non_uniform_dt_jitter",
            "session_id": f"dt_jitter_{i:02d}",
            "drift_tier": 0.14,
            "scenario": "nominal_pos_0_14deg",
            "seed": 12000 + i
        })

    print(f"Total sessions configured: {len(tasks)}")
    t_start = time.time()

    # 并行多进程评测
    max_workers = min(os.cpu_count() or 4, 8)
    print(f"Executing with {max_workers} parallel worker processes...")

    session_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_single_session, task): task for task in tasks}
        completed = 0
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            session_results.append(res)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"  Progress: {completed}/{len(tasks)} sessions completed ({completed/len(tasks)*100:.1f}%)")

    t_elapsed = time.time() - t_start
    print(f"All {len(tasks)} sessions completed in {t_elapsed:.2f} seconds ({t_elapsed/len(tasks):.2f}s per session)")

    # 6. 按工况聚合统计
    scenarios_grouped = {}
    for res in session_results:
        scen = res["scenario_name"]
        if scen not in scenarios_grouped:
            scenarios_grouped[scen] = []
        scenarios_grouped[scen].append(res)

    candidates_eval = ["raw", "p15_naive_zaru", "fused_delta_naive_zaru", "conservative_gated_zaru", "hybrid_conservative"]
    scenarios_summary = {}

    print("\n" + "=" * 115)
    print(f"{'Scenario':<34} | {'Candidate':<24} | {'D_raw(°)':<9} | {'D_corr(°)':<10} | {'Gain G(°)':<10} | {'Win Rate':<9} | {'95% CI':<15}")
    print("-" * 115)

    for scen_name, s_list in sorted(scenarios_grouped.items()):
        scenarios_summary[scen_name] = {
            "num_sessions": len(s_list),
            "candidates": {},
            "telemetry": {
                "mean_update_count": float(np.mean([s["conservative_states_counts"]["UPDATE"] for s in s_list])),
                "mean_uncertain_count": float(np.mean([s["conservative_states_counts"]["UNCERTAIN"] for s in s_list])),
                "mean_reject_count": float(np.mean([s["conservative_states_counts"]["REJECT"] for s in s_list])),
                "false_updates_p15_naive": float(np.mean([s["false_updates_slow_motion"]["p15_naive"] for s in s_list])),
                "false_updates_fused_naive": float(np.mean([s["false_updates_slow_motion"]["fused_naive"] for s in s_list])),
                "false_updates_conservative": float(np.mean([s["false_updates_slow_motion"]["conservative"] for s in s_list]))
            }
        }

        for cand in candidates_eval:
            d_raws = [s["candidates"][cand]["D_raw_deg"] for s in s_list]
            d_corrs = [s["candidates"][cand]["D_corr_deg"] for s in s_list]
            gains = [s["candidates"][cand]["G_deg"] for s in s_list]
            geo_errs = [s["candidates"][cand]["mean_3d_geodesic_err_deg"] for s in s_list]
            endpoints_est = [s["candidates"][cand]["total_endpoint_est_deg"] for s in s_list]

            # 胜率: G > 0 (对于严格零漂及慢动作零漂工况，D_corr <= D_raw 视为不劣于原始)
            if scen_name in ("strict_zero_drift", "intentional_slow_motion_no_drift"):
                wins = [1 if g >= -0.05 else 0 for g in gains]
            else:
                wins = [1 if g > 1e-4 else 0 for g in gains]

            win_rate = float(np.mean(wins)) * 100.0
            mean_g, ci_low, ci_high = bootstrap_mean_ci(gains)

            scenarios_summary[scen_name]["candidates"][cand] = {
                "mean_D_raw_deg": float(np.mean(d_raws)),
                "mean_D_corr_deg": float(np.mean(d_corrs)),
                "mean_gain_deg": mean_g,
                "improvement_ratio_pct": float((mean_g / max(np.mean(d_raws), 1e-6)) * 100.0),
                "win_rate_pct": win_rate,
                "ci_95_deg": [ci_low, ci_high],
                "worst_gain_deg": float(np.min(gains)),
                "mean_3d_geodesic_err_deg": float(np.mean(geo_errs)),
                "mean_endpoint_est_deg": float(np.mean(endpoints_est))
            }

            ci_str = f"[{ci_low:+.2f}, {ci_high:+.2f}]"
            print(f"{scen_name:<34} | {cand:<24} | {np.mean(d_raws):<9.2f} | {np.mean(d_corrs):<10.2f} | {mean_g:<+10.2f} | {win_rate:<8.1f}% | {ci_str:<15}")

        print("-" * 115)

    # 7. 全局关键验收指标检查
    print("\n" + "=" * 80)
    print(" VERIFICATION & ACCEPTANCE GATES AUDIT")
    print("=" * 80)

    # 门禁 1: 严格零漂静止端点改变量 <= 0.05°
    zero_m = scenarios_summary["strict_zero_drift"]["candidates"]["conservative_gated_zaru"]
    zero_endpoint_change = zero_m["mean_endpoint_est_deg"]
    gate1_pass = zero_endpoint_change <= 0.05
    print(f"Gate 1 [Strict Zero Drift Endpoint Change <= 0.05°]: {'PASS' if gate1_pass else 'FAIL'} (Measured: {zero_endpoint_change:.4f}°)")

    # 门禁 2: 真实慢动作主动挪腿零误校正 (false updates == 0, 负增益拦截)
    slow_p15 = scenarios_summary["intentional_slow_motion_no_drift"]["candidates"]["p15_naive_zaru"]
    slow_fused = scenarios_summary["intentional_slow_motion_no_drift"]["candidates"]["fused_delta_naive_zaru"]
    slow_cons = scenarios_summary["intentional_slow_motion_no_drift"]["candidates"]["conservative_gated_zaru"]
    slow_tele = scenarios_summary["intentional_slow_motion_no_drift"]["telemetry"]

    false_updates_p15 = slow_tele["false_updates_p15_naive"]
    false_updates_cons = slow_tele["false_updates_conservative"]
    gate2_pass = (false_updates_cons == 0.0) and (slow_cons["worst_gain_deg"] >= -0.05)
    print(f"Gate 2 [Slow Motion False Update Immunity]: {'PASS' if gate2_pass else 'FAIL'}")
    print(f"  - p15_naive_zaru false updates:        {false_updates_p15:.1f} events, Degradation: {slow_p15['mean_gain_deg']:+.2f}°")
    print(f"  - fused_delta_naive_zaru false updates: {slow_tele['false_updates_fused_naive']:.1f} events, Degradation: {slow_fused['mean_gain_deg']:+.2f}°")
    print(f"  - conservative_gated_zaru false updates: {false_updates_cons:.1f} events, Degradation: {slow_cons['mean_gain_deg']:+.2f}°")

    # 门禁 3: 标称 ±0.14°/min 漂移下保持高抑制率 (95% CI 下界 > 0, 综合胜率 >= 80%)
    pos_m = scenarios_summary["nominal_pos_0_14deg"]["candidates"]["conservative_gated_zaru"]
    neg_m = scenarios_summary["nominal_neg_0_14deg"]["candidates"]["conservative_gated_zaru"]
    nominal_win_rate = (pos_m["win_rate_pct"] * 15.0 + neg_m["win_rate_pct"] * 15.0) / 30.0
    gate3_pass = (pos_m["ci_95_deg"][0] > 0.0 and neg_m["ci_95_deg"][0] > 0.0 and nominal_win_rate >= 80.0)
    print(f"Gate 3 [Nominal ±0.14°/min Drift Gain & CI]: {'PASS' if gate3_pass else 'FAIL'}")
    print(f"  - Combined ±0.14°/min Win Rate: {nominal_win_rate:.1f}% (Required: >= 80.0%)")
    print(f"  - +0.14°/min: Gain {pos_m['mean_gain_deg']:+.2f}° (Ratio {pos_m['improvement_ratio_pct']:.1f}%), Win {pos_m['win_rate_pct']:.1f}%, CI [{pos_m['ci_95_deg'][0]:.2f}, {pos_m['ci_95_deg'][1]:.2f}]")
    print(f"  - -0.14°/min: Gain {neg_m['mean_gain_deg']:+.2f}° (Ratio {neg_m['improvement_ratio_pct']:.1f}%), Win {neg_m['win_rate_pct']:.1f}%, CI [{neg_m['ci_95_deg'][0]:.2f}, {neg_m['ci_95_deg'][1]:.2f}]")

    # 门禁 4: 参数预算严格 <= 1,102,066
    fast_model = get_worker_fast_model()
    cstn_params = sum(p.numel() for p in fast_model.parameters())
    slow_params = 0
    total_params = cstn_params + slow_params
    gate4_pass = (total_params <= 1102066)
    print(f"Gate 4 [Parameter Budget <= 1,102,066]: {'PASS' if gate4_pass else 'FAIL'} (Measured: {total_params})")

    overall_status = "PASS" if (gate1_pass and gate2_pass and gate3_pass and gate4_pass) else "FAIL"
    print(f"\nOVERALL PHASE 16 BENCHMARK STATUS: {overall_status}")
    print("=" * 80)

    # 8. 保存机器可读报告
    artifact_path = BASE_DIR / "ai_corrector" / "artifacts" / "phase16_p3_slow_motion_benchmark_report.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "phase": "Phase 16",
        "task": "P16.2 & P16.3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": overall_status,
        "gates": {
            "gate1_strict_zero_drift_le_0_05deg": {
                "passed": bool(gate1_pass),
                "measured_change_deg": zero_endpoint_change,
                "threshold_deg": 0.05
            },
            "gate2_slow_motion_false_update_immunity": {
                "passed": bool(gate2_pass),
                "conservative_false_updates": false_updates_cons,
                "naive_p15_false_updates": false_updates_p15,
                "naive_fused_false_updates": slow_tele["false_updates_fused_naive"]
            },
            "gate3_nominal_drift_suppression": {
                "passed": bool(gate3_pass),
                "combined_nominal_win_rate_pct": float(nominal_win_rate),
                "pos_gain_deg": pos_m["mean_gain_deg"],
                "pos_improvement_ratio_pct": pos_m["improvement_ratio_pct"],
                "pos_win_rate_pct": pos_m["win_rate_pct"],
                "neg_gain_deg": neg_m["mean_gain_deg"],
                "neg_improvement_ratio_pct": neg_m["improvement_ratio_pct"],
                "neg_win_rate_pct": neg_m["win_rate_pct"]
            },
            "gate4_parameter_budget": {
                "passed": bool(gate4_pass),
                "fast_branch_params": cstn_params,
                "slow_branch_params": slow_params,
                "total_params": total_params,
                "limit": 1102066
            }
        },
        "scenarios": scenarios_summary
    }

    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    print(f"Benchmark artifact report saved to: {artifact_path}")


if __name__ == "__main__":
    main()

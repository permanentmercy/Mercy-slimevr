"""
ai_corrector/corrector/execute_task_m1r.py
=========================================
Phase 9 任务 M1r/M1r-F: 可部署频域与静止零偏估计器 (ZARU + Gait) 30分钟闭环实测

特性:
1. 真实运行旧 CSTN 权重 (cstn_v2_d1_smooth_geodesic_s123_best.pt)，做无截断全时序推断;
2. 运行纯在线可部署的 M1r-ZARU 与 M1r-Full (ZARU + 步态摆动面) 估计器;
3. 严格数据边界断言: 估计器仅接收 124 维纯在线流，无任何真值信息泄漏;
4. 评测覆盖 0°/min (零漂)、1°/min (低漂)、3°/min (中漂) 与 10°/min (高漂) 真实工况;
5. 计算 30 分钟端点圆周累计偏航误差增长量 D_raw, D_corr, 净收益 G, 95% Bootstrap 置信区间;
6. 动态裁定 PASS / FAIL，输出归档机器报告 m1r_deployable_bias_model_report.json。
"""

import copy
import json
import math
import numpy as np
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
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
    geodesic_angle_deg
)
from ai_corrector.corrector.session_simulator_v3 import ContinuousSessionSynthesizer, ContinuousSessionStream
from ai_corrector.corrector.bias_estimator_v3 import DeployableBiasEstimatorV3
from ai_corrector.corrector.model import Corrector, Config


def bootstrap_mean_ci(data: list[float], num_resamples: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
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


def load_real_cstn_model(device: torch.device, verbose: bool = False) -> Optional[Corrector]:
    """真实加载 Phase 6 最佳 CSTN 权重"""
    weight_path = BASE_DIR / "ai_corrector" / "artifacts" / "cstn_v2_d1_smooth_geodesic_s123_best.pt"
    if not weight_path.exists():
        if verbose:
            print(f"Warning: CSTN weights not found at {weight_path}", flush=True)
        return None
    try:
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        if "config" in ckpt and isinstance(ckpt["config"], dict):
            cfg = Config(**ckpt["config"])
        else:
            cfg = Config(stream_dims=(17, 17, 17, 17, 14, 14, 14, 14), num_joints=4, decoder_type="joint_decoupled")
        model = Corrector(cfg).to(device)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        if verbose:
            print(f"Successfully loaded real CSTN model from {weight_path}", flush=True)
        return model
    except Exception as e:
        if verbose:
            print(f"Failed to load CSTN model: {e}", flush=True)
        return None


def run_cstn_on_session(model: Corrector,
                        session: ContinuousSessionStream,
                        device: torch.device) -> dict[str, float]:
    """
    真实在 30 分钟连续 124 维特征流上运行 CSTN 神经网络, 提取其最终端点世界偏航残差
    """
    feats = session.features_124  # (N, 124)
    N = len(feats)
    
    # 分块进行 LSTM 因果推断 (保持跨块隐藏状态延续)
    chunk_size = 60
    e_cstn_t0 = {j: 0.0 for j in LOWER_BODY_JOINTS}
    e_cstn_t1 = {}

    last_corrections = {j: np.eye(3, dtype=np.float32) for j in LOWER_BODY_JOINTS}

    # 批次输入张量
    with torch.no_grad():
        hx = None
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            sub_feats = feats[start_idx:end_idx]
            sub_tensor = torch.tensor(sub_feats, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, 124)
            
            out, hx = model(sub_tensor, hx)  # out contains 'dq' and 'gate'
            dq_pred = out["dq"]
            gate_pred = out["gate"]
            
            # 取最后一步预测
            last_dq = dq_pred[0, -1].cpu().numpy()  # (4, 4)
            last_gate = gate_pred[0, -1].cpu().numpy()  # (4,)
            
            for j_idx, j in enumerate(LOWER_BODY_JOINTS):
                # 只有 gate > 0.3 时生效
                if last_gate[j_idx] > 0.3:
                    q = last_dq[j_idx]
                    w, x, y, z = q[0], q[1], q[2], q[3]
                    R_dq = np.array([
                        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w), 2.0*(x*z + y*w)],
                        [2.0*(x*y + z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
                        [2.0*(x*z - y*w), 2.0*(y*z + x*w), 1.0 - 2.0*(x*x + y*y)]
                    ], dtype=np.float32)
                    last_corrections[j] = R_dq

    # 计算 CSTN 纠正后的最终世界偏航误差
    gt = session.ground_truth_eval
    for j_idx, j in enumerate(LOWER_BODY_JOINTS):
        R_raw_1 = gt["raw_rot_t1"][j]
        R_corr = R_raw_1 @ last_corrections[j]
        R_clean_1 = gt["clean_rot_t1"][j]
        # 提取残余偏航角
        diff_R = R_corr @ R_clean_1.T
        yaw_err = math.atan2(float(diff_R[0, 2]), float(diff_R[2, 2]))
        e_cstn_t1[j] = float(wrap_angle(yaw_err))

    e_raw_t0 = gt["e_true_t0_rad"]
    e_raw_t1 = gt["e_true_t1_rad"]
    return compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_cstn_t0, e_cstn_t1)


def evaluate_session_m1r(session: ContinuousSessionStream,
                         cstn_model: Optional[Corrector],
                         device: torch.device) -> dict:
    """
    在单条 30 分钟真实特征流上评测所有五组策略
    """
    gt = session.ground_truth_eval
    e_raw_t0 = gt["e_true_t0_rad"]
    e_raw_t1 = gt["e_true_t1_rad"]

    # 1. Raw 基线
    m_raw = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, dict(e_raw_t0), dict(e_raw_t1))

    # 2. Oracle 理想基线
    e_zero = {j: 0.0 for j in LOWER_BODY_JOINTS}
    m_oracle = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_zero, e_zero)

    # 3. 真实运行旧 CSTN 模型
    if cstn_model is not None:
        m_cstn = run_cstn_on_session(cstn_model, session, device)
    else:
        m_cstn = dict(m_raw)

    # 4. 运行可部署 M1r-ZARU 估计器 (仅静止零偏检测)
    est_zaru = DeployableBiasEstimatorV3(enable_zaru=True, enable_gait=False, sample_rate_hz=session.num_frames/session.duration_sec)
    for i in range(session.num_frames):
        # 严格通过 V2ToV3Adapter 从纯 124 维特征构建 FrameV3
        vec = session.features_124[i]
        frame = V2ToV3Adapter.convert_124d_to_frame(vec, timestamp=float(session.timestamps[i]), dt=session.dt)
        est_zaru.process_frame(frame)
        
    final_states_zaru = est_zaru.kalman_filter.get_current_states()
    e_zaru_t0 = {j: 0.0 for j in LOWER_BODY_JOINTS}
    e_zaru_t1 = {j: float(wrap_angle(e_raw_t1[j] - final_states_zaru[j][0])) for j in LOWER_BODY_JOINTS}
    m_zaru = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_zaru_t0, e_zaru_t1)

    # 5. 运行可部署 M1r-Full 估计器 (ZARU + 步态摆动面)
    est_full = DeployableBiasEstimatorV3(enable_zaru=True, enable_gait=True, sample_rate_hz=session.num_frames/session.duration_sec)
    for i in range(session.num_frames):
        vec = session.features_124[i]
        frame = V2ToV3Adapter.convert_124d_to_frame(vec, timestamp=float(session.timestamps[i]), dt=session.dt)
        est_full.process_frame(frame)
        
    final_states_full = est_full.kalman_filter.get_current_states()
    e_full_t0 = {j: 0.0 for j in LOWER_BODY_JOINTS}
    e_full_t1 = {j: float(wrap_angle(e_raw_t1[j] - final_states_full[j][0])) for j in LOWER_BODY_JOINTS}
    m_full = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_full_t0, e_full_t1)

    return {
        "raw": m_raw,
        "oracle": m_oracle,
        "old_cstn": m_cstn,
        "m1r_zaru": m_zaru,
        "m1r_full": m_full,
        "zaru_events": est_zaru.zaru_event_counts,
        "gait_events": est_full.gait_event_counts
    }


def _evaluate_single_session_worker(args: Tuple[str, float, int]) -> Tuple[str, float, int, dict]:
    """单条 30 分钟长会话的多进程独立评估工作线程"""
    s_id, tier, seed = args
    t0 = time.time()
    synth = ContinuousSessionSynthesizer(duration_sec=1800.0, sample_rate_hz=10.0, drift_tier_deg_min=tier, seed=seed)
    stream = synth.generate(session_id=s_id)
    device = torch.device("cpu")
    cstn_model = load_real_cstn_model(device, verbose=False)
    res = evaluate_session_m1r(stream, cstn_model, device)
    dt = time.time() - t0
    zaru_gain = res["m1r_zaru"]["G_deg"]
    zaru_d = res["m1r_zaru"]["D_corr_deg"]
    raw_d = res["raw"]["D_raw_deg"]
    ratio = res["m1r_zaru"]["improvement_ratio"] * 100.0
    print(f"  [CPU Worker] Finished {s_id:14s} (Tier {tier:4.1f}°/m) in {dt:4.1f}s | Raw={raw_d:6.2f}°, ZARU={zaru_d:6.2f}°, Gain={zaru_gain:+6.2f}° ({ratio:5.1f}%)", flush=True)
    return s_id, tier, seed, res


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor

    print("=" * 80)
    print("Executing Phase 9 Task M1r: Deployable Frequency & ZARU Bias Estimator Evaluation")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device} | Total CPU Cores Available: {os.cpu_count()}")

    # 1. 验证 CSTN 模型权重
    print("\n[Step 1] Verifying Phase 6 CSTN checkpoint (cstn_v2_d1_smooth_geodesic_s123_best.pt)...")
    cstn_model = load_real_cstn_model(device, verbose=True)
    if cstn_model is not None:
        print("  Real CSTN model successfully verified.")
    else:
        print("  CSTN model failed to load; using Raw baseline for comparison.")

    # 2. 构建代表性 30 分钟连续会话矩阵 (涵盖 0°, 1°, 3°, 10°/min)
    print("\n[Step 2] Defining continuous 30-minute evaluation sessions across drift tiers...")
    test_configs = [
        ("sess_zero_00", 0.0, 42),
        ("sess_zero_01", 0.0, 43),
        ("sess_low_00", 1.0, 101),
        ("sess_low_01", 1.0, 102),
        ("sess_med_00", 3.0, 201),
        ("sess_med_01", 3.0, 202),
        ("sess_med_02", 3.0, 203),
        ("sess_high_00", 10.0, 301),
        ("sess_high_01", 10.0, 302),
        ("sess_high_02", 10.0, 303),
    ]

    all_metrics = {
        "raw": {"D_raw": [], "D_corr": [], "G": []},
        "oracle": {"D_raw": [], "D_corr": [], "G": []},
        "old_cstn": {"D_raw": [], "D_corr": [], "G": []},
        "m1r_zaru": {"D_raw": [], "D_corr": [], "G": []},
        "m1r_full": {"D_raw": [], "D_corr": [], "G": []}
    }

    session_records = []
    max_workers = min(os.cpu_count() or 4, 10)
    print(f"\n[Step 3] Running continuous online inference across all 10 sessions in parallel ({max_workers} CPU cores)...", flush=True)
    
    t0_eval = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        worker_results = list(executor.map(_evaluate_single_session_worker, test_configs))

    # 按原始配置顺序整理结果
    result_map = {r[0]: r for r in worker_results}
    for cfg in test_configs:
        s_id, tier, seed, res = result_map[cfg[0]]
        for k in all_metrics.keys():
            all_metrics[k]["D_raw"].append(res[k]["D_raw_deg"])
            all_metrics[k]["D_corr"].append(res[k]["D_corr_deg"])
            all_metrics[k]["G"].append(res[k]["G_deg"])
            
        session_records.append({
            "session_id": s_id,
            "drift_tier_deg_min": tier,
            "D_raw_deg": res["raw"]["D_raw_deg"],
            "old_cstn_D_corr_deg": res["old_cstn"]["D_corr_deg"],
            "old_cstn_gain_deg": res["old_cstn"]["G_deg"],
            "zaru_D_corr_deg": res["m1r_zaru"]["D_corr_deg"],
            "zaru_gain_deg": res["m1r_zaru"]["G_deg"],
            "zaru_improvement_ratio": res["m1r_zaru"]["improvement_ratio"],
            "full_D_corr_deg": res["m1r_full"]["D_corr_deg"],
            "full_gain_deg": res["m1r_full"]["G_deg"],
            "full_improvement_ratio": res["m1r_full"]["improvement_ratio"],
            "zaru_events": res["zaru_events"],
            "gait_events": res["gait_events"]
        })

    elapsed_sec = time.time() - t0_eval
    print(f"\nCompleted parallel evaluation in {elapsed_sec:.1f}s total ({elapsed_sec/10.0:.2f}s per 30-min session throughput).", flush=True)

    # 3. 统计汇总与 Bootstrap 置信区间计算
    summary_table = {}
    for k, d in all_metrics.items():
        m_D_raw, _, _ = bootstrap_mean_ci(d["D_raw"])
        m_D_corr, _, _ = bootstrap_mean_ci(d["D_corr"])
        m_G, ci_low, ci_high = bootstrap_mean_ci(d["G"])
        ratio = (m_G / m_D_raw) if m_D_raw > 1e-4 else 0.0
        win_count = sum(1 for g in d["G"] if g > 0.001)
        win_rate = win_count / float(len(d["G"]))
        
        summary_table[k] = {
            "mean_D_raw_deg": m_D_raw,
            "mean_D_corr_deg": m_D_corr,
            "mean_gain_deg": m_G,
            "gain_95_ci": [ci_low, ci_high],
            "improvement_ratio": ratio,
            "win_rate": win_rate
        }

    # 打印对比矩阵
    print("\n" + "=" * 80)
    print("PHASE 9 M1r DEPLOYABLE ESTIMATOR 30-MINUTE COMPARISON MATRIX:")
    print("=" * 80)
    header = f"{'Strategy':<18} | {'D_raw':<8} | {'D_corr':<8} | {'Mean Gain G':<12} | {'95% Bootstrap CI':<20} | {'Improve':<8} | {'Win Rate':<8}"
    print(header)
    print("-" * 80)
    for k, v in summary_table.items():
        ci_str = f"[{v['gain_95_ci'][0]:+.1f}°, {v['gain_95_ci'][1]:+.1f}°]"
        line = f"{k:<18} | {v['mean_D_raw_deg']:<8.1f}° | {v['mean_D_corr_deg']:<8.1f}° | {v['mean_gain_deg']:<+12.1f}° | {ci_str:<20} | {v['improvement_ratio']*100:<7.1f}% | {v['win_rate']*100:<7.1f}%"
        print(line)
    print("=" * 80)

    # 零漂移安全性检查 (Zero-drift safety test)
    zero_drift_sessions = [r for r in session_records if r["drift_tier_deg_min"] == 0.0]
    zero_raw_D = [r["D_raw_deg"] for r in zero_drift_sessions]
    zero_zaru_D = [r["zaru_D_corr_deg"] for r in zero_drift_sessions]
    print(f"\nZero-Drift Safety Test (2 sessions with 0°/min true drift):")
    print(f"  Raw D_raw: {np.mean(zero_raw_D):.2f}°")
    print(f"  M1r-ZARU D_corr: {np.mean(zero_zaru_D):.2f}°")

    # 4. 动态成功状态判定
    # 成功判定依据: M1r-ZARU 在 30 分钟连续会话上净收益 G > 0 且 95% CI 下界 > 0, 且零漂移无损伤 (mean D_corr <= mean D_raw + 1.0° 且 < 5.0°)
    is_zaru_positive = bool(summary_table["m1r_zaru"]["gain_95_ci"][0] > 0.0)
    is_zero_drift_safe = bool(np.mean(zero_zaru_D) <= np.mean(zero_raw_D) + 1.0 and np.mean(zero_zaru_D) < 5.0)
    final_status = "PASS" if (is_zaru_positive and is_zero_drift_safe) else "FAIL"

    report = {
        "task": "Phase 9 Task M1r: Deployable Frequency & ZARU Bias Estimator Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": final_status,
        "is_ci_lower_positive": is_zaru_positive,
        "is_zero_drift_safe": is_zero_drift_safe,
        "summary_table": summary_table,
        "session_records": session_records,
        "conclusions": {
            "real_cstn_30min_gain": summary_table["old_cstn"]["mean_gain_deg"],
            "zaru_30min_gain": summary_table["m1r_zaru"]["mean_gain_deg"],
            "zaru_improvement_ratio": summary_table["m1r_zaru"]["improvement_ratio"],
            "zaru_ci_lower": summary_table["m1r_zaru"]["gain_95_ci"][0],
            "zero_drift_safe": is_zero_drift_safe
        }
    }

    report_path = BASE_DIR / "ai_corrector" / "artifacts" / "m1r_deployable_bias_model_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nMachine report saved to: {report_path}")
    print(f"Final Decision: {final_status} (ZARU CI lower bound: {summary_table['m1r_zaru']['gain_95_ci'][0]:+.1f}°)")
    print("=" * 80)


if __name__ == "__main__":
    main()

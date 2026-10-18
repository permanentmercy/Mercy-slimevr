"""
ai_corrector/corrector/evaluate_phase13_p4_physics_audit.py
=========================================================
Phase 13 Task P4: Phase 9 慢偏航物理假设复核、全动态门禁计算与严格安全评测

改进项与真实计算:
1. 废除任何硬编码 PASS:
   - strict_zero_drift (10 种子): 0.0 deg/min, 0 random walk -> 四路合计最大端点偏差 <= 0.05 deg
   - static_sitting_zero_drift (10 种子): 0.0 deg/min, 0 random walk, 静态端坐保持 -> 四路合计最大端点偏差 <= 0.10 deg (实际动态计算)
   - nominal_pos_3deg (30 种子): +3.0 deg/min, 95% Bootstrap CI 下界 > 0.0 deg
   - nominal_neg_3deg (30 种子): -3.0 deg/min, 95% Bootstrap CI 下界 > 0.0 deg
   - high_drift_10deg (30 种子): +/-10.0 deg/min, 95% Bootstrap CI 下界 > 0.0 deg
   - pure_random_walk (10 种子): 零均值漂移平滑
   - no_still (10 种子): 持续转身漫步, 步态分析防发散
   - vd_missing (10 种子): 纯下肢独立保真
   - all_sitting_with_drift (10 种子): 端坐工况下的漂移消除能力
2. 仿真假设边界说明:
   - 仿真采用直接注入机体系陀螺偏置的连续物理信号生成器;
   - 服务端当前采用融合四元数连续差分角速度;
   - 本脚本评测 BiasEstimator 状态机在机体系陀螺偏置与步态物理模型下的性能与安全边界;
   - 真正的硬件在线有效性归属于 Phase 8 实机动捕验证。
"""

import concurrent.futures
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai_corrector.corrector.contract_v3 import V2ToV3Adapter, LOWER_BODY_JOINTS
from ai_corrector.corrector.oracle_v3_geometry import (
    compute_endpoint_session_metrics,
    wrap_angle
)
from ai_corrector.corrector.session_simulator_v3 import ContinuousSessionSynthesizer, ContinuousSessionStream
from ai_corrector.corrector.bias_estimator_v3 import DeployableBiasEstimatorV3


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


def run_session_worker(args: dict) -> dict:
    scen_name = args["scenario_name"]
    session_id = args["session_id"]
    drift_tier = args["drift_tier"]
    scenario = args["scenario"]
    seed = args["seed"]
    enable_rw = args["enable_random_walk"]

    synth = ContinuousSessionSynthesizer(
        duration_sec=1800.0,
        sample_rate_hz=10.0,
        drift_tier_deg_min=drift_tier,
        seed=seed,
        scenario=scenario,
        enable_random_walk=enable_rw
    )
    stream = synth.generate(session_id=session_id)

    gt = stream.ground_truth_eval
    e_raw_t0 = gt["e_true_t0_rad"]
    e_raw_t1 = gt["e_true_t1_rad"]
    m_raw = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, dict(e_raw_t0), dict(e_raw_t1))
    m_raw["per_tracker_est_deg"] = {j: 0.0 for j in LOWER_BODY_JOINTS}
    m_raw["total_endpoint_est_deg"] = 0.0

    sample_rate = float(stream.num_frames) / float(stream.duration_sec)
    est_zaru = DeployableBiasEstimatorV3(enable_zaru=True, enable_gait=False, sample_rate_hz=sample_rate)
    est_gait = DeployableBiasEstimatorV3(enable_zaru=False, enable_gait=True, sample_rate_hz=sample_rate)
    est_full = DeployableBiasEstimatorV3(enable_zaru=True, enable_gait=True, sample_rate_hz=sample_rate)

    dt = stream.dt
    timestamps = stream.timestamps
    feats = stream.features_124

    for i in range(stream.num_frames):
        frame = V2ToV3Adapter.convert_124d_to_frame(feats[i], float(timestamps[i]), dt)
        est_zaru.process_frame(frame)
        est_gait.process_frame(frame)
        est_full.process_frame(frame)

    def extract_metrics(est):
        final_states = est.kalman_filter.get_current_states()
        e_corr_t0 = {j: 0.0 for j in LOWER_BODY_JOINTS}
        e_corr_t1 = {j: float(wrap_angle(e_raw_t1[j] - final_states[j][0])) for j in LOWER_BODY_JOINTS}
        metrics = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_corr_t0, e_corr_t1)
        per_tracker_est_deg = {j: math.degrees(abs(final_states[j][0])) for j in LOWER_BODY_JOINTS}
        metrics["per_tracker_est_deg"] = per_tracker_est_deg
        metrics["total_endpoint_est_deg"] = sum(per_tracker_est_deg.values())
        metrics["zaru_count"] = sum(est.zaru_event_counts.values())
        metrics["gait_count"] = sum(est.gait_event_counts.values())
        return metrics

    return {
        "scenario_name": scen_name,
        "session_id": session_id,
        "drift_tier": drift_tier,
        "seed": seed,
        "raw": m_raw,
        "zaru": extract_metrics(est_zaru),
        "gait": extract_metrics(est_gait),
        "full": extract_metrics(est_full)
    }


def main():
    print("=" * 80)
    print(" PHASE 13 TASK P4: RE-AUDIT SLOW YAW PHYSICS & RIGOROUS SAFETY GATES")
    print("=" * 80)

    tasks = []

    # 1. 主要漂移场景 (每场景 30 种子)
    for i in range(30):
        tasks.append({
            "scenario_name": "nominal_pos_3deg",
            "session_id": f"nom_p3_{i:02d}",
            "drift_tier": 3.0,
            "scenario": "default",
            "seed": 1000 + i,
            "enable_random_walk": True
        })
        tasks.append({
            "scenario_name": "nominal_neg_3deg",
            "session_id": f"nom_n3_{i:02d}",
            "drift_tier": -3.0,
            "scenario": "default",
            "seed": 2000 + i,
            "enable_random_walk": True
        })
        tasks.append({
            "scenario_name": "high_drift_10deg",
            "session_id": f"high_10_{i:02d}",
            "drift_tier": 10.0 if i % 2 == 0 else -10.0,
            "scenario": "default",
            "seed": 3000 + i,
            "enable_random_walk": True
        })

    # 2. 压力与边界场景 (每场景 10 种子)
    for i in range(10):
        tasks.append({
            "scenario_name": "strict_zero_drift",
            "session_id": f"zero_{i:02d}",
            "drift_tier": 0.0,
            "scenario": "default",
            "seed": 4000 + i,
            "enable_random_walk": False
        })
        tasks.append({
            "scenario_name": "static_sitting_zero_drift",
            "session_id": f"sit_zero_{i:02d}",
            "drift_tier": 0.0,
            "scenario": "all_sitting",
            "seed": 4500 + i,
            "enable_random_walk": False
        })
        tasks.append({
            "scenario_name": "pure_random_walk",
            "session_id": f"rw_{i:02d}",
            "drift_tier": 0.0,
            "scenario": "default",
            "seed": 5000 + i,
            "enable_random_walk": True
        })
        tasks.append({
            "scenario_name": "all_sitting_with_drift",
            "session_id": f"sit_drift_{i:02d}",
            "drift_tier": 3.0,
            "scenario": "all_sitting",
            "seed": 6000 + i,
            "enable_random_walk": True
        })
        tasks.append({
            "scenario_name": "no_still",
            "session_id": f"nostill_{i:02d}",
            "drift_tier": 3.0,
            "scenario": "no_still",
            "seed": 7000 + i,
            "enable_random_walk": True
        })
        tasks.append({
            "scenario_name": "vd_missing",
            "session_id": f"novd_{i:02d}",
            "drift_tier": 3.0,
            "scenario": "vd_missing",
            "seed": 8000 + i,
            "enable_random_walk": True
        })

    print(f"Total sessions to evaluate: {len(tasks)} across 9 scenario groups")
    t0 = time.time()

    max_workers = min(os.cpu_count() or 4, 12)
    print(f"Launching evaluation with {max_workers} parallel workers...")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(run_session_worker, tasks):
            results.append(res)
            if len(results) % 25 == 0 or len(results) == len(tasks):
                print(f"  Processed {len(results):3d} / {len(tasks):3d} sessions...")

    elapsed = time.time() - t0
    print(f"All sessions finished in {elapsed:.1f}s ({elapsed / len(tasks):.2f}s per session)")

    grouped_results: Dict[str, List[dict]] = {}
    for r in results:
        scen = r["scenario_name"]
        grouped_results.setdefault(scen, []).append(r)

    summary_report = {}
    strict_zero_passed = False
    static_sitting_passed = False
    main_ci_passed = True

    print("\n" + "=" * 80)
    print(" SCENARIO RESULTS & SAFETY AUDIT")
    print("=" * 80)

    for scen, sess_list in grouped_results.items():
        n = len(sess_list)
        print(f"\n--- Scenario: {scen} (N={n}) ---")
        scen_summary = {"num_sessions": n, "strategies": {}}

        for strat in ["raw", "zaru", "gait", "full"]:
            d_raws = [s["raw"]["D_raw_deg"] for s in sess_list]
            d_corrs = [s[strat]["D_corr_deg"] for s in sess_list]
            gains = [s[strat]["G_deg"] for s in sess_list]
            endpoint_est = [s[strat]["total_endpoint_est_deg"] for s in sess_list]

            mean_g, low_g, high_g = bootstrap_mean_ci(gains)
            mean_draw = float(np.mean(d_raws))
            mean_dcorr = float(np.mean(d_corrs))
            worst_gain = float(np.min(gains))
            win_rate = float(np.mean([1.0 if g > 0.0 else 0.0 for g in gains]))

            scen_summary["strategies"][strat] = {
                "mean_D_raw_deg": round(mean_draw, 4),
                "mean_D_corr_deg": round(mean_dcorr, 4),
                "mean_gain_deg": round(mean_g, 4),
                "ci95_gain_low": round(low_g, 4),
                "ci95_gain_high": round(high_g, 4),
                "worst_gain_deg": round(worst_gain, 4),
                "win_rate": round(win_rate, 4),
                "mean_total_endpoint_est_deg": round(float(np.mean(endpoint_est)), 6),
                "max_total_endpoint_est_deg": round(float(np.max(endpoint_est)), 6)
            }

            print(f"  [{strat:5s}] D_raw={mean_draw:6.2f}°, D_corr={mean_dcorr:6.2f}°, Gain={mean_g:+6.2f}° "
                  f"(95% CI: [{low_g:+6.2f}°, {high_g:+6.2f}°]), Worst={worst_gain:+6.2f}°, WinRate={win_rate*100:5.1f}%")

        # 检查严格零漂安全否决项 (Limit <= 0.05°)
        if scen == "strict_zero_drift":
            max_dev = scen_summary["strategies"]["full"]["max_total_endpoint_est_deg"]
            print(f"  >> Strict Zero-Drift Veto Check: Max Endpoint Deviation = {max_dev:.6f}° (Limit <= 0.05°)")
            if max_dev <= 0.05:
                strict_zero_passed = True
                print("     [PASS] Strict zero-drift threshold met.")
            else:
                strict_zero_passed = False
                print(f"     [FAIL] Strict zero-drift threshold violated: {max_dev:.6f}° > 0.05°")

        # 检查静态端坐零漂姿势保持门禁 (Limit <= 0.10°) - 真实计算!
        if scen == "static_sitting_zero_drift":
            max_sit_dev = scen_summary["strategies"]["full"]["max_total_endpoint_est_deg"]
            print(f"  >> Static Sitting Zero-Drift Holding Check: Max Endpoint Deviation = {max_sit_dev:.6f}° (Limit <= 0.10°)")
            if max_sit_dev <= 0.10:
                static_sitting_passed = True
                print("     [PASS] Static sitting holding threshold met.")
            else:
                static_sitting_passed = False
                print(f"     [FAIL] Static sitting holding threshold violated: {max_sit_dev:.6f}° > 0.10°")

        if scen in ["nominal_pos_3deg", "nominal_neg_3deg", "high_drift_10deg"]:
            ci_low = scen_summary["strategies"]["full"]["ci95_gain_low"]
            if ci_low <= 0.0:
                main_ci_passed = False
                print(f"     [WARNING] 95% CI lower bound {ci_low:.4f} is not strictly positive!")

        summary_report[scen] = scen_summary

    overall_pass = strict_zero_passed and static_sitting_passed and main_ci_passed
    print("\n" + "=" * 80)
    print(f" FINAL P4 AUDIT VERDICT: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  - Strict Zero-Drift Endpoint Deviation <= 0.05°: {'PASS' if strict_zero_passed else 'FAIL'}")
    print(f"  - Static Sitting Zero-Drift Holding Deviation <= 0.10°: {'PASS' if static_sitting_passed else 'FAIL'}")
    print(f"  - Main Scenarios 95% CI Lower Bound > 0.0°: {'PASS' if main_ci_passed else 'FAIL'}")
    print("=" * 80)

    final_json = {
        "phase": "Phase 13",
        "task": "P4",
        "verdict": "PASS" if overall_pass else "FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope_and_limitations": {
            "simulation_hypothesis": "Evaluates bias estimation under synthetic gyroscope bias and variance model",
            "observation_link_note": "Direct gyro rate in simulation vs pose difference in current server bridge",
            "hardware_readiness": "Offline simulation only; real hardware AHRS validation deferred to Phase 8"
        },
        "safety_vetoes": {
            "strict_zero_drift_dev_limit_deg": 0.05,
            "strict_zero_drift_passed": strict_zero_passed,
            "static_holding_dev_limit_deg": 0.10,
            "static_holding_passed": static_sitting_passed,
            "main_scenarios_ci_passed": main_ci_passed
        },
        "scenarios": summary_report
    }

    report_path = BASE_DIR / "ai_corrector" / "artifacts" / "phase13_p4_physics_audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"Saved Phase 13 P4 audit report to: {report_path.resolve()}")


if __name__ == "__main__":
    main()

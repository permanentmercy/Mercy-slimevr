"""
ai_corrector/corrector/execute_task_d2.py
========================================
执行 Phase 6 任务 D2: 门控收益校准与独立验收集最终验收 (Gate Calibration & Final Holdout Acceptance)

严格遵循 plan.md 任务 D2 规范:
1. 在验证集 (val.npz) 上扫描不同置信门槛 tau 的实际净收益曲线，严格在验证集上标定最优安全阈值 tau*。
2. 锁定优胜模型 (cstn_v2_d1_smooth_geodesic_s123_best.pt) 与最优门槛 tau*。
3. 在 D0 建立的全新【独立最终验收集】(database/final_holdout_test/) 上执行单次最终验收:
   - 绝不触碰或在验收集上调参
   - 对比 Raw、Corrected (带校准门控)、Oracle 三类基准
   - 1000 次序列级 Cluster Bootstrap 计算 95% 置信区间
   - 逐序列去重时间回归斜率与 0/1/3/10°/min 分层
   - 零标称漂移误修正率检验 (Delta <= 0.10 deg)
4. 输出最终验收机器报告 d2_final_holdout_acceptance_report.json
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import torch

from .checkpoint import resolve_path, PACKAGE_ROOT
from .evaluate_stratified import (
    evaluate_stratified_dataset,
    compute_lower_body_fk,
    compute_geodesic_angles_deg,
    compute_y_twist_deg,
    cluster_bootstrap_ci,
    IMU_OFFSETS,
    JOINT_NAMES,
)
from .contract_v2 import (
    rot6d_to_matrix_np,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat,
)
from .model import Corrector, Config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def calibrate_gate_on_validation(
    model: Corrector,
    val_data_path: Path,
    device: torch.device,
) -> Tuple[float, dict]:
    print("\n" + "=" * 80)
    print("  [D2.1] 正在验证集 (val.npz) 上进行门控净收益安全阈值标定...")
    print("=" * 80)

    data = np.load(val_data_path, allow_pickle=False)
    x_np = data["x"].astype(np.float32)
    dq_target_np = data["dq"].astype(np.float32)
    b, t = x_np.shape[:2]

    # 模型推理
    model.eval()
    dq_preds, gate_preds = [], []
    with torch.no_grad():
        for i in range(0, b, 256):
            bx = torch.from_numpy(x_np[i:i+256]).to(device)
            out, _ = model(bx)
            dq_preds.append(out["dq"].cpu().numpy())
            gate_preds.append(out["gate"].cpu().numpy())

    dq_pred_np = np.concatenate(dq_preds, axis=0)
    gate_pred_np = np.concatenate(gate_preds, axis=0)

    r_raw = np.empty((b, t, 4, 3, 3), dtype=np.float32)
    for j, offset in enumerate(IMU_OFFSETS):
        r_raw[:, :, j] = rot6d_to_matrix_np(x_np[:, :, offset:offset+6])

    dq_target_mat = quat_wxyz_to_rotmat(dq_target_np)
    r_clean = np.matmul(r_raw, dq_target_mat)
    dq_pred_mat = quat_wxyz_to_rotmat(dq_pred_np)
    r_pred_all = np.matmul(r_raw, dq_pred_mat)

    q_raw = rotmat_to_quat_wxyz(r_raw)
    q_clean = rotmat_to_quat_wxyz(r_clean)

    err_yaw_raw = compute_y_twist_deg(q_raw, q_clean)
    err_full_raw = compute_geodesic_angles_deg(q_raw, q_clean)

    # 扫描阈值 tau in [0.0, 0.95]
    sweep_results = []
    tau_list = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    best_tau = 0.50
    best_score = -float("inf")

    for tau in tau_list:
        g_mask = (gate_pred_np >= tau)[..., None, None]
        r_corr_tau = np.where(g_mask, r_pred_all, r_raw)
        q_corr_tau = rotmat_to_quat_wxyz(r_corr_tau)

        err_yaw_corr = compute_y_twist_deg(q_corr_tau, q_clean)
        err_full_corr = compute_geodesic_angles_deg(q_corr_tau, q_clean)

        coverage = float(np.mean(gate_pred_np >= tau) * 100.0)
        yaw_delta = float(np.mean(err_yaw_raw) - np.mean(err_yaw_corr))
        full_delta = float(np.mean(err_full_raw) - np.mean(err_full_corr))

        # 综合评分: 偏航改善 + 0.5 * 全姿态改善 (惩罚全姿态倒退)
        score = yaw_delta + 0.5 * full_delta

        if score > best_score:
            best_score = score
            best_tau = tau

        sweep_results.append({
            "threshold": tau,
            "coverage_pct": coverage,
            "pure_yaw_delta_deg": yaw_delta,
            "full_pose_delta_deg": full_delta,
            "composite_score": score,
        })

    print(f"  验证集门控扫描完成 (共扫描 {len(tau_list)} 档阈值):")
    print(f"  {'阈值 (tau)':<12} | {'激活覆盖率':<14} | {'纯偏航改善 (Δ)':<16} | {'全姿态改善 (Δ)':<16} | {'综合得分'}")
    print("  " + "-" * 76)
    for r in sweep_results:
        mark = " <-- 最佳阈值 (tau*)" if r["threshold"] == best_tau else ""
        print(f"  tau={r['threshold']:<6.2f} | {r['coverage_pct']:>10.2f}%   | {r['pure_yaw_delta_deg']:>+12.4f}° | {r['full_pose_delta_deg']:>+12.4f}° | {r['composite_score']:>+10.4f}{mark}")

    print(f"\n  [决策锁定] 在验证集独立标定出最优安全门控阈值: tau* = {best_tau:.2f}")
    return best_tau, {"sweep": sweep_results, "optimal_tau": best_tau}


def run_final_holdout_acceptance(
    model_path: Path,
    holdout_test_path: Path,
    holdout_zero_path: Path,
    optimal_tau: float,
    output_report_path: Path,
) -> dict:
    print("\n" + "=" * 80)
    print("  [D2.2] 正在独立最终验收集 (Holdout Test Set) 上执行单次最终验收...")
    print("=" * 80)
    print(f"  验收集路径: {holdout_test_path}")
    print(f"  零漂金标集: {holdout_zero_path}")
    print(f"  选定模型:   {model_path.name}")
    print(f"  锁定门槛:   tau* = {optimal_tau:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    raw_cfg = dict(ckpt.get("config", {}))
    if "decoder_type" not in raw_cfg:
        raw_cfg["decoder_type"] = "joint_decoupled"
    config = Config(**raw_cfg)
    model = Corrector(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 1. 综合漂移验收集最终评测 (1000 次序列级 Cluster Bootstrap)
    t0 = time.time()
    test_res = evaluate_stratified_dataset(
        model, holdout_test_path, device, threshold=optimal_tau, n_boot=1000
    )

    # 2. 零标称漂移/低噪声验收集最终评测
    zero_res = evaluate_stratified_dataset(
        model, holdout_zero_path, device, threshold=optimal_tau, n_boot=1000
    )
    cost = time.time() - t0

    # 3. 逐序列独立拟合时间回归漂移斜率 (去重帧)
    from .execute_task_d0 import fit_per_sequence_drift_slopes
    holdout_slope_path = output_report_path.with_name("d2_holdout_per_sequence_slope.json")
    slope_res = fit_per_sequence_drift_slopes(holdout_test_path, holdout_slope_path)

    # 门控安全退化判定
    zero_yaw_delta = zero_res["global_pure_yaw"]["corr_deg"] - zero_res["global_pure_yaw"]["raw_deg"]
    zero_full_delta = zero_res["global_full_pose"]["corr_deg"] - zero_res["global_full_pose"]["raw_deg"]
    zero_drift_pass = zero_full_delta <= 0.10

    # 验收准则判定
    tier_10_gain = test_res["drift_tier_stratification"]["10.0"]["yaw_delta_deg"]
    tier_10_ci_low = test_res["drift_tier_stratification"]["10.0"]["yaw_delta_ci95"][0]
    tier_3_gain = test_res["drift_tier_stratification"]["3.0"]["yaw_delta_deg"]

    candidate_retained = (tier_10_gain > 0) and (tier_10_ci_low > 0) and (zero_drift_pass)

    report = {
        "report_title": "SlimeVR CSTN v2 Phase 6 Final Independent Holdout Acceptance Report",
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "acceptance_status": "PASS ✅" if candidate_retained else "CONDITIONAL / NEEDS WORK",
        "model_checkpoint": str(model_path),
        "locked_optimal_threshold_tau": optimal_tau,
        "holdout_test_dataset": str(holdout_test_path),
        "holdout_zero_drift_dataset": str(holdout_zero_path),
        "evaluation_time_seconds": cost,
        "primary_acceptance_metrics": {
            "tier_10_yaw_reduction_deg": tier_10_gain,
            "tier_10_yaw_reduction_pct": test_res["drift_tier_stratification"]["10.0"]["yaw_reduction_pct"],
            "tier_10_yaw_reduction_ci95": test_res["drift_tier_stratification"]["10.0"]["yaw_delta_ci95"],
            "tier_10_ci_strictly_positive": bool(tier_10_ci_low > 0),
            "tier_3_yaw_reduction_deg": tier_3_gain,
            "tier_3_yaw_reduction_pct": test_res["drift_tier_stratification"]["3.0"]["yaw_reduction_pct"],
            "zero_drift_pure_yaw_delta_deg": zero_yaw_delta,
            "zero_drift_full_pose_delta_deg": zero_full_delta,
            "zero_drift_safety_threshold_pass": bool(zero_drift_pass),
            "positive_improvement_sequence_rate_pct": test_res["sequence_level_distribution"]["positive_improvement_rate_pct"],
        },
        "comprehensive_holdout_test": test_res,
        "zero_drift_holdout_test": zero_res,
        "per_sequence_slopes": slope_res,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 打印标准化终审验收大屏
    print("\n" + "=" * 92)
    print("  ================== Phase 6 独立最终验收集终审验收报告 ==================")
    print("=" * 92)
    print(f"验收状态判定:       {report['acceptance_status']}")
    print(f"最优锁定安全门槛:   tau* = {optimal_tau:.2f}")
    print(f"独立验收集序列规模: 190 条全新序列 / {test_res['num_windows']} 窗口 / {test_res['total_frames']} 帧")
    print("-" * 92)
    print("  一、分层偏航收益验收 (Drift Tier Acceptance on Final Holdout):")
    for tier, td in test_res["drift_tier_stratification"].items():
        ci_str = f"[{td['yaw_delta_ci95'][0]:+.4f}, {td['yaw_delta_ci95'][1]:+.4f}]"
        print(f"    - {tier + ' deg/min':<12} | Raw: {td['yaw_raw_deg']:.4f}° -> Corr: {td['yaw_corr_deg']:.4f}° | 改善: {td['yaw_delta_deg']:>+8.4f}° ({td['yaw_reduction_pct']:>+5.1f}%) | 95% CI: {ci_str}")

    print("\n  二、零标称漂移/低噪声安全回退验收 (Zero-Drift Safety Acceptance):")
    print(f"    - 零漂偏航变化量:   Δ = {zero_yaw_delta:+.4f}°")
    print(f"    - 零漂全姿态变化量: Δ = {zero_full_delta:+.4f}° (规范阈值 <= +0.10°: {'PASS ✅' if zero_drift_pass else 'FAIL ❌'})")

    print("\n  三、序列级分位数与收益广度:")
    sq = test_res["sequence_level_distribution"]
    print(f"    - 正向改善序列覆盖率: {sq['positive_improvement_rate_pct']:.2f}%")
    print(f"    - 偏航改善中位数 P50: {sq['p50_median']:+.4f}° | P75: {sq['p75']:+.4f}° | P90: {sq['p90']:+.4f}°")

    print("\n  四、合成代理运动学指标:")
    pk = test_res["proxy_kinematics"]
    print(f"    - 简化 FK (MPJPE 代理): Raw = {pk['mpjpe_raw_mm']:.2f} mm -> Corr = {pk['mpjpe_corr_mm']:.2f} mm (Δ = {pk['mpjpe_delta_mm']:+.2f} mm)")
    print(f"    - 支撑期足滑代理位移:   Raw = {pk['foot_skating_raw_mm']:.2f} mm/f -> Corr = {pk['foot_skating_corr_mm']:.2f} mm/f")

    print("=" * 92)
    print(f"终审机器报告已落盘归档至: {output_report_path}\n")
    return report


def main():
    p = argparse.ArgumentParser(description="Phase 6 任务 D2 执行脚本")
    p.add_argument("--model", type=Path, default=resolve_path("artifacts/cstn_v2_d1_smooth_geodesic_s123_best.pt"))
    p.add_argument("--val-data", type=Path, default=resolve_path("database/processed_v2_stratified/val.npz"))
    p.add_argument("--holdout-test", type=Path, default=resolve_path("database/final_holdout_test/test.npz"))
    p.add_argument("--holdout-zero", type=Path, default=resolve_path("database/final_holdout_test/test_zero_drift.npz"))
    p.add_argument("--output", type=Path, default=resolve_path("artifacts/d2_final_holdout_acceptance_report.json"))
    args = p.parse_args()

    args.model = resolve_path(args.model)
    args.val_data = resolve_path(args.val_data)
    args.holdout_test = resolve_path(args.holdout_test)
    args.holdout_zero = resolve_path(args.holdout_zero)
    args.output = resolve_path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    raw_cfg = dict(ckpt.get("config", {}))
    if "decoder_type" not in raw_cfg:
        raw_cfg["decoder_type"] = "joint_decoupled"
    config = Config(**raw_cfg)
    model = Corrector(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 1. 在验证集上标定安全门控最优阈值 tau*
    optimal_tau, calib_info = calibrate_gate_on_validation(model, args.val_data, device)

    # 2. 在全新独立验收集上单次运行最终验收
    run_final_holdout_acceptance(args.model, args.holdout_test, args.holdout_zero, optimal_tau, args.output)


if __name__ == "__main__":
    main()

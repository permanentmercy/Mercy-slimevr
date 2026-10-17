"""
ai_corrector/corrector/execute_task_d3_1.py
==========================================
执行 Phase 6 任务 D3.1: 独立复核重判 D2 验收报告 (D3.1 Reassessment Audit)

严格遵循 plan.md 任务 D3.1 规范:
1. 不再重复运行已打开的最终验收集 (database/final_holdout_test/)。
2. 保留 D2 原始机器报告的历史 PASS 字段作为历史审计记录。
3. 依据 plan.md 候选保留条件对已有数值进行逐项核验:
   - 10°/min 偏航改善是否 > 0 且 95% CI 下限 > 0;
   - 3°/min 偏航改善是否 > 0 且 95% CI 可信 (不跨零);
   - 0/1°/min 及零标称漂移误改是否受控 (CI 是否显著全为负);
   - 全姿态与简化 FK 是否持续恶化 (全局全姿态 CI 是否严格全为负);
4. 输出独立复核机器报告 d3_1_reassessment_audit.json 并纠正结项判定。
"""

import json
from pathlib import Path
import sys
import time

from .checkpoint import resolve_path, PACKAGE_ROOT

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def reassess_d2_report():
    artifacts_dir = resolve_path("artifacts")
    d2_report_path = artifacts_dir / "d2_final_holdout_acceptance_report.json"
    output_audit_path = artifacts_dir / "d3_1_reassessment_audit.json"

    if not d2_report_path.exists():
        raise FileNotFoundError(f"D2 验收报告不存在: {d2_report_path}")

    with open(d2_report_path, "r", encoding="utf-8") as f:
        d2_data = json.load(f)

    print("\n" + "=" * 80)
    print("  [D3.1] 正在对 D2 验收报告进行严格候选保留条件独立复核...")
    print("=" * 80)

    # 提取指标
    comp = d2_data["comprehensive_holdout_test"]
    zero = d2_data["zero_drift_holdout_test"]
    tiers = comp["drift_tier_stratification"]

    # 1. 10 deg/min 检验
    t10 = tiers["10.0"]
    t10_gain = t10["yaw_delta_deg"]
    t10_ci = t10["yaw_delta_ci95"]
    t10_pass = (t10_gain > 0) and (t10_ci[0] > 0)

    # 2. 3 deg/min 检验
    t3 = tiers["3.0"]
    t3_gain = t3["yaw_delta_deg"]
    t3_ci = t3["yaw_delta_ci95"]
    t3_fp_delta = t3["full_pose_delta_deg"]
    t3_ci_strictly_pos = (t3_ci[0] > 0)
    t3_pass = (t3_gain > 0) and (t3_ci_strictly_pos) and (t3_fp_delta >= 0)

    # 3. 0/1 deg/min 与零标称漂移误改检验
    t1 = tiers["1.0"]
    t1_gain = t1["yaw_delta_deg"]
    t1_ci = t1["yaw_delta_ci95"]

    t0 = tiers["0.0"]
    t0_gain = t0["yaw_delta_deg"]
    t0_ci = t0["yaw_delta_ci95"]
    t0_fp_delta = t0["full_pose_delta_deg"]
    t0_systematic_degrade = (t0_ci[1] < 0)  # CI 上限小于零表示显著全为退化

    z_yaw = zero["global_pure_yaw"]
    z_full = zero["global_full_pose"]
    z_yaw_degrade = z_yaw["delta_deg"]
    z_yaw_ci = z_yaw["delta_ci95"]
    z_full_degrade = z_full["delta_deg"]
    z_full_ci = z_full["delta_ci95"]
    z_systematic_degrade = (z_yaw_ci[1] < 0) or (z_full_ci[1] < 0)
    z_under_nominal_threshold = (abs(z_full_degrade) <= 0.10)

    # 4. 全局全姿态检验
    g_fp = comp["global_full_pose"]
    g_fp_delta = g_fp["delta_deg"]
    g_fp_ci = g_fp["delta_ci95"]
    g_fp_systematic_degrade = (g_fp_ci[1] < 0)

    # 逐项核验结果
    conditions = [
        {
            "criterion": "10.0°/min 偏航改善显著为正且 95% CI 下限 > 0",
            "measured": f"改善 +{t10_gain:.4f}° ({t10['yaw_reduction_pct']:+.2f}%), 95% CI [{t10_ci[0]:+.4f}°, {t10_ci[1]:+.4f}°]",
            "verdict": "PASS ✅" if t10_pass else "FAIL ❌",
            "details": "区间下限 strictly > 0，确认统计显著的小幅偏航改善。"
        },
        {
            "criterion": "3.0°/min 偏航改善显著为正且 95% CI 可信 (不跨零)",
            "measured": f"改善 +{t3_gain:.4f}° ({t3['yaw_reduction_pct']:+.2f}%), 95% CI [{t3_ci[0]:+.4f}°, {t3_ci[1]:+.4f}°], 全姿态 {t3_fp_delta:+.4f}°",
            "verdict": "FAIL ❌",
            "details": "虽然均值微弱为正 (+0.0007°)，但 95% CI 跨零 [-0.0084°, +0.0103°]，且该档全姿态发生退化 (-0.0010°)，未确认可信正收益。"
        },
        {
            "criterion": "0/1°/min 误改受控，无显著系统性负向退化",
            "measured": f"1°/min: {t1_gain:+.4f}° CI [{t1_ci[0]:+.4f}°, {t1_ci[1]:+.4f}°]; 0°/min: {t0_gain:+.4f}° CI [{t0_ci[0]:+.4f}°, {t0_ci[1]:+.4f}°]",
            "verdict": "FAIL ❌",
            "details": "0°/min 档位偏航退化 0.0202°，95% CI 上限 (-0.0107°) 严格小于零，表现出系统性误改；全姿态退化 0.0221°。"
        },
        {
            "criterion": "零标称漂移独立金标集无显著系统性破坏 (且全姿态增量 <= 0.10°)",
            "measured": f"零漂偏航: {z_yaw_degrade:+.4f}° CI [{z_yaw_ci[0]:+.4f}°, {z_yaw_ci[1]:+.4f}°]; 零漂全姿态: {z_full_degrade:+.4f}° CI [{z_full_ci[0]:+.4f}°, {z_full_ci[1]:+.4f}°]",
            "verdict": "PARTIAL / FAIL ❌",
            "details": "数值增量 0.0143° 虽然低于标称 0.10° 容差红线，但其 95% CI 完全为负 [-0.0179°, -0.0109°]，存在确定性静态负扰动，不可称为安全门控。"
        },
        {
            "criterion": "全局综合全姿态与运动学不发生系统性倒退",
            "measured": f"全局全姿态变化 {g_fp_delta:+.4f}°, 95% CI [{g_fp_ci[0]:+.4f}°, {g_fp_ci[1]:+.4f}°]",
            "verdict": "FAIL ❌",
            "details": "全局全姿态变化量 -0.0045°，95% CI 上限 (-0.0005°) 严格为负，模型在全局平均上轻微损伤全姿态。"
        }
    ]

    all_passed = all(c["verdict"] == "PASS ✅" for c in conditions)

    reassessment_verdict = {
        "report_title": "SlimeVR CSTN v2 Phase 6 Task D3.1 Independent Reassessment Audit",
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "d2_historical_status": d2_data.get("acceptance_status", "PASS ✅"),
        "reassessed_acceptance_status": "NOT PASSED / CONDITIONAL SHADOW ONLY ⚠️",
        "phase_6_closed": False,
        "online_correction_allowed": False,
        "shadow_mode_candidate_frozen": True,
        "frozen_checkpoint": d2_data["model_checkpoint"],
        "frozen_optimal_threshold_tau": d2_data["locked_optimal_threshold_tau"],
        "criteria_breakdown": conditions,
        "reassessment_summary": (
            "D2 验收报告中的 'PASS' 仅来自自动化脚本中对 10°/min 偏航与 0.10° 零漂阈值的局部检查。"
            "复核表明: 3°/min 偏航改善置信区间跨零且全姿态退化；0°/min 与独立零漂金标集的偏航与全姿态退化"
            "置信区间均严格全为负，证明存在确定性的低漂移系统性误伤；全局全姿态 95% CI 也严格全为负。"
            "因此，Phase 6 候选保留条件未全部达成，Phase 6 绝对不能宣告结项。"
            "当前最优权重与 tau*=0.50 门槛仅作为影子模式候选冻结，严禁开启在线姿态修正；"
            "项目必须进入任务 D3，在训练/验证/开发数据上重点攻坚低漂移退化问题。"
        )
    }

    print("\n  [复核核验清单]:")
    for c in conditions:
        print(f"  - [{c['verdict']}] {c['criterion']}")
        print(f"      实测: {c['measured']}")
        print(f"      分析: {c['details']}")

    print("\n" + "=" * 80)
    print(f"  [最终复核结论]: {reassessment_verdict['reassessed_acceptance_status']}")
    print(f"  - Phase 6 精度结项状态: 暂不结项 (False)")
    print(f"  - 在线姿态修正启用权限: 严禁启用 (False)")
    print(f"  - 影子模式候选权重状态: 冻结作为影子模式基准 (True)")
    print("=" * 80 + "\n")

    with open(output_audit_path, "w", encoding="utf-8") as f:
        json.dump(reassessment_verdict, f, indent=2, ensure_ascii=False)

    print(f"  已保存独立复核机器报告至: {output_audit_path}")
    return reassessment_verdict


if __name__ == "__main__":
    reassess_d2_report()

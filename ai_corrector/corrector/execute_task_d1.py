"""
ai_corrector/corrector/execute_task_d1.py
========================================
执行 Phase 6 任务 D1: 受控验证补偿模型 (Controlled Rotation Model Ablation)

严格遵循 plan.md 任务 D1 规范:
1. 在新版分层 Train/Val 数据集上固定同等预算 (15 epochs, batch 256, lr=1e-3)
2. 单因素严格对照三种旋转损失头:
   - "chordal" (默认基线弦长损失)
   - "smooth_geodesic" (平滑测地角旋转损失)
   - "yaw_aware" (平滑测地角 + 显式世界向上垂直轴偏航 Twist 监督)
3. 多种子重复 (Seed 42 & Seed 123)，记录验证集选模指标与开发测试集表现
4. 详细记录残差预测幅度 (是否打破 0.138° 压缩瓶颈)、方向余弦相似度与各档位净收益
5. 生成 d1_controlled_ablation_matrix.json 并在 validation set 上评定保留候选模型
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

from .checkpoint import resolve_path, PACKAGE_ROOT
from .evaluate_stratified import evaluate_stratified_dataset
from .model import Corrector, Config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_training_experiment(
    exp_name: str,
    loss_type: str,
    seed: int,
    train_data: Path,
    val_data: Path,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    rot_weight: float = None,
    output_pt: Path = None,
) -> Path:
    print(f"\n{'='*80}")
    print(f"  [D1 训练] 正在启动受控实验: {exp_name} | 损失={loss_type} | 种子={seed}")
    print(f"{'='*80}")

    cmd = [
        sys.executable,
        "-m", "ai_corrector.corrector.train",
        str(train_data),
        str(output_pt),
        "--val-data", str(val_data),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--loss-type", loss_type,
        "--seed", str(seed),
        "--print-iters", "50",
    ]
    if rot_weight is not None:
        cmd.extend(["--rot-weight", str(rot_weight)])

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PACKAGE_ROOT.parent), capture_output=True, text=True, encoding="utf-8")
    cost = time.time() - t0

    if res.returncode != 0:
        print(f"训练异常报错:\n{res.stderr}")
        raise RuntimeError(f"实验 {exp_name} 训练失败 (code {res.returncode})")

    best_pt = output_pt.with_name(output_pt.stem + "_best.pt")
    if not best_pt.exists():
        best_pt = output_pt

    print(f"  训练完成 (耗时 {cost:.2f}s) -> 最优权重已保存至: {best_pt.name}")
    return best_pt


def evaluate_checkpoint_on_sets(
    model_path: Path,
    val_data: Path,
    test_data: Path,
    zero_drift_data: Path,
    device: torch.device,
) -> dict:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    raw_cfg = dict(ckpt.get("config", {}))
    if "decoder_type" not in raw_cfg:
        raw_cfg["decoder_type"] = "joint_decoupled"
    config = Config(**raw_cfg)
    model = Corrector(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 1. 验证集评测 (门控关闭/全通，以真实衡量旋转头能力)
    res_val = evaluate_stratified_dataset(model, val_data, device, threshold=0.0, n_boot=200)

    # 2. 开发测试集评测 (门控全通)
    res_test = evaluate_stratified_dataset(model, test_data, device, threshold=0.0, n_boot=500)

    # 3. 零漂移金标集评测
    res_zero = evaluate_stratified_dataset(model, zero_drift_data, device, threshold=0.0, n_boot=200)

    return {
        "validation_metrics": {
            "pure_yaw_raw_deg": res_val["global_pure_yaw"]["raw_deg"],
            "pure_yaw_corr_deg": res_val["global_pure_yaw"]["corr_deg"],
            "pure_yaw_delta_deg": res_val["global_pure_yaw"]["delta_deg"],
            "pure_yaw_reduction_pct": res_val["global_pure_yaw"]["reduction_pct"],
            "full_pose_raw_deg": res_val["global_full_pose"]["raw_deg"],
            "full_pose_corr_deg": res_val["global_full_pose"]["corr_deg"],
            "full_pose_delta_deg": res_val["global_full_pose"]["delta_deg"],
            "residual_magnitude_mean_deg": res_val["residual_audit"]["predicted_magnitude_mean_deg"],
            "residual_magnitude_median_deg": res_val["residual_audit"]["predicted_magnitude_median_deg"],
            "direction_cosine_similarity": res_val["residual_audit"]["direction_cosine_similarity"],
        },
        "dev_test_metrics": {
            "pure_yaw_raw_deg": res_test["global_pure_yaw"]["raw_deg"],
            "pure_yaw_corr_deg": res_test["global_pure_yaw"]["corr_deg"],
            "pure_yaw_delta_deg": res_test["global_pure_yaw"]["delta_deg"],
            "pure_yaw_reduction_pct": res_test["global_pure_yaw"]["reduction_pct"],
            "full_pose_raw_deg": res_test["global_full_pose"]["raw_deg"],
            "full_pose_corr_deg": res_test["global_full_pose"]["corr_deg"],
            "full_pose_delta_deg": res_test["global_full_pose"]["delta_deg"],
            "residual_magnitude_mean_deg": res_test["residual_audit"]["predicted_magnitude_mean_deg"],
            "residual_magnitude_median_deg": res_test["residual_audit"]["predicted_magnitude_median_deg"],
            "direction_cosine_similarity": res_test["residual_audit"]["direction_cosine_similarity"],
            "tier_0_yaw_delta_deg": res_test["drift_tier_stratification"]["0.0"]["yaw_delta_deg"],
            "tier_1_yaw_delta_deg": res_test["drift_tier_stratification"]["1.0"]["yaw_delta_deg"],
            "tier_3_yaw_delta_deg": res_test["drift_tier_stratification"]["3.0"]["yaw_delta_deg"],
            "tier_10_yaw_delta_deg": res_test["drift_tier_stratification"]["10.0"]["yaw_delta_deg"],
            "tier_10_ci95": res_test["drift_tier_stratification"]["10.0"]["yaw_delta_ci95"],
        },
        "zero_drift_metrics": {
            "pure_yaw_delta_deg": res_zero["global_pure_yaw"]["delta_deg"],
            "full_pose_delta_deg": res_zero["global_full_pose"]["delta_deg"],
        }
    }


def main():
    p = argparse.ArgumentParser(description="Phase 6 任务 D1 受控消融全流水线")
    p.add_argument("--train-data", type=Path, default=resolve_path("database/processed_v2_stratified/train.npz"))
    p.add_argument("--val-data", type=Path, default=resolve_path("database/processed_v2_stratified/val.npz"))
    p.add_argument("--test-data", type=Path, default=resolve_path("database/processed_v2_stratified/test.npz"))
    p.add_argument("--zero-data", type=Path, default=resolve_path("database/processed_v2_stratified/test_zero_drift.npz"))
    p.add_argument("--output-json", type=Path, default=resolve_path("artifacts/d1_controlled_ablation_matrix.json"))
    p.add_argument("--epochs", type=int, default=15)
    args = p.parse_args()

    args.train_data = resolve_path(args.train_data)
    args.val_data = resolve_path(args.val_data)
    args.test_data = resolve_path(args.test_data)
    args.zero_data = resolve_path(args.zero_data)
    args.output_json = resolve_path(args.output_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 启动任务 D1 受控验证，使用设备: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    experiments = [
        # 变体 1: 弦长损失基线 (Seed 42)
        {"id": "chordal_s42", "loss": "chordal", "seed": 42, "rot_weight": 200.0},
        # 变体 2: 平滑测地角损失 (Seed 42)
        {"id": "smooth_geodesic_s42", "loss": "smooth_geodesic", "seed": 42, "rot_weight": 10.0},
        # 变体 3: 显式偏航监督损失 (Seed 42)
        {"id": "yaw_aware_s42", "loss": "yaw_aware", "seed": 42, "rot_weight": 10.0},
        # 变体 4: 弦长损失重复种子 (Seed 123)
        {"id": "chordal_s123", "loss": "chordal", "seed": 123, "rot_weight": 200.0},
        # 变体 5: 平滑测地角重复种子 (Seed 123)
        {"id": "smooth_geodesic_s123", "loss": "smooth_geodesic", "seed": 123, "rot_weight": 10.0},
    ]

    all_results = {}

    for exp in experiments:
        out_pt = resolve_path(f"artifacts/cstn_v2_d1_{exp['id']}.pt")
        best_pt = run_training_experiment(
            exp_name=exp["id"],
            loss_type=exp["loss"],
            seed=exp["seed"],
            train_data=args.train_data,
            val_data=args.val_data,
            epochs=args.epochs,
            rot_weight=exp["rot_weight"],
            output_pt=out_pt,
        )

        metrics = evaluate_checkpoint_on_sets(
            best_pt, args.val_data, args.test_data, args.zero_data, device=device
        )
        all_results[exp["id"]] = {
            "config": exp,
            "checkpoint": str(best_pt),
            "metrics": metrics,
        }

    # 保存总矩阵
    matrix_obj = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "role": "Phase 6 Task D1 Controlled Rotation Loss Ablation",
        "budget": f"{args.epochs} epochs, AdamW lr=1e-3, CosineAnnealing, batch_size=256",
        "results": all_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(matrix_obj, indent=2), encoding="utf-8")

    # 打印标准化对比汇总表
    print("\n" + "=" * 96)
    print("  === Phase 6 任务 D1 受控旋转损失消融矩阵汇总 ===")
    print("=" * 96)
    print(f"{'实验配置':<24} | {'Val 偏航改善(Δ)':<16} | {'Val 全姿态(Δ)':<14} | {'预测残差均值':<14} | {'Test 10°/min 偏航改善'}")
    print("-" * 96)
    for exp_id, data in all_results.items():
        vm = data["metrics"]["validation_metrics"]
        tm = data["metrics"]["dev_test_metrics"]
        print(f"{exp_id:<24} | {vm['pure_yaw_delta_deg']:>+10.4f}° ({vm['pure_yaw_reduction_pct']:>+5.1f}%) | {vm['full_pose_delta_deg']:>+10.4f}° | {vm['residual_magnitude_mean_deg']:>10.4f}° | {tm['tier_10_yaw_delta_deg']:>+10.4f}° (CI: [{tm['tier_10_ci95'][0]:+.3f}, {tm['tier_10_ci95'][1]:+.3f}])")

    print("\n" + "=" * 96)
    print(f"完整受控消融报告已写入: {args.output_json}\n")


if __name__ == "__main__":
    main()

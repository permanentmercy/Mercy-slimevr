"""
ai_corrector/corrector/prepare_p11_dataset.py
============================================
Phase 11: 合成动作相关 Swing 扰动的时序数据集生成器

建模边界:
真实 AHRS 的动态倾斜误差可能受线加速度、重力观测、融合器参数和传感器安装影响；本生成器没有从加速度计比力运行 AHRS，
也没有用实机数据标定误差分布。它只构造一个可控的合成 Swing 扰动：轴取 omega × up，角度取
clip(0.040 * ||omega × up||, 0, 6°)。其中 0.040 是启发式比例系数（omega 用 rad/s 时量纲为秒），不是经实测拟合的传感器常数。
因此数据可用于验证网络能否学习这一类动作相关扰动，不能证明模型已学会校正真实 SlimeVR AHRS 的动态加速度误差。

产出目录:
database/p11_continuous/
- train_continuous.npz
- val_continuous.npz
- final_confirm_continuous.npz
- p11_dataset_manifest.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .checkpoint import resolve_path, PACKAGE_ROOT
from .contract_v2 import (
    TOTAL_INPUT_DIM_V2,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat,
    rot6d_to_matrix_np,
)
from .contract_v3 import LOWER_BODY_JOINTS
from .hybrid_composer import swing_twist_decomposition_y, rotmat_to_quat_wxyz_safe

TARGET_LOWER_JOINTS = [1, 2, 4, 5]

SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]


def rotmat_to_rot6d(m: np.ndarray) -> np.ndarray:
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1).astype(np.float32)


def world_pose(local_rot: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(local_rot)
    world = np.zeros((n, 22, 3, 3), dtype=np.float32)
    pos = np.zeros((n, 22, 3), dtype=np.float32)
    pos[:, 0] = trans
    world[:, 0] = local_rot[:, 0]
    for i in range(1, 22):
        parent = SMPL_PARENTS[i]
        world[:, i] = np.einsum("nij,njk->nik", world[:, parent], local_rot[:, i])
        pos[:, i] = pos[:, parent] + world[:, parent][:, :, 1] * 0.2
    return world, pos


def angular_velocity(r: np.ndarray) -> np.ndarray:
    if len(r) < 2:
        return np.zeros((len(r), 3), dtype=np.float32)
    rel = np.einsum("nij,njk->nik", np.swapaxes(r[:-1], 1, 2), r[1:])
    v = Rotation.from_matrix(rel).as_rotvec().astype(np.float32)
    return np.vstack([v[:1], v]) * 60.0


def acceleration(p: np.ndarray) -> np.ndarray:
    v = np.gradient(p, axis=0) * 60.0
    return np.gradient(v, axis=0) * 60.0


def generate_noise_profile(rng: np.random.Generator, num_frames: int, drift_tier: float) -> dict:
    mounting_error_euler = rng.normal(0.0, 0.3, size=(4, 3)).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=4).astype(np.float32)

    tier_val = float(drift_tier)
    if tier_val == 0.0:
        actual_rates = np.zeros(4, dtype=np.float32)
        rw_sigma = np.deg2rad(0.002)
    else:
        actual_rates = (signs * rng.normal(tier_val, tier_val * 0.15, size=4)).astype(np.float32)
        rw_sigma = np.deg2rad(0.005 * np.sqrt(tier_val))

    random_walk = np.cumsum(rng.normal(0, rw_sigma, (num_frames, 4)), axis=0).astype(np.float32)

    has_vd_chest = bool(rng.random() > 0.20)
    vd_chest_age = float(rng.uniform(0.0, 0.02))
    vd_left_arm_age = float(rng.uniform(0.0, 0.02))
    vd_right_arm_age = float(rng.uniform(0.0, 0.02))
    imu_valid = (rng.random((4, num_frames)) > 0.01).astype(np.float32)

    return {
        "mounting_error_euler": mounting_error_euler,
        "drift_rates": actual_rates,
        "random_walk": random_walk,
        "has_vd_chest": has_vd_chest,
        "vd_chest_age": vd_chest_age,
        "vd_left_arm_age": vd_left_arm_age,
        "vd_right_arm_age": vd_right_arm_age,
        "imu_valid": imu_valid,
        "drift_tier": tier_val,
    }


def synthesize_p11_sequence(
    clean_world: np.ndarray,
    positions: np.ndarray,
    noise_profile: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    num_frames = len(clean_world)
    time_minutes = np.arange(num_frames, dtype=np.float32) / (60.0 * 60.0)

    mounting_rot = Rotation.from_euler("xyz", noise_profile["mounting_error_euler"], degrees=True).as_matrix()

    base_rates = noise_profile["drift_rates"]
    linear_drift = np.outer(time_minutes, base_rates)
    total_yaw_drift = np.deg2rad(linear_drift) + noise_profile["random_walk"]

    raw_world = clean_world.copy()
    rotvec_yaw = np.zeros((num_frames, 4, 3), dtype=np.float32)
    rotvec_yaw[..., 1] = total_yaw_drift
    drift_rot = Rotation.from_rotvec(rotvec_yaw.reshape(-1, 3)).as_matrix().reshape(num_frames, 4, 3, 3)

    u_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # 为四路下肢生成可控的合成动作相关 Swing 扰动；该启发式未经过真实 AHRS/加速度计标定。
    for k, joint in enumerate(TARGET_LOWER_JOINTS):
        omega_k = angular_velocity(clean_world[:, joint])  # (N, 3)
        # 倾斜轴取角速度与重力竖直轴的叉乘 (处于水平面 X-Z, 作为纯 Swing 扰动)
        tilt_axis = np.cross(omega_k, u_world)  # (N, 3)
        norm_tilt = np.linalg.norm(tilt_axis, axis=-1, keepdims=True)
        unit_tilt_axis = np.where(norm_tilt > 1e-6, tilt_axis / np.maximum(norm_tilt, 1e-6), np.zeros_like(tilt_axis))
        # 合成扰动幅值随角速度线性增长并截断；不是从线加速度或 AHRS 状态估计得到。
        tilt_angle = np.clip(0.040 * norm_tilt, 0.0, np.deg2rad(6.0)).astype(np.float32)
        tilt_rotvec = unit_tilt_axis * tilt_angle
        R_dyn_tilt = Rotation.from_rotvec(tilt_rotvec).as_matrix()  # (N, 3, 3)

        # 组合合成观测: R_raw = R_yaw_drift @ R_dyn_tilt @ R_clean @ R_mounting
        temp = np.einsum("nij,njk->nik", R_dyn_tilt, clean_world[:, joint])
        temp_yaw = np.einsum("nij,njk->nik", drift_rot[:, k], temp)
        raw_world[:, joint] = np.einsum("nij,jk->nik", temp_yaw, mounting_rot[k])

    raw_omega = np.stack([angular_velocity(raw_world[:, j]) for j in range(22)], axis=1)
    raw_acc = acceleration(positions)

    # 组装 124 维特征流
    feat = np.zeros((num_frames, TOTAL_INPUT_DIM_V2), dtype=np.float32)
    head = 15
    feat[:, 0:3] = 0.0
    feat[:, 3:9] = rotmat_to_rot6d(clean_world[:, head])
    feat[:, 9:12] = np.gradient(positions[:, head], axis=0) * 60.0
    feat[:, 12:15] = angular_velocity(clean_world[:, head])
    feat[:, 15] = 1.0
    feat[:, 16] = 0.0

    if noise_profile["has_vd_chest"]:
        feat[:, 17:20] = positions[:, 6] - positions[:, head]
        feat[:, 20:26] = rotmat_to_rot6d(clean_world[:, 6])
        feat[:, 26:29] = np.gradient(positions[:, 6], axis=0) * 60.0
        feat[:, 29:32] = angular_velocity(clean_world[:, 6])
        feat[:, 32] = 1.0
        feat[:, 33] = noise_profile["vd_chest_age"]

    feat[:, 34:37] = positions[:, 16] - positions[:, head]
    feat[:, 37:43] = rotmat_to_rot6d(clean_world[:, 16])
    feat[:, 43:46] = np.gradient(positions[:, 16], axis=0) * 60.0
    feat[:, 46:49] = angular_velocity(clean_world[:, 16])
    feat[:, 49] = 1.0
    feat[:, 50] = noise_profile["vd_left_arm_age"]

    feat[:, 51:54] = positions[:, 17] - positions[:, head]
    feat[:, 54:60] = rotmat_to_rot6d(clean_world[:, 17])
    feat[:, 60:63] = np.gradient(positions[:, 17], axis=0) * 60.0
    feat[:, 63:66] = angular_velocity(clean_world[:, 17])
    feat[:, 66] = 1.0
    feat[:, 67] = noise_profile["vd_right_arm_age"]

    offsets_14d = [68, 82, 96, 110]
    for k, (start, joint) in enumerate(zip(offsets_14d, TARGET_LOWER_JOINTS)):
        feat[:, start:start+6] = rotmat_to_rot6d(raw_world[:, joint])
        feat[:, start+6:start+9] = raw_acc[:, joint]
        feat[:, start+9:start+12] = raw_omega[:, joint]
        feat[:, start+12] = noise_profile["imu_valid"][k]
        feat[:, start+13] = 0.0

    clean_target = clean_world[:, TARGET_LOWER_JOINTS]
    raw_target = raw_world[:, TARGET_LOWER_JOINTS]

    # 保存局部右乘真值残差；训练时根据原始姿态转换到世界系提取 Swing。
    dq_mat = np.einsum("ntji,ntjk->ntik", raw_target, clean_target)
    dq = rotmat_to_quat_wxyz(dq_mat)

    # 门控值: 动态倾斜误差较大时门控接近 1.0, 静态接近 0.0
    dot = np.abs(np.einsum("ntk,ntk->nt", rotmat_to_quat_wxyz(raw_target), rotmat_to_quat_wxyz(clean_target)))
    dot = np.clip(dot, -1.0, 1.0)
    angle_err_deg = 2.0 * np.arccos(dot) * 180.0 / np.pi
    gate = np.clip((angle_err_deg - 0.5) / 1.5, 0.0, 1.0).astype(np.float32)

    foot_speed = np.linalg.norm(np.gradient(positions[:, [7, 8]], axis=0) * 60.0, axis=-1)
    contact = (foot_speed < 0.35).astype(np.float32)

    meta = {
        "drift_tier": noise_profile["drift_tier"],
        "drift_rates": noise_profile["drift_rates"],
        "num_frames": num_frames,
    }
    return feat, dq, gate, contact, meta


def process_p11_sequence_item(item: Tuple) -> Optional[dict]:
    path_str, seq_id, subset_name, split_name, seed, drift_tier = item
    try:
        data = np.load(path_str, allow_pickle=False)
        poses = data["poses"].astype(np.float32)[:, :66].reshape(-1, 22, 3)
        trans = data["trans"].astype(np.float32)
        local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
        clean_world, positions = world_pose(local, trans)

        num_frames = len(local)
        if num_frames < 20:
            return None

        rng = np.random.default_rng(seed)
        noise_profile = generate_noise_profile(rng, num_frames, drift_tier)
        seq_name = Path(path_str).stem

        feat, dq, gate, contact, meta = synthesize_p11_sequence(clean_world, positions, noise_profile)

        return {
            "x": feat,
            "dq": dq,
            "gate": gate,
            "contact": contact,
            "seq_len": np.int32(num_frames),
            "seq_id": np.int32(seq_id),
            "seq_name": str(seq_name),
            "subset": str(subset_name),
            "split": str(split_name),
            "drift_tier": np.float32(meta["drift_tier"]),
            "drift_rates": meta["drift_rates"].astype(np.float32),
            "sample_seed": np.int64(seed),
            "fps": np.float32(60.0),
        }
    except Exception:
        return None


def pad_and_pack_p11(seq_list: List[dict], max_len: int = 300) -> dict:
    n = len(seq_list)
    x_pad = np.zeros((n, max_len, TOTAL_INPUT_DIM_V2), dtype=np.float32)
    dq_pad = np.zeros((n, max_len, 4, 4), dtype=np.float32)
    dq_pad[..., 0] = 1.0
    gate_pad = np.zeros((n, max_len, 4), dtype=np.float32)
    contact_pad = np.zeros((n, max_len, 2), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=np.bool_)

    for i, s in enumerate(seq_list):
        l = min(s["seq_len"], max_len)
        x_pad[i, :l] = s["x"][:l]
        dq_pad[i, :l] = s["dq"][:l]
        gate_pad[i, :l] = s["gate"][:l]
        contact_pad[i, :l] = s["contact"][:l]
        mask[i, :l] = True

    return {
        "x": x_pad,
        "dq": dq_pad,
        "gate": gate_pad,
        "contact": contact_pad,
        "mask": mask,
        "seq_lens": np.array([min(s["seq_len"], max_len) for s in seq_list], dtype=np.int32),
        "seq_ids": np.array([s["seq_id"] for s in seq_list], dtype=np.int32),
        "seq_names": np.array([s["seq_name"] for s in seq_list], dtype="U64"),
        "subsets": np.array([s["subset"] for s in seq_list], dtype="U32"),
        "splits": np.array([s["split"] for s in seq_list], dtype="U32"),
        "drift_tiers": np.array([s["drift_tier"] for s in seq_list], dtype=np.float32),
        "drift_rates": np.stack([s["drift_rates"] for s in seq_list]).astype(np.float32),
        "sample_seeds": np.array([s["sample_seed"] for s in seq_list], dtype=np.int64),
        "fps": np.array([s["fps"] for s in seq_list], dtype=np.float32),
    }


def generate_p11_datasets(source_root: Path, output_dir: Path, seed: int = 20261018):
    print("=" * 80)
    print("生成 Phase 11 动态倾斜失真时序数据集 (Dynamic Tilt Dataset)...")
    print(f"数据源: {source_root} -> 输出: {output_dir}")
    print("=" * 80)

    subsets = sorted([d for d in source_root.iterdir() if d.is_dir()])
    tier_cycle = [0.0, 1.0, 3.0, 10.0]

    train_items, val_items, confirm_items = [], [], []
    global_seq_id = 0

    for sid_idx, sdir in enumerate(subsets):
        npz_files = sorted(list(sdir.glob("*.npz")))
        if len(npz_files) < 41:
            continue
        # 前 21 个训练
        for p in npz_files[:21]:
            sid = global_seq_id
            global_seq_id += 1
            drift = tier_cycle[sid % 4]
            train_items.append((str(p), sid, sdir.name, "train", seed + sid * 31, drift))
        # 21..24 验证
        for p in npz_files[21:25]:
            sid = global_seq_id
            global_seq_id += 1
            drift = tier_cycle[sid % 4]
            val_items.append((str(p), sid, sdir.name, "val", seed + sid * 31, drift))
        # 36..40 独立未见确认集
        for p in npz_files[36:41]:
            sid = global_seq_id
            global_seq_id += 1
            drift = tier_cycle[sid % 4]
            confirm_items.append((str(p), sid, sdir.name, "final_confirm", seed + sid * 31, drift))

    print(f"划分: Train={len(train_items)} | Val={len(val_items)} | FinalConfirm={len(confirm_items)}")

    def run_split(items):
        res_list = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            for r in executor.map(process_p11_sequence_item, items, chunksize=16):
                if r is not None:
                    res_list.append(r)
        return res_list

    t0 = time.time()
    print("  -> 生成 Train 集...")
    train_data = run_split(train_items)
    print("  -> 生成 Val 集...")
    val_data = run_split(val_items)
    print("  -> 生成 FinalConfirm 集 (未见源序列 36~40)...")
    conf_data = run_split(confirm_items)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "train_continuous.npz", **pad_and_pack_p11(train_data))
    np.savez_compressed(output_dir / "val_continuous.npz", **pad_and_pack_p11(val_data))
    np.savez_compressed(output_dir / "final_confirm_continuous.npz", **pad_and_pack_p11(conf_data))

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "role": "Phase 11 Synthetic Motion-Correlated Swing Perturbation Dataset",
        "train_sequences": len(train_data),
        "val_sequences": len(val_data),
        "final_confirm_sequences": len(conf_data),
    }
    with open(output_dir / "p11_dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"  数据集生成完毕 (耗时 {time.time()-t0:.2f}s) -> 目录: {output_dir.name}")
    return manifest


if __name__ == "__main__":
    source = resolve_path("database/amass/HumanML3D_SMPLH52")
    output = resolve_path("database/p11_continuous")
    generate_p11_datasets(source, output)

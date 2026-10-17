"""
ai_corrector/corrector/prepare_d3c_dataset.py
============================================
Phase 6 任务 D3c: 逐帧连续时序特征与配对反事实数据集生成器
(D3c Continuous Sequential & Strictly Paired Counterfactual Dataset Generator)

严格遵循 plan.md 任务 D3c 规范:
1. 逐帧完整连续时序:
   - 彻底摒弃 20 帧短窗重叠切片, 保留每个源序列由 t=0 至 T-1 的完整逐帧连续输入;
   - 每帧恰好输入一次, 严格按时序推进, 支持真正的连续因果截断反传 (TBPTT) 与连续重放;
2. 严格 1-to-1 配对反事实 (Strictly Paired Counterfactuals):
   - 标准 (std) 与反事实 (cf) 序列拥有完全相同的 clean 运动、安装误差、随机游走、VD 缺失及 IMU 丢包掩码;
   - 唯一差异为四路 IMU 线性漂移率符号严格相反 (actual_rates -> -actual_rates);
   - 拥有全局唯一的稳定 pair_id, 保证逐序列、逐帧、逐关节点的严格 1-to-1 配对;
3. 严格数据隔离划分:
   - 训练集 Train: 每子集前 21 个源文件 (0..20);
   - 验证集 Val: 每子集文件 21..24;
   - 开发测试集 DevTest (含对齐反事实): 每子集文件 25..29;
   - 独立确认集 FinalConfirm (含对齐反事实): 从每子集第 36 条开始抽取从未参与任何历史调优的未见源序列 (36..40);
4. 产出目录:
   - database/d3c_continuous/
   - 产出: train_continuous.npz, val_continuous.npz, dev_test_continuous.npz, dev_test_cf_continuous.npz,
          final_confirm_continuous.npz, final_confirm_cf_continuous.npz, d3c_dataset_manifest.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
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

TARGET_LOWER_JOINTS = [1, 2, 4, 5]

SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def rotmat_to_rot6d(m: np.ndarray) -> np.ndarray:
    """提取 3x3 旋转矩阵前两列拼接为 6 维连续向量 [col0, col1]。"""
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
    mounting_error_euler = rng.normal(0.0, 0.5, size=(4, 3)).astype(np.float32)
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


def synthesize_full_sequence(
    clean_world: np.ndarray,
    positions: np.ndarray,
    noise_profile: dict,
    is_counterfactual: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    num_frames = len(clean_world)
    time_minutes = np.arange(num_frames, dtype=np.float32) / (60.0 * 60.0)

    mounting_rot = Rotation.from_euler("xyz", noise_profile["mounting_error_euler"], degrees=True).as_matrix()

    base_rates = noise_profile["drift_rates"]
    effective_rates = -base_rates if is_counterfactual else base_rates
    effective_tier = -noise_profile["drift_tier"] if is_counterfactual else noise_profile["drift_tier"]

    linear_drift = np.outer(time_minutes, effective_rates)
    total_yaw_drift = np.deg2rad(linear_drift) + noise_profile["random_walk"]

    raw_world = clean_world.copy()
    rotvec = np.zeros((num_frames, 4, 3), dtype=np.float32)
    rotvec[..., 1] = total_yaw_drift
    drift_rot = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(num_frames, 4, 3, 3)

    for k, joint in enumerate(TARGET_LOWER_JOINTS):
        temp = np.einsum("nij,njk->nik", drift_rot[:, k], clean_world[:, joint])
        raw_world[:, joint] = np.einsum("nij,jk->nik", temp, mounting_rot[k])

    raw_omega = np.stack([angular_velocity(raw_world[:, j]) for j in range(22)], axis=1)
    raw_acc = acceleration(positions)

    # 组装 124 维逐帧连续输入特征
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
    dq_mat = np.einsum("ntji,ntjk->ntik", raw_target, clean_target)
    dq = rotmat_to_quat_wxyz(dq_mat)

    dot = np.abs(np.einsum("ntk,ntk->nt", rotmat_to_quat_wxyz(raw_target), rotmat_to_quat_wxyz(clean_target)))
    dot = np.clip(dot, -1.0, 1.0)
    angle_err_deg = 2.0 * np.arccos(dot) * 180.0 / np.pi
    gate = np.clip((angle_err_deg - 0.25) / 1.25, 0.0, 1.0).astype(np.float32)

    foot_speed = np.linalg.norm(np.gradient(positions[:, [7, 8]], axis=0) * 60.0, axis=-1)
    contact = (foot_speed < 0.35).astype(np.float32)

    meta = {
        "drift_tier": effective_tier,
        "drift_rates": effective_rates,
        "is_counterfactual": is_counterfactual,
        "num_frames": num_frames,
    }
    return feat, dq, gate, contact, meta


def process_sequence_item(item: Tuple) -> Optional[Tuple[dict, Optional[dict]]]:
    (path_str, seq_id, subset_name, split_name, seed, drift_tier, make_counterfactual, pair_id) = item
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

        # 1. 完整连续标准序列
        feat_std, dq_std, gate_std, contact_std, meta_std = synthesize_full_sequence(
            clean_world, positions, noise_profile, is_counterfactual=False
        )

        std_dict = {
            "x": feat_std,
            "dq": dq_std,
            "gate": gate_std,
            "contact": contact_std,
            "seq_len": np.int32(num_frames),
            "pair_id": np.int32(pair_id),
            "seq_id": np.int32(seq_id),
            "seq_name": str(seq_name),
            "subset": str(subset_name),
            "split": str(split_name),
            "drift_tier": np.float32(meta_std["drift_tier"]),
            "drift_rates": meta_std["drift_rates"].astype(np.float32),
            "drift_signs": np.sign(meta_std["drift_rates"]).astype(np.float32),
            "sample_seed": np.int64(seed),
            "is_counterfactual": np.bool_(False),
            "fps": np.float32(60.0),
        }

        cf_dict = None
        if make_counterfactual:
            # 2. 严格 1-to-1 配对连续反事实序列
            feat_cf, dq_cf, gate_cf, contact_cf, meta_cf = synthesize_full_sequence(
                clean_world, positions, noise_profile, is_counterfactual=True
            )
            cf_dict = {
                "x": feat_cf,
                "dq": dq_cf,
                "gate": gate_cf,
                "contact": contact_cf,
                "seq_len": np.int32(num_frames),
                "pair_id": np.int32(pair_id),
                "seq_id": np.int32(seq_id),
                "seq_name": str(seq_name),
                "subset": str(subset_name),
                "split": str(split_name),
                "drift_tier": np.float32(meta_cf["drift_tier"]),
                "drift_rates": meta_cf["drift_rates"].astype(np.float32),
                "drift_signs": np.sign(meta_cf["drift_rates"]).astype(np.float32),
                "sample_seed": np.int64(seed),
                "is_counterfactual": np.bool_(True),
                "fps": np.float32(60.0),
            }

        return std_dict, cf_dict
    except Exception as e:
        return None


def pad_and_pack_continuous(seq_list: List[dict], max_len: int = 300) -> dict:
    n = len(seq_list)
    x_pad = np.zeros((n, max_len, TOTAL_INPUT_DIM_V2), dtype=np.float32)
    dq_pad = np.zeros((n, max_len, 4, 4), dtype=np.float32)
    # 单位四元数初始化
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
        "pair_ids": np.array([s["pair_id"] for s in seq_list], dtype=np.int32),
        "seq_ids": np.array([s["seq_id"] for s in seq_list], dtype=np.int32),
        "seq_names": np.array([s["seq_name"] for s in seq_list], dtype="U64"),
        "subsets": np.array([s["subset"] for s in seq_list], dtype="U32"),
        "splits": np.array([s["split"] for s in seq_list], dtype="U32"),
        "drift_tiers": np.array([s["drift_tier"] for s in seq_list], dtype=np.float32),
        "drift_rates": np.stack([s["drift_rates"] for s in seq_list]).astype(np.float32),
        "drift_signs": np.stack([s["drift_signs"] for s in seq_list]).astype(np.float32),
        "sample_seeds": np.array([s["sample_seed"] for s in seq_list], dtype=np.int64),
        "is_counterfactuals": np.array([s["is_counterfactual"] for s in seq_list], dtype=np.bool_),
        "fps": np.array([s["fps"] for s in seq_list], dtype=np.float32),
    }


def generate_d3c_datasets(
    source_root: Path,
    output_dir: Path,
    seed: int = 20261017,
):
    print("\n" + "=" * 80)
    print("  [D3c.1] 正在生成 D3c 逐帧完整连续时序与配对反事实实验数据集...")
    print("=" * 80)

    subsets = sorted([d for d in source_root.iterdir() if d.is_dir()])
    tier_cycle = [0.0, 1.0, 3.0, 10.0]

    train_items = []
    val_items = []
    dev_items = []
    confirm_items = []

    global_seq_id = 0
    global_pair_id = 0

    for s_idx, sdir in enumerate(subsets):
        sfiles = sorted(sdir.glob("*.npz"))
        # 严格隔离划分:
        # Train: 0..20
        train_files = sfiles[:21]
        # Val: 21..24
        val_files = sfiles[21:25]
        # DevTest: 25..29
        dev_files = sfiles[25:30]
        # FinalConfirm: 从第 36 条开始抽取 5 条全新独立序列
        confirm_files = sfiles[36:41] if len(sfiles) >= 41 else sfiles[30:35]

        for p in train_files:
            sid = global_seq_id
            pid = global_pair_id
            global_seq_id += 1
            global_pair_id += 1
            drift = tier_cycle[sid % 4]
            train_items.append((str(p), sid, sdir.name, "train", seed + sid * 41, drift, False, pid))

        for p in val_files:
            sid = global_seq_id
            pid = global_pair_id
            global_seq_id += 1
            global_pair_id += 1
            drift = tier_cycle[sid % 4]
            val_items.append((str(p), sid, sdir.name, "val", seed + sid * 41, drift, False, pid))

        for p in dev_files:
            sid = global_seq_id
            pid = global_pair_id
            global_seq_id += 1
            global_pair_id += 1
            drift = tier_cycle[sid % 4]
            dev_items.append((str(p), sid, sdir.name, "dev_test", seed + sid * 41, drift, True, pid))

        for p in confirm_files:
            sid = global_seq_id
            pid = global_pair_id
            global_seq_id += 1
            global_pair_id += 1
            drift = tier_cycle[sid % 4]
            confirm_items.append((str(p), sid, sdir.name, "final_confirm", seed + sid * 41, drift, True, pid))

    print(f"  序列划分清单: Train={len(train_items)} | Val={len(val_items)} | DevTest={len(dev_items)} | Confirm={len(confirm_items)}")

    def run_split(items):
        std_list, cf_list = [], []
        with ProcessPoolExecutor(max_workers=6) as executor:
            for res in executor.map(process_sequence_item, items, chunksize=16):
                if res is not None:
                    s_dict, c_dict = res
                    std_list.append(s_dict)
                    if c_dict is not None:
                        cf_list.append(c_dict)
        return std_list, cf_list

    t0 = time.time()
    print("  -> 生成连续 Train 集中...")
    train_std, _ = run_split(train_items)
    print("  -> 生成连续 Val 集中...")
    val_std, _ = run_split(val_items)
    print("  -> 生成连续 DevTest 集中 (含 1-to-1 配对反事实)...")
    dev_std, dev_cf = run_split(dev_items)
    print("  -> 生成全新独立 FinalConfirm 集中 (从未参与历史调参的序列 36~40)...")
    conf_std, conf_cf = run_split(confirm_items)

    assert len(dev_std) == len(dev_cf), "DevTest 配对数量不一致!"
    assert len(conf_std) == len(conf_cf), "FinalConfirm 配对数量不一致!"

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "train_continuous.npz", **pad_and_pack_continuous(train_std))
    np.savez_compressed(output_dir / "val_continuous.npz", **pad_and_pack_continuous(val_std))
    np.savez_compressed(output_dir / "dev_test_continuous.npz", **pad_and_pack_continuous(dev_std))
    np.savez_compressed(output_dir / "dev_test_cf_continuous.npz", **pad_and_pack_continuous(dev_cf))
    np.savez_compressed(output_dir / "final_confirm_continuous.npz", **pad_and_pack_continuous(conf_std))
    np.savez_compressed(output_dir / "final_confirm_cf_continuous.npz", **pad_and_pack_continuous(conf_cf))

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "role": "D3c Full-Sequence Continuous and Strictly Paired Counterfactual Dataset",
        "format": "Full-length sequential, no overlapping window slicing, exact single frame input",
        "train_sequences": len(train_std),
        "val_sequences": len(val_std),
        "dev_test_sequences": len(dev_std),
        "dev_test_cf_sequences": len(dev_cf),
        "final_confirm_sequences": len(conf_std),
        "final_confirm_cf_sequences": len(conf_cf),
        "subsets_count": len(subsets),
        "total_train_frames": int(sum(s["seq_len"] for s in train_std)),
        "total_dev_frames": int(sum(s["seq_len"] for s in dev_std)),
    }
    with open(output_dir / "d3c_dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"  连续数据集生成完毕 (耗时 {time.time()-t0:.2f}s):")
    print(f"    - train_continuous.npz:         {len(train_std)} 条完整时序 ({manifest['total_train_frames']} 帧)")
    print(f"    - val_continuous.npz:           {len(val_std)} 条完整时序")
    print(f"    - dev_test_continuous.npz:      {len(dev_std)} 条完整时序")
    print(f"    - dev_test_cf_continuous.npz:   {len(dev_cf)} 条严格配对反事实时序")
    print(f"    - final_confirm_continuous.npz: {len(conf_std)} 条未见独立确认时序")
    return manifest


if __name__ == "__main__":
    source = resolve_path("database/amass/HumanML3D_SMPLH52")
    output = resolve_path("database/d3c_continuous")
    generate_d3c_datasets(source, output)

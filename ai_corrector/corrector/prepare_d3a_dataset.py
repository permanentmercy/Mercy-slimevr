"""
ai_corrector/corrector/prepare_d3a_dataset.py
============================================
Phase 6 任务 D3a: 无捷径与配对反事实数据生成器 (D3a Unbiased & Counterfactual Dataset Generator)

严格遵循 plan.md 任务 D3a 规范:
1. 消除安装误差与漂移档位关联: 所有档位统一安装误差分布 N(0, 0.5 deg);
2. 消除正向偏航单向漂移捷径: 四路下肢独立随机采样有符号漂移方向 (+/-);
3. 覆盖 0.0, ±1.0, ±3.0, ±10.0 deg/min 及随机游走;
4. 生成配对反事实测试样本 (Counterfactual Pairs): 同一干净动作施加相反漂移, 用于检验网络是否响应真实传感器物理漂移;
5. 记录逐窗时序元数据 (seq_id, start_frame, window_idx_in_seq, is_seq_start), 支持长因果历史连续重放。
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
    rot6d_to_matrix_np,
)

TARGET_LOWER_JOINTS = [1, 2, 4, 5]


def rotmat_to_rot6d(m: np.ndarray) -> np.ndarray:
    # Contract v2 stores the complete first column, then the complete second column.
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1).astype(np.float32)


def verify_rot6d_contract() -> None:
    rotations = Rotation.from_euler("xyz", [[10, 20, 30], [0, 90, 0], [45, -20, 70]], degrees=True).as_matrix()
    np.testing.assert_allclose(rot6d_to_matrix_np(rotmat_to_rot6d(rotations)), rotations, atol=1e-6)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]


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


def build_sequence_d3a(
    path: Path,
    rng: np.random.Generator,
    drift_tier: float = 0.0,
    drift_signs: Optional[np.ndarray] = None,
    fixed_mounting: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    poses = data["poses"].astype(np.float32)[:, :66].reshape(-1, 22, 3)
    trans = data["trans"].astype(np.float32)
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
    clean_world, positions = world_pose(local, trans)

    num_frames = len(local)
    if num_frames < 20:
        raise ValueError("Sequence too short")

    time_minutes = np.arange(num_frames, dtype=np.float32) / (60.0 * 60.0)

    # 1. 统一安装误差分布: 严格无论漂移档位均 N(0, 0.5°)
    if fixed_mounting is not None:
        mounting_error_euler = fixed_mounting
    else:
        mounting_error_euler = rng.normal(0.0, 0.5, size=(4, 3))
    mounting_rot = Rotation.from_euler("xyz", mounting_error_euler, degrees=True).as_matrix()

    # 2. 四路独立有符号漂移
    if drift_signs is None:
        signs = rng.choice([-1.0, 1.0], size=4)
    else:
        signs = np.array(drift_signs, dtype=np.float32)

    tier_val = float(drift_tier)
    if tier_val == 0.0:
        # 零标称漂移: 纯微小随机游走
        rw_sigma = np.deg2rad(0.002)
        random_walk = np.cumsum(rng.normal(0, rw_sigma, (num_frames, 4)), axis=0)
        total_yaw_drift = random_walk
        actual_rates = np.zeros(4, dtype=np.float32)
    else:
        # 四路独立有符号线漂移率 (deg/min)
        actual_rates = signs * rng.normal(tier_val, tier_val * 0.15, size=4).astype(np.float32)
        rw_sigma = np.deg2rad(0.005 * np.sqrt(tier_val))
        random_walk = np.cumsum(rng.normal(0, rw_sigma, (num_frames, 4)), axis=0)
        linear_drift = np.outer(time_minutes, actual_rates)
        total_yaw_drift = np.deg2rad(linear_drift) + random_walk

    # 施加偏航漂移 (+Y 轴世界坐标) 与局部安装误差
    raw_world = clean_world.copy()
    rotvec = np.zeros((num_frames, 4, 3), dtype=np.float32)
    rotvec[..., 1] = total_yaw_drift
    drift_rot = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(num_frames, 4, 3, 3)

    for k, joint in enumerate(TARGET_LOWER_JOINTS):
        temp = np.einsum("nij,njk->nik", drift_rot[:, k], clean_world[:, joint])
        raw_world[:, joint] = np.einsum("nij,jk->nik", temp, mounting_rot[k])

    raw_omega = np.stack([angular_velocity(raw_world[:, j]) for j in range(22)], axis=1)
    raw_acc = acceleration(positions)

    # 3. 组装 124 维输入特征
    feat = np.zeros((num_frames, TOTAL_INPUT_DIM_V2), dtype=np.float32)
    head = 15
    feat[:, 0:3] = 0.0
    feat[:, 3:9] = rotmat_to_rot6d(clean_world[:, head])
    feat[:, 9:12] = np.gradient(positions[:, head], axis=0) * 60.0
    feat[:, 12:15] = angular_velocity(clean_world[:, head])
    feat[:, 15] = 1.0
    feat[:, 16] = 0.0

    # VD 胸部 (20% 缺省)
    has_vd_chest = rng.random() > 0.20
    if has_vd_chest:
        feat[:, 17:20] = positions[:, 6] - positions[:, head]
        feat[:, 20:26] = rotmat_to_rot6d(clean_world[:, 6])
        feat[:, 26:29] = np.gradient(positions[:, 6], axis=0) * 60.0
        feat[:, 29:32] = angular_velocity(clean_world[:, 6])
        feat[:, 32] = 1.0
        feat[:, 33] = rng.uniform(0.0, 0.02)

    # VD 左上臂
    feat[:, 34:37] = positions[:, 16] - positions[:, head]
    feat[:, 37:43] = rotmat_to_rot6d(clean_world[:, 16])
    feat[:, 43:46] = np.gradient(positions[:, 16], axis=0) * 60.0
    feat[:, 46:49] = angular_velocity(clean_world[:, 16])
    feat[:, 49] = 1.0
    feat[:, 50] = rng.uniform(0.0, 0.02)

    # VD 右上臂
    feat[:, 51:54] = positions[:, 17] - positions[:, head]
    feat[:, 54:60] = rotmat_to_rot6d(clean_world[:, 17])
    feat[:, 60:63] = np.gradient(positions[:, 17], axis=0) * 60.0
    feat[:, 63:66] = angular_velocity(clean_world[:, 17])
    feat[:, 66] = 1.0
    feat[:, 67] = rng.uniform(0.0, 0.02)

    # 4 路下肢 IMU (各 14 维)
    offsets_14d = [68, 82, 96, 110]
    for start, joint in zip(offsets_14d, TARGET_LOWER_JOINTS):
        feat[:, start:start+6] = rotmat_to_rot6d(raw_world[:, joint])
        feat[:, start+6:start+9] = raw_acc[:, joint]
        feat[:, start+9:start+12] = raw_omega[:, joint]
        is_valid = (rng.random(num_frames) > 0.01).astype(np.float32)
        feat[:, start+12] = is_valid
        feat[:, start+13] = 0.0

    # 监督目标
    clean_target = clean_world[:, TARGET_LOWER_JOINTS]
    raw_target = raw_world[:, TARGET_LOWER_JOINTS]
    dq_mat = np.einsum("ntji,ntjk->ntik", raw_target, clean_target)
    dq = rotmat_to_quat_wxyz(dq_mat)

    # 门控标签 (按真实角度误差)
    dot = np.abs(np.einsum("ntk,ntk->nt", rotmat_to_quat_wxyz(raw_target), rotmat_to_quat_wxyz(clean_target)))
    dot = np.clip(dot, -1.0, 1.0)
    angle_err_deg = 2.0 * np.arccos(dot) * 180.0 / np.pi
    gate = np.clip((angle_err_deg - 0.25) / 1.25, 0.0, 1.0).astype(np.float32)

    # 双足触地代理
    foot_speed = np.linalg.norm(np.gradient(positions[:, [7, 8]], axis=0) * 60.0, axis=-1)
    contact = (foot_speed < 0.35).astype(np.float32)

    meta = {
        "mounting_error_euler": mounting_error_euler,
        "drift_signs": signs,
        "drift_rates": actual_rates,
        "num_frames": num_frames,
    }
    return feat, dq, gate, contact, meta


def process_file_d3a(item: Tuple) -> Optional[Tuple[List[dict], Optional[List[dict]]]]:
    (path_str, seq_id, subset_name, split_name, seed,
     window, stride, windows_per_sequence, drift_tier, make_counterfactual) = item
    try:
        rng = np.random.default_rng(seed)
        feat, dq, gate, contact, meta = build_sequence_d3a(
            Path(path_str), rng, drift_tier=drift_tier
        )
        num_frames = meta["num_frames"]
        max_start = max(0, num_frames - window)
        if max_start == 0:
            starts = [0]
        else:
            n_win = min(windows_per_sequence, max_start // max(1, stride) + 1)
            if n_win <= 1:
                starts = [0]
            else:
                step = max_start / max(1, n_win - 1)
                starts = sorted(list(set(int(round(i * step)) for i in range(n_win))))

        if not starts:
            return None

        seq_name = Path(path_str).stem
        std_rows = []
        for w_idx, s in enumerate(starts):
            std_rows.append({
                "x": feat[s:s+window],
                "dq": dq[s:s+window],
                "gate": gate[s:s+window],
                "contact": contact[s:s+window],
                "seq_id": np.int32(seq_id),
                "seq_name": str(seq_name),
                "subset": str(subset_name),
                "split": str(split_name),
                "drift_tier": np.float32(drift_tier),
                "drift_rates": meta["drift_rates"].astype(np.float32),
                "drift_signs": meta["drift_signs"].astype(np.float32),
                "sample_seed": np.int64(seed),
                "start_frame": np.int32(s),
                "window_idx_in_seq": np.int32(w_idx),
                "is_seq_start": np.bool_(w_idx == 0),
                "is_counterfactual": np.bool_(False),
                "fps": np.float32(60.0),
                "time_seconds": np.float32(s / 60.0),
            })

        cf_rows = None
        if make_counterfactual and drift_tier != 0.0:
            # 配对反事实: 相同安装误差, 漂移符号严格取反
            rng_cf = np.random.default_rng(seed + 999999)
            cf_signs = -meta["drift_signs"]
            c_feat, c_dq, c_gate, c_contact, c_meta = build_sequence_d3a(
                Path(path_str), rng_cf, drift_tier=drift_tier,
                drift_signs=cf_signs, fixed_mounting=meta["mounting_error_euler"]
            )
            cf_rows = []
            for w_idx, s in enumerate(starts):
                cf_rows.append({
                    "x": c_feat[s:s+window],
                    "dq": c_dq[s:s+window],
                    "gate": c_gate[s:s+window],
                    "contact": c_contact[s:s+window],
                    "seq_id": np.int32(seq_id + 500000),  # 独立 ID
                    "seq_name": str(seq_name) + "_cf",
                    "subset": str(subset_name),
                    "split": str(split_name) + "_cf",
                    "drift_tier": np.float32(-drift_tier),
                    "drift_rates": c_meta["drift_rates"].astype(np.float32),
                    "drift_signs": c_meta["drift_signs"].astype(np.float32),
                    "sample_seed": np.int64(seed + 999999),
                    "start_frame": np.int32(s),
                    "window_idx_in_seq": np.int32(w_idx),
                    "is_seq_start": np.bool_(w_idx == 0),
                    "is_counterfactual": np.bool_(True),
                    "fps": np.float32(60.0),
                    "time_seconds": np.float32(s / 60.0),
                })

        return std_rows, cf_rows
    except Exception as e:
        return None


def pack_d3a(rows: List[dict]) -> dict:
    return {
        "x": np.stack([r["x"] for r in rows]).astype(np.float32),
        "dq": np.stack([r["dq"] for r in rows]).astype(np.float32),
        "gate": np.stack([r["gate"] for r in rows]).astype(np.float32),
        "contact": np.stack([r["contact"] for r in rows]).astype(np.float32),
        "seq_id": np.array([r["seq_id"] for r in rows], dtype=np.int32),
        "seq_name": np.array([r["seq_name"] for r in rows], dtype="U64"),
        "subset": np.array([r["subset"] for r in rows], dtype="U32"),
        "split": np.array([r["split"] for r in rows], dtype="U32"),
        "drift_tier": np.array([r["drift_tier"] for r in rows], dtype=np.float32),
        "drift_rates": np.stack([r["drift_rates"] for r in rows]).astype(np.float32),
        "drift_signs": np.stack([r["drift_signs"] for r in rows]).astype(np.float32),
        "sample_seed": np.array([r["sample_seed"] for r in rows], dtype=np.int64),
        "start_frame": np.array([r["start_frame"] for r in rows], dtype=np.int32),
        "window_idx_in_seq": np.array([r["window_idx_in_seq"] for r in rows], dtype=np.int32),
        "is_seq_start": np.array([r["is_seq_start"] for r in rows], dtype=np.bool_),
        "is_counterfactual": np.array([r["is_counterfactual"] for r in rows], dtype=np.bool_),
        "fps": np.array([r["fps"] for r in rows], dtype=np.float32),
        "time_seconds": np.array([r["time_seconds"] for r in rows], dtype=np.float32),
    }


def generate_d3a_datasets(
    source_root: Path,
    output_dir: Path,
    per_subset: int = 30,
    seed: int = 20261017,
):
    verify_rot6d_contract()
    print("\n" + "=" * 80)
    print("  [D3a.1] 正在生成 D3a 无捷径与配对反事实分层实验数据集...")
    print("=" * 80)

    subsets = sorted([d for d in source_root.iterdir() if d.is_dir()])
    tier_cycle = [0.0, 1.0, 3.0, 10.0]

    train_items, val_items, dev_items = [], [], []
    global_seq_id = 0

    for s_idx, sdir in enumerate(subsets):
        sfiles = sorted(sdir.glob("*.npz"))[:per_subset]
        train_files = sfiles[:21]
        val_files = sfiles[21:25]
        dev_files = sfiles[25:30]

        for p in train_files:
            sid = global_seq_id
            global_seq_id += 1
            drift = tier_cycle[sid % 4]
            train_items.append((str(p), sid, sdir.name, "train", seed + sid * 41, 20, 5, 12, drift, False))

        for p in val_files:
            sid = global_seq_id
            global_seq_id += 1
            drift = tier_cycle[sid % 4]
            val_items.append((str(p), sid, sdir.name, "val", seed + sid * 41, 20, 5, 12, drift, False))

        for p in dev_files:
            sid = global_seq_id
            global_seq_id += 1
            drift = tier_cycle[sid % 4]
            dev_items.append((str(p), sid, sdir.name, "dev_test", seed + sid * 41, 20, 5, 12, drift, True))

    print(f"  序列划分: Train={len(train_items)} | Val={len(val_items)} | DevTest={len(dev_items)}")

    def run_split(items):
        std_list, cf_list = [], []
        with ProcessPoolExecutor(max_workers=6) as executor:
            for res in executor.map(process_file_d3a, items, chunksize=32):
                if res is not None:
                    s_rows, c_rows = res
                    std_list.extend(s_rows)
                    if c_rows is not None:
                        cf_list.extend(c_rows)
        return std_list, cf_list

    t0 = time.time()
    print("  -> 处理 Train 集中...")
    train_std, _ = run_split(train_items)
    print("  -> 处理 Val 集中...")
    val_std, _ = run_split(val_items)
    print("  -> 处理 DevTest 集中 (同时生成配对反事实样本)...")
    dev_std, dev_cf = run_split(dev_items)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "train.npz", **pack_d3a(train_std))
    np.savez_compressed(output_dir / "val.npz", **pack_d3a(val_std))
    np.savez_compressed(output_dir / "dev_test.npz", **pack_d3a(dev_std))
    np.savez_compressed(output_dir / "dev_test_cf.npz", **pack_d3a(dev_cf))

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "role": "D3a Unbiased and Counterfactual Paired Dataset",
        "mounting_error_std_deg": 0.5,
        "drift_distribution": "Independent 4-tracker signed linear drifts +/- 0, 1, 3, 10 deg/min + random walk",
        "train_windows": len(train_std),
        "val_windows": len(val_std),
        "dev_test_windows": len(dev_std),
        "dev_test_cf_windows": len(dev_cf),
        "subsets_count": len(subsets),
    }
    with open(output_dir / "d3a_dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"  数据集生成完毕 (耗时 {time.time()-t0:.2f}s):")
    print(f"    - train.npz:        {len(train_std)} 窗口")
    print(f"    - val.npz:          {len(val_std)} 窗口")
    print(f"    - dev_test.npz:     {len(dev_std)} 窗口")
    print(f"    - dev_test_cf.npz:  {len(dev_cf)} 窗口 (配对反事实)")
    return manifest


if __name__ == "__main__":
    source = resolve_path("database/amass/HumanML3D_SMPLH52")
    output = resolve_path("database/d3a_unbiased")
    generate_d3a_datasets(source, output)

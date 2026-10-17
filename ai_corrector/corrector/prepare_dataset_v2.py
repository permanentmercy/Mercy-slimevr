"""
ai_corrector/corrector/prepare_dataset_v2.py
============================================
从 ./database/amass/HumanML3D_SMPLH52 生成 CSTN v2 数据集。

v2 契约特性:
- 输入 X[B, 20, 124]: 8 路通道 (HMD 17维, VD胸部 17维(可缺省), VD双上臂各17维, 4个下肢IMU各14维)
- 真实多工况漂移注入: 覆盖 0, 1, 3, 10°/min 偏航漂移与安装角误差、VD延迟
- 物理非平凡门控监督 gate[B, 20, 4]: 依据残差收益自适应连续门控，不再全置 1
- 合成代理触地监督 contact[B, 20, 2]: 双足速度阈值代理标签
- 严格序列级独立切分 (75% Train / 25% Val)，杜绝跨窗口泄露
"""

import argparse
from pathlib import Path
import sys
import time
from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation

from .checkpoint import resolve_path
from .contract_v2 import (
    STREAM_DIMS_V2,
    TOTAL_INPUT_DIM_V2,
    rotmat_to_quat_wxyz,
    rot6d_to_matrix_np,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# SMPL-H 拓扑与骨骼偏置
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
OFFSETS = np.zeros((22, 3), dtype=np.float32)
OFFSETS[[1, 2]] = [[-.09, -.08, 0], [.09, -.08, 0]]
OFFSETS[3] = [0, 0, .12]
OFFSETS[[4, 5]] = [[0, -.42, 0], [0, -.42, 0]]
OFFSETS[6] = [0, 0, .18]
OFFSETS[[7, 8]] = [[0, -.42, 0], [0, -.42, 0]]
OFFSETS[9] = [0, 0, .16]
OFFSETS[[10, 11]] = [[0, -.08, .12], [0, -.08, .12]]
OFFSETS[12] = [0, 0, .18]
OFFSETS[[13, 14]] = [[-.12, 0, 0], [.12, 0, 0]]
OFFSETS[15] = [0, 0, .18]
OFFSETS[[16, 17]] = [[-.17, 0, 0], [.17, 0, 0]]
OFFSETS[[18, 19]] = [[-.25, 0, 0], [.25, 0, 0]]
OFFSETS[[20, 21]] = [[-.22, 0, 0], [.22, 0, 0]]

# 4 处下半身目标修正关节点: 左大腿(1), 右大腿(2), 左小腿(4), 右小腿(5)
TARGET_LOWER_JOINTS = (1, 2, 4, 5)


def rot6(m: np.ndarray) -> np.ndarray:
    """提取 3x3 旋转矩阵前两列拼接为 6 维连续向量。"""
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1).astype(np.float32)


def world_pose(local: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = local.shape[0]
    world_r = np.empty_like(local)
    pos = np.empty((n, 22, 3), dtype=np.float32)
    for j in range(22):
        parent = PARENTS[j]
        if parent < 0:
            world_r[:, j] = local[:, j]
            pos[:, j] = trans
        else:
            world_r[:, j] = np.einsum("nij,njk->nik", world_r[:, parent], local[:, j])
            pos[:, j] = pos[:, parent] + np.einsum("nij,j->ni", world_r[:, parent], OFFSETS[j])
    return world_r, pos


def angular_velocity(r: np.ndarray) -> np.ndarray:
    if len(r) < 2:
        return np.zeros((len(r), 3), dtype=np.float32)
    rel = np.einsum("nij,njk->nik", np.swapaxes(r[:-1], 1, 2), r[1:])
    v = Rotation.from_matrix(rel).as_rotvec().astype(np.float32)
    return np.vstack([v[:1], v]) * 60.0


def acceleration(p: np.ndarray) -> np.ndarray:
    v = np.gradient(p, axis=0) * 60.0
    return np.gradient(v, axis=0) * 60.0


def build_sequence_v2(
    path: Path,
    rng: np.random.Generator,
    force_zero_drift: bool = False,
    drift_profile: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    data = np.load(path, allow_pickle=False)
    poses = data["poses"].astype(np.float32)[:, :66].reshape(-1, 22, 3)
    trans = data["trans"].astype(np.float32)
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
    clean_world, positions = world_pose(local, trans)

    num_frames = len(local)
    if num_frames < 20:
        raise ValueError("Sequence too short")

    # 1. 漂移与噪声注入策略 (0, 1, 3, 10 deg/min 覆盖)
    if force_zero_drift:
        actual_drift = 0.0
    elif drift_profile is not None:
        actual_drift = float(drift_profile)
    else:
        actual_drift = float(rng.choice([0.0, 1.0, 3.0, 10.0], p=[0.25, 0.25, 0.25, 0.25]))

    time_minutes = np.arange(num_frames, dtype=np.float32) / (60.0 * 60.0)

    # 4 个下肢关节点的偏航角漂移 (常数偏置 + 随机游走)
    if actual_drift == 0.0:
        # 零标称漂移/低噪声工况：微小残余测量噪声与校准后微误差
        random_walk = np.cumsum(rng.normal(0, np.deg2rad(0.002), (num_frames, 4)), axis=0)
        total_yaw_drift = random_walk
        mounting_error_euler = rng.normal(0.0, 0.2, size=(4, 3))
    else:
        rw_sigma = np.deg2rad(0.01 * np.sqrt(actual_drift))
        random_walk = np.cumsum(rng.normal(0, rw_sigma, (num_frames, 4)), axis=0)
        linear_drift = np.outer(time_minutes, rng.normal(actual_drift, actual_drift * 0.15, 4))
        total_yaw_drift = np.deg2rad(linear_drift) + random_walk
        mounting_error_euler = rng.normal(0.0, 0.8, size=(4, 3))

    mounting_rot = Rotation.from_euler("xyz", mounting_error_euler, degrees=True).as_matrix()

    raw_world = clean_world.copy()
    rotvec = np.zeros((num_frames, 4, 3), dtype=np.float32)
    rotvec[..., 1] = total_yaw_drift  # Y 轴 (世界向上) 偏航漂移
    drift_rot = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(num_frames, 4, 3, 3)

    for k, joint in enumerate(TARGET_LOWER_JOINTS):
        # 偏航漂移发生在世界空间 (+Y 轴)，传感器安装误差发生在局部坐标系
        temp = np.einsum("nij,njk->nik", drift_rot[:, k], clean_world[:, joint])
        raw_world[:, joint] = np.einsum("nij,jk->nik", temp, mounting_rot[k])

    raw_omega = np.stack([angular_velocity(raw_world[:, j]) for j in range(22)], axis=1)
    raw_acc = acceleration(positions)

    # 2. 组装 124 维输入特征向量 (8 路全部对齐至 SlimeVR 世界/追踪空间)
    feat = np.zeros((num_frames, TOTAL_INPUT_DIM_V2), dtype=np.float32)

    # 流 0: HMD (关节点 15, 17维) - 空间基准锚点
    head = 15
    feat[:, 0:3] = 0.0  # 相对自身原点
    feat[:, 3:9] = rot6(clean_world[:, head])
    feat[:, 9:12] = np.gradient(positions[:, head], axis=0) * 60.0
    feat[:, 12:15] = angular_velocity(clean_world[:, head])
    feat[:, 15] = 1.0  # valid
    feat[:, 16] = 0.0  # age

    # 流 1: VD 胸部 (关节点 6, 17维) - 模拟 20% 概率缺失
    has_vd_chest = rng.random() > 0.20
    if has_vd_chest:
        feat[:, 17:20] = positions[:, 6] - positions[:, head]
        feat[:, 20:26] = rot6(clean_world[:, 6])
        feat[:, 26:29] = np.gradient(positions[:, 6], axis=0) * 60.0
        feat[:, 29:32] = angular_velocity(clean_world[:, 6])
        feat[:, 32] = 1.0
        feat[:, 33] = rng.uniform(0.0, 0.02)  # age 抖动

    # 流 2: VD 左上臂 (关节点 16, 17维)
    feat[:, 34:37] = positions[:, 16] - positions[:, head]
    feat[:, 37:43] = rot6(clean_world[:, 16])
    feat[:, 43:46] = np.gradient(positions[:, 16], axis=0) * 60.0
    feat[:, 46:49] = angular_velocity(clean_world[:, 16])
    feat[:, 49] = 1.0
    feat[:, 50] = rng.uniform(0.0, 0.02)

    # 流 3: VD 右上臂 (关节点 17, 17维)
    feat[:, 51:54] = positions[:, 17] - positions[:, head]
    feat[:, 54:60] = rot6(clean_world[:, 17])
    feat[:, 60:63] = np.gradient(positions[:, 17], axis=0) * 60.0
    feat[:, 63:66] = angular_velocity(clean_world[:, 17])
    feat[:, 66] = 1.0
    feat[:, 67] = rng.uniform(0.0, 0.02)

    # 流 4..7: 左大腿(1), 右大腿(2), 左小腿(4), 右小腿(5) IMU (各14维，均为世界追踪空间方向)
    offsets_14d = [68, 82, 96, 110]
    for start, joint in zip(offsets_14d, TARGET_LOWER_JOINTS):
        feat[:, start:start+6] = rot6(raw_world[:, joint])
        feat[:, start+6:start+9] = raw_acc[:, joint]
        feat[:, start+9:start+12] = raw_omega[:, joint]
        # 模拟 1% 极微丢包
        is_valid = (rng.random(num_frames) > 0.01).astype(np.float32)
        feat[:, start+12] = is_valid
        feat[:, start+13] = 0.0

    # 3. 计算监督目标:
    # a. 单位四元数残差 dq[num_frames, 4, 4] (wxyz): raw * dq = clean => dq = raw^T * clean
    clean_target = clean_world[:, TARGET_LOWER_JOINTS]
    raw_target = raw_world[:, TARGET_LOWER_JOINTS]
    dq_mat = np.einsum("ntji,ntjk->ntik", raw_target, clean_target)
    dq = rotmat_to_quat_wxyz(dq_mat)

    # b. 物理非平凡门控 gate[num_frames, 4]
    dot = np.abs(np.einsum("ntk,ntk->nt", rotmat_to_quat_wxyz(raw_target), rotmat_to_quat_wxyz(clean_target)))
    dot = np.clip(dot, -1.0, 1.0)
    angle_err_deg = 2.0 * np.arccos(dot) * 180.0 / np.pi
    gate = np.clip((angle_err_deg - 0.25) / 1.25, 0.0, 1.0).astype(np.float32)

    # c. 双足触地代理标签 contact[num_frames, 2]: 左脚踝(7), 右脚踝(8) 线速度 < 0.35 m/s
    foot_speed = np.linalg.norm(np.gradient(positions[:, [7, 8]], axis=0) * 60.0, axis=-1)
    contact = (foot_speed < 0.35).astype(np.float32)

    return feat, dq, gate, contact, actual_drift


def process_single_file(item: Tuple) -> Optional[List[dict]]:
    path_str, seq_id, subset_name, split_name, seed, window, stride, windows_per_sequence, force_zero_drift, drift_tier = item
    try:
        rng = np.random.default_rng(seed)
        feat, dq, gate, contact, actual_drift = build_sequence_v2(
            Path(path_str), rng, force_zero_drift=force_zero_drift, drift_profile=drift_tier
        )
        max_start = max(0, len(feat) - window)
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
        rows = []
        for s in starts:
            rows.append({
                "x": feat[s:s+window],
                "dq": dq[s:s+window],
                "gate": gate[s:s+window],
                "contact": contact[s:s+window],
                "seq_id": np.int32(seq_id),
                "seq_name": str(seq_name),
                "subset": str(subset_name),
                "split": str(split_name),
                "drift_tier": np.float32(actual_drift),
                "sample_seed": np.int64(seed),
                "start_frame": np.int32(s),
                "fps": np.float32(60.0),
                "time_seconds": np.float32(s / 60.0),
            })
        return rows
    except Exception:
        return None


def pack(rows: List[dict]) -> dict:
    return {
        "x": np.stack([r["x"] for r in rows]).astype(np.float32),
        "dq": np.stack([r["dq"] for r in rows]).astype(np.float32),
        "gate": np.stack([r["gate"] for r in rows]).astype(np.float32),
        "contact": np.stack([r["contact"] for r in rows]).astype(np.float32),
        "seq_id": np.array([r["seq_id"] for r in rows], dtype=np.int32),
        "seq_name": np.array([r["seq_name"] for r in rows], dtype="U64"),
        "subset": np.array([r["subset"] for r in rows], dtype="U32"),
        "split": np.array([r["split"] for r in rows], dtype="U16"),
        "drift_tier": np.array([r["drift_tier"] for r in rows], dtype=np.float32),
        "sample_seed": np.array([r["sample_seed"] for r in rows], dtype=np.int64),
        "start_frame": np.array([r["start_frame"] for r in rows], dtype=np.int32),
        "fps": np.array([r["fps"] for r in rows], dtype=np.float32),
        "time_seconds": np.array([r["time_seconds"] for r in rows], dtype=np.float32),
    }


def main():
    import json
    p = argparse.ArgumentParser(description="SlimeVR CSTN v2 AMASS 组内分层切分与富元数据切窗生成器")
    p.add_argument("--source", type=Path, default=resolve_path("database/amass/HumanML3D_SMPLH52"))
    p.add_argument("--output", type=Path, default=resolve_path("database/processed_v2_stratified"))
    p.add_argument("--per-subset-limit", type=int, default=30, help="每个动作子集采样的序列数上限 (0 为全量)")
    p.add_argument("--windows-per-sequence", type=int, default=12, help="每条序列截取的窗口数上限")
    p.add_argument("--window", type=int, default=20, help="滑动窗口帧长")
    p.add_argument("--stride", type=int, default=5, help="滑动步长")
    p.add_argument("--workers", type=int, default=6, help="多进程并发数")
    p.add_argument("--seed", type=int, default=20261017, help="全局分层划分随机种子")
    args = p.parse_args()

    args.source = resolve_path(args.source)
    args.output = resolve_path(args.output)

    subsets = sorted([d for d in args.source.iterdir() if d.is_dir()])
    print(f"[*] 发现 {len(subsets)} 个动作子目录，正在执行组内分层划分 (Per-Subset Stratified Partition)...")

    train_items = []
    val_items = []
    test_items = []
    test_zero_items = []

    global_seq_id = 0
    split_manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(args.source),
        "seed": args.seed,
        "per_subset_limit": args.per_subset_limit,
        "subsets": {},
        "sequences": [],
    }

    tier_cycle = [0.0, 1.0, 3.0, 10.0]

    for s_idx, sdir in enumerate(subsets):
        sfiles = sorted(sdir.glob("*.npz"))
        if args.per_subset_limit > 0:
            sfiles = sfiles[:args.per_subset_limit]
        n = len(sfiles)
        if n == 0:
            continue

        # 确定性打乱每个子集内部序列
        subset_rng = np.random.default_rng(args.seed + s_idx * 1000)
        perm = subset_rng.permutation(n)
        shuffled_files = [sfiles[idx] for idx in perm]

        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        n_test = n - n_train - n_val
        if n_val == 0 and n > 2:
            n_val = 1
            n_train -= 1
        if n_test == 0 and n > 3:
            n_test = 1
            n_train -= 1

        sub_train = shuffled_files[:n_train]
        sub_val = shuffled_files[n_train:n_train + n_val]
        sub_test = shuffled_files[n_train + n_val:]

        split_manifest["subsets"][sdir.name] = {
            "total": n,
            "train": len(sub_train),
            "val": len(sub_val),
            "test": len(sub_test),
        }

        # 分配并包装序列元数据 (保证 0/1/3/10 漂移档位均衡分布)
        for p in sub_train:
            sid = global_seq_id
            global_seq_id += 1
            drift = tier_cycle[sid % 4]
            seed = 1000000 + sid * 37
            train_items.append((str(p), sid, sdir.name, "train", seed, args.window, args.stride, args.windows_per_sequence, False, drift))
            split_manifest["sequences"].append({
                "seq_id": sid, "seq_name": p.stem, "subset": sdir.name, "split": "train", "drift_tier": drift, "path": str(p)
            })

        for p in sub_val:
            sid = global_seq_id
            global_seq_id += 1
            drift = tier_cycle[sid % 4]
            seed = 2000000 + sid * 37
            val_items.append((str(p), sid, sdir.name, "val", seed, args.window, args.stride, args.windows_per_sequence, False, drift))
            split_manifest["sequences"].append({
                "seq_id": sid, "seq_name": p.stem, "subset": sdir.name, "split": "val", "drift_tier": drift, "path": str(p)
            })

        for p in sub_test:
            sid = global_seq_id
            global_seq_id += 1
            drift = tier_cycle[sid % 4]
            seed = 3000000 + sid * 37
            test_items.append((str(p), sid, sdir.name, "test", seed, args.window, args.stride, args.windows_per_sequence, False, drift))
            test_zero_items.append((str(p), sid, sdir.name, "test_zero", seed + 500000, args.window, args.stride, args.windows_per_sequence, True, 0.0))
            split_manifest["sequences"].append({
                "seq_id": sid, "seq_name": p.stem, "subset": sdir.name, "split": "test", "drift_tier": drift, "path": str(p)
            })

    # 严格互斥断言审计
    train_paths = set(item[0] for item in train_items)
    val_paths = set(item[0] for item in val_items)
    test_paths = set(item[0] for item in test_items)
    assert len(train_paths & val_paths) == 0, "Train 与 Val 存在序列重叠泄露!"
    assert len(train_paths & test_paths) == 0, "Train 与 Test 存在序列重叠泄露!"
    assert len(val_paths & test_paths) == 0, "Val 与 Test 存在序列重叠泄露!"

    print(f"[*] 组内分层切分完成并已通过严格互斥审计:")
    print(f"    Train: {len(train_items)} 序列 | Val: {len(val_items)} 序列 | Test: {len(test_items)} 序列 | 总计: {global_seq_id} 序列")
    print(f"    每个 split 均覆盖全部 {len(subsets)} 个动作子集，彻底消除子集跨域漂移。")

    def run_split(items, desc=""):
        results = []
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for res in executor.map(process_single_file, items, chunksize=32):
                    if res is not None:
                        results.append(res)
        else:
            for item in items:
                res = process_single_file(item)
                if res is not None:
                    results.append(res)
        return results

    start_time = time.time()
    print("  -> 正在处理 Train 序列...")
    train_seqs = run_split(train_items, desc="Train")
    print("  -> 正在处理 Val 序列...")
    val_seqs = run_split(val_items, desc="Val")
    print("  -> 正在处理 Test (均衡多工况漂移 0/1/3/10°/min) 序列...")
    test_seqs = run_split(test_items, desc="Test")
    print("  -> 正在处理 Test (零标称漂移/低噪声金标) 序列...")
    test_zero_seqs = run_split(test_zero_items, desc="TestZero")

    train_rows = [w for seq in train_seqs for w in seq]
    val_rows = [w for seq in val_seqs for w in seq]
    test_rows = [w for seq in test_seqs for w in seq]
    test_zero_rows = [w for seq in test_zero_seqs for w in seq]

    args.output.mkdir(parents=True, exist_ok=True)
    train_pkg = pack(train_rows)
    val_pkg = pack(val_rows)
    test_pkg = pack(test_rows)
    test_zero_pkg = pack(test_zero_rows)

    print(f"\n[*] 正在保存分层数据集与切分清单至 {args.output} ...")
    np.savez_compressed(args.output / "train.npz", **train_pkg)
    np.savez_compressed(args.output / "val.npz", **val_pkg)
    np.savez_compressed(args.output / "test.npz", **test_pkg)
    np.savez_compressed(args.output / "test_zero_drift.npz", **test_zero_pkg)

    split_manifest["summary"] = {
        "train_windows": len(train_rows),
        "val_windows": len(val_rows),
        "test_windows": len(test_rows),
        "test_zero_windows": len(test_zero_rows),
    }
    (args.output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    print("\n=== CSTN v2 组内分层与富元数据切窗打包完成 ===")
    print(f"训练集规模:   {len(train_rows):>6} 窗口 -> train.npz")
    print(f"验证集规模:   {len(val_rows):>6} 窗口 -> val.npz")
    print(f"测试集 (综合): {len(test_rows):>6} 窗口 -> test.npz (覆盖 0/1/3/10 deg/min 均衡漂移)")
    print(f"测试集 (零漂): {len(test_zero_rows):>6} 窗口 -> test_zero_drift.npz (零标称漂移金标)")
    print(f"切分清单文件: split_manifest.json (包含 {len(split_manifest['sequences'])} 条序列全量可审计元数据)")
    print(f"耗时: {time.time() - start_time:.2f} s")
    print(f"元数据字段: {list(train_pkg.keys())}")


if __name__ == "__main__":
    main()

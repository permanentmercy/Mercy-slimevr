"""
corrector/prepare_dataset.py
=============================
从 ./database/amass/HumanML3D_SMPLH52 下全量 38 个子集提取切片，
构建全量训练与验证数据集 (train.npz / val.npz)。

特性:
- 均衡采样遍历 38 个动作子集 (CMU 日常 23 组 + HDM05 技巧 15 组)
- 注入 6 处下半身独立偏航角累积随机游走与真实动力学高频噪声
- 计算目标增量四元数 dq 与双足支撑相真实触地标签 contact
- 序列级独立划分 (75% Train / 25% Val)，杜绝跨窗口数据泄露
"""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .checkpoint import resolve_path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

# 目标关节点：胸(6)、髋(0)、左大腿(1)、右大腿(2)、左小腿(4)、右小腿(5)
TARGET = (6, 0, 1, 2, 4, 5)


def rot6(m):
    return m[..., :, :2].reshape(*m.shape[:-2], 6)


def quat_wxyz(m):
    original = m.shape[:-2]
    q = Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat().reshape(*original, 4)  # xyzw
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1).astype(np.float32)


def world_pose(local, trans):
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


def angular_velocity(r):
    if len(r) < 2:
        return np.zeros((len(r), 3), dtype=np.float32)
    rel = np.einsum("nij,njk->nik", np.swapaxes(r[:-1], 1, 2), r[1:])
    v = Rotation.from_matrix(rel).as_rotvec().astype(np.float32)
    return np.vstack([v[:1], v]) * 60.0


def acceleration(p):
    v = np.gradient(p, axis=0) * 60.0
    return np.gradient(v, axis=0) * 60.0


def build_sequence(path, rng):
    data = np.load(path, allow_pickle=False)
    poses = data["poses"].astype(np.float32)[:, :66].reshape(-1, 22, 3)
    trans = data["trans"].astype(np.float32)
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
    _, positions = world_pose(local, trans)

    # 注入独立角随机游走与零偏偏航漂移
    drift = np.cumsum(rng.normal(0, np.deg2rad(.035), (len(local), 6)), axis=0)
    raw_local = local.copy()
    rotvec = np.zeros((len(local), 6, 3), dtype=np.float32)
    rotvec[..., 2] = drift
    z = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(len(local), 6, 3, 3)
    for k, joint in enumerate(TARGET):
        raw_local[:, joint] = np.einsum("nij,njk->nik", z[:, k], raw_local[:, joint])

    raw_world, raw_positions = world_pose(raw_local, trans)
    omega = np.stack([angular_velocity(raw_local[:, j]) for j in range(22)], axis=1)
    acc = acceleration(raw_positions)

    feat = np.zeros((len(local), 135), dtype=np.float32)
    # 流 0: HMD/头显参考锚 (17维)
    head = 15
    feat[:, :3] = positions[:, head] - positions[:, 15]
    feat[:, 3:9] = rot6(raw_world[:, head])
    feat[:, 9:12] = np.gradient(positions[:, head], axis=0) * 60.0
    feat[:, 12:15] = angular_velocity(raw_world[:, head])
    feat[:, 15] = 1.0  # valid
    feat[:, 16] = 0.0  # age

    # 流 1: 胸部 (6) 与髋部 (0) IMU (共28维)
    for start, joint in ((17, 6), (31, 0)):
        feat[:, start:start+6] = rot6(raw_local[:, joint])
        feat[:, start+6:start+9] = acc[:, joint]
        feat[:, start+9:start+12] = omega[:, joint]
        feat[:, start+12] = 1.0

    # 流 2, 3: 左手 (20) 与右手 (21) 视觉追踪锚 (各17维)
    for start, joint in ((45, 20), (62, 21)):
        feat[:, start:start+3] = positions[:, joint] - positions[:, head]
        feat[:, start+3:start+9] = rot6(raw_world[:, joint])
        feat[:, start+9:start+12] = np.gradient(positions[:, joint], axis=0) * 60.0
        feat[:, start+12:start+15] = angular_velocity(raw_world[:, joint])
        feat[:, start+15] = 1.0

    # 流 4..7: 左大腿(1)、右大腿(2)、左小腿(4)、右小腿(5) IMU (各14维)
    for start, joint in ((79, 1), (93, 2), (107, 4), (121, 5)):
        feat[:, start:start+6] = rot6(raw_local[:, joint])
        feat[:, start+6:start+9] = acc[:, joint]
        feat[:, start+9:start+12] = omega[:, joint]
        feat[:, start+12] = 1.0

    # 监督残差单位四元数 dq: [N, 6, 4] (wxyz)
    clean = local[:, TARGET]
    raw = raw_local[:, TARGET]
    dq = quat_wxyz(np.einsum("ntij,ntjk->ntik", np.swapaxes(raw, -1, -2), clean))

    # 触地判定标签: 基于脚踝线速度 (< 0.35 m/s 为支撑相)
    foot_speed = np.linalg.norm(np.gradient(positions[:, [7, 8]], axis=0) * 60.0, axis=-1)
    contact = (foot_speed < .35).astype(np.float32)

    return feat, dq, np.ones((len(local), 6), dtype=np.float32), contact


def process_single_file(item):
    path_str, seed, window, stride, windows_per_sequence = item
    try:
        rng = np.random.default_rng(seed)
        feat, dq, gate, contact = build_sequence(Path(path_str), rng)
        starts = list(range(0, max(0, len(feat) - window + 1), stride))[:windows_per_sequence]
        if not starts:
            return None
        return [(feat[s:s+window], dq[s:s+window], gate[s:s+window], contact[s:s+window]) for s in starts]
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=resolve_path("database/amass/HumanML3D_SMPLH52"))
    p.add_argument("--output", type=Path, default=resolve_path("database/processed"))
    p.add_argument("--per-subset-limit", type=int, default=0, help="每个动作子集最多采样的序列数 (0为全部启用)")
    p.add_argument("--windows-per-sequence", type=int, default=12, help="每条序列最多截取的窗口数")
    p.add_argument("--window", type=int, default=20, help="滑动窗口帧长")
    p.add_argument("--stride", type=int, default=5, help="滑动步长")
    p.add_argument("--workers", type=int, default=8, help="并发多进程数 (0或1为单进程)")
    args = p.parse_args()

    args.source = resolve_path(args.source)
    args.output = resolve_path(args.output)

    # 遍历所有 38 个子目录，均衡收集序列文件
    subsets = sorted([d for d in args.source.iterdir() if d.is_dir()])
    print(f"[*] 发现 {len(subsets)} 个动作子目录，正在均衡采样...")

    selected_files = []
    for sdir in subsets:
        sfiles = sorted(sdir.glob("*.npz"))
        if args.per_subset_limit > 0:
            selected_files.extend(sfiles[:args.per_subset_limit])
        else:
            selected_files.extend(sfiles)

    print(f"[*] 总计选取 {len(selected_files)} 条动捕序列准备切窗...")

    sequence_windows = []
    processed = 0
    start_time = time.time()

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        items = [(str(p), 20261017 + i, args.window, args.stride, args.windows_per_sequence) for i, p in enumerate(selected_files)]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i, res in enumerate(executor.map(process_single_file, items, chunksize=32)):
                if res is not None:
                    sequence_windows.append(res)
                    processed += 1
                if (i + 1) % 1000 == 0 or (i + 1) == len(selected_files):
                    elapsed = time.time() - start_time
                    rate = (i + 1) / max(0.1, elapsed)
                    eta = (len(selected_files) - (i + 1)) / max(0.1, rate)
                    print(f"  进度: [{i+1}/{len(selected_files)}] ({rate:.1f} seq/s, ETA: {eta:.0f}s) | 有效序列: {processed} | 累计窗口: {sum(map(len, sequence_windows))}")
    else:
        rng = np.random.default_rng(20261017)
        for i, path in enumerate(selected_files):
            try:
                feat, dq, gate, contact = build_sequence(path, rng)
            except Exception:
                continue

            starts = list(range(0, max(0, len(feat) - args.window + 1), args.stride))[:args.windows_per_sequence]
            if starts:
                sequence_windows.append([(feat[s:s+args.window], dq[s:s+args.window], gate[s:s+args.window], contact[s:s+args.window]) for s in starts])
                processed += 1

            if (i + 1) % 1000 == 0 or (i + 1) == len(selected_files):
                elapsed = time.time() - start_time
                rate = (i + 1) / max(0.1, elapsed)
                eta = (len(selected_files) - (i + 1)) / max(0.1, rate)
                print(f"  进度: [{i+1}/{len(selected_files)}] ({rate:.1f} seq/s, ETA: {eta:.0f}s) | 有效序列: {processed} | 累计窗口: {sum(map(len, sequence_windows))}")

    if not sequence_windows:
        raise SystemExit("错误：未提取到任何有效窗口")

    # 序列级严格切分 (75% Train / 25% Val)，杜绝跨窗口泄露
    split_idx = max(1, int(len(sequence_windows) * 0.75))
    train_rows = [w for seq in sequence_windows[:split_idx] for w in seq]
    val_rows = [w for seq in sequence_windows[split_idx:] for w in seq]

    def pack(rows):
        return {k: np.stack([r[idx] for r in rows]).astype(np.float32) for idx, k in enumerate(("x", "dq", "gate", "contact"))}

    args.output.mkdir(parents=True, exist_ok=True)
    train_pkg = pack(train_rows)
    val_pkg = pack(val_rows)

    print(f"\n[*] 正在保存全量数据集至 {args.output} ...")
    np.savez_compressed(args.output / "train.npz", **train_pkg)
    np.savez_compressed(args.output / "val.npz", **val_pkg)

    train_mb = (args.output / "train.npz").stat().st_size / 1024 / 1024
    val_mb = (args.output / "val.npz").stat().st_size / 1024 / 1024

    print("\n=== 数据集打包完成 ===")
    print(f"有效序列数: {len(sequence_windows)} (Train: {split_idx}, Val: {len(sequence_windows)-split_idx})")
    print(f"训练集规模: {len(train_rows)} 窗口 (张量形状: {train_pkg['x'].shape}) -> train.npz ({train_mb:.2f} MB)")
    print(f"验证集规模: {len(val_rows)} 窗口 (张量形状: {val_pkg['x'].shape}) -> val.npz ({val_mb:.2f} MB)")


if __name__ == "__main__":
    main()

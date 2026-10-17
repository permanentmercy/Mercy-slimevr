"""Build a small, reproducible training/validation subset from HumanML3D/AMASS.

The source files contain SMPL-H axis-angle poses. This converter uses the first
22 body joints, approximate SMPL offsets for sensor-like positions, and injects
independent yaw random walks into the six lower-body observations. It is an
engineering/training smoke pipeline; it is not a replacement for calibrated
SMPL-H forward kinematics or real sensor capture.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


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

# chest, pelvis, left/right thigh, left/right shin
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
    # Independent random walks: a separate offset for each lower-body sensor.
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
    # HMD/head anchor: position, 6D rotation, velocity, angular velocity, valid, age.
    head = 15
    feat[:, :3] = positions[:, head] - positions[:, 15]
    feat[:, 3:9] = rot6(raw_world[:, head])
    feat[:, 9:12] = np.gradient(positions[:, head], axis=0) * 60.0
    feat[:, 12:15] = angular_velocity(raw_world[:, head])
    feat[:, 15] = 1.0
    # Stream 1: chest + pelvis IMU, 14 dimensions each.
    for start, joint in ((17, 6), (31, 0)):
        feat[:, start:start+6] = rot6(raw_local[:, joint])
        feat[:, start+6:start+9] = acc[:, joint]
        feat[:, start+9:start+12] = omega[:, joint]
        feat[:, start+12] = 1.0
    # Hand anchors.
    for start, joint in ((45, 20), (62, 21)):
        feat[:, start:start+3] = positions[:, joint] - positions[:, head]
        feat[:, start+3:start+9] = rot6(raw_world[:, joint])
        feat[:, start+9:start+12] = np.gradient(positions[:, joint], axis=0) * 60.0
        feat[:, start+12:start+15] = angular_velocity(raw_world[:, joint])
        feat[:, start+15] = 1.0
    # Four leg IMU streams.
    for start, joint in ((79, 1), (93, 2), (107, 4), (121, 5)):
        feat[:, start:start+6] = rot6(raw_local[:, joint])
        feat[:, start+6:start+9] = acc[:, joint]
        feat[:, start+9:start+12] = omega[:, joint]
        feat[:, start+12] = 1.0
    # The documented stream layout totals 135: 17+28+17+17+14*4.
    clean = local[:, TARGET]
    raw = raw_local[:, TARGET]
    dq = quat_wxyz(np.einsum("ntij,ntjk->ntik", np.swapaxes(raw, -1, -2), clean))
    foot_speed = np.linalg.norm(np.gradient(positions[:, [7, 8]], axis=0) * 60.0, axis=-1)
    contact = (foot_speed < .35).astype(np.float32)
    return feat, dq, np.ones((len(local), 6), dtype=np.float32), contact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("database/amass/HumanML3D_SMPLH52"))
    p.add_argument("--output", type=Path, default=Path("database/processed"))
    p.add_argument("--sequences", type=int, default=12)
    p.add_argument("--windows-per-sequence", type=int, default=8)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--stride", type=int, default=5)
    args = p.parse_args()
    if min(args.sequences, args.windows_per_sequence, args.window, args.stride) < 1:
        p.error("all limits must be positive")
    files = sorted(args.source.rglob("*.npz"))[:args.sequences]
    if len(files) < args.sequences:
        raise SystemExit(f"requested {args.sequences} sequences, found {len(files)}")
    rng = np.random.default_rng(20261017)
    sequence_windows = []
    for path in files:
        try:
            sequence = build_sequence(path, rng)
        except (KeyError, ValueError):
            continue
        x, dq, gate, contact = sequence
        starts = list(range(0, max(0, len(x)-args.window+1), args.stride))[:args.windows_per_sequence]
        sequence_windows.append([(x[s:s+args.window], dq[s:s+args.window], gate[s:s+args.window], contact[s:s+args.window]) for s in starts])
    if sum(map(len, sequence_windows)) < 4:
        raise SystemExit("not enough complete windows")
    # Split by source sequence before windowing to prevent near-duplicate
    # windows from the same motion leaking into validation.
    sequence_split = max(1, int(len(sequence_windows)*.75))
    train_rows = [row for group in sequence_windows[:sequence_split] for row in group]
    val_rows = [row for group in sequence_windows[sequence_split:] for row in group]
    if not val_rows:
        val_rows = train_rows[-1:]
        train_rows = train_rows[:-1]
    def pack(rows):
        return {k: np.stack([row[i] for row in rows]).astype(np.float32) for i, k in enumerate(("x", "dq", "gate", "contact"))}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/"subset_train.npz", **pack(train_rows))
    np.savez_compressed(args.output/"subset_val.npz", **pack(val_rows))
    print(f"sequences={len(files)} train_sequences={sequence_split} val_sequences={len(sequence_windows)-sequence_split} windows={len(train_rows)+len(val_rows)} train={len(train_rows)} val={len(val_rows)}")


if __name__ == "__main__":
    main()

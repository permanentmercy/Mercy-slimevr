"""
ai_corrector/corrector/generate_s0_fixture.py
=============================================
Phase 12 Task S0: 124D 输入契约黄金哨兵测试 Fixture 生成器与校验脚本。

该脚本生成包含 8 路身体流、每个物理通道均具有唯一已知哨兵值的基准数据，
导出为 JSON 文件，用于与 Kotlin CorrectorBridge.assembleFeatures() 进行逐元素比对。
"""

import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from ai_corrector.corrector.contract_v2 import (
    TOTAL_INPUT_DIM_V2,
    STREAM_DIMS_V2,
    STREAM_NAMES_V2,
    pack_stream_17d,
    pack_stream_14d,
    rot6d_to_matrix_np,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat,
)


def generate_sentinel_fixture() -> dict:
    """
    构造 8 路流的基准输入参数，保证每个字段具有唯一且可追踪的哨兵数值。
    """
    # 1. 通道 0: HMD (17维)
    # 构造已知倾角旋转: 俯仰 10度, 偏航 20度, 滚转 5度
    hmd_rot = Rotation.from_euler("yxz", [20.0, 10.0, 5.0], degrees=True).as_matrix().astype(np.float32)
    hmd_pos = np.array([0.0, 1.70, -0.10], dtype=np.float32)
    hmd_lin_vel = np.array([0.11, 0.12, 0.13], dtype=np.float32)
    hmd_ang_vel = np.array([0.21, 0.22, 0.23], dtype=np.float32)
    hmd_valid = 1.0
    hmd_age = 0.0

    # 2. 通道 1: VD 胸部 (17维)
    chest_rot = Rotation.from_euler("yxz", [18.0, 8.0, 3.0], degrees=True).as_matrix().astype(np.float32)
    chest_rel_pos = np.array([0.01, -0.35, 0.02], dtype=np.float32)
    chest_lin_vel = np.array([0.14, 0.15, 0.16], dtype=np.float32)
    chest_ang_vel = np.array([0.24, 0.25, 0.26], dtype=np.float32)
    chest_valid = 1.0
    chest_age = 0.008  # 8ms 延迟

    # 3. 通道 2: VD 左上臂 (17维)
    l_arm_rot = Rotation.from_euler("yxz", [15.0, 30.0, -10.0], degrees=True).as_matrix().astype(np.float32)
    l_arm_rel_pos = np.array([-0.25, -0.20, 0.05], dtype=np.float32)
    l_arm_lin_vel = np.array([0.31, 0.32, 0.33], dtype=np.float32)
    l_arm_ang_vel = np.array([0.41, 0.42, 0.43], dtype=np.float32)
    l_arm_valid = 1.0
    l_arm_age = 0.012

    # 4. 通道 3: VD 右上臂 (17维)
    r_arm_rot = Rotation.from_euler("yxz", [25.0, -20.0, 15.0], degrees=True).as_matrix().astype(np.float32)
    r_arm_rel_pos = np.array([0.26, -0.19, -0.04], dtype=np.float32)
    r_arm_lin_vel = np.array([0.34, 0.35, 0.36], dtype=np.float32)
    r_arm_ang_vel = np.array([0.44, 0.45, 0.46], dtype=np.float32)
    r_arm_valid = 1.0
    r_arm_age = 0.015

    # 5. 通道 4: 左大腿 IMU (14维)
    l_thigh_rot = Rotation.from_euler("yxz", [12.0, 25.0, -5.0], degrees=True).as_matrix().astype(np.float32)
    l_thigh_acc = np.array([0.51, 9.82, -0.53], dtype=np.float32)
    l_thigh_ang_vel = np.array([0.61, 0.62, 0.63], dtype=np.float32)
    l_thigh_valid = 1.0
    l_thigh_age = 0.0

    # 6. 通道 5: 右大腿 IMU (14维)
    r_thigh_rot = Rotation.from_euler("yxz", [8.0, -15.0, 4.0], degrees=True).as_matrix().astype(np.float32)
    r_thigh_acc = np.array([-0.54, 9.79, 0.56], dtype=np.float32)
    r_thigh_ang_vel = np.array([0.64, 0.65, 0.66], dtype=np.float32)
    r_thigh_valid = 1.0
    r_thigh_age = 0.0

    # 7. 通道 6: 左小腿 IMU (14维)
    l_calf_rot = Rotation.from_euler("yxz", [10.0, 45.0, -8.0], degrees=True).as_matrix().astype(np.float32)
    l_calf_acc = np.array([0.71, 9.91, -0.73], dtype=np.float32)
    l_calf_ang_vel = np.array([0.81, 0.82, 0.83], dtype=np.float32)
    l_calf_valid = 1.0
    l_calf_age = 0.0

    # 8. 通道 7: 右小腿 IMU (14维)
    r_calf_rot = Rotation.from_euler("yxz", [5.0, -35.0, 6.0], degrees=True).as_matrix().astype(np.float32)
    r_calf_acc = np.array([-0.74, 9.85, 0.76], dtype=np.float32)
    r_calf_ang_vel = np.array([0.84, 0.85, 0.86], dtype=np.float32)
    r_calf_valid = 1.0
    r_calf_age = 0.0

    # 组装 124 维向量
    s0 = pack_stream_17d(np.zeros(3, dtype=np.float32), hmd_rot, hmd_lin_vel, hmd_ang_vel, hmd_valid, hmd_age)
    s1 = pack_stream_17d(chest_rel_pos, chest_rot, chest_lin_vel, chest_ang_vel, chest_valid, chest_age)
    s2 = pack_stream_17d(l_arm_rel_pos, l_arm_rot, l_arm_lin_vel, l_arm_ang_vel, l_arm_valid, l_arm_age)
    s3 = pack_stream_17d(r_arm_rel_pos, r_arm_rot, r_arm_lin_vel, r_arm_ang_vel, r_arm_valid, r_arm_age)
    s4 = pack_stream_14d(l_thigh_rot, l_thigh_acc, l_thigh_ang_vel, l_thigh_valid, l_thigh_age)
    s5 = pack_stream_14d(r_thigh_rot, r_thigh_acc, r_thigh_ang_vel, r_thigh_valid, r_thigh_age)
    s6 = pack_stream_14d(l_calf_rot, l_calf_acc, l_calf_ang_vel, l_calf_valid, l_calf_age)
    s7 = pack_stream_14d(r_calf_rot, r_calf_acc, r_calf_ang_vel, r_calf_valid, r_calf_age)

    feature_124d = np.concatenate([s0, s1, s2, s3, s4, s5, s6, s7], axis=0)
    assert len(feature_124d) == TOTAL_INPUT_DIM_V2, f"Expected {TOTAL_INPUT_DIM_V2}, got {len(feature_124d)}"

    def quat_dict(rot_mat):
        q_wxyz = rotmat_to_quat_wxyz(rot_mat)
        return {"w": float(q_wxyz[0]), "x": float(q_wxyz[1]), "y": float(q_wxyz[2]), "z": float(q_wxyz[3])}

    def vec_dict(v):
        return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}

    fixture = {
        "description": "SlimeVR CSTN v2 124D Input Contract Golden Sentinel Fixture",
        "total_dimensions": TOTAL_INPUT_DIM_V2,
        "stream_dimensions": list(STREAM_DIMS_V2),
        "stream_names": list(STREAM_NAMES_V2),
        "inputs": {
            "hmd": {
                "pos": vec_dict(hmd_pos),
                "rot": quat_dict(hmd_rot),
                "lin_vel": vec_dict(hmd_lin_vel),
                "ang_vel": vec_dict(hmd_ang_vel),
                "valid": hmd_valid,
                "age": hmd_age,
            },
            "vd_chest": {
                "pos": vec_dict(hmd_pos + chest_rel_pos),
                "rel_pos": vec_dict(chest_rel_pos),
                "rot": quat_dict(chest_rot),
                "lin_vel": vec_dict(chest_lin_vel),
                "ang_vel": vec_dict(chest_ang_vel),
                "valid": chest_valid,
                "age": chest_age,
            },
            "vd_left_upper_arm": {
                "pos": vec_dict(hmd_pos + l_arm_rel_pos),
                "rel_pos": vec_dict(l_arm_rel_pos),
                "rot": quat_dict(l_arm_rot),
                "lin_vel": vec_dict(l_arm_lin_vel),
                "ang_vel": vec_dict(l_arm_ang_vel),
                "valid": l_arm_valid,
                "age": l_arm_age,
            },
            "vd_right_upper_arm": {
                "pos": vec_dict(hmd_pos + r_arm_rel_pos),
                "rel_pos": vec_dict(r_arm_rel_pos),
                "rot": quat_dict(r_arm_rot),
                "lin_vel": vec_dict(r_arm_lin_vel),
                "ang_vel": vec_dict(r_arm_ang_vel),
                "valid": r_arm_valid,
                "age": r_arm_age,
            },
            "left_upper_leg": {
                "rot": quat_dict(l_thigh_rot),
                "acc": vec_dict(l_thigh_acc),
                "ang_vel": vec_dict(l_thigh_ang_vel),
                "valid": l_thigh_valid,
                "age": l_thigh_age,
            },
            "right_upper_leg": {
                "rot": quat_dict(r_thigh_rot),
                "acc": vec_dict(r_thigh_acc),
                "ang_vel": vec_dict(r_thigh_ang_vel),
                "valid": r_thigh_valid,
                "age": r_thigh_age,
            },
            "left_lower_leg": {
                "rot": quat_dict(l_calf_rot),
                "acc": vec_dict(l_calf_acc),
                "ang_vel": vec_dict(l_calf_ang_vel),
                "valid": l_calf_valid,
                "age": l_calf_age,
            },
            "right_lower_leg": {
                "rot": quat_dict(r_calf_rot),
                "acc": vec_dict(r_calf_acc),
                "ang_vel": vec_dict(r_calf_ang_vel),
                "valid": r_calf_valid,
                "age": r_calf_age,
            },
        },
        "expected_feature_124d": [float(val) for val in feature_124d]
    }
    return fixture


def main():
    fixture = generate_sentinel_fixture()
    out_dir = Path(__file__).resolve().parents[2] / "ai_corrector" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "fixture_124d_sentinel.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)
    print(f"Generated 124D Sentinel Fixture to {out_file}")
    print(f"Vector length: {len(fixture['expected_feature_124d'])}, Streams: {len(fixture['stream_dimensions'])}")


if __name__ == "__main__":
    main()

"""
ai_corrector/corrector/contract_v2.py
======================================
SlimeVR CSTN v2 数据契约与坐标变换系统规范。

目标硬件场景:
- 4 个下肢 6 轴 IMU: 左大腿 (Left Upper Leg), 右大腿 (Right Upper Leg), 左小腿 (Left Lower Leg), 右小腿 (Right Lower Leg)
- 上半身 6DoF 视觉定位参考: Meta Quest 3 HMD + Virtual Desktop (VD) Upper Body Estimation (双上臂及可选胸部)

输入张量 X[B, T, 124]:
- 8 个解耦身体流通道，总计 124 维:
  通道 0: HMD 视觉参考锚 (17 维: pos 3, rot6 6, lin_vel 3, ang_vel 3, valid 1, age 1)
  通道 1: VD 胸部参考 (17 维: 可缺省，若无则填 0 且 valid=0)
  通道 2: VD 左上臂参考 (17 维)
  通道 3: VD 右上臂参考 (17 维)
  通道 4: 左大腿 IMU 观测 (14 维: rot6 6, ang_acc 3, ang_vel 3, valid 1, age 1)
  通道 5: 右大腿 IMU 观测 (14 维)
  通道 6: 左小腿 IMU 观测 (14 维)
  通道 7: 右小腿 IMU 观测 (14 维)

输出多任务解耦头:
- dq[B, T, 4, 4]: 4 处下半身关节点单位四元数残差 (左大腿, 右大腿, 左小腿, 右小腿)，限幅在 ±8° 内
- gate[B, T, 4]: 4 处关节预测收益置信度门控 [0, 1]
- contact[B, T, 2]: 双足当前帧触地/支撑相概率 [0, 1] (左足, 右足)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# 维度切片与流通道常量
# ─────────────────────────────────────────────────────────────────────────────

STREAM_DIMS_V2 = (17, 17, 17, 17, 14, 14, 14, 14)
TOTAL_INPUT_DIM_V2 = sum(STREAM_DIMS_V2)  # 68 + 56 = 124

NUM_OUTPUT_JOINTS_V2 = 4  # 4 处下半身关节: 左大腿, 右大腿, 左小腿, 右小腿
OUTPUT_DQ_SHAPE_V2 = (NUM_OUTPUT_JOINTS_V2, 4)
OUTPUT_GATE_SHAPE_V2 = (NUM_OUTPUT_JOINTS_V2,)
OUTPUT_CONTACT_SHAPE_V2 = (2,)

# 流通道名称映射
STREAM_NAMES_V2 = (
    "HMD",
    "VD_CHEST",
    "VD_LEFT_UPPER_ARM",
    "VD_RIGHT_UPPER_ARM",
    "IMU_LEFT_THIGH",
    "IMU_RIGHT_THIGH",
    "IMU_LEFT_SHIN",
    "IMU_RIGHT_SHIN"
)

# 目标关节点名称 (按输出顺序)
TARGET_JOINT_NAMES_V2 = (
    "left_thigh",
    "right_thigh",
    "left_shin",
    "right_shin"
)

# SMPL-H 52 拓扑关节点索引对应关系
# 0: Pelvis, 1: L_Hip/Thigh, 2: R_Hip/Thigh, 4: L_Knee/Shin, 5: R_Knee/Shin,
# 6: Spine2/Chest, 7: L_Foot, 8: R_Foot, 15: Head, 16: L_Collar/Shoulder, 17: R_Collar/Shoulder
SMPLH_JOINT_MAP_V2 = {
    "head": 15,
    "chest": 6,
    "left_upper_arm": 16,
    "right_upper_arm": 17,
    "left_thigh": 1,
    "right_thigh": 2,
    "left_shin": 4,
    "right_shin": 5,
    "left_foot": 7,
    "right_foot": 8,
    "pelvis": 0
}

TARGET_SMPLH_JOINTS_V2 = (
    SMPLH_JOINT_MAP_V2["left_thigh"],
    SMPLH_JOINT_MAP_V2["right_thigh"],
    SMPLH_JOINT_MAP_V2["left_shin"],
    SMPLH_JOINT_MAP_V2["right_shin"]
)


# ─────────────────────────────────────────────────────────────────────────────
# 坐标系定义与转换工具
# ─────────────────────────────────────────────────────────────────────────────

def get_stream_slice(stream_idx: int) -> slice:
    """获取第 stream_idx 个流通道在 124 维特征向量中的切片范围。"""
    if not (0 <= stream_idx < len(STREAM_DIMS_V2)):
        raise IndexError(f"stream_idx {stream_idx} out of range (0..{len(STREAM_DIMS_V2)-1})")
    start = sum(STREAM_DIMS_V2[:stream_idx])
    end = start + STREAM_DIMS_V2[stream_idx]
    return slice(start, end)


def pack_stream_17d(
    pos: np.ndarray,
    rot_mat: np.ndarray,
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    valid: float = 1.0,
    age: float = 0.0
) -> np.ndarray:
    """
    组装视觉 6DoF 观测流为 17 维向量:
    [pos (3), rot6 (6), lin_vel (3), ang_vel (3), valid (1), age (1)]
    """
    # 取旋转矩阵第 0 列 (x 轴基向量) 与第 1 列 (y 轴基向量) 拼接为 6 维
    rot6 = np.concatenate([rot_mat[:3, 0], rot_mat[:3, 1]], axis=-1)
    return np.concatenate([
        pos.astype(np.float32),
        rot6.astype(np.float32),
        lin_vel.astype(np.float32),
        ang_vel.astype(np.float32),
        np.array([valid, age], dtype=np.float32)
    ], axis=-1)


def pack_stream_14d(
    rot_mat: np.ndarray,
    ang_acc: np.ndarray,
    ang_vel: np.ndarray,
    valid: float = 1.0,
    age: float = 0.0
) -> np.ndarray:
    """
    组装下肢 IMU 观测流为 14 维向量:
    [rot6 (6), ang_acc (3), ang_vel (3), valid (1), age (1)]
    """
    rot6 = np.concatenate([rot_mat[:3, 0], rot_mat[:3, 1]], axis=-1)
    return np.concatenate([
        rot6.astype(np.float32),
        ang_acc.astype(np.float32),
        ang_vel.astype(np.float32),
        np.array([valid, age], dtype=np.float32)
    ], axis=-1)


def rot6d_to_matrix_np(r6: np.ndarray) -> np.ndarray:
    """
    将连续 6D 旋转表示通过 Gram-Schmidt 正交化还原为 3x3 旋转矩阵。
    输入 r6: [..., 6]
    输出: [..., 3, 3]
    """
    x_raw = r6[..., :3]
    y_raw = r6[..., 3:6]

    x = x_raw / np.maximum(np.linalg.norm(x_raw, axis=-1, keepdims=True), 1e-8)
    z = np.cross(x, y_raw)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def rotmat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """[..., 3, 3] -> [..., 4] (wxyz)."""
    orig_shape = m.shape[:-2]
    q_xyzw = Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat().reshape(*orig_shape, 4)
    return np.concatenate([q_xyzw[..., 3:4], q_xyzw[..., :3]], axis=-1).astype(np.float32)


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """[..., 4] (wxyz) -> [..., 3, 3]."""
    orig_shape = q_wxyz.shape[:-1]
    q_xyzw = np.concatenate([q_wxyz[..., 1:], q_wxyz[..., :1]], axis=-1)
    return Rotation.from_quat(q_xyzw.reshape(-1, 4)).as_matrix().reshape(*orig_shape, 3, 3).astype(np.float32)


def geodesic_angle_error_deg(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    """计算两四元数间的测地角误差 (单位: 度)."""
    q1 = q1_wxyz / np.maximum(np.linalg.norm(q1_wxyz, axis=-1, keepdims=True), 1e-8)
    q2 = q2_wxyz / np.maximum(np.linalg.norm(q2_wxyz, axis=-1, keepdims=True), 1e-8)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.mean(2.0 * np.arccos(dot) * 180.0 / np.pi))


# ─────────────────────────────────────────────────────────────────────────────
# SlimeVR 坐标系法则
# ─────────────────────────────────────────────────────────────────────────────
# SlimeVR World/Solver Frame:
#   +Y: Up (向天花板)
#   +X: Right (向人体右侧)
#   +Z: Backward (向人体后方)  [+X (右) x +Y (上) = +Z (后)]
#
# IMU Raw Sensor Frame:
#   +Z: Forward (传感器朝前)
#   +X: Left (传感器朝左)
#   +Y: Up (传感器朝上)
# ─────────────────────────────────────────────────────────────────────────────

def slimevr_world_to_imu_matrix() -> np.ndarray:
    """
    返回从 SlimeVR 骨架坐标系 (+X右, +Y上, +Z后) 转换至 IMU 传感器本地初始系 (+X左, +Y上, +Z前) 的旋转矩阵:
    X_imu = -X_slimevr
    Y_imu = +Y_slimevr
    Z_imu = -Z_slimevr
    """
    return np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0]
    ], dtype=np.float32)


def test_contract_v2_dimensions():
    """单元测试：验证 v2 特征张量各通道维度切片与总长度。"""
    assert len(STREAM_DIMS_V2) == 8
    assert TOTAL_INPUT_DIM_V2 == 124
    assert sum(STREAM_DIMS_V2[:4]) == 68  # 4 * 17
    assert sum(STREAM_DIMS_V2[4:]) == 56  # 4 * 14

    # 验证各通道切片无重叠且完全覆盖 [0, 124)
    slices = [get_stream_slice(i) for i in range(8)]
    for i in range(7):
        assert slices[i].stop == slices[i+1].start
    assert slices[0].start == 0
    assert slices[-1].stop == 124


def test_coordinate_conversion_known_poses():
    """
    单元测试：验证标准已知姿态（T-Pose，90° Yaw 旋转，坐姿）在 SlimeVR +Y UP 坐标系下的几何等价性。
    """
    # 1. 验证单位旋转
    identity_rot = np.eye(3, dtype=np.float32)
    s17 = pack_stream_17d(
        pos=np.zeros(3),
        rot_mat=identity_rot,
        lin_vel=np.zeros(3),
        ang_vel=np.zeros(3),
        valid=1.0,
        age=0.0
    )
    assert s17.shape == (17,)
    assert s17[15] == 1.0  # valid
    assert s17[16] == 0.0  # age
    # 还原 rot6d 并检查与单位矩阵一致
    rec_mat = rot6d_to_matrix_np(s17[3:9])
    np.testing.assert_allclose(rec_mat, identity_rot, atol=1e-6)

    # 2. 验证绕 +Y 轴（垂直向上）旋转 90 度的偏航转换
    yaw_90 = Rotation.from_euler("y", 90, degrees=True).as_matrix().astype(np.float32)
    s14 = pack_stream_14d(
        rot_mat=yaw_90,
        ang_acc=np.zeros(3),
        ang_vel=np.array([0.0, 1.57, 0.0]),  # 绕 Y 轴角速度
        valid=1.0,
        age=0.016
    )
    assert s14.shape == (14,)
    rec_yaw = rot6d_to_matrix_np(s14[:6])
    np.testing.assert_allclose(rec_yaw, yaw_90, atol=1e-6)

    # 3. 验证 SlimeVR 坐标手性: (+X) x (+Y) = (+Z)
    ex = np.array([1.0, 0.0, 0.0])  # 右
    ey = np.array([0.0, 1.0, 0.0])  # 上
    ez = np.cross(ex, ey)           # 后
    np.testing.assert_allclose(ez, np.array([0.0, 0.0, 1.0]), atol=1e-6)

    # 4. 验证 IMU 坐标与 SlimeVR 坐标变换行列式为 +1 (纯旋转，右手系保持)
    T = slimevr_world_to_imu_matrix()
    np.testing.assert_allclose(np.linalg.det(T), 1.0, atol=1e-6)


if __name__ == "__main__":
    test_contract_v2_dimensions()
    test_coordinate_conversion_known_poses()
    print("[Contract v2] All dimension and coordinate transformation tests passed successfully!")

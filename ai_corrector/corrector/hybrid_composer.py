"""
ai_corrector/corrector/hybrid_composer.py
=========================================
Phase 11 任务 H1: Phase 9 慢偏航与 Phase 6 快姿态的正交 Swing-Twist 合成算子

核心数学原理:
1. 世界竖直轴 u_world = [0, 1, 0]^T (+Y).
2. Phase 9 慢分支世界左乘逆偏航:
     R9_i = R_y(-e_hat_i) @ R_raw_i
3. Phase 6 快分支候选姿态:
     R6_candidate = R9_i @ delta_R6
4. 世界增量差分:
     D = R6_candidate @ R9_i.T
5. 世界竖直轴 Swing-Twist 分解:
     D = D_swing @ D_twist
   其中 D_twist 严格为绕世界 Y 轴旋转, D_swing 绕世界垂直轴的旋转角严格为 0。
6. 严格丢弃 D_twist, 仅将经门控调制的 D_swing 作用于 R9:
     Rout_i = D_swing_gated @ R9_i
7. SlimeVR 服务端局部右乘等价增量:
     delta_R_server = R_raw_i.T @ Rout_i  (使得 R_raw_i @ delta_R_server == Rout_i)

正交性保证:
无论 Phase 6 预测出何种姿态残差, D_swing 绝对不改变 R9 的世界偏航角;
Phase 9 慢估计的偏航漂移状态具有 100% 免疫保护。
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .contract_v3 import LOWER_BODY_JOINTS
from .oracle_v3_geometry import rot_y_matrix, geodesic_angle_rad, geodesic_angle_deg, wrap_angle


def rotmat_to_quat_wxyz_safe(R: np.ndarray) -> np.ndarray:
    """
    将 3x3 旋转矩阵转换为归一化四元数 [w, x, y, z], 保证 w >= 0
    采用稳健的 Shepperd 算法分支, 避免退化
    """
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return q


def quat_wxyz_to_rotmat_safe(q: np.ndarray) -> np.ndarray:
    """归一化四元数 [w, x, y, z] 转换为 3x3 旋转矩阵"""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)]
    ], dtype=np.float64)


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 q_out = q1 * q2, 约定 [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=np.float64)


def swing_twist_decomposition_y(q_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    针对世界竖直 Y 轴 (u = [0, 1, 0]^T) 的 Swing-Twist 分解。
    q = q_swing * q_twist
    
    数学特性:
    - q_twist 具有纯 Y 轴旋转: q_twist = [cos(theta/2), 0, sin(theta/2), 0]
    - q_swing 的 Y 轴四元数分量恒为 0: q_swing[2] == 0.0 (数值精度内)
    - q_swing 绕世界 Y 轴的投影角恒为 0
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    w, x, y, z = q

    twist_norm_sq = w * w + y * y
    if twist_norm_sq < 1e-12:
        # 旋转角接近 180 度且轴正交于 Y，twist 分量退化为单位元
        q_twist = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        q_swing = q / np.linalg.norm(q)
    else:
        twist_norm = math.sqrt(twist_norm_sq)
        q_twist = np.array([w / twist_norm, 0.0, y / twist_norm, 0.0], dtype=np.float64)
        # q_twist 的共轭/逆: [tw, -tx, -ty, -tz] = [w_t, 0, -y_t, 0]
        q_twist_inv = np.array([q_twist[0], 0.0, -q_twist[2], 0.0], dtype=np.float64)
        q_swing = quat_multiply_wxyz(q, q_twist_inv)
        q_swing /= np.linalg.norm(q_swing)

    if q_swing[0] < 0.0:
        q_swing = -q_swing
    return q_swing, q_twist


def slerp_from_identity_wxyz(q_wxyz: np.ndarray, weight: float) -> np.ndarray:
    """
    四元数从单位元 [1, 0, 0, 0] 到 q_wxyz 的球面线性插值 (SLERP)。
    weight in [0.0, 1.0]
    """
    w_clamped = max(0.0, min(1.0, float(weight)))
    if w_clamped <= 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if w_clamped >= 1.0 - 1e-6:
        return q_wxyz

    q = np.asarray(q_wxyz, dtype=np.float64)
    if q[0] < 0.0:
        q = -q

    cos_half = np.clip(q[0], -1.0, 1.0)
    theta = 2.0 * math.acos(cos_half)
    if theta < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    sin_half = math.sin(theta / 2.0)
    axis = q[1:] / sin_half
    theta_gated = theta * w_clamped
    return np.array([
        math.cos(theta_gated / 2.0),
        axis[0] * math.sin(theta_gated / 2.0),
        axis[1] * math.sin(theta_gated / 2.0),
        axis[2] * math.sin(theta_gated / 2.0)
    ], dtype=np.float64)


@dataclass
class SingleJointHybridOutput:
    """单颗 Tracker 的正交组合输出与审计记录"""
    imu_id: str
    R_raw: np.ndarray                   # 原始姿态 (3x3)
    R_p9: np.ndarray                    # Phase 9 慢偏航修正姿态 (3x3)
    R_p6_neutral: np.ndarray            # 仅应用 Phase 6 非偏航姿态 (3x3)
    R_out: np.ndarray                   # 最终正交合成姿态 (3x3)
    delta_R_server: np.ndarray          # 供 SlimeVR 服务端局部右乘增量 (3x3)
    e_hat_rad: float                    # Phase 9 估计偏航角 (rad)
    gate_p6: float                      # Phase 6 预测置信门控
    gate_active: bool                   # 门控是否超过阈值生效
    discarded_twist_deg: float          # 从 Phase 6 滤除的世界偏航扭转角 (deg)
    applied_swing_deg: float            # 实际生效的非偏航 Swing 摆动角 (deg)
    p9_yaw_diff_deg: float              # 合成姿态相对 P9 的偏航差异 (应恒为 0.0)


def compose_single_tracker_hybrid(
    R_raw: np.ndarray,
    e_hat_rad: float,
    delta_R6: np.ndarray,
    gate_p6: float = 1.0,
    gate_threshold: float = 0.3,
    imu_id: str = "tracker"
) -> SingleJointHybridOutput:
    """
    单颗 Tracker 的单一正交组合函数。
    
    输入:
    - R_raw: 原始 3x3 旋转矩阵
    - e_hat_rad: Phase 9 慢偏航估计 (绕世界 Y 轴旋转弧度)
    - delta_R6: Phase 6 预测局部右乘残差 (3x3 矩阵 或 4 维四元数 [w,x,y,z])
    - gate_p6: Phase 6 门控标量 [0.0, 1.0]
    - gate_threshold: 门控激活阈值 (默认 0.3)
    """
    # 1. 确保 delta_R6 为 3x3 矩阵
    if isinstance(delta_R6, np.ndarray) and delta_R6.shape == (4,):
        delta_R6_mat = quat_wxyz_to_rotmat_safe(delta_R6)
    else:
        delta_R6_mat = np.asarray(delta_R6, dtype=np.float64)

    # 2. Phase 9 慢世界左乘逆偏航
    R9 = rot_y_matrix(-e_hat_rad) @ R_raw

    # 3. 构造 Phase 6 候选姿态 R6_candidate = R9 @ delta_R6
    R6_candidate = R9 @ delta_R6_mat

    # 4. 构造世界差分矩阵 D = R6_candidate @ R9.T
    D = R6_candidate @ R9.T

    # 5. Swing-Twist 分解 D = D_swing @ D_twist (绕世界 Y 轴)
    q_D = rotmat_to_quat_wxyz_safe(D)
    q_swing, q_twist = swing_twist_decomposition_y(q_D)

    # 计算滤除的 Twist 角度与 Swing 角度
    twist_half_angle = math.acos(np.clip(abs(q_twist[0]), 0.0, 1.0))
    discarded_twist_deg = math.degrees(2.0 * twist_half_angle)

    swing_half_angle = math.acos(np.clip(abs(q_swing[0]), 0.0, 1.0))
    raw_swing_deg = math.degrees(2.0 * swing_half_angle)

    # 6. 门控过滤与 SLERP 衰减
    gate_val = float(gate_p6)
    gate_active = bool(gate_val >= gate_threshold)
    if gate_active:
        q_swing_gated = slerp_from_identity_wxyz(q_swing, gate_val)
        applied_swing_deg = raw_swing_deg * gate_val
    else:
        q_swing_gated = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        applied_swing_deg = 0.0

    D_swing = quat_wxyz_to_rotmat_safe(q_swing_gated)

    # 7. 正交合成最终姿态 Rout = D_swing @ R9
    R_out = D_swing @ R9

    # 8. 计算 Phase 6-only 非偏航对照输出 (当 e_hat = 0 时)
    # R_raw 作用 D_swing
    D_raw_cand = (R_raw @ delta_R6_mat) @ R_raw.T
    q_raw_d = rotmat_to_quat_wxyz_safe(D_raw_cand)
    q_raw_s, _ = swing_twist_decomposition_y(q_raw_d)
    if gate_active:
        q_raw_s_gated = slerp_from_identity_wxyz(q_raw_s, gate_val)
    else:
        q_raw_s_gated = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    R_p6_neutral = quat_wxyz_to_rotmat_safe(q_raw_s_gated) @ R_raw

    # 9. 导出供 SlimeVR 服务端消费的局部右乘增量 delta_R_server:
    # R_raw @ delta_R_server = R_out  =>  delta_R_server = R_raw.T @ R_out
    delta_R_server = R_raw.T @ R_out

    # 10. 验证世界竖直轴偏航严格不变性:
    # 计算 Rout @ R9.T 的 Twist 旋转分量
    diff_out_9 = R_out @ R9.T
    q_diff = rotmat_to_quat_wxyz_safe(diff_out_9)
    _, q_twist_check = swing_twist_decomposition_y(q_diff)
    diff_twist_half = math.acos(np.clip(abs(q_twist_check[0]), 0.0, 1.0))
    p9_yaw_diff_deg = math.degrees(2.0 * diff_twist_half)

    return SingleJointHybridOutput(
        imu_id=imu_id,
        R_raw=R_raw,
        R_p9=R9,
        R_p6_neutral=R_p6_neutral,
        R_out=R_out,
        delta_R_server=delta_R_server,
        e_hat_rad=e_hat_rad,
        gate_p6=gate_val,
        gate_active=gate_active,
        discarded_twist_deg=discarded_twist_deg,
        applied_swing_deg=applied_swing_deg,
        p9_yaw_diff_deg=p9_yaw_diff_deg
    )


class HybridComposer:
    """
    四路下肢 IMU 的统一正交合成处理器与审计状态记录器
    """
    def __init__(self, gate_threshold: float = 0.3):
        self.gate_threshold = gate_threshold
        self.history_records: List[Dict[str, SingleJointHybridOutput]] = []

    def reset(self):
        """重置内部历史审计记录"""
        self.history_records.clear()

    def step(
        self,
        rots_raw: Dict[str, np.ndarray],
        e_hats_rad: Dict[str, float],
        deltas_p6: Dict[str, np.ndarray],
        gates_p6: Dict[str, float]
    ) -> Dict[str, SingleJointHybridOutput]:
        """
        单帧四路下肢 IMU 姿态合成
        """
        outputs = {}
        for joint in LOWER_BODY_JOINTS:
            r_raw = rots_raw[joint]
            e_hat = e_hats_rad.get(joint, 0.0)
            dq_p6 = deltas_p6.get(joint, np.eye(3, dtype=np.float64))
            g_p6 = gates_p6.get(joint, 1.0)

            out = compose_single_tracker_hybrid(
                R_raw=r_raw,
                e_hat_rad=e_hat,
                delta_R6=dq_p6,
                gate_p6=g_p6,
                gate_threshold=self.gate_threshold,
                imu_id=joint
            )
            outputs[joint] = out

        self.history_records.append(outputs)
        return outputs

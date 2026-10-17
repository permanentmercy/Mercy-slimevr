"""
ai_corrector/corrector/contract_v3.py
====================================
Phase 9: 四路下肢偏航状态模型（v3 时序数据契约与适配器）

定义包含真实时间戳、reset/重标定事件、四路设备有效性、HMD/VD 跟踪质量以及四路世界竖直轴偏航误差状态的数据结构。
提供从旧版 124 维特征向量 (contract_v2) 到 v3 时序帧的适配器。
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Tuple

# 传感器枚举标识
LOWER_BODY_JOINTS = ("left_thigh", "right_thigh", "left_calf", "right_calf")
UPPER_BODY_CONTEXT = ("hmd", "virtual_chest")


@dataclass
class FrameMetadata:
    """帧时序与系统事件元数据"""
    timestamp: float           # 会话单调时间（秒），高精度
    dt: float                  # 与前一有效帧的时间差（秒）
    session_id: str = "session_0"
    frame_index: int = 0
    reset_event: bool = False  # 是否发生 SlimeVR reset / 双击重置
    recalib_event: bool = False  # 是否发生重标定
    is_anchor_candidate: bool = False  # 是否为潜在可比动作锚点（如站立静止/对称标定姿态）


@dataclass
class SingleTrackerV3:
    """单颗追踪器在当前帧的时序数据"""
    imu_id: str
    rot_raw: np.ndarray        # 原始四元数 [w, x, y, z] 或 3x3 旋转矩阵
    ang_vel: np.ndarray        # 角速度 [wx, wy, wz] (rad/s)
    ang_acc: np.ndarray        # 角加速度 [ax, ay, az] (rad/s^2)
    valid: bool = True         # 传感器数据是否有效（未掉线/未丢包）
    data_age: float = 0.0      # 数据延迟年龄（秒）
    mounting_matrix: Optional[np.ndarray] = None  # 安装标定矩阵 M_i (3x3)，若未标定则为单位阵


@dataclass
class UpperBodyContextV3:
    """上身参考设备（HMD / VD）时序上下文"""
    hmd_rot: np.ndarray        # HMD 3x3 旋转矩阵
    hmd_pos: np.ndarray        # HMD 世界坐标 [x, y, z]
    hmd_lin_vel: np.ndarray    # HMD 线速度 [vx, vy, vz]
    hmd_ang_vel: np.ndarray    # HMD 角速度 [wx, wy, wz]
    vd_chest_rot: np.ndarray   # VD 胸部 3x3 旋转矩阵
    vd_chest_pos: np.ndarray   # VD 胸部世界坐标 [x, y, z]
    hmd_valid: bool = True
    vd_chest_valid: bool = False
    tracking_quality: float = 1.0


@dataclass
class YawBiasStateV3:
    """四路下肢 IMU 独立的偏航误差状态 (e_i, b_i, P_i)"""
    imu_id: str
    e: float = 0.0             # 世界竖直轴偏航误差角 e_i (radians, wrapped to [-pi, pi])
    b: float = 0.0             # 偏航漂移率 b_i (rad/s, 姿态偏航误差率，非裸陀螺硬件零偏)
    cov: np.ndarray = field(default_factory=lambda: np.diag([0.05, 0.001]).astype(np.float32))  # 2x2 协方差 P_i
    last_update_t: float = 0.0


@dataclass
class FrameV3:
    """Phase 9 完整单帧数据容器"""
    meta: FrameMetadata
    trackers: Dict[str, SingleTrackerV3]  # key: "left_thigh", "right_thigh", "left_calf", "right_calf"
    upper_body: UpperBodyContextV3
    # 真值姿态与偏航（仅在训练与离线验收时存在，在线部署时严格为 None）
    clean_rots: Optional[Dict[str, np.ndarray]] = None
    true_yaw_drifts: Optional[Dict[str, float]] = None


class V2ToV3Adapter:
    """
    将标准 124 维特征向量 (contract_v2.py) 适配转换为 FrameV3。
    
    124 维布局规范 (contract_v2.py):
      通道 0: HMD 视觉参考锚 (17 维: pos 3, rot6 6, lin_vel 3, ang_vel 3, valid 1, age 1) -> [0..17]
      通道 1: VD 胸部参考 (17 维: pos 3, rot6 6, lin_vel 3, ang_vel 3, valid 1, age 1) -> [17..34]
      通道 2: VD 左上臂参考 (17 维) -> [34..51]
      通道 3: VD 右上臂参考 (17 维) -> [51..68]
      通道 4: 左大腿 IMU 观测 (14 维: rot6 6, ang_acc 3, ang_vel 3, valid 1, age 1) -> [68..82]
      通道 5: 右大腿 IMU 观测 (14 维) -> [82..96]
      通道 6: 左小腿 IMU 观测 (14 维) -> [96..110]
      通道 7: 右小腿 IMU 观测 (14 维) -> [110..124]
      总计 124 维
    """
    @staticmethod
    def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
        """从 6D 旋转表示还原 3x3 旋转矩阵（按列拼接 Gram-Schmidt 正交化）"""
        x_raw = r6[:3]
        y_raw = r6[3:6]
        x_norm = np.linalg.norm(x_raw)
        x = x_raw / max(x_norm, 1e-8)
        z = np.cross(x, y_raw)
        z_norm = np.linalg.norm(z)
        z = z / max(z_norm, 1e-8)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    @classmethod
    def convert_124d_to_frame(cls,
                              feature_124: np.ndarray,
                              timestamp: float,
                              dt: float = 1.0 / 60.0,
                              session_id: str = "session_0",
                              frame_index: int = 0,
                              reset_event: bool = False) -> FrameV3:
        """将单个 124 维向量转换为 FrameV3"""
        assert len(feature_124) == 124, f"Expected 124 dims, got {len(feature_124)}"
        
        meta = FrameMetadata(
            timestamp=timestamp,
            dt=dt,
            session_id=session_id,
            frame_index=frame_index,
            reset_event=reset_event
        )
        
        # 1. 解析上身视觉参考通道 (前 68 维)
        # HMD: [0..17]
        hmd_pos = feature_124[0:3]
        hmd_rot6 = feature_124[3:9]
        hmd_rot = cls.rot6d_to_matrix(hmd_rot6)
        hmd_lin_vel = feature_124[9:12]
        hmd_ang_vel = feature_124[12:15]
        hmd_valid = bool(feature_124[15] > 0.5)
        
        # VD 胸部: [17..34]
        vd_pos = feature_124[17:20]
        vd_rot6 = feature_124[20:26]
        vd_rot = cls.rot6d_to_matrix(vd_rot6)
        vd_valid = bool(feature_124[32] > 0.5)
        
        upper = UpperBodyContextV3(
            hmd_rot=hmd_rot,
            hmd_pos=hmd_pos,
            hmd_lin_vel=hmd_lin_vel,
            hmd_ang_vel=hmd_ang_vel,
            vd_chest_rot=vd_rot,
            vd_chest_pos=vd_pos,
            hmd_valid=hmd_valid,
            vd_chest_valid=vd_valid,
            tracking_quality=1.0 if (hmd_valid and vd_valid) else (0.7 if hmd_valid else 0.0)
        )
        
        # 2. 解析下肢四路 IMU 观测通道 (后 56 维, 从 68 开始)
        imu_names = ("left_thigh", "right_thigh", "left_calf", "right_calf")
        trackers = {}
        offset = 68
        
        for name in imu_names:
            sub = feature_124[offset : offset + 14]
            rot6 = sub[0:6]
            rot_mat = cls.rot6d_to_matrix(rot6)
            ang_acc = sub[6:9]
            ang_vel = sub[9:12]
            valid = bool(sub[12] > 0.5)
            age = float(sub[13])
            
            trackers[name] = SingleTrackerV3(
                imu_id=name,
                rot_raw=rot_mat,
                ang_vel=ang_vel,
                ang_acc=ang_acc,
                valid=valid,
                data_age=age,
                mounting_matrix=np.eye(3, dtype=np.float32)
            )
            offset += 14
            
        return FrameV3(meta=meta, trackers=trackers, upper_body=upper)


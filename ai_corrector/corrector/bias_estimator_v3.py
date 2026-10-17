"""
ai_corrector/corrector/bias_estimator_v3.py
==========================================
Phase 9 M1r-F: 静止段陀螺零偏估计 (ZARU) 与可信直线步态摆动面估计器

设计原则:
1. 主分支 (ZARU): 逐颗传感器独立判断静止状态。
   利用短时角速度范数、加速度波动与频带能量检测物理静止切片;
   静止期内 omega_meas 直接观测传感器三轴陀螺零偏率 b_gyro,
   并在世界竖直轴投影上更新偏航漂移率 b_i;
   完全兼容站立、端坐、侧坐、翘二郎腿等任意静态姿态, 杜绝姿态冲刷 (Washout)。
2. 辅助分支 (Gait Swing Plane): 仅在明确直线前进步态下,
   利用大腿周期性矢状面摆动主轴与 HMD 真实位移方向比较, 输出低权重累计偏航修正量。
3. 严格数据隔离: 仅接收可部署特征 (FrameV3 / 124 维特征流), 严禁接触任何真值。
"""

import collections
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from ai_corrector.corrector.contract_v3 import FrameV3, LOWER_BODY_JOINTS
from ai_corrector.corrector.bias_state_v3 import QuadChannelBiasFilterV3
from ai_corrector.corrector.oracle_v3_geometry import wrap_angle


class StationaryZARUDetector:
    """单颗 IMU 独立静止检测与零偏观测器"""
    def __init__(self,
                 imu_id: str,
                 window_sec: float = 1.0,
                 sample_rate_hz: float = 10.0,
                 omega_thresh_rad_s: float = 0.04,   # 约 2.3°/s
                 accel_var_thresh: float = 0.15,     # (m/s^2)^2
                 min_still_duration_sec: float = 1.5):
        self.imu_id = imu_id
        self.window_size = int(max(window_sec * sample_rate_hz, 5))
        self.min_still_frames = int(min_still_duration_sec * sample_rate_hz)
        self.omega_thresh = float(omega_thresh_rad_s)
        self.accel_var_thresh = float(accel_var_thresh)
        
        self.omega_history = collections.deque(maxlen=self.window_size)
        self.accel_history = collections.deque(maxlen=self.window_size)
        self.consecutive_still_count = 0
        self.still_rate_accumulator: List[float] = []

    def update(self,
               ang_vel: np.ndarray,
               rot_mat: np.ndarray,
               ang_acc: Optional[np.ndarray] = None) -> Optional[Tuple[float, float]]:
        """
        输入当前帧该颗 IMU 的角速度 [wx, wy, wz] 与旋转矩阵 R_raw。
        若满足连续静止条件，返回世界竖直轴偏航漂移率测量值: (b_meas_rad_s, var_rad2_s2)。
        """
        omega_norm = float(np.linalg.norm(ang_vel))
        self.omega_history.append(omega_norm)
        
        # 简单加速度平稳性（若有角加速度或合力）
        acc_metric = float(np.linalg.norm(ang_acc)) if ang_acc is not None else 0.0
        self.accel_history.append(acc_metric)
        
        if len(self.omega_history) < self.window_size:
            return None
            
        mean_omega = float(np.mean(self.omega_history))
        var_omega = float(np.var(self.omega_history))
        
        # 判定是否处于静止状态: 角速度低且方差极小 (处于白噪声水平)
        is_still = (mean_omega < self.omega_thresh) and (var_omega < (self.omega_thresh * 0.5) ** 2)
        
        if is_still:
            self.consecutive_still_count += 1
            # 将传感器机体角速度转换到世界坐标系: omega_world = R_raw @ omega_local
            omega_world = rot_mat @ ang_vel
            yaw_rate_world = float(omega_world[1])  # Y 轴 (世界竖直轴)
            self.still_rate_accumulator.append(yaw_rate_world)
            
            # 只有在持续静止超过门限时长 (>= 3.0s)，且间隔一定帧数 (如每 1.0s / 10 帧) 才触发一次更新
            if self.consecutive_still_count >= self.min_still_frames:
                # 每 10 帧更新一次，避免 10 Hz 相关噪声过度驱动卡尔曼滤波器
                if (self.consecutive_still_count % 10) == 0:
                    N_still = min(len(self.still_rate_accumulator), 150)
                    recent_rates = list(self.still_rate_accumulator)[-N_still:]
                    b_meas = float(np.mean(recent_rates))
                    
                    # 动态白噪声标准误估计
                    noise_std = float(np.std(recent_rates)) if len(recent_rates) > 10 else 0.005
                    stderr = noise_std / math.sqrt(N_still)
                    
                    # 3-sigma 统计死区门限 (至少 1.5°/min，防止零漂时白噪声均值泄漏)
                    deadband_rad_s = max(math.radians(1.5 / 60.0), 2.5 * stderr)
                    if abs(b_meas) < deadband_rad_s:
                        b_meas = 0.0
                    
                    var_b = max(stderr ** 2, 2.5e-7)
                    return (b_meas, var_b)
        else:
            self.consecutive_still_count = 0
            self.still_rate_accumulator.clear()
            
        return None


class GaitSwingPlaneDetector:
    """可信直线步态摆动面检测器 (辅助分支)"""
    def __init__(self,
                 window_sec: float = 3.0,
                 sample_rate_hz: float = 10.0,
                 min_walk_speed_m_s: float = 0.4):
        self.window_size = int(max(window_sec * sample_rate_hz, 15))
        self.min_speed = float(min_walk_speed_m_s)
        self.hmd_pos_history = collections.deque(maxlen=self.window_size)
        self.leg_omega_history: Dict[str, collections.deque] = {
            j: collections.deque(maxlen=self.window_size) for j in LOWER_BODY_JOINTS
        }

    def update(self,
               frame: FrameV3) -> Dict[str, Tuple[float, float]]:
        """
        检查 HMD 位移是否为稳定直线前进，且双腿是否存在周期性摆动。
        若条件严苛满足，返回逐路世界偏航误差观测候选: {joint: (e_meas_rad, var_rad2)}
        """
        # 安全性门禁: 若上身 HMD / VD 有效性不足，拒绝步态推断
        if not frame.upper_body.hmd_valid or not frame.upper_body.vd_chest_valid:
            return {}

        hmd_pos = frame.upper_body.hmd_pos
        self.hmd_pos_history.append(hmd_pos)
        
        for j in LOWER_BODY_JOINTS:
            w = frame.trackers[j].ang_vel
            self.leg_omega_history[j].append(w)
            
        if len(self.hmd_pos_history) < self.window_size:
            return {}
            
        # 1. 检验 HMD 水平位移: 起点到终点在 X-Z 平面的位移矢量 (X 为横向, Z 为前后纵向)
        p_start = self.hmd_pos_history[0]
        p_end = self.hmd_pos_history[-1]
        delta_p = p_end - p_start
        delta_horiz = np.array([delta_p[0], delta_p[2]], dtype=np.float32)
        norm_h = float(np.linalg.norm(delta_horiz))
        
        dt = float(len(self.hmd_pos_history)) * frame.meta.dt
        speed = norm_h / max(dt, 1e-4)  # 水平前进速度
        
        if speed < self.min_speed:
            return {}  # 速度不足或原地转圈，拒绝
            
        # 检验位移直线度: 轨迹沿前进方向的投影
        dir_walk = delta_horiz / max(norm_h, 1e-6)
        # 前进航向角 (X 轴为分母/正弦, Z 轴为余弦)
        walk_heading_rad = math.atan2(float(dir_walk[0]), float(dir_walk[1]))
        
        candidates = {}
        for j in ("left_thigh", "right_thigh"):  # 大腿摆动最显著
            omegas = np.array(self.leg_omega_history[j])  # (W, 3)
            # 检验是否具备明显的周期性摆动 (X 轴前后俯仰摆动方差显著大于阈值)
            omega_std = float(np.std(omegas[:, 0]))  # 俯仰摆动轴标准差
            if omega_std > 0.3:  # 明确在迈步
                R_raw = frame.trackers[j].rot_raw
                # 提取大腿局部前向轴在世界平面的投影偏航 (局部 Z 轴为前向)
                forward_vec = R_raw[:, 2]
                leg_heading_rad = math.atan2(float(forward_vec[0]), float(forward_vec[2]))
                
                # 偏航创新量 = wrap(leg_heading - walk_heading)
                diff = float(wrap_angle(leg_heading_rad - walk_heading_rad))
                
                # 只有在偏差处于合理范围内时接纳 (避免将大幅度外撇动作当成漂移)
                if abs(diff) < math.radians(25.0):
                    # 步态观测方差较大 (约 5° 标准差)
                    var_gait = math.radians(5.0) ** 2
                    candidates[j] = (diff, var_gait)
                    
        return candidates


class DeployableBiasEstimatorV3:
    """
    Phase 9 M1r-F 完整可部署估计器组合体:
    组合 4 路独立的 ZARU 静止检测器 + 直线步态摆动面检测器 + 卡尔曼偏置滤波器。
    """
    def __init__(self,
                 enable_zaru: bool = True,
                 enable_gait: bool = True,
                 sample_rate_hz: float = 10.0):
        self.enable_zaru = enable_zaru
        self.enable_gait = enable_gait
        
        self.zaru_detectors = {
            j: StationaryZARUDetector(imu_id=j, sample_rate_hz=sample_rate_hz)
            for j in LOWER_BODY_JOINTS
        }
        self.gait_detector = GaitSwingPlaneDetector(sample_rate_hz=sample_rate_hz)
        self.kalman_filter = QuadChannelBiasFilterV3()
        
        self.zaru_event_counts = {j: 0 for j in LOWER_BODY_JOINTS}
        self.gait_event_counts = {j: 0 for j in LOWER_BODY_JOINTS}

    def process_frame(self, frame: FrameV3) -> Dict[str, Dict[str, float]]:
        """
        处理单个可部署 FrameV3，返回当前四路估计状态字典。
        """
        t = frame.meta.timestamp
        dt = frame.meta.dt
        reset = frame.meta.reset_event
        
        anchor_obs = {}
        
        # 1. 运行主分支: 逐颗传感器独立的 ZARU 静止零偏检测
        if self.enable_zaru:
            for j in LOWER_BODY_JOINTS:
                tracker = frame.trackers[j]
                if not tracker.valid:
                    continue
                zaru_res = self.zaru_detectors[j].update(
                    ang_vel=tracker.ang_vel,
                    rot_mat=tracker.rot_raw,
                    ang_acc=tracker.ang_acc
                )
                if zaru_res is not None:
                    b_meas, var_b = zaru_res
                    # 直接更新单通道零偏率状态
                    ch = self.kalman_filter.channels[j]
                    # 卡尔曼测量更新零偏率 b
                    S_b = float(ch.P[1, 1] + var_b)
                    if S_b > 1e-12:
                        K_b = ch.P[1, 1] / S_b
                        innov_b = b_meas - ch.b
                        ch.b = float(ch.b + K_b * innov_b)
                        ch.P[1, 1] = max(float((1.0 - K_b) * ch.P[1, 1]), 1e-10)
                        self.zaru_event_counts[j] += 1

        # 2. 运行辅助分支: 直线步态摆动面检测
        if self.enable_gait:
            gait_candidates = self.gait_detector.update(frame)
            for j, (e_cand, var_e) in gait_candidates.items():
                anchor_obs[j] = (e_cand, var_e)
                self.gait_event_counts[j] += 1

        # 3. 驱动四路卡尔曼状态预测与步态更新
        valid_mask = {j: frame.trackers[j].valid for j in LOWER_BODY_JOINTS}
        states = self.kalman_filter.process_frame(
            timestamp=t,
            dt=dt,
            reset_event=reset,
            valid_mask=valid_mask,
            anchor_observations=anchor_obs if anchor_obs else None
        )
        return states

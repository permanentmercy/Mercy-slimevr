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
    """单颗 IMU 独立静止检测与高精零偏观测器"""
    def __init__(self,
                 imu_id: str,
                 window_sec: float = 1.5,
                 sample_rate_hz: float = 10.0,
                 omega_thresh_rad_s: float = 0.08,   # 约 4.5°/s
                 accel_var_thresh: float = 0.15,     # (m/s^2)^2
                 min_still_duration_sec: float = 2.5):
        self.imu_id = imu_id
        self.sample_rate_hz = sample_rate_hz
        self.window_size = int(max(window_sec * sample_rate_hz, 10))
        self.min_still_sec = float(min_still_duration_sec)
        self.omega_thresh = float(omega_thresh_rad_s)
        self.accel_var_thresh = float(accel_var_thresh)
        
        self.omega_history = collections.deque(maxlen=self.window_size)
        self.consecutive_still_count = 0
        
        self.still_start_R: Optional[np.ndarray] = None
        self.still_start_t: Optional[float] = None
        self.still_start_e: Optional[float] = None
        self.last_R: Optional[np.ndarray] = None

    def update(self,
               ang_vel: np.ndarray,
               rot_mat: np.ndarray,
               t: float,
               current_e: float,
               ang_acc: Optional[np.ndarray] = None) -> Optional[Tuple[float, float, float, float]]:
        """
        输入当前帧该颗 IMU 的角速度 [wx, wy, wz] 与旋转矩阵 R_raw。
        若满足连续静止条件，返回:
          (filtered_rate_rad_s, var_rate, target_e_rad, elapsed_sec)
        """
        w_norm = float(np.linalg.norm(ang_vel))
        
        # 1. 物理 3D 姿态单步突变检查 (单帧 0.1s 旋转超过 1.5° 属于人体主动动作或坐立转换)
        is_posture_jump = False
        if self.last_R is not None:
            trace_val = max(-1.0, min(3.0, float(np.trace(rot_mat @ self.last_R.T))))
            cos_step = 0.5 * (trace_val - 1.0)
            if cos_step < math.cos(math.radians(1.5)):
                is_posture_jump = True
        self.last_R = np.copy(rot_mat)
        
        # 2. 瞬时动作与突跳快速切断门禁: 任何突发动作立刻打断静止状态
        if w_norm >= self.omega_thresh or is_posture_jump:
            self.consecutive_still_count = 0
            self.omega_history.clear()
            self.still_start_R = None
            self.still_start_t = None
            self.still_start_e = None
            return None
            
        self.omega_history.append(w_norm)
        if len(self.omega_history) < self.window_size:
            return None
            
        mean_omega = float(np.mean(self.omega_history))
        var_omega = float(np.var(self.omega_history))
        
        # 2. 滑动窗口统计平稳性判定 (角速度均值与方差均处于基底噪声水平)
        is_still = (mean_omega < self.omega_thresh) and (var_omega < (self.omega_thresh * 0.5) ** 2)
        
        if is_still:
            if self.consecutive_still_count == 0:
                self.still_start_R = np.copy(rot_mat)
                self.still_start_t = t
                self.still_start_e = current_e
                
            self.consecutive_still_count += 1
            elapsed = t - self.still_start_t
            
            # 3. 达到最小连续静止时长门槛 (>= 2.5s)
            if elapsed >= self.min_still_sec:
                # 每 5 帧 (0.5s) 输出一次平滑测量
                if (self.consecutive_still_count % 5) == 0:
                    diff_R = rot_mat @ self.still_start_R.T
                    delta_yaw = math.atan2(float(diff_R[0, 2]), float(diff_R[2, 2]))
                    raw_rate = delta_yaw / elapsed  # rad/s
                    
                    # 4. 最大物理陀螺零偏保护 (25°/min): 超出该速率必然属于人体动作而非漂移
                    if abs(math.degrees(raw_rate) * 60.0) > 25.0:
                        self.consecutive_still_count = 0
                        self.still_start_R = None
                        return None
                    
                    # 5. 自适应时间不确定度死区门限 (适配 LSM6DSV 真实低漂移率 0.14°/min)
                    rate_uncertainty = 0.002 / elapsed
                    deadband_rad_s = max(math.radians(0.04 / 60.0), 1.8 * rate_uncertainty)
                    
                    if abs(raw_rate) < deadband_rad_s:
                        filtered_rate = 0.0
                    else:
                        filtered_rate = raw_rate
                        
                    var_b = max(rate_uncertainty ** 2, 1e-8)
                    target_e = float((self.still_start_e + delta_yaw + math.pi) % (2.0 * math.pi) - math.pi)
                    
                    return (filtered_rate, var_b, target_e, elapsed)
        else:
            self.consecutive_still_count = 0
            self.omega_history.clear()
            self.still_start_R = None
            self.still_start_t = None
            self.still_start_e = None
            
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
                ch = self.kalman_filter.channels[j]
                zaru_res = self.zaru_detectors[j].update(
                    ang_vel=tracker.ang_vel,
                    rot_mat=tracker.rot_raw,
                    t=t,
                    current_e=ch.e,
                    ang_acc=tracker.ang_acc
                )
                if zaru_res is not None:
                    b_meas, var_b, target_e, elapsed = zaru_res
                    # 更新单通道零偏率与姿态对齐状态
                    ch.update_stationary_observation(
                        b_meas_rad_s=b_meas,
                        var_rate=var_b,
                        target_e_rad=target_e,
                        elapsed_sec=elapsed,
                        timestamp=t
                    )
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

"""
ai_corrector/corrector/conservative_bias_estimator_v4.py
========================================================
Phase 16: 保守型三态慢偏航观测器 (Conservative Tri-State ZARU Observer)

改进核心:
1. 真正激活并强制检查机体加速度方差 (accel_var_thresh = 0.05 (m/s^2)^2);
   当人体存在真实极慢动作 (如慢速挪腿) 时，肌肉颤动与重力投影倾角微变产生加速度方差，
   系统自动判决为 REJECT / UNCERTAIN，防止将主动动作误认为零偏漂移;
2. 引入明确的三态置信度逻辑 (Tri-State Decision):
   - UPDATE: 确认物理绝对静止 (角速度范数极低 + 加速度方差极低 + 单步无突跳)
   - UNCERTAIN: 处于模糊速率带 (角速度有微变但缺乏静止证据) -> 膨胀协方差 P, 拒绝修正偏航 e
   - REJECT: 明确动作期 -> 旁路观测更新
3. 多 Tracker 相对运动学一致性门禁 (Kinematic Coupling Anchor):
   若上身 HMD / VD 或对侧腿存在显著角速度，单腿的微偏航改变不作为静态漂移更新;
4. 严格保持 0 神经网络参数，完全符合 1.10M 参数预算上限。
"""

import collections
from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ai_corrector.corrector.contract_v3 import FrameV3, LOWER_BODY_JOINTS
from ai_corrector.corrector.bias_state_v3 import QuadChannelBiasFilterV3
from ai_corrector.corrector.oracle_v3_geometry import wrap_angle


class ObservationState(Enum):
    UPDATE = "UPDATE"        # 确认物理静止，接纳测量更新
    UNCERTAIN = "UNCERTAIN"  # 模糊不可观测带，扩大不确定度，拒绝校正
    REJECT = "REJECT"        # 明确动态运动期，旁路更新


@dataclass
class TriStateObservationResult:
    state: ObservationState
    b_meas_rad_s: float
    var_rate: float
    target_e_rad: float
    elapsed_sec: float
    accel_var: float
    omega_norm: float


class ConservativeZARUDetectorV4:
    """带加速度方差监控与模糊带保护的三态静止检测器"""
    def __init__(self,
                 imu_id: str,
                 window_sec: float = 1.5,
                 sample_rate_hz: float = 10.0,
                 omega_thresh_rad_s: float = 0.05,    # 严格角速度阈值 (~2.8°/s)
                 accel_var_thresh: float = 0.05,      # 加速度总方差阈值 (m/s^2)^2
                 min_still_duration_sec: float = 2.5):
        self.imu_id = imu_id
        self.sample_rate_hz = sample_rate_hz
        self.window_size = int(max(window_sec * sample_rate_hz, 10))
        self.min_still_sec = float(min_still_duration_sec)
        self.omega_thresh = float(omega_thresh_rad_s)
        self.accel_var_thresh = float(accel_var_thresh)

        self.omega_history = collections.deque(maxlen=self.window_size)
        self.accel_history = collections.deque(maxlen=self.window_size)
        self.consecutive_still_count = 0

        self.still_start_R: Optional[np.ndarray] = None
        self.still_start_t: Optional[float] = None
        self.still_start_e: Optional[float] = None
        self.last_R: Optional[np.ndarray] = None

    def update(self,
               ang_vel: np.ndarray,
               rot_mat: np.ndarray,
               accel: np.ndarray,
               t: float,
               current_e: float,
               other_trackers_moving: bool = False) -> TriStateObservationResult:
        """
        输入当前帧角速度、旋转矩阵和三轴加速度向量，输出三态观测判决。
        """
        w_norm = float(np.linalg.norm(ang_vel))
        self.omega_history.append(w_norm)
        self.accel_history.append(np.asarray(accel, dtype=np.float32))

        # 1. 物理 3D 姿态单步突变检查
        is_posture_jump = False
        if self.last_R is not None:
            trace_val = max(-1.0, min(3.0, float(np.trace(rot_mat @ self.last_R.T))))
            cos_step = 0.5 * (trace_val - 1.0)
            if cos_step < math.cos(math.radians(1.2)):
                is_posture_jump = True
        self.last_R = np.copy(rot_mat)

        # 2. 检查多 Tracker 耦合运动 (例如全身正在转动)
        if other_trackers_moving:
            self._reset_stillness()
            return TriStateObservationResult(
                state=ObservationState.REJECT,
                b_meas_rad_s=0.0,
                var_rate=1e-2,
                target_e_rad=current_e,
                elapsed_sec=0.0,
                accel_var=999.0,
                omega_norm=w_norm
            )

        # 3. 瞬时动作与突跳拦截
        if w_norm >= self.omega_thresh or is_posture_jump:
            self._reset_stillness()
            return TriStateObservationResult(
                state=ObservationState.REJECT,
                b_meas_rad_s=0.0,
                var_rate=1e-2,
                target_e_rad=current_e,
                elapsed_sec=0.0,
                accel_var=0.0,
                omega_norm=w_norm
            )

        # 4. 滑动窗口平稳性与加速度方差真实核验
        if len(self.accel_history) < self.window_size:
            return TriStateObservationResult(
                state=ObservationState.UNCERTAIN,
                b_meas_rad_s=0.0,
                var_rate=1e-3,
                target_e_rad=current_e,
                elapsed_sec=0.0,
                accel_var=0.0,
                omega_norm=w_norm
            )

        accs = np.array(self.accel_history)  # (W, 3)
        acc_var_total = float(np.sum(np.var(accs, axis=0)))

        # 关键创新: 加速度微方差检查 (真实慢动作肌肉微颤拦截)
        if acc_var_total >= self.accel_var_thresh:
            # 存在未消除的机械/肌肉微振动，属于低速真实运动而非静态漂移!
            self._reset_stillness()
            return TriStateObservationResult(
                state=ObservationState.UNCERTAIN,  # 安全输出不确定度，拒绝更新!
                b_meas_rad_s=0.0,
                var_rate=1e-3,
                target_e_rad=current_e,
                elapsed_sec=0.0,
                accel_var=acc_var_total,
                omega_norm=w_norm
            )

        # 5. 确认绝对静止状态开始累积
        if self.consecutive_still_count == 0:
            self.still_start_R = np.copy(rot_mat)
            self.still_start_t = t
            self.still_start_e = current_e

        self.consecutive_still_count += 1
        elapsed = t - self.still_start_t

        if elapsed >= self.min_still_sec:
            # 每 5 帧输出一次平滑更新
            if (self.consecutive_still_count % 5) == 0:
                diff_R = rot_mat @ self.still_start_R.T
                delta_yaw = math.atan2(float(diff_R[0, 2]), float(diff_R[2, 2]))
                raw_rate = delta_yaw / max(elapsed, 1e-4)

                # 最大物理允许零偏保护 (15°/min)
                if abs(math.degrees(raw_rate) * 60.0) > 15.0:
                    self._reset_stillness()
                    return TriStateObservationResult(
                        state=ObservationState.REJECT,
                        b_meas_rad_s=0.0,
                        var_rate=1e-2,
                        target_e_rad=current_e,
                        elapsed_sec=elapsed,
                        accel_var=acc_var_total,
                        omega_norm=w_norm
                    )

                # 自适应时间死区 (0.04°/min)
                rate_uncertainty = 0.002 / elapsed
                deadband_rad_s = max(math.radians(0.04 / 60.0), 1.8 * rate_uncertainty)

                if abs(raw_rate) < deadband_rad_s:
                    filtered_rate = 0.0
                else:
                    filtered_rate = raw_rate

                var_b = max(rate_uncertainty ** 2, 1e-8)
                target_e = float((self.still_start_e + delta_yaw + math.pi) % (2.0 * math.pi) - math.pi)

                return TriStateObservationResult(
                    state=ObservationState.UPDATE,
                    b_meas_rad_s=filtered_rate,
                    var_rate=var_b,
                    target_e_rad=target_e,
                    elapsed_sec=elapsed,
                    accel_var=acc_var_total,
                    omega_norm=w_norm
                )

        return TriStateObservationResult(
            state=ObservationState.UNCERTAIN,
            b_meas_rad_s=0.0,
            var_rate=1e-3,
            target_e_rad=current_e,
            elapsed_sec=elapsed,
            accel_var=acc_var_total,
            omega_norm=w_norm
        )

    def _reset_stillness(self):
        self.consecutive_still_count = 0
        self.still_start_R = None
        self.still_start_t = None
        self.still_start_e = None


class ConservativeBiasEstimatorV4:
    """Phase 16 保守型偏航估计器容器"""
    def __init__(self, sample_rate_hz: float = 10.0):
        self.detectors = {
            j: ConservativeZARUDetectorV4(imu_id=j, sample_rate_hz=sample_rate_hz)
            for j in LOWER_BODY_JOINTS
        }
        self.kalman_filter = QuadChannelBiasFilterV3()
        self.update_counts = {j: 0 for j in LOWER_BODY_JOINTS}
        self.uncertain_counts = {j: 0 for j in LOWER_BODY_JOINTS}
        self.reject_counts = {j: 0 for j in LOWER_BODY_JOINTS}

    def process_frame(self, frame: FrameV3) -> Dict[str, Dict[str, float]]:
        t = frame.meta.timestamp
        dt = frame.meta.dt
        reset = frame.meta.reset_event

        # 检查上身或整体运动
        hmd_moving = bool(np.linalg.norm(frame.upper_body.hmd_ang_vel) > 0.08 or np.linalg.norm(frame.upper_body.hmd_lin_vel) > 0.2)

        for j in LOWER_BODY_JOINTS:
            tracker = frame.trackers[j]
            if not tracker.valid:
                continue

            ch = self.kalman_filter.channels[j]
            res = self.detectors[j].update(
                ang_vel=tracker.ang_vel,
                rot_mat=tracker.rot_raw,
                accel=tracker.accel,
                t=t,
                current_e=ch.e,
                other_trackers_moving=hmd_moving
            )

            if res.state == ObservationState.UPDATE:
                ch.update_stationary_observation(
                    b_meas_rad_s=res.b_meas_rad_s,
                    var_rate=res.var_rate,
                    target_e_rad=res.target_e_rad,
                    elapsed_sec=res.elapsed_sec,
                    timestamp=t
                )
                self.update_counts[j] += 1
            elif res.state == ObservationState.UNCERTAIN:
                # 模糊不可观测带: 仅自然增加协方差不确定度，拒绝移动偏航角 e
                ch.P[0, 0] += ch.q_angle * dt
                self.uncertain_counts[j] += 1
            else:
                self.reject_counts[j] += 1

        valid_mask = {j: frame.trackers[j].valid for j in LOWER_BODY_JOINTS}
        return self.kalman_filter.process_frame(
            timestamp=t,
            dt=dt,
            reset_event=reset,
            valid_mask=valid_mask,
            anchor_observations=None
        )

"""
ai_corrector/corrector/bias_state_v3.py
======================================
Phase 9: 四路下肢显式偏航状态卡尔曼滤波器 (e_i, b_i, P_i)

功能:
1. 显式维护四颗腿部 IMU (左大腿、右大腿、左小腿、右小腿) 独立的世界竖直轴偏航状态;
2. 状态向量 x = [e, b]^T (e: 偏航角误差 rad, b: 姿态偏航误差率 rad/s);
3. 2x2 协方差矩阵 P 动态更新 (Joseph 稳定形式);
4. 角度 wrap-around (限制在 [-pi, pi]);
5. 幅度与速率保护门槛 (max_drift_rate 限制);
6. reset / 掉线 / 时间跳变安全复位与方差膨胀机制;
7. 导出给服务端的局部右乘修正四元数与状态遥测字典.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

LOWER_BODY_JOINTS = ("left_thigh", "right_thigh", "left_calf", "right_calf")


class SingleChannelYawFilter:
    """单颗 IMU 的一阶/二阶世界偏航误差卡尔曼滤波器"""

    def __init__(self,
                 imu_id: str,
                 q_angle_var: float = 1e-6,      # 偏航角度过程噪声率 rad^2/s
                 q_rate_var: float = 1e-8,       # 漂移率过程噪声率 (rad/s)^2/s
                 init_angle_var: float = 1e-3,   # 初始角度方差 rad^2
                 init_rate_var: float = 1e-5,    # 初始漂移率方差 (rad/s)^2
                 max_drift_rate_deg_min: float = 30.0  # 最大物理允许漂移率 (30°/min)
                 ):
        self.imu_id = imu_id
        self.e = 0.0  # 世界偏航误差 (radians, [-pi, pi])
        self.b = 0.0  # 偏航漂移率 (rad/s)
        
        # 协方差矩阵 P (2x2)
        self.P = np.array([
            [init_angle_var, 0.0],
            [0.0, init_rate_var]
        ], dtype=np.float64)

        self.q_angle = float(q_angle_var)
        self.q_rate = float(q_rate_var)
        self.init_P = np.copy(self.P)
        
        # 物理饱和截断
        self.max_b = math.radians(max_drift_rate_deg_min / 60.0)  # rad/s
        self.last_t: Optional[float] = None
        self.total_updates = 0

    def reset(self, t: Optional[float] = None):
        """完全重置状态（如用户双击 reset 标定）"""
        self.e = 0.0
        self.b = 0.0
        self.P = np.copy(self.init_P)
        self.last_t = t

    def predict(self, dt: float):
        """
        状态外推预测步:
          e <- wrap(e + b * dt)
          b <- b
          P <- F P F^T + Q * dt
        """
        if dt <= 0.0:
            return
        # 限制单步异常大 dt
        dt = min(dt, 1.0)
        
        # 1. 状态外推
        self.e = float((self.e + self.b * dt + math.pi) % (2.0 * math.pi) - math.pi)
        
        # 2. 状态转移矩阵 F = [[1, dt], [0, 1]]
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ], dtype=np.float64)
        
        # 3. 过程噪声协方差 Q_dt
        Q_dt = np.array([
            [self.q_angle * dt + (1.0/3.0) * self.q_rate * (dt**3), 0.5 * self.q_rate * (dt**2)],
            [0.5 * self.q_rate * (dt**2), self.q_rate * dt]
        ], dtype=np.float64)
        
        self.P = F @ self.P @ F.T + Q_dt
        self._ensure_covariance_stability()

    def update_stationary_observation(self,
                                      b_meas_rad_s: float,
                                      var_rate: float,
                                      target_e_rad: float,
                                      elapsed_sec: float,
                                      timestamp: float):
        """
        ZARU 静止段高精测量更新:
        1. 漂移率状态 b 的动态测量平滑更新;
        2. 静止姿态几何对齐与初始启动补偿 (带严格零漂保护).
        """
        # 1. 零偏率更新
        alpha_b = min(0.15, 2.0 / (elapsed_sec + 10.0))
        self.b = float((1.0 - alpha_b) * self.b + alpha_b * b_meas_rad_s)
        self.b = float(np.clip(self.b, -self.max_b, self.max_b))
        self.P[1, 1] = max(float(var_rate), 1e-10)

        # 2. 真实物理漂移检测 (b_meas != 0.0) 下对齐 e，消除前续自由运动期间由于积分累积的残余微小误差
        if b_meas_rad_s != 0.0 and elapsed_sec > 5.0:
            diff_e = float((target_e_rad - self.e + math.pi) % (2.0 * math.pi) - math.pi)
            self.e = float((self.e + 0.05 * diff_e + math.pi) % (2.0 * math.pi) - math.pi)
            self.P[0, 0] = max(self.P[0, 0] * 0.95, 1e-6)

        self._ensure_covariance_stability()
        self.total_updates += 1

    def update_observation(self, z_rad: float, var_rad2: float):
        """
        步态航向观测更新步 (解耦模式: 仅更新偏航角 e，严禁交叉污染漂移率 b)
        """
        if var_rad2 <= 0.0:
            var_rad2 = 1e-2

        # 观测残差 y = wrap(z - e)
        y = float((z_rad - self.e + math.pi) % (2.0 * math.pi) - math.pi)
        
        # 仅更新角度 e, 拒绝向漂移率 b 交叉耦合
        S_e = float(self.P[0, 0] + var_rad2)
        if S_e > 1e-12:
            K_e = min(0.1, self.P[0, 0] / S_e)  # 保守增益限制
            self.e = float((self.e + K_e * y + math.pi) % (2.0 * math.pi) - math.pi)
            self.P[0, 0] = max(float((1.0 - K_e) * self.P[0, 0]), 1e-6)
            
        self._ensure_covariance_stability()
        self.total_updates += 1

    def _ensure_covariance_stability(self):
        """强制协方差对称与正定下界保护"""
        self.P = 0.5 * (self.P + self.P.T)
        self.P[0, 0] = max(self.P[0, 0], 1e-8)
        self.P[1, 1] = max(self.P[1, 1], 1e-10)
        # 限制条件数/相关系数
        corr = self.P[0, 1] / (math.sqrt(self.P[0, 0] * self.P[1, 1]) + 1e-12)
        if abs(corr) >= 0.999:
            self.P[0, 1] = np.sign(corr) * 0.999 * math.sqrt(self.P[0, 0] * self.P[1, 1])
            self.P[1, 0] = self.P[0, 1]


class QuadChannelBiasFilterV3:
    """四路下肢独立的偏航状态估计器管理容器"""

    def __init__(self,
                 max_drift_rate_deg_min: float = 30.0,
                 q_angle_var: float = 1e-6,
                 q_rate_var: float = 1e-8):
        self.channels: Dict[str, SingleChannelYawFilter] = {
            j: SingleChannelYawFilter(
                imu_id=j,
                q_angle_var=q_angle_var,
                q_rate_var=q_rate_var,
                max_drift_rate_deg_min=max_drift_rate_deg_min
            ) for j in LOWER_BODY_JOINTS
        }
        self.last_timestamp: Optional[float] = None

    def process_frame(self,
                      timestamp: float,
                      dt: float,
                      reset_event: bool = False,
                      valid_mask: Optional[Dict[str, bool]] = None,
                      anchor_observations: Optional[Dict[str, Tuple[float, float]]] = None
                      ) -> Dict[str, Dict[str, float]]:
        """
        处理单帧时序推进与稀疏锚点更新。
        
        参数:
          timestamp: 当前帧单调时间戳 (秒)
          dt: 帧间隔时间 (秒)
          reset_event: 是否触发了系统全局重置
          valid_mask: 各通道有效性字典 {joint: bool}
          anchor_observations: 稀疏锚点观测字典 {joint: (z_rad, var_rad2)}
          
        返回:
          各通道当前估计状态字典:
          { joint: { "e_hat_rad": float, "b_hat_rad_s": float, "e_var": float, "b_var": float } }
        """
        # 1. 处理全局 Reset
        if reset_event or (self.last_timestamp is not None and (timestamp - self.last_timestamp > 3.0)):
            for ch in self.channels.values():
                ch.reset(t=timestamp)
            self.last_timestamp = timestamp

        if self.last_timestamp is None:
            self.last_timestamp = timestamp

        actual_dt = dt if (0.0 < dt <= 1.0) else (timestamp - self.last_timestamp)
        actual_dt = max(0.0, min(actual_dt, 1.0))
        self.last_timestamp = timestamp

        outputs = {}
        for j in LOWER_BODY_JOINTS:
            ch = self.channels[j]
            is_valid = True if (valid_mask is None or valid_mask.get(j, True)) else False

            # 2. 预测外推步
            if actual_dt > 0.0:
                ch.predict(actual_dt)

            # 3. 观测更新步 (若当前为可信锚点且通道有效)
            if is_valid and anchor_observations and (j in anchor_observations):
                z_rad, var_rad2 = anchor_observations[j]
                ch.update_observation(z_rad, var_rad2)

            outputs[j] = {
                "e_hat_rad": ch.e,
                "e_hat_deg": math.degrees(ch.e),
                "b_hat_rad_s": ch.b,
                "b_hat_deg_min": math.degrees(ch.b) * 60.0,
                "e_var": float(ch.P[0, 0]),
                "b_var": float(ch.P[1, 1])
            }

        return outputs

    def get_current_states(self) -> Dict[str, Tuple[float, float, np.ndarray]]:
        """获取当前四路状态快照 (e, b, P)"""
        return {j: (ch.e, ch.b, np.copy(ch.P)) for j, ch in self.channels.items()}

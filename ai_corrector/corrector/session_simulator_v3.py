"""
ai_corrector/corrector/session_simulator_v3.py
=============================================
Phase 9 M1r: 30分钟连续运动会话仿真器 (完整运动流与严格可部署特征接口)

特性:
1. 完整的人体宏观行为状态机 (直立静止、直线行走、端坐/侧坐休息、原地转身);
2. 真实传感器物理模型:
   - 纯净动作角速度 omega_motion;
   - 陀螺零偏向量 b_gyro + 真实偏航漂移累计 e_true(t);
   - 传感器裸读数 omega_raw = omega_motion + b_gyro + noise;
   - 原始姿态 R_raw(t) = R_y(e_true(t)) @ R_clean(t);
3. 严格数据边界:
   - 输出符合 contract_v2 规范的 124 维特征流 (前 68 维视觉, 后 56 维 IMU);
   - 在线模型仅接收可部署特征 (FrameV3 / 124 维数组), 严禁泄漏 true_drift_angles 或真值姿态;
4. 终点独立真值仅供离线评估使用 (t0 与 t1 处世界偏航误差 wrap 计算).
"""

from dataclasses import dataclass
import math
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai_corrector.corrector.contract_v2 import pack_stream_17d, pack_stream_14d, TOTAL_INPUT_DIM_V2
from ai_corrector.corrector.contract_v3 import FrameV3, V2ToV3Adapter, LOWER_BODY_JOINTS
from ai_corrector.corrector.oracle_v3_geometry import (
    rot_y_matrix,
    world_left_correction,
    world_to_local_dq,
    wrong_local_right_correction,
    compute_endpoint_session_metrics,
    wrap_angle,
    geodesic_angle_rad
)


@dataclass
class ContinuousSessionStream:
    """30 分钟连续会话完整数据包"""
    session_id: str
    duration_sec: float
    dt: float
    num_frames: int
    timestamps: np.ndarray                  # (N,)
    features_124: np.ndarray                # (N, 124) 纯在线可部署特征矩阵
    # 离线真值字典 (仅用于验收，严禁在线访问)
    ground_truth_eval: dict


class ContinuousSessionSynthesizer:
    """30 分钟多工况物理连续会话合成器"""
    def __init__(self,
                 duration_sec: float = 1800.0,
                 sample_rate_hz: float = 10.0,
                 drift_tier_deg_min: float = 3.0,
                 seed: int = 123,
                 scenario: str = "default",
                 enable_random_walk: bool = True,
                 enable_dynamic_tilt: bool = True):
        self.duration_sec = float(duration_sec)
        self.sample_rate_hz = float(sample_rate_hz)
        self.dt = 1.0 / self.sample_rate_hz
        self.num_frames = int(self.duration_sec * self.sample_rate_hz)
        self.drift_tier = float(drift_tier_deg_min)
        self.rng = np.random.RandomState(seed)
        self.scenario = scenario
        self.enable_random_walk = enable_random_walk
        self.enable_dynamic_tilt = enable_dynamic_tilt

    def generate(self, session_id: str) -> ContinuousSessionStream:
        t = np.linspace(0.0, self.duration_sec, self.num_frames, endpoint=False)
        
        # 1. 为四路 IMU 分配真实线性漂移率 (以 drift_tier 为基准波动)
        true_drift_rates_deg_min = {}
        for j in LOWER_BODY_JOINTS:
            if abs(self.drift_tier) < 1e-9:
                true_drift_rates_deg_min[j] = 0.0
            else:
                factor = self.rng.uniform(0.6, 1.4) * self.rng.choice([-1.0, 1.0])
                true_drift_rates_deg_min[j] = self.drift_tier * factor
            
        true_rates_rad_s = {
            j: math.radians(true_drift_rates_deg_min[j] / 60.0) for j in LOWER_BODY_JOINTS
        }
        
        # 2. 生成真实连续漂移曲线 e_true(t) (线性漂移 + 极慢随机游走)
        true_drift_curves = {}
        for j in LOWER_BODY_JOINTS:
            rate = true_rates_rad_s[j]
            lin = rate * t
            if self.enable_random_walk:
                rw = np.cumsum(self.rng.normal(0.0, 0.0002, size=self.num_frames))
            else:
                rw = np.zeros(self.num_frames, dtype=np.float64)
            true_drift_curves[j] = lin + rw

        # 3. 构造宏观行为状态时间线 (按分段循环)
        # 0: 直立静止 (Standing rest)
        # 1: 直线行走 (Walking forward)
        # 2: 端坐/侧坐休息 (Sitting / quiet)
        # 3: 原地转身/漫步 (Turning / wandering)
        behavior_states = np.zeros(self.num_frames, dtype=np.int32)
        if self.scenario == "all_sitting":
            behavior_states[:] = 2
        elif self.scenario == "no_still":
            for i in range(self.num_frames):
                behavior_states[i] = 1 if (i // int(30.0 * self.sample_rate_hz)) % 2 == 0 else 3
        elif self.scenario == "complex_gait":
            cycle_len = int(240.0 * self.sample_rate_hz)
            for i in range(self.num_frames):
                c_pos = i % cycle_len
                t_in = c_pos * self.dt
                if t_in < 40.0:
                    behavior_states[i] = 1  # 前进
                elif t_in < 80.0:
                    behavior_states[i] = 3  # 转身
                elif t_in < 140.0:
                    behavior_states[i] = 0  # 站立
                else:
                    behavior_states[i] = 2  # 静坐
        else:
            cycle_len = int(420.0 * self.sample_rate_hz)  # 7 分钟一个周期
            for i in range(self.num_frames):
                c_pos = i % cycle_len
                t_in_cycle = c_pos * self.dt
                if t_in_cycle < 60.0:
                    behavior_states[i] = 0  # 站立静止
                elif t_in_cycle < 180.0:
                    behavior_states[i] = 1  # 直线行走
                elif t_in_cycle < 360.0:
                    behavior_states[i] = 2  # 端坐/侧坐静止
                else:
                    behavior_states[i] = 3  # 转身漫步

        # 4. 生成每帧的物理信号
        features_124 = np.zeros((self.num_frames, TOTAL_INPUT_DIM_V2), dtype=np.float32)
        
        # HMD 运动轨迹模拟
        hmd_pos = np.array([0.0, 1.70, 0.0], dtype=np.float32)
        hmd_heading = 0.0  # rad
        
        # 离线真值快照
        clean_rot_t0 = {}
        clean_rot_t1 = {}
        raw_rot_t0 = {}
        raw_rot_t1 = {}

        walk_phase = 0.0
        step_freq = 1.2  # 步频 1.2 Hz

        for i in range(self.num_frames):
            state = behavior_states[i]
            
            # (A) 人体动作角速度与局部姿态生成
            leg_omega_motion = {}
            leg_rot_clean = {}
            
            if state == 0:
                # 站立静止: 腿部完全静止, 干净姿态为双腿垂直
                for j in LOWER_BODY_JOINTS:
                    leg_omega_motion[j] = np.zeros(3, dtype=np.float32)
                    leg_rot_clean[j] = np.eye(3, dtype=np.float32)
                hmd_lin_vel = np.zeros(3, dtype=np.float32)
                hmd_ang_vel = np.zeros(3, dtype=np.float32)
            elif state == 1:
                # 直线行走: 沿世界 Z 轴前进 (速度 1.0 m/s), 双腿周期性前后矢状摆动
                walk_phase += 2.0 * math.pi * step_freq * self.dt
                swing_l = 1.2 * math.sin(walk_phase)
                swing_r = 1.2 * math.sin(walk_phase + math.pi)
                
                leg_omega_motion["left_thigh"] = np.array([swing_l, 0.0, 0.0], dtype=np.float32)
                leg_omega_motion["right_thigh"] = np.array([swing_r, 0.0, 0.0], dtype=np.float32)
                leg_omega_motion["left_calf"] = np.array([swing_l * 1.3, 0.0, 0.0], dtype=np.float32)
                leg_omega_motion["right_calf"] = np.array([swing_r * 1.3, 0.0, 0.0], dtype=np.float32)
                
                # 行走姿态 (俯仰角摆动)
                pitch_l = 0.4 * math.cos(walk_phase)
                pitch_r = 0.4 * math.cos(walk_phase + math.pi)
                R_pitch_l = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(pitch_l), -math.sin(pitch_l)],
                    [0.0, math.sin(pitch_l), math.cos(pitch_l)]
                ], dtype=np.float32)
                R_pitch_r = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(pitch_r), -math.sin(pitch_r)],
                    [0.0, math.sin(pitch_r), math.cos(pitch_r)]
                ], dtype=np.float32)
                
                leg_rot_clean["left_thigh"] = R_pitch_l
                leg_rot_clean["right_thigh"] = R_pitch_r
                leg_rot_clean["left_calf"] = R_pitch_l
                leg_rot_clean["right_calf"] = R_pitch_r
                
                hmd_lin_vel = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                hmd_pos[2] += 1.0 * self.dt
                hmd_ang_vel = np.array([0.05 * math.sin(walk_phase * 2), 0.0, 0.0], dtype=np.float32)
            elif state == 2:
                # 端坐/侧坐休息: 膝盖弯曲 70°, 双腿静止不动
                R_sit = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(1.2), -math.sin(1.2)],
                    [0.0, math.sin(1.2), math.cos(1.2)]
                ], dtype=np.float32)
                for j in LOWER_BODY_JOINTS:
                    leg_omega_motion[j] = np.zeros(3, dtype=np.float32)
                    leg_rot_clean[j] = R_sit
                hmd_lin_vel = np.zeros(3, dtype=np.float32)
                hmd_ang_vel = np.zeros(3, dtype=np.float32)
            else:
                # 转身漫步
                hmd_heading += 0.2 * self.dt
                for j in LOWER_BODY_JOINTS:
                    leg_omega_motion[j] = self.rng.normal(0.0, 0.1, 3).astype(np.float32)
                    leg_rot_clean[j] = rot_y_matrix(hmd_heading).astype(np.float32)
                hmd_lin_vel = np.array([0.3 * math.sin(hmd_heading), 0.0, 0.3 * math.cos(hmd_heading)], dtype=np.float32)
                hmd_pos += hmd_lin_vel * self.dt
                hmd_ang_vel = np.array([0.0, 0.2, 0.0], dtype=np.float32)

            # (B) 组装视觉通道 (前 68 维)
            hmd_rot = rot_y_matrix(hmd_heading).astype(np.float32)
            s0 = pack_stream_17d(hmd_pos, hmd_rot, hmd_lin_vel, hmd_ang_vel, valid=1.0, age=0.0)
            vd_chest_pos = hmd_pos - np.array([0.0, 0.3, 0.0], dtype=np.float32)
            if self.scenario == "vd_missing":
                s1 = pack_stream_17d(vd_chest_pos, hmd_rot, hmd_lin_vel, hmd_ang_vel, valid=0.0, age=999.0)
            else:
                s1 = pack_stream_17d(vd_chest_pos, hmd_rot, hmd_lin_vel, hmd_ang_vel, valid=1.0, age=0.0)
            s2 = np.zeros(17, dtype=np.float32)
            s3 = np.zeros(17, dtype=np.float32)

            # (C) 组装四路 IMU 通道 (后 56 维)
            imu_streams = []
            for j in LOWER_BODY_JOINTS:
                e_t = float(true_drift_curves[j][i])
                
                # 动态 AHRS 倾斜失真 (只在肢体运动时产生，静态为 0)
                norm_omega = float(np.linalg.norm(leg_omega_motion[j]))
                if self.enable_dynamic_tilt and norm_omega > 0.05:
                    tilt_axis = np.cross(leg_omega_motion[j], np.array([0.0, 1.0, 0.0], dtype=np.float32))
                    n_t = float(np.linalg.norm(tilt_axis))
                    if n_t > 1e-6:
                        u_t = tilt_axis / n_t
                        ang_t = float(np.clip(0.040 * n_t, 0.0, np.deg2rad(6.0)))
                        c_t, s_t = math.cos(ang_t), math.sin(ang_t)
                        ux, uy, uz = u_t
                        R_dyn_tilt = np.array([
                            [c_t + ux*ux*(1-c_t), ux*uy*(1-c_t) - uz*s_t, ux*uz*(1-c_t) + uy*s_t],
                            [uy*ux*(1-c_t) + uz*s_t, c_t + uy*uy*(1-c_t), uy*uz*(1-c_t) - ux*s_t],
                            [uz*ux*(1-c_t) - uy*s_t, uz*uy*(1-c_t) + ux*s_t, c_t + uz*uz*(1-c_t)]
                        ], dtype=np.float32)
                    else:
                        R_dyn_tilt = np.eye(3, dtype=np.float32)
                else:
                    R_dyn_tilt = np.eye(3, dtype=np.float32)

                R_raw = rot_y_matrix(e_t).astype(np.float32) @ R_dyn_tilt @ leg_rot_clean[j]
                
                # 传感器测量角速度 = 动作角速度 + 机体陀螺零偏 + 测量白噪声
                # 世界竖直轴偏航漂移率在机体轴投影: b_body = R_raw^T @ [0, rate, 0]
                b_world = np.array([0.0, true_rates_rad_s[j], 0.0], dtype=np.float32)
                b_body = R_raw.T @ b_world
                noise_w = self.rng.normal(0.0, 0.005, 3).astype(np.float32)
                omega_raw = leg_omega_motion[j] + b_body + noise_w
                ang_acc = np.zeros(3, dtype=np.float32)
                
                s_imu = pack_stream_14d(R_raw, ang_acc, omega_raw, valid=1.0, age=0.0)
                imu_streams.append(s_imu)
                
                if i == 0:
                    clean_rot_t0[j] = np.copy(leg_rot_clean[j])
                    raw_rot_t0[j] = np.copy(R_raw)
                if i == self.num_frames - 1:
                    clean_rot_t1[j] = np.copy(leg_rot_clean[j])
                    raw_rot_t1[j] = np.copy(R_raw)

            features_124[i] = np.concatenate([s0, s1, s2, s3, *imu_streams], axis=-1)

        gt_eval = {
            "drift_tier_deg_min": self.drift_tier,
            "true_drift_rates_deg_min": true_drift_rates_deg_min,
            "true_drift_curves_rad": true_drift_curves,
            "clean_rot_t0": clean_rot_t0,
            "clean_rot_t1": clean_rot_t1,
            "raw_rot_t0": raw_rot_t0,
            "raw_rot_t1": raw_rot_t1,
            "e_true_t0_rad": {j: float(true_drift_curves[j][0]) for j in LOWER_BODY_JOINTS},
            "e_true_t1_rad": {j: float(true_drift_curves[j][-1]) for j in LOWER_BODY_JOINTS}
        }

        return ContinuousSessionStream(
            session_id=session_id,
            duration_sec=self.duration_sec,
            dt=self.dt,
            num_frames=self.num_frames,
            timestamps=t,
            features_124=features_124,
            ground_truth_eval=gt_eval
        )

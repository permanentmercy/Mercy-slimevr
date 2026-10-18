"""
ai_corrector/corrector/physics_session_simulator_v4.py
======================================================
Phase 16: 物理自洽动力学与连续传感器信号合成器 (Kinematic Quaternion Integrator)

相较于 Phase 15 的重大改进:
1. 废除 R_raw 与 omega_raw 的独立生成模式;
2. 姿态采用严格四元数微分方程积分生成:
   q(t + dt) = q(t) * exp(0.5 * (omega_motion + b_gyro + noise) * dt)
   保证 R_raw 与 omega_imu 在数学与动力学上 100% 耦合自洽;
3. 真实物理加速度计模型:
   a_meas(t) = R(t)^T * (a_motion(t) + [0, 9.81, 0]^T) + b_acc + noise_acc
   当肢体有哪怕微弱的真实肌肉动作时，加速度计在机体轴会产生微振方差，为 ZARU 静止判定提供第二独立物理观测源;
4. 真实慢动作 (Intentional Slow Motion) 物理切片生成:
   支持生成极其微慢 (0.1°/min ~ 1.0°/min) 的人体主动姿态转换 (如坐姿缓慢挪腿、慢速转身),
   用于严格测试慢偏航观测器能否识别真实动作与传感器漂移。
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
from ai_corrector.corrector.contract_v3 import LOWER_BODY_JOINTS
from ai_corrector.corrector.oracle_v3_geometry import rot_y_matrix, wrap_angle


def quat_exp_so3(omega_dt: np.ndarray) -> np.ndarray:
    """SO(3) 轴角向量转增量四元数 [w, x, y, z]"""
    theta = float(np.linalg.norm(omega_dt))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half = theta * 0.5
    axis = omega_dt / theta
    return np.array([math.cos(half), axis[0]*math.sin(half), axis[1]*math.sin(half), axis[2]*math.sin(half)], dtype=np.float64)


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """四元数转 3x3 旋转矩阵"""
    q_norm = q / np.linalg.norm(q)
    w, x, y, z = q_norm
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w), 2.0*(x*z + y*w)],
        [2.0*(x*y + z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w), 2.0*(y*z + x*w), 1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float64)


@dataclass
class PhysicsSessionStreamV4:
    """物理动力学自洽连续会话数据包"""
    session_id: str
    duration_sec: float
    dt: float
    num_frames: int
    timestamps: np.ndarray                  # (N,)
    features_124: np.ndarray                # (N, 124) 纯在线可部署特征矩阵
    angular_velocity_sources: Dict[str, np.ndarray]  # {"raw_gyro": (N,4,3), "fused_rotation_delta": (N,4,3)}
    ground_truth_eval: dict                 # 离线真值字典
    dts: Optional[np.ndarray] = None        # (N,) 每帧具体 dt (含真实传输抖动)


class PhysicsSessionSynthesizerV4:
    """四元数动力学积分物理会话合成器"""
    def __init__(self,
                 duration_sec: float = 600.0,
                 sample_rate_hz: float = 10.0,
                 drift_tier_deg_min: float = 0.14,
                 scenario: str = "default",
                 seed: int = 42,
                 enable_random_walk: bool = True,
                 enable_accel_noise: bool = True):
        self.duration_sec = float(duration_sec)
        self.sample_rate_hz = float(sample_rate_hz)
        self.dt = 1.0 / self.sample_rate_hz
        self.num_frames = int(self.duration_sec * self.sample_rate_hz)
        self.drift_tier = float(drift_tier_deg_min)
        self.scenario = scenario
        self.rng = np.random.RandomState(seed)
        self.enable_random_walk = enable_random_walk
        self.enable_accel_noise = enable_accel_noise

    def generate(self, session_id: str) -> PhysicsSessionStreamV4:
        if self.scenario == "non_uniform_dt_jitter":
            # 真实非均匀时间戳抖动 (80ms ~ 120ms, 均值 100ms)
            jitter_factors = self.rng.uniform(0.80, 1.20, self.num_frames)
            dts = (self.dt * jitter_factors).astype(np.float64)
            t = np.zeros(self.num_frames, dtype=np.float64)
            t[1:] = np.cumsum(dts[:-1])
        else:
            dts = np.full(self.num_frames, self.dt, dtype=np.float64)
            t = np.linspace(0.0, self.duration_sec, self.num_frames, endpoint=False)

        # 1. 分配陀螺零偏向量 b_gyro (绕世界竖直 Y 轴投影为主)
        b_gyro_world = {}
        for j in LOWER_BODY_JOINTS:
            if abs(self.drift_tier) < 1e-9 or self.scenario in ("strict_zero_drift", "intentional_slow_motion_no_drift"):
                rate_deg_min = 0.0
            elif self.scenario in ("nominal_pos_0_14deg", "nominal_drift_with_rest"):
                factor = self.rng.uniform(0.85, 1.15)
                rate_deg_min = abs(self.drift_tier) * factor
            elif self.scenario == "nominal_neg_0_14deg":
                factor = self.rng.uniform(0.85, 1.15) * (-1.0)
                rate_deg_min = abs(self.drift_tier) * factor
            else:
                factor = self.rng.uniform(0.8, 1.2) * self.rng.choice([-1.0, 1.0])
                rate_deg_min = self.drift_tier * factor
            rate_rad_s = math.radians(rate_deg_min / 60.0)
            b_gyro_world[j] = np.array([0.0, rate_rad_s, 0.0], dtype=np.float64)

        # 2. 生成每帧的人体主动运动指令 (角速度与加速度)
        # 行为状态码:
        # 0: 站立静止 (Standing Still)
        # 1: 直线行走 (Walking)
        # 2: 静坐保持 (Sitting Still)
        # 3: 转身漫步 (Turning / Wandering)
        # 4: 极慢动作/主动挪腿 (Intentional Slow Motion: ~0.3°/min 连续微转，含生理微颤 Var(a) > 0.05)
        # 5: 单腿独立动作 (Left leg moving, Right leg stationary)
        # 6: 全身慢速同向转向 (Whole body turning, HMD moving)
        # 7: 静坐抖腿 (Sitting with fidgeting, high accel variance)
        # 8: 静止+周期性外力敲击震动 (Sitting Still with periodic table bump disturbances)
        behavior_states = np.zeros(self.num_frames, dtype=np.int32)

        if self.scenario in ("pure_drift_static", "strict_zero_drift", "all_sitting"):
            behavior_states[:] = 2
        elif self.scenario == "intentional_slow_motion_no_drift":
            behavior_states[:] = 4
        elif self.scenario == "slow_motion_plus_drift":
            # 循环: 静止(2) 60s -> 慢动作挪腿(4) 60s
            cycle_half = int(60.0 * self.sample_rate_hz)
            for i in range(self.num_frames):
                behavior_states[i] = 4 if ((i // cycle_half) % 2 == 1) else 2
        elif self.scenario == "single_leg_isolated_motion":
            behavior_states[:] = 5
        elif self.scenario == "whole_body_slow_turn":
            behavior_states[:] = 6
        elif self.scenario == "sitting_leg_fidget":
            behavior_states[:] = 7
        elif self.scenario == "accel_disturbance_bump":
            behavior_states[:] = 8
        elif self.scenario == "no_still":
            for i in range(self.num_frames):
                behavior_states[i] = 1 if (i // int(30.0 * self.sample_rate_hz)) % 2 == 0 else 3
        else:
            # 标准混合工况 (循环: 静止 60s -> 行走 90s -> 静坐 90s -> 转身漫步 60s)
            for i in range(self.num_frames):
                t_in = float(t[i] % 300.0)
                if t_in < 60.0:
                    behavior_states[i] = 0  # 站立静止
                elif t_in < 150.0:
                    behavior_states[i] = 1  # 行走
                elif t_in < 240.0:
                    behavior_states[i] = 2  # 静坐
                else:
                    behavior_states[i] = 3  # 转身漫步

        # 3. 四元数积分初始状态
        q_clean = {j: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) for j in LOWER_BODY_JOINTS}
        q_raw = {j: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) for j in LOWER_BODY_JOINTS}

        # 预分配时序数组
        R_clean_arr = {j: np.zeros((self.num_frames, 3, 3), dtype=np.float32) for j in LOWER_BODY_JOINTS}
        R_raw_arr = {j: np.zeros((self.num_frames, 3, 3), dtype=np.float32) for j in LOWER_BODY_JOINTS}
        omega_raw_arr = {j: np.zeros((self.num_frames, 3), dtype=np.float32) for j in LOWER_BODY_JOINTS}
        accel_raw_arr = {j: np.zeros((self.num_frames, 3), dtype=np.float32) for j in LOWER_BODY_JOINTS}
        omega_derived_arr = {j: np.zeros((self.num_frames, 3), dtype=np.float32) for j in LOWER_BODY_JOINTS}

        walk_phase = 0.0
        hmd_heading = 0.0
        hmd_pos = np.array([0.0, 1.70, 0.0], dtype=np.float32)

        features_124 = np.zeros((self.num_frames, TOTAL_INPUT_DIM_V2), dtype=np.float32)

        for i in range(self.num_frames):
            dt_i = float(dts[i])
            state = behavior_states[i]

            # 生成肢体真实主动运动
            omega_body = {}
            acc_body = {}

            if state in (0, 2):
                # 纯静止 (0 动作角速度, 0 线加速度)
                for j in LOWER_BODY_JOINTS:
                    omega_body[j] = np.zeros(3, dtype=np.float64)
                    acc_body[j] = np.zeros(3, dtype=np.float64)
                hmd_lin_vel = np.zeros(3, dtype=np.float32)
                hmd_ang_vel = np.zeros(3, dtype=np.float32)

            elif state == 1:
                # 步态行走
                walk_phase += 2.0 * math.pi * 1.2 * dt_i
                swing_l = 1.2 * math.sin(walk_phase)
                swing_r = 1.2 * math.sin(walk_phase + math.pi)
                omega_body["left_thigh"] = np.array([swing_l, 0.0, 0.0], dtype=np.float64)
                omega_body["right_thigh"] = np.array([swing_r, 0.0, 0.0], dtype=np.float64)
                omega_body["left_calf"] = np.array([swing_l * 1.3, 0.0, 0.0], dtype=np.float64)
                omega_body["right_calf"] = np.array([swing_r * 1.3, 0.0, 0.0], dtype=np.float64)
                for j in LOWER_BODY_JOINTS:
                    acc_body[j] = np.array([0.0, 0.3 * math.sin(walk_phase * 2), 0.5 * math.cos(walk_phase)], dtype=np.float64)
                hmd_lin_vel = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                hmd_pos[2] += 1.0 * dt_i
                hmd_ang_vel = np.zeros(3, dtype=np.float32)

            elif state == 4:
                # 极慢主动转腿动作 (Intentional Slow Motion, 例如 ~0.3°/min 极微慢转动)
                # 关键物理特征: 真实肌肉运动带有微弱的生理颤动与加速度微方差 (Var > 0.05)
                slow_rate = math.radians(0.3 / 60.0)  # 0.3°/min
                for j in LOWER_BODY_JOINTS:
                    omega_body[j] = np.array([0.0, slow_rate, 0.0], dtype=np.float64)
                    acc_body[j] = self.rng.normal(0.0, 0.25, 3).astype(np.float64)
                hmd_lin_vel = np.zeros(3, dtype=np.float32)
                hmd_ang_vel = np.zeros(3, dtype=np.float32)

            elif state == 5:
                # 单腿独立动作 (左腿慢转挪动并伴随肌肉颤动，右腿严格静止)
                slow_rate = math.radians(0.3 / 60.0)
                tremor = self.rng.normal(0.0, 0.25, 3).astype(np.float64)
                omega_body["left_thigh"] = np.array([0.0, slow_rate, 0.0], dtype=np.float64)
                omega_body["left_calf"] = np.array([0.0, slow_rate, 0.0], dtype=np.float64)
                acc_body["left_thigh"] = tremor
                acc_body["left_calf"] = tremor
                omega_body["right_thigh"] = np.zeros(3, dtype=np.float64)
                omega_body["right_calf"] = np.zeros(3, dtype=np.float64)
                acc_body["right_thigh"] = np.zeros(3, dtype=np.float64)
                acc_body["right_calf"] = np.zeros(3, dtype=np.float64)
                hmd_lin_vel = np.zeros(3, dtype=np.float32)
                hmd_ang_vel = np.zeros(3, dtype=np.float32)

            elif state == 6:
                # 全身慢速同向转向 (Whole body turn, HMD 角速度 > 0.08)
                turn_rate = 0.10  # rad/s (~5.7°/s)
                hmd_heading += turn_rate * dt_i
                hmd_ang_vel = np.array([0.0, turn_rate, 0.0], dtype=np.float32)
                hmd_lin_vel = np.zeros(3, dtype=np.float32)
                for j in LOWER_BODY_JOINTS:
                    omega_body[j] = np.array([0.0, turn_rate, 0.0], dtype=np.float64)
                    acc_body[j] = np.zeros(3, dtype=np.float64)

            elif state == 7:
                # 静坐抖腿 (微小角速度摆动 < 0.03 rad/s，但高加速度颤动方差 > 0.08)
                fidget_w = 0.02 * math.sin(2.0 * math.pi * 3.0 * t[i])
                hmd_lin_vel = np.zeros(3, dtype=np.float32)
                hmd_ang_vel = np.zeros(3, dtype=np.float32)
                for j in LOWER_BODY_JOINTS:
                    omega_body[j] = np.array([fidget_w, 0.0, 0.0], dtype=np.float64)
                    acc_body[j] = self.rng.normal(0.0, 0.20, 3).astype(np.float64)

            elif state == 8:
                # 静止+周期性外力敲击震动 (每 30 秒发生一次 2.5 秒的桌面/机械震动)
                is_bump = (i % int(30.0 * self.sample_rate_hz)) < int(2.5 * self.sample_rate_hz)
                hmd_lin_vel = np.zeros(3, dtype=np.float32)
                hmd_ang_vel = np.zeros(3, dtype=np.float32)
                bump_acc = self.rng.normal(0.0, 0.6, 3).astype(np.float64) if is_bump else np.zeros(3, dtype=np.float64)
                for j in LOWER_BODY_JOINTS:
                    omega_body[j] = np.zeros(3, dtype=np.float64)
                    acc_body[j] = bump_acc

            else:
                # 转身漫步
                hmd_heading += 0.2 * dt_i
                for j in LOWER_BODY_JOINTS:
                    omega_body[j] = self.rng.normal(0.0, 0.1, 3).astype(np.float64)
                    acc_body[j] = self.rng.normal(0.0, 0.2, 3).astype(np.float64)
                hmd_lin_vel = np.array([0.3 * math.sin(hmd_heading), 0.0, 0.3 * math.cos(hmd_heading)], dtype=np.float32)
                hmd_pos += hmd_lin_vel * dt_i
                hmd_ang_vel = np.array([0.0, 0.2, 0.0], dtype=np.float32)

            # 4. 动力学四元数积分
            for j in LOWER_BODY_JOINTS:
                # (A) 干净姿态演化
                dq_clean = quat_exp_so3(omega_body[j] * dt_i)
                q_clean[j] = quat_mult(q_clean[j], dq_clean)
                q_clean[j] /= np.linalg.norm(q_clean[j])
                R_clean_arr[j][i] = quat_to_rotmat(q_clean[j])

                # (B) 含陀螺零偏与噪声的传感器测量角速度 (机体系)
                # 将世界竖直轴慢漂移投影到机体轴: b_body = R_raw^T * b_world
                R_curr_raw = quat_to_rotmat(q_raw[j])
                b_body = R_curr_raw.T @ b_gyro_world[j]
                noise_gyro = self.rng.normal(0.0, 0.002, 3).astype(np.float64) if self.enable_random_walk else np.zeros(3, dtype=np.float64)
                
                omega_imu_physical = omega_body[j] + b_body + noise_gyro

                # (C) 真实原始姿态动力学积分
                dq_raw = quat_exp_so3(omega_imu_physical * dt_i)
                q_raw[j] = quat_mult(q_raw[j], dq_raw)
                q_raw[j] /= np.linalg.norm(q_raw[j])
                R_raw = quat_to_rotmat(q_raw[j])
                R_raw_arr[j][i] = R_raw

                # (D) 物理加速度计测量值: a_meas = R_raw^T * (a_body + gravity) + noise
                gravity_world = np.array([0.0, 9.81, 0.0], dtype=np.float64)
                total_acc_world = acc_body[j] + gravity_world
                a_meas = R_raw.T @ total_acc_world
                if self.enable_accel_noise:
                    a_meas += self.rng.normal(0.0, 0.02, 3).astype(np.float64)

                omega_raw_arr[j][i] = omega_imu_physical.astype(np.float32)
                accel_raw_arr[j][i] = a_meas.astype(np.float32)

                # (E) 依据服务端 computeAngularVelocity 计算的因果姿态差分角速度
                if i == 0:
                    omega_derived_arr[j][i] = np.zeros(3, dtype=np.float32)
                else:
                    R_prev = R_raw_arr[j][i - 1]
                    diff_local = R_prev.T @ R_raw
                    tr = float(np.trace(diff_local))
                    cos_theta = np.clip(0.5 * (tr - 1.0), -1.0, 1.0)
                    theta = math.acos(cos_theta)
                    if theta < 1e-6:
                        omega_derived_arr[j][i] = np.zeros(3, dtype=np.float32)
                    else:
                        ax = float(diff_local[2, 1] - diff_local[1, 2])
                        ay = float(diff_local[0, 2] - diff_local[2, 0])
                        az = float(diff_local[1, 0] - diff_local[0, 1])
                        axis = np.array([ax, ay, az], dtype=np.float32)
                        norm = float(np.linalg.norm(axis))
                        if norm < 1e-6:
                            omega_derived_arr[j][i] = np.zeros(3, dtype=np.float32)
                        else:
                            omega_derived_arr[j][i] = (theta / dt_i) * (axis / norm)

            # 5. 组装 124D 特征
            hmd_rot = rot_y_matrix(hmd_heading).astype(np.float32)
            s0 = pack_stream_17d(np.zeros(3, dtype=np.float32), hmd_rot, hmd_lin_vel, hmd_ang_vel, valid=1.0, age=0.0)
            vd_chest_pos = hmd_pos - np.array([0.0, 0.3, 0.0], dtype=np.float32)
            s1 = pack_stream_17d(vd_chest_pos - hmd_pos, hmd_rot, hmd_lin_vel, hmd_ang_vel, valid=1.0, age=0.0)
            s2 = np.zeros(17, dtype=np.float32)
            s3 = np.zeros(17, dtype=np.float32)

            imu_streams = []
            for j in LOWER_BODY_JOINTS:
                is_valid = 1.0
                age = 0.0
                if self.scenario == "tracker_dropout_recovery" and j == "left_calf":
                    if 150.0 <= t[i] < 210.0:
                        is_valid = 0.0
                        age = 999.0

                s_imu = pack_stream_14d(
                    R_raw_arr[j][i],
                    accel_raw_arr[j][i],
                    omega_derived_arr[j][i],  # 服务端实际可得特征
                    valid=is_valid,
                    age=age
                )
                imu_streams.append(s_imu)

            features_124[i] = np.concatenate([s0, s1, s2, s3, *imu_streams], axis=-1)

        # 离线评估真值快照
        clean_rot_t0 = {j: R_clean_arr[j][0] for j in LOWER_BODY_JOINTS}
        clean_rot_t1 = {j: R_clean_arr[j][-1] for j in LOWER_BODY_JOINTS}
        raw_rot_t0 = {j: R_raw_arr[j][0] for j in LOWER_BODY_JOINTS}
        raw_rot_t1 = {j: R_raw_arr[j][-1] for j in LOWER_BODY_JOINTS}

        # 计算真实偏航漂移角
        e_true_t0 = {}
        e_true_t1 = {}
        for j in LOWER_BODY_JOINTS:
            d0 = raw_rot_t0[j] @ clean_rot_t0[j].T
            e_true_t0[j] = float(wrap_angle(math.atan2(float(d0[0, 2]), float(d0[2, 2]))))
            d1 = raw_rot_t1[j] @ clean_rot_t1[j].T
            e_true_t1[j] = float(wrap_angle(math.atan2(float(d1[0, 2]), float(d1[2, 2]))))

        gt_eval = {
            "drift_tier_deg_min": self.drift_tier,
            "scenario": self.scenario,
            "clean_rot_t0": clean_rot_t0,
            "clean_rot_t1": clean_rot_t1,
            "raw_rot_t0": raw_rot_t0,
            "raw_rot_t1": raw_rot_t1,
            "e_true_t0_rad": e_true_t0,
            "e_true_t1_rad": e_true_t1,
            "behavior_states": behavior_states
        }

        return PhysicsSessionStreamV4(
            session_id=session_id,
            duration_sec=self.duration_sec,
            dt=self.dt,
            num_frames=self.num_frames,
            timestamps=t,
            features_124=features_124,
            angular_velocity_sources={
                "raw_gyro": omega_raw_arr,
                "fused_rotation_delta": omega_derived_arr
            },
            ground_truth_eval=gt_eval,
            dts=dts
        )

"""
ai_corrector/corrector/test_p4_coordinate_and_simulation.py
===========================================================
Phase 13 Task P4: 世界坐标逆偏航左乘修正解析验证与观测语义对照单测

验证内容:
1. 解析验证世界坐标偏航修正顺序:
   R_raw = R_y(e_true) @ R_clean @ R_mount
   - 证明: 世界逆旋转左乘 R_y(-e_true) @ R_raw 严格精确复原 R_clean @ R_mount (误差 < 1e-6 rad)
   - 反证: 局部右乘 R_raw @ R_y(-e_true) 在非中立姿态 (俯仰、滚转、坐姿) 下产生严重的轴系畸变
2. 姿态差分角速度与陀螺角速度对照:
   - 证明因果连续四元数差分角速度在光滑运动下的解析一致性
   - 明确仿真直接注入陀螺偏置 vs 服务端姿态差分角速度的假设边界
"""

import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation


def rot_y_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float64)


def geodesic_angle_rad(R1: np.ndarray, R2: np.ndarray) -> float:
    R_diff = R1.T @ R2
    tr = float(np.trace(R_diff))
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(math.acos(cos_theta))


class TestPhase13P4PhysicsCoordinates(unittest.TestCase):

    def test_01_world_inverse_yaw_left_multiply_exactness(self):
        """解析验证: 世界逆偏航左乘修正能够精确恢复任意倾斜姿态与安装偏置下的姿态"""
        rng = np.random.RandomState(42)

        # 测试涵盖各种极端非中立姿态: 端坐 (大腿水平90度), 盘腿 (滚转+偏航), 俯冲等
        test_poses = [
            # 1. 大腿端坐水平姿势 (pitch = +90 deg)
            Rotation.from_euler("xyz", [90.0, 0.0, 0.0], degrees=True).as_matrix(),
            # 2. 盘腿姿势 (pitch = +60 deg, roll = +45 deg)
            Rotation.from_euler("xyz", [60.0, 0.0, 45.0], degrees=True).as_matrix(),
            # 3. 随机人体动作姿态 (50组随机姿态)
        ] + [Rotation.random(random_state=rng).as_matrix() for _ in range(50)]

        # 测试涵盖各种安装变换
        mountings = [
            np.eye(3, dtype=np.float64),
            Rotation.from_euler("xyz", [0.0, 180.0, 0.0], degrees=True).as_matrix(), # 反向安装
            Rotation.from_euler("xyz", [5.0, -10.0, 3.0], degrees=True).as_matrix(), # 现实安装倾斜
        ]

        # 测试涵盖各种正负偏航漂移角
        yaw_drifts_deg = [-179.0, -90.0, -35.2, -3.0, -0.05, 0.0, 0.05, 3.0, 45.0, 120.0, 180.0]

        for R_clean in test_poses:
            for R_mount in mountings:
                R_true_mounted = R_clean @ R_mount

                for yaw_deg in yaw_drifts_deg:
                    yaw_rad = math.radians(yaw_deg)
                    R_drift_world = rot_y_matrix(yaw_rad)

                    # 物理合成观测: 世界偏航漂移作用在世界坐标系 (最左侧)
                    R_raw = R_drift_world @ R_true_mounted

                    # 1. 正确的物理逆解: 世界逆偏航左乘
                    R_corrected_left = rot_y_matrix(-yaw_rad) @ R_raw
                    err_left = geodesic_angle_rad(R_corrected_left, R_true_mounted)

                    self.assertLess(
                        err_left, 1e-6,
                        f"Left-multiplication inverse yaw must perfectly recover pose, err={err_left} rad"
                    )

                    # 2. 错误的局部右乘反例测试 (Phase 6 发现的早期漏洞):
                    if abs(yaw_deg) > 5.0 and not np.allclose(R_clean, np.eye(3), atol=1e-3):
                        R_wrong_right = R_raw @ rot_y_matrix(-yaw_rad)
                        err_right = geodesic_angle_rad(R_wrong_right, R_true_mounted)
                        self.assertGreater(
                            err_right, 1e-3,
                            f"Right-multiplication of world yaw must fail on non-neutral poses, got err={err_right}"
                        )

    def test_02_observation_link_differentiation(self):
        """对比姿态因果差分角速度与真实角速度的连续极限"""
        dt = 1.0 / 60.0
        # 匀速旋转: 绕世界轴以 1.0 rad/s 旋转
        omega_true = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        R0 = np.eye(3, dtype=np.float64)
        R1 = rot_y_matrix(1.0 * dt)

        # 连续四元数差分
        q0 = Rotation.from_matrix(R0).as_quat() # [x, y, z, w] in scipy
        q1 = Rotation.from_matrix(R1).as_quat()

        # q_rel = q0.inv() * q1
        rot0 = Rotation.from_quat(q0)
        rot1 = Rotation.from_quat(q1)
        rot_rel = rot0.inv() * rot1
        rotvec = rot_rel.as_rotvec()
        omega_derived = rotvec / dt

        np.testing.assert_allclose(omega_derived, omega_true, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

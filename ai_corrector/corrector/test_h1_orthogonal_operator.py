"""
ai_corrector/corrector/test_h1_orthogonal_operator.py
====================================================
Phase 11 任务 H1: 正交 Swing-Twist 合成算子数值验证与退化测试

验证项:
1. 偏航绝对不变性: 快分支任意 3D 姿态输入下, R_out 相对 R_p9 的世界偏航变化在数值精度内恒为零 (< 1e-6 deg);
2. 门控退化: gate=0 或低于阈值时, R_out 严格等于 R_p9;
3. 慢偏航中立退化: e_hat=0 时, R_out 严格退化为 yaw-neutral Phase 6;
4. 快分支单位元退化: delta_R6=I 时, R_out 严格等于 R_p9;
5. 纯世界偏航扰动完全剔除: 当 delta_R6 仅含纯偏航时, D_swing 严格为单位阵, 全部偏航被丢弃;
6. 服务端局部右乘等价性: R_raw @ delta_R_server == R_out (误差 < 1e-7);
7. 极端姿态与奇点覆盖: 俯仰 90 度 (Gimbal lock)、180 度大翻转、负数四元数标量等。
"""

import math
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai_corrector.corrector.contract_v3 import LOWER_BODY_JOINTS
from ai_corrector.corrector.oracle_v3_geometry import (
    rot_y_matrix,
    geodesic_angle_rad,
    geodesic_angle_deg,
    extract_world_yaw_rad,
    wrap_angle,
)
from ai_corrector.corrector.hybrid_composer import (
    rotmat_to_quat_wxyz_safe,
    quat_wxyz_to_rotmat_safe,
    swing_twist_decomposition_y,
    slerp_from_identity_wxyz,
    compose_single_tracker_hybrid,
    HybridComposer,
)


class TestTaskH1OrthogonalOperator(unittest.TestCase):
    """Phase 11 任务 H1: 正交合成算子全面单元测试"""

    def test_01_zero_yaw_crosstalk_randomized_1000(self):
        """1000 次随机复合三维姿态测试: R_out 相对 R_p9 世界竖直偏航变化量必须 < 1e-6 度"""
        rng = np.random.RandomState(42)
        max_yaw_leak_deg = 0.0

        for _ in range(1000):
            # 随机原始姿态 R_raw
            R_raw = Rotation.random(random_state=rng).as_matrix()
            # 随机慢偏航估计 e_hat in [-pi, pi]
            e_hat = rng.uniform(-math.pi, math.pi)
            # 随机快分支预测残差 delta_R6 (任意 3D 旋转)
            delta_R6 = Rotation.random(random_state=rng).as_matrix()
            # 随机门控 in [0.3, 1.0]
            gate = rng.uniform(0.3, 1.0)

            out = compose_single_tracker_hybrid(
                R_raw=R_raw,
                e_hat_rad=e_hat,
                delta_R6=delta_R6,
                gate_p6=gate,
                gate_threshold=0.3
            )

            max_yaw_leak_deg = max(max_yaw_leak_deg, out.p9_yaw_diff_deg)
            self.assertLess(out.p9_yaw_diff_deg, 1e-6,
                            f"Yaw crosstalk detected: {out.p9_yaw_diff_deg:.2e} deg")

        self.assertLess(max_yaw_leak_deg, 1e-6)

    def test_02_gate_zero_degeneracy(self):
        """门控为 0 或低于阈值时，输出必须严格等于 Phase 9 慢修正结果"""
        rng = np.random.RandomState(123)
        for _ in range(50):
            R_raw = Rotation.random(random_state=rng).as_matrix()
            e_hat = rng.uniform(-math.pi, math.pi)
            delta_R6 = Rotation.random(random_state=rng).as_matrix()

            # (A) gate = 0.0
            out_zero = compose_single_tracker_hybrid(
                R_raw=R_raw, e_hat_rad=e_hat, delta_R6=delta_R6, gate_p6=0.0, gate_threshold=0.3
            )
            np.testing.assert_allclose(out_zero.R_out, out_zero.R_p9, atol=1e-10)
            self.assertFalse(out_zero.gate_active)
            self.assertEqual(out_zero.applied_swing_deg, 0.0)

            # (B) gate = 0.25 (低于 threshold 0.3)
            out_sub = compose_single_tracker_hybrid(
                R_raw=R_raw, e_hat_rad=e_hat, delta_R6=delta_R6, gate_p6=0.25, gate_threshold=0.3
            )
            np.testing.assert_allclose(out_sub.R_out, out_sub.R_p9, atol=1e-10)
            self.assertFalse(out_sub.gate_active)

    def test_03_zero_slow_drift_degeneracy(self):
        """当慢偏航估计 e_hat = 0 时，R_p9 严格等于 R_raw, 且 R_out 严格退化为 Phase 6-neutral"""
        rng = np.random.RandomState(456)
        for _ in range(50):
            R_raw = Rotation.random(random_state=rng).as_matrix()
            delta_R6 = Rotation.random(random_state=rng).as_matrix()
            gate = rng.uniform(0.5, 1.0)

            out = compose_single_tracker_hybrid(
                R_raw=R_raw, e_hat_rad=0.0, delta_R6=delta_R6, gate_p6=gate
            )
            np.testing.assert_allclose(out.R_p9, R_raw, atol=1e-10)
            np.testing.assert_allclose(out.R_out, out.R_p6_neutral, atol=1e-7)

    def test_04_identity_fast_delta_degeneracy(self):
        """当快分支预测单位元 (delta_R6 = I) 时，R_out 严格等于 R_p9"""
        rng = np.random.RandomState(789)
        for _ in range(50):
            R_raw = Rotation.random(random_state=rng).as_matrix()
            e_hat = rng.uniform(-math.pi, math.pi)

            out = compose_single_tracker_hybrid(
                R_raw=R_raw, e_hat_rad=e_hat, delta_R6=np.eye(3), gate_p6=1.0
            )
            np.testing.assert_allclose(out.R_out, out.R_p9, atol=1e-10)
            self.assertAlmostEqual(out.discarded_twist_deg, 0.0, places=5)
            self.assertAlmostEqual(out.applied_swing_deg, 0.0, places=5)

    def test_05_pure_yaw_perturbation_complete_rejection(self):
        """当 Phase 6 错误地输出纯偏航残差时，D_swing 必须为单位阵，所有偏航必须被 100% 丢弃"""
        rng = np.random.RandomState(999)
        for _ in range(50):
            R_raw = Rotation.random(random_state=rng).as_matrix()
            e_hat = rng.uniform(-math.pi, math.pi)

            # 注入在局部坐标系的纯偏航 (绕局部机体轴，或世界 Y 轴)
            # 无论输入何种纯偏航角 (例如 15 度)
            unwanted_yaw_deg = rng.uniform(-30.0, 30.0)
            unwanted_yaw_rad = math.radians(unwanted_yaw_deg)
            # 设 candidate 与 R9 仅差一个世界 Y 旋转
            delta_R6 = R_raw.T @ rot_y_matrix(unwanted_yaw_rad) @ R_raw

            out = compose_single_tracker_hybrid(
                R_raw=R_raw, e_hat_rad=e_hat, delta_R6=delta_R6, gate_p6=1.0
            )

            # 输出必须严格维持 R_p9，绝不接受 unwanted_yaw
            geo_diff = geodesic_angle_deg(out.R_out, out.R_p9)
            self.assertLess(geo_diff, 1e-5, f"Pure yaw was not completely rejected: diff={geo_diff:.2e} deg")
            # 滤除的角度应与注入的偏航角绝对值一致
            self.assertAlmostEqual(out.discarded_twist_deg, abs(unwanted_yaw_deg), delta=1e-4)

    def test_06_server_local_delta_equivalence(self):
        """验证供 SlimeVR 服务端局部消费的增量 delta_R_server 严格复现 R_out"""
        rng = np.random.RandomState(1337)
        for _ in range(100):
            R_raw = Rotation.random(random_state=rng).as_matrix()
            e_hat = rng.uniform(-math.pi, math.pi)
            delta_R6 = Rotation.random(random_state=rng).as_matrix()
            gate = rng.uniform(0.0, 1.0)

            out = compose_single_tracker_hybrid(
                R_raw=R_raw, e_hat_rad=e_hat, delta_R6=delta_R6, gate_p6=gate
            )

            # 服务端在收到 delta_R_server 后计算 R_applied = R_raw @ delta_R_server
            R_applied = R_raw @ out.delta_R_server
            max_err = float(np.max(np.abs(R_applied - out.R_out)))
            self.assertLess(max_err, 1e-7, f"Server local delta did not reproduce R_out: {max_err}")

    def test_07_extreme_gimbal_lock_and_singularity(self):
        """极端俯仰角 (+/-90 度万向节锁)、180 度翻转等临界姿态稳定性测试"""
        extreme_eulers = [
            [90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [0.0, 90.0, 0.0],
            [0.0, -90.0, 0.0],
            [0.0, 0.0, 180.0],
            [180.0, 0.0, 180.0],
            [89.999, 45.0, -89.999],
        ]

        for euler in extreme_eulers:
            R_raw = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
            e_hat = math.radians(45.0)
            delta_R6 = Rotation.from_euler("xyz", [10.0, 20.0, -5.0], degrees=True).as_matrix()

            out = compose_single_tracker_hybrid(
                R_raw=R_raw, e_hat_rad=e_hat, delta_R6=delta_R6, gate_p6=1.0
            )

            self.assertFalse(np.isnan(out.R_out).any())
            self.assertFalse(np.isinf(out.R_out).any())
            self.assertLess(out.p9_yaw_diff_deg, 1e-6)

    def test_08_hybrid_composer_multi_joint_stream(self):
        """测试四路 IMU 连续时序流的 HybridComposer 处理器与重置机制"""
        composer = HybridComposer(gate_threshold=0.3)
        composer.reset()

        # 模拟 10 帧四路输入
        rng = np.random.RandomState(2026)
        for _ in range(10):
            rots_raw = {j: Rotation.random(random_state=rng).as_matrix() for j in LOWER_BODY_JOINTS}
            e_hats = {j: rng.uniform(-1.0, 1.0) for j in LOWER_BODY_JOINTS}
            deltas_p6 = {j: Rotation.random(random_state=rng).as_matrix() for j in LOWER_BODY_JOINTS}
            gates_p6 = {j: rng.uniform(0.0, 1.0) for j in LOWER_BODY_JOINTS}

            step_out = composer.step(rots_raw, e_hats, deltas_p6, gates_p6)
            self.assertEqual(len(step_out), 4)
            for j in LOWER_BODY_JOINTS:
                self.assertIn(j, step_out)
                self.assertLess(step_out[j].p9_yaw_diff_deg, 1e-6)

        self.assertEqual(len(composer.history_records), 10)
        composer.reset()
        self.assertEqual(len(composer.history_records), 0)


if __name__ == "__main__":
    unittest.main()

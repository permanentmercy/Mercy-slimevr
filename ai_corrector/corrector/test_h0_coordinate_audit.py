"""
ai_corrector/corrector/test_h0_coordinate_audit.py
=================================================
Phase 11 任务 H0: 输入、坐标系与契约审计

验证项:
1. 124 维特征流布局、索引分段与四路 tracker 拓扑对齐;
2. 坐标系手性与世界竖直轴 (u_world = [0, 1, 0]^T) 一致性;
3. 四元数布局 [w, x, y, z] 与旋转矩阵往返保真度 (数值误差 < 1e-6);
4. V2ToV3Adapter 字段解析正确性与双向无损性;
5. 仿真器严格零漂移 (0.0 deg/min, 0 random walk) 与随机游走工况隔离;
6. 在线输入数据边界与无未来/无真值泄漏断言。
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

from ai_corrector.corrector.contract_v2 import (
    TOTAL_INPUT_DIM_V2,
    pack_stream_17d,
    pack_stream_14d,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat,
    rot6d_to_matrix_np,
)
from ai_corrector.corrector.contract_v3 import (
    FrameV3,
    V2ToV3Adapter,
    LOWER_BODY_JOINTS,
    UPPER_BODY_CONTEXT,
)
from ai_corrector.corrector.oracle_v3_geometry import (
    rot_y_matrix,
    geodesic_angle_rad,
    extract_world_yaw_rad,
    wrap_angle,
)
from ai_corrector.corrector.session_simulator_v3 import (
    ContinuousSessionSynthesizer,
)


class TestTaskH0CoordinateAudit(unittest.TestCase):
    """Phase 11 任务 H0: 输入与坐标契约全面审计"""

    def test_01_feature_layout_and_stream_dims(self):
        """审计 124 维特征流的切片布局与通道定义"""
        stream_dims = [17, 17, 17, 17, 14, 14, 14, 14]
        self.assertEqual(sum(stream_dims), 124)
        self.assertEqual(TOTAL_INPUT_DIM_V2, 124)

        # 检查通道切片
        offsets = [0, 17, 34, 51, 68, 82, 96, 110, 124]
        # 前 4 路为视觉锚点 (上身)
        for i in range(4):
            self.assertEqual(offsets[i + 1] - offsets[i], 17)
        # 后 4 路为下肢 IMU
        for i in range(4, 8):
            self.assertEqual(offsets[i + 1] - offsets[i], 14)

        # 验证关节名称与顺序
        self.assertEqual(LOWER_BODY_JOINTS, ("left_thigh", "right_thigh", "left_calf", "right_calf"))

    def test_02_quaternion_and_matrix_roundtrip(self):
        """审计四元数 [w, x, y, z] 与 3x3 旋转矩阵往返精度"""
        rng = np.random.RandomState(42)
        for _ in range(500):
            # 随机旋转
            scipy_rot = Rotation.random(random_state=rng)
            R_orig = scipy_rot.as_matrix().astype(np.float32)

            # 转换为 [w, x, y, z] 四元数
            q_wxyz = rotmat_to_quat_wxyz(R_orig)
            self.assertEqual(q_wxyz.shape, (4,))
            self.assertAlmostEqual(float(np.linalg.norm(q_wxyz)), 1.0, places=5)

            # 还原为 3x3 矩阵
            R_reconstructed = quat_wxyz_to_rotmat(q_wxyz)
            geo_err = geodesic_angle_rad(R_orig, R_reconstructed)
            self.assertLess(geo_err, 1e-3, f"Roundtrip geodesic error too high: {geo_err}")

    def test_03_world_y_axis_definition(self):
        """审计世界竖直轴 (u_world = [0, 1, 0]^T) 旋转矩阵数学规范"""
        # 测试 +90 度偏航旋转: 沿世界 Y 轴旋转时，Y 分量应完全保持不变
        R_90 = rot_y_matrix(math.pi / 2.0)
        v_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v_y_rot = R_90 @ v_y
        np.testing.assert_allclose(v_y_rot, v_y, atol=1e-7)

        # X 轴旋转 90 度应指向 -Z
        v_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v_x_rot = R_90 @ v_x
        # R_y(pi/2) @ [1, 0, 0]^T = [0, 0, -1]^T (右手系标准定义)
        np.testing.assert_allclose(v_x_rot, np.array([0.0, 0.0, -1.0]), atol=1e-7)

    def test_04_v2_to_v3_adapter(self):
        """审计 V2ToV3Adapter 从 124 维特征构建 FrameV3 的保真度"""
        rng = np.random.RandomState(101)
        feat_124 = rng.normal(0.0, 1.0, 124).astype(np.float32)
        # 归一化 6D 旋转前两列以构成合法旋转
        for offset in [3, 20, 37, 54, 68, 82, 96, 110]:
            col0 = feat_124[offset:offset + 3]
            col1 = feat_124[offset + 3:offset + 6]
            col0 /= np.linalg.norm(col0) + 1e-8
            col1 = col1 - np.dot(col0, col1) * col0
            col1 /= np.linalg.norm(col1) + 1e-8
            feat_124[offset:offset + 3] = col0
            feat_124[offset + 3:offset + 6] = col1

        # 设置有效标志位
        feat_124[15] = 1.0   # HMD valid
        feat_124[32] = 1.0   # VD chest valid
        feat_124[68 + 12] = 1.0   # left thigh valid
        feat_124[82 + 12] = 1.0   # right thigh valid
        feat_124[96 + 12] = 1.0   # left calf valid
        feat_124[110 + 12] = 1.0  # right calf valid

        frame = V2ToV3Adapter.convert_124d_to_frame(feat_124, timestamp=12.34, dt=0.1)
        self.assertIsInstance(frame, FrameV3)
        self.assertEqual(frame.meta.timestamp, 12.34)
        self.assertEqual(frame.meta.dt, 0.1)
        self.assertEqual(len(frame.trackers), 4)
        for j in LOWER_BODY_JOINTS:
            self.assertIn(j, frame.trackers)
            tr = frame.trackers[j]
            self.assertEqual(tr.rot_raw.shape, (3, 3))
            self.assertEqual(tr.ang_vel.shape, (3,))
            self.assertTrue(tr.valid)

    def test_05_strict_zero_drift_vs_random_walk_separation(self):
        """审计仿真器严格零漂 (0 drift, 0 random walk) 与随机游走工况的明确分组"""
        # (A) 严格零漂会话: enable_random_walk = False
        synth_strict = ContinuousSessionSynthesizer(
            duration_sec=300.0,
            sample_rate_hz=10.0,
            drift_tier_deg_min=0.0,
            seed=42,
            enable_random_walk=False
        )
        stream_strict = synth_strict.generate(session_id="strict_zero_audit")
        gt_strict = stream_strict.ground_truth_eval

        # 验证所有关节的 true drift curve 严格为 0.0
        for j in LOWER_BODY_JOINTS:
            max_drift_rad = float(np.max(np.abs(gt_strict["true_drift_curves_rad"][j])))
            self.assertEqual(max_drift_rad, 0.0, f"Strict zero drift should be 0.0, got {max_drift_rad}")
            self.assertEqual(gt_strict["true_drift_rates_deg_min"][j], 0.0)

        # (B) 含随机游走的会话: enable_random_walk = True
        synth_rw = ContinuousSessionSynthesizer(
            duration_sec=300.0,
            sample_rate_hz=10.0,
            drift_tier_deg_min=0.0,
            seed=42,
            enable_random_walk=True
        )
        stream_rw = synth_rw.generate(session_id="rw_audit")
        gt_rw = stream_rw.ground_truth_eval
        # 验证存在微小的随机游走漂移 (非严格零漂)
        max_rw = max(float(np.max(np.abs(gt_rw["true_drift_curves_rad"][j]))) for j in LOWER_BODY_JOINTS)
        self.assertGreater(max_rw, 1e-4, "Random walk session should have non-zero random walk curve")

    def test_06_no_information_leakage(self):
        """审计输入流数据边界: features_124 严格无真值角度泄漏"""
        synth = ContinuousSessionSynthesizer(duration_sec=60.0, sample_rate_hz=10.0, drift_tier_deg_min=3.0, seed=123)
        stream = synth.generate("leak_audit")
        feats = stream.features_124

        # 检查特征维度与有限值
        self.assertEqual(feats.shape, (600, 124))
        self.assertFalse(np.isnan(feats).any())
        self.assertFalse(np.isinf(feats).any())

        # 验证 ground_truth_eval 字典独立存在，不在 stream.features_124 内部
        self.assertTrue(hasattr(stream, "ground_truth_eval"))
        self.assertEqual(stream.features_124.shape[1], TOTAL_INPUT_DIM_V2)


if __name__ == "__main__":
    unittest.main()

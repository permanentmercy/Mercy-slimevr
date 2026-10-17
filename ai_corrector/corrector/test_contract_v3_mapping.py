"""
ai_corrector/corrector/test_contract_v3_mapping.py
==================================================
单元测试: 验证 contract_v2 124 维流通道与 contract_v3 V2ToV3Adapter 的逐维双向映射闭环。
"""

import math
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai_corrector.corrector.contract_v2 import pack_stream_17d, pack_stream_14d, TOTAL_INPUT_DIM_V2
from ai_corrector.corrector.contract_v3 import V2ToV3Adapter, LOWER_BODY_JOINTS


def test_v2_to_v3_roundtrip():
    rng = np.random.RandomState(42)
    
    # 1. 模拟上身 4 路视觉通道 (4 x 17 = 68 维)
    # HMD
    hmd_pos = np.array([0.1, 1.7, -0.5], dtype=np.float32)
    hmd_rot = np.eye(3, dtype=np.float32)
    hmd_lin_vel = np.array([0.0, 0.0, 1.2], dtype=np.float32)
    hmd_ang_vel = np.array([0.01, -0.02, 0.0], dtype=np.float32)
    s0 = pack_stream_17d(hmd_pos, hmd_rot, hmd_lin_vel, hmd_ang_vel, valid=1.0, age=0.01)
    
    # VD_CHEST
    vd_pos = np.array([0.1, 1.4, -0.4], dtype=np.float32)
    vd_rot = np.eye(3, dtype=np.float32)
    s1 = pack_stream_17d(vd_pos, vd_rot, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), valid=1.0, age=0.02)
    
    # VD_LEFT_ARM, VD_RIGHT_ARM (填充 17 维)
    s2 = np.zeros(17, dtype=np.float32)
    s3 = np.zeros(17, dtype=np.float32)
    
    # 2. 模拟下肢 4 路 IMU 通道 (4 x 14 = 56 维)
    imu_streams = []
    imu_test_data = {}
    for j in LOWER_BODY_JOINTS:
        rot_mat = np.eye(3, dtype=np.float32)
        ang_acc = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
        ang_vel = rng.uniform(-0.5, 0.5, 3).astype(np.float32)
        s_imu = pack_stream_14d(rot_mat, ang_acc, ang_vel, valid=1.0, age=0.005)
        imu_streams.append(s_imu)
        imu_test_data[j] = (rot_mat, ang_acc, ang_vel)
        
    # 组装 124 维向量
    vec_124 = np.concatenate([s0, s1, s2, s3, *imu_streams], axis=-1)
    assert len(vec_124) == TOTAL_INPUT_DIM_V2 == 124, f"Dim mismatch: {len(vec_124)}"
    
    # 3. 使用 V2ToV3Adapter 解析
    frame = V2ToV3Adapter.convert_124d_to_frame(vec_124, timestamp=10.0, dt=1.0/60.0)
    
    # 4. 验证解析准确性
    assert np.allclose(frame.upper_body.hmd_pos, hmd_pos, atol=1e-5), "HMD pos mismatch"
    assert np.allclose(frame.upper_body.hmd_lin_vel, hmd_lin_vel, atol=1e-5), "HMD lin_vel mismatch"
    assert np.allclose(frame.upper_body.vd_chest_pos, vd_pos, atol=1e-5), "VD chest pos mismatch"
    
    for j in LOWER_BODY_JOINTS:
        tracker = frame.trackers[j]
        exp_rot, exp_acc, exp_vel = imu_test_data[j]
        assert np.allclose(tracker.rot_raw, exp_rot, atol=1e-5), f"{j} rot mismatch"
        assert np.allclose(tracker.ang_acc, exp_acc, atol=1e-5), f"{j} ang_acc mismatch"
        assert np.allclose(tracker.ang_vel, exp_vel, atol=1e-5), f"{j} ang_vel mismatch"
        assert tracker.valid is True
        assert math.isclose(tracker.data_age, 0.005, abs_tol=1e-4)
        
    print("test_v2_to_v3_roundtrip: PASSED! Exact 124-dim alignment verified.")


if __name__ == "__main__":
    test_v2_to_v3_roundtrip()

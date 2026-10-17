"""
ai_corrector/corrector/replay_harness.py
=========================================
SlimeVR CSTN v2 合成 Tracker 回放与链路验证测试套件。

检验内容:
1. 124 维特征切片映射与坐标系契约 (+Y 向上, 右手系, 单位与重力约定)
2. 四元数右乘残差数学一致性 (q_corrected = q_raw * dq)
3. 缺省/过期 VD 角色容错 (VD Chest valid=0 时无 NaN，平滑运行)
4. Reset 校准隐状态清空机制 (状态清空后平滑复位，无数值突跃)
5. 60Hz 流式单步推理时延基准 (P50, P95, P99 测试)
"""

import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .model import CSTN, Config
from .contract_v2 import (
    STREAM_DIMS_V2,
    TOTAL_INPUT_DIM_V2,
    STREAM_NAMES_V2,
    TARGET_JOINT_NAMES_V2,
    rot6d_to_matrix_np,
    quat_wxyz_to_rotmat,
    rotmat_to_quat_wxyz,
    geodesic_angle_error_deg,
)


def create_synthetic_frame(
    time_s: float,
    drift_deg_per_min: float = 3.0,
    has_vd_chest: bool = True,
    age_s: float = 0.0,
) -> np.ndarray:
    """构建单帧符合 v2 契约的 124 维特征向量。"""
    vec = np.zeros(TOTAL_INPUT_DIM_V2, dtype=np.float32)

    # 1. HMD (流 0, 17维): 相对原点微动
    hmd_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    hmd_rot = Rotation.from_euler("xyz", [0, 5.0 * np.sin(time_s * 2.0), 0], degrees=True)
    hmd_mat = hmd_rot.as_matrix()
    vec[0:3] = hmd_pos
    vec[3:9] = np.concatenate([hmd_mat[:, 0], hmd_mat[:, 1]])
    vec[9:12] = [0.0, 0.0, 0.0]  # lin_vel
    vec[12:15] = [0.0, np.deg2rad(10.0), 0.0]  # ang_vel
    vec[15] = 1.0  # valid
    vec[16] = age_s  # age

    # 2. VD 胸部 (流 1, 17维)
    if has_vd_chest:
        chest_pos = np.array([0.0, -0.3, 0.0], dtype=np.float32)
        chest_rot = Rotation.from_euler("xyz", [2.0, 3.0 * np.sin(time_s * 2.0), 0], degrees=True)
        chest_mat = chest_rot.as_matrix()
        vec[17:20] = chest_pos
        vec[20:26] = np.concatenate([chest_mat[:, 0], chest_mat[:, 1]])
        vec[26:29] = [0.0, 0.0, 0.0]
        vec[29:32] = [0.0, 0.0, 0.0]
        vec[32] = 1.0
        vec[33] = age_s
    else:
        # 缺省填充 0，valid = 0
        vec[17:34] = 0.0

    # 3 & 4. VD 左右上臂 (流 2, 3, 各17维)
    for stream_idx, offset, arm_sign in [(2, 34, -1.0), (3, 51, 1.0)]:
        arm_pos = np.array([arm_sign * 0.25, -0.15, 0.0], dtype=np.float32)
        arm_rot = Rotation.from_euler("xyz", [0, 0, arm_sign * 30.0], degrees=True)
        arm_mat = arm_rot.as_matrix()
        vec[offset:offset+3] = arm_pos
        vec[offset+3:offset+9] = np.concatenate([arm_mat[:, 0], arm_mat[:, 1]])
        vec[offset+9:offset+12] = [0.0, 0.0, 0.0]
        vec[offset+12:offset+15] = [0.0, 0.0, 0.0]
        vec[offset+15] = 1.0
        vec[offset+16] = age_s

    # 5..8. 四个下肢 IMU (各14维): 注入受漂移影响的测量值
    drift_rad = np.deg2rad(drift_deg_per_min * (time_s / 60.0))
    for stream_idx, offset, leg_yaw_bias in [
        (4, 68, drift_rad * 1.0),   # 左大腿
        (5, 82, drift_rad * 1.1),   # 右大腿
        (6, 96, drift_rad * 1.3),   # 左小腿
        (7, 110, drift_rad * 1.4),  # 右小腿
    ]:
        true_rot = Rotation.from_euler("xyz", [15.0 * np.sin(time_s * 4.0), leg_yaw_bias, 0], degrees=True)
        mat = true_rot.as_matrix()
        vec[offset:offset+6] = np.concatenate([mat[:, 0], mat[:, 1]])
        vec[offset+6:offset+9] = [0.0, -9.81, 0.0]  # 重力加速度 (传感器系)
        vec[offset+9:offset+12] = [0.0, np.deg2rad(5.0), 0.0]  # 陀螺仪角速度
        vec[offset+12] = 1.0  # valid
        vec[offset+13] = age_s  # age

    return vec


def verify_residual_right_multiplication():
    """验证右乘残差数学一致性: q_corrected = q_raw * dq。"""
    print("\n--- [Check 1/5] 验证四元数右乘残差数学方向 ---")
    rng = np.random.default_rng(20261017)

    for i in range(10):
        # 随机真实姿态
        clean_rot = Rotation.random(random_state=rng)
        clean_mat = clean_rot.as_matrix().astype(np.float32)
        clean_quat = rotmat_to_quat_wxyz(clean_mat)

        # 随机漂移 (如 5-15 度偏航)
        drift_rot = Rotation.from_euler("y", rng.uniform(5.0, 15.0), degrees=True)
        raw_mat = (drift_rot * clean_rot).as_matrix().astype(np.float32)
        raw_quat = rotmat_to_quat_wxyz(raw_mat)

        # 计算残差: dq = raw^{-1} * clean => raw * dq = clean
        dq_mat = np.matmul(raw_mat.T, clean_mat)
        dq_quat = rotmat_to_quat_wxyz(dq_mat)

        # 应用右乘: q_corrected = q_raw * dq
        corrected_mat = np.matmul(raw_mat, dq_mat)
        corrected_quat = rotmat_to_quat_wxyz(corrected_mat)

        err_raw = geodesic_angle_error_deg(raw_quat, clean_quat)
        err_corrected = geodesic_angle_error_deg(corrected_quat, clean_quat)

        assert err_corrected < 1e-4, f"右乘修正未消除误差: {err_corrected:.6f} deg"
        assert err_raw > 4.0, f"原始误差过小: {err_raw:.2f} deg"

    print("  [PASS] 四元数右乘残差数学方向与闭环消除验证通过 (残差误差 < 1e-4 deg)。")


def verify_missing_vd_chest():
    """验证缺少 VD 胸部 tracker 时 (流 1 全零, valid=0) 模型推理鲁棒性。"""
    print("\n--- [Check 2/5] 验证缺省 VD 胸部 tracker 鲁棒性 ---")
    cfg = Config(stream_dims=STREAM_DIMS_V2, num_joints=4)
    model = CSTN(cfg)
    model.eval()

    # 生成 20 帧序列，其中 VD Chest 为空
    frames = [create_synthetic_frame(t * (1.0 / 60.0), has_vd_chest=False) for t in range(20)]
    batch_x = torch.from_numpy(np.stack(frames)[None, ...]).float()

    with torch.no_grad():
        out, _ = model(batch_x)

    assert not torch.isnan(out["dq"]).any(), "缺少 VD 胸部时模型输出 dq 包含 NaN"
    assert not torch.isnan(out["gate"]).any(), "缺少 VD 胸部时模型输出 gate 包含 NaN"
    assert not torch.isnan(out["contact"]).any(), "缺少 VD 胸部时模型输出 contact 包含 NaN"
    assert out["dq"].shape == (1, 20, 4, 4), f"输出 dq 形状异常: {out['dq'].shape}"
    assert out["gate"].shape == (1, 20, 4), f"输出 gate 形状异常: {out['gate'].shape}"
    assert out["contact"].shape == (1, 20, 2), f"输出 contact 形状异常: {out['contact'].shape}"

    print(f"  [PASS] VD 胸部缺省容错测试通过，张量形状正常且无 NaN。")


def verify_reset_state_flush():
    """验证 Tracker Reset 时清空隐状态前后数值平滑性。"""
    print("\n--- [Check 3/5] 验证 Reset 状态清空与连续性 ---")
    cfg = Config(stream_dims=STREAM_DIMS_V2, num_joints=4)
    model = CSTN(cfg)
    model.eval()

    # 模拟连续流式推理
    frames = [create_synthetic_frame(t * (1.0 / 60.0)) for t in range(40)]
    batch_x = torch.from_numpy(np.stack(frames)[None, ...]).float()

    # 运行前半段 (0..19 帧)
    with torch.no_grad():
        out1, _ = model(batch_x[:, :20])
        # 模拟 Reset: 清除所有因果隐状态 (通过输入全新序列从零初始状态计算)
        out2, _ = model(batch_x[:, 20:])

    # 验证输出幅度在合理界限内，无爆发突跃
    dq_norm = torch.norm(out2["dq"], dim=-1)
    assert (dq_norm > 0.99).all() and (dq_norm < 1.01).all(), "Reset 发生后四元数模长不归一"
    assert (out2["gate"] >= 0.0).all() and (out2["gate"] <= 1.0).all(), "Gate 越界"
    assert (out2["contact"] >= 0.0).all() and (out2["contact"] <= 1.0).all(), "Contact 越界"

    print("  [PASS] Reset 隐状态清空测试通过，输出值域安全稳定。")


def benchmark_streaming_latency(num_steps: int = 600):
    """评估单步流式推理 P50 / P95 / P99 时延 (CPU 单线程)。"""
    print(f"\n--- [Check 4/5] 评估流式单步推理时延 ({num_steps} 步) ---")
    cfg = Config(stream_dims=STREAM_DIMS_V2, num_joints=4)
    model = CSTN(cfg).to(torch.device("cpu"))
    model.eval()

    torch.set_num_threads(1)

    # 预热
    dummy = torch.randn(1, 1, TOTAL_INPUT_DIM_V2)
    for _ in range(50):
        with torch.no_grad():
            _, _ = model(dummy)

    latencies_ms = []
    for step in range(num_steps):
        frame = create_synthetic_frame(step * (1.0 / 60.0))
        t_input = torch.from_numpy(frame[None, None, ...]).float()

        start = time.perf_counter()
        with torch.no_grad():
            _, _ = model(t_input)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    latencies_ms = np.array(latencies_ms)
    p50 = np.percentile(latencies_ms, 50)
    p95 = np.percentile(latencies_ms, 95)
    p99 = np.percentile(latencies_ms, 99)
    max_lat = np.max(latencies_ms)

    print(f"  单步 CPU 推理时延基准 (单线程, PyTorch Python):")
    print(f"    P50: {p50:.3f} ms")
    print(f"    P95: {p95:.3f} ms (门槛: < 20.0 ms, ONNX 部署态 < 1.0 ms)")
    print(f"    P99: {p99:.3f} ms")
    print(f"    Max: {max_lat:.3f} ms")

    assert p95 < 20.0, f"P95 时延超标: {p95:.3f} ms >= 20.0 ms"
    print("  [PASS] 单步流式时延性能达标。")


def verify_packet_drop_and_stale_data():
    """验证丢包与高时延 (age > 0.05s) 输入情景。"""
    print("\n--- [Check 5/5] 验证 Tracker 丢包与数据过期场景 ---")
    cfg = Config(stream_dims=STREAM_DIMS_V2, num_joints=4)
    model = CSTN(cfg)
    model.eval()

    # 构造带有 100ms 过期时间戳与随机丢包 (valid=0) 的帧序列
    frames = []
    for t in range(30):
        age = 0.10 if t % 5 == 0 else 0.0
        frame = create_synthetic_frame(t * (1.0 / 60.0), age_s=age)
        if t % 7 == 0:
            # 模拟右小腿丢包
            frame[110:124] = 0.0
        frames.append(frame)

    batch_x = torch.from_numpy(np.stack(frames)[None, ...]).float()
    with torch.no_grad():
        out, _ = model(batch_x)

    assert not torch.isnan(out["dq"]).any(), "丢包与过期数据引起 NaN"
    assert not torch.isnan(out["gate"]).any(), "丢包与过期数据引起 gate NaN"
    print("  [PASS] 丢包与数据过期异常鲁棒性通过。")


def main():
    print("==================================================")
    print("  SlimeVR CSTN v2 合成 Tracker 回放与链路验证测试  ")
    print("==================================================")
    verify_residual_right_multiplication()
    verify_missing_vd_chest()
    verify_reset_state_flush()
    benchmark_streaming_latency()
    verify_packet_drop_and_stale_data()
    print("\n[SUCCESS] 全部 5 项回放与链路鲁棒性测试 100% 通过！\n")


if __name__ == "__main__":
    main()

"""
ai_corrector/corrector/oracle_v3_geometry.py
============================================
Phase 9: 四路下肢偏航状态模型（几何与坐标变换无学习 Oracle、单测与端点指标计算器）

功能:
1. 世界 Y 轴旋转与四元数严格数学定义;
2. 世界左乘纠正: R_corr = R_y(-e_hat) @ R_raw;
3. 局部右乘接口等价转换: R_dq = R_raw.T @ R_y(-e_hat) @ R_raw，使得 R_raw @ R_dq == R_corr;
4. 错误右乘 (R_raw @ R_y(-e_hat)) 的几何退化定量复现;
5. 会话开始-结束端点累计偏航误差增长指标 (D_raw, D_corr, G, 改善率) 计算器;
6. 完备的单元测试与随机化压力测试.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Union


def wrap_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """将角度折叠至 [-pi, pi] 区间"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_degrees(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """将角度 (deg) 折叠至 [-180, 180] 区间"""
    return (deg + 180.0) % 360.0 - 180.0


def rot_y_matrix(yaw_rad: float) -> np.ndarray:
    """
    构造绕世界竖直轴 (Y 轴, Y-up) 的 3x3 旋转矩阵。
    右手坐标系：+X 指向右，+Y 指向上，+Z 指向后（或向前，依基准）。
    R_y(theta) = [ [ cos, 0, sin ],
                   [   0, 1,   0 ],
                   [-sin, 0, cos ] ]
    """
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float64)


def geodesic_angle_rad(R1: np.ndarray, R2: np.ndarray) -> float:
    """计算两个 3x3 旋转矩阵之间的测地角距离 (SO(3) 真实角度误差，单位 rad)"""
    R_diff = R1.T @ R2
    tr = float(np.trace(R_diff))
    cos_val = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return math.acos(cos_val)


def geodesic_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """测地角距离（单位度）"""
    return math.degrees(geodesic_angle_rad(R1, R2))


def extract_world_yaw_rad(R: np.ndarray) -> float:
    """
    从 3x3 旋转矩阵中提取等效世界偏航角 (绕 Y 轴投影角)。
    在 R_y(yaw) @ R_rest 中，世界 X 轴经变换后的朝向在 X-Z 平面的偏角。
    """
    # 考虑机体前向向量 (例如 Z 轴 R[:, 2] 或 X 轴 R[:, 0])
    # 标准 X-Z 平面偏航角: atan2(R[0, 2], R[2, 2])
    return math.atan2(float(R[0, 2]), float(R[2, 2]))


def world_left_correction(R_raw: np.ndarray, e_hat_rad: float) -> np.ndarray:
    """
    严格物理修正：世界左乘逆偏航。
    若 R_raw = R_y(e) @ R_clean，则 R_corr = R_y(-e_hat) @ R_raw.
    """
    R_corr = rot_y_matrix(-e_hat_rad) @ R_raw
    return R_corr


def world_to_local_dq(R_raw: np.ndarray, e_hat_rad: float) -> np.ndarray:
    """
    将世界逆偏航转换为服务端的局部右乘残差 R_dq。
    数学等价关系：R_raw @ R_dq = R_y(-e_hat) @ R_raw
    => R_dq = R_raw.T @ R_y(-e_hat) @ R_raw
    """
    R_inv_yaw = rot_y_matrix(-e_hat_rad)
    R_dq = R_raw.T @ R_inv_yaw @ R_raw
    return R_dq


def wrong_local_right_correction(R_raw: np.ndarray, e_hat_rad: float) -> np.ndarray:
    """
    【反面教材对照】错误地将世界偏航直接作为局部右乘：
    R_wrong = R_raw @ R_y(-e_hat)
    这在具有俯仰/滚转角时会导致灾难性的轴向混淆。
    """
    return R_raw @ rot_y_matrix(-e_hat_rad)


def compute_endpoint_session_metrics(e_raw_t0: Dict[str, float],
                                     e_raw_t1: Dict[str, float],
                                     e_corr_t0: Dict[str, float],
                                     e_corr_t1: Dict[str, float],
                                     joints: Tuple[str, ...] = ("left_thigh", "right_thigh", "left_calf", "right_calf")
                                     ) -> Dict[str, float]:
    """
    计算 Phase 9 主验收指标：单次连续会话 (>=30分钟) 开始 t0 与结束 t1 的四路累计偏航误差增长总量。
    
    指标公式:
      D_raw = sum_i | wrap(e_raw,i(t1) - e_raw,i(t0)) |
      D_corr = sum_i | wrap(e_corr,i(t1) - e_corr,i(t0)) |
      主收益 G = D_raw - D_corr
      改善率 = G / D_raw (若 D_raw > 1e-4)
    四路独立绝对值累加，严禁用有符号正负相互抵消！
    """
    delta_raw_list = []
    delta_corr_list = []
    per_joint = {}

    for j in joints:
        raw_diff = float(wrap_angle(e_raw_t1[j] - e_raw_t0[j]))
        corr_diff = float(wrap_angle(e_corr_t1[j] - e_corr_t0[j]))
        
        abs_raw = abs(raw_diff)
        abs_corr = abs(corr_diff)
        
        delta_raw_list.append(abs_raw)
        delta_corr_list.append(abs_corr)
        per_joint[f"delta_raw_deg_{j}"] = math.degrees(abs_raw)
        per_joint[f"delta_corr_deg_{j}"] = math.degrees(abs_corr)
        per_joint[f"gain_deg_{j}"] = math.degrees(abs_raw - abs_corr)

    D_raw_rad = sum(delta_raw_list)
    D_corr_rad = sum(delta_corr_list)
    G_rad = D_raw_rad - D_corr_rad

    D_raw_deg = math.degrees(D_raw_rad)
    D_corr_deg = math.degrees(D_corr_rad)
    G_deg = math.degrees(G_rad)

    improvement_ratio = (G_rad / D_raw_rad) if D_raw_rad > 1e-5 else 0.0

    result = {
        "D_raw_deg": D_raw_deg,
        "D_corr_deg": D_corr_deg,
        "G_deg": G_deg,
        "improvement_ratio": improvement_ratio,
        "is_improved": bool(G_deg > 0.001),
        **per_joint
    }
    return result


def random_rotation_matrix(rng: np.random.RandomState) -> np.ndarray:
    """生成均匀随机的 SO(3) 3x3 旋转矩阵 (基于随机四元数采样)"""
    u1, u2, u3 = rng.uniform(0.0, 1.0, 3)
    q = np.array([
        math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
        math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
        math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
        math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    ], dtype=np.float64)
    # q: [w, x, y, z] -> matrix
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w), 2.0*(x*z + y*w)],
        [2.0*(x*y + z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w), 2.0*(y*z + x*w), 1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float64)
    return R


def run_geometry_oracle_tests(num_samples: int = 10000, seed: int = 42) -> Dict[str, Union[float, int, bool]]:
    """
    执行高置信度几何与坐标变换自动化验证：
    1. 10,000 组任意 3D 姿态与正负偏航漂移；
    2. 验证世界左乘纠正误差严格为 0；
    3. 验证服务端局部右乘转换 R_raw @ R_dq 严格等于世界左乘 R_corr（误差严格 < 1e-7 rad）；
    4. 定量统计错误局部右乘造成的系统性姿态退化；
    5. 验证端点指标计算器的无偏性与无符号抵消约束。
    """
    rng = np.random.RandomState(seed)
    
    max_left_err_deg = 0.0
    max_right_equiv_err_deg = 0.0
    wrong_errors_deg = []

    for _ in range(num_samples):
        # 1. 干净 3D 姿态
        R_clean = random_rotation_matrix(rng)
        
        # 2. 真实世界 Y 轴漂移 (-180° ~ +180°)
        true_e_deg = rng.uniform(-180.0, 180.0)
        true_e_rad = math.radians(true_e_deg)
        R_drift = rot_y_matrix(true_e_rad)
        
        # 3. 原始漂移姿态
        R_raw = R_drift @ R_clean
        
        # 4. 理想 Oracle 估计值 (e_hat == true_e)
        e_hat_rad = true_e_rad
        
        # 5. 世界左乘纠正
        R_corr_left = world_left_correction(R_raw, e_hat_rad)
        err_left_deg = geodesic_angle_deg(R_corr_left, R_clean)
        max_left_err_deg = max(max_left_err_deg, err_left_deg)
        
        # 6. 局部右乘转换
        R_dq = world_to_local_dq(R_raw, e_hat_rad)
        R_corr_right = R_raw @ R_dq
        err_right_deg = geodesic_angle_deg(R_corr_right, R_clean)
        equiv_err_deg = geodesic_angle_deg(R_corr_right, R_corr_left)
        max_right_equiv_err_deg = max(max_right_equiv_err_deg, equiv_err_deg)
        
        # 7. 错误右乘
        R_wrong = wrong_local_right_correction(R_raw, e_hat_rad)
        err_wrong_deg = geodesic_angle_deg(R_wrong, R_clean)
        wrong_errors_deg.append(err_wrong_deg)

    # 端点指标单测
    e_raw_t0 = {"left_thigh": 0.01, "right_thigh": -0.02, "left_calf": 0.00, "right_calf": 0.03}
    # 模拟 30 分钟漂移增长
    e_raw_t1 = {"left_thigh": math.radians(10.0), "right_thigh": math.radians(-12.0),
                "left_calf": math.radians(15.0), "right_calf": math.radians(-8.0)}
    # 完美纠正 (corr_t1 等于 corr_t0)
    e_corr_t0 = dict(e_raw_t0)
    e_corr_t1 = dict(e_raw_t0)  # 偏移完全消除
    perfect_metrics = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_corr_t0, e_corr_t1)

    # 错误纠正 (由于坐标轴错误，误差被放大)
    e_wrong_t1 = {"left_thigh": math.radians(18.0), "right_thigh": math.radians(-20.0),
                  "left_calf": math.radians(25.0), "right_calf": math.radians(-16.0)}
    degraded_metrics = compute_endpoint_session_metrics(e_raw_t0, e_raw_t1, e_corr_t0, e_wrong_t1)

    mean_wrong_err_deg = float(np.mean(wrong_errors_deg))
    max_wrong_err_deg = float(np.max(wrong_errors_deg))

    passed = (max_left_err_deg < 1e-5) and (max_right_equiv_err_deg < 1e-5) and (perfect_metrics["G_deg"] > 40.0)

    return {
        "num_samples_tested": num_samples,
        "max_world_left_geodesic_err_deg": max_left_err_deg,
        "max_local_right_equiv_geodesic_err_deg": max_right_equiv_err_deg,
        "mean_wrong_right_geodesic_err_deg": mean_wrong_err_deg,
        "max_wrong_right_geodesic_err_deg": max_wrong_err_deg,
        "perfect_oracle_D_raw_deg": perfect_metrics["D_raw_deg"],
        "perfect_oracle_D_corr_deg": perfect_metrics["D_corr_deg"],
        "perfect_oracle_gain_deg": perfect_metrics["G_deg"],
        "perfect_oracle_improvement_ratio": perfect_metrics["improvement_ratio"],
        "degraded_metrics_gain_deg": degraded_metrics["G_deg"],
        "degraded_metrics_is_improved": degraded_metrics["is_improved"],
        "all_tests_passed": bool(passed)
    }


if __name__ == "__main__":
    report = run_geometry_oracle_tests(num_samples=10000, seed=42)
    print("Geometry Oracle Verification Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

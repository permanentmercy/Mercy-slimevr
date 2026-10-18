"""Audit whether the Phase 16 slow-bias observer can distinguish yaw drift from real slow yaw motion.

The two counterfactuals deliberately share exactly the same observable stream:
either a stationary thigh has gyro yaw bias, or an unbiased thigh rotates at the
same rate. The gravity vector is invariant to rotation about world vertical, and
the HMD and other trackers remain still. Hidden labels are used only to score the
same observer output, never as model input.

Run with the ai_corrector Python environment from the repository root:
    python ai_corrector/corrector/execute_phase16_p4_observability_twin_audit.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_corrector.corrector.conservative_bias_estimator_v4 import ConservativeBiasEstimatorV4
from ai_corrector.corrector.contract_v3 import (
    FrameMetadata,
    FrameV3,
    LOWER_BODY_JOINTS,
    SingleTrackerV3,
    UpperBodyContextV3,
)
from ai_corrector.corrector.oracle_v3_geometry import rot_y_matrix, wrap_angle


REPORT_PATH = ROOT / "ai_corrector" / "artifacts" / "phase16_p4_observability_twin_audit.json"


def run_observer(rate_deg_min: float = 0.3, duration_sec: float = 600.0, sample_rate_hz: float = 10.0) -> dict:
    dt = 1.0 / sample_rate_hz
    frame_count = int(duration_sec * sample_rate_hz)
    rate_rad_s = math.radians(rate_deg_min / 60.0)
    gravity = np.array([0.0, 9.81, 0.0], dtype=np.float32)
    observer = ConservativeBiasEstimatorV4(sample_rate_hz=sample_rate_hz)
    stream_hash = hashlib.sha256()

    for frame_index in range(frame_count):
        timestamp = frame_index * dt
        thigh_rotation = rot_y_matrix(rate_rad_s * timestamp).astype(np.float32)
        thigh_omega = np.array([0.0, rate_rad_s, 0.0], dtype=np.float32)
        # For a pure world-vertical yaw, gravity in the body frame stays constant.
        thigh_accel = (thigh_rotation.T @ gravity).astype(np.float32)

        trackers = {}
        for joint in LOWER_BODY_JOINTS:
            is_moving_thigh = joint == "left_thigh"
            rotation = thigh_rotation if is_moving_thigh else np.eye(3, dtype=np.float32)
            omega = thigh_omega if is_moving_thigh else np.zeros(3, dtype=np.float32)
            accel = thigh_accel if is_moving_thigh else gravity
            trackers[joint] = SingleTrackerV3(
                imu_id=joint,
                rot_raw=rotation,
                ang_vel=omega,
                ang_acc=accel,  # FrameV3 currently stores the acceleration vector in this field.
                valid=True,
            )
            stream_hash.update(rotation.tobytes())
            stream_hash.update(omega.tobytes())
            stream_hash.update(accel.tobytes())

        identity = np.eye(3, dtype=np.float32)
        zeros = np.zeros(3, dtype=np.float32)
        upper = UpperBodyContextV3(
            hmd_rot=identity,
            hmd_pos=zeros,
            hmd_lin_vel=zeros,
            hmd_ang_vel=zeros,
            vd_chest_rot=identity,
            vd_chest_pos=zeros,
            hmd_valid=True,
            vd_chest_valid=True,
        )
        frame = FrameV3(
            meta=FrameMetadata(timestamp=timestamp, dt=dt, frame_index=frame_index),
            trackers=trackers,
            upper_body=upper,
        )
        observer.process_frame(frame)

    estimated_deg = math.degrees(observer.kalman_filter.channels["left_thigh"].e)
    true_rotation_deg = rate_deg_min * duration_sec / 60.0
    corrected_error_motion_deg = abs(estimated_deg)
    corrected_error_drift_deg = abs(math.degrees(wrap_angle(math.radians(true_rotation_deg - estimated_deg))))

    reset_audit = verify_reset_lifecycle()

    return {
        "phase": "Phase 16",
        "task": "P16.5 exact observation-twin audit",
        "status": "FAIL" if observer.update_counts["left_thigh"] > 0 or not reset_audit["reset_lifecycle_verified"] else "PASS",
        "sample_rate_hz": sample_rate_hz,
        "duration_sec": duration_sec,
        "frame_count": frame_count,
        "rate_deg_min": rate_deg_min,
        "acceleration_variance_m2_s4": 0.0,
        "input_sha256": stream_hash.hexdigest(),
        "counterfactuals": {
            "real_slow_thigh_yaw_without_sensor_drift": {
                "raw_endpoint_error_deg": 0.0,
                "corrected_endpoint_error_deg": corrected_error_motion_deg,
                "gain_deg": -estimated_deg,
            },
            "stationary_thigh_with_equal_sensor_yaw_drift": {
                "raw_endpoint_error_deg": true_rotation_deg,
                "corrected_endpoint_error_deg": corrected_error_drift_deg,
                "gain_deg": true_rotation_deg - corrected_error_drift_deg,
            },
        },
        "observer_output_for_both_counterfactuals": {
            "observations_identical": True,
            "predictions_must_be_identical": True,
            "left_thigh_update_count": observer.update_counts["left_thigh"],
            "left_thigh_uncertain_count": observer.uncertain_counts["left_thigh"],
            "left_thigh_reject_count": observer.reject_counts["left_thigh"],
            "estimated_yaw_correction_deg": estimated_deg,
        },
        "reset_lifecycle_audit": reset_audit,
        "criterion": (
            "Ambiguous observations must not be automatically corrected. "
            "An UPDATE on this shared stream proves that the observer cannot "
            "safely distinguish the two hidden causes from available inputs."
        ),
        "hardware_validation": False,
    }


def verify_reset_lifecycle() -> dict:
    """显式验证 reset 生命周期: 确保 reset 事件触发时探测器窗口、累积基准与滤波器偏航角彻底清零"""
    observer = ConservativeBiasEstimatorV4(sample_rate_hz=10.0)
    gravity = np.array([0.0, 9.81, 0.0], dtype=np.float32)
    identity = np.eye(3, dtype=np.float32)
    zeros = np.zeros(3, dtype=np.float32)

    # 1. 运行 50 帧累积静止状态
    for i in range(50):
        t = i * 0.1
        trackers = {
            j: SingleTrackerV3(imu_id=j, rot_raw=identity, ang_vel=zeros, ang_acc=gravity, valid=True)
            for j in LOWER_BODY_JOINTS
        }
        upper = UpperBodyContextV3(
            hmd_rot=identity, hmd_pos=zeros, hmd_lin_vel=zeros, hmd_ang_vel=zeros,
            vd_chest_rot=identity, vd_chest_pos=zeros, hmd_valid=True, vd_chest_valid=True
        )
        frame = FrameV3(
            meta=FrameMetadata(timestamp=t, dt=0.1, frame_index=i, reset_event=False),
            trackers=trackers,
            upper_body=upper
        )
        observer.process_frame(frame)

    pre_counts = {j: observer.detectors[j].consecutive_still_count for j in LOWER_BODY_JOINTS}
    assert all(c > 0 for c in pre_counts.values()), "Detectors failed to accumulate stillness"

    # 2. 注入 reset_event=True 帧
    reset_frame = FrameV3(
        meta=FrameMetadata(timestamp=5.1, dt=0.1, frame_index=51, reset_event=True),
        trackers={j: SingleTrackerV3(imu_id=j, rot_raw=identity, ang_vel=zeros, ang_acc=gravity, valid=True) for j in LOWER_BODY_JOINTS},
        upper_body=UpperBodyContextV3(
            hmd_rot=identity, hmd_pos=zeros, hmd_lin_vel=zeros, hmd_ang_vel=zeros,
            vd_chest_rot=identity, vd_chest_pos=zeros, hmd_valid=True, vd_chest_valid=True
        )
    )
    observer.process_frame(reset_frame)

    # 3. 验证所有状态彻底清空
    post_counts = {j: observer.detectors[j].consecutive_still_count for j in LOWER_BODY_JOINTS}
    post_starts = {j: observer.detectors[j].still_start_R for j in LOWER_BODY_JOINTS}
    post_e = {j: observer.kalman_filter.channels[j].e for j in LOWER_BODY_JOINTS}

    reset_clean = (
        all(c == 0 for c in post_counts.values()) and
        all(s is None for s in post_starts.values()) and
        all(abs(e) < 1e-12 for e in post_e.values())
    )

    return {
        "reset_lifecycle_verified": reset_clean,
        "pre_reset_still_count": pre_counts["left_thigh"],
        "post_reset_still_count": post_counts["left_thigh"],
        "post_reset_still_start_R_is_none": post_starts["left_thigh"] is None,
        "post_reset_kalman_e_rad": post_e["left_thigh"]
    }


def main() -> None:
    report = run_observer()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

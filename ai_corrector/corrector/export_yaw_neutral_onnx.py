"""
ai_corrector/corrector/export_yaw_neutral_onnx.py
=================================================
Phase 12 Task S0: Phase 11 Yaw-Neutral CSTN 专用的 ONNX 流式导出器与 ABI 核验脚本。

核心特性:
1. 包含世界竖直 Y 轴 Swing-Twist 正交投影算子 (torch_world_swing_local_delta);
2. 导出标准流式 ABI:
   - 输入: x [1, 1, 124], h [8, 1, 1, hidden], c [8, 1, 1, hidden]
   - 输出: dq [1, 1, 4, 4], gate [1, 1, 4], contact [1, 1, 2], hn [8, 1, 1, hidden], cn [8, 1, 1, hidden]
3. 验证 ONNX checker 与 8 帧因果时序递归回放一致性 (atol <= 2e-5);
4. 记录并输出包含 SHA-256、张量维度与 ABI 签名的报告。
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

from ai_corrector.corrector.model import Corrector, Config
from ai_corrector.corrector.yaw_neutral_cstn import YawNeutralCSTN


class StreamingYawNeutralWrapper(nn.Module):
    """
    ONNX 流式推理包装器:
    接收单帧 [1, 1, 124] 与递归隐状态 (h, c)，输出局部右乘 Swing 四元数残差、门控、触地概率及下时刻状态。
    """
    def __init__(self, yaw_neutral_model: YawNeutralCSTN):
        super().__init__()
        self.model = yaw_neutral_model

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        out, (hn, cn) = self.model(x, (h, c))
        # 提取经过世界 Y 轴 Swing-Twist 投影后的局部 Swing 四元数
        dq = out["dq"]          # [1, 1, 4, 4]
        gate = out["gate"]      # [1, 1, 4]
        contact = out["contact"]  # [1, 1, 2]
        return dq, gate, contact, hn, cn


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def export_and_verify(checkpoint_path: Path, output_onnx_path: Path) -> dict:
    torch.manual_seed(42)
    torch.set_num_threads(1)

    ckpt_sha256 = compute_file_sha256(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_dict = ckpt["config"]
    config = Config(**config_dict)

    base_model = Corrector(config)
    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()

    yaw_neutral_model = YawNeutralCSTN(base_model).eval()
    wrapper = StreamingYawNeutralWrapper(yaw_neutral_model).eval()

    in_dim = sum(config.stream_dims)  # 124
    hidden = config.hidden            # 96
    num_streams = len(config.stream_dims)  # 8

    # 构造单帧输入与状态测试张量
    x = torch.randn(1, 1, in_dim, dtype=torch.float32)
    # 构造非平凡的 6D 连续旋转以通过 Gram-Schmidt
    for offset in (68, 82, 96, 110):
        x[:, :, offset:offset + 3] = torch.tensor([1.0, 0.0, 0.0])
        x[:, :, offset + 3:offset + 6] = torch.tensor([0.0, 1.0, 0.0])

    h, c = base_model.initial_state(1)

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting PyTorch model to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        wrapper,
        (x, h, c),
        str(output_onnx_path),
        opset_version=14,
        input_names=["x", "h", "c"],
        output_names=["dq", "gate", "contact", "hn", "cn"],
        dynamic_axes=None,  # 影子流式固定为单帧 batch=1, time=1
        dynamo=False
    )

    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX structure check passed.")

    onnx_sha256 = compute_file_sha256(output_onnx_path)

    # 验证 ONNXRuntime 与 PyTorch 8 帧递归对齐
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output_onnx_path), options, providers=["CPUExecutionProvider"])

    oh = h.detach().numpy().copy()
    oc = c.detach().numpy().copy()

    max_diffs = {"dq": 0.0, "gate": 0.0, "contact": 0.0, "hn": 0.0, "cn": 0.0}

    with torch.no_grad():
        for frame_idx in range(8):
            x_test = torch.randn(1, 1, in_dim, dtype=torch.float32)
            for offset in (68, 82, 96, 110):
                x_test[:, :, offset:offset + 3] = torch.tensor([1.0, 0.0, 0.0])
                x_test[:, :, offset + 3:offset + 6] = torch.tensor([0.0, 1.0, 0.0])

            py_dq, py_gate, py_contact, py_hn, py_cn = wrapper(x_test, h, c)
            ort_out = session.run(None, {"x": x_test.numpy(), "h": oh, "c": oc})

            diff_dq = float(np.max(np.abs(py_dq.numpy() - ort_out[0])))
            diff_gate = float(np.max(np.abs(py_gate.numpy() - ort_out[1])))
            diff_contact = float(np.max(np.abs(py_contact.numpy() - ort_out[2])))
            diff_hn = float(np.max(np.abs(py_hn.numpy() - ort_out[3])))
            diff_cn = float(np.max(np.abs(py_cn.numpy() - ort_out[4])))

            max_diffs["dq"] = max(max_diffs["dq"], diff_dq)
            max_diffs["gate"] = max(max_diffs["gate"], diff_gate)
            max_diffs["contact"] = max(max_diffs["contact"], diff_contact)
            max_diffs["hn"] = max(max_diffs["hn"], diff_hn)
            max_diffs["cn"] = max(max_diffs["cn"], diff_cn)

            for key, val in max_diffs.items():
                assert val < 2e-5, f"Frame {frame_idx} {key} parity failed: max_diff={val}"

            h, c = py_hn, py_cn
            oh, oc = ort_out[3], ort_out[4]

    print(f"8-frame recurrent parity passed! Max diffs: {max_diffs}")

    # 性能冒烟测试 (100 次推理取均值)
    times_ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        session.run(None, {"x": x_test.numpy(), "h": oh, "c": oc})
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    report = {
        "task": "Phase 12 Task S0 Model ABI & ONNX Export",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": "PASS",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": ckpt_sha256,
            "config": config_dict
        },
        "onnx_model": {
            "path": str(output_onnx_path),
            "sha256": onnx_sha256,
            "size_bytes": output_onnx_path.stat().st_size,
            "opset_version": 14,
            "io_abi": {
                "inputs": {
                    "x": {"shape": [1, 1, in_dim], "type": "float32"},
                    "h": {"shape": [num_streams, 1, 1, hidden], "type": "float32"},
                    "c": {"shape": [num_streams, 1, 1, hidden], "type": "float32"}
                },
                "outputs": {
                    "dq": {"shape": [1, 1, 4, 4], "type": "float32", "desc": "local right-multiplied swing quaternion [w,x,y,z]"},
                    "gate": {"shape": [1, 1, 4], "type": "float32", "desc": "per-joint confidence [0,1]"},
                    "contact": {"shape": [1, 1, 2], "type": "float32", "desc": "foot contact probability [left, right]"},
                    "hn": {"shape": [num_streams, 1, 1, hidden], "type": "float32"},
                    "cn": {"shape": [num_streams, 1, 1, hidden], "type": "float32"}
                }
            }
        },
        "verification": {
            "onnx_checker": "PASS",
            "recurrent_parity_frames": 8,
            "max_discrepancy": max_diffs,
            "tolerance_atol": 2e-5,
            "cpu_single_thread_ms": {
                "mean": float(np.mean(times_ms[10:])),
                "p95": float(np.percentile(times_ms[10:], 95))
            }
        }
    }

    report_path = output_onnx_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved ABI report to {report_path}")
    return report


def main():
    base_dir = Path(__file__).resolve().parents[2]
    ckpt_path = base_dir / "ai_corrector" / "artifacts" / "cstn_v2_yaw_neutral_world_best.pt"
    out_onnx = base_dir / "ai_corrector" / "artifacts" / "cstn_v2_yaw_neutral_world.onnx"
    export_and_verify(ckpt_path, out_onnx)


if __name__ == "__main__":
    main()

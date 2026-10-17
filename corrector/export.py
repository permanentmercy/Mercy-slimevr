"""Export FP32 streaming ONNX with explicit recurrent state and parity checks."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import Corrector, Config


class Streaming(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, h, c):
        out, (hn, cn) = self.model(x, (h, c))
        return out["dq"], out["gate"], out["contact"], hn, cn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Corrector(Config(**ckpt["config"])).eval()
    model.load_state_dict(ckpt["state_dict"])
    wrapper = Streaming(model).eval()
    x = torch.randn(1, 1, 135)
    h, c = model.initial_state(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, (x, h, c), str(args.output), opset_version=14,
                      input_names=["x", "h", "c"], output_names=["dq", "gate", "contact", "hn", "cn"],
                      dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(args.output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"])
    oh, oc = h.detach().numpy(), c.detach().numpy()
    # Check recurrent rollout, not just a zero-state frame.
    with torch.no_grad():
        for _ in range(8):
            x = torch.randn(1, 1, 135)
            expected = wrapper(x, h, c)
            actual = session.run(None, {"x": x.numpy(), "h": oh, "c": oc})
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b.numpy(), atol=2e-5, rtol=2e-4)
            h, c = expected[-2:]
            oh, oc = actual[-2:]
    times = []
    for i in range(1030):
        start = time.perf_counter_ns()
        actual = session.run(None, {"x": x.numpy(), "h": oh, "c": oc})
        elapsed = (time.perf_counter_ns()-start)/1e6
        oh, oc = actual[-2:]
        if i >= 30:
            times.append(elapsed)
    report = {"onnx_checker": "PASS", "recurrent_parity_frames": 8,
              "atol": 2e-5, "rtol": 2e-4,
              "bytes": args.output.stat().st_size,
              "config": ckpt["config"], "onnxruntime": ort.__version__,
              "cpu_single_thread_ms": {"mean": float(np.mean(times)), "p95": float(np.percentile(times, 95)), "frames": len(times)},
              "scope": "Engineering smoke weights; no accuracy or end-to-end certification"}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ONNX checker and 8-frame recurrent parity passed: {args.output}")


if __name__ == "__main__":
    main()

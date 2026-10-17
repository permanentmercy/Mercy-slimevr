"""Executable engineering acceptance. Does not certify tracking accuracy."""
import argparse
import io
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch
from .model import Corrector, Config
from .rotations import rotvec_to_quat, multiply, residual_target, blend
from .loss import objective
from .train import load_data


def run(hidden=256, frames=1000):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = Corrector(Config(hidden=hidden)).eval()
    # Exercise nonzero, near-bound residuals instead of only identity initialization.
    with torch.no_grad():
        model.pose[-1].weight.normal_(0, .1)
        model.pose[-1].bias.fill_(4.)
    x = torch.randn(2, 7, 135)
    with torch.no_grad():
        full, _ = model(x)
        changed = x.clone()
        changed[:, 4:] += 10
        future, _ = model(changed)
        for key in full:
            torch.testing.assert_close(full[key][:, :4], future[key][:, :4])
        state, collected = None, {k: [] for k in full}
        for i in range(x.shape[1]):
            out, state = model(x[:, i:i+1], state)
            for key in out:
                collected[key].append(out[key])
        for key in full:
            torch.testing.assert_close(full[key], torch.cat(collected[key], 1), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(full["dq"].norm(dim=-1), torch.ones(2, 7, 6))
        angle = 2*torch.atan2(full["dq"][..., 1:].norm(dim=-1), full["dq"][..., 0])
        assert angle.max() <= math.radians(8)+1e-6
        for key in ("gate", "contact"):
            assert ((full[key] >= 0) & (full[key] <= 1)).all()
        raw = rotvec_to_quat(torch.randn(2, 6, 3))
        clean = rotvec_to_quat(torch.randn(2, 6, 3))
        dq = residual_target(raw, clean)
        torch.testing.assert_close(multiply(raw, dq), clean, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(blend(raw, dq, torch.zeros(2, 6)), raw)
        blended = blend(raw, dq, torch.ones(2, 6))
        torch.testing.assert_close((blended*clean).sum(-1).abs(), torch.ones(2, 6))
        # Identity SLERP must have finite outputs.
        identity = rotvec_to_quat(torch.zeros(2, 6, 3))
        assert torch.isfinite(blend(raw, identity, torch.ones(2, 6))).all()
    zero = torch.zeros(2, 6, 3, requires_grad=True)
    rotvec_to_quat(zero).sum().backward()
    assert torch.isfinite(zero.grad).all()
    try:
        model(torch.zeros(1, 2, 134))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid feature width accepted")
    # All heads receive finite gradients and optimization reduces a fixed tiny loss.
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    targets = {"dq": rotvec_to_quat(torch.full((2, 7, 6, 3), .02)),
               "gate": torch.ones(2, 7, 6), "contact": torch.zeros(2, 7, 2)}
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        pred, _ = model(x)
        loss = objective(pred, targets)
        loss.backward()
        for head in (model.pose, model.gate, model.contact):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses
    arrays = {"x": x.numpy(), **{k: v.numpy() for k, v in targets.items()}}
    dataset_buffer = io.BytesIO()
    np.savez(dataset_buffer, **arrays)
    dataset_buffer.seek(0)
    assert len(load_data(dataset_buffer)) == 2
    for key, value in (("gate", 2.), ("contact", float("nan")), ("dq", 0.)):
        bad = {k: v.copy() for k, v in arrays.items()}
        bad[key][...] = value
        dataset_buffer = io.BytesIO()
        np.savez(dataset_buffer, **bad)
        dataset_buffer.seek(0)
        try:
            load_data(dataset_buffer)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {key} accepted")
    opposite = {**targets, "dq": -targets["dq"]}
    torch.testing.assert_close(objective(pred, targets), objective(pred, opposite))
    # Round-trip checkpoint restores identical output.
    model.eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = Corrector(Config(hidden=hidden)).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        for key, value in model(x)[0].items():
            torch.testing.assert_close(value, restored(x)[0][key])
        sample = torch.randn(1, 1, 135)
        state = None
        for _ in range(30):
            _, state = model(sample, state)
        timings = []
        for _ in range(frames):
            start = time.perf_counter_ns()
            _, state = model(sample, state)
            timings.append((time.perf_counter_ns()-start)/1e6)
    timings.sort()
    return {"engineering_checks": "PASS", "hidden": hidden,
            "parameters": sum(p.numel() for p in model.parameters()),
            "fp32_state_dict_bytes": buffer.getbuffer().nbytes,
            "cpu_streaming_ms": {"mean": sum(timings)/frames, "p95": timings[int(.95*(frames-1))], "frames": frames},
            "tiny_training_loss": [losses[0], losses[-1]],
            "torch": torch.__version__, "python": platform.python_version(),
            "device": platform.processor(), "threads": 1,
            "accuracy_acceptance": "NOT EVALUATED: no AMASS/DIP or real-device test set",
            "integration_acceptance": "NOT EVALUATED: no live SlimeVR/VD/VRChat session",
            "latency_scope": "CPU PyTorch batch=1 single-frame model only; not ONNX or end-to-end",
            "checks": ["future-frame isolation", "stream/full sequence parity", "quaternion unit norm and 8-degree bound",
                       "right-multiplication target", "gate bypass and full correction", "finite three-head gradients",
                       "tiny-batch loss decrease", "quaternion sign invariance", "checkpoint round trip",
                       "NPZ loading and invalid-label rejection", "zero-angle finite gradients", "invalid feature width rejection"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("corrector/acceptance-report.json"))
    p.add_argument("--frames", type=int, default=1000)
    args = p.parse_args()
    if args.frames < 1:
        p.error("frames must be positive")
    result = [run(h, args.frames) for h in (256, 96)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Train on already prepared, sequence-split NPZ tensors; not raw AMASS files."""
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .model import Corrector, Config
from .loss import objective


def load_data(path):
    with np.load(path, allow_pickle=False) as a:
        data = [torch.from_numpy(a[k].astype(np.float32)) for k in ("x", "dq", "gate", "contact")]
    x, dq, gate, contact = data
    n, t = x.shape[:2]
    if x.shape != (n, t, 135) or dq.shape != (n, t, 6, 4) or gate.shape != (n, t, 6) or contact.shape != (n, t, 2):
        raise ValueError("invalid NPZ shapes")
    if n == 0 or t == 0 or any(not torch.isfinite(v).all() for v in data):
        raise ValueError("empty or nonfinite dataset")
    if not torch.allclose(dq.norm(dim=-1), torch.ones_like(gate), atol=1e-3):
        raise ValueError("dq must contain unit quaternions")
    if any(((v < 0) | (v > 1)).any() for v in (gate, contact)):
        raise ValueError("labels must be probabilities")
    return TensorDataset(*data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    torch.manual_seed(42)
    config = Config(hidden=args.hidden)
    model = Corrector(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loader = DataLoader(load_data(args.data), batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        total = 0.
        for x, dq, gate, contact in loader:
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = objective(pred, dict(dq=dq, gate=gate, contact=contact))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"epoch={epoch+1} loss={total/len(loader):.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=asdict(config), state_dict=model.state_dict(),
                    trained_on=str(args.data), epochs=args.epochs), args.output)


if __name__ == "__main__":
    main()

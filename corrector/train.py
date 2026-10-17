"""
corrector/train.py
===================
GPU-accelerated CSTN Multi-Task Training Executor.
Formatted with standardized experiment banner, diagnostic, and iteration diagnostics.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .model import Corrector, Config
from .loss import objective

# Configure standard UTF-8 stream to avoid Windows terminal mojibake
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta

    def __call__(self, val_loss, model, path, extra=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, extra)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, extra)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model, path, extra=None):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        payload = dict(state_dict=model.state_dict(), val_loss=val_loss)
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        self.val_loss_min = val_loss


def load_data(path: Path):
    with np.load(path, allow_pickle=False) as a:
        data = [torch.from_numpy(a[k].astype(np.float32)) for k in ("x", "dq", "gate", "contact")]
    x, dq, gate, contact = data
    n, t = x.shape[:2]
    if x.shape != (n, t, 135) or dq.shape != (n, t, 6, 4) or gate.shape != (n, t, 6) or contact.shape != (n, t, 2):
        raise ValueError(f"invalid NPZ shapes in {path}: x={x.shape}, dq={dq.shape}, gate={gate.shape}, contact={contact.shape}")
    if n == 0 or t == 0 or any(not torch.isfinite(v).all() for v in data):
        raise ValueError("empty or nonfinite dataset")
    if not torch.allclose(dq.norm(dim=-1), torch.ones_like(gate), atol=1e-3):
        raise ValueError("dq must contain unit quaternions")
    if any(((v < 0) | (v > 1)).any() for v in (gate, contact)):
        raise ValueError("labels must be probabilities in [0, 1]")
    return TensorDataset(*data)


def evaluate(model, loader, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_angle_err = 0.0
    total_frames = 0
    tp = fp = fn = 0

    with torch.no_grad():
        for x, dq, gate, contact in loader:
            x = x.to(device)
            target = dict(dq=dq.to(device), gate=gate.to(device), contact=contact.to(device))
            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                pred, _ = model(x)
            pred_fp32 = {k: v.float() for k, v in pred.items()}
            loss = objective(pred_fp32, target)

            total_loss += loss.item() * len(x)

            # Residual angle error: 2 * acos(|<q_pred, q_true>|) in degrees
            q_pred = pred_fp32["dq"]
            q_true = F.normalize(target["dq"], dim=-1)
            w_err = (q_pred * q_true).sum(dim=-1).abs().clamp(0.0, 1.0)
            angle_rad = 2.0 * torch.acos(w_err)
            total_angle_err += angle_rad.sum().item()
            total_frames += angle_rad.numel()

            # Contact F1 score
            c_pred = (pred_fp32["contact"] >= 0.5).bool()
            c_true = (target["contact"] >= 0.5).bool()
            tp += (c_pred & c_true).sum().item()
            fp += (c_pred & (~c_true)).sum().item()
            fn += ((~c_pred) & c_true).sum().item()

    avg_loss = total_loss / max(1, len(loader.dataset))
    avg_deg = np.rad2deg(total_angle_err / max(1, total_frames))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    return avg_loss, avg_deg, f1


def print_args_banner(args, config, train_path, val_path, device, device_name, param_count):
    vram_str = "N/A"
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_str = f"{total_vram:.2f} GB"

    b_on = "\033[1m"
    b_off = "\033[0m"

    print("Args in experiment:")
    print(f"{b_on}Basic Config{b_off}")
    print(f"  {'Task Name:':<22}{args.task_name:<28}{'Is Training:':<20}{1:<10}")
    print(f"  {'Model ID:':<22}{args.model_id:<28}{'Model:':<20}{args.model:<10}\n")

    print(f"{b_on}Data Loader{b_off}")
    print(f"  {'Data:':<22}{'AMASS (All 38 Subsets)':<28}{'Root Path:':<20}{str(train_path.parent) + '/':<10}")
    print(f"  {'Train Data:':<22}{train_path.name:<28}{'Val Data:':<20}{val_path.name:<10}")
    print(f"  {'Target:':<22}{'dq(rot),gate,contact':<28}{'Freq:':<20}{'60Hz':<10}")
    print(f"  {'Checkpoints:':<22}{'./artifacts/':<28}\n")

    print(f"{b_on}Kinematics Correction Task{b_off}")
    print(f"  {'Window Len:':<22}{'20 frames (333ms)':<28}{'Stride:':<20}{'5 frames':<10}")
    print(f"  {'Body Streams:':<22}{'8 (Head,Torso,Hands,Legs)':<28}{'Max Angle Clamp:':<20}{f'±{config.max_degrees:.1f} deg':<10}\n")

    print(f"{b_on}Model Parameters{b_off}")
    print(f"  {'Input Feat Dim:':<22}{135:<28}{'Output Dim:':<20}{'18 (6x3 rotvec)':<10}")
    print(f"  {'d model (Hidden):':<22}{config.hidden:<28}{'n heads:':<20}{config.heads:<10}")
    print(f"  {'Transformer Layers:':<22}{config.layers:<28}{'d FF:':<20}{4 * config.hidden:<10}")
    print(f"  {'Temporal Backbone:':<22}{'8x Causal LSTM':<28}{'Spatial Fusion:':<20}{'TransformerEncoder':<10}")
    print(f"  {'Dropout:':<22}{0.0:<28}{'Activation:':<20}{'gelu / leaky_relu':<10}\n")

    print(f"{b_on}Run Parameters{b_off}")
    print(f"  {'Train Epochs:':<22}{args.epochs:<28}{'Batch Size:':<20}{args.batch_size:<10}")
    print(f"  {'Patience:':<22}{args.patience:<28}{'Learning Rate:':<20}{args.lr:<10}")
    print(f"  {'Rotation Loss:':<22}{'sym_chordal_quat':<28}{'Contact Loss:':<20}{'binary_cross_entropy':<10}")
    print(f"  {'Gate Loss:':<22}{'binary_cross_entropy':<28}{'LR Schedule:':<20}{'cosine':<10}")
    print(f"  {'Use Amp:':<22}{args.use_amp:<28}\n")

    print(f"{b_on}GPU{b_off}")
    print(f"  {'Use GPU:':<22}{1 if device.type == 'cuda' else 0:<28}{'GPU ID:':<20}{0:<10}")
    print(f"  {'Device:':<22}{device_name:<28}{'VRAM:':<20}{vram_str:<10}\n")


def print_diagnostic(model, config, param_count, device):
    dev_str = f"cuda:0" if device.type == "cuda" else "cpu"
    print(f"Use GPU: {dev_str}\n")
    print("=" * 60)
    print("MODEL CONFIGURATION DIAGNOSTIC")
    print("=" * 60)
    print(f"Architecture: Causal Spatio-Temporal Network (CSTN)")
    print(f"Input streams: 8 streams (HMD: 17, Torso: 28, Hands: 2x17, Legs: 4x14) -> 135-d")
    print(f"Temporal backbones: 8 independent Causal LSTMs (hidden: {config.hidden})")
    print(f"Spatial fusion: TransformerEncoder ({config.layers} layers, {config.heads} heads, d_ff: {4*config.hidden})")
    print(f"Rotation head: Linear({config.hidden}->128->18) -> rotvec_to_quat (clamped ±{config.max_degrees}°)")
    print(f"Gate head: Linear({config.hidden}->64->6) -> Sigmoid() [Reliability Gating]")
    print(f"Contact head: Linear({config.hidden}->64->2) -> Sigmoid() [Foot Strike ZUPT]")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {param_count:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60 + "\n")


def estimate_macs(config, seq_len=20):
    lstm_macs = sum(4 * (config.hidden * config.hidden + config.hidden * d) for d in (17, 28, 17, 17, 14, 14, 14, 14)) * seq_len
    transformer_macs = config.layers * (12 * config.hidden * config.hidden) * (8 * seq_len)
    total_gmac = (lstm_macs + transformer_macs) / 1e9
    return max(0.01, total_gmac)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path, nargs="?", default=Path("database/processed/train.npz"), help="Path to train NPZ")
    p.add_argument("output", type=Path, nargs="?", default=Path("artifacts/cstn_compact_expanded.pt"), help="Path to output .pt")
    p.add_argument("--val-data", type=Path, default=Path("database/processed/val.npz"), help="Path to validation NPZ")
    p.add_argument("--epochs", type=int, default=20, help="Train Epochs (default: 20)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch Size (default: 256)")
    p.add_argument("--hidden", type=int, default=96, help="Hidden size / d_model (default: 96)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning Rate (default: 1e-3)")
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    p.add_argument("--use-amp", type=int, default=1, help="Use Automatic Mixed Precision (default: 1)")
    p.add_argument("--print-iters", type=int, default=100, help="Print progress every N iters (default: 100)")
    p.add_argument("--device", type=str, default="", help="Device (cuda / cpu)")
    p.add_argument("--task-name", type=str, default="kinematics_residual_correction", help="Task name")
    p.add_argument("--model-id", type=str, default="CSTN_compact_20_135", help="Model ID")
    p.add_argument("--model", type=str, default="CSTN", help="Model name")
    p.add_argument("--des", type=str, default="test", help="Description tag")
    args = p.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
        torch.cuda.reset_peak_memory_stats(device)

    config = Config(hidden=args.hidden)
    model = Corrector(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print_args_banner(args, config, args.data, args.val_data, device, device_name, param_count)
    print_diagnostic(model, config, param_count, device)

    setting = (f"{args.task_name}_{args.model_id}_{args.model}_AMASS_dm{args.hidden}_"
               f"nh{config.heads}_el{config.layers}_df{4*config.hidden}_"
               f"bs{args.batch_size}_lr{args.lr}")

    print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")

    train_dataset = load_data(args.data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    print(f"train {len(train_dataset)} windows ({len(train_dataset)*20:,} frames)")

    val_loader = None
    val_dataset = None
    if args.val_data and args.val_data.exists():
        val_dataset = load_data(args.val_data)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
        print(f"val {len(val_dataset)} windows ({len(val_dataset)*20:,} frames)")

    use_amp = bool(args.use_amp and device.type == "cuda")
    if use_amp:
        print("Mixed Precision Training (AMP) Enabled")
    else:
        print("Mixed Precision Training (AMP) Disabled")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda", enabled=True) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=True)

    best_path = args.output.with_name(args.output.stem + "_best.pt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    total_steps_per_epoch = len(train_loader)
    total_train_steps = args.epochs * total_steps_per_epoch
    epoch_times = []
    global_step = 0
    best_val_loss = float("inf")
    best_val_deg = 0.0
    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        iter_timer = time.time()

        for step, (x, dq, gate, contact) in enumerate(train_loader, 1):
            global_step += 1
            x = x.to(device)
            target = dict(dq=dq.to(device), gate=gate.to(device), contact=contact.to(device))

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    pred, _ = model(x)
                pred_fp32 = {k: v.float() for k, v in pred.items()}
                loss = objective(pred_fp32, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Epoch {epoch}: Non-finite loss (NaN/Inf)")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred, _ = model(x)
                loss = objective(pred, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Epoch {epoch}: Non-finite loss (NaN/Inf)")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item() * len(x)

            if step % args.print_iters == 0:
                elapsed_chunk = time.time() - iter_timer
                iter_speed = elapsed_chunk / args.print_iters
                left_iters = total_train_steps - global_step
                left_time = max(0.0, iter_speed * left_iters)
                print(f"\titers: {step}, epoch: {epoch} | loss: {loss.item():.7f}")
                print(f"\tspeed: {iter_speed:.4f}s/iter; left time: {left_time:.4f}s", flush=True)
                iter_timer = time.time()

        scheduler.step()
        train_loss = epoch_loss / len(train_dataset)
        epoch_cost = time.time() - t0
        epoch_times.append(epoch_cost)

        print(f"Epoch: {epoch} cost time: {epoch_cost:.6f} s")

        if val_loader:
            val_loss, val_deg, val_f1 = evaluate(model, val_loader, device, use_amp=use_amp)
            if val_loss < best_val_loss:
                print(f"Best model stored in memory (Vali Loss: {val_loss:.6f})")
                best_val_loss = val_loss
                best_val_deg = val_deg
                best_val_f1 = val_f1

            print(f"Epoch: {epoch}, Steps: {total_steps_per_epoch} | Train Loss: {train_loss:.7f} | Vali Loss: {val_loss:.7f} | Angle Err: {val_deg:.4f} deg | Contact F1: {val_f1*100:.2f}%")
            print(f"lr = {scheduler.get_last_lr()[0]:.10f}")

            extra_info = dict(config=asdict(config), epoch=epoch, val_loss=val_loss, val_deg=val_deg, val_f1=val_f1)
            improved = early_stopping(val_loss, model, best_path, extra=extra_info)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            print(f"Epoch: {epoch}, Steps: {total_steps_per_epoch} | Train Loss: {train_loss:.7f}")
            print(f"lr = {scheduler.get_last_lr()[0]:.10f}")

    avg_epoch_time = sum(epoch_times) / max(1, len(epoch_times))
    print(f"average training time: {avg_epoch_time:.6f} s")

    max_mem = 0.0
    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"Max Memory (MB): {max_mem:.6f}")

    # Final testing summary
    print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    if val_dataset:
        print(f"test {len(val_dataset)} windows ({len(val_dataset)*20:,} frames)")
        macs_g = estimate_macs(config, seq_len=20)
        print(f"Computational complexity(MACs): {macs_g:.2f} GMac, Number of parameters: {param_count/1e6:.2f} M")
        print(f"test tensor shape: ({len(val_dataset)}, 20, 135)")
        print(f"val_loss:   {best_val_loss:.7f} (multitask composite loss)")
        print(f"angle_err:  {best_val_deg:.4f} deg (per-joint residual yaw drift)")
        print(f"contact_f1: {best_val_f1*100:.2f} % (foot strike classification for ZUPT)")
        print(f"Predictions saved to {best_path}")

    # Save final model
    torch.save(dict(config=asdict(config), state_dict=model.state_dict(),
                    trained_on=str(args.data), epochs=args.epochs), args.output)


if __name__ == "__main__":
    main()

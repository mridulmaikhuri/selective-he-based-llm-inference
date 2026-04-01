"""
train.py
========
Robust next-token prediction training loop for TinyGPT.

CLI usage examples:
    # Minimal run (10 steps, quick smoke-test)
    python train.py --data-dir data/ --max-steps 10

    # Full training run
    python train.py --data-dir data/ --max-steps 5000 --batch-size 16

    # 1-epoch run with FP16 and gradient accumulation
    python train.py --data-dir data/ --epochs 1 --fp16 --accum-steps 4

    # Custom LR and output directory
    python train.py --data-dir data/ --max-steps 5000 --lr 3e-4 --out-dir runs/exp1/
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    print("[warn] tqdm not found — install with: pip install tqdm")
    # Minimal fallback so the script still runs
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kw):
            self._it = iterable
        def __iter__(self):
            return iter(self._it)
        def set_postfix(self, **kw): pass
        def update(self, n=1): pass
        def close(self): pass

from model import TinyGPT, count_parameters


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """
    Wraps a 1-D or 2-D long tensor of token IDs into (input, target) pairs
    for next-token prediction.

    If the tensor is 1-D (flat token sequence) it is chunked into fixed-length
    windows of `seq_len` tokens; the target is the sequence shifted by 1.

    If the tensor is 2-D (B, S) each row is treated as one sample and the
    target is shifted by 1 along the sequence axis (last token of target is
    ignored via ignore_index in the loss).
    """

    def __init__(self, data: torch.Tensor, seq_len: int = 128):
        if data.ndim == 1:
            # Chunk flat sequence into (seq_len + 1) windows; drop remainder
            n_chunks = (len(data) - 1) // seq_len
            self.data = data[: n_chunks * seq_len + 1]
            self.seq_len = seq_len
            self.flat = True
        elif data.ndim == 2:
            self.data = data
            self.seq_len = data.shape[1]
            self.flat = False
        else:
            raise ValueError(f"Expected 1-D or 2-D tensor, got shape {data.shape}")

    def __len__(self) -> int:
        if self.flat:
            return (len(self.data) - 1) // self.seq_len
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.flat:
            start = idx * self.seq_len
            chunk = self.data[start : start + self.seq_len + 1]
            x = chunk[:-1].long()
            y = chunk[1:].long()
        else:
            row = self.data[idx].long()
            x = row[:-1]
            y = row[1:]
        return x, y


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    """Linearly warm up then cosine-decay to lr/10."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return lr / 10.0
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr / 10.0 + cosine * (lr - lr / 10.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    max_batches: int = 50,
) -> tuple[float, float]:
    """Return (avg_loss, perplexity) over up to `max_batches` val batches."""
    model.eval()
    total_loss, n_batches = 0.0, 0

    for x, y in loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, enabled=use_fp16):
            logits = model(x)                               # (B, S, V)
            B, S, V = logits.shape
            loss = nn.functional.cross_entropy(
                logits.reshape(B * S, V),
                y.reshape(B * S),
                ignore_index=-100,
            )

        total_loss += loss.item()
        n_batches  += 1

    avg_loss = total_loss / max(1, n_batches)
    perplexity = math.exp(min(avg_loss, 20))   # cap to avoid overflow
    model.train()
    return avg_loss, perplexity


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict,
    step: int,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler_state,
            "step":            step,
            "best_val_loss":   best_val_loss,
        },
        path,
    )
    print(f"  💾  Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train TinyGPT on next-token prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data-dir",    type=str, default="data/",
                   help="Directory containing train_inputs.pt and val_inputs.pt")
    p.add_argument("--seq-len",     type=int, default=128,
                   help="Sequence length (used only if tensors are 1-D)")
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)

    # Training duration
    p.add_argument("--max-steps",   type=int, default=5000,
                   help="Stop after this many optimiser steps")
    p.add_argument("--epochs",      type=int, default=None,
                   help="Stop after this many epochs (whichever comes first with --max-steps)")

    # Optimiser / schedule
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",type=float, default=0.01)
    p.add_argument("--warmup-steps",type=int,   default=100)
    p.add_argument("--accum-steps", type=int,   default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--clip-grad",   type=float, default=1.0,
                   help="Gradient norm clip value (0 = disabled)")

    # Model
    p.add_argument("--num-layers",  type=int, default=4)
    p.add_argument("--d-model",     type=int, default=128)
    p.add_argument("--d-ff",        type=int, default=512)
    p.add_argument("--num-heads",   type=int, default=4)
    p.add_argument("--dropout",     type=float, default=0.1)

    # Misc
    p.add_argument("--fp16",        action="store_true",
                   help="Enable automatic mixed precision (AMP) if available")
    p.add_argument("--out-dir",     type=str, default=".",
                   help="Directory to write checkpoint.pt")
    p.add_argument("--log-every",   type=int, default=100)
    p.add_argument("--val-every",   type=int, default=500)
    p.add_argument("--seed",        type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] Using {device}")

    # FP16 only makes sense on CUDA
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and not use_fp16:
        print("[warn] --fp16 requested but no CUDA device found; running in fp32")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train_inputs.pt"
    val_path   = data_dir / "val_inputs.pt"

    for p in (train_path, val_path):
        if not p.exists():
            print(
                f"[error] Dataset file not found: {p}\n"
                f"        Run data_prep.py first, or point --data-dir to the "
                f"directory containing train_inputs.pt and val_inputs.pt."
            )
            sys.exit(1)

    print(f"[data]  Loading {train_path} …", end=" ", flush=True)
    train_data = torch.load(train_path, map_location="cpu", weights_only=True)
    print(f"shape={tuple(train_data.shape)}")

    print(f"[data]  Loading {val_path} …", end=" ", flush=True)
    val_data = torch.load(val_path, map_location="cpu", weights_only=True)
    print(f"shape={tuple(val_data.shape)}")

    train_ds = TokenDataset(train_data, seq_len=args.seq_len)
    val_ds   = TokenDataset(val_data,   seq_len=args.seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Infer vocab_size from the data (max token + 1)
    vocab_size = int(train_data.max().item()) + 1
    print(f"[data]  Vocab size inferred from data : {vocab_size:,}")
    print(f"[data]  Train samples : {len(train_ds):,}  |  Val samples : {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TinyGPT(
        num_layers=args.num_layers,
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    print("\n[model] TinyGPT architecture summary")
    print(f"        Layers={args.num_layers}, d_model={args.d_model}, "
          f"heads={args.num_heads}, d_ff={args.d_ff}")
    n_params = count_parameters(model)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Separate weight-decay params (matrices) from non-decay (biases, norms)
    decay_params     = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.ndim >= 2]
    no_decay_params  = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.ndim < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # LambdaLR wrapping our custom schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, args.warmup_steps, args.max_steps, 1.0),
    )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ── Training ──────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step           = 0
    epoch          = 0
    best_val_loss  = float("inf")
    running_loss   = 0.0
    accum_count    = 0
    start_time     = time.time()

    max_epochs = args.epochs if args.epochs is not None else 10**9

    print(f"\n[train] Starting — max_steps={args.max_steps}, "
          f"max_epochs={args.epochs or '∞'}, "
          f"accum_steps={args.accum_steps}, fp16={use_fp16}\n")

    model.train()
    optimizer.zero_grad()

    pbar = tqdm(total=args.max_steps, desc="Training", unit="step", dynamic_ncols=True)

    try:
        while epoch < max_epochs and step < args.max_steps:
            epoch += 1

            for x, y in train_loader:
                if step >= args.max_steps:
                    break

                x, y = x.to(device), y.to(device)

                # ── Forward ──────────────────────────────────────────────────
                with torch.autocast(device_type=device.type, enabled=use_fp16):
                    logits = model(x)                         # (B, S, V)
                    B, S, V = logits.shape
                    loss = nn.functional.cross_entropy(
                        logits.reshape(B * S, V),
                        y.reshape(B * S),
                        ignore_index=-100,
                    )
                    loss = loss / args.accum_steps            # scale for accumulation

                # ── Backward ─────────────────────────────────────────────────
                scaler.scale(loss).backward()
                running_loss += loss.item() * args.accum_steps   # un-scale for logging
                accum_count  += 1

                if accum_count < args.accum_steps:
                    continue  # accumulate more gradients

                # ── Optimiser step ───────────────────────────────────────────
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                step        += 1
                accum_count  = 0
                avg_loss     = running_loss / args.log_every \
                               if step % args.log_every != 0 \
                               else running_loss / args.log_every

                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss.item() * args.accum_steps:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=step,
                )

                # ── Logging ──────────────────────────────────────────────────
                if step % args.log_every == 0:
                    elapsed  = time.time() - start_time
                    avg      = running_loss / args.log_every
                    print(
                        f"\n  step {step:>6} | "
                        f"train_loss {avg:.4f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"{elapsed:.1f}s elapsed"
                    )
                    running_loss = 0.0

                # ── Validation ───────────────────────────────────────────────
                if step % args.val_every == 0:
                    val_loss, ppl = evaluate(model, val_loader, device, use_fp16)
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                    print(
                        f"  ── val @ step {step} ── "
                        f"val_loss {val_loss:.4f} | "
                        f"perplexity {ppl:.2f}"
                        + (" ← best" if is_best else "")
                    )
                    model.train()

    except KeyboardInterrupt:
        print("\n[train] Interrupted by user.")

    pbar.close()

    # ── Final validation ──────────────────────────────────────────────────────
    print("\n[eval]  Running final validation …")
    val_loss, ppl = evaluate(model, val_loader, device, use_fp16)
    print(f"        Final val_loss={val_loss:.4f}  perplexity={ppl:.2f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = out_dir / "checkpoint.pt"
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler_state=scheduler.state_dict(),
        step=step,
        best_val_loss=best_val_loss,
    )

    elapsed = time.time() - start_time
    print(f"\n[done]  {step} optimiser steps in {elapsed:.1f}s  "
          f"({step / max(elapsed, 1):.1f} steps/s)")
    print(f"        Checkpoint → {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
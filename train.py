"""
Training Script for TinyGPT (step-driven, no epochs)

Usage examples:
    python train.py --data-dir data/ --max-steps 5000
    python train.py --data-dir data/ --max-steps 100 --batch-size 16 --accum-steps 2 --fp16
"""

import argparse
import os
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model import TinyGPT, count_parameters


class TokenDataset(Dataset):
    """Dataset expecting a 1D token stream (will flatten if needed). Produces (x, y) pairs of length seq_len."""
    def __init__(self, data_path, seq_len=128):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        print(f"Loading dataset from {data_path}...")
        self.data = torch.load(data_path)

        if isinstance(self.data, dict):
            # accept {"input_ids": ...} style files
            if 'input_ids' in self.data:
                self.data = self.data['input_ids']
            elif 'tokens' in self.data:
                self.data = self.data['tokens']
            else:
                raise ValueError(f"Unknown data format in {data_path}")

        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data, dtype=torch.long)

        # Flatten multi-d tensors to 1D token stream
        if self.data.dim() > 1:
            print(f"  Warning: Data has shape {self.data.shape}, flattening to 1D")
            self.data = self.data.view(-1)

        if self.data.dtype not in [torch.long, torch.int, torch.int32, torch.int64]:
            raise ValueError(f"Data must contain integer token IDs, got dtype {self.data.dtype}")

        self.seq_len = seq_len
        print(f"  Data shape after loading: {self.data.shape}")
        print(f"  Loaded {len(self.data)} tokens")
        print(f"  Number of sequences: {len(self)}")

    def __len__(self):
        return max(1, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        if end > len(self.data):
            end = len(self.data)
            start = max(0, end - self.seq_len - 1)

        chunk = self.data[start:end]
        x = chunk[:-1]
        y = chunk[1:]

        if len(x) < self.seq_len:
            pad = self.seq_len - len(x)
            x = torch.cat([x, torch.zeros(pad, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(pad, dtype=torch.long)])

        assert x.dim() == 1 and y.dim() == 1 and len(x) == self.seq_len
        return x, y


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, device, scaler, accumulation_steps, use_fp16):
    """
    Perform forward/backward for one micro-batch (does NOT step optimizer).
    Returns the scalar (unscaled) loss for logging.
    """
    input_ids, target_ids = batch

    # If single-sample (1D), ensure batch dim
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        target_ids = target_ids.unsqueeze(0)

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    if use_fp16 and scaler is not None:
        with torch.cuda.amp.autocast():
            logits = model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            targets = target_ids.view(-1)
            loss = nn.functional.cross_entropy(logits, targets, ignore_index=0)
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        return loss.item() * accumulation_steps
    else:
        logits = model(input_ids)
        logits = logits.view(-1, logits.size(-1))
        targets = target_ids.view(-1)
        loss = nn.functional.cross_entropy(logits, targets, ignore_index=0)
        loss = loss / accumulation_steps
        loss.backward()
        return loss.item() * accumulation_steps


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        logits = logits.view(-1, logits.size(-1))
        targets = y.view(-1)
        loss = nn.functional.cross_entropy(logits, targets, reduction='sum', ignore_index=0)
        nonpad = (targets != 0).sum().item()
        total_loss += loss.item()
        total_tokens += nonpad
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    model.train()
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, scheduler, step, args, filepath='checkpoint.pt'):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'args': vars(args),
    }
    torch.save(ckpt, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer, scheduler, device):
    print(f"Loading checkpoint from {filepath}...")
    ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_step = ckpt.get('step', 0)
    print(f"Resumed from step {start_step}")
    return start_step


def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("Device:", device)

    use_fp16 = args.fp16 and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    if use_fp16:
        print("Mixed precision enabled (fp16)")

    # data
    data_dir = Path(args.data_dir)
    train_path = data_dir / 'train_inputs.pt'
    val_path = data_dir / 'val_inputs.pt'
    try:
        train_ds = TokenDataset(train_path, seq_len=args.seq_len)
        val_ds = TokenDataset(val_path, seq_len=args.seq_len)
    except FileNotFoundError as e:
        print(e)
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # model
    model = TinyGPT(
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=args.seq_len,
        dropout=args.dropout
    ).to(device)
    total_params = count_parameters(model)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # total steps (step-driven)
    total_steps = int(args.max_steps)
    if total_steps <= 0:
        raise ValueError("max_steps must be > 0 for step-driven training")

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # resume from checkpoint (start_step must be known before printing config)
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    print("\nTraining configuration:")
    print(f"  Target steps: {total_steps}")
    print(f"  Batches per data pass: {len(train_loader)}")
    steps_per_pass = max(1, len(train_loader) // max(1, args.accum_steps))
    print(f"  Steps per data pass (after accumulation): {steps_per_pass}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation steps: {args.accum_steps}")
    print(f"  Effective batch size: {args.batch_size * args.accum_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Starting from step: {start_step}")
    print(f"  Total model params: {total_params:,}")

    # training state
    model.train()
    global_step = int(start_step)
    micro_step = 0  # counts microbatches since last optimizer step
    accumulated_loss = 0.0

    pbar = tqdm(total=total_steps, initial=global_step, desc="Training")

    # iterate over dataloader repeatedly until we hit total_steps
    train_iter = iter(train_loader)
    data_pass = 0

    while global_step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            # restart the iterator
            train_iter = iter(train_loader)
            data_pass += 1
            # continue to next micro-batch
            batch = next(train_iter)

        # forward & backward for micro-batch
        loss = train_step(model, batch, device, scaler, args.accum_steps, use_fp16)
        accumulated_loss += loss
        micro_step += 1

        # perform optimizer step every accumulation_steps micro-batches
        if micro_step % args.accum_steps == 0:
            # unscale & clip & step
            if use_fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            pbar.update(1)

            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr

            # logging
            if args.log_interval > 0 and (global_step % args.log_interval == 0 or args.log_interval == 1):
                avg_loss = accumulated_loss / max(1, args.log_interval)
                print(f"[Step {global_step}/{total_steps}] loss={loss:.4f} avg_loss={avg_loss:.4f} lr={current_lr:.2e}")
                # reset accumulated_loss only when we used it for averaging
                if args.log_interval != 0:
                    accumulated_loss = 0.0

            # validation & optional checkpointing
            if args.val_interval > 0 and (global_step % args.val_interval == 0):
                val_loss, perplexity = validate(model, val_loader, device)
                print(f"\nValidation @ step {global_step}: loss={val_loss:.4f}, ppl={perplexity:.2f}\n")
                if args.save_best:
                    save_checkpoint(model, optimizer, scheduler, global_step, args,
                                    filepath=f"checkpoint_step_{global_step}.pt")

            # reset micro_step after optimizer step
            micro_step = 0

        # end while loop - continue until global_step >= total_steps

    pbar.close()

    # final validation and checkpoint
    print("\nTraining finished — running final validation...")
    val_loss, perplexity = validate(model, val_loader, device)
    print(f"Final Validation - Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

    save_checkpoint(model, optimizer, scheduler, global_step, args, args.checkpoint_path)
    print(f"Saved final checkpoint to {args.checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyGPT language model (step-driven)")

    # Data args
    parser.add_argument('--data-dir', type=str, default='data/', help='Directory containing train_inputs.pt and val_inputs.pt')
    parser.add_argument('--seq-len', type=int, default=128, help='Maximum sequence length')

    # Model args
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--vocab-size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')

    # Training args (step-driven)
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--accum-steps', type=int, default=4, help='Gradient accumulation steps (microbatches per optimizer step)')
    parser.add_argument('--max-steps', type=int, default=5000, help='Maximum number of optimizer steps to run (required)')

    # Optimizer / LR
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=100, help='Number of warmup steps')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value')

    # Logging/checkpoint
    parser.add_argument('--log-interval', type=int, default=1, help='Print training logs every N steps (1 = every step)')
    parser.add_argument('--val-interval', type=int, default=500, help='Run validation every N steps (0 = never)')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint.pt', help='Path to save final checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save-best', action='store_true', help='Save checkpoint at each validation step')

    # System
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training (CUDA only)')
    parser.add_argument('--num-workers', type=int, default=0, help='Dataloader worker count')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    main(args)

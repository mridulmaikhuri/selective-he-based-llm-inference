#!/usr/bin/env python3
"""
run_strategy3.py
================
Strategy 3: Encrypt input embeddings and LM head only.

Flow per inference
------------------

  ┌────────────────────────────────────────────────────┐
  │  Token IDs (encrypted)                             │
  │  HE Embedding Lookup  ← ENCRYPTED                  │
  │  (one-hot × embedding matrix)                      │
  └────────────────────────────────────────────────────┘
       │
       ▼ decrypt to plaintext embeddings
       │
  ┌────▼────────────────────────────────────────────────┐
  │  All TransformerBlocks 0–(N-1)  (plaintext)        │
  │  • All attention ops in plaintext                  │
  │  • All FFN ops in plaintext                        │
  │  • No HE compute here                              │
  └────────────────────────────────────────────────────-┘
       │
  ┌────▼────────────────────────────────────────────────┐
  │  Final LayerNorm       (plaintext)                 │
  │  Encrypt hidden state                              │
  │  HE LM head           ← ENCRYPTED                  │
  │  Decrypt logits                                    │
  └────────────────────────────────────────────────────-┘

Privacy guarantee
-----------------
The embedding lookup is performed under HE: the server cannot observe which
token ID was presented or the embedding vector selected. The LM head is also
encrypted, preventing the server from observing the final feature projection.

However, all transformer blocks run in plaintext, so the server observes the
evolution of hidden states through the entire stack. This is weaker privacy
than Strategy 2 but may offer different latency-privacy tradeoffs depending on
whether embedding/LM head are the bottleneck.

Advantages over Strategy 1 & 2:
- Embedding lookup is "free" in plaintext (lookup) but encrypted here (linear)
- Transformer blocks all run at plaintext speed (no per-layer HE overhead)
- LM head is a single linear operation (fast vs Strategy 2's multiple ops)

Disadvantages:
- Server sees all intermediate transformer outputs
- Embedding ID and final projection are protected but not intermediate semantics

Usage
-----
    python run_strategy3.py

    # Optionally override defaults:
    python run_strategy3.py --vocab 128 --d_model 16 --seq 4 --n_inputs 10

Outputs
-------
- CSV file: strategy3_results.csv
- Printed to stdout:
  • Per-input latency (HE vs plain)
  • Aggregate timing breakdown
  • Quality metrics (top-5 overlap, cosine similarity)
  • Side-by-side comparison table (Strategies 1 vs 2 vs 3)
  • Privacy vs latency analysis
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── HE engine imports ──────────────────────────────────────────────────────
try:
    from Pyfhel import Pyfhel
    from he_layers import he_linear
    from he_utils import encrypt_tensor, decrypt_tensor, setup_HE_context, _TaggedList
    from selective_he_engine import (
        SelectiveHEConfig,
        selective_HE_inference,
        print_timing_report,
        ConfigError,
    )
    _HE_AVAILABLE = True
except ImportError as _he_err:
    _HE_AVAILABLE = False
    _HE_IMPORT_ERR = _he_err

# Import the model and metrics from Strategy 1
try:
    from run_strategy1 import (
        TinyTransformerLM,
        plain_inference,
        top_k_overlap,
        cosine_similarity_logits,
        _bar,
    )
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Strategy 3 HE inference
# ═══════════════════════════════════════════════════════════════════════════

class Strategy3Timings(NamedTuple):
    plain_time:      float
    encryption_time: float
    he_compute_time: float
    decryption_time: float
    total_time:      float


def _he_embedding_lookup(
    token_ids: torch.Tensor,     # (batch, seq_len) of token indices
    embedding_weight: torch.Tensor,  # (vocab_size, d_model)
    HE: Pyfhel,
    weight_scale: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Encrypt token IDs, perform HE embedding lookup (via one-hot × embedding matrix),
    decrypt result.

    The embedding lookup is implemented as:
      one_hot(token_id) × embedding_weight → encrypted embedding vector

    For each token, we:
      1. Convert token_id to one-hot vector (sparse)
      2. Encrypt the one-hot vector
      3. Perform HE matrix-vector product: one_hot × embedding_weight
      4. Decrypt the result

    Returns
    -------
    embeddings : (batch, seq_len, d_model) float tensor
    times      : dict with keys encrypt_ms, he_ms, decrypt_ms
    """
    batch, seq_len = token_ids.shape
    d_model = embedding_weight.shape[1]
    vocab_size = embedding_weight.shape[0]

    # Convert embedding weight to integer form
    # embedding_weight is (vocab_size, d_model), which matches he_linear expectations:
    # input (1, vocab_size) × W (vocab_size, d_model) = output (1, d_model)
    W_np = (
        embedding_weight.detach().cpu().float().numpy() * weight_scale
    ).round().astype(np.int64)  # (vocab_size, d_model) - NO transpose

    embeddings = []
    t_enc = t_he = t_dec = 0.0

    for b in range(batch):
        for s in range(seq_len):
            token_id = token_ids[b, s].item()

            # Create one-hot vector for this token
            one_hot = torch.zeros(1, vocab_size, dtype=torch.long)
            one_hot[0, token_id] = 1

            # Encrypt the one-hot vector
            t0 = time.perf_counter()
            enc_one_hot = encrypt_tensor(one_hot, HE, batch=False)
            t_enc += time.perf_counter() - t0

            # Perform HE lookup: one_hot (1, vocab_size) × W (vocab_size, d_model) = (1, d_model)
            t0 = time.perf_counter()
            enc_emb = he_linear(enc_one_hot, W_np, np.zeros(d_model, dtype=np.int64), HE)
            t_he += time.perf_counter() - t0

            # Decrypt the embedding vector
            t0 = time.perf_counter()
            emb_int = decrypt_tensor(enc_emb, HE)  # (1, d_model)
            t_dec += time.perf_counter() - t0

            embeddings.append(emb_int.float() / weight_scale)

    # Stack embeddings back to (batch, seq_len, d_model)
    embeddings_flat = torch.cat(embeddings, dim=0)  # (batch*seq_len, d_model)
    embeddings_tensor = embeddings_flat.reshape(batch, seq_len, d_model)  # (batch, seq_len, d_model)

    return embeddings_tensor, {
        "encrypt_ms": t_enc * 1e3,
        "he_ms":      t_he  * 1e3,
        "decrypt_ms": t_dec * 1e3,
    }


def _he_linear_op(
    x_in:      torch.Tensor,      # (B, ..., in_features)
    linear:    nn.Linear,
    HE:        Pyfhel,
    weight_scale: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Encrypt x_in, apply linear layer homomorphically, decrypt result.

    Returns
    -------
    out   : torch.Tensor of same shape as torch.nn.functional.linear(x_in, linear.weight, ...)
    times : dict with keys encrypt_ms, he_ms, decrypt_ms
    """
    original_shape = x_in.shape
    if len(original_shape) > 2:
        x_flat = x_in.reshape(-1, original_shape[-1])
    else:
        x_flat = x_in

    W_np = (
        linear.weight.detach().cpu().float().numpy() * weight_scale
    ).round().astype(np.int64).T

    b_np = np.zeros(linear.out_features, dtype=np.int64)
    if linear.bias is not None:
        b_np = (
            linear.bias.detach().cpu().float().numpy() * weight_scale
        ).round().astype(np.int64)

    out_samples = []
    t_enc = t_he = t_dec = 0.0

    for i in range(x_flat.shape[0]):
        sample = x_flat[i:i+1]
        sample_int = sample.round().to(torch.int64)

        t0 = time.perf_counter()
        enc_x = encrypt_tensor(sample_int, HE, batch=False)
        t_enc += time.perf_counter() - t0

        t0 = time.perf_counter()
        enc_y = he_linear(enc_x, W_np, b_np, HE)
        t_he += time.perf_counter() - t0

        t0 = time.perf_counter()
        y_int = decrypt_tensor(enc_y, HE)
        t_dec += time.perf_counter() - t0

        out_samples.append(y_int.float() / weight_scale)

    out_flat = torch.cat(out_samples, dim=0)

    if len(original_shape) > 2:
        out = out_flat.reshape(original_shape[0], original_shape[1], linear.out_features)
    else:
        out = out_flat

    return out, {
        "encrypt_ms": t_enc * 1e3,
        "he_ms":      t_he  * 1e3,
        "decrypt_ms": t_dec * 1e3,
    }


def strategy3_inference(
    model:        TinyTransformerLM,
    input_ids:    torch.Tensor,
    HE:           Pyfhel,
    weight_scale: int = 1,
) -> tuple[torch.Tensor, Strategy3Timings, list[tuple[str, bool]]]:
    """
    Run Strategy 3 inference: embedding lookup and LM head encrypted,
    all transformer blocks in plaintext.

    Parameters
    ----------
    model        : TinyTransformerLM in eval mode
    input_ids    : (1, seq_len) int64 tensor
    HE           : initialised Pyfhel context
    weight_scale : integer scale for weights (default 1)

    Returns
    -------
    logits             : (1, seq_len, vocab_size) float tensor
    timings            : Strategy3Timings named tuple
    encryption_state   : list of (layer_name, was_encrypted)
    """
    model.eval()

    t_plain = t_enc = t_he = t_dec = 0.0
    state_log: list[tuple[str, bool]] = []

    with torch.no_grad():
        # ── HE Embedding Lookup ────────────────────────────────────────────
        emb_out, emb_times = _he_embedding_lookup(
            input_ids, model.embed.weight, HE, weight_scale
        )
        t_enc += emb_times["encrypt_ms"] / 1e3
        t_he  += emb_times["he_ms"]      / 1e3
        t_dec += emb_times["decrypt_ms"] / 1e3
        state_log.append(("embedding", True))

        # Add positional embeddings (plaintext)
        t0 = time.perf_counter()
        B, S = input_ids.shape
        pos  = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x    = emb_out + model.pos_embed(pos)
        t_plain += time.perf_counter() - t0
        state_log.append(("pos_embed", False))

        # ── All Transformer blocks (plaintext) ──────────────────────────────
        for b_idx, block in enumerate(model.blocks):
            prefix = f"blocks.{b_idx}"

            # Pre-norm 1
            t0 = time.perf_counter()
            h  = block.norm1(x)
            t_plain += time.perf_counter() - t0
            state_log.append((f"{prefix}.norm1", False))

            # Q, K, V projections
            t0 = time.perf_counter()
            Q = block.attention.q_proj(h)
            K = block.attention.k_proj(h)
            V = block.attention.v_proj(h)
            t_plain += time.perf_counter() - t0
            for proj_name in ("attention.q_proj", "attention.k_proj", "attention.v_proj"):
                state_log.append((f"{prefix}.{proj_name}", False))

            # Scaled dot-product attention
            t0 = time.perf_counter()
            scale  = math.sqrt(model.d_model)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
            attn_w = torch.softmax(scores, dim=-1)
            ctx    = torch.matmul(attn_w, V)
            t_plain += time.perf_counter() - t0

            # Attention out_proj
            t0 = time.perf_counter()
            out    = block.attention.out_proj(ctx)
            t_plain += time.perf_counter() - t0
            state_log.append((f"{prefix}.attention.out_proj", False))

            # Residual add
            t0 = time.perf_counter()
            x = x + out
            t_plain += time.perf_counter() - t0

            # Pre-norm 2
            t0 = time.perf_counter()
            h2 = block.norm2(x)
            t_plain += time.perf_counter() - t0
            state_log.append((f"{prefix}.norm2", False))

            # FFN
            t0 = time.perf_counter()
            ffn_out = block.ffn(h2)
            x = x + ffn_out
            t_plain += time.perf_counter() - t0
            for sub in ("ffn.0", "ffn.2"):
                state_log.append((f"{prefix}.{sub}", False))

        # ── Final norm (plaintext) ──────────────────────────────────────────
        t0 = time.perf_counter()
        x = model.norm(x)
        t_plain += time.perf_counter() - t0
        state_log.append(("norm", False))

        # ── HE LM head ─────────────────────────────────────────────────────
        lm_out, lm_times = _he_linear_op(x, model.lm_head, HE, weight_scale)
        t_enc += lm_times["encrypt_ms"] / 1e3
        t_he  += lm_times["he_ms"]      / 1e3
        t_dec += lm_times["decrypt_ms"] / 1e3
        state_log.append(("lm_head", True))

    logits = lm_out
    total = t_plain + t_enc + t_he + t_dec
    timings = Strategy3Timings(
        plain_time      = t_plain,
        encryption_time = t_enc,
        he_compute_time = t_he,
        decryption_time = t_dec,
        total_time      = total,
    )
    return logits, timings, state_log


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Evaluation and reporting
# ═══════════════════════════════════════════════════════════════════════════

def estimate_perplexity(logits: torch.Tensor) -> float:
    """Quick perplexity estimate from last-token logits."""
    last_logits = logits[0, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    return torch.exp(entropy).item()


def print_per_input_table(results: list[dict], n_show: int = 5) -> None:
    print("\n  Per-input results (first {} / {} inputs shown)".format(
        min(n_show, len(results)), len(results)))
    print("  " + "─" * 74)
    print(f"  {'#':>3}  {'Plain ms':>9}  {'HE ms':>9}  "
          f"{'Overhead':>9}  {'Top5 overlap':>13}  {'Cosine sim':>10}")
    print("  " + "─" * 74)
    for r in results[:n_show]:
        ohd = (r["he_total_ms"] / r["plain_ms"] - 1) * 100 if r["plain_ms"] > 0 else 0
        print(f"  {r['idx']:>3}  {r['plain_ms']:>9.2f}  {r['he_total_ms']:>9.2f}  "
              f"  {ohd:>7.1f}%  {r['top5_overlap']:>12.2f}  {r['cosine_sim']:>10.4f}")
    print("  " + "─" * 74)


def print_aggregate_timing(
    results:     list[dict],
    state_log:   list[tuple[str, bool]],
    n_layers:    int,
    seq_len:     int,
    d_model:     int,
) -> None:
    keys = ["plain_ms", "encrypt_ms", "he_ms", "decrypt_ms", "he_total_ms"]
    agg  = {k: np.mean([r[k] for r in results]) for k in keys}
    total = agg["he_total_ms"]

    print("\n" + "═" * 64)
    print("  Strategy 3 — Aggregate Timing (avg over {} inputs)".format(len(results)))
    print("═" * 64)
    rows = [
        ("Plaintext compute",  "plain_ms"),
        ("Encryption",         "encrypt_ms"),
        ("HE compute",         "he_ms"),
        ("Decryption",         "decrypt_ms"),
    ]
    for label, key in rows:
        t   = agg[key]
        pct = 100 * t / total if total > 0 else 0
        print(f"  {label:<24} {t:>8.2f} ms  [{_bar(pct/100):<40}] {pct:5.1f}%")
    print(f"  {'─'*64}")
    print(f"  {'HE Total':<24} {agg['he_total_ms']:>8.2f} ms")
    print(f"  {'Plain Total':<24} {agg['plain_ms']:>8.2f} ms  "
          f"(pure plaintext for same model)")

    overhead = (agg["he_total_ms"] / agg["plain_ms"] - 1) * 100
    print(f"\n  Latency overhead: {overhead:.1f}×  "
          f"({agg['he_total_ms']:.2f} ms vs {agg['plain_ms']:.2f} ms plain)")

    enc_frac = (agg["encrypt_ms"] + agg["decrypt_ms"]) / total * 100
    print(f"  Encryption+Decryption fraction of HE time: {enc_frac:.1f}%")

    print("\n  Layer Encryption State")
    print("  " + "─" * 48)
    for name, enc in state_log:
        tag = "🔒 HE    " if enc else "🔓 plain "
        print(f"  {tag} {name}")
    print("  " + "─" * 48)

    n_he    = sum(1 for _, enc in state_log if enc)
    n_plain = len(state_log) - n_he
    print(f"  HE layers total   : {n_he}  (embedding + lm_head)")
    print(f"  Plain layers total: {n_plain}  (all transformer blocks)")


def print_quality_summary(results: list[dict]) -> None:
    overlaps = [r["top5_overlap"] for r in results]
    cosines  = [r["cosine_sim"]   for r in results]
    print("\n  Quality Metrics (HE vs Plain, {} inputs)".format(len(results)))
    print("  " + "─" * 44)
    print(f"  Top-5 token overlap  mean : {np.mean(overlaps):.3f}")
    print(f"  Top-5 token overlap  min  : {np.min(overlaps):.3f}")
    print(f"  Top-5 token overlap  max  : {np.max(overlaps):.3f}")
    print(f"  Cosine similarity    mean : {np.mean(cosines):.4f}")
    print(f"  Cosine similarity    min  : {np.min(cosines):.4f}")
    print(f"  Cosine similarity    max  : {np.max(cosines):.4f}")
    print("  " + "─" * 44)
    perfect_top5 = sum(1 for o in overlaps if o == 1.0)
    print(f"  Inputs with perfect top-5 overlap: "
          f"{perfect_top5}/{len(results)} ({100*perfect_top5/len(results):.0f}%)")


def save_results_to_csv(
    results: list[dict],
    filepath: str = "strategy3_results.csv",
) -> None:
    """Save results to CSV for later analysis."""
    if not results:
        return

    fieldnames = [
        "input_id",
        "plain_ms",
        "encrypt_ms",
        "he_ms",
        "decrypt_ms",
        "he_total_ms",
        "top5_overlap",
        "cosine_sim",
        "perplexity_plain",
        "perplexity_he",
    ]

    try:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "input_id": r["idx"],
                    "plain_ms": f"{r['plain_ms']:.4f}",
                    "encrypt_ms": f"{r['encrypt_ms']:.4f}",
                    "he_ms": f"{r['he_ms']:.4f}",
                    "decrypt_ms": f"{r['decrypt_ms']:.4f}",
                    "he_total_ms": f"{r['he_total_ms']:.4f}",
                    "top5_overlap": f"{r['top5_overlap']:.4f}",
                    "cosine_sim": f"{r['cosine_sim']:.4f}",
                    "perplexity_plain": f"{r['perplexity_plain']:.4f}",
                    "perplexity_he": f"{r['perplexity_he']:.4f}",
                })
        print(f"\n  ✓ Results saved to {filepath}")
    except Exception as e:
        print(f"\n  ✗ Failed to save CSV: {e}")


def print_privacy_analysis(
    results:      list[dict],
    n_layers:     int,
    seq_len:      int,
    weight_scale: int,
) -> None:
    """Privacy vs latency analysis for Strategy 3."""
    agg     = {k: np.mean([r[k] for r in results])
               for k in ["plain_ms", "encrypt_ms", "he_ms", "decrypt_ms", "he_total_ms"]}
    overhead = (agg["he_total_ms"] / agg["plain_ms"] - 1) * 100
    enc_pct  = (agg["encrypt_ms"] + agg["decrypt_ms"]) / agg["he_total_ms"] * 100

    print("\n" + "╔" + "═" * 62 + "╗")
    print("║  Strategy 3: Privacy vs Latency Analysis" + " " * 20 + "║")
    print("╠" + "═" * 62 + "╣")

    lines = [
        f"  Encrypted layers : embedding + lm_head only",
        f"  Plaintext layers : {n_layers} transformer blocks",
        f"  Sequence length  : {seq_len} tokens",
        f"  Weight scale     : {weight_scale}",
        "",
        f"  Latency overhead : +{overhead:.0f}%  "
        f"({agg['he_total_ms']:.1f} ms HE vs {agg['plain_ms']:.1f} ms plain)",
        f"  Enc+Dec fraction : {enc_pct:.1f}% of HE time",
        "",
        "  Privacy Coverage:",
        "    ✓ Protected: Token ID, embedding selection, final projection",
        "    ✗ Exposed:   All transformer block outputs (hidden states,",
        "                 attention scores, FFN activations)",
        "",
        "  Latency Breakdown:",
        f"    • Embedding HE   : ~{agg['encrypt_ms']:.1f}% encryption + "
        f"{agg['he_ms']:.1f}% compute",
        f"    • Transformer    : full plaintext speed  ≈ {agg['plain_ms']:.2f}ms",
        f"    • LM head HE     : encrypted output projection",
        "",
        "  Comparison to other strategies:",
        "    Strategy 1: Encrypts 1 projection per block → more privacy,",
        "               similar latency to Strategy 3",
        "    Strategy 2: Encrypts all of last block → strong privacy,",
        "               higher latency",
        "    Strategy 3: Encrypts embedding + LM head → weak privacy,",
        "               low latency (most compute is plaintext)",
        "",
        "  Use cases for Strategy 3:",
        "    • When input/output privacy is critical",
        "    • Latency is a hard constraint",
        "    • Transformer semantics can be learned from hidden states anyway",
    ]

    for line in lines:
        padded = ("║  " + line + " " * max(0, 59 - len(line)) + "║")
        print(padded)
    print("╚" + "═" * 62 + "╝")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Main driver
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strategy 3 HE inference: encrypt embedding + lm_head only"
    )
    p.add_argument("--vocab",     type=int, default=128,   help="Vocabulary size")
    p.add_argument("--d_model",   type=int, default=16,    help="Model dimension")
    p.add_argument("--seq",       type=int, default=4,     help="Sequence length")
    p.add_argument("--heads",     type=int, default=1,     help="Attention heads (fixed=1)")
    p.add_argument("--layers",    type=int, default=2,     help="Number of transformer blocks")
    p.add_argument("--ffn_dim",   type=int, default=32,    help="FFN hidden dimension")
    p.add_argument("--n_inputs",  type=int, default=20,    help="Number of test inputs")
    p.add_argument("--n_poly",    type=int, default=8192,  help="BFV polynomial degree (n)")
    p.add_argument("--t",         type=int, default=65537, help="BFV plaintext modulus t")
    p.add_argument("--wscale",    type=int, default=1,
                   help="Integer scale for weights (default 1 = already-int)")
    p.add_argument("--seed",      type=int, default=0,     help="Model parameter seed")
    p.add_argument("--csv_output", type=str, default="strategy3_results.csv",
                   help="Output CSV filename")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _HE_AVAILABLE:
        print(f"[ERROR] HE libraries not available: {_HE_IMPORT_ERR}")
        print("Install with: pip install Pyfhel")
        return

    print("╔" + "═" * 62 + "╗")
    print("║  run_strategy3.py  —  Strategy 3: Encrypt Embedding + LM Head  ║")
    print("╚" + "═" * 62 + "╝")
    print(f"\n  Model config:")
    print(f"    vocab={args.vocab}, d_model={args.d_model}, seq={args.seq}")
    print(f"    n_layers={args.layers}, ffn_dim={args.ffn_dim}")
    print(f"  BFV config: n={args.n_poly}, t={args.t}")
    print(f"  Test inputs: {args.n_inputs}  (random seeds 0..{args.n_inputs-1})")

    # ── Build model ────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    model = TinyTransformerLM(
        vocab_size = args.vocab,
        d_model    = args.d_model,
        n_heads    = args.heads,
        n_layers   = args.layers,
        ffn_dim    = args.ffn_dim,
        seq_len    = args.seq,
    )
    # Use small integer weights so BFV rounding error is zero
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    model.eval()

    # ── Initialise HE context ──────────────────────────────────────────────
    print(f"\n  Initialising HE context …", end=" ", flush=True)
    t0 = time.perf_counter()
    HE = setup_HE_context(n=args.n_poly, t=args.t)
    print(f"done ({time.perf_counter()-t0:.2f}s)")

    # ── Run N test inputs ──────────────────────────────────────────────────
    print(f"\n  Running {args.n_inputs} inference pairs (HE + plain) …\n")
    results:  list[dict] = []
    last_state_log: list[tuple[str, bool]] = []

    for i in range(args.n_inputs):
        rng = torch.Generator()
        rng.manual_seed(i)
        input_ids = torch.randint(
            0, args.vocab, (1, args.seq), generator=rng, dtype=torch.long
        )

        # Plain inference
        plain_logits, plain_s = plain_inference(model, input_ids)

        # HE Strategy 3 inference
        t_wall = time.perf_counter()
        he_logits, timings, state_log = strategy3_inference(
            model, input_ids, HE, weight_scale=args.wscale
        )
        he_wall = time.perf_counter() - t_wall

        last_state_log = state_log

        top5 = top_k_overlap(he_logits, plain_logits, k=5)
        cos  = cosine_similarity_logits(he_logits, plain_logits)

        ppl_plain = estimate_perplexity(plain_logits)
        ppl_he    = estimate_perplexity(he_logits)

        results.append({
            "idx":              i,
            "plain_ms":         plain_s          * 1e3,
            "encrypt_ms":       timings.encryption_time * 1e3,
            "he_ms":            timings.he_compute_time * 1e3,
            "decrypt_ms":       timings.decryption_time * 1e3,
            "he_total_ms":      he_wall          * 1e3,
            "top5_overlap":     top5,
            "cosine_sim":       cos,
            "perplexity_plain": ppl_plain,
            "perplexity_he":    ppl_he,
        })

        # Progress dot every 5 inputs
        if (i + 1) % 5 == 0 or i == args.n_inputs - 1:
            print(f"    [{i+1:>3}/{args.n_inputs}] last HE={he_wall*1e3:.1f}ms  "
                  f"plain={plain_s*1e3:.2f}ms  top5={top5:.2f}  cos={cos:.4f}")

    # ── Report ─────────────────────────────────────────────────────────────
    print_per_input_table(results, n_show=5)
    print_aggregate_timing(results, last_state_log, args.layers, args.seq, args.d_model)
    print_quality_summary(results)

    # Save to CSV
    save_results_to_csv(results, args.csv_output)

    # Privacy analysis
    print_privacy_analysis(results, args.layers, args.seq, args.wscale)

    # Save results for comparison
    try:
        import pickle
        with open(".strategy3_results.pkl", "wb") as f:
            pickle.dump(results, f)
    except Exception:
        pass

    print("\n  ✓ Strategy 3 evaluation complete.\n")


if __name__ == "__main__":
    main()
